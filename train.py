#!/usr/bin/env python3
"""
train.py – Comprehensive training & evaluation script for PA-CSI-HAR.

Reproduces results from:
    "Enhanced Human Activity Recognition Using Wi-Fi Sensing:
     Leveraging Phase and Amplitude with Attention Mechanisms"
    Sensors 2025, 25, 1038. https://doi.org/10.3390/s25041038

Usage examples
--------------
# MINE lab dataset (5-fold CV, data already in datasets/):
python train.py --dataset mine --data_dir datasets/

# StanWiFi (80/20 split, raw data pre-processed into .npy by dataset.py):
python train.py --dataset stanwifi --data_dir datasets/stanwifi/

# MultiEnv environment 1 (office LOS):
python train.py --dataset multienv --data_dir datasets/multienv/ --env E1

Saved artefacts
---------------
results/metrics.csv        – one row per run (appended)
results/history_<tag>.json – per-epoch loss/accuracy curves
checkpoints/<tag>/         – best-validation-accuracy model weights
"""

import argparse
import csv
import json
import os
import sys
import time
import datetime
import warnings

import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    accuracy_score,
)

from models import build_model

warnings.filterwarnings("ignore", category=UserWarning)

# ── Per-dataset hyperparameters (paper Table 3) ──────────────────────────────
DATASET_CONFIGS = {
    "stanwifi": {
        "maxlen":       2000,
        "embed_dim":    90,    # 30 subcarriers × 3 rx antennas
        "num_class":    6,
        "hlayers":      4,
        "hheads":       9,
        "K":            10,
        "sample":       2,
        "filter_sizes": [10, 40],
        "dropout":      0.1,
        "grn_units":    256,
        "batch_size":   8,
        "epochs":       200,
        "lr":           0.001,
        "decay_rate":   0.9,
        "decay_steps":  10000,
        "split":        "holdout",
        "test_size":    0.2,
        "label_names":  ["Fall", "Run", "Lie_down", "Walk", "Sit_down", "Stand_up"],
    },
    "multienv": {
        "maxlen":       850,
        "embed_dim":    90,
        "num_class":    6,
        "hlayers":      4,
        "hheads":       9,
        "K":            10,
        "sample":       2,
        "filter_sizes": [10, 40],
        "dropout":      0.1,
        "grn_units":    256,
        "batch_size":   8,
        "epochs":       200,
        "lr":           0.001,
        "decay_rate":   0.9,
        "decay_steps":  10000,
        "split":        "holdout",
        "test_size":    0.2,
        "label_names":  [
            "No_movement", "Falling", "Sit_stand",
            "Walking", "Turning", "Picking_up",
        ],
    },
    "mine": {
        # Paper says (1000, 90) but available data has 270 features
        # (30 subcarriers × 9 antenna pairs = 270).  We match the data.
        "maxlen":       1000,
        "embed_dim":    270,
        "num_class":    6,
        "hlayers":      4,
        "hheads":       9,
        "K":            10,
        "sample":       2,
        "filter_sizes": [10, 40],
        "dropout":      0.1,
        "grn_units":    256,
        "batch_size":   8,
        "epochs":       200,
        "lr":           0.001,
        "decay_rate":   0.9,
        "decay_steps":  10000,
        "split":        "kfold",
        "n_folds":      5,
        "label_names":  [
            "Squat", "Raise_hand", "Open_arms",
            "Kick_right", "Kick_left", "Other",
        ],
    },
}

CSV_COLUMNS = [
    "timestamp_utc", "dataset", "env", "model", "seed", "device",
    "fold", "epochs",
    "final_train_accuracy", "final_train_loss",
    "validation_loss", "validation_accuracy",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
    "micro_precision", "micro_recall", "micro_f1",
    "balanced_accuracy",
    "total_parameters", "trainable_parameters", "non_trainable_parameters",
    "inference_forward_time_s", "inference_latency_ms_per_sample",
    "inference_latency_ms_per_batch", "inference_throughput_samples_per_s",
    "training_time_s", "evaluation_time_s", "total_runtime_s",
    "checkpoint_path",
]


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_mine(data_dir: str):
    """Load the MINE lab dataset (pre-processed .npy files)."""
    # Try the filename used by the preprocessing step (stride=150)
    candidates = [
        ("our_data_amp_1000_270_150.npy",
         "our_data_phase_1000_270_150.npy",
         "our_data_label_1000_270_150.npy"),
        ("our_data_amp_1000_270_200.npy",
         "our_data_phase_1000_270_200.npy",
         "our_data_label_1000_270_200.npy"),
    ]
    for amp_f, phase_f, label_f in candidates:
        amp_path   = os.path.join(data_dir, amp_f)
        phase_path = os.path.join(data_dir, phase_f)
        label_path = os.path.join(data_dir, label_f)
        if os.path.exists(amp_path):
            print(f"[data] Loading MINE lab from {amp_f} …")
            x_amp   = np.load(amp_path,   allow_pickle=True).astype(np.float32)
            x_phase = np.load(phase_path, allow_pickle=True).astype(np.float32)
            labels  = np.load(label_path, allow_pickle=True).astype(np.int32)
            break
    else:
        raise FileNotFoundError(
            f"MINE lab .npy files not found in {data_dir}.\n"
            "Expected: our_data_amp_1000_270_150.npy etc."
        )

    # Min-max normalise amplitude (paper Eq. 6)
    x_amp = (x_amp - x_amp.min()) / (x_amp.max() - x_amp.min() + 1e-8)
    # Unwrap phase (paper Eq. 7 / Section 3.3.6)
    x_phase = np.unwrap(x_phase, axis=1).astype(np.float32)

    print(f"[data] amp={x_amp.shape}, phase={x_phase.shape}, "
          f"labels={labels.shape}, classes={np.unique(labels)}")
    return x_amp, x_phase, labels


def load_stanwifi(data_dir: str):
    """Load StanWiFi pre-processed dataset.

    Expected files (from the PA-CSI Google Drive):
        <data_dir>/data_amp_2000.npy    – amplitude  (N, 2000, 90)
        <data_dir>/data_phase_2000.npy  – phase      (N, 2000, 90)  already unwrapped
        <data_dir>/label_2000.npy       – labels     (N,)
    """
    amp_path   = os.path.join(data_dir, "data_amp_2000.npy")
    phase_path = os.path.join(data_dir, "data_phase_2000.npy")
    label_path = os.path.join(data_dir, "label_2000.npy")

    for p in (amp_path, phase_path, label_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"StanWiFi file not found: {p}\n"
                "Download pre-processed data from the PA-CSI Google Drive\n"
                "and place in datasets/StanWiFi/"
            )

    x_amp   = np.load(amp_path,   allow_pickle=True).astype(np.float32)
    x_phase = np.load(phase_path, allow_pickle=True).astype(np.float32)
    labels  = np.load(label_path, allow_pickle=True).astype(np.int32)

    # Min-max normalise amplitude (paper Eq. 6)
    x_amp = (x_amp - x_amp.min()) / (x_amp.max() - x_amp.min() + 1e-8)
    # Phase is already unwrapped – apply np.unwrap again is harmless but
    # skip to preserve the signal as delivered by the authors.

    print(f"[data] StanWiFi: amp={x_amp.shape}, phase={x_phase.shape}, "
          f"labels={labels.shape}, classes={np.unique(labels)}")
    return x_amp, x_phase, labels


def load_multienv(data_dir: str, env: str):
    """Load MultiEnv dataset for a specific environment (E1/E2/E3).

    Expected files (from the PA-CSI Google Drive), numbered 1/2/3:
        <data_dir>/data_6c_{idx}.npy        – amplitude   (3000, 850, 90)
        <data_dir>/data_angle_6c_{idx}.npy  – phase       (3000, 850, 90)  wrapped
        <data_dir>/label_6c_{idx}.npy       – labels      (3000,)
    Environment mapping: E1→1 (Office LOS), E2→2 (Hall LOS), E3→3 (NLOS)
    """
    env_idx = {"E1": 1, "E2": 2, "E3": 3}.get(env.upper())
    if env_idx is None:
        raise ValueError(f"Unknown env '{env}'. Use E1, E2, or E3.")

    amp_path   = os.path.join(data_dir, f"data_6c_{env_idx}.npy")
    phase_path = os.path.join(data_dir, f"data_angle_6c_{env_idx}.npy")
    label_path = os.path.join(data_dir, f"label_6c_{env_idx}.npy")

    for p in (amp_path, phase_path, label_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"MultiEnv file not found: {p}\n"
                "Download pre-processed data from the PA-CSI Google Drive\n"
                "and place in datasets/MultiEnv/"
            )

    x_amp   = np.load(amp_path,   allow_pickle=True).astype(np.float32)
    x_phase = np.load(phase_path, allow_pickle=True).astype(np.float32)
    labels  = np.load(label_path, allow_pickle=True).astype(np.int32)

    x_amp = (x_amp - x_amp.min()) / (x_amp.max() - x_amp.min() + 1e-8)
    x_phase = np.unwrap(x_phase, axis=1).astype(np.float32)

    print(f"[data] MultiEnv {env}: amp={x_amp.shape}, labels={labels.shape}")
    return x_amp, x_phase, labels


# ── Model utilities ───────────────────────────────────────────────────────────

def make_optimizer(cfg):
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg["lr"],
        decay_steps=cfg["decay_steps"],
        decay_rate=cfg["decay_rate"],
    )
    return optimizers.Adam(learning_rate=lr_schedule)


def count_parameters(model):
    total        = model.count_params()
    trainable    = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    non_trainable = total - trainable
    return total, trainable, non_trainable


def measure_inference(model, x_amp_sample, x_phase_sample, n_trials=20):
    """Time a single fixed-size batch (warmup excluded)."""
    # Warmup
    _ = model([x_amp_sample[:1], x_phase_sample[:1]], training=False)
    batch_size = len(x_amp_sample)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = model([x_amp_sample, x_phase_sample], training=False)
        times.append(time.perf_counter() - t0)
    batch_time = float(np.mean(times))
    return {
        "inference_forward_time_s":          batch_time,
        "inference_latency_ms_per_sample":   batch_time * 1000 / batch_size,
        "inference_latency_ms_per_batch":    batch_time * 1000,
        "inference_throughput_samples_per_s": batch_size / batch_time,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, x_amp, x_phase, y_true, batch_size):
    """Full evaluation: loss/accuracy via model.evaluate + sklearn metrics."""
    eval_start = time.perf_counter()

    # Loss + accuracy
    loss, acc = model.evaluate(
        [x_amp, x_phase], y_true,
        batch_size=batch_size, verbose=0,
    )

    # Per-sample predictions for sklearn metrics
    proba = model.predict(
        [x_amp, x_phase], batch_size=batch_size, verbose=0
    )
    y_pred = np.argmax(proba, axis=1)
    eval_time = time.perf_counter() - eval_start

    metrics = {
        "validation_loss":     float(loss),
        "validation_accuracy": float(acc),
    }
    for avg in ("macro", "weighted", "micro"):
        metrics[f"{avg}_precision"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"{avg}_recall"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"{avg}_f1"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
        )
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["evaluation_time_s"] = eval_time
    return metrics, y_pred, confusion_matrix(y_true, y_pred)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_csv(path, row: dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


# ── Single-fold training ──────────────────────────────────────────────────────

def train_fold(
    cfg, x_amp_train, x_phase_train, y_train,
    x_amp_val, x_phase_val, y_val,
    checkpoint_path, args, fold=0,
):
    """Build, compile, train, and evaluate model for one fold."""
    # Set seed for reproducibility
    tf.random.set_seed(args.seed + fold)
    np.random.seed(args.seed + fold)

    model = build_model(cfg)
    model.compile(
        optimizer=make_optimizer(cfg),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if fold == 0:
        model.summary()

    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt_file = os.path.join(checkpoint_path, "best.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_file,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]

    train_start = time.perf_counter()
    history = model.fit(
        [x_amp_train, x_phase_train], y_train,
        validation_data=([x_amp_val, x_phase_val], y_val),
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.perf_counter() - train_start

    # Reload best checkpoint for evaluation
    model.load_weights(ckpt_file)

    eval_metrics, y_pred, conf_mat = evaluate_model(
        model, x_amp_val, x_phase_val, y_val, cfg["batch_size"]
    )

    # Inference timing on one batch
    batch_size = min(cfg["batch_size"], len(x_amp_val))
    inf_metrics = measure_inference(
        model,
        x_amp_val[:batch_size],
        x_phase_val[:batch_size],
    )

    total, trainable, non_trainable = count_parameters(model)

    # Detect device
    gpus = tf.config.list_physical_devices("GPU")
    device = gpus[0].name if gpus else "CPU"

    row = {
        "timestamp_utc":          datetime.datetime.utcnow().isoformat(),
        "dataset":                args.dataset,
        "env":                    getattr(args, "env", ""),
        "model":                  "PA-CSI",
        "seed":                   args.seed,
        "device":                 device,
        "fold":                   fold,
        "epochs":                 cfg["epochs"],
        "final_train_accuracy":   float(history.history["accuracy"][-1]),
        "final_train_loss":       float(history.history["loss"][-1]),
        "total_parameters":       total,
        "trainable_parameters":   trainable,
        "non_trainable_parameters": non_trainable,
        "training_time_s":        training_time,
        "total_runtime_s":        training_time + eval_metrics["evaluation_time_s"],
        "checkpoint_path":        ckpt_file,
        **eval_metrics,
        **inf_metrics,
    }

    return row, history, conf_mat, y_pred


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train & evaluate the PA-CSI model"
    )
    p.add_argument(
        "--dataset", choices=["mine", "stanwifi", "multienv"],
        default="mine", help="Dataset to use",
    )
    p.add_argument(
        "--data_dir", default="datasets/",
        help="Root directory containing the dataset files",
    )
    p.add_argument(
        "--env", default="E1",
        help="MultiEnv environment: E1 (office LOS), E2 (hall LOS), E3 (NLOS)",
    )
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs from config",
    )
    p.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size from config",
    )
    p.add_argument(
        "--results_dir", default="results/",
        help="Directory for CSV results",
    )
    p.add_argument(
        "--gpu", default="0",
        help="GPU index to use (set to '' to use CPU)",
    )
    p.add_argument(
        "--fold", type=int, default=None,
        help="Run only this fold (1-indexed). Omit to run all folds.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # GPU setup
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[device] GPUs available: {[g.name for g in gpus] or 'None (CPU)'}")

    # Config
    cfg = dict(DATASET_CONFIGS[args.dataset])
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    print(f"\n[config] Dataset={args.dataset}  epochs={cfg['epochs']}  "
          f"batch={cfg['batch_size']}  lr={cfg['lr']}")

    # Data loading
    if args.dataset == "mine":
        x_amp, x_phase, labels = load_mine(args.data_dir)
    elif args.dataset == "stanwifi":
        x_amp, x_phase, labels = load_stanwifi(args.data_dir)
    else:
        x_amp, x_phase, labels = load_multienv(args.data_dir, args.env)

    # Results path
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "metrics.csv")
    init_csv(csv_path)

    tag = f"{args.dataset}_{getattr(args, 'env', '')}_{args.seed}"

    # ── Training strategy ──────────────────────────────────────────────────
    if cfg["split"] == "kfold":
        # 5-fold stratified cross-validation (MINE lab dataset)
        n_folds = cfg["n_folds"]
        kf      = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=args.seed)
        fold_rows  = []
        all_conf   = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(x_amp, labels)):
            if args.fold is not None and fold + 1 != args.fold:
                continue
            print(f"\n{'='*60}")
            print(f"  FOLD {fold+1}/{n_folds}")
            print(f"{'='*60}")

            ckpt = os.path.join("checkpoints", f"{tag}_fold{fold+1}")
            row, hist, conf, y_pred = train_fold(
                cfg,
                x_amp[train_idx], x_phase[train_idx], labels[train_idx],
                x_amp[val_idx],   x_phase[val_idx],   labels[val_idx],
                ckpt, args, fold=fold + 1,
            )
            fold_rows.append(row)
            all_conf = conf if all_conf is None else all_conf + conf
            append_csv(csv_path, row)

            # Save per-fold history
            hist_path = os.path.join(
                args.results_dir, f"history_{tag}_fold{fold+1}.json"
            )
            with open(hist_path, "w") as f:
                json.dump(
                    {k: [float(v) for v in vs]
                     for k, vs in hist.history.items()},
                    f, indent=2,
                )

        # Save averaged-fold confusion matrix (only when all folds were run)
        if args.fold is None:
            np.save(
                os.path.join(args.results_dir, f"confusion_{tag}_all_folds.npy"),
                all_conf,
            )

        # Print fold summary
        print("\n" + "="*60)
        print("  CROSS-VALIDATION SUMMARY")
        print("="*60)
        for metric in ("validation_accuracy", "macro_f1", "weighted_f1",
                       "balanced_accuracy"):
            vals = [r[metric] for r in fold_rows]
            print(f"  {metric:<30s}  "
                  f"mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    else:
        # 80/20 holdout split (StanWiFi, MultiEnv)
        (x_amp_train, x_amp_val,
         x_phase_train, x_phase_val,
         y_train, y_val) = train_test_split(
            x_amp, x_phase, labels,
            test_size=cfg["test_size"],
            stratify=labels,
            random_state=args.seed,
        )
        print(f"[split] train={len(y_train)}, val={len(y_val)}")

        ckpt = os.path.join("checkpoints", tag)
        row, hist, conf, y_pred = train_fold(
            cfg,
            x_amp_train, x_phase_train, y_train,
            x_amp_val,   x_phase_val,   y_val,
            ckpt, args, fold=0,
        )
        append_csv(csv_path, row)

        # Save history
        hist_path = os.path.join(args.results_dir, f"history_{tag}.json")
        with open(hist_path, "w") as f:
            json.dump(
                {k: [float(v) for v in vs]
                 for k, vs in hist.history.items()},
                f, indent=2,
            )

        # Save confusion matrix
        np.save(
            os.path.join(args.results_dir, f"confusion_{tag}.npy"),
            conf,
        )

        # Print summary
        print("\n" + "="*60)
        print(f"  FINAL RESULTS  [{args.dataset}"
              f"{' ' + args.env if args.dataset == 'multienv' else ''}]")
        print("="*60)
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k:<40s}  {v:.6f}")

    print(f"\n[done] Results appended to {csv_path}")


if __name__ == "__main__":
    main()
