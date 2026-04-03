#!/usr/bin/env python3
"""
inference_timing.py – Re-measure inference latency at batch_size=1 for all
saved checkpoints, without touching metrics.csv or any training artefact.

Outputs: results/inference_latency.csv  (created fresh, never appended)

Checkpoints used:
  StanWiFi  : checkpoints/stanwifi_E1_42/best.h5
  MultiEnv  : checkpoints/multienv_{E1,E2,E3}_42_cw/best.h5   (class-weighted)
  MINE lab  : checkpoints/mine_E1_42_fold{1..5}/best.h5
"""

import csv
import os
import time
import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold

from models import build_model
from train import (
    DATASET_CONFIGS,
    load_stanwifi,
    load_multienv,
    load_mine,
    count_parameters,
)

# ── Config ────────────────────────────────────────────────────────────────────

WORKDIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKDIR, "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "inference_latency.csv")

CHECKPOINTS = {
    "stanwifi_E1":     "checkpoints/stanwifi_E1_42/best.h5",
    "multienv_E1_cw":  "checkpoints/multienv_E1_42_cw/best.h5",
    "multienv_E2_cw":  "checkpoints/multienv_E2_42_cw/best.h5",
    "multienv_E3_cw":  "checkpoints/multienv_E3_42_cw/best.h5",
    "mine_fold1":      "checkpoints/mine_E1_42_fold1/best.h5",
    "mine_fold2":      "checkpoints/mine_E1_42_fold2/best.h5",
    "mine_fold3":      "checkpoints/mine_E1_42_fold3/best.h5",
    "mine_fold4":      "checkpoints/mine_E1_42_fold4/best.h5",
    "mine_fold5":      "checkpoints/mine_E1_42_fold5/best.h5",
}

N_TRIALS    = 50   # more trials at bs=1 for stable estimate
SEED        = 42

CSV_COLUMNS = [
    "timestamp_utc", "run_id", "dataset", "env", "fold",
    "checkpoint_path", "batch_size",
    "total_parameters",
    "inference_forward_time_s",
    "inference_latency_ms_per_sample",
    "inference_throughput_samples_per_s",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def measure_inference_bs1(model, x_amp_1, x_phase_1, n_trials=N_TRIALS):
    """Time a single-sample forward pass (warmup excluded)."""
    # Warmup
    _ = model([x_amp_1, x_phase_1], training=False)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = model([x_amp_1, x_phase_1], training=False)
        times.append(time.perf_counter() - t0)
    fwd = float(np.mean(times))
    return {
        "batch_size":                        1,
        "inference_forward_time_s":          fwd,
        "inference_latency_ms_per_sample":   fwd * 1000,
        "inference_throughput_samples_per_s": 1.0 / fwd,
    }


def build_and_load(cfg, checkpoint_path):
    model = build_model(cfg)
    model.load_weights(checkpoint_path)
    return model


def get_single_sample(x_amp, x_phase):
    """Return a (1, maxlen, embed_dim) batch for timing."""
    return x_amp[:1], x_phase[:1]


# ── Per-dataset data loaders (reuse train.py loaders, grab one sample) ────────

def get_stanwifi_sample():
    cfg = DATASET_CONFIGS["stanwifi"]
    x_amp, x_phase, labels = load_stanwifi(
        os.path.join(WORKDIR, "datasets/StanWiFi")
    )
    x_amp_tr, x_amp_val, x_phase_tr, x_phase_val, _, _ = train_test_split(
        x_amp, x_phase, labels, test_size=cfg["test_size"],
        random_state=SEED, shuffle=True,
    )
    return get_single_sample(x_amp_val, x_phase_val)


def get_multienv_sample(env):
    cfg = DATASET_CONFIGS["multienv"]
    x_amp, x_phase, labels = load_multienv(
        os.path.join(WORKDIR, "datasets/MultiEnv"), env
    )
    x_amp_tr, x_amp_val, x_phase_tr, x_phase_val, _, _ = train_test_split(
        x_amp, x_phase, labels, test_size=cfg["test_size"],
        random_state=SEED, shuffle=True,
    )
    return get_single_sample(x_amp_val, x_phase_val)


def get_mine_sample(fold_idx):
    """fold_idx: 1-based (1..5)"""
    cfg = DATASET_CONFIGS["mine"]
    x_amp, x_phase, labels = load_mine(os.path.join(WORKDIR, "datasets/"))
    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=SEED)
    splits = list(skf.split(x_amp, labels))
    _, val_idx = splits[fold_idx - 1]
    return get_single_sample(x_amp[val_idx], x_phase[val_idx])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    device_str = gpus[0].name if gpus else "CPU"
    print(f"[device] Running on: {device_str}")

    rows = []

    # ── StanWiFi ──────────────────────────────────────────────────────────────
    print("\n[stanwifi] Loading data and checkpoint …")
    cfg  = DATASET_CONFIGS["stanwifi"]
    ckpt = os.path.join(WORKDIR, CHECKPOINTS["stanwifi_E1"])
    model = build_and_load(cfg, ckpt)
    x_amp_1, x_phase_1 = get_stanwifi_sample()
    total, _, _ = count_parameters(model)
    inf = measure_inference_bs1(model, x_amp_1, x_phase_1)
    print(f"  latency: {inf['inference_latency_ms_per_sample']:.2f} ms/sample")
    rows.append({
        "timestamp_utc":   datetime.datetime.utcnow().isoformat(),
        "run_id":          "stanwifi_E1",
        "dataset":         "stanwifi",
        "env":             "E1",
        "fold":            "",
        "checkpoint_path": ckpt,
        "total_parameters": total,
        **inf,
    })
    del model

    # ── MultiEnv ──────────────────────────────────────────────────────────────
    for env in ("E1", "E2", "E3"):
        key  = f"multienv_{env}_cw"
        ckpt = os.path.join(WORKDIR, CHECKPOINTS[key])
        print(f"\n[multienv {env}] Loading data and checkpoint …")
        cfg   = DATASET_CONFIGS["multienv"]
        model = build_and_load(cfg, ckpt)
        x_amp_1, x_phase_1 = get_multienv_sample(env)
        total, _, _ = count_parameters(model)
        inf = measure_inference_bs1(model, x_amp_1, x_phase_1)
        print(f"  latency: {inf['inference_latency_ms_per_sample']:.2f} ms/sample")
        rows.append({
            "timestamp_utc":   datetime.datetime.utcnow().isoformat(),
            "run_id":          key,
            "dataset":         "multienv",
            "env":             env,
            "fold":            "",
            "checkpoint_path": ckpt,
            "total_parameters": total,
            **inf,
        })
        del model

    # ── MINE lab ──────────────────────────────────────────────────────────────
    for fold in range(1, 6):
        key  = f"mine_fold{fold}"
        ckpt = os.path.join(WORKDIR, CHECKPOINTS[key])
        print(f"\n[mine fold {fold}] Loading data and checkpoint …")
        cfg   = DATASET_CONFIGS["mine"]
        model = build_and_load(cfg, ckpt)
        x_amp_1, x_phase_1 = get_mine_sample(fold)
        total, _, _ = count_parameters(model)
        inf = measure_inference_bs1(model, x_amp_1, x_phase_1)
        print(f"  latency: {inf['inference_latency_ms_per_sample']:.2f} ms/sample")
        rows.append({
            "timestamp_utc":   datetime.datetime.utcnow().isoformat(),
            "run_id":          key,
            "dataset":         "mine",
            "env":             "E1",
            "fold":            fold,
            "checkpoint_path": ckpt,
            "total_parameters": total,
            **inf,
        })
        del model

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[done] Results written to: {OUT_CSV}")
    for r in rows:
        print(f"  {r['run_id']:25s}  {r['inference_latency_ms_per_sample']:8.2f} ms/sample")


if __name__ == "__main__":
    main()
