#!/bin/bash
# =============================================================================
# SLURM job script for PA-CSI-HAR training on the Ruche cluster.
# =============================================================================
#
# Usage:
#   sbatch slurm_job.sh                    # trains all available datasets
#   sbatch --export=DATASET=mine slurm_job.sh   # mine only
#
# Adjust #SBATCH directives below as needed for your Ruche allocation.
# Common GPU partitions on Ruche: gpu, gpua100, gpuv100
# =============================================================================

#SBATCH --job-name=PA-CSI-HAR
#SBATCH --output=logs/PA-CSI-HAR_%j.out
#SBATCH --error=logs/PA-CSI-HAR_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com    # uncomment and set your e-mail

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail

WORKDIR=/gpfs/workdir/balazsk/PA-CSI-HAR
VENV=/gpfs/workdir/balazsk/.venvs/pa-csi-har
RESULTS_DIR="${WORKDIR}/results"
LOG_DIR="${WORKDIR}/logs"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

cd "${WORKDIR}"

# Activate virtual environment
source "${VENV}/bin/activate"

echo "=============================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU(s)       : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python       : $(python --version)"
echo "TF version   : $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Working dir  : ${WORKDIR}"
echo "Start time   : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

# ── Select which datasets to train ───────────────────────────────────────────
# Override by setting the DATASET environment variable, e.g.:
#   sbatch --export=DATASET=stanwifi slurm_job.sh
RUN_MINE=${DATASET:-all}   # default: run all available datasets

# ── MINE lab dataset (always available) ───────────────────────────────────────
if [[ "${RUN_MINE}" == "all" || "${RUN_MINE}" == "mine" ]]; then
    echo ""
    echo "=============================="
    echo "  Training: MINE lab (5-fold)"
    echo "=============================="
    python train.py \
        --dataset   mine \
        --data_dir  "${WORKDIR}/datasets/" \
        --seed      42 \
        --results_dir "${RESULTS_DIR}" \
        --gpu       0
fi

# ── StanWiFi dataset ──────────────────────────────────────────────────────────
# Pre-processed files from PA-CSI Google Drive, placed in datasets/StanWiFi/
#   data_amp_2000.npy  |  data_phase_2000.npy  |  label_2000.npy
STANWIFI_DIR="${WORKDIR}/datasets/StanWiFi"
if [[ "${RUN_MINE}" == "all" || "${RUN_MINE}" == "stanwifi" ]]; then
    if [[ -f "${STANWIFI_DIR}/data_amp_2000.npy" ]]; then
        echo ""
        echo "=============================="
        echo "  Training: StanWiFi"
        echo "=============================="
        python train.py \
            --dataset   stanwifi \
            --data_dir  "${STANWIFI_DIR}/" \
            --seed      42 \
            --results_dir "${RESULTS_DIR}" \
            --gpu       0
    else
        echo "[SKIP] StanWiFi data not found at ${STANWIFI_DIR}/data_amp_2000.npy"
    fi
fi

# ── MultiEnv dataset (one run per environment) ────────────────────────────────
# Pre-processed files from PA-CSI Google Drive, placed in datasets/MultiEnv/
#   data_6c_{1,2,3}.npy  |  data_angle_6c_{1,2,3}.npy  |  label_6c_{1,2,3}.npy
# E1→1 (Office LOS),  E2→2 (Hall LOS),  E3→3 (NLOS)
MULTIENV_DIR="${WORKDIR}/datasets/MultiEnv"
if [[ "${RUN_MINE}" == "all" || "${RUN_MINE}" == "multienv" ]]; then
    for ENV in E1 E2 E3; do
        IDX="${ENV: -1}"          # last character: 1, 2, or 3
        PROBE="${MULTIENV_DIR}/data_6c_${IDX}.npy"
        if [[ -f "${PROBE}" ]]; then
            echo ""
            echo "=============================="
            echo "  Training: MultiEnv ${ENV}"
            echo "=============================="
            python train.py \
                --dataset   multienv \
                --data_dir  "${MULTIENV_DIR}/" \
                --env       "${ENV}" \
                --seed      42 \
                --results_dir "${RESULTS_DIR}" \
                --gpu       0
        else
            echo "[SKIP] MultiEnv ${ENV} not found (expected ${PROBE})"
        fi
    done
fi

echo ""
echo "=============================================="
echo "  All done.  Results: ${RESULTS_DIR}/metrics.csv"
echo "  End time : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="
