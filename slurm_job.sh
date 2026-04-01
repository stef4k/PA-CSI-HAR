#!/bin/bash
# =============================================================================
# SLURM job script for PA-CSI-HAR training on the Ruche cluster.
# =============================================================================
#
# Usage:
#   bash slurm_job.sh                           # submits all datasets as separate jobs
#   sbatch --export=DATASET=mine slurm_job.sh   # single job: MINE lab
#   sbatch --export=DATASET=stanwifi ...        # single job: StanWiFi
#   sbatch --export=DATASET=multienv,ENV=E1 ... # single job: MultiEnv E1
# =============================================================================

#SBATCH --job-name=PA-CSI-HAR
#SBATCH --output=logs/PA-CSI-HAR_%j.out
#SBATCH --error=logs/PA-CSI-HAR_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ── Launcher mode (run with `bash slurm_job.sh`, not sbatch) ─────────────────
# When called outside SLURM, submit one job per dataset and exit.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
    WORKDIR="$(dirname "${SCRIPT}")"
    STANWIFI_DIR="${WORKDIR}/datasets/StanWiFi"
    MULTIENV_DIR="${WORKDIR}/datasets/MultiEnv"

    echo "Submitting individual training jobs..."

    for FOLD in 1 2 3 4 5; do
        sbatch --export=DATASET=mine,FOLD=${FOLD} "${SCRIPT}"
    done

    if [[ -f "${STANWIFI_DIR}/data_amp_2000.npy" ]]; then
        sbatch --export=DATASET=stanwifi "${SCRIPT}"
    else
        echo "[SKIP] StanWiFi data not found — not submitting StanWiFi job"
    fi

    for ENV in E1 E2 E3; do
        IDX="${ENV: -1}"
        if [[ -f "${MULTIENV_DIR}/data_6c_${IDX}.npy" ]]; then
            sbatch --export=DATASET=multienv,ENV=${ENV} "${SCRIPT}"
        else
            echo "[SKIP] MultiEnv ${ENV} data not found — not submitting"
        fi
    done

    echo "Done. Check job status with: squeue -u \$USER"
    exit 0
fi

# ── Training mode (running as a SLURM job) ────────────────────────────────────
set -euo pipefail

WORKDIR=/gpfs/workdir/balazsk/PA-CSI-HAR
VENV=/gpfs/workdir/balazsk/.venvs/pa-csi-har
PYTHON="${VENV}/bin/python"
RESULTS_DIR="${WORKDIR}/results"
LOG_DIR="${WORKDIR}/logs"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

cd "${WORKDIR}"

module purge
module load gcc/13.4.0/gcc-15.1.0
module load anaconda3/2023.09-0/none-none
module load cuda/12.2.2/none-none

echo "=============================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU(s)       : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python       : $("${PYTHON}" --version)"
echo "TF version   : $("${PYTHON}" -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Working dir  : ${WORKDIR}"
echo "Dataset      : ${DATASET:-not set}"
echo "Start time   : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

STANWIFI_DIR="${WORKDIR}/datasets/StanWiFi"
MULTIENV_DIR="${WORKDIR}/datasets/MultiEnv"

case "${DATASET:-}" in

    mine)
        echo ""
        echo "=============================="
        echo "  Training: MINE lab (fold ${FOLD:-all})"
        echo "=============================="
        "${PYTHON}" train.py \
            --dataset     mine \
            --data_dir    "${WORKDIR}/datasets/" \
            --seed        42 \
            --results_dir "${RESULTS_DIR}" \
            --gpu         0 \
            ${FOLD:+--fold ${FOLD}}
        ;;

    stanwifi)
        echo ""
        echo "=============================="
        echo "  Training: StanWiFi"
        echo "=============================="
        "${PYTHON}" train.py \
            --dataset     stanwifi \
            --data_dir    "${STANWIFI_DIR}/" \
            --seed        42 \
            --results_dir "${RESULTS_DIR}" \
            --gpu         0 \
            --batch_size  32
        ;;

    multienv)
        ENV="${ENV:-E1}"
        echo ""
        echo "=============================="
        echo "  Training: MultiEnv ${ENV}"
        echo "=============================="
        "${PYTHON}" train.py \
            --dataset     multienv \
            --data_dir    "${MULTIENV_DIR}/" \
            --env         "${ENV}" \
            --seed        42 \
            --results_dir "${RESULTS_DIR}" \
            --gpu         0
        ;;

    *)
        echo "ERROR: DATASET='${DATASET:-}' is not set or not recognised."
        echo "Submit via:  bash slurm_job.sh  (to launch all datasets)"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "  Done.  Results: ${RESULTS_DIR}/metrics.csv"
echo "  End time : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="
