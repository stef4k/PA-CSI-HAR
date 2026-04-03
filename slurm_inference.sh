#!/bin/bash
# =============================================================================
# SLURM job: re-measure inference latency at batch_size=1 for all checkpoints.
# Outputs to results/inference_latency.csv (never touches metrics.csv).
#
# Usage:
#   sbatch slurm_inference.sh
# =============================================================================

#SBATCH --job-name=PA-CSI-latency
#SBATCH --output=logs/PA-CSI-latency_%j.out
#SBATCH --error=logs/PA-CSI-latency_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

WORKDIR=/gpfs/workdir/balazsk/PA-CSI-HAR
VENV=/gpfs/workdir/balazsk/.venvs/pa-csi-har
PYTHON="${VENV}/bin/python"

mkdir -p "${WORKDIR}/logs"

cd "${WORKDIR}"

module purge
module load gcc/13.4.0/gcc-15.1.0
module load anaconda3/2023.09-0/none-none
module load cuda/12.2.2/none-none

echo "=============================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU(s)     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Python     : $("${PYTHON}" --version)"
echo "Start time : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

"${PYTHON}" inference_timing.py

echo ""
echo "=============================================="
echo "Done. Results: ${WORKDIR}/results/inference_latency.csv"
echo "End time : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="
