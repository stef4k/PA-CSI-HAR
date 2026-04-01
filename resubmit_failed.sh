#!/bin/bash
# =============================================================================
# Resubmit only the two jobs that did not finish (MINE and StanWiFi).
# MINE is split into 5 per-fold jobs (~11h each).
# StanWiFi uses batch_size=32 to fit within 24h (~12-13h expected).
# MultiEnv E1/E2/E3 already finished — not resubmitted.
# =============================================================================

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
WORKDIR="$(dirname "${SCRIPT}")"
JOB_SCRIPT="${WORKDIR}/slurm_job.sh"

echo "Submitting MINE lab folds 1-5..."
for FOLD in 1 2 3 4 5; do
    sbatch --partition=gpua100 --export=DATASET=mine,FOLD=${FOLD} "${JOB_SCRIPT}"
done

echo "Submitting StanWiFi (batch_size=32)..."
sbatch --partition=gpua100 --export=DATASET=stanwifi "${JOB_SCRIPT}"

echo ""
echo "6 jobs submitted. Check status with: squeue -u \$USER"
