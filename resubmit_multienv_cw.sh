#!/bin/bash
# =============================================================================
# Re-run MultiEnv E1/E2/E3 with balanced class weighting to address the
# class imbalance issue (No_movement 40% vs Picking_up 6.7%).
# Results are saved with the "_cw" suffix to distinguish from the baseline:
#   results/history_multienv_{E1,E2,E3}_42_cw.json
#   results/confusion_multienv_{E1,E2,E3}_42_cw.npy
#   checkpoints/multienv_{E1,E2,E3}_42_cw/best.h5
# The metrics.csv row will also carry the _cw tag in checkpoint_path.
# =============================================================================

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
WORKDIR="$(dirname "${SCRIPT}")"
JOB_SCRIPT="${WORKDIR}/slurm_job.sh"

echo "Submitting MultiEnv E1/E2/E3 with class weighting (gpua100)..."
for ENV in E1 E2 E3; do
    sbatch --partition=gpua100 --export=DATASET=multienv,ENV=${ENV},CLASS_WEIGHT=1 "${JOB_SCRIPT}"
done

echo ""
echo "3 jobs submitted. Results will be tagged with '_cw'."
echo "Check status with: squeue -u \$USER"
