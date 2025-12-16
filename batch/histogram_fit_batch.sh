#!/bin/bash
#SBATCH --job-name=xtalk_array
#SBATCH --output=logs/xtalk_%A_%a.out
#SBATCH --error=logs/xtalk_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100%101
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676
#SBATCH --image=legendexp/legend-software:latest

# -----------------------------
# Setup
# -----------------------------

# Always work from the repository root
REPO_ROOT=$SLURM_SUBMIT_DIR
cd "$REPO_ROOT" || exit 1

mkdir -p logs

date
hostname
echo "Running histogram_fitter.py on task ${SLURM_ARRAY_TASK_ID}"

# -----------------------------
# Run inside container
# -----------------------------

shifter python -m pip install --user -e .

shifter python scripts/histogram_fitter.py "${SLURM_ARRAY_TASK_ID}"

echo "Done."
date
