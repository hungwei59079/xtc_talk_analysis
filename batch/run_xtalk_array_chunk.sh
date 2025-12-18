#!/bin/bash
#SBATCH --job-name=xtalk_array
#SBATCH --output=logs/xtalk_%A_%a.out
#SBATCH --error=logs/xtalk_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=57-58%2
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676

# -----------------------------
# Setup
# -----------------------------

# Always work from the repository root
REPO_ROOT=$SLURM_SUBMIT_DIR
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate

mkdir -p logs

date
hostname
echo "Running xtalk_batch.py on task ${SLURM_ARRAY_TASK_ID}"

# -----------------------------
# Run inside container
# -----------------------------

python scripts/xtalk_batch.py "${SLURM_ARRAY_TASK_ID}"

echo "Done."
date
