#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=logs/baseline_%A_%a.out
#SBATCH --error=logs/baseline_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100%50
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676

# -----------------------------
# Configuration
# -----------------------------
# Modify these variables as needed
CONFIG_PATH="configs/xtc_config.json"
CONFIG_NAME="default"
# Optional: path to data filter JSON file (comment out if not needed)
# DATA_DICT_PATH="configs/data_filter_example.json"

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
echo "Running prepare_baseline.py on task ${SLURM_ARRAY_TASK_ID}"

# -----------------------------
# Run the baseline computation
# -----------------------------

# Build the command with optional data_dict_path
CMD="python scripts/prepare_baseline.py --config_path ${CONFIG_PATH} --config_name ${CONFIG_NAME}"

# Uncomment the following line if using a data filter
# CMD="${CMD} --data_dict_path ${DATA_DICT_PATH}"

CMD="${CMD} ${SLURM_ARRAY_TASK_ID}"

echo "Executing: ${CMD}"
eval ${CMD}

echo "Done."
date
