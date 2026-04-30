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
# Configuration (now primarily using Environment Variables from orchestrator)
# -----------------------------
if [ -z "$CONFIG_PATH" ]; then
    echo "WARNING: CONFIG_PATH not set by environment. Falling back to default." >&2
    CONFIG_PATH="configs/xtc_config.json"
fi

if [ -z "$CONFIG_NAME" ]; then
    echo "WARNING: CONFIG_NAME not set by environment. Falling back to default." >&2
    CONFIG_NAME="xtc_p16"
fi

if [ -z "$TEMP_RESULT_LOC" ]; then
    echo "WARNING: TEMP_RESULT_LOC not set by environment. Falling back to default." >&2
    TEMP_RESULT_LOC="${SCRATCH}"
fi

# Optional: path to data filter JSON file
# DATA_DICT_PATH=${DATA_DICT_PATH:-"configs/data_filter_example.json"}

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
echo "Running prepare_baseline.py on task ${SLURM_ARRAY_TASK_ID}, using config ${CONFIG_NAME} at ${CONFIG_PATH}, storing results in ${TEMP_RESULT_LOC}"

# -----------------------------
# Run the baseline computation
# -----------------------------

# Build the command with optional data_dict_path
CMD="python scripts/prepare_baseline.py --config_path ${CONFIG_PATH} --config_name ${CONFIG_NAME} --temp_result_target ${TEMP_RESULT_LOC}"

if [ -n "$DATA_DICT_PATH" ]; then
    CMD="${CMD} --data_dict_path ${DATA_DICT_PATH}"
fi

CMD="${CMD} ${SLURM_ARRAY_TASK_ID}"

echo "Executing: ${CMD}"
eval ${CMD}

echo "Done."
date
