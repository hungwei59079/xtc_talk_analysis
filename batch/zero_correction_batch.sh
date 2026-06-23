#!/bin/bash
#SBATCH --job-name=zero_corr
#SBATCH --output=logs/zero_corr_%A_%a.out
#SBATCH --error=logs/zero_corr_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-59%60
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676

# -----------------------------
# Configuration
# -----------------------------
CONFIG_PATH="configs/xtc_config.json"
CONFIG_NAME="xtc_p16_phy"
# Total crosstalk matrices (all runs r008-r018), restrained variants.
# These live directly under results/, not in a per-run subdirectory.
POS_MATRIX="results/pos_restrained_xtalk_matrix.csv"
NEG_MATRIX="results/neg_restrained_xtalk_matrix.csv"
OUT_DIR="results/Inspected_histograms/zero_correction"

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
echo "Running zero_correction.py on detector index ${SLURM_ARRAY_TASK_ID}"

# -----------------------------
# Run the crosstalk correction
# -----------------------------

python test_scripts/zero_correction.py "${SLURM_ARRAY_TASK_ID}" \
    --config_path "${CONFIG_PATH}" \
    --config_name "${CONFIG_NAME}" \
    --pos_matrix "${POS_MATRIX}" \
    --neg_matrix "${NEG_MATRIX}" \
    --out_dir "${OUT_DIR}"

echo "Done."
date
