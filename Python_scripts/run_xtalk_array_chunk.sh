#!/bin/bash
#SBATCH --job-name=xtalk_array
#SBATCH --output=sbatch_logs/log_%A_%a.txt
#SBATCH --error=sbatch_logs/log_%A_%a.txt
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100%101
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676
#SBATCH --image=legendexp/legend-software:latest

cd $SLURM_SUBMIT_DIR

date
hostname
echo "Running xtalk_batch.py on task $SLURM_ARRAY_TASK_ID"

shifter python xtalk_batch.py $SLURM_ARRAY_TASK_ID

echo "Done."
date
