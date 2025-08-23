#!/bin/bash
#SBATCH --job-name=xtalk_array
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=4000-5999%200
#SBATCH -A m2676
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH --image=legendexp/legend-software:latest

date
hostname
echo "Running xtalk_batch.py on task $SLURM_ARRAY_TASK_ID"

shifter python xtalk_batch.py $SLURM_ARRAY_TASK_ID

echo "Done."
date

