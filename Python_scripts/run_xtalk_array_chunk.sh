#!/bin/bash
#SBATCH --job-name=xtalk_array
#SBATCH --output=sbatch_logs/log_%A_%a.txt
#SBATCH --error=sbatch_logs/log_%A_%a.txt
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1020%200
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676
#SBATCH --image=legendexp/legend-software:latest

cd $SLURM_SUBMIT_DIR

date
hostname

# Compute start and end indices for this array task
START=$(( SLURM_ARRAY_TASK_ID * 10 ))
END=$(( START + 9 ))
if [ $END -gt 10200 ]; then
    END=10200
fi

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID handling tasks $START to $END"

for i in $(seq $START $END); do
    echo "Running xtalk_batch.py on task $i"
    shifter python xtalk_batch.py $i
done

echo "Done."
date
