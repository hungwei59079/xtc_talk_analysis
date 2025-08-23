#!/bin/bash

# Max script runtime: 2 days (in seconds)
TIMEOUT=$((2 * 24 * 60 * 60))  # 172800 seconds
START_TIME=$(date +%s)

# List of chunk scripts
chunks=(
 # run_xtalk_array_chunk0.sh
  run_xtalk_array_chunk1.sh
  run_xtalk_array_chunk2.sh
  run_xtalk_array_chunk3.sh
  run_xtalk_array_chunk4.sh
  run_xtalk_array_chunk5.sh
)

for chunk_script in "${chunks[@]}"; do
  echo "Submitting $chunk_script..."
  jobid=$(sbatch --parsable "$chunk_script")
  echo "Submitted job array $jobid"

  echo "Waiting for job array $jobid to finish..."

  # Wait until the job finishes or global timeout reached
  while :
  do
    # Check if total script runtime has exceeded 2 days
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    if (( ELAPSED >= TIMEOUT )); then
      echo "⏰ Script reached 2-day timeout. Exiting..."
      exit 1
    fi

    # Check job status using squeue first (more reliable during runtime)
    if squeue -j "$jobid" > /dev/null 2>&1; then
      if squeue -j "$jobid" | grep -q "$jobid"; then
        echo "Job array $jobid still running... sleeping 60s"
        sleep 60
        continue
      fi
    fi

    # Fallback: Check sacct for final job state
    state=$(sacct -j "$jobid" --format=JobID,State --parsable2 --noheader | grep -E "^$jobid($|\..*)" | awk -F'|' '{print $2}' | sort | uniq)

    if [[ -z "$state" ]]; then
      echo "⚠️  Warning: Could not determine job state. Retrying in 60s..."
      sleep 60
      continue
    fi

    echo "Job array $jobid finished with state: $state"
    break
  done
done

echo "✅ All chunks submitted and completed."

