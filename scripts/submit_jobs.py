import json
import os
import subprocess
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Submit SLURM jobs from a JSON list.")
parser.add_argument("--jobs_file", type=str, default="configs/jobs_list.json", help="Path to the jobs list JSON file.")
args = parser.parse_args()

# Load master job list
with open(args.jobs_file, "r") as f:
    jobs = json.load(f)

for job_id, params in jobs.items():
    temp_dir = Path(params["temp_result_loc"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Dump the specific data_dict to a file for this single run
    filter_path = temp_dir / "data_filter.json"
    with open(filter_path, "w") as f:
        json.dump(params["data_dict"], f, indent=2)
        
    # Pass parameters to sbatch via --export
    # Variables exported: CONFIG_PATH, CONFIG_NAME, TEMP_RESULT_LOC, DATA_DICT_PATH
    export_vars = (
        f"CONFIG_PATH={params['config_path']},"
        f"CONFIG_NAME={params['config_name']},"
        f"TEMP_RESULT_LOC={temp_dir},"
        f"DATA_DICT_PATH={filter_path}"
    )
    
    print(f"Submitting {job_id}...")
    subprocess.run([
        "sbatch",
        f"--export=ALL,{export_vars}",
        "batch/prepare_baseline_batch.sh"
    ])
