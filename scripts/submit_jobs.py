import json
import os
import subprocess
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Submit SLURM jobs from a JSON list.")
parser.add_argument("--jobs_file", type=str, default="configs/jobs_list.json", help="Path to the jobs list JSON file.")
parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], required=True, help="Which pipeline step to run")
args = parser.parse_args()

# Load master job list
with open(args.jobs_file, "r") as f:
    jobs = json.load(f)

for job_id, params in jobs.items():
    temp_dir = params["temp_result_loc"]
    print(f"--- Processing {job_id} for Step {args.step} ---")
    
    if args.step == 1:
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        
        # Dump the specific data_dict to a file for this single run
        filter_path = temp_path / "data_filter.json"
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
        
        subprocess.run([
            "sbatch",
            f"--export=ALL,{export_vars}",
            "batch/prepare_baseline_batch.sh"
        ])
    
    elif args.step == 2:
        subprocess.run(["./scripts/merge_baseline.sh", f"{temp_dir}/temp_results"])
        
    elif args.step == 3:
        export_vars = f"TEMP_RESULT_LOC={temp_dir}/temp_results"
        subprocess.run([
            "sbatch",
            f"--export=ALL,{export_vars}",
            "batch/run_xtalk_array_chunk.sh"
        ])
        
    elif args.step == 4:
        subprocess.run(["./scripts/histogram_fitter.sh", f"{temp_dir}/temp_results"])
