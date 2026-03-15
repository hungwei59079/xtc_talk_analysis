from pathlib import Path
import numpy as np
import json
import argparse
from datetime import datetime
from xtc_utils import XTCConfig

def merge_baseline_results(input_dir: Path, output_dir: Path):
    result_files = sorted(input_dir.glob("baseline_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No baseline_*.json files found in {input_dir}")
    print(f"Found {len(result_files)} individual result files")

    with open(result_files[0]) as f:
        data_ref = json.load(f)
        parameters_ref = data_ref["parameters"]
        
    config = XTCConfig(parameters_ref["config_path"], parameters_ref["config_name"])
    n_detectors = config.number_of_detectors
        
    positive_baseline = np.full(n_detectors, np.nan)
    negative_baseline = np.full(n_detectors, np.nan)
    skipped_channels = []
    
    #index catalog - used to see if indices are missing
    idx_catalog = set()
    
    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)
        idx = data["detector_index"]
        idx_catalog.add(idx)
        if data["parameters"] != parameters_ref:
            raise RuntimeError("Data corrupted: json from other runs detected")
        if not data["success"]:
            skipped_channels.append(data["detector_id"])
            continue
        positive_baseline[idx] = data["positive_baseline"]
        negative_baseline[idx] = data["negative_baseline"]

    missing_indices = set(np.arange(n_detectors)) - idx_catalog
    if missing_indices:
        print(f"Missing indices: {missing_indices}")
        raise RuntimeError("Indices Missing. Perhaps baseline computation crashed. Check log for more details.")
    
    # Save merged results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "positive_baseline.npy", positive_baseline)
    np.save(output_dir / "negative_baseline.npy", negative_baseline)
    
    if skipped_channels:
        np.save(output_dir / "skipped_channels.npy", np.array(skipped_channels))
    else:
        np.save(output_dir / "skipped_channels.npy", np.array([], dtype=int))
    
    # Save metadata
    n_success = np.sum(~np.isnan(positive_baseline))
    n_failed = len(skipped_channels)
    
    metadata = {
        "parameters": parameters_ref,
        "merged_at": datetime.now().isoformat(),
        "total_detectors": n_detectors,
        "successful": int(n_success),
        "failed": n_failed,
        "skipped_channel_ids": skipped_channels,
    }
    
    with open(output_dir / "baseline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    summary = {
        "total": n_detectors,
        "successful": n_success,
        "failed": n_failed,
        "missing": len(missing_indices),
    }
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge individual baseline results into combined arrays."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing individual baseline_XXXX.json files. "
             "Default: temp_results/parameters/json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write merged results. "
             "Default: temp_results/parameters",
    )
    parser.add_argument(
    "--data_dict_path",
    type=str,
    default=None,
    )

    args = parser.parse_args()
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir) if args.input_dir else REPO_ROOT / "temp_results" / "parameters" / "baseline_individuals" / "json"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "temp_results" / "parameters" 
    
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    summary = merge_baseline_results(input_dir, output_dir)
    
    print()
    print("=" * 50)
    print("Merge Summary")
    print("=" * 50)
    print(f"Total detectors:  {summary['total']}")
    print(f"Successful:       {summary['successful']}")
    print(f"Failed:           {summary['failed']}")
    print(f"Missing:          {summary['missing']}")
    print()
    print(f"Results saved to: {output_dir}")
    print("  - positive_baseline.npy")
    print("  - negative_baseline.npy")
    print("  - skipped_channels.npy")
    print("  - baseline_metadata.json")
