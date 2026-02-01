"""
Merge individual baseline results from batch processing into combined arrays.

This script reads all baseline_XXXX.json files from the individual results directory
and combines them into the standard format (positive_baseline.npy, negative_baseline.npy, 
skipped_channels.npy) expected by downstream scripts like xtalk_batch.py.
"""

from pathlib import Path
import numpy as np
import json
import argparse
from datetime import datetime


def merge_baseline_results(input_dir: Path, output_dir: Path, expected_count: int = None):
    """
    Merge individual baseline JSON files into combined numpy arrays.
    
    Args:
        input_dir: Directory containing baseline_XXXX.json files
        output_dir: Directory to write merged results
        expected_count: Expected number of detectors (optional, for validation)
    
    Returns:
        dict with summary statistics
    """
    # Find all individual result files
    result_files = sorted(input_dir.glob("baseline_*.json"))
    
    if not result_files:
        raise FileNotFoundError(f"No baseline_*.json files found in {input_dir}")
    
    print(f"Found {len(result_files)} individual result files")
    
    # Load all results
    results = {}
    metadata_sample = None
    
    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)
        idx = data["detector_index"]
        results[idx] = data
        if metadata_sample is None:
            metadata_sample = {
                "config_path": data.get("config_path"),
                "config_name": data.get("config_name"),
                "data_filter": data.get("data_filter"),
            }
    
    # Determine the range of indices
    indices = sorted(results.keys())
    max_idx = max(indices)
    
    if expected_count is not None:
        if len(results) != expected_count:
            print(f"⚠️  Warning: Expected {expected_count} results, found {len(results)}")
        max_idx = expected_count - 1
    
    # Check for missing indices
    expected_indices = set(range(max_idx + 1))
    actual_indices = set(indices)
    missing_indices = expected_indices - actual_indices
    
    if missing_indices:
        print(f"⚠️  Warning: Missing results for indices: {sorted(missing_indices)}")
    
    # Build arrays
    n_detectors = max_idx + 1
    positive_baseline = np.full(n_detectors, np.nan)
    negative_baseline = np.full(n_detectors, np.nan)
    skipped_channels = []
    
    for idx, data in results.items():
        if idx >= n_detectors:
            print(f"⚠️  Warning: Index {idx} exceeds expected count, skipping")
            continue
            
        if data["success"]:
            positive_baseline[idx] = data["positive_baseline"]
            negative_baseline[idx] = data["negative_baseline"]
        else:
            skipped_channels.append(data["detector_id"])
    
    # Also mark missing indices as skipped (we don't have their detector IDs, so use -1 as placeholder)
    for idx in missing_indices:
        print(f"⚠️  Index {idx} has no result file, marking as skipped")
    
    # Save merged results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "positive_baseline.npy", positive_baseline)
    np.save(output_dir / "negative_baseline.npy", negative_baseline)
    
    if skipped_channels:
        np.save(output_dir / "skipped_channels.npy", np.array(skipped_channels))
    else:
        # Save empty array if no skipped channels
        np.save(output_dir / "skipped_channels.npy", np.array([], dtype=int))
    
    # Save metadata
    n_success = np.sum(~np.isnan(positive_baseline))
    n_failed = len(skipped_channels)
    
    metadata = {
        "config_path": metadata_sample["config_path"] if metadata_sample else None,
        "config_name": metadata_sample["config_name"] if metadata_sample else None,
        "data_filter": metadata_sample["data_filter"] if metadata_sample else None,
        "merged_at": datetime.now().isoformat(),
        "total_detectors": n_detectors,
        "successful": int(n_success),
        "failed": n_failed,
        "missing_indices": sorted(missing_indices) if missing_indices else [],
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
             "Default: temp_results/parameters/baseline_individual",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write merged results. "
             "Default: temp_results/parameters",
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=None,
        help="Expected number of detectors (for validation). "
             "If not provided, inferred from the highest index found.",
    )
    args = parser.parse_args()
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    input_dir = Path(args.input_dir) if args.input_dir else REPO_ROOT / "temp_results" / "parameters" / "baseline_individual"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "temp_results" / "parameters"
    
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    summary = merge_baseline_results(input_dir, output_dir, args.expected_count)
    
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
