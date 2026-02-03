from pathlib import Path
from lgdo.lh5 import read
import numpy as np
import argparse
import json
from datetime import datetime
from xtc_utils import files_and_chnid, relevant_events, XTCConfig


def get_single_baseline_energy(new_hit_list, new_dsp_list, detector, detector_idx, flag_datasets, flag_conditions):
    """
    Computes mean baseline energy for a single detector.

    Args:
        new_hit_list: list of hit files
        new_dsp_list: list of dsp files
        detector: detector channel ID
        detector_idx: index of detector in the channel list
        flag_datasets: list of flag dataset names for filtering
        flag_conditions: dict of conditions for filtering

    Returns:
        positive_baseline: mean positive baseline (trapTmax), or np.nan if failed
        negative_baseline: mean negative baseline (trapTmin), or np.nan if failed
        success: bool indicating if the computation succeeded
    """
    try:
        energies, idxs = relevant_events(
            table_path=f"ch{detector}/hit/",
            files=new_hit_list,
            ene_dataset="cuspEmax_ctc_cal",
            flag_datasets=flag_datasets,
            conditions=flag_conditions,
            return_index=True
        )
        table = read(f"ch{detector}/dsp/", new_dsp_list, field_mask=["trapTmin", "trapTmax"], idx=idxs)
        trapTmin = table["trapTmin"].nda
        trapTmax = table["trapTmax"].nda
        positive_baseline = np.mean(trapTmax)
        negative_baseline = np.mean(trapTmin)
        print(f"✅ Baseline energy evaluated for detector #{detector_idx} (ID={detector}).")
        return positive_baseline, negative_baseline, True
    except Exception as e:
        print(f"❌ Failed for detector #{detector_idx} (ID={detector}): {e}")
        return np.nan, np.nan, False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the configuration JSON file.",
)
parser.add_argument(
    "--config_name",
    type=str,
    required=True,
    help="Name of the configuration to use from the JSON file.",
)
parser.add_argument(
    "--data_dict_path",
    type=str,
    default=None,
    help="Path to a JSON file specifying which periods/runs to use. "
         "Format: {\"p08\": [\"r015\", \"r016\"], \"p09\": [\"r001\"]}. "
         "If not provided, all periods/runs from the config will be used.",
)
parser.add_argument(
    "detector_index",
    type=int,
    help="Index of the detector to process (0-based index into the channel list)."
)
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(args.config_path)
OUTDIR = REPO_ROOT / "temp_results" / "parameters" / "baseline_individual"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Load configuration using XTCConfig
config = XTCConfig(CONFIG_PATH, args.config_name)

# Load data_dict from file if provided
data_dict = None
if args.data_dict_path is not None:
    data_dict_path = Path(args.data_dict_path)
    if not data_dict_path.exists():
        raise FileNotFoundError(f"Data dictionary file not found: {data_dict_path}")
    with open(data_dict_path) as f:
        data_dict = json.load(f)
    print(f"Loaded data filter from: {data_dict_path}")
    print(f"Filtering to periods/runs: {data_dict}")

new_hit_list, new_dsp_list, chn_id = files_and_chnid(config, data_dict)

# Validate detector index
detector_index = args.detector_index
if detector_index < 0 or detector_index >= len(chn_id):
    raise ValueError(f"detector_index {detector_index} out of range. Valid range: 0-{len(chn_id)-1}")

detector = chn_id[detector_index]
print(f"Processing detector index {detector_index} (channel ID: {detector})")

# Compute baseline for this single detector
positive_baseline, negative_baseline, success = get_single_baseline_energy(
    new_hit_list, new_dsp_list, detector, detector_index,
    config.baseline_flag_datasets, config.baseline_conditions
)

# Save individual results
result = {
    "detector_index": detector_index,
    "detector_id": detector,
    "positive_baseline": float(positive_baseline) if not np.isnan(positive_baseline) else None,
    "negative_baseline": float(negative_baseline) if not np.isnan(negative_baseline) else None,
    "success": success,
    "config_path": str(CONFIG_PATH.resolve()),
    "config_name": args.config_name,
    "data_filter": data_dict, 
    "processed_at": datetime.now().isoformat(),
}

out_path = OUTDIR / f"baseline_{detector_index:04d}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Result saved to: {out_path}")