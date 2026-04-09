from pathlib import Path
import numpy as np
import argparse
import json
from datetime import datetime
from xtc_utils import files_and_chnid, EventSelector, XTCConfig

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
parser.add_argument(
    "--temp_result_dir",
    type=str,
    default=None,
    help="Path to the temp_results directory where outputs will be saved. "
         "If not provided, defaults to <repo_root>/temp_results.",
)
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(args.config_path)
OUTDIR = Path(args.temp_result_dir) / "parameters" / "baseline_individuals"
JSONDIR = OUTDIR / "json"
TRAPTMIN_DIR = OUTDIR / "trapTmin"
TRAPTMAX_DIR = OUTDIR / "trapTmax"
TRAPTMIN_DIR.mkdir(parents=True, exist_ok=True)
TRAPTMAX_DIR.mkdir(parents=True, exist_ok=True)
JSONDIR.mkdir(parents=True, exist_ok=True)
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
successful_extraction = True
try:
    cuspEmax_selection = EventSelector(
        table_path=f"ch{detector}/hit/",
        files=new_hit_list,
        ene_dataset="cuspEmax_ctc_cal",
        conditions=config.baseline_conditions,
    )
    trapTmax_selection = EventSelector(
        table_path=f"ch{detector}/dsp/",
        files=new_dsp_list,
        ene_dataset="trapTmax",
        idx=cuspEmax_selection.selected_idxs
    )
    trapTmin_selection = EventSelector(
        table_path=f"ch{detector}/dsp/",
        files=new_dsp_list,
        ene_dataset="trapTmin",
        idx=cuspEmax_selection.selected_idxs
    )

    positive_baseline = np.mean(trapTmax_selection.selected_energies)
    negative_baseline = np.mean(trapTmin_selection.selected_energies)
    trapTmax_selection.draw(TRAPTMAX_DIR / f"trapTmax_detector_{detector_index}.png")
    trapTmin_selection.draw(TRAPTMIN_DIR / f"trapTmin_detector_{detector_index}.png")
    
except Exception as e:
    print(f"exception occurred at detector {detector_index}: {e}")
    successful_extraction = False
    positive_baseline = np.nan
    negative_baseline = np.nan
    

# Save individual results
result = {
    "detector_index": detector_index,
    "detector_id": detector,
    "positive_baseline": float(positive_baseline) if not np.isnan(positive_baseline) else None,
    "negative_baseline": float(negative_baseline) if not np.isnan(negative_baseline) else None,
    "success": successful_extraction,
    "processed_at": datetime.now().isoformat(),
    "parameters": { 
        "config_path": str(CONFIG_PATH.resolve()),
        "config_name": args.config_name,
        "baseline_conditions": config.baseline_conditions,
        "data_filter": data_dict, 
    }
}

out_path = JSONDIR / f"baseline_{detector_index:04d}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Result saved to: {out_path}")