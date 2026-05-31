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

args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(args.config_path)
OUTDIR = REPO_ROOT / "results" / "Inspected_histograms"
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

new_hit_list, new_dsp_list, evt_list, chn_id = files_and_chnid(config, data_dict, tiers=("hit", "dsp", "evt"), on_mismatch="drop")

# Validate detector index
detector_index = args.detector_index
if detector_index < 0 or detector_index >= len(chn_id):
    raise ValueError(f"detector_index {detector_index} out of range. Valid range: 0-{len(chn_id)-1}")

detector = chn_id[detector_index]

pulser_selector = EventSelector(
    table_path=f"evt/coincident/",
    files=evt_list,
    ene_dataset="geds",
    conditions={"puls" : False}
)
trapTmax_selector = EventSelector(
    table_path=f"ch{detector}/dsp/",
    files=new_dsp_list,
    ene_dataset="trapTmax",
    idx=pulser_selector.selected_idxs,
)

trapTmax_selector.draw(OUTDIR / f"detector_{detector}_trapTmax_histogram.png")