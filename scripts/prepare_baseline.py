from pathlib import Path
import numpy as np
import argparse
import json
from datetime import datetime
from xtc_utils import files_and_chnid, get_baseline_energy

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
args = parser.parse_args()


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(args.config_path)
OUTDIR = REPO_ROOT / "temp_results" / "parameters"
OUTDIR.mkdir(parents=True, exist_ok=True)

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

new_hit_list, new_dsp_list, chn_id = files_and_chnid(CONFIG_PATH, args.config_name, data_dict)

positive_baseline, negative_baseline, skipped_channels = (
    get_baseline_energy(new_hit_list, new_dsp_list, chn_id)
)

np.save(OUTDIR / "positive_baseline.npy", positive_baseline)
np.save(OUTDIR / "negative_baseline.npy", negative_baseline)

if skipped_channels:
    np.save(OUTDIR / "skipped_channels.npy", np.array(skipped_channels))

# Save metadata recording what was used in this processing
metadata = {
    "config_path": str(CONFIG_PATH.resolve()),
    "config_name": args.config_name,
    "data_filter": data_dict,  # None if all periods/runs were used
    "processed_at": datetime.now().isoformat(),
}
metadata_path = OUTDIR / "baseline_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")