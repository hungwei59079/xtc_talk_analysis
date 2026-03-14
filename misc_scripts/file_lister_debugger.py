from pathlib import Path
import argparse
import json
from xtc_utils import files_and_chnid, relevant_events, XTCConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dict_path",
    type=str,
    default=None,
    help="Path to a JSON file specifying which periods/runs to use. "
         "Format: {\"p08\": [\"r015\", \"r016\"], \"p09\": [\"r001\"]}. "
         "If not provided, all periods/runs from the config will be used.",
)

parser.add_argument(
    "--config_name",
    type=str,
    default="xtc_p16",
)
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[1]
config_path_str = "/global/u2/h/hungwei/xtc_talk_analysis/configs/xtc_config.json"
config_name = args.config_name
CONFIG_PATH = Path(config_path_str)
config = XTCConfig(CONFIG_PATH, config_name)

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

hit_timestrings = [f.split("-")[-2] for f in new_hit_list]
dsp_timestrings = [f.split("-")[-2] for f in new_dsp_list]

print(hit_timestrings[5:])
print(dsp_timestrings[5:])

if len(hit_timestrings) != len(dsp_timestrings):
    raise ValueError("mismatched file number")

strings_complete_match = True
for i in range(len(hit_timestrings)):
    if hit_timestrings[i] != dsp_timestrings[i]:
        print(f"mismatch detected: hit timestring = {hit_timestrings[i]}, dsp timestring = {dsp_timestrings[i]}")
        strings_complete_match = False

if strings_complete_match:
    print("All timestrings match.")