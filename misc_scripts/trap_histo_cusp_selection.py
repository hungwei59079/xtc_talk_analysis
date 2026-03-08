import sys
import json
import numpy as np
from lgdo import lh5
from xtc_utils import files_and_chnid, relevant_events, XTCConfig
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "channel_number",
    type=str,
    help="channel to draw histogram for"
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
    "--selection",
    action="store_true",
    help="do selection or not"
)

args = parser.parse_args()

# ---------- Setup ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
config_path_str = "/global/u2/h/hungwei/xtc_talk_analysis/configs/xtc_config.json"
config_name = "xtc_old"

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
print("File listing complete.")

j1 = int(args.channel_number)
raw_id_1 = chn_id[j1]

if args.selection:
    try:
        energy_1, idxs = relevant_events(
            table_path=f"ch{raw_id_1}/hit/",
            files=new_hit_list,
            ene_dataset="cuspEmax_ctc_cal",
            flag_datasets=config.xtalk_flag_trigger_datasets,
            conditions=config.xtalk_flag_trigger_conditions,
            energy_range=(1500, 4500),
            return_index=True
        )
                
        trapTmax_1 = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list, idx=idxs).nda
        # mask = ~np.isnan(energy_1)
        mask = ~np.isnan(trapTmax_1) & (trapTmax_1 > 200)
        trapTmax_1 = trapTmax_1[mask]
        trapTmax_map_1 = dict(zip(idxs, trapTmax_1)) # used later for secondary selection
        print("Trigger channel event extraction complete.")
        trig_extract_complete = True
    except Exception as e:
        print(f"Exception occurred at trigger channel {j1} extraction: {e}")
        trig_extract_complete = False

else:
    energy_1 = lh5.read(f"ch{raw_id_1}/hit/cuspEmax_ctc_cal", new_hit_list).nda
    mask = ~np.isnan(energy_1)
    energy_1 = energy_1[mask]
    trapTmax_1 = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list).nda
    print(f"trapTmax maximum: {max(trapTmax_1)}")
    # mask = ~np.isnan(trapTmax_1)
    mask = ~np.isnan(trapTmax_1) & (trapTmax_1 > 200)
    trapTmax_1 = trapTmax_1[mask]

histogram = np.histogram(energy_1, bins=100)
plt.figure(figsize=(10,6))
plt.bar(histogram[1][:-1], histogram[0], width=np.diff(histogram[1]), align='edge', label=f"cuspEmax ({len(energy_1)})")
plt.xlabel('cuspEmax_ctc_cal')
plt.ylabel('Counts')
plt.title(f'Channel {j1} (raw {raw_id_1}) cuspEmax Distribution')
plt.grid()
plt.legend()
plt.savefig(REPO_ROOT / "results" / "Inspected_histograms" / f"cuspEmax_histogram_j{j1}.png")
plt.close()

histogram = np.histogram(trapTmax_1, bins=100)
plt.figure(figsize=(10,6))
plt.bar(histogram[1][:-1], histogram[0], width=np.diff(histogram[1]), align='edge', label=f"trapTmax ({len(trapTmax_1)})")
plt.xlabel('trapTmax')
plt.ylabel('Counts')
plt.title(f'Channel {j1} (raw {raw_id_1}) trapTmax Distribution')
plt.grid()
plt.legend()
plt.savefig(REPO_ROOT / "results" / "Inspected_histograms" / f"trapTmax_histogram_j{j1}.png")
plt.close()
