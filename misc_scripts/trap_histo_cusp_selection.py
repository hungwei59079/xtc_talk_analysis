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

# Always load pre-selection data (with NaN filtering only)
energy_1_raw = lh5.read(f"ch{raw_id_1}/hit/cuspEmax_ctc_cal", new_hit_list).nda
mask_energy_raw = ~np.isnan(energy_1_raw)
energy_1_presel = energy_1_raw[mask_energy_raw]

trapTmax_1_raw = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list).nda
print(f"trapTmax maximum: {np.nanmax(trapTmax_1_raw)}")
mask_trap_raw = ~np.isnan(trapTmax_1_raw) & (trapTmax_1_raw > 200)
trapTmax_1_presel = trapTmax_1_raw[mask_trap_raw]

# Selected data (if selection is enabled)
energy_1_sel = None
trapTmax_1_sel = None
trig_extract_complete = False

if args.selection:
    try:
        energy_1_sel, idxs = relevant_events(
            table_path=f"ch{raw_id_1}/hit/",
            files=new_hit_list,
            ene_dataset="cuspEmax_ctc_cal",
            flag_datasets=config.xtalk_flag_trigger_datasets,
            conditions=config.xtalk_flag_trigger_conditions,
            energy_range=(1500, 4500),
            return_index=True
        )
                
        trapTmax_1_sel_raw = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list, idx=idxs).nda
        mask_sel = ~np.isnan(trapTmax_1_sel_raw) & (trapTmax_1_sel_raw > 200)
        trapTmax_1_sel = trapTmax_1_sel_raw[mask_sel]
        trapTmax_map_1 = dict(zip(idxs, trapTmax_1_sel)) # used later for secondary selection
        print("Trigger channel event extraction complete.")
        trig_extract_complete = True
    except Exception as e:
        print(f"Exception occurred at trigger channel {j1} extraction: {e}")
        trig_extract_complete = False

# ---------- Plotting ----------
# cuspEmax histogram
plt.figure(figsize=(10,6))
hist_presel = np.histogram(energy_1_presel, bins=100)
plt.bar(hist_presel[1][:-1], hist_presel[0], width=np.diff(hist_presel[1]), align='edge', 
        alpha=0.5, label=f"cuspEmax pre-sel ({len(energy_1_presel)})")
if args.selection and energy_1_sel is not None:
    hist_sel = np.histogram(energy_1_sel, bins=hist_presel[1])  # Use same bins
    plt.bar(hist_sel[1][:-1], hist_sel[0], width=np.diff(hist_sel[1]), align='edge', 
            alpha=0.7, label=f"cuspEmax selected ({len(energy_1_sel)})")
plt.xlabel('cuspEmax_ctc_cal')
plt.ylabel('Counts')
plt.yscale('log')
plt.title(f'Channel {j1} (raw {raw_id_1}) cuspEmax Distribution')
plt.grid()
plt.legend()
plt.savefig(REPO_ROOT / "results" / "Inspected_histograms" / f"cuspEmax_histogram_j{j1}.png")
plt.close()

# trapTmax histogram
plt.figure(figsize=(10,6))
hist_trap_presel = np.histogram(trapTmax_1_presel, bins=100)
plt.bar(hist_trap_presel[1][:-1], hist_trap_presel[0], width=np.diff(hist_trap_presel[1]), align='edge', 
        alpha=0.5, label=f"trapTmax pre-sel ({len(trapTmax_1_presel)})")
if args.selection and trapTmax_1_sel is not None:
    hist_trap_sel = np.histogram(trapTmax_1_sel, bins=hist_trap_presel[1])  # Use same bins
    plt.bar(hist_trap_sel[1][:-1], hist_trap_sel[0], width=np.diff(hist_trap_sel[1]), align='edge', 
            alpha=0.7, label=f"trapTmax selected ({len(trapTmax_1_sel)})")
plt.xlabel('trapTmax')
plt.ylabel('Counts')
plt.yscale('log')
plt.title(f'Channel {j1} (raw {raw_id_1}) trapTmax Distribution')
plt.grid()
plt.legend()
plt.savefig(REPO_ROOT / "results" / "Inspected_histograms" / f"trapTmax_histogram_j{j1}.png")
plt.close()
