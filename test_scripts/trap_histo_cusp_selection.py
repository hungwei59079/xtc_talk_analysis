import sys
import os
import json
import numpy as np
from lgdo import lh5
from xtc_utils import files_and_chnid, relevant_events, XTCConfig
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from dbetto import TextDB

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
    "--config_name",
    type=str,
    default="xtc_p16",
)

args = parser.parse_args()

# ---------- Setup ----------
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
"""
period, run = "p16", "r008"
data_path = "/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind"
prod_dir = f"{data_path}/auto/v2.0.0"
lmeta = TextDB(path=f"{prod_dir}/inputs")
dsp_dir = f"{prod_dir}/generated/tier/dsp/ssc/{period}/{run}"
hit_dir = f"{prod_dir}/generated/tier/hit/ssc/{period}/{run}"
new_dsp_list = sorted([os.path.join(dsp_dir, fname) for fname in os.listdir(dsp_dir)])
new_hit_list = sorted([os.path.join(hit_dir, fname) for fname in os.listdir(hit_dir)])
time_string = new_dsp_list[0].split("/")[-1].split("-")[4]
chmap = lmeta.hardware.configuration.channelmaps.on(time_string)
# config = lmeta.dataprod.config.on(time_string)
geds = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds']
"""
print("File listing complete.")

j1 = int(args.channel_number)
raw_id_1 = chn_id[j1]
# raw_id_1 = chmap[geds[j1]]['daq']['rawid']

# pre-selection data (with NaN filtering only)
energy_1_raw = lh5.read(f"ch{raw_id_1}/hit/cuspEmax_ctc_cal", new_hit_list).nda
mask_energy_raw = ~np.isnan(energy_1_raw)
energy_1_presel = energy_1_raw[mask_energy_raw]

table = lh5.read(f"ch{raw_id_1}/dsp/", new_dsp_list, field_mask=["trapTmax","trapTmin"])
trapTmax_1_raw = table["trapTmax"].nda
trapTmin_1_raw = table["trapTmin"].nda
max_mask_cusp_raw = ~np.isnan(trapTmax_1_raw)
min_mask_cusp_raw = ~np.isnan(trapTmin_1_raw)
trapTmax_1_presel = trapTmax_1_raw[max_mask_cusp_raw]
trapTmin_1_presel = trapTmin_1_raw[min_mask_cusp_raw]

# Selected data
trig_extract_complete = False

try:
    results = relevant_events(
        table_path=f"ch{raw_id_1}/hit/",
        files=new_hit_list,
        ene_dataset="cuspEmax_ctc_cal",
        # flag_datasets=config.baseline_flag_datasets,
        # conditions=config.baseline_conditions,
        conditions=config.xtalk_flag_trigger_conditions,
        energy_range=(1500, 4500),
    )
    idxs = results["indices"]
    energy_1_sel = results["energy_sel"]
    
    #thr = 1500
    #energy_1 = lh5.read(f"ch{raw_id_1}/hit/cuspEmax_ctc_cal", new_hit_list).nda
    #idxs = np.where(energy_1 > thr)[0]
    # energy_1_sel = energy_1[idxs]

    table = lh5.read(f"ch{raw_id_1}/dsp/", new_dsp_list, field_mask=["trapTmin", "trapTmax"], idx=idxs)
    trapTmax_1_sel = table["trapTmax"].nda
    trapTmin_1_sel = table["trapTmin"].nda
    # trapTmax_1_sel = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list, idx=idxs)
    # trapTmin_1_sel = lh5.read(f"ch{raw_id_1}/dsp/trapTmin", new_dsp_list, idx=idxs)
    
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
hist_trapTmax_presel = np.histogram(trapTmax_1_presel, bins=100)
plt.bar(hist_trapTmax_presel[1][:-1], hist_trapTmax_presel[0], width=np.diff(hist_trapTmax_presel[1]), align='edge', 
        alpha=0.5, label=f"trapTmax pre-sel ({len(trapTmax_1_presel)})")
hist_trapTmax_sel = np.histogram(trapTmax_1_sel, bins=hist_trapTmax_presel[1])  # Use same bins
plt.bar(hist_trapTmax_sel[1][:-1], hist_trapTmax_sel[0], width=np.diff(hist_trapTmax_sel[1]), align='edge', 
        alpha=0.7, label=f"trapTmax selected ({len(trapTmax_1_sel)})")

plt.xlabel('trapTmax')
plt.ylabel('Counts')
plt.yscale('log')
plt.title(f'Channel {j1} (raw {raw_id_1}) trapTmax Distribution')
plt.grid()
plt.legend()
plt.savefig(REPO_ROOT / "results" / "Inspected_histograms" / f"trapTmax_histogram_j{j1}.png")
plt.close()

#trapTmin histogram
plt.figure(figsize=(10,6))
hist_trapTmin_presel = np.histogram(trapTmin_1_presel, bins=100)
plt.bar(hist_trapTmin_presel[1][:-1], hist_trapTmin_presel[0], width=np.diff(hist_trapTmin_presel[1]), align='edge', 
        alpha=0.5, label=f"trapTmin pre-sel ({len(trapTmin_1_presel)})")
hist_trapTmin_sel = np.histogram(trapTmin_1_sel, bins=hist_trapTmin_presel[1])  # Use same bins
plt.bar(hist_trapTmin_sel[1][:-1], hist_trapTmin_sel[0], width=np.diff(hist_trapTmin_sel[1]), align='edge', 
       alpha=0.7, label=f"trapTmin selected ({len(trapTmin_1_sel)})")
    
plt.xlabel('trapTmin')
plt.ylabel('Counts')
plt.yscale('log')
plt.title(f'Channel {j1} (raw {raw_id_1}) trapTmin Distribution')
plt.grid()
plt.legend()
plt.savefig(REPO_ROOT / "results" / "Inspected_histograms" / f"trapTmin_histogram_j{j1}.png")
plt.close()
