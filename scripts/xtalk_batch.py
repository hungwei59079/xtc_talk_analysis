# xtalk_batch.py

import sys
import os
import json
import numpy as np
from lgdo import lh5
from xtc_utils import files_and_chnid, relevant_events, xtalk_element, XTCConfig
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Setup ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "temp_results"/ "parameters"
baseline_metadata_file = PARAMS_PATH / "baseline_metadata.json"
with open(baseline_metadata_file, "r") as f:
    baseline_metadata = json.load(f)
    config_path_str = baseline_metadata["config_path"]
    config_name = baseline_metadata["config_name"]
    data_filter = baseline_metadata["data_filter"]

print(data_filter) # I got None
CONFIG_PATH = Path(config_path_str)
config = XTCConfig(CONFIG_PATH, config_name)

OUTDIR = REPO_ROOT / "temp_results" / "histograms"
OUTDIR.mkdir(parents=True, exist_ok=True)
new_hit_list, new_dsp_list, chn_id = files_and_chnid(config, data_dict=data_filter)
print("File listing complete.")

# Load parameters
positive_baseline = np.load(PARAMS_PATH / "positive_baseline.npy")
negative_baseline = np.load(PARAMS_PATH / "negative_baseline.npy")
skipped_channels = set(np.load(PARAMS_PATH / "skipped_channels.npy"))

j1 = int(sys.argv[1])
raw_id_1 = chn_id[j1]

if raw_id_1 in skipped_channels:
    print(f"Trigger channel j1={j1} (raw {raw_id_1}) is in skipped_channels; saving empty histograms for all j2.")
    for j2 in range(0, 101):
        out_path = OUTDIR / f"xtalk_{j1}_{j2}.npz"
        np.savez_compressed(
            out_path,
            neg_counts=np.array([], dtype=int),
            neg_bins=np.array([]),
            pos_counts=np.array([], dtype=int),
            pos_bins=np.array([]),
            neg_vals=np.array([]),
        )
    sys.exit(0)
    
#Trigger channel event extraction
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
    trapTmax_map_1 = dict(zip(idxs, trapTmax_1)) # used later for secondary selection
    print("Trigger channel event extraction complete.")
    trig_extract_complete = True
except Exception as e:
    print(f"Exception occurred at trigger channel {j1} extraction.")
    trig_extract_complete = False

NBINS = 700

for j2 in range(len(chn_id)):
    raw_id_2 = chn_id[j2]
    neg_vals = np.array([])
    pos_vals = np.array([])
    
# Skip if self-interaction or either channel is missing
    if raw_id_1 == raw_id_2:
        print(f"Self-interaction at ({j1}, {j2}) ignored.")
    elif raw_id_2 in skipped_channels:
        print(f"Skipping job ({j1}, {j2}) due to missing channel(s).")
    else:
        energy_2, secondary_idxs = relevant_events(
            table_path=f"ch{raw_id_2}/hit/",
            files=new_hit_list,
            ene_dataset="cuspEmax_ctc_cal",
            flag_datasets=config.xtalk_flag_response_datasets,
            conditions=config.xtalk_flag_response_conditions,
            energy_range=(-9999, 100),
            idx = idxs,
            return_index=True
        )
        selected_trapTmax_1 = np.array([trapTmax_map_1[i] for i in secondary_idxs])

        table_2 = lh5.read(f"ch{raw_id_2}/dsp/", new_dsp_list, field_mask=["trapTmin", "trapTmax"], idx=secondary_idxs)
        trapTmin_2 = table_2["trapTmin"].nda
        trapTmax_2 = table_2["trapTmax"].nda

        neg_vals = np.asarray(xtalk_element(selected_trapTmax_1, trapTmin_2, negative_baseline[j2]))
        pos_vals = np.asarray(xtalk_element(selected_trapTmax_1, trapTmax_2, positive_baseline[j2]))

    # Build histograms (counts + bin edges). For empty arrays save empty arrays.
    if neg_vals.size:
        # neg_counts, neg_bins = np.histogram(neg_vals, bins=NBINS, range=(max(min(neg_vals), -5),0.5))
        neg_counts, neg_bins = np.histogram(neg_vals, bins=NBINS, range=(min(neg_vals),0.5))
    else:
        neg_counts = np.array([], dtype=int)
        neg_bins = np.array([])

    if pos_vals.size:
        # pos_counts, pos_bins = np.histogram(pos_vals, bins=NBINS, range=(-0.5,min(max(pos_vals), 5)))
        pos_counts, pos_bins = np.histogram(pos_vals, bins=NBINS, range=(-0.5, max(pos_vals)))
    else:
        pos_counts = np.array([], dtype=int)
        pos_bins = np.array([])

    # Save histogram numeric data to compressed .npz
    out_path = OUTDIR / f"xtalk_{j1}_{j2}.npz"
    np.savez_compressed(
        out_path,
        neg_counts=neg_counts,
        neg_bins=neg_bins,
        pos_counts=pos_counts,
        pos_bins=pos_bins,
        # also save raw values in case you later want to re-bin / inspect them
        neg_vals=neg_vals,
        pos_vals=pos_vals
    )

    print(f"histogram ({j1},{j2}) saved: {out_path}.")
