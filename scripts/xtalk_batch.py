# xtalk_batch.py

import sys
import argparse
import os
import json
import numpy as np
from lgdo import lh5
from xtc_utils import files_and_chnid, EventSelector, xtalk_element, XTCConfig
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Setup ----------
parser = argparse.ArgumentParser()
parser.add_argument("--temp_result_dir", 
                    help="Path to the temp_results directory containing parameters and metadata.",
                    default=Path(__file__).resolve().parents[1] / "temp_results"
                    )
parser.add_argument("j1", type=int, help="Index of the trigger channel (0-based).")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(args.temp_result_dir) / "histograms" / "histograms_not_fitted"
OUTDIR.mkdir(parents=True, exist_ok=True)
PARAMS_PATH = args.temp_result_dir / "parameters"
baseline_metadata_file = PARAMS_PATH / "baseline_metadata.json"
with open(baseline_metadata_file, "r") as f:
    baseline_metadata = json.load(f)
parameters = baseline_metadata["parameters"]
n_detectors = baseline_metadata["total_detectors"]
config_path_str = parameters["config_path"]
config_name = parameters["config_name"]
data_filter = parameters["data_filter"]

CONFIG_PATH = Path(config_path_str)
config = XTCConfig(CONFIG_PATH, config_name)

with open(CONFIG_PATH, "r") as f:
    full_cfg = json.load(f)
    fit_params = full_cfg.get("fit_parameters", {})
    nbins = fit_params.get("histogram", {}).get("nbins", 700)
    range_mult = fit_params.get("histogram", {}).get("range_multiplier", 3)

new_hit_list, new_dsp_list, chn_id = files_and_chnid(config, data_dict=data_filter)
j1 = int(sys.argv[1])
raw_id_1 = chn_id[j1]
print("File listing complete.")

# Metadata
parameters["xtalk_flag_trigger_conditions"] = config.xtalk_flag_trigger_conditions
parameters["xtalk_flag_response_conditions"] = config.xtalk_flag_response_conditions

if j1 == 0:
    from datetime import datetime
    xtalk_metadata = {
        "parameters": parameters,
        "number_of_detectors": n_detectors,
        "processed_at": datetime.now().isoformat(),
    }
    xtalk_metadata_file = OUTDIR / "xtalk_metadata.json"
    with open(xtalk_metadata_file, "w") as f:
        json.dump(xtalk_metadata, f, indent=2)
    print(f"Metadata saved to {xtalk_metadata_file}")


# Load parameters
positive_baseline = np.load(PARAMS_PATH / "positive_baseline.npy")
negative_baseline = np.load(PARAMS_PATH / "negative_baseline.npy")
skipped_channels = set(np.load(PARAMS_PATH / "skipped_channels.npy"))

if raw_id_1 in skipped_channels:
    print(f"Trigger channel j1={j1} (raw {raw_id_1}) is in skipped_channels; saving empty histograms for all j2.")
    for j2 in range(n_detectors):
        out_path = OUTDIR / f"xtalk_{j1}_{j2}.npz"
        np.savez_compressed(
            out_path,
            neg_counts=np.array([], dtype=int),
            neg_bins=np.array([]),
            pos_counts=np.array([], dtype=int),
            pos_bins=np.array([]),
            neg_counts_restrained=np.array([], dtype=int),
            neg_bins_restrained=np.array([]),
            pos_counts_restrained=np.array([], dtype=int),
            pos_bins_restrained=np.array([]),
            neg_vals=np.array([]),
            pos_vals=np.array([]),
        )
    sys.exit(0)
    
#Trigger channel event extraction
try:
    cuspEmax_selection_1 = EventSelector(
        table_path=f"ch{raw_id_1}/hit/",
        files=new_hit_list,
        ene_dataset="cuspEmax_ctc_cal",
        conditions=config.xtalk_flag_trigger_conditions,
        energy_range=(1500, 99999),
    )
    print(f"trigger selection indices: {cuspEmax_selection_1.selected_idxs}")
    trapTmax_1_presel = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list).nda
    print("Trigger channel event extraction complete.")
    trig_extract_complete = True
except Exception as e:
    print(f"Exception occurred at trigger channel {j1} extraction: {e}")
    trig_extract_complete = False


def build_hist(vals, mean, stdev, name):
    if vals.size and not np.isnan(stdev) and stdev > 0:
        counts, bins = np.histogram(vals, bins=nbins)
        counts_restr, bins_restr = np.histogram(
            vals,
            bins=nbins,
            range=(mean - range_mult*stdev, mean + range_mult*stdev)
            )
        return counts, bins, counts_restr, bins_restr
    else:
        print(f"warning: {name}_histogram empty or invalid")
        return np.array([], dtype=int), np.array([]), np.array([], dtype=int), np.array([])

for j2 in range(n_detectors):
    raw_id_2 = chn_id[j2]
    neg_vals = np.array([])
    pos_vals = np.array([])
    neg_mean = float('nan')
    pos_mean = float('nan')
    neg_stdev = float('nan')
    pos_stdev = float('nan')
    
# Skip if self-interaction or either channel is missing
    if raw_id_1 == raw_id_2:
        print(f"Self-interaction at ({j1}, {j2}) ignored.")
    elif raw_id_2 in skipped_channels:
        print(f"Skipping job ({j1}, {j2}) due to missing channel(s).")
    else:
        cuspEmax_selection_2 = EventSelector(
            table_path=f"ch{raw_id_2}/hit/",
            files=new_hit_list,
            ene_dataset="cuspEmax_ctc_cal",
            conditions=config.xtalk_flag_response_conditions,
            energy_range=(-99999, 100),
            idx = cuspEmax_selection_1.selected_idxs
        )
        print(f"response selection indices: {cuspEmax_selection_2.selected_idxs}")
        trapTmax_1 = trapTmax_1_presel[cuspEmax_selection_2.selected_idxs] #Beware: .selected_idxs reuses indices from original array!
        
        table_2 = lh5.read(f"ch{raw_id_2}/dsp/", new_dsp_list, field_mask=["trapTmin", "trapTmax"], idx=cuspEmax_selection_2.selected_idxs)
        trapTmin_2 = table_2["trapTmin"].nda
        trapTmax_2 = table_2["trapTmax"].nda
        
        print(f"len of trapTmax_1: {len(trapTmax_1)}; len of trapTmax_2: {len(trapTmax_2)}")

        neg_vals = np.asarray(xtalk_element(trapTmax_1, trapTmin_2, negative_baseline[j2]))
        pos_vals = np.asarray(xtalk_element(trapTmax_1, trapTmax_2, positive_baseline[j2]))

        neg_mean = np.mean(neg_vals) if neg_vals.size else float('nan')
        pos_mean = np.mean(pos_vals) if pos_vals.size else float('nan')
        neg_stdev = np.std(neg_vals) if neg_vals.size else float('nan')
        pos_stdev = np.std(pos_vals) if pos_vals.size else float('nan')

    neg_counts, neg_bins, neg_counts_restrained, neg_bins_restrained = build_hist(neg_vals, neg_mean, neg_stdev, "neg")
    pos_counts, pos_bins, pos_counts_restrained, pos_bins_restrained = build_hist(pos_vals, pos_mean, pos_stdev, "pos")

# Save histogram numeric data to compressed .npz
    out_path = OUTDIR / f"xtalk_{j1}_{j2}.npz"
    np.savez_compressed(
        out_path,
        neg_counts=neg_counts,
        neg_counts_restrained=neg_counts_restrained,
        neg_bins=neg_bins,
        neg_bins_restrained=neg_bins_restrained,
        pos_counts=pos_counts,
        pos_counts_restrained=pos_counts_restrained,
        pos_bins=pos_bins,
        pos_bins_restrained=pos_bins_restrained,
        neg_vals=neg_vals,
        pos_vals=pos_vals
    )

    print(f"histogram ({j1},{j2}) saved: {out_path}.")
