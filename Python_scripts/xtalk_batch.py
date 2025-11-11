# xtalk_batch.py

import sys
import os
import numpy as np
from lgdo import lh5
from self_defined_functions import files_and_chnid, relevant_events, xtalk_element
import matplotlib.pyplot as plt

# ---------- Setup ----------
new_hit_list, new_dsp_list, chn_id = files_and_chnid()
print("File listing complete.")

# Load parameters
os.chdir("../parameters")
baseline_energy = np.load("baseline_energy.npy")
skipped_channels = set(np.load("skipped_channels.npy"))

os.chdir("../")

j1 = int(sys.argv[1])
raw_id_1 = chn_id[j1]

if raw_id_1 in skipped_channels:
    print(f"Trigger channel j1={j1} (raw {raw_id_1}) is in skipped_channels; saving empty histograms for all j2.")
    for j2 in range(0, 101):
        out_base = f"histograms/xtalk_{j1}_{j2}"
        np.savez_compressed(
            out_base + ".npz",
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
        flag_datasets=["is_discharge", "is_valid_0vbb_old"],
        conditions={"is_discharge": False, "is_valid_0vbb_old": True},
        energy_range=(1500, 4500),
        return_index=True
    )
    trapTmax_1 = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list, idx=idxs).nda
    print("Trigger channel event extraction complete.")
    trig_extract_complete = True
except Exception as e:
    print(f"Exception occurred at trigger channel {j1} extraction.")
    trig_extract_complete = False

NBINS = 700

for j2 in range(0,101):
    raw_id_2 = chn_id[j2]
    neg_vals = np.array([])
    pos_vals = np.array([])
    
# Skip if self-interaction or either channel is missing
    if raw_id_1 == raw_id_2:
        print(f"Self-interaction at ({j1}, {j2}) ignored.")
    elif raw_id_2 in skipped_channels:
        print(f"Skipping job ({j1}, {j2}) due to missing channel(s).")
    else:
        energy_2 = lh5.read(f"ch{raw_id_2}/hit/cuspEmax_ctc_cal", new_hit_list, idx=idxs).nda
        secondary_selection = (energy_2 < 100)
        secondary_idxs = idxs[secondary_selection]
        selected_trapTmax_1 = trapTmax_1[secondary_selection]

        table_2 = lh5.read(f"ch{raw_id_2}/dsp/", new_dsp_list, field_mask=["trapTmin", "trapTmax"], idx=secondary_idxs)
        trapTmin_2 = table_2["trapTmin"].nda
        trapTmax_2 = table_2["trapTmax"].nda

        neg_vals = np.asarray(xtalk_element(selected_trapTmax_1, trapTmin_2, baseline_energy[j2]))
        pos_vals = np.asarray(xtalk_element(selected_trapTmax_1, trapTmax_2, baseline_energy[j2]))

    # Build histograms (counts + bin edges). For empty arrays save empty arrays.
    if neg_vals.size:
        neg_counts, neg_bins = np.histogram(neg_vals, bins=NBINS, range=(min(neg_vals),0.5))
    else:
        neg_counts = np.array([], dtype=int)
        neg_bins = np.array([])

    if pos_vals.size:
        pos_counts, pos_bins = np.histogram(pos_vals, bins=NBINS, range=(-0.5,max(pos_vals)))
    else:
        pos_counts = np.array([], dtype=int)
        pos_bins = np.array([])

    # Save histogram numeric data to compressed .npz
    out_base = f"histograms/xtalk_{j1}_{j2}"
    np.savez_compressed(
        out_base + ".npz",
        neg_counts=neg_counts,
        neg_bins=neg_bins,
        pos_counts=pos_counts,
        pos_bins=pos_bins,
        # also save raw values in case you later want to re-bin / inspect them
        neg_vals=neg_vals,
        pos_vals=pos_vals
    )

    print(f"histogram ({j1},{j2}) saved: {out_base}.npz.")
