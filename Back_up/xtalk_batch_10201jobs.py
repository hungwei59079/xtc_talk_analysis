# xtalk_batch.py

import sys
import os
import numpy as np
from lgdo import lh5
from self_defined_functions import files_and_chnid, relevant_events, xtalk_element

# ---------- Setup ----------
new_hit_list, new_dsp_list, chn_id = files_and_chnid()

# Load parameters
os.chdir("../parameters")
baseline_energy = np.load("baseline_energy.npy")
skipped_channels = set(np.load("skipped_channels.npy"))

os.chdir("../")

# ---------- Job index -> (j1, j2 block) mapping ----------
job_index = int(sys.argv[1])

matrix_size = 101
j1 = job_index // matrix_size
j2 = job_index % matrix_size

# Get corresponding raw channel IDs
raw_id_1 = chn_id[j1]
raw_id_2 = chn_id[j2]

# Default values
neg_matrix_element = np.nan
pos_matrix_element = np.nan

# Skip if self-interaction or either channel is missing
if raw_id_1 == raw_id_2:
    print(f"Self-interaction at ({j1}, {j2}) ignored.")
elif raw_id_1 in skipped_channels or raw_id_2 in skipped_channels:
    print(f"Skipping job ({j1}, {j2}) due to missing channel(s).")
else:
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
        energy_2 = lh5.read(f"ch{raw_id_2}/hit/cuspEmax_ctc_cal", new_hit_list, idx=idxs).nda

        secondary_selection = (energy_2 < 100)
        secondary_idxs = idxs[secondary_selection]
        selected_trapTmax_1 = trapTmax_1[secondary_selection]

        table_2 = lh5.read(f"ch{raw_id_2}/dsp/", new_dsp_list, field_mask=["trapTmin", "trapTmax"], idx=secondary_idxs)
        trapTmin_2 = table_2["trapTmin"].nda
        trapTmax_2 = table_2["trapTmax"].nda

        neg_matrix_element = np.mean(xtalk_element(selected_trapTmax_1, trapTmin_2, baseline_energy[j2]))
        pos_matrix_element = np.mean(xtalk_element(selected_trapTmax_1, trapTmax_2, baseline_energy[j2]))
    except Exception as e:
        print(f"Exception occurred in job ({j1}, {j2}): {e}")

# Save results (including skipped jobs with np.nan)
with open(f"matrix_elements/xtalk_{j1}_{j2}.txt", "w") as f:
    f.write(f"{neg_matrix_element},{pos_matrix_element}\n")
