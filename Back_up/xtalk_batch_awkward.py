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

# Load skipped channels if it exists
try:
    skipped_channels = set(np.load("skipped_channels.npy"))
except FileNotFoundError:
    skipped_channels = set()
os.chdir("../")

# ---------- Job index -> (j1, j2 block) mapping ----------
job_index = int(sys.argv[1])

ROWS = 101
COLS = 101

q = job_index // 10          # row index (which trigger channel)
r = job_index % 10           # which 10-wide column block within the row

j1 = q                       # fixed trigger index for this job
j2_start = 10 * r
j2_end = j2_start + 10
if r == 9:                   # last block includes the 101st (index 100) column
    j2_end += 1              # -> makes range(..., 101), i.e., includes 100

# Guard against accidental out-of-bounds
if not (0 <= j1 < ROWS):
    raise ValueError(f"Computed j1={j1} out of bounds for ROWS={ROWS} from job_index={job_index}")
if not (0 <= j2_start < COLS) or not (0 < j2_end <= COLS):
    raise ValueError(f"Computed j2 range [{j2_start}, {j2_end}) out of bounds for COLS={COLS}")

raw_id_1 = chn_id[j1]

# Helper: write NaNs for all j2 in this block
def write_block_nans():
    for j2 in range(j2_start, j2_end):
        with open(f"matrix_elements/xtalk_{j1}_{j2}.txt", "w") as f:
            f.write(f"{np.nan},{np.nan}\n")

# ---------- Fast exit if trigger channel is skipped ----------
if raw_id_1 in skipped_channels:
    print(f"Trigger channel j1={j1} (raw {raw_id_1}) is in skipped_channels; writing NaNs for j2 in [{j2_start}, {j2_end-1}].")
    write_block_nans()
    sys.exit(0)

# ---------- Extract trigger-channel events once ----------
extraction_complete = True
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
    # Read trapTmax for trigger channel once (indexed by the same idxs)
    trapTmax_1 = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list, idx=idxs).nda
    print(f"Trigger extraction complete for j1={j1} (raw {raw_id_1}); #events={len(idxs)}.")
except Exception as e:
    print(f"Trigger channel j1={j1} (raw {raw_id_1}) extraction failed: {e}")
    extraction_complete = False

# ---------- Loop over the j2 block ----------
for j2 in range(j2_start, j2_end):
    neg_matrix_element = np.nan
    pos_matrix_element = np.nan

    if not extraction_complete:
        # Already failed at the trigger; write NaNs and continue.
        with open(f"matrix_elements/xtalk_{j1}_{j2}.txt", "w") as f:
            f.write(f"{neg_matrix_element},{pos_matrix_element}\n")
        continue

    raw_id_2 = chn_id[j2]

    # Skip self-interaction
    if raw_id_1 == raw_id_2:
        print(f"Self-interaction at ({j1}, {j2}) ignored.")
    # Skip if victim channel is missing
    elif raw_id_2 in skipped_channels:
        print(f"Skipping ({j1}, {j2}) due to victim raw_id {raw_id_2} in skipped_channels.")
    else:
        try:
            # Read energy_2 for the same primary indices
            energy_2 = lh5.read(f"ch{raw_id_2}/hit/cuspEmax_ctc_cal", new_hit_list, idx=idxs).nda

            # Secondary selection depends on channel 2, so compute mask per j2
            secondary_selection = (energy_2 < 100)
            if not np.any(secondary_selection):
                print(f"No secondary events passed selection for ({j1}, {j2}).")
            else:
                secondary_idxs = idxs[secondary_selection]
                selected_trapTmax_1 = trapTmax_1[secondary_selection]

                table_2 = lh5.read(
                    f"ch{raw_id_2}/dsp/",
                    new_dsp_list,
                    field_mask=["trapTmin", "trapTmax"],
                    idx=secondary_idxs
                )
                trapTmin_2 = table_2["trapTmin"].nda
                trapTmax_2 = table_2["trapTmax"].nda

                neg_matrix_element = np.mean(
                    xtalk_element(selected_trapTmax_1, trapTmin_2, baseline_energy[j2])
                )
                pos_matrix_element = np.mean(
                    xtalk_element(selected_trapTmax_1, trapTmax_2, baseline_energy[j2])
                )
        except Exception as e:
            print(f"Exception at ({j1}, {j2}) with raw_ids ({raw_id_1}, {raw_id_2}): {e}")

    # Save result for this (j1, j2)
    with open(f"matrix_elements/xtalk_{j1}_{j2}.txt", "w") as f:
        f.write(f"{neg_matrix_element},{pos_matrix_element}\n")
