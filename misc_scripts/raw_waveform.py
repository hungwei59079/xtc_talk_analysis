import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from lgdo import lh5
from pathlib import Path
from xtc_utils import files_and_chnid, relevant_events

parser = argparse.ArgumentParser()
parser.add_argument(
    "j1", 
    help="Trigger channel index"
)
parser.add_argument(
    "j2",
    help="Response channel index"
)
args = parser.parse_args()

j1 = int(args.j1)
j2 = int(args.j2)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "temp_results"/ "parameters"
CONFIG_PATH = REPO_ROOT / "xtc_config.json"
OUT_DIR = REPO_ROOT / "results" / "waveforms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

raw_dir = "/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/ref-raw/generated/tier/raw/xtc/p08/r015/"
skipped_channels = set(np.load(PARAMS_PATH / "skipped_channels.npy"))

new_hit_list, new_dsp_list, chn_id = files_and_chnid(CONFIG_PATH)
print("file listing complete")
raw_list = []
for path in new_hit_list:
    hit_file = Path(path).name
    raw_file = hit_file.replace("hit","raw")
    raw_path = raw_dir + raw_file
    raw_list.append(raw_path)
print("raw data file list construction is complete.")

"""
print(f"Checking {len(new_hit_list)} file pairs for event count consistency...")

mismatches = []

for i, (hit_path, raw_path) in enumerate(zip(new_hit_list, raw_list)):
    # We pick one channel to check. Assuming chn_id[i] corresponds to the files.
    # Usually, LH5 paths are formatted as 'chXXX/raw' or 'chXXX/hit'
    ch_name = f"ch{chn_id[0]}" 
    
    try:
        # Read only the table structure/length for the hit file
        # We read a small field like 'event_id' or 'timestamp' to get the count
        # Note: 'hit' files usually have a 'hit' group, 'raw' files have a 'raw' group
        hit_data = lh5.read(f"{ch_name}/hit", hit_path)
        raw_data = lh5.read(f"{ch_name}/raw", raw_path)

        if len(hit_data) != len(raw_data):
            print(f"[!] Mismatch found at index {i}:")
            print(f"    File: {Path(hit_path).name}")
            print(f"    Hit rows: {len(hit_data)} | Raw rows: {len(raw_data)}")
            mismatches.append((hit_path, len(hit_data), len(raw_data)))
            
    except Exception as e:
        print(f"[X] Error reading file pair {i}: {e}")

if not mismatches:
    print("✅ Success: All file pairs have matching event counts.")
else:
    print(f"❌ Found {len(mismatches)} pairs with mismatched event counts.")

"""

raw_id_1 = chn_id[j1]
raw_id_2 = chn_id[j2]

if raw_id_1 in skipped_channels:
    print(f"Trigger channel j1={j1} (raw {raw_id_1}) is in skipped_channels; Change a trigger channel.")
    sys.exit(0)
if raw_id_2 in skipped_channels:
    print(f"Response channel j2={j2} (raw {raw_id_2}) is in skipped_channels; Change a response channel.")
    sys.exit(0)
if raw_id_1 == raw_id_2:
    print("Are you stupid? Why using the same channels? (I'm joking)")
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
    # trapTmax_1 = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list, idx=idxs).nda
    print("Trigger channel event extraction complete.")
    trig_extract_complete = True
except Exception as e:
    print(f"Exception occurred at trigger channel {j1} extraction: {e}")
    trig_extract_complete = False

    
energy_2 = lh5.read(f"ch{raw_id_2}/hit/cuspEmax_ctc_cal", new_hit_list, idx=idxs).nda
print("response channel energy extraction complete.")
secondary_selection = (energy_2 < 100)
secondary_idxs = idxs[secondary_selection]
print("secondary selection complete.")

raw_waveform_table_1 = lh5.read(f"ch{raw_id_1}/raw/waveform_windowed", raw_list, idx=secondary_idxs)
raw_waveform_table_2 = lh5.read(f"ch{raw_id_2}/raw/waveform_windowed", raw_list, idx=secondary_idxs)
raw_waveform_1 = raw_waveform_table_1.values.nda
raw_waveform_2 = raw_waveform_table_2.values.nda

print(f"Generating plots for {len(raw_waveform_1)} events...")

for i in range(len(raw_waveform_1)):
    if i > 50:
        break
    # Extract specific waveforms for this event
    adc_1 = raw_waveform_1[i]
    adc_2 = raw_waveform_2[i]
    
    # Create x-axis (sample indices)
    x = np.arange(len(adc_1))
    
    # Create a canvas with 2 subplots (2 rows, 1 column)
    # sharex=True ensures both plots have the same time window
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot first waveform (e.g., from raw_waveform_1)
    ax1.plot(x, adc_1, label=f"Waveform 1 - Event {i}", color="blue", linewidth=1)
    ax1.set_ylabel("ADC")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Plot second waveform (e.g., from raw_waveform_2)
    ax2.plot(x, adc_2, label=f"Waveform 2 - Event {i}", color="red", linewidth=1)
    ax2.set_ylabel("ADC")
    ax2.set_xlabel("Sample Index")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure to OUT_DIR
    save_path = OUT_DIR / f"waveform_comparison_{j1}_{j2}_event_{i}.png"
    plt.savefig(save_path)
    
    # Close the figure to free up memory during the loop
    plt.close(fig)

print(f"Done. Plots saved to {OUT_DIR}")

"""
something_wrong = False
for path in raw_list:
    file_path = Path(path)
    if file_path.is_file():
        pass
    else:
        something_wrong = True
        print(f"The file {file_path} does not exist or is not a file.")

if not something_wrong:
    print("All files exist.")
"""
