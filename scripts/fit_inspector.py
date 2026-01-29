import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
from xtc_utils import files_and_chnid

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reason",
    help="reason of failure one wants to inspect",
)
args = parser.parse_args()

reason_dict = {None : None,
               "no_stats" : "no_stats",
               "low_stats" : "low_stats",
              "delta" : "Optimal parameters not found: Number of calls to function has reached maxfev = 800.",
              "sharp" : "ok but with insufficient points"}
actual_reason = reason_dict[args.reason]

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "xtc_config.json"
IN_DIR = REPO_ROOT / "temp_results" / "fit_results"
SKIP_DIR = REPO_ROOT / "temp_results" / "parameters"
OUT_DIR = REPO_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

neg_xtalk_matrix = np.full((101, 101),np.nan)
pos_xtalk_matrix = np.full((101, 101),np.nan)
neg_fail_list = []
pos_fail_list = []
scenarios = set()

new_hit_list, new_dsp_list, chn_id = files_and_chnid(CONFIG_PATH)
skipped_channels = set(np.load(SKIP_DIR / "skipped_channels.npy"))

for j1 in range(101):
    raw_id_1 = chn_id[j1]
    for j2 in range(101):
        raw_id_2 = chn_id[j2]
        npz_path = IN_DIR / f"fit_neg_{j1}_{j2}.npz"
        try:
            with np.load(npz_path) as data:
                success = data["success"]
                reason = str(data["reason"])
                mu = float(data["mu"])
                sigma = float(data["sigma"])
        except Exception as e:
            print(f"Exception {e} occurs. Skipping.")
            continue
        if reason not in scenarios:
            scenarios.add(reason)
        neg_xtalk_matrix[j1,j2] = mu
        if neg_xtalk_matrix[j1,j2] < -2:
            print(f"neg xtalk value for {(j1,j2)} is {neg_xtalk_matrix[j1,j2]}")
        if reason == actual_reason:
            if raw_id_1 not in skipped_channels and raw_id_2 not in skipped_channels:
                if raw_id_1 != raw_id_2:
                    neg_fail_list.append(f"{j1},{j2}")

        npz_path = IN_DIR / f"fit_pos_{j1}_{j2}.npz"
        with np.load(npz_path) as data:
            success = data["success"]
            reason = str(data["reason"])
            mu = float(data["mu"])
            sigma = float(data["sigma"])
        if reason not in scenarios:
            scenarios.add(reason)
        pos_xtalk_matrix[j1,j2] = mu
        if pos_xtalk_matrix[j1,j2] > 0.05:
            print(f"pos xtalk value for {(j1,j2)} is {pos_xtalk_matrix[j1,j2]}")
        if reason == actual_reason:
            if raw_id_1 not in skipped_channels and raw_id_2 not in skipped_channels:
                if raw_id_1 != raw_id_2:
                    pos_fail_list.append(f"{j1},{j2}")

# -----------Negative Plot ---------------

# Set up value range and colormap
vmin = -0.3  # red
vmax = 0.1   # blue
cmap = plt.cm.jet_r  # matches your uploaded colorbar

# Create the plot
plt.figure(figsize=(8, 6))
im = plt.imshow(neg_xtalk_matrix, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label('Negative Xtalk Value (%)')

# Axis labels
plt.xlabel('Response Channel Index')
plt.ylabel('Trigger Channel Index')
plt.title('Negative Crosstalk Matrix Heatmap')

# Save & display
plt.tight_layout()
plt.savefig(OUT_DIR / "Neg_xtk_map_fitted.png")

# -----------Positive Plot----------------

# Set up value range and colormap
vmin = -0.07  # red
vmax = 0.3   # blue
cmap = plt.cm.jet_r  # matches your uploaded colorbar

# Create the plot
plt.figure(figsize=(8, 6))
im = plt.imshow(pos_xtalk_matrix, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label('Positive Xtalk Value (%)')

# Axis labels
plt.xlabel('Response Channel Index')
plt.ylabel('Trigger Channel Index')
plt.title('Positive Crosstalk Matrix Heatmap')
plt.tight_layout()
plt.savefig(OUT_DIR / "Pos_xtk_map_fitted.png")

if actual_reason:
    print("negative fail list:\n")
    print(neg_fail_list)
    print("positive fail list:\n")
    print(pos_fail_list)