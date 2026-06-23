"""Apply a crosstalk correction to a single detector's low-energy spectrum.

For the detector selected by ``detector_index``, this script selects events
with cuspEmax_ctc_cal < 25, computes a per-event crosstalk correction from the
energies deposited in all other detectors, and overlays the uncorrected and
corrected histograms.

The correction for event e is:

    correction[e] = - sum_{j != i} (C[j, i] / 100) * A_j[e] * delta_j[e]

where i is ``detector_index``, C[j, i] is the aggregated crosstalk matrix
(trigger j -> response i), A_j[e] is the cuspEmax energy of detector j in
event e, and delta_j[e] = 1 if A_j[e] > 50 else 0. C is divided by 100
because the crosstalk matrices are stored in percent.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

from xtc_utils import files_and_chnid, EventSelector, XTCConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the configuration JSON file.",
)
parser.add_argument(
    "--config_name",
    type=str,
    required=True,
    help="Name of the configuration to use from the JSON file.",
)
parser.add_argument(
    "detector_index",
    type=int,
    help="Index of the detector to correct (0-based index into the channel list).",
)
parser.add_argument(
    "--pos_matrix",
    type=str,
    required=True,
    help="Path to the positive crosstalk matrix CSV.",
)
parser.add_argument(
    "--neg_matrix",
    type=str,
    required=True,
    help="Path to the negative crosstalk matrix CSV.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Directory to save the output histogram. "
         "Defaults to <repo>/results/zero_correction.",
)
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(args.config_path)
OUT_DIR = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "zero_correction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds.
SELECTION_EMAX = 3  
SIGNAL_THRESHOLD = 50  
SIGNAL_MAX = 9999

config = XTCConfig(CONFIG_PATH, args.config_name)

new_hit_list, new_dsp_list, evt_list, chn_id = files_and_chnid(
    config, tiers=("hit", "dsp", "evt"), on_mismatch="drop"
)
n_det = len(chn_id)

detector_index = args.detector_index
if detector_index < 0 or detector_index >= n_det:
    raise ValueError(
        f"detector_index {detector_index} out of range. Valid range: 0-{n_det - 1}"
    )
detector = chn_id[detector_index]
print(f"Correcting detector index {detector_index} (channel ID: {detector})")

pos_matrix = np.loadtxt(args.pos_matrix, delimiter=",")
neg_matrix = np.loadtxt(args.neg_matrix, delimiter=",")
for name, m in (("pos_matrix", pos_matrix), ("neg_matrix", neg_matrix)):
    if m.shape != (n_det, n_det):
        raise ValueError(
            f"{name} has shape {m.shape}, expected ({n_det}, {n_det}) "
            f"to match the {n_det} detectors from the config."
        )

abs_pos = np.where(np.isnan(pos_matrix), -np.inf, np.abs(pos_matrix))
abs_neg = np.where(np.isnan(neg_matrix), -np.inf, np.abs(neg_matrix))
agg_matrix = np.where(abs_pos >= abs_neg, pos_matrix, neg_matrix)

crosstalk_col = np.nan_to_num(
    agg_matrix[:, detector_index], nan=0.0, posinf=0.0, neginf=0.0
) / 100.0

event_selection = EventSelector(
    table_path=f"evt/coincident/",
    files=evt_list,
    ene_dataset="geds",
    conditions={"puls" : False}
)
broad_selection = EventSelector(
    table_path=f"ch{detector}/hit/",
    files=new_hit_list,
    ene_dataset="cuspEmax_ctc_cal",
    idx=event_selection.selected_idxs,
)
raw_energies = broad_selection.selected_energies
raw_idxs = broad_selection.selected_idxs

rough_range = (-50, 50)
counts, bin_edges = np.histogram(raw_energies, bins=1000, range=rough_range)

# Find the peak height and calculate the 5% threshold
max_counts = counts.max()
threshold = 0.05 * max_counts
above_thresh_indices = np.where(counts >= threshold)[0]

if len(above_thresh_indices) == 0:
    lower_bound, upper_bound = -10.0, 10.0
else:
    lower_bound = bin_edges[above_thresh_indices[0]]
    upper_bound = bin_edges[above_thresh_indices[-1] + 1]

print(f"Dynamic 5% threshold energy range for detector {detector}: ({lower_bound:.2f}, {upper_bound:.2f})")

# 6. Apply the dynamic mask to filter your final arrays
mask = (raw_energies >= lower_bound) & (raw_energies <= upper_bound)

uncorrected = raw_energies[mask]
selected_idxs = raw_idxs[mask]

correction = np.zeros(len(selected_idxs))
has_correction = np.zeros(len(selected_idxs), dtype=bool) # Track non-zero contributions. Should be removed in the future.
n_capped_total = 0
for j in range(n_det):
    if j == detector_index:
        continue
    # A zero coefficient contributes nothing; skip the (expensive) read.
    if crosstalk_col[j] == 0.0:
        continue
    try:
        donor = EventSelector(
            table_path=f"ch{chn_id[j]}/hit/",
            files=new_hit_list,
            ene_dataset="cuspEmax_ctc_cal",
            idx=selected_idxs,
        )
        a_j = np.nan_to_num(
            donor.energy_all_indexed, nan=0.0, posinf=0.0, neginf=0.0
        )
    except Exception as e:
        print(f"  detector {j} (ch{chn_id[j]}) unreadable, skipping: {e}")
        continue
    fired = a_j > SIGNAL_THRESHOLD
    over_cap = a_j > SIGNAL_MAX
    n_over_cap = int(np.count_nonzero(over_cap))
    if n_over_cap:
        n_capped_total += n_over_cap
        print(
            f"  [cap] detector {j} (ch{chn_id[j]}): {n_over_cap} events with "
            f"energy > {SIGNAL_MAX} (max {a_j.max():.4g}) excluded from the "
            f"correction -- likely saturated / mis-calibrated"
        )

    valid_correction_event = fired & ~over_cap # These two lines are for tracking nonzero contributions. 
    has_correction |= valid_correction_event # Should be removed in the future.

    delta_j = (fired & ~over_cap).astype(float)
    correction += crosstalk_col[j] * a_j * delta_j
    print(f"correction contribution from detector {j} completed")

# Uncomment these two line when debugging is done
# correction = -correction 
# corrected = uncorrected + correction

# ========== Non zero contributions tracking (for debugging) ==========
correction = -correction
corrected_full = uncorrected + correction
 
# Calculate proportions
n_total = len(selected_idxs)
n_corrected = np.count_nonzero(has_correction)
proportion = (n_corrected / n_total) * 100 if n_total > 0 else 0.0

print("\n" + "="*40)
print(f"Correction Summary for Detector {detector_index}:")
print(f"  Total events in peak window: {n_total}")
print(f"  Events receiving crosstalk correction: {n_corrected} ({proportion:.2f}%)")
print("="*40 + "\n")

# Filter arrays to keep ONLY events that received a nonzero correction
print(f"Number of events before filtering: {len(uncorrected)}")
uncorrected = uncorrected[has_correction]
corrected = corrected_full[has_correction]
selected_idxs = selected_idxs[has_correction]
print(f"Number of events after filtering to nonzero correction: {len(uncorrected)}")
# ===============================================

if n_capped_total:
    print(
        f"Total donor events excluded by the {SIGNAL_MAX} keV cap: "
        f"{n_capped_total} -- if this is non-zero, oversized finite donor "
        f"energies were the cause of the runaway correction."
    )
else:
    print(f"No donor energies exceeded the {SIGNAL_MAX} keV cap.")


# --- Step 6: overlay histogram -----------------------------------------------
lo = min(uncorrected.min(), corrected.min())
hi = max(uncorrected.max(), corrected.max())
bins = np.linspace(lo, hi, 5001)

plt.figure(figsize=(10, 6))
plt.hist(uncorrected, bins=bins, histtype="step", linewidth=1.5,
         label=f"Uncorrected (mean={np.mean(uncorrected):.3f}, count={len(uncorrected)})")
plt.hist(corrected, bins=bins, histtype="step", linewidth=1.5,
         label=f"Corrected (mean={np.mean(corrected):.3f}, count={len(corrected)})")
plt.xlabel("cuspEmax_ctc_cal energy")
plt.ylabel("Counts")
plt.title(f"Crosstalk correction - detector index {detector_index} (ch{detector})")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / f"zero_correction_detector_{detector_index}.png"
plt.savefig(out_path)
plt.close()
print(f"Histogram saved to: {out_path}")
