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
SELECTION_EMAX = 25   # keep events with detector-i energy below this
SIGNAL_THRESHOLD = 50  # a detector "fired" (delta_j = 1) above this energy

config = XTCConfig(CONFIG_PATH, args.config_name)

new_hit_list, new_dsp_list, chn_id = files_and_chnid(config)
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

crosstalk_col = np.nan_to_num(agg_matrix[:, detector_index], nan=0.0) / 100.0

target_selection = EventSelector(
    table_path=f"ch{detector}/hit/",
    files=new_hit_list,
    ene_dataset="cuspEmax_ctc_cal",
    energy_range=(-np.inf, SELECTION_EMAX),
)
uncorrected = target_selection.selected_energies
selected_idxs = target_selection.selected_idxs
print(f"Selected {len(uncorrected)} events with cuspEmax_ctc_cal < {SELECTION_EMAX}")

# --- Step 4: per-event crosstalk correction ----------------------------------
# correction[e] = - sum_{j != i} (C[j,i]/100) * A_j[e] * delta_j[e]
correction = np.zeros(len(selected_idxs))
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
        # energy_all_indexed is aligned event-by-event with selected_idxs.
        a_j = np.nan_to_num(donor.energy_all_indexed, nan=0.0)
    except Exception as e:
        print(f"  detector {j} (ch{chn_id[j]}) unreadable, skipping: {e}")
        continue
    delta_j = (a_j > SIGNAL_THRESHOLD).astype(float)
    correction += crosstalk_col[j] * a_j * delta_j
    print(f"correction contribution from detector {j} completed")

correction = -correction
corrected = uncorrected + correction


# --- Step 6: overlay histogram -----------------------------------------------
lo = min(uncorrected.min(), corrected.min())
hi = max(uncorrected.max(), corrected.max())
bins = np.linspace(lo, hi, 101)

plt.figure(figsize=(10, 6))
plt.hist(uncorrected, bins=bins, alpha=0.5,
         label=f"Uncorrected (mean={np.mean(uncorrected):.3f})")
plt.hist(corrected, bins=bins, alpha=0.5,
         label=f"Corrected (mean={np.mean(corrected):.3f})")
plt.xlabel("cuspEmax_ctc_cal energy")
plt.ylabel("Counts")
plt.title(f"Crosstalk correction - detector index {detector_index} (ch{detector})")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / f"zero_correction_detector_{detector_index}.png"
plt.savefig(out_path)
plt.close()
print(f"Histogram saved to: {out_path}")
