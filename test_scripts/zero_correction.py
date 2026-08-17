from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
from xtc_utils import get_test_args, files_and_chnid, EventSelector, XTCConfig

args = get_test_args(Path(sys.argv[1]), Path(sys.argv[0]).stem)
print(args)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(args["config_path"])
OUT_DIR = Path(args["out_dir"]) if args["out_dir"] else REPO_ROOT / "results" / "zero_correction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

config = XTCConfig(CONFIG_PATH, args["config_name"])

new_hit_list, new_dsp_list, evt_list, tcm_list, chn_id = files_and_chnid(
    config, tiers=("hit", "dsp", "evt", "tcm"), on_mismatch="drop"
)
n_det = len(chn_id)

detector_index = args["detector_index"]
trigger_index = args["trigger_index"]
detector = chn_id[detector_index]
trigger = chn_id[trigger_index]

pos_matrix = np.loadtxt(Path(REPO_ROOT / args["pos_matrix"]), delimiter=",")
neg_matrix = np.loadtxt(Path(REPO_ROOT / args["neg_matrix"]), delimiter=",")
for name, m in (("pos_matrix", pos_matrix), ("neg_matrix", neg_matrix)):
    if m.shape != (n_det, n_det):
        raise ValueError(
            f"{name} has shape {m.shape}, expected ({n_det}, {n_det}) "
            f"to match the {n_det} detectors from the config."
        )

abs_pos = np.where(np.isnan(pos_matrix), -np.inf, np.abs(pos_matrix))
abs_neg = np.where(np.isnan(neg_matrix), -np.inf, np.abs(neg_matrix))
agg_matrix = np.where(abs_pos >= abs_neg, pos_matrix, neg_matrix)
crosstalk_col = np.nan_to_num(agg_matrix[:, detector_index], nan=0.0, posinf=0.0, neginf=0.0) / 100.0

pulser_selection = EventSelector(
    table_path=f"evt/coincident/",
    files=evt_list,
    ene_dataset="geds",
    conditions={"puls" : False}
)
trigger_selection = EventSelector(
    table_path=f"ch{trigger}/hit/",
    files=new_hit_list,
    ene_dataset="cuspEmax_ctc_cal",
    energy_range=(args["trigger_threshold"], args["trigger_max"]),
    idx=pulser_selection.selected_idxs,
)
response_selection = EventSelector(
    table_path=f"ch{detector}/hit/",
    files=new_hit_list,
    ene_dataset="cuspEmax_ctc_cal",
    energy_range=(-10, 50),
    idx=trigger_selection.selected_idxs,
)

response_energy = response_selection.selected_energies
trigger_energy = trigger_selection.energy_all[response_selection.selected_idxs]

if len(response_energy) != len(trigger_energy):
    raise ValueError(
        f"Length mismatch: response_energy ({len(response_energy)}) "
        f"and trigger_energy ({len(trigger_energy)}) must be the same."
    )

crosstalk_value = crosstalk_col[trigger_index]

print(f"Crosstalk value: {crosstalk_value:.6f} ")

correction = crosstalk_value * trigger_energy
correction = -correction
corrected_energy = response_energy + correction

lo = min(response_energy.min(), corrected_energy.min())
hi = max(response_energy.max(), corrected_energy.max())
bins = np.linspace(lo, hi, 501)

plt.figure(figsize=(10, 6))
plt.hist(response_energy, bins=bins, histtype="step", linewidth=1.5,
         label=f"response_energy (mean={np.mean(response_energy):.3f}, count={len(response_energy)})")
plt.hist(corrected_energy, bins=bins, histtype="step", linewidth=1.5,
         label=f"corrected_energy (mean={np.mean(corrected_energy):.3f}, count={len(corrected_energy)})")
plt.xlabel("cuspEmax_ctc_cal energy")
plt.ylabel("Counts")
plt.title(f"Crosstalk correction - detector index {detector_index} (ch{detector})")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / f"zero_correction_detector_{detector_index}.png"
plt.savefig(out_path)
plt.close()
print(f"Histogram saved to: {out_path}")