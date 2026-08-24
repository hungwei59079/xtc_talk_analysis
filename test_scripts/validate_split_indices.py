# validate_split_indices.py
#
# Validate the below/above event indices from a xtalk_split_*.npz by re-reading
# trapTmax/trapTmin straight from the dsp files (no event selection), evaluating
# the neg xtalk values at those indices, and histogramming the two categories.
# If the indices are correct the two histograms must tile the restrained range.

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lgdo import lh5
from xtc_utils import files_and_chnid, xtalk_element, XTCConfig
from pathlib import Path

parser = argparse.ArgumentParser(description="Validate below/above indices of a xtalk_split_*.npz.")
parser.add_argument("npz", type=str, help="Path to the xtalk_split_*.npz file.")
parser.add_argument("--config_path", type=str, default="configs/xtc_config.json")
parser.add_argument("--config_name", type=str, default="xtc_p16_ssc",
                    help="Must match the dataset the npz was produced from.")
parser.add_argument("--temp_result_dir", type=str, default=None,
                    help="temp_results dir holding parameters/ (default: npz's parent's parent).")
parser.add_argument("--nbins", type=int, default=700)
parser.add_argument("--sigma", type=float, default=3.0)
parser.add_argument("--outdir", type=str, default=None)
args = parser.parse_args()

npz_path = Path(args.npz)
data = np.load(npz_path, allow_pickle=True)
j1, j2 = int(data["j1"]), int(data["j2"])
raw_id_1, raw_id_2 = int(data["raw_id_1"]), int(data["raw_id_2"])
below_idx, above_idx = data["below_idx"], data["above_idx"]
cut = float(data["cut"])

outdir = Path(args.outdir) if args.outdir else npz_path.parent
outdir.mkdir(parents=True, exist_ok=True)
temp_result_dir = Path(args.temp_result_dir) if args.temp_result_dir else npz_path.parent.parent

# Restrained range from the stored values (same convention as inspect_split.py).
stored_neg = np.concatenate([data["below_neg_vals"], data["above_neg_vals"]])
mean, std = stored_neg.mean(), stored_neg.std()
lo, hi = mean - args.sigma * std, mean + args.sigma * std
print(f"stored neg_vals: total={stored_neg.size}, mean={mean:.4f}, std={std:.4f}")
print(f"restrained range=({lo:.4f}, {hi:.4f})")

# ---------- Re-read the dsp files, no selection ----------
config = XTCConfig(Path(args.config_path), args.config_name)
new_hit_list, new_dsp_list, chn_id = files_and_chnid(config)

if chn_id[j1] != raw_id_1 or chn_id[j2] != raw_id_2:
    raise SystemExit(
        f"Config mismatch: config '{args.config_name}' gives chn_id[{j1}]={chn_id[j1]}, "
        f"chn_id[{j2}]={chn_id[j2]}, but the npz was made with raw ids "
        f"({raw_id_1}, {raw_id_2}). Indices would be meaningless."
    )

negative_baseline = np.load(temp_result_dir / "parameters" / "negative_baseline.npy")

trapTmax_1_full = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list).nda
trapTmin_2_full = lh5.read(f"ch{raw_id_2}/dsp/trapTmin", new_dsp_list).nda
n_total = trapTmax_1_full.size
print(f"full dsp arrays: trigger={n_total}, response={trapTmin_2_full.size}")
print(f"max index used: below={below_idx.max()}, above={above_idx.max()}")

neg_below = np.asarray(xtalk_element(trapTmax_1_full[below_idx], trapTmin_2_full[below_idx],
                                     negative_baseline[j2]))
neg_above = np.asarray(xtalk_element(trapTmax_1_full[above_idx], trapTmin_2_full[above_idx],
                                     negative_baseline[j2]))

# Recomputed values must reproduce the stored ones exactly.
ok_below = np.allclose(neg_below, data["below_neg_vals"], equal_nan=True)
ok_above = np.allclose(neg_above, data["above_neg_vals"], equal_nan=True)
print(f"recomputed == stored: below={ok_below}, above={ok_above}")
print(f"below: n={neg_below.size}, range=({neg_below.min():.4f}, {neg_below.max():.4f})")
print(f"above: n={neg_above.size}, range=({neg_above.min():.4f}, {neg_above.max():.4f})")

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, vals, label in ((axes[0], neg_below, "below"), (axes[1], neg_above, "above")):
    ax.hist(vals, bins=args.nbins, range=(lo, hi))
    ax.axvline(cut, color="red", ls="--", label=f"cut = {cut}")
    ax.set_xlim(lo, hi)
    ax.set_title(f"{label} cut (N={vals.size})")
    ax.set_xlabel("neg xtalk [%]")
    ax.set_ylabel("counts")
    ax.legend()

fig.suptitle(f"neg xtalk recomputed from dsp via split indices, j1={j1}, j2={j2}")
fig.tight_layout()
out_path = outdir / f"validate_split_indices_{j1}_{j2}.png"
fig.savefig(out_path)
plt.close(fig)
print(f"Saved: {out_path}")
