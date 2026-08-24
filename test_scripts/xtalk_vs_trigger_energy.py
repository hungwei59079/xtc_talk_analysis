# xtalk_vs_trigger_energy.py
#
# Compare the trigger-detector (j1) trapTmax distribution between the below/above
# cut categories, and show the full (trapTmax, neg xtalk) 2D distribution.
# Events beyond the 3 sigma xtalk range are trimmed.

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lgdo import lh5
from xtc_utils import files_and_chnid, XTCConfig
from pathlib import Path

parser = argparse.ArgumentParser(description="Trigger trapTmax vs neg xtalk for a xtalk_split_*.npz.")
parser.add_argument("npz", type=str, help="Path to the xtalk_split_*.npz file.")
parser.add_argument("--config_path", type=str, default="configs/xtc_config.json")
parser.add_argument("--config_name", type=str, default="xtc_p16_ssc")
parser.add_argument("--sigma", type=float, default=3.0, help="Trim range = mean +/- sigma*std.")
parser.add_argument("--ebins", type=int, default=150, help="Bins for the 1D energy histograms.")
parser.add_argument("--bins2d", type=int, default=200, help="Bins per axis for the 2D histogram.")
parser.add_argument("--erange", type=float, nargs=2, default=None, help="trapTmax range override.")
parser.add_argument("--outdir", type=str, default=None)
args = parser.parse_args()

npz_path = Path(args.npz)
data = np.load(npz_path, allow_pickle=True)
j1, j2 = int(data["j1"]), int(data["j2"])
raw_id_1, raw_id_2 = int(data["raw_id_1"]), int(data["raw_id_2"])
below_idx, above_idx = data["below_idx"], data["above_idx"]
below_neg, above_neg = data["below_neg_vals"], data["above_neg_vals"]
cut = float(data["cut"])

outdir = Path(args.outdir) if args.outdir else npz_path.parent
outdir.mkdir(parents=True, exist_ok=True)

# 3 sigma range from the stored xtalk values.
stored_neg = np.concatenate([below_neg, above_neg])
mean, std = stored_neg.mean(), stored_neg.std()
lo, hi = mean - args.sigma * std, mean + args.sigma * std
print(f"neg xtalk: total={stored_neg.size}, mean={mean:.4f}, std={std:.4f}")
print(f"{args.sigma} sigma range=({lo:.4f}, {hi:.4f})")

# ---------- Read trigger trapTmax at the split indices ----------
config = XTCConfig(Path(args.config_path), args.config_name)
new_hit_list, new_dsp_list, chn_id = files_and_chnid(config)
if chn_id[j1] != raw_id_1 or chn_id[j2] != raw_id_2:
    raise SystemExit(f"Config mismatch: chn_id[{j1}]={chn_id[j1]}, chn_id[{j2}]={chn_id[j2]} "
                     f"vs npz raw ids ({raw_id_1}, {raw_id_2}).")

# below_idx/above_idx are each ascending, so they work directly as lh5 idx.
E_below = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list, idx=below_idx).nda
E_above = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list, idx=above_idx).nda
print(f"read trigger trapTmax: below={E_below.size}, above={E_above.size}")

# Trim events with extreme xtalk values.
keep_b = (below_neg >= lo) & (below_neg <= hi)
keep_a = (above_neg >= lo) & (above_neg <= hi)
Eb, xb = E_below[keep_b], below_neg[keep_b]
Ea, xa = E_above[keep_a], above_neg[keep_a]
print(f"after trim: below={Eb.size} (dropped {(~keep_b).sum()}), "
      f"above={Ea.size} (dropped {(~keep_a).sum()})")

E_all = np.concatenate([Eb, Ea])
x_all = np.concatenate([xb, xa])

e_lo, e_hi = args.erange if args.erange else (E_all.min(), E_all.max())
print(f"trapTmax range=({e_lo:.1f}, {e_hi:.1f}); p0.1/p50/p99.9 = "
      f"{np.percentile(E_all, 0.1):.1f}/{np.percentile(E_all, 50):.1f}/"
      f"{np.percentile(E_all, 99.9):.1f}")

# ---------- 1D: trigger energy per category ----------
ebins = np.linspace(e_lo, e_hi, args.ebins + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, density in ((axes[0], False), (axes[1], True)):
    ax.hist(Eb, bins=ebins, density=density, histtype="step", label=f"below cut (N={Eb.size})")
    ax.hist(Ea, bins=ebins, density=density, histtype="step", label=f"above cut (N={Ea.size})")
    ax.set_xlabel("trigger trapTmax")
    ax.legend()
axes[0].set_yscale("log")
axes[0].set_ylabel("counts")
axes[0].set_title("raw counts (log y)")
axes[1].set_ylabel("probability density")
axes[1].set_title("normalized (shape comparison)")
fig.suptitle(f"trigger (j1={j1}) trapTmax by xtalk category, cut={cut}")
fig.tight_layout()
out1 = outdir / f"trigger_energy_by_category_{j1}_{j2}.png"
fig.savefig(out1)
plt.close(fig)
print(f"Saved: {out1}")

# ---------- 2D: xtalk vs trigger energy, all events ----------
fig2, ax2 = plt.subplots(figsize=(9, 6))
h = ax2.hist2d(E_all, x_all, bins=[args.bins2d, args.bins2d],
               range=[[e_lo, e_hi], [lo, hi]], norm=LogNorm())
fig2.colorbar(h[3], ax=ax2, label="counts")
ax2.axhline(cut, color="red", ls="--", label=f"cut = {cut}")
ax2.set_xlabel("trigger trapTmax")
ax2.set_ylabel("neg xtalk [%]")
ax2.set_title(f"neg xtalk vs trigger trapTmax, j1={j1}, j2={j2} (N={E_all.size})")
ax2.legend()
fig2.tight_layout()
out2 = outdir / f"xtalk_vs_trigger_energy_2d_{j1}_{j2}.png"
fig2.savefig(out2)
plt.close(fig2)
print(f"Saved: {out2}")
