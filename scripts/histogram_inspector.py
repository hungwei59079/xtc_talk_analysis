import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "j1", 
    help="Trigger channel index"
)
parser.add_argument(
    "j2",
    help="Response channel index"
)
parser.add_argument(
    "--label",
    default="neg",
    help="neg or pos",
)
parser.add_argument(
    "--min",
    help="min for visualized histogram"
)
parser.add_argument(
    "--max",
    help="max for visualized histogram"
)
args = parser.parse_args()

j1 = args.j1
j2 = args.j2
label = args.label
xmin = float(args.min) if args.min is not None else None
xmax = float(args.max) if args.max is not None else None

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_DIR = REPO_ROOT / "temp_results" / "fit_results"
HISTO_DIR = REPO_ROOT / "temp_results" / "histograms"
fit_npz_path = FIT_DIR / f"fit_{label}_{j1}_{j2}.npz"
histo_npz_path = HISTO_DIR / f"xtalk_{j1}_{j2}.npz"
OUT_DIR = REPO_ROOT / "results" / "Inspected_histograms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with np.load(histo_npz_path) as data:
    neg_counts = data["neg_counts"]
    neg_bins = data["neg_bins"]
    pos_counts = data["pos_counts"]
    pos_bins = data["pos_bins"]

with np.load(fit_npz_path) as data:
    success = bool(data["success"])
    reason = data["reason"]
    if success:
        A = float(data["A"])
    mu = float(data["mu"])
    sigma = float(data["sigma"])
    total_events = int(data["total_events"])

if label == "neg":
    x = 0.5 * (neg_bins[1:] + neg_bins[:-1])
    y = neg_counts
    bins = neg_bins
elif label == "pos":
    x = 0.5 * (pos_bins[1:] + pos_bins[:-1])
    y = pos_counts
    bins = pos_bins
else:
    raise ValueError("Incorrect label")

if not success:
    print(f"No plot due to {reason}")
    plt.figure()
    plt.bar(x, y, width=np.diff(bins), alpha=0.5, label=f"data (N={total_events})")
    plt.title(f"{label} histogram j1={j1}, j2={j2}")
    plt.legend()
    plt.show()
    plt.savefig(OUT_DIR / f"{label}_histogram_{j1},{j2}_.png")
    sys.exit(0)

# Apply x-range restriction if requested
if xmin is not None or xmax is not None:
    mask_range = np.ones_like(x, dtype=bool)
    if xmin is not None:
        mask_range &= (x >= xmin)
    if xmax is not None:
        mask_range &= (x <= xmax)

    # Apply mask to histogram values
    x = x[mask_range]
    y = y[mask_range]

    # bins has length N+1; select corresponding edges
    idx = np.where(mask_range)[0]
    bins = bins[idx[0] : idx[-1] + 2]

mask = y > 0.05 * np.max(y)
x_fit = x[mask]

plt.figure()
plt.bar(x, y, width=np.diff(bins), alpha=0.5, label=f"data (N={total_events})")
x_dense = np.linspace(min(x_fit), max(x_fit), 300)
plt.plot(
    x_dense,
    gaussian(x_dense, A, mu, sigma),
    'r-',
    label=f"Fit μ={mu:.3f}, σ={sigma:.3f}"
)
plt.title(f"{label} histogram j1={j1}, j2={j2}")
plt.legend()
plt.show()
plt.savefig(OUT_DIR / f"{label}_histogram_{j1},{j2}.png")

