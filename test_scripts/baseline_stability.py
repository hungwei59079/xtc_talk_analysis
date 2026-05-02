import argparse
import fnmatch
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
RUN_RE = re.compile(r"r\d+")

parser = argparse.ArgumentParser(
    description="Plot the cross-talk value for a single (trigger, response) detector "
    "pair across all runs found under results/."
)
parser.add_argument("trigger", type=int, help="Trigger detector index (row).")
parser.add_argument("response", type=int, help="Response detector index (column).")
parser.add_argument(
    "--label",
    choices=["neg", "pos", "neg_restrained", "pos_restrained"],
    required=True,
)
parser.add_argument(
    "--results_dir",
    type=Path,
    default=DEFAULT_RESULTS_DIR,
    help=f"Directory containing per-job result subdirectories (default: {DEFAULT_RESULTS_DIR}).",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Path to save the plot. Defaults to results/baseline_stability_{label}_{i}_{j}.png.",
)
parser.add_argument(
    "--filter",
    dest="job_filter",
    type=str,
    default="*",
    help='Glob pattern matched against job-directory names, e.g. "job_ssc_*" or '
    '"job_phy_r00[1-3]". Default "*" includes all jobs.',
)
args = parser.parse_args()

csv_name = f"{args.label}_xtalk_matrix.csv"
sigma_csv_name = f"{args.label}_xtalk_matrix_sigma.csv"

points = []
for job_dir in sorted(args.results_dir.iterdir()):
    if not job_dir.is_dir():
        continue
    if not fnmatch.fnmatch(job_dir.name, args.job_filter):
        continue
    csv_path = job_dir / csv_name
    if not csv_path.exists():
        print(f"Skipping {job_dir.name}: {csv_name} not found.")
        continue

    matrix = np.loadtxt(csv_path, delimiter=",")
    value = matrix[args.trigger, args.response]

    sigma_path = job_dir / sigma_csv_name
    if sigma_path.exists():
        sigma_matrix = np.loadtxt(sigma_path, delimiter=",")
        sigma = sigma_matrix[args.trigger, args.response]
    else:
        print(f"  {job_dir.name}: {sigma_csv_name} not found, plotting without error bar.")
        sigma = np.nan

    run_match = RUN_RE.search(job_dir.name)
    run_label = run_match.group(0) if run_match else job_dir.name
    points.append((run_label, job_dir.name, value, sigma))

if not points:
    raise SystemExit(f"No matching results found in {args.results_dir}.")

points.sort(key=lambda p: p[0])
run_labels = [p[0] for p in points]
job_names = [p[1] for p in points]
values = np.array([p[2] for p in points])
sigmas = np.array([p[3] for p in points])

fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(points)), 4))
ax.errorbar(range(len(points)), values, yerr=sigmas, marker="o", linestyle="-", capsize=3)
ax.set_xticks(range(len(points)))
ax.set_xticklabels(run_labels, rotation=45, ha="right")
ax.set_xlabel("Run")
ax.set_ylabel(f"{args.label} cross-talk (trigger={args.trigger}, response={args.response})")
ax.set_title(f"Baseline stability: {args.label} xtalk for pair ({args.trigger}, {args.response})")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = args.output or args.results_dir / f"baseline_stability_{args.label}_{args.trigger}_{args.response}.png"
fig.savefig(out_path)
plt.close(fig)
print(f"Saved plot to {out_path}")

for run_label, job_name, value, sigma in zip(run_labels, job_names, values, sigmas):
    print(f"  {run_label} ({job_name}): {value:.6f} +/- {sigma:.6f}")
