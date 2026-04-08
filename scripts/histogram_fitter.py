import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "channel_number",
    type=str,
    help="channel to fit"
)
parser.add_argument(
    "--histo_dir",
    type=str,
    help="directory storing histograms",
)
args = parser.parse_args()

j1 = int(args.channel_number)

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = Path(args.histo_dir)
OUT_DIR = REPO_ROOT / "temp_results" / "fit_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

xtalk_metadata_file = IN_DIR / "xtalk_metadata.json"
with open(xtalk_metadata_file, "r") as f:
    xtalk_metadata = json.load(f)
    number_of_detectors = xtalk_metadata["number_of_detectors"]
    parameters = xtalk_metadata["parameters"]
    CONFIG_PATH = Path(parameters["config_path"])

with open(CONFIG_PATH, "r") as f:
    full_cfg = json.load(f)
    fit_params = full_cfg.get("fit_parameters", {})
    low_stats_threshold = fit_params.get("low_stats_threshold", 100)
    y_mask_threshold = fit_params.get("y_mask_threshold", 0.05)
    sharp_fit_min_points = fit_params.get("sharp_fit_min_points", 5)


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

for j2 in range(number_of_detectors):
    npz_path = IN_DIR / f"xtalk_{j1}_{j2}.npz"

    with np.load(npz_path) as data:
        neg_counts = data["neg_counts"]
        neg_bins = data["neg_bins"]
        pos_counts = data["pos_counts"]
        pos_bins = data["pos_bins"]
        neg_counts_restrained = data["neg_counts_restrained"]
        neg_bins_restrained = data["neg_bins_restrained"]
        pos_counts_restrained = data["pos_counts_restrained"]
        pos_bins_restrained = data["pos_bins_restrained"]

    for label, counts, bins in [("neg", neg_counts, neg_bins), ("pos", pos_counts, pos_bins), ("neg_restrained", neg_counts_restrained, neg_bins_restrained), ("pos_restrained", pos_counts_restrained, pos_bins_restrained)]:
        total_events = np.sum(counts)
        # Case 1: No events at all, skipping.
        if total_events == 0:
            result = dict(success=False, reason="no_stats", A=np.nan, mu=np.nan, sigma=np.nan, total_events=total_events)
            np.savez(OUT_DIR / f"fit_{label}_{j1}_{j2}.npz", **result)
            print(f"{label}_{j1}_{j2} has no stats")
            continue
            
        x = 0.5 * (bins[1:] + bins[:-1])
        y = counts
        # Case 2: less than 100 events, taking mean directly.
        if total_events < low_stats_threshold:
            mu = np.sum(x * y) / total_events
            sigma = np.sqrt(np.sum(y * (x - mu)**2) / total_events)
            result = dict(success=False, reason="low_stats", A=np.nan, mu=mu, sigma=sigma, total_events=total_events)
            np.savez(OUT_DIR / f"fit_{label}_{j1}_{j2}.npz", **result)
            print(f"{label}_{j1}_{j2} has low stats, tracing back to taking mean.")
            continue          
                        
        # Smooth counts to avoid spikes due to noise
        smooth_y = uniform_filter1d(y, size=3)
        peak_idx = np.argmax(smooth_y)
        peak_x = x[peak_idx]

        mask = y > y_mask_threshold * np.max(y)
        x_fit = x[mask]
        y_fit = y[mask]

        #Case 3: Sharp distribution
        if len(x_fit) < sharp_fit_min_points:
            print(f"{label}_{j1}_{j2} has insufficient_fit_points after masking. Trace back to non_masked x,y")
            A0 = np.max(y)
            mu0 = np.average(x, weights=y)
            sigma0 = np.sqrt(np.average((x - mu0)**2, weights=y))
            try:
                popt, pcov = curve_fit(gaussian, x, y, p0=[A0, mu0, sigma0])
                A, mu, sigma = popt
                success = True
                reason = "ok but with insufficient points"
            except Exception as e:
                A, mu, sigma = np.nan, np.nan, np.nan
                success = False
                reason = str(e)
                print("Failed with:")
                print(e)

        else:
            A0 = np.max(y_fit)
            mu0 = np.average(x_fit, weights=y_fit)
            sigma0 = np.sqrt(np.average((x_fit - mu0)**2, weights=y_fit))

            try:
                popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=[A0, mu0, sigma0])
                A, mu, sigma = popt
                success = True
                reason = "ok"
            except Exception as e:
                A, mu, sigma = np.nan, np.nan, np.nan
                success = False
                reason = str(e)
                print("Failed with:")
                print(e)

        # Save results
        result = dict(success=success, reason=reason, A=A, mu=mu, sigma=sigma, total_events=total_events)
        np.savez(OUT_DIR / f"fit_{label}_{j1}_{j2}.npz", **result)