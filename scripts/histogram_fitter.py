import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from pathlib import Path

j1 = int(sys.argv[1])

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "temp_results" / "histograms"
OUT_DIR = REPO_ROOT / "temp_results" / "fit_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# out_dir_figs = os.path.join(base_dir, "../fit_figures")

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

for j2 in range(101):
    npz_path = IN_DIR / f"xtalk_{j1}_{j2}.npz"

    with np.load(npz_path) as data:
        neg_counts = data["neg_counts"]
        neg_bins = data["neg_bins"]
        pos_counts = data["pos_counts"]
        pos_bins = data["pos_bins"]

    #neg_x = (neg_bins[1:]+neg_bins[:-1])/2
    #neg_y = neg_counts
    #pos_x = (pos_bins[1:]+pos_bins[:-1])/2
    #pos_y = pos_counts

    for label, counts, bins in [("neg", neg_counts, neg_bins), ("pos", pos_counts, pos_bins)]:
        total_events = np.sum(counts)
        if total_events == 0:
            # No event at all, skipping.
            result = dict(success=False, reason="no_stats", A=np.nan, mu=np.nan, sigma=np.nan, total_events=total_events)
            np.savez(OUT_DIR / f"fit_{label}_{j1}_{j2}.npz", **result)
            print(f"{label}_{j1}_{j2} has no stats")
            continue
            
        x = 0.5 * (bins[1:] + bins[:-1])
        y = counts

        if total_events < 100:
            # Too few stats to fit with Gaussian, trace back to taking mean.
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

        # Restrict fit region: within ±3 std estimates around peak
        # Use only bins where y is > 5% of the max to ignore far tails
        mask = y > 0.05 * np.max(y)
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 5:
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