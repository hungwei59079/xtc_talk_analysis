import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from xtc_utils import files_and_chnid, XTCConfig

reason_dict = {"no_stats" : "no_stats",
               "low_stats" : "low_stats",
               "Optimal parameters not found: Number of calls to function has reached maxfev = 800.": "delta" ,
               "ok but with insufficient points" : "sharp"}

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_PATH = REPO_ROOT / "temp_results"/ "parameters"
IN_DIR = REPO_ROOT / "temp_results"/ "fit_results"
metadata_file = IN_DIR / "xtalk_metadata.json"
with open(metadata_file, "r") as f:
    xtalk_metadata = json.load(f)
parameters = xtalk_metadata["parameters"]
number_of_detectors = xtalk_metadata["number_of_detectors"]
config_path_str = parameters["config_path"]
config_name = parameters["config_name"]
data_filter = parameters["data_filter"]

CONFIG_PATH = Path(config_path_str)
config = XTCConfig(CONFIG_PATH, config_name)

xtalk_matrices = {"neg_xtalk_matrix": np.full((number_of_detectors, number_of_detectors),np.nan),
                "pos_xtalk_matrix": np.full((number_of_detectors, number_of_detectors),np.nan),
                "neg_restrained_xtalk_matrix": np.full((number_of_detectors, number_of_detectors),np.nan),
                "pos_restrained_xtalk_matrix": np.full((number_of_detectors, number_of_detectors),np.nan),
               }
fail_dict = {"no_stats": [], "low_stats": [], "delta": [], "sharp": []}
scenarios = set()

new_hit_list, new_dsp_list, chn_id = files_and_chnid(config, data_dict=data_filter)
skipped_channels = set(np.load(PARAMS_PATH / "skipped_channels.npy"))

for j1 in range(number_of_detectors):
    raw_id_1 = chn_id[j1]
    for j2 in range(number_of_detectors):
        raw_id_2 = chn_id[j2]
        for label in ["neg", "pos", "neg_restrained", "pos_restrained"]:
            npz_path = IN_DIR / f"fit_{label}_{j1}_{j2}.npz"
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
            xtalk_matrices[f"{label}_xtalk_matrix"][j1,j2] = mu
            if reason == reason_dict:
                abb_reason = reason_dict[reason]
                if raw_id_1 not in skipped_channels and raw_id_2 not in skipped_channels:
                    if raw_id_1 != raw_id_2:
                        fail_dict[abb_reason].append(f"{label}_{j1}_{j2}")

print(scenarios)

plot_settings = {
    "neg_xtalk_matrix": {
        "vmin": -0.3,
        "vmax": 0.1,
        "title": "Negative Crosstalk Matrix Heatmap",
        "cbar_label": "Negative Xtalk Value (%)",
        "filename": "Neg_xtk_map_fitted.png",
    },
    "pos_xtalk_matrix": {
        "vmin": -0.07,
        "vmax": 0.3,
        "title": "Positive Crosstalk Matrix Heatmap",
        "cbar_label": "Positive Xtalk Value (%)",
        "filename": "Pos_xtk_map_fitted.png",
    },
    "neg_restrained_xtalk_matrix": {
        "vmin": -0.3,
        "vmax": 0.1,
        "title": "Negative Restrained Crosstalk Matrix Heatmap",
        "cbar_label": "Negative Restrained Xtalk Value (%)",
        "filename": "Neg_restrained_xtk_map_fitted.png",
    },
    "pos_restrained_xtalk_matrix": {
        "vmin": -0.07,
        "vmax": 0.3,
        "title": "Positive Restrained Crosstalk Matrix Heatmap",
        "cbar_label": "Positive Restrained Xtalk Value (%)",
        "filename": "Pos_restrained_xtk_map_fitted.png",
    },
}

cmap = plt.cm.jet_r

for matrix_key, cfg in plot_settings.items():
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        xtalk_matrices[matrix_key],
        origin='lower',
        cmap=cmap,
        vmin=cfg["vmin"],
        vmax=cfg["vmax"],
    )

    cbar = plt.colorbar(im)
    cbar.set_label(cfg["cbar_label"])

    plt.xlabel('Response Channel Index')
    plt.ylabel('Trigger Channel Index')
    plt.title(cfg["title"])
    plt.tight_layout()
    plt.savefig(OUT_DIR / cfg["filename"])
    plt.close()