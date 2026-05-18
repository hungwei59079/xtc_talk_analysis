import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from xtc_utils import files_and_chnid, XTCConfig, XTCMatrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--temp_result_dir",
    type=str,
    default=None,
)
parser.add_argument(
    "--job_id",
    type=str,
    default=None,
    help="Subdirectory under results/ to save outputs into; avoids overwriting between jobs.",
)
args = parser.parse_args()


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results"
if args.job_id:
    OUT_DIR = OUT_DIR / args.job_id
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_PATH = Path(args.temp_result_dir) / "parameters"
IN_DIR = Path(args.temp_result_dir) / "fit_results"
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

new_hit_list, new_dsp_list, chn_id, det_names = files_and_chnid(
    config, data_dict=data_filter, return_names=True
)
skipped_channels = set(np.load(PARAMS_PATH / "skipped_channels.npy"))

map_path = OUT_DIR / "detector_map.csv"
with open(map_path, "w") as f:
    f.write("index,detector_name,channel_id\n")
    for idx, (name, rawid) in enumerate(zip(det_names, chn_id)):
        f.write(f"{idx},{name},{rawid}\n")
print(f"Saved detector index mapping to {map_path}")

for label in ["neg", "pos", "neg_restrained", "pos_restrained"]:
    xtalk_matrix = XTCMatrix(number_of_detectors, label)
    xtalk_matrix.load(chn_id, skipped_channels, path=IN_DIR)
    xtalk_matrix.save_csv(path=OUT_DIR)
    xtalk_matrix.plot(path=OUT_DIR)
    xtalk_matrix.diagnose()