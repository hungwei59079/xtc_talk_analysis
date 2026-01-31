from pathlib import Path
import numpy as np
from xtc_utils import files_and_chnid, get_baseline_energy

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = REPO_ROOT / "xtc_config.json"
data_dict = {"p08": ["r014","r015"]}
OUTDIR = REPO_ROOT / "temp_results" / "parameters"

OUTDIR.mkdir(parents=True, exist_ok=True)

new_hit_list, new_dsp_list, chn_id = files_and_chnid(CONFIG_PATH, "xtc_old", data_dict)

positive_baseline, negative_baseline, skipped_channels = (
    get_baseline_energy(new_hit_list, new_dsp_list, chn_id)
)

np.save(OUTDIR / "positive_baseline.npy", positive_baseline)
np.save(OUTDIR / "negative_baseline.npy", negative_baseline)

if skipped_channels:
    np.save(OUTDIR / "skipped_channels.npy", np.array(skipped_channels))