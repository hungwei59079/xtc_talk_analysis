import sys
import json
import numpy as np
from lgdo import lh5
from xtc_utils import files_and_chnid, relevant_events, XTCConfig
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Setup ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
config_path_str = "/global/u2/h/hungwei/xtc_talk_analysis/configs/xtc_config.json"
config_name = "xtc_p16"

CONFIG_PATH = Path(config_path_str)
config = XTCConfig(CONFIG_PATH, config_name)

new_hit_list, new_dsp_list, chn_id = files_and_chnid(config)
print("File listing complete.")

j1 = int(sys.argv[1])
raw_id_1 = chn_id[j1]

try:
    energy_1, idxs = relevant_events(
        table_path=f"ch{raw_id_1}/hit/",
        files=new_hit_list,
        ene_dataset="trapEmax_ctc_cal",
        flag_datasets=config.xtalk_flag_trigger_datasets,
        conditions=config.xtalk_flag_trigger_conditions,
        energy_range=(1500, 4500),
        return_index=True
    )
    trapEmax_1 = lh5.read(f"ch{raw_id_1}/dsp/trapEmax", new_dsp_list, idx=idxs).nda
    trapEmax_map_1 = dict(zip(idxs, trapEmax_1)) # used later for secondary selection
    print("Trigger channel event extraction complete.")
    trig_extract_complete = True
except Exception as e:
    print(f"Exception occurred at trigger channel {j1} extraction.")
    trig_extract_complete = False

# mask = cuspEmax_1 < 1000
# selected_cuspEmax_1 = cuspEmax_1[mask]
histogram = np.histogram(trapEmax_1, bins=100)
plt.figure(figsize=(10,6))
plt.bar(histogram[1][:-1], histogram[0], width=np.diff(histogram[1]), align='edge')
plt.xlabel('trapEmax')
plt.ylabel('Counts')
plt.title(f'Channel {j1} (raw {raw_id_1}) trapEmax Distribution')
plt.grid()
plt.savefig(REPO_ROOT / "results" / f"trapEmax_histogram_j{j1}.png")
plt.close()
