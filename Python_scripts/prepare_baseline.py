from self_defined_functions import files_and_chnid, get_baseline_energy
import numpy as np
import os

new_hit_list, new_dsp_list, chn_id = files_and_chnid()

positive_baseline, negative_baseline, skipped_channels = get_baseline_energy(new_hit_list, new_dsp_list, chn_id)

os.chdir("../parameters")
np.save("positive_baseline.npy", positive_baseline)
np.save("negative_baseline.npy", negative_baseline)

# Optionally save the skipped channels
if skipped_channels:
    np.save("skipped_channels.npy", np.array(skipped_channels))