import argparse
import json
import numpy as np
from lgdo import lh5
from xtc_utils import files_and_chnid, EventSelector, xtalk_element, XTCConfig
from pathlib import Path


def find_repo_root(start):
    """Walk up from `start` until a directory containing pyproject.toml is found."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume this file lives one level below the repo root.
    return start.parents[1]


# ---------- Setup ----------
parser = argparse.ArgumentParser(
    description="Split the xtalk events of a single (trigger, response) pair "
                "into below/above a cut for double-peak inspection."
)
parser.add_argument("j1", type=int, help="Index of the trigger channel (0-based).")
parser.add_argument("j2", type=int, help="Index of the response channel (0-based).")
parser.add_argument("--cut", type=float, required=True,
                    help="Cut value on the xtalk array. Events with value < cut "
                         "go to the 'below' category, value >= cut to 'above'.")
parser.add_argument("--cut_on", choices=["neg", "pos"], default="neg",
                    help="Which xtalk array the cut is applied to (default: neg, "
                         "since the double peak appears in the neg_restrained histograms).")
parser.add_argument("--temp_result_dir",
                    help="Path to the temp_results directory containing parameters and metadata.",
                    default=find_repo_root(Path(__file__).resolve()) / "temp_results"
                    )
parser.add_argument("--outdir", type=str, default=None,
                    help="Output directory for the split indices and values. ")
args = parser.parse_args()

j1 = int(args.j1)
j2 = int(args.j2)
cut_value = float(args.cut)

REPO_ROOT = find_repo_root(Path(__file__).resolve())
if args.outdir is None:
    OUTDIR = REPO_ROOT / "results" / "Inspected_histograms"
else:
    OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

PARAMS_PATH = Path(args.temp_result_dir) / "parameters"
baseline_metadata_file = PARAMS_PATH / "baseline_metadata.json"
with open(baseline_metadata_file, "r") as f:
    baseline_metadata = json.load(f)
parameters = baseline_metadata["parameters"]
n_detectors = baseline_metadata["total_detectors"]
config_path_str = parameters["config_path"]
config_name = parameters["config_name"]
data_filter = parameters["data_filter"]

CONFIG_PATH = Path(config_path_str)
config = XTCConfig(CONFIG_PATH, config_name)

new_hit_list, new_dsp_list, chn_id = files_and_chnid(config, data_dict=data_filter)

if not (0 <= j1 < n_detectors) or not (0 <= j2 < n_detectors):
    raise SystemExit(f"j1={j1}, j2={j2} out of range for {n_detectors} detectors.")

raw_id_1 = chn_id[j1]
raw_id_2 = chn_id[j2]
print(f"File listing complete. Pair (j1={j1}, j2={j2}) -> (raw {raw_id_1}, raw {raw_id_2}).")

# Load parameters
positive_baseline = np.load(PARAMS_PATH / "positive_baseline.npy")
negative_baseline = np.load(PARAMS_PATH / "negative_baseline.npy")
skipped_channels = set(np.load(PARAMS_PATH / "skipped_channels.npy"))

if raw_id_1 == raw_id_2:
    raise SystemExit(f"Self-interaction (j1==j2=={j1}); nothing to split.")
if raw_id_1 in skipped_channels:
    raise SystemExit(f"Trigger channel j1={j1} (raw {raw_id_1}) is in skipped_channels.")
if raw_id_2 in skipped_channels:
    raise SystemExit(f"Response channel j2={j2} (raw {raw_id_2}) is in skipped_channels.")

# ---------- Trigger channel event extraction ----------
cuspEmax_selection_1 = EventSelector(
    table_path=f"ch{raw_id_1}/hit/",
    files=new_hit_list,
    ene_dataset="cuspEmax_ctc_cal",
    conditions=config.xtalk_flag_trigger_conditions,
    energy_range=(1500, 99999),
)
print(f"trigger selection: {len(cuspEmax_selection_1.selected_idxs)} events")
trapTmax_1_presel = lh5.read(f"ch{raw_id_1}/dsp/trapTmax", new_dsp_list).nda
print("Trigger channel event extraction complete.")

# ---------- Response channel event extraction ----------
cuspEmax_selection_2 = EventSelector(
    table_path=f"ch{raw_id_2}/hit/",
    files=new_hit_list,
    ene_dataset="cuspEmax_ctc_cal",
    conditions=config.xtalk_flag_response_conditions,
    energy_range=(-99999, 100),
    idx=cuspEmax_selection_1.selected_idxs,
)
print(f"response selection: {len(cuspEmax_selection_2.selected_idxs)} events")

# Chained EventSelector => selected_idxs are indices into the original length-N
# event table, row-aligned across channels; valid to re-read any hit/dsp property.
final_idxs = cuspEmax_selection_2.selected_idxs

# Beware: .selected_idxs reuses indices from the original (concatenated) array!
trapTmax_1 = trapTmax_1_presel[final_idxs]

table_2 = lh5.read(f"ch{raw_id_2}/dsp/", new_dsp_list,
                   field_mask=["trapTmin", "trapTmax"], idx=final_idxs)
trapTmin_2 = table_2["trapTmin"].nda
trapTmax_2 = table_2["trapTmax"].nda

print(f"len of trapTmax_1: {len(trapTmax_1)}; len of trapTmax_2: {len(trapTmax_2)}")

neg_vals = np.asarray(xtalk_element(trapTmax_1, trapTmin_2, negative_baseline[j2]))
pos_vals = np.asarray(xtalk_element(trapTmax_1, trapTmax_2, positive_baseline[j2]))

if neg_vals.size == 0:
    raise SystemExit(f"No events for pair (j1={j1}, j2={j2}); nothing to split.")

# ---------- Split into below / above the cut ----------
# Masks index selected-event position; final_idxs maps them to original indices.
cut_array = neg_vals if args.cut_on == "neg" else pos_vals
below_mask = cut_array < cut_value
above_mask = ~below_mask

below_idx = final_idxs[below_mask]
above_idx = final_idxs[above_mask]

print(f"Cut on {args.cut_on}_vals at {cut_value}: "
      f"below={below_idx.size}, above={above_idx.size}, total={cut_array.size}")

out_path = OUTDIR / f"xtalk_split_{j1}_{j2}_{args.cut_on}.npz"
np.savez_compressed(
    out_path,
    below_idx=below_idx,
    above_idx=above_idx,
    below_neg_vals=neg_vals[below_mask],
    above_neg_vals=neg_vals[above_mask],
    below_pos_vals=pos_vals[below_mask],
    above_pos_vals=pos_vals[above_mask],
    cut=np.array(cut_value),
    cut_on=np.array(args.cut_on),
    j1=np.array(j1),
    j2=np.array(j2),
    raw_id_1=np.array(raw_id_1),
    raw_id_2=np.array(raw_id_2),
    n_selected=np.array(cut_array.size),
)
print(f"Saved below_idx ({below_idx.size}) and above_idx ({above_idx.size}) to {out_path}")
