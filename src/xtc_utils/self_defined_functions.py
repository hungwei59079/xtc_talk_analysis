import os
import numpy as np
from lgdo import lh5
from dbetto import TextDB, Props
from pathlib import Path
import matplotlib.pyplot as plt

from .config import XTCConfig


def files_and_chnid(config: XTCConfig, data_dict: dict = None, return_names: bool = False):
    """Get hit/dsp file lists and channel IDs from configuration.

    Parameters
    ----------
    config : XTCConfig
        Configuration object for the dataset.
    data_dict : dict, optional
        Dictionary specifying which periods/runs to use.
        Format: {"p08": ["r015", "r016"], "p09": ["r001"]}.
        If not provided, all available periods/runs will be used.
    return_names : bool, optional
        If True, additionally return the list of germanium detector names.
        Default False, which preserves the original 3-value return.

    Returns
    -------
    new_hit_list : list
        List of hit file paths.
    new_dsp_list : list
        List of DSP file paths.
    chn_id : list
        List of channel IDs (rawid) for germanium detectors.
    det_names : list, optional
        Only returned when ``return_names`` is True. Germanium detector
        names in the same order as ``chn_id``; i.e. ``det_names[j]`` and
        ``chn_id[j]`` refer to the same detector, which is also the
        detector at row/column index ``j`` of the crosstalk matrix.
    """
    xtc_dir = config.xtc_dir
    dsp_dir_template = config.dsp_dir_template
    hit_dir_template = config.hit_dir_template
    full_data_dict = config.available_periods

    # check if all periods and runs in data_dict are in full_data_dict
    if data_dict is not None:
        for period in data_dict.keys():
            if period not in full_data_dict.keys():
                print(f"Warning: Period {period} not found in configuration. It will be skipped.")
            for run in data_dict[period]:
                if run not in full_data_dict[period]:
                    print(f"Warning: Run {run} not found in configuration for period {period}. It will be skipped.")

    lmeta = TextDB(path=f"{xtc_dir}/inputs")
    new_hit_list = []
    new_dsp_list = []

    for period in full_data_dict.keys():
        for run in full_data_dict[period]:
            if data_dict is not None:
                if period not in data_dict.keys():
                    continue
                if run not in data_dict[period]:
                    continue
            print(f"searching for files for {period}, {run}......")
            dsp_dir = dsp_dir_template.format(xtc_dir=xtc_dir, period=period, run=run)
            hit_dir = hit_dir_template.format(xtc_dir=xtc_dir, period=period, run=run)

            try:
                valid_file = f"{xtc_dir}/generated/par/valid_keys/l200-{period}-{run}-valid_xtc.json"
                valid_keys = list(Props.read_from(valid_file)["valid_keys"])
                time_string = valid_keys[0].split("-")[-1]
            
                dsp_list = [f"{dsp_dir}/{key}-tier_dsp.lh5" for key in valid_keys] 
                hit_list = [f"{hit_dir}/{key}-tier_hit.lh5" for key in valid_keys]
            
                # Remove non-existent files from the lists
            
                listed_files = set(os.path.basename(f) for f in hit_list)
                actual_files = set(os.listdir(hit_dir))
                non_existent_files = listed_files - actual_files
                for f in hit_list:
                    if os.path.basename(f) not in non_existent_files:
                        new_hit_list.append(f)
            
                listed_files = set(os.path.basename(f) for f in dsp_list)
                actual_files = set(os.listdir(dsp_dir))
                non_existent_files = listed_files - actual_files
                for f in dsp_list:
                    if os.path.basename(f) not in non_existent_files:
                        new_dsp_list.append(f)
            except:
                print("Could not find valid_xtc.json; using all files in hit and dsp directories.")
                new_hit_list += [f"{hit_dir}/{f}" for f in os.listdir(hit_dir) if f.endswith(".lh5")]
                new_dsp_list += [f"{dsp_dir}/{f}" for f in os.listdir(dsp_dir) if f.endswith(".lh5")]

    new_dsp_list = sorted(new_dsp_list)
    new_hit_list = sorted(new_hit_list)

    # Print summary of collected hit files
    print(f"\n{'='*50}")
    print(f"Summary of collected hit files:")
    print(f"  Total number of files: {len(new_hit_list)}")
    
    # Extract periods and runs from file paths
    periods_runs = {}
    for f in new_hit_list:
        # Parse the path to extract period and run (e.g., .../p10/r005/...)
        parts = f.split('/')
        for i, part in enumerate(parts):
            if part.startswith('p') and part[1:].isdigit():
                period = part
                if i + 1 < len(parts) and parts[i + 1].startswith('r'):
                    run = parts[i + 1]
                    if period not in periods_runs:
                        periods_runs[period] = {}
                    if run not in periods_runs[period]:
                        periods_runs[period][run] = 0
                    periods_runs[period][run] += 1
                break
    
    print(f"  Periods and runs included:")
    for period in sorted(periods_runs.keys()):
        runs_info = ", ".join([f"{run} ({count} files)" for run, count in sorted(periods_runs[period].items())])
        print(f"    {period}: {runs_info}")
    print(f"{'='*50}\n")

    #Check if timestrings match
    hit_timestrings = [f.split("-")[-2] for f in new_hit_list]
    dsp_timestrings = [f.split("-")[-2] for f in new_dsp_list]
    # print(hit_timestrings)
    if set(hit_timestrings) != set(dsp_timestrings):
        raise ValueError("Hit and DSP files have mismatched time strings.")
    time_string = hit_timestrings[0]

    #Check if number of events match
    """
    hit_table = lh5.read('/ch1078400/hit/', new_hit_list)
    dsp_table = lh5.read('/ch1078400/dsp/', new_dsp_list)
    print(f"hit_table_length: {len(hit_table)}; dsp_table_length:{len(dsp_table)}")
    if len(hit_table) != len(dsp_table):
        raise ValueError("Hit and dsp tables have unequal number of events.")
    """
    
    #Now we can obtain the raw ids using the .on() utility.
    
    chmap = lmeta.hardware.configuration.channelmaps.on(time_string)
    geds = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds']
    chn_id = []
    for detector in geds:
        chn_id.append(chmap[detector]['daq']['rawid'])

    if return_names:
        return new_hit_list, new_dsp_list, chn_id, geds
    return new_hit_list, new_dsp_list, chn_id

def xtalk_element(E_trig, E_response, baseline_value):
    # Check if baseline_value is numerical
    if not isinstance(baseline_value, (int, float)):
        raise TypeError("baseline_value must be a numerical type (int or float).")
    
    # Case 1: both inputs are lists
    if isinstance(E_trig, np.ndarray) and isinstance(E_response, np.ndarray):
        if len(E_trig) != len(E_response):
            raise ValueError("E_trig and E_response must have the same length.")
        baseline = np.full(len(E_trig), baseline_value)
        return (E_response - baseline) / E_trig * 100
        # return E_response / E_trig * 100
    
    # Case 2: all inputs are scalars (numerical values)
    elif isinstance(E_trig, (int, float)) and isinstance(E_response, (int, float)):
        return (E_response - baseline_value) / E_trig * 100
    
    # Case 3: unsupported input types
    else:
        raise TypeError("E_trig and E_response must either both be lists of equal length or both be numerical values.")

# To be deprecated: use the EventSelector class instead for better modularity and reusability.
def relevant_events(
    table_path,
    files,
    ene_dataset,
    conditions=None,
    energy_range=None,
    idx = None):
    """
    Select events from a LH5 file based on multiple flag conditions and an optional energy range.

    Parameters:
    - table_path: str
        The table where the energy and flag datasets are stored.
    - files: str or list
        The LH5 file(s) to read.
    - ene_dataset: str
        The name of the dataset containing the energy values.
    - conditions: dict, optional
        Dictionary mapping each flag dataset name to its condition.
        If a flag dataset is listed but no condition is given, defaults to True.
    - energy_range: tuple (emin, emax), optional
        If provided, only energies within this inclusive range will be kept.

    Returns: A dictionary that contains:
    - energy_presel: energy without BOTH INDEXING and flag selection.
    - energy_sel: np.ndarray
        1D array of energy values satisfying all given conditions.
    - indices: np.ndarray
        array of indices of the events being selected.
    """
    if conditions is None:
        conditions = {}
    flag_datasets = list(conditions.keys())

    all_fields = [ene_dataset] + flag_datasets
    table = lh5.read(table_path, files, field_mask=all_fields)
    energy_all = table[ene_dataset].nda

    if idx is not None:
        energy_all_indexed = energy_all[idx]
        selection_array = ~np.isnan(energy_all_indexed)
    else:
        selection_array = ~np.isnan(energy_all)

    for flag in flag_datasets:
        if idx is not None:
            flag_array = table[flag].nda[idx]
        else:
            flag_array = table[flag].nda
        condition = conditions.get(flag, True)
        selection_array &= (flag_array == condition)

    if energy_range is not None:
        emin, emax = energy_range
        selection_array &= (energy_all >= emin) & (energy_all <= emax)

    if idx is not None:
        selected_energies = energy_all_indexed[selection_array]
        selected_idxs = idx[selection_array]
    else:
        selected_energies = energy_all[selection_array]
        selected_idxs = np.arange(len(energy_all))
        selected_idxs = selected_idxs[selection_array]

    results = {
        "energy_presel": energy_all,
        "energy_sel": selected_energies,
        "indices": selected_idxs,
    }
    
    return results


