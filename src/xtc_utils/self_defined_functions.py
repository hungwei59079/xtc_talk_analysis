import os
import numpy as np
from lgdo import lh5
from dbetto import TextDB, Props
import json
from pathlib import Path

def files_and_chnid(config_path: Path, config_name: str, data_dict: dict = None):
    with config_path.open() as f:
        config = json.load(f)
    xtc_dir = config["datasets"][config_name]["xtc_dir"]
    dsp_dir_template = config["datasets"][config_name]["path_templates"]["dsp_dir"]
    hit_dir_template = config["datasets"][config_name]["path_templates"]["hit_dir"]
    full_data_dict = config["datasets"][config_name]["periods"]

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
    
    #Now we can obtain the raw ids using the .on() utility.
    
    chmap = lmeta.hardware.configuration.channelmaps.on(time_string)
    geds = [ch for ch in chmap.keys() if chmap[ch]['system']=='geds']
    chn_id = []
    for detector in geds:
        chn_id.append(chmap[detector]['daq']['rawid'])
        
    return new_hit_list, new_dsp_list, chn_id

def relevant_events(
    table_path,
    files,
    ene_dataset,
    flag_datasets=None,
    conditions=None,
    energy_range=None,
    return_index=False):
    """
    Select events from a LH5 file based on multiple flag conditions and an optional energy range.

    Parameters:
    - table_path: str
        The table where the energy and flag datasets are stored.
    - files: str or list
        The LH5 file(s) to read.
    - ene_dataset: str
        The name of the dataset containing the energy values.
    - flag_datasets: list of str, optional
        List of flag dataset names to apply conditions to.
    - conditions: dict, optional
        Dictionary mapping each flag dataset name to its condition.
        If a flag dataset is listed but no condition is given, defaults to True.
    - energy_range: tuple (emin, emax), optional
        If provided, only energies within this inclusive range will be kept.
    - return_index: If specified as True, then return the indices that meet the condition.

    Returns:
    - selected_energies: np.ndarray
        1D array of energy values satisfying all given conditions.
    - idxs: np.ndarray
        array of indices of the events being selected.
    """
    if flag_datasets is None:
        flag_datasets = []
    if conditions is None:
        conditions = {}

    all_fields = [ene_dataset] + flag_datasets
    table = lh5.read(table_path, files, field_mask=all_fields)
    energy_all = table[ene_dataset].nda

    selection_array = ~np.isnan(energy_all)

    for flag in flag_datasets:
        flag_array = table[flag].nda
        condition = conditions.get(flag, True)
        selection_array &= (flag_array == condition)

    if energy_range is not None:
        emin, emax = energy_range
        selection_array &= (energy_all >= emin) & (energy_all <= emax)

    selected_energies = energy_all[selection_array]
    if return_index == True:
        idxs = np.arange(len(energy_all))
        idxs = idxs[selection_array]
        return selected_energies, idxs
    return selected_energies

def get_baseline_energy(new_hit_list, new_dsp_list, chn_id):
    """
    Computes mean baseline energy for each detector (chn_id).
    Skips detectors that cause an error and logs them.

    Returns:
        baseline_energy: list of mean baseline energies (float)
        skipped: list of detector IDs that failed
    """
    positive_baseline = []
    negative_baseline = []
    skipped = []

    for j, detector in enumerate(chn_id):
        try:
            energies, idxs = relevant_events(
                table_path=f"ch{detector}/hit/",
                files=new_hit_list,
                ene_dataset="cuspEmax_ctc_cal",
                flag_datasets=["is_baseline"],
                conditions={"is_baseline": 63},
                return_index=True
            )
            table = lh5.read(f"ch{detector}/dsp/", new_dsp_list, field_mask=["trapTmin", "trapTmax"], idx=idxs)
            trapTmin = table["trapTmin"].nda
            trapTmax = table["trapTmax"].nda
            positive_baseline.append(np.mean(trapTmax))
            negative_baseline.append(np.mean(trapTmin))
            print(f"✅ Baseline energy evaluated for detector #{j} (ID={detector}).")
        except Exception as e:
            print(f"❌ Skipping detector #{j} (ID={detector}): {e}")
            skipped.append(detector)
            positive_baseline.append(np.nan)
            negative_baseline.append(np.nan)

    print(f"\nSummary: {len(skipped)} detector(s) skipped.")
    if skipped:
        print("Skipped detector IDs:", skipped)

    return positive_baseline, negative_baseline, skipped

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


