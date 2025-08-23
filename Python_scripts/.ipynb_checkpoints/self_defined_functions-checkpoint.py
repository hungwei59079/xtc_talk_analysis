import os
import numpy as np
from lgdo import lh5
from dbetto import TextDB, Props

def files_and_chnid():
    """
    Currently it's a copy and paste. Might include logic in future update.
    """
    period, run = "p08", "r015"
    xtc_dir = "/global/cfs/cdirs/m2676/data/lngs/l200/scratch/crosstalk_data/xtc"
    lmeta = TextDB(path=f"{xtc_dir}/inputs")

    valid_file = f"{xtc_dir}/generated/par/valid_keys/l200-{period}-{run}-valid_xtc.json"
    valid_keys = list(Props.read_from(valid_file)["valid_keys"])
    time_string = valid_keys[0].split("-")[-1]

    dsp_dir = f"{xtc_dir}/generated/tier/dsp/xtc/{period}/{run}"
    hit_dir = f"{xtc_dir}/generated/tier/hit/xtc/{period}/{run}"
    dsp_list = [f"{dsp_dir}/{key}-tier_dsp.lh5" for key in valid_keys]
    hit_list = [f"{hit_dir}/{key}-tier_hit.lh5" for key in valid_keys]

    #The next part is to remove missing files from the list obtained from valid_xtc.json. Should be removed when moved to LNGS.

    listed_files = set(os.path.basename(f) for f in hit_list)
    actual_files = set(os.listdir(hit_dir))
    non_existent_files = listed_files - actual_files
    new_hit_list = []
    for f in hit_list:
        if os.path.basename(f) not in non_existent_files:
            new_hit_list.append(f)

    listed_files = set(os.path.basename(f) for f in dsp_list)
    actual_files = set(os.listdir(dsp_dir))
    non_existent_files = listed_files - actual_files
    new_dsp_list = []
    for f in dsp_list:
        if os.path.basename(f) not in non_existent_files:
            new_dsp_list.append(f)

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

def get_baseline_energy(new_hit_list, chn_id):
    """
    Computes mean baseline energy for each detector (chn_id).
    Skips detectors that cause an error and logs them.

    Returns:
        baseline_energy: list of mean baseline energies (float)
        skipped: list of detector IDs that failed
    """
    baseline_energy = []
    skipped = []

    for j, detector in enumerate(chn_id):
        try:
            energies = relevant_events(
                table_path=f"ch{detector}/hit/",
                files=new_hit_list,
                ene_dataset="cuspEmax_ctc_cal",
                flag_datasets=["is_baseline"],
                conditions={"is_baseline": 63}
            )
            mean_energy = np.mean(energies)
            baseline_energy.append(mean_energy)
            print(f"✅ Baseline energy evaluated for detector #{j} (ID={detector}).")
        except Exception as e:
            print(f"❌ Skipping detector #{j} (ID={detector}): {e}")
            skipped.append(detector)
            baseline_energy.append(np.nan)

    print(f"\nSummary: {len(skipped)} detector(s) skipped.")
    if skipped:
        print("Skipped detector IDs:", skipped)

    return baseline_energy, skipped

def xtalk_element(E_trig, E_response, baseline_value):
    # Check if baseline_value is numerical
    if not isinstance(baseline_value, (int, float)):
        raise TypeError("baseline_value must be a numerical type (int or float).")
    
    # Case 1: both inputs are lists
    if isinstance(E_trig, np.ndarray) and isinstance(E_response, np.ndarray):
        if len(E_trig) != len(E_response):
            raise ValueError("E_trig and E_response must have the same length.")
        baseline = np.full(len(E_trig), baseline_value)
        #return (E_response - baseline) / E_trig * 100
        return E_response / E_trig * 100
    
    # Case 2: all inputs are scalars (numerical values)
    elif isinstance(E_trig, (int, float)) and isinstance(E_response, (int, float)):
        return (E_response - baseline_value) / E_trig * 100
    
    # Case 3: unsupported input types
    else:
        raise TypeError("E_trig and E_response must either both be lists of equal length or both be numerical values.")


