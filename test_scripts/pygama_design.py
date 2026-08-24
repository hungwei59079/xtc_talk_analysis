def prepare_baseline(hit_files: str | list, dsp_files: str | list, 
                     chn_id: str | list, temp_dir: str | Path) -> None: 
    raise NotImplementedError()

# merge baseline? 

def xtalk_column(hit_files: str | list, dsp_files: str | list, 
                 trigger_detector_id: str, chn_id_list: list, baseline: dict) -> None:
    raise NotImplementedError()

# Or maybe baseline should be two separate arguments? 
# Or maybe baseline is a file path?

def xtalk_histogram_fitter(trigger_detector_id: str, histogram_file: str | Path, 
                           config: dict | None) -> None:
    raise NotImplementedError()

def build_xtalk_matrix(chn_id_list: list, histogram_files: list, fitted_files: list, 
                       out_path: str | Path, config: dict | None) -> None:
    raise NotImplementedError()