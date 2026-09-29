from legenddataflow.methods.paths import tmp_par_path
from pathlib import Path

def get_pattern_xtc_tmp(config, filename):
    return Path(f"{tmp_par_path(config)}") / "xtc" / filename


def get_pattern_xtc_tmp_channel(config, filename):
    return Path(f"{tmp_par_path(config)}") / "xtc" / filename


rule xtc_baseline:
    input:
        hit=...,
        dsp=...,
    output:
        baseline=get_pattern_xtc_tmp(
            config,
            "{experiment}-{period}-{run}-cal-{timestamp}-xtc-baseline.npz",
        )


rule xtc_histograms:
    input:
        baseline=get_pattern_xtc_tmp(
            config,
            "{experiment}-{period}-{run}-cal-{timestamp}-xtc-baseline.npz",
        ),
        hit=...,
        dsp=...,
    output:
        histograms=temp(
            get_pattern_xtc_tmp_channel(
                config,
                "{experiment}-{period}-{run}-cal-{timestamp}-{channel}-xtc-histograms.npz",
            )
        )


rule xtc_fit:
    input:
        histograms=get_pattern_xtc_tmp_channel(
            config,
            "{experiment}-{period}-{run}-cal-{timestamp}-{channel}-xtc-histograms.npz",
        ),
    output:
        fit=temp(
            get_pattern_xtc_tmp_channel(
                config,
                "{experiment}-{period}-{run}-cal-{timestamp}-{channel}-xtc-fit.npz",
            )
        )


rule xtc_combine:
    input:
        fits=expand(
            get_pattern_xtc_tmp_channel(
                config,
                "{experiment}-{period}-{run}-cal-{timestamp}-{channel}-xtc-fit.npz",
            ),
            channel=CHANNELS,
        ),
    output:
        matrix=get_pattern_xtc_tmp(
            
            "{experiment}-{period}-{run}-cal-{timestamp}-xtc-matrix.npz",
        ),