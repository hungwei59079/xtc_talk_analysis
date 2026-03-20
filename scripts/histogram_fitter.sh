#!/bin/bash

cd ~/xtc_talk_analysis
metadata_file=temp_results/fit_results/
if [ ! -f $metadata_file ]; then
    echo "cloning metadata to fit_results directory......"
    cp temp_results/histograms/xtalk_metadata.json $metadata_file
fi
source .venv/bin/activate

for i in {0..100}; do
    echo "Running histogram_fitter.py $i ......"
    python scripts/histogram_fitter.py $i
    echo "Finish fitting detector $i."
done