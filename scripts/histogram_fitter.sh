#!/bin/bash

cd ~/xtc_talk_analysis
histogram_directory="histograms_not_fitted"
metadata_file_target=temp_results/fit_results/
cp temp_results/histograms/$histogram_directory/xtalk_metadata.json $metadata_file_target

for i in {0..100}; do
    echo "Running histogram_fitter.py $i ......"
    uv run scripts/histogram_fitter.py $i --histo_dir temp_results/histograms/$histogram_directory
    echo "Finish fitting detector $i."
done

uv run scripts/fit_inspector.py

cd temp_results/histograms/
timestring=$(date +"%Y%m%d_%H%M%S")
mkdir "histograms_$timestring"
mv "$histogram_directory"/* "histograms_$timestring"/
rm -r "$histogram_directory"/