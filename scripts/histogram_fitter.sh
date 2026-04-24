#!/bin/bash

output_root=$1
histogram_directory="histograms_not_fitted"
metadata_file_target="fit_results"
mkdir -p "$output_root/$metadata_file_target"
cp "$output_root/histograms/$histogram_directory/xtalk_metadata.json" "$output_root/$metadata_file_target"

for i in {0..100}; do
    echo "Running histogram_fitter.py $i ......"
    uv run scripts/histogram_fitter.py $i --temp_result_dir "$output_root/"
    echo "Finish fitting detector $i."
done

uv run scripts/fit_inspector.py --temp_result_dir "$output_root/"

cd "$output_root/histograms/"
timestring=$(date +"%Y%m%d_%H%M%S")
mkdir "histograms_$timestring"
mv "$histogram_directory"/* "histograms_$timestring"/
rm -r "$histogram_directory"/