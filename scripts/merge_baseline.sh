#!/bin/bash

cd ~/xtc_talk_analysis

$TEMP_RESULT_DIR = $1
uv run scripts/merge_baseline.py --input_dir "$TEMP_RESULT_DIR/parameters/baseline_individuals/json" --output_dir "$TEMP_RESULT_DIR/parameters"
timestring=$(date +"%Y%m%d_%H%M%S")
cd $TEMP_RESULT_DIR/parameters
mkdir "baseline_individuals_$timestring"
mv baseline_individuals/* "baseline_individuals_$timestring"/
rm -r baseline_individuals/
