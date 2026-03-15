#!/bin/bash

uv run scripts/merge_baseline.py
timestring=$(date +"%Y%m%d_%H%M%S")
cd temp_results/parameters
mkdir "baseline_individuals_$timestring"
mv baseline_individuals/* "baseline_individuals_$timestring"/
rm -r baseline_individuals/
