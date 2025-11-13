## **A brief explanation on file hierarchy**

- `Back_up`, `single_logs`, `results_no_fit`: Some legacy stuff, just ignore them.
- `parameters`: Intermediate files obtained after running `prepare_baseline.py`.
- `results`: Interactive notebooks that could be used to inspect on final results.
- `Python_scripts`: All the codes that I actually used in my workflow.

## **Steps**:
1. Run
```
cd Python_scripts
python3 prepare_baseline.py
```

2. Outside whatever container you're using, run:
```
sbatch run_xtalk_array_chunk.sh
```
This will set up 101 jobs, each job actually runs:
```
python xtalk_batch.py $SLURM_ARRAY_TASK_ID
```
where `$SLURM_ARRAY_TASK_ID` is some integer from 0 to 100, in the legend-software container. This step creates ~10000 `.npz` files, each of them contains the bins and counts of the histogram that's filled with xtalk events that passes selection, for a certain trigger-response pair.
The `.npz` files will be stored in a git-ignored directory called `histograms`.

3. Outside container, run:
```
sbatch histogram_fit_batch.sh
```
which fits each histogram with Gaussian, and save the fitted parameters ($\mu$, $\sigma$, etc) in another bunch of .npz files.
The `.npz` files will be stored in a git-ignored directory called `fit_results`.

4. Finally, one can go to the `results` directory and run `fit_inspector.ipynb` for the heatmap, and `Histogram_inspector.ipynb` if you want to inspect on the histogram of a certain channel pair (Change the parameters j1, j2, and label if you want to do so.). Ignore attempt 1 in the latter notebook.  
