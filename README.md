## LEGEND-200 Cross Talk Analysis package

### Usage:
Note that all the commands should be done at project root directory `xtc_talk_analysis/`.
#### Step 0 - Environment setup:
Run the followings:
```
git clone https://github.com/hungwei59079/xtc_talk_analysis.git
cd xtc_talk_analysis
git checkout v1.0.0
uv sync
```
To pull the repository and download the necessary packages.

#### Step 1 - Prepare baseline in batch:
Go to `batch/prepare_baseline_batch.sh` and edit the following lines:
```
CONFIG_PATH="configs/xtc_config.json"
CONFIG_NAME="xtc_old"
```
to whatever you would like to use. 
Currently two configuration `xtc_old` and `xtc_p16` are provided in `xtc_config.json`. 
One could add new configurations by oneself as long as the keys are the same. 

After modifying the parameters, one could run
```
sbatch batch/prepare_baseline_batch.sh
```
to submit the baseline computation jobs to SLURM. 
Note that by default, the temporary results are generated in `${SCRATCH}/temp_results` as configured by the `--temp_result_target` flag in the script.

#### Step 2 - Merge Baseline:
Run:
```
./scripts/merge_baseline.sh $SCRATCH/temp_results
```
to merge independent baseline results into one single set of `.npy` files. 
You need to specify the path to your `temp_results` directory as an argument.
The directory storing the individual results will not be removed but instead be renamed with a time string appended to it for reusability.

#### Step 3 - Xtalk Element Computation and Binning:
Before running, be sure to open `batch/run_xtalk_array_chunk.sh` and ensure the `--temp_result_dir` argument passed to `xtalk_batch.py` points to your `temp_results` directory (by default `"$SCRATCH/temp_results"`).
Then, run:
```
sbatch batch/run_xtalk_array_chunk.sh
```
to compute the xtalk elements. They will be filled into the histograms. Note that in order to remove extreme values, only the data within 3 standard deviation will be filled in.

#### Step 4 - Fitting and Plotting the Xtalk Matrices:
Run:
```
./scripts/histogram_fitter.sh $SCRATCH/temp_results
```
You need to specify the path to your `temp_results` directory as an argument.
Gaussian fit will be done on previous histograms and xtalk matrices will be produced after the fitting is complete.
The results will be stored in `results/`.
