## LEGEND-200 Cross Talk Analysis package

### Usage:
Note that all the commands should be done at project root directory `xtc_talk_analysis/`.
#### Step 0 - Environment setup:
Run the followings:
```
git clone https://github.com/hungwei59079/xtc_talk_analysis.git
cd xtc_talk_analysis
git checkout v1.1.0
uv sync
```
To pull the repository and download the necessary packages.

#### Step 1 - Prepare baseline in batch:
Review `configs/jobs_list.json` and ensure jobs and configurations are set to whatever you would like to use. 
Currently configurations like `xtc_old` and `xtc_p16` are provided in `xtc_config.json`. 
One could add new configurations by oneself as long as the keys are the same. 

After modifying the parameters, one could run
```
uv run scripts/submit_jobs.py --step 1
```
to submit the baseline computation jobs to SLURM. 
The temporary results are generated in the `temp_result_loc` mapped out in the job list JSON.

#### Step 2 - Merge Baseline:
Run:
```
uv run scripts/submit_jobs.py --step 2
```
to merge independent baseline results into one single set of `.npy` files across all target tasks. 
The directory storing the individual results will not be removed but instead be renamed with a time string appended to it for reusability.

#### Step 3 - Xtalk Element Computation and Binning:
Run:
```
uv run scripts/submit_jobs.py --step 3
```
to submit batch calculations for the xtalk elements. They will be filled into the histograms. Note that in order to remove extreme values, only the data within 3 standard deviation will be filled in.

#### Step 4 - Fitting and Plotting the Xtalk Matrices:
Run:
```
uv run scripts/submit_jobs.py --step 4
```
Gaussian fit will be done on previous histograms and xtalk matrices will be produced after the fitting is complete.
The results will be stored in `results/`.
