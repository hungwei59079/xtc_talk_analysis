import os
from lgdo import lh5

# Step 1: Set your directory path here
target_dir = "/global/cfs/cdirs/m2676/data/lngs/l200/scratch/crosstalk_data/xtc/generated/tier/hit/xtc/"
os.chdir(target_dir)

# Step 2: List all .lh5 files in the directory
lh5_files = [f for f in os.listdir() if f.endswith('.lh5')]

# Step 3: Dictionary to store each file's channel list
channel_lists = {}

# Step 4: Iterate over each file and get the channel list
for fname in lh5_files:
    try:
        hit_file = fname
        channel_list = lh5.ls(hit_file, "/")
        channel_lists[hit_file] = channel_list
    except Exception as e:
        print(f"Error processing {fname}: {e}")

# Step 5: Compare channel lists
all_channels = list(channel_lists.values())
all_same = all(channel == all_channels[0] for channel in all_channels[1:])

if all_same:
    print("All .lh5 files have the same channel list.")
else:
    print("Not all .lh5 files have the same channel list.")
    for fname, clist in channel_lists.items():
        print(f"{fname}: {clist}")