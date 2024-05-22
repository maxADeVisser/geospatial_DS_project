# %%
import os

import pandas as pd

from utils.preprocessing import extend_main_trajectory_file

# see https://stackoverflow.com/questions/47191675/pandas-write-dataframe-to-parquet-format-with-append

# Create existing file
all_data = "test/out/small10k.parquet"
start_idx = 3
end_idx = 6

# read in "existing data"
df = pd.read_parquet(all_data)
existing_file_path = "test/out/traj.parquet"
df[:start_idx].to_parquet(existing_file_path)

# %%

# "new" data
new_data = df[start_idx:end_idx]
extend_main_trajectory_file(new_data, existing_file_path)

pd.read_parquet(existing_file_path)
