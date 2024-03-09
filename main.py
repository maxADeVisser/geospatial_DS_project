"""Main file to run the pipeline"""

import datetime as dt

import dask.dataframe
import matplotlib.pyplot as plt

from utils.plotting import plot_AIS_trace, plot_static
from utils.preprocessing import (change_data_frequency, download_ais_data,
                                 extend_main_trajectory_file,
                                 load_raw_ais_file, to_geodf, unzip_ais_data)
from utils.project_types import TimeFrequency

# Inputs
DATE = dt.datetime(2023, 8, 2, 0, 0)
OUTPUT_PATH = "test/out"
DATA_FILES = "data_files"

# %% 1. download files
downloaded_data_path: str = download_ais_data(DATE, DATA_FILES, verbose=True)
# extracted_files_path: list[str] = unzip_ais_data(downloaded_data_path, DATA_FILES)
extracted_files_path: list[str] = unzip_ais_data(
    "data_files/aisdk-2023-08-02.zip", DATA_FILES
)

# %% 2. Intantiate Dask parser
ddf: dask.dataframe = load_raw_ais_file(extracted_files_path[0])

# %% 3. (Clean data) Remove faulty AIS readings and filter for Sailboats
# Filter for sailing boats:
ddf = ddf.map_partitions(
    lambda df_partition: df_partition.query('ship_type == "Sailing"')
)

# Filter out faulty ais readings (readings that are outside Danish waters)

# TODO write these to types file as constants
min_lon = 4.250
min_lat = 53.6
max_lon = 19.5
max_lat = 61.0

query_str = (
    f"lon > {min_lon} and lat > {min_lat} and lon < {max_lon} and lat < {max_lat}"
)
ddf = ddf.map_partitions(lambda df_partition: df_partition.query(query_str))

# see data:
# data = ddf.partitions[0].compute()
# plot_static(to_geodf(data))

# %% 4. Grouping
ddf = ddf.map_partitions(lambda df_partition: df_partition.groupby("MMSI"))

# %% 5. Change data frequency to every 15 min
ddf = ddf.map_partitions(lambda df_partition: change_data_frequency(df_partition, TimeFrequency.min_15)) # TODO FIX THIS
# the problem is that ddf has groupby objects as the partitions instead of pandas dataframes. We also need to have time and MMSi as index. Maybe transform the grouby object directly into a multiindex dataframe before passing it to this step.

# %% 6. write to traj.parquet (main file with trajectories)
extend_main_trajectory_file()

# %% 7. delete downloaded .zip and data files before the next iteration of the loop
