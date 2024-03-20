"""Main file to run the pipeline"""

# %%

import datetime as dt

import dask.dataframe
import pandas as pd
from dask.dataframe.utils import make_meta

from utils.plotting import plot_AIS_trace, plot_static
from utils.preprocessing import (
    change_data_frequency,
    download_ais_data,
    extend_main_trajectory_file,
    load_raw_ais_file,
    to_geodf,
    unzip_ais_data,
)
from utils.project_types import (
    AIS_MAX_LAT,
    AIS_MAX_LON,
    AIS_MIN_LAT,
    AIS_MIN_LON,
    TimeFrequency,
)

# Pipeline Inputs
DATE = dt.datetime(2023, 8, 2)
OUTPUT_PATH = "test/out"
DATA_FILES = "data_files"

# %% 1. download files
downloaded_data_path: str = download_ais_data(DATE, DATA_FILES, verbose=True)
# extracted_files_path: list[str] = unzip_ais_data(downloaded_data_path, DATA_FILES)
extracted_files_path: list[str] = unzip_ais_data(
    "data_files/aisdk-2023-08-02.zip", DATA_FILES
)

# %% 2. Intantiate Dask parser
# ddf: dask.dataframe = load_raw_ais_file(extracted_files_path[0])
ddf: dask.dataframe = load_raw_ais_file("data_files/aisdk-2023-08-02.csv")  # TESTING


# %% 3. (Clean data) Remove faulty AIS readings and filter for Sailboats
# Filter for only sailing boats:
ddf = ddf.map_partitions(
    lambda df_partition: df_partition.query('ship_type == "Sailing"')
)
# Filter out faulty ais readings (readings that are outside Danish waters)
query_str = f"lon > {AIS_MIN_LON} and lat > {AIS_MIN_LAT} and lon < {AIS_MAX_LON} and lat < {AIS_MAX_LAT}"
ddf = ddf.map_partitions(lambda df_partition: df_partition.query(query_str))

# TESTING. see data:
data = ddf.partitions[1].compute()
plot_static(to_geodf(data), alpha=0.03)

# %% 4. Grouping by MMSI and change data frequency to every 15 min
meta = make_meta(
    ddf
)  # metadata that dasks needs to know the structure of the dataframe

grouped: pd.DataFrame = (
    ddf.groupby("MMSI").apply(change_data_frequency, meta=meta).compute()
)

# Plot for the whole day:
plot_static(to_geodf(grouped), alpha=0.03)

# %% 6. write to traj.parquet (main file with trajectories)
extend_main_trajectory_file()

# %% 7. delete downloaded .zip and data files before the next iteration of the loop

# %%
# TODO make if main run main logic:
