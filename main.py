"""Main file to run the pipeline"""

# %%
import datetime as dt
import os

import dask.dataframe
import pandas as pd
from dask.dataframe.utils import make_meta
from tqdm import tqdm

from utils.plotting import plot_static
from utils.preprocessing import (
    change_data_frequency,
    download_ais_data,
    extend_main_trajectory_file,
    load_raw_ais_file,
    to_geodf,
    unzip_ais_data,
)
from utils.project_types import AIS_MAX_LAT, AIS_MAX_LON, AIS_MIN_LAT, AIS_MIN_LON


def main(DATE: dt.datetime) -> None:
    # %% 1. download files
    downloaded_data_path: str = download_ais_data(DATE, DATA_FILES, verbose=True)
    print("Extracting files...")
    extracted_files_path: list[str] = unzip_ais_data(downloaded_data_path, DATA_FILES)

    # TESTING:
    # extracted_files_path: list[str] = unzip_ais_data(
    #     "data_files/aisdk-2023-08-02.zip", DATA_FILES
    # )

    # %% 2. Intantiate Dask parser
    ddf: dask.dataframe = load_raw_ais_file(
        extracted_files_path[0]
    )  # there is only one file in the list

    # TESTING:
    # ddf: dask.dataframe = load_raw_ais_file(
    #     "data_files/aisdk-2023-08-02.csv"
    # )

    # %% 3. (Clean data) Remove faulty AIS readings and filter for Sailboats
    # Filter for only sailing boats:
    print("Filtering data...")
    ddf = ddf.map_partitions(
        lambda df_partition: df_partition.query('ship_type == "Sailing"')
    )
    # Filter out faulty ais readings (readings that are outside Danish waters)
    query_str = f"lon > {AIS_MIN_LON} and lat > {AIS_MIN_LAT} and lon < {AIS_MAX_LON} and lat < {AIS_MAX_LAT}"
    ddf = ddf.map_partitions(lambda df_partition: df_partition.query(query_str))

    # TESTING. plot data:
    # data = ddf.partitions[1].compute()
    # plot_static(to_geodf(data), alpha=0.03)

    # %% 4. Grouping by MMSI and change data frequency to every 15 min
    print("Grouping and resampling data...")
    meta = make_meta(
        ddf
    )  # metadata that dasks needs to know the structure of the dataframe

    print("Resampling data...")
    grouped_resampled_df: pd.DataFrame = (
        ddf.groupby("MMSI").apply(change_data_frequency, meta=meta).compute()
    )

    # Plot for the whole day: TESTING
    # plot_static(to_geodf(grouped_resampled_df), alpha=0.03)

    # Flatten the multiindex and clean up data before storing it
    flattened_df = (
        grouped_resampled_df.drop(columns=["MMSI"])
        .reset_index()  # remove multiindex
        .dropna()  # remove NaN values from the resampling process
        .drop(columns=["level_1", "ship_type"])
        .sort_values(by=["MMSI", "timestamp"])
        .reset_index(drop=True)
    )

    # %% 6. write to traj.parquet (main file with trajectories)
    print("Writing to main file...")
    extend_main_trajectory_file(flattened_df, main_file_path=OUTPUT_FILE_PATH)

    # %% 7. delete downloaded .zip and data files before the next iteration of the loop
    os.remove(downloaded_data_path)
    for file in extracted_files_path:
        os.remove(file)

    print(f"Done processing {DATE.date()}!")


# %%
if __name__ == "__main__":
    # PIPELINE INPUTS:
    ALL_DATES = [
        dt.datetime(2023, 8, 1),
        dt.datetime(2023, 8, 2),
        dt.datetime(2023, 8, 3),
        dt.datetime(2023, 8, 4),
        dt.datetime(2023, 8, 5),
        dt.datetime(2023, 8, 6),
        dt.datetime(2023, 8, 7),
        dt.datetime(2023, 8, 8),
        dt.datetime(2023, 8, 9),
        dt.datetime(2023, 8, 10),
    ]  # list of dates to process
    OUTPUT_FILE_PATH = "test/out/traj.parquet"  # main file with trajectories
    DATA_FILES = "data_files"  # folder to store the downloaded data (temporarily)
    RESET_MAIN_FILE = True  # Set to True to reset the main file

    # Run pipeline:
    # Reset main file if needed:
    if os.path.exists(OUTPUT_FILE_PATH) and RESET_MAIN_FILE:
        os.remove(OUTPUT_FILE_PATH)

    for i in tqdm(range(0, len(ALL_DATES)), desc="Processing dates"):
        main(ALL_DATES[i])
