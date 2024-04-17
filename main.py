"""Main file to run the pipeline"""

# %%
import datetime as dt
import os

import dask.dataframe
import pandas as pd
from dask.dataframe.utils import make_meta
from tqdm import tqdm

from utils.preprocessing import (
    change_data_frequency,
    download_ais_data,
    extend_main_trajectory_file,
    load_raw_ais_file,
    to_geodf,
    unzip_ais_data,
)
from utils.project_types import AIS_MAX_LAT, AIS_MAX_LON, AIS_MIN_LAT, AIS_MIN_LON

# %%


def main(DATE: dt.datetime) -> None:
    # %% 1. download files
    downloaded_data_path: str = download_ais_data(DATE, TEMP_DATA_DIR, verbose=True)
    extracted_files_path: list[str] = unzip_ais_data(
        downloaded_data_path, TEMP_DATA_DIR
    )

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
    query_str = f"ship_type == '{SHIP_TYPE}'"
    ddf = ddf.map_partitions(lambda df_partition: df_partition.query(query_str))
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

    print(f"Done processing {DATE.date()}!\n\n")


# %%
if __name__ == "__main__":
    # PIPELINE INPUTS:
    OUTPUT_FILE_PATH = (
        "out/ais_data/fishing_first_10_days.parquet"  # main file with trajectories
    )
    TEMP_DATA_DIR = "out/temp_data"  # temp dir to store downloaded data
    SHIP_TYPE = "Fishingxhhh"
    RESET_MAIN_FILE = True  # Set to True to reset the main file if it exists
    PROCESS_DATES = [
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
        # dt.datetime(2023, 8, 11),
        # dt.datetime(2023, 8, 12),
        # dt.datetime(2023, 8, 13),
        # dt.datetime(2023, 8, 14),
        # dt.datetime(2023, 8, 15),
        # dt.datetime(2023, 8, 16),
        # dt.datetime(2023, 8, 17),
        # dt.datetime(2023, 8, 18),
        # dt.datetime(2023, 8, 19),
        # dt.datetime(2023, 8, 20),
        # dt.datetime(2023, 8, 21),
        # dt.datetime(2023, 8, 22),
        # dt.datetime(2023, 8, 23),
        # dt.datetime(2023, 8, 24),
        # dt.datetime(2023, 8, 25),
        # dt.datetime(2023, 8, 26),
        # dt.datetime(2023, 8, 27),
        # dt.datetime(2023, 8, 28),
        # dt.datetime(2023, 8, 29),
        # dt.datetime(2023, 8, 30),
        # dt.datetime(2023, 8, 31),
    ]

    # Run pipeline:
    # Reset main file if needed:
    if os.path.exists(OUTPUT_FILE_PATH) and RESET_MAIN_FILE:
        os.remove(OUTPUT_FILE_PATH)

    for i in tqdm(range(0, len(PROCESS_DATES)), desc="Downloading AIS data"):
        main(PROCESS_DATES[i])
