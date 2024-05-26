"""Main file to run the pipeline"""

# %%
import datetime as dt
import os

import dask.dataframe
import movingpandas as mpd
import pandas as pd
from dask.dataframe.utils import make_meta
from tqdm import tqdm

from utils.postprocessing import (
    export_trajs_as_gdf,
    filter_min_readings,
    split_by_time_gaps,
    stop_detection,
)
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
    ShipType,
)

# %%


def fetch_AIS_for_day(DATE: dt.datetime) -> None:
    downloaded_data_path: str = download_ais_data(DATE, TEMP_DATA_DIR, verbose=True)
    extracted_files_path: list[str] = unzip_ais_data(
        downloaded_data_path, TEMP_DATA_DIR
    )

    ddf: dask.dataframe = load_raw_ais_file(
        extracted_files_path[0]
    )  # there is only one file in the list

    # Filter for only sailing boats:
    print("Filtering data...")
    query_str = f"ship_type == '{SHIP_TYPE}'"
    ddf = ddf.map_partitions(lambda df_partition: df_partition.query(query_str))
    # Filter out faulty ais readings (readings that are outside Danish waters)
    query_str = f"lon > {AIS_MIN_LON} and lat > {AIS_MIN_LAT} and lon < {AIS_MAX_LON} and lat < {AIS_MAX_LAT}"
    ddf = ddf.map_partitions(lambda df_partition: df_partition.query(query_str))

    print("Grouping and resampling data...")
    meta = make_meta(
        ddf
    )  # metadata that dasks needs to know the structure of the dataframe

    print("Resampling data...")
    grouped_resampled_df: pd.DataFrame = (
        ddf.groupby("MMSI").apply(change_data_frequency, meta=meta).compute()
    )

    # Flatten the multiindex and clean up data before storing it
    flattened_df = (
        grouped_resampled_df.drop(columns=["MMSI"])
        .reset_index()  # remove multiindex
        .dropna()  # remove NaN values from the resampling process
        .drop(columns=["level_1", "ship_type"])
        .sort_values(by=["MMSI", "timestamp"])
        .reset_index(drop=True)
    )

    print("Writing to main file...")
    extend_main_trajectory_file(flattened_df, main_file_path=OUTPUT_FILE_PATH)

    # Remove temp files:
    os.remove(downloaded_data_path)
    for file in extracted_files_path:
        os.remove(file)

    print(f"Done fetching data and preprocessing for {DATE.date()}!\n\n")


# %%
if __name__ == "__main__":
    # PIPELINE INPUTS:
    OUTPUT_FILE_PATH = (
        "out/ais_data/traj_all.parquet"  # main file with all trajectories
    )
    TEMP_DATA_DIR = "out/temp_data"  # temp dir to store downloaded data
    SHIP_TYPE = ShipType.sailboat.value  # Ship type to analyse
    RESET_MAIN_FILE = True  # Set to True to reset the main file if it exists
    PROCESS_DATES = [dt.datetime(2023, 8, day) for day in range(1, 32)]

    # Run pipeline:
    # Reset main file if needed:
    if os.path.exists(OUTPUT_FILE_PATH) and RESET_MAIN_FILE:
        os.remove(OUTPUT_FILE_PATH)

    # Run data fetching pipeline for each date in PROCESS_DATES:
    for i in tqdm(
        range(0, len(PROCESS_DATES)), desc="Downloading AIS data and preprocessing ..."
    ):
        fetch_AIS_for_day(PROCESS_DATES[i])

    # ------------ Post processing of the downloaded data ---------------
    df = pd.read_parquet(OUTPUT_FILE_PATH)

    # filter out MMSI with less than 50 readings (12,5 hours of sailing time when 1 reading per 15 minutes):
    df = filter_min_readings(
        df, min_readings=50
    )  # NOTE: only a small handful of MMSIs have less than 50 readings

    # Create trajectory collection
    gdf = to_geodf(df)  # convert to geodataframe and add CRS
    trajs = mpd.TrajectoryCollection(
        gdf.set_index("timestamp"),
        traj_id_col="MMSI",
        t="timestamp",
        min_length=10_000,  # min_length is in meters
    )

    # split trajectories if there are more than 10 hours time gaps:
    obs_splitted_trajs = split_by_time_gaps(trajs, time_gap_threshold=10)

    # Apply stop detection (takes time - 29 minutes for all of august):
    stop_splitted_trajs = stop_detection(
        obs_splitted_trajs, max_diameter=1_000, min_duration=dt.timedelta(hours=3)
    )

    # add speeds
    stop_splitted_trajs.add_speed(
        overwrite=True, units=("km", "h"), name="speed (km/h)"
    )

    # convert to geodataframe
    gdf = stop_splitted_trajs.to_point_gdf()

    # Remove speed outliers (there was some crazy high speeds in the data due to faulty readings probably):
    max_speed_threshold = 100
    gdf_speed_filtered = gdf[gdf["speed (km/h)"] < max_speed_threshold]

    final_trajs = mpd.TrajectoryCollection(
        gdf_speed_filtered, traj_id_col="MMSI", t="timestamp"
    )

    # Write out file
    export_trajs_as_gdf(
        gdf_speed_filtered,
        "out/post_processed_ais_data/august_speed_filtered_trajectories.shp",
    )
