"""This module contains functions for postprocessing the trajectories after they have been resample and cleaned"""

import datetime as dt

import geopandas as gpd
import matplotlib.pyplot as plt
import movingpandas as mpd
import pandas as pd
from tqdm import tqdm


def filter_min_readings(df: pd.DataFrame, min_readings: int = 16) -> pd.DataFrame:
    """
    Filter out MMSIs with less than @min_readings readings.
    Default is 16 readings which is minimum 4 hours of sailing time
    """
    return df.groupby("MMSI").filter(lambda x: len(x) > min_readings)


def plot_time_gap_elbow(
    trajs: mpd.TrajectoryCollection, time_gap_threshold: list[int]
) -> None:
    """Plot the elbow curve to determine the optimal time gap threshold for splitting the trajectories."""
    base = len(trajs.trajectories)
    n_trajs_all = []
    for t in tqdm(time_gap_threshold):
        obs_splitted_trajs = mpd.ObservationGapSplitter(trajs).split(
            gap=dt.timedelta(hours=t),
            min_length=10_000,
        )
        n_trajs_all.append(len(obs_splitted_trajs.trajectories) - base)
    plt.plot(time_gap_threshold, n_trajs_all, "ro-")
    plt.xlabel("Time gap threshold split (hours)")
    plt.ylabel("Increase in the number of trajectories after splitting")
    plt.grid()
    plt.show()


def split_by_time_gaps(
    trajs: mpd.TrajectoryCollection, time_gap_threshold: int = 10
) -> mpd.TrajectoryCollection:
    """Split the trajectories when there is a time gap in the readings"""
    return mpd.ObservationGapSplitter(trajs).split(
        gap=dt.timedelta(minutes=time_gap_threshold * 60),
        min_length=10_000,
    )


def stop_detection(
    trajs: mpd.TrajectoryCollection,
    max_diameter: int,  # in CRS units
    min_duration: dt.timedelta,
    min_length: int = 10_000,  # in CRS units
) -> mpd.TrajectoryCollection:
    """Detect stops in the trajectories and splits them"""
    return mpd.StopSplitter(trajs).split(
        max_diameter=max_diameter, min_duration=min_duration, min_length=min_length
    )


def split_ids(traj_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    After applying the observation and stop splitting, there are prefixes added to the individuals trajectory ids. This functions splits these into seperate columns
    """
    traj_gdf = traj_gdf.rename(columns={"MMSI": "traj_id"})
    traj_gdf["MMSI"] = traj_gdf["traj_id"].apply(lambda x: x.split("_")[0]).astype(int)
    traj_gdf["obs_split_id"] = (
        traj_gdf["traj_id"].apply(lambda x: x.split("_")[1]).astype(int)
    )
    traj_gdf["stop_split_id"] = pd.to_datetime(
        traj_gdf["traj_id"].apply(lambda x: x.split("_")[2])
    )
    return traj_gdf


def export_trajs_as_gdf(trajs_gdf: gpd.GeoDataFrame, path: str) -> None:
    """NOTE: This function assumes that observation and stop splitting has been applied to the trajectories!"""
    trajs_gdf = split_ids(trajs_gdf).reset_index()
    # export driver does not support datetime objects, so convert to strings:
    trajs_gdf["stop_split_id"] = trajs_gdf["stop_split_id"].astype(str)
    trajs_gdf["timestamp"] = trajs_gdf["timestamp"].astype(str)
    trajs_gdf.to_file(path)


def load_and_parse_gdf_from_file(file_path: str) -> gpd.GeoDataFrame:
    """Load and parse the geodataframe from file"""
    gdf = gpd.read_file(file_path)
    # ! the export driver truncates the names of the columns. Grrrr. annoying
    gdf["stop_split"] = pd.to_datetime(gdf["stop_split"])
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"])
    return gdf
