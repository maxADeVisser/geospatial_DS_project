"""After the main trajectory files are created, run the movingpandas and other stuff here"""

# %%
import datetime as dt

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import movingpandas as mpd
import pandas as pd
import seaborn as sns

from utils.plotting import (
    plot_AIS_trace,
    plot_static,
    plot_traj,
    plot_traj_length_distribution,
    plot_trajs,
)
from utils.preprocessing import to_geodf

# ! TESTING:
df = pd.read_parquet("test/out/traj_first_10_days.parquet")


# %% 1. filter out trajectories with less than X readings
def filter_min_readings(df: pd.DataFrame, min_readings: int = 16) -> pd.DataFrame:
    """Filter out trajectories with less than @min_readings readings.
    Default is 16 readings which is minimum 4 hours of sailing time"""
    return df.groupby("MMSI").filter(lambda x: len(x) > min_readings)


df = filter_min_readings(df, min_readings=50)

# %% 2. Create a GeoDataFrame from the main trajectory file
gdf = to_geodf(df)
# plot_static(gdf, alpha=0.03)


# %% 3. Create a TrajectoryCollection from the GeoDataFrame
# TODO THERE NEEDS TO BE SOME CLEANING OF THE TRAJECTORIES !!!
# ! TESTING
trajs = mpd.TrajectoryCollection(
    gdf.set_index("timestamp"),
    traj_id_col="MMSI",
    t="timestamp",
    min_length=5_000,  # exclude trajectories less than 5 km
)
plot_traj_length_distribution(trajs)

# Remove trajectories more than 300 kilometers
filtered_trajs = mpd.TrajectoryCollection(
    [traj for traj in trajs.trajectories if traj.get_length() < 2_000_000]
)
long = [traj for traj in trajs.trajectories if traj.get_length() > 2_000_000][0]
plot_traj(long)

plot_traj_length_distribution(filtered_trajs)

# Add speed:
# trajs.add_speed(overwrite=True, units=("km", "h"), name="speed (km/h)")
# trajs.add_timedelta(overwrite=True, name="time_since_last")
# trajs.add_direction(overwrite=True)


# %% 4. Stop Detection
# ! FOR TESTING:
# Split the trajectories into smaller ones at detected stops:
splitted_trajs: mpd.TrajectoryCollection = mpd.StopSplitter(trajs).split(
    max_diameter=300, min_duration=dt.timedelta(minutes=30), min_length=10_000
)

plot_trajs(splitted_trajs)

plot_traj_length_distribution(splitted_trajs)
long_traj = [
    x for _, x in enumerate(splitted_trajs.trajectories) if x.get_length() > 270_000
][0]
plot_traj(long_traj)


# %% 5. Create a FlowMap from the TrajectoryCollection
