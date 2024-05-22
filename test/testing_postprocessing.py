# %%
import datetime as dt

import movingpandas as mpd
import pandas as pd

from utils.plotting import plot_traj_length_distribution, plot_trajs
from utils.postprocessing import (
    export_trajs_as_gdf,
    filter_min_readings,
    load_and_parse_gdf_from_file,
    plot_time_gap_elbow,
    split_by_time_gaps,
    split_ids,
    stop_detection,
)
from utils.preprocessing import to_geodf

# %% 1.
# filter out trajectories with less than 50 readings (12,5 hours of sailing time when 1 reading per 15 minutes):
df = pd.read_parquet("out/ais_data/traj_first_10_days.parquet")
df = filter_min_readings(df, min_readings=50)

# create a TrajectoryCollection:
gdf = to_geodf(df)
trajs = mpd.TrajectoryCollection(
    gdf.set_index("timestamp"),
    traj_id_col="MMSI",
    t="timestamp",
    min_length=10_000,  # min_length is in meters
)
# plot_traj_length_distribution(trajs)

# %% 2. Split trajectories by time gaps
# Investigate where to split the trajectories by time gaps:
# threshold_to_try = list(range(48, 10, -4)) + list(range(10, 0, -1))
# plot_time_gap_elbow(trajs, threshold_to_try)

# ! when doing the below, i think a incrementing prefix is added to the trajectory id (the MMSI in this case) to keep track of the original trajectory.
obs_splitted_trajs = split_by_time_gaps(trajs, time_gap_threshold=10)
# plot_traj_length_distribution(obs_splitted_trajs)

# Remove trajectories more than 300 kilometers # TODO should be removed ?
# filtered_trajs = mpd.TrajectoryCollection(
#     [traj for traj in trajs.trajectories if traj.get_length() < 2_000_000]
# )
# long = [traj for traj in trajs.trajectories if traj.get_length() > 2_000_000][0]
# plot_traj(long)
# plot_traj_length_distribution(filtered_trajs)

# %% 3. Apply Stop Detection
# ! when doing the below, a datetime prefix is added the to the trajectory id to keep track of the original trajectory
stop_splitted_trajs = stop_detection(
    obs_splitted_trajs, max_diameter=1_000, min_duration=dt.timedelta(hours=3)
)
# means 12 points (3 hours = 12 * 15 minutes) within 1 km distance are considered a stop
# plot_traj_length_distribution(stop_splitted_trajs)


# %% 4. add speed, timedelta, and direction to the trajectories
stop_splitted_trajs.add_speed(overwrite=True, units=("km", "h"), name="speed (km/h)")
# plot_speed_for_traj(stop_splitted_trajs.trajectories[0])

stop_splitted_trajs.add_timedelta(overwrite=True, name="time_since_last")
stop_splitted_trajs.add_direction(overwrite=True)

# %% 5. Remove present speed outliers (due to faulty AIS readings)
# ? maybe we should interpolate between points when speed is above X threshold and the point is removed??
# manually filter out extreme speeds (faulty AIS readings)

gdf = stop_splitted_trajs.to_point_gdf()
# gdf["speed (km/h)"].plot(kind="hist", bins=100)

max_speed_threshold = 100  # ! arbitrarly set # TODO set proper threshold
gdf_speed_filtered = gdf[gdf["speed (km/h)"] < max_speed_threshold]
gdf_speed_filtered["speed (km/h)"].plot(kind="hist", bins=100)
print(f"{gdf.shape[0] - gdf_speed_filtered.shape[0]} AIS readings removed")

final_trajs = mpd.TrajectoryCollection(
    gdf_speed_filtered, traj_id_col="MMSI", t="timestamp"
)
print(
    f"{len(stop_splitted_trajs.trajectories) - len(final_trajs.trajectories)} trajectories removed after speed filtering"
)

# ? could not get this to work:
# speed_filtered_trajs = mpd.OutlierCleaner(stop_splitted_trajs).clean(
#     v_max=600, units=("km", "h")
# )
# plot_trajs(speed_filtered_trajs)


# %% 5. Save the trajectories to a file
# exports the individual points (with splitting ids/prefixes though):
export_trajs_as_gdf(
    final_trajs, "out/post_processed_ais_data/speed_filtered_trajectories.shp"
)


# %% load back and parse a file:
saved_trajs = load_and_parse_gdf_from_file(
    "out/post_processed_ais_data/trajectories_test.shp"
)
saved_trajs
