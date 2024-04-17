# %%
import datetime as dt

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import movingpandas as mpd
import pandas as pd
from tqdm import tqdm

from utils.plotting import (
    plot_speed_for_traj,
    plot_traj,
    plot_traj_length_distribution,
    plot_trajs,
)
from utils.postprocessing import (
    create_traj_collection,
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
trajs = create_traj_collection(df, min_length=10_000)
plot_traj_length_distribution(trajs)

# %% 2. Split trajectories by time gaps
# threshold_to_try = list(range(48, 10, -4)) + list(range(10, 0, -1))
# plot_time_gap_elbow(trajs, threshold_to_try)

# ! when doing the below, i think a incrementing prefix is added to the trajectory id (the MMSI in this case) to keep track of the original trajectory.
# ! This needs to be handled when saving the file.
obs_splitted_trajs = split_by_time_gaps(trajs, time_gap_threshold=10)
plot_traj_length_distribution(obs_splitted_trajs)

# Remove trajectories more than 300 kilometers
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
plot_traj_length_distribution(stop_splitted_trajs)


# %% 4. add speed, timedelta, and direction to the trajectories
stop_splitted_trajs.add_speed(overwrite=True, units=("km", "h"), name="speed (km/h)")
# plot_speed_for_traj(stop_splitted_trajs.trajectories[0])

stop_splitted_trajs.add_timedelta(overwrite=True, name="time_since_last")
stop_splitted_trajs.add_direction(overwrite=True)

# %% 5. Save the trajectories to a file
# exports the individual points (with splitting ids/prefixes though):

export_trajs_as_gdf(
    stop_splitted_trajs, "out/post_processed_ais_data/trajectories_test.shp"
)

# %% load back and parse a file:
# trajs = load_and_parse_gdf_from_file(
#     "out/post_processed_ais_data/trajectories_test.shp"
# )
