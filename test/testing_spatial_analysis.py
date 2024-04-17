# %% imports

import contextily as cx
import movingpandas as mpd
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

from utils.plotting import plot_traj_length_distribution, plot_trajs
from utils.postprocessing import create_traj_collection, load_and_parse_gdf_from_file

trajs_gdf = load_and_parse_gdf_from_file(
    "out/post_processed_ais_data/speed_filtered_trajectories.shp"
)
# %% 1. Cluster the trajectories with DBSCAN by start and end points
# Create trajectory collection:
trajs = mpd.TrajectoryCollection(
    trajs_gdf.set_index("timestamp"), traj_id_col="traj_id", t="timestamp"
)

# Get origin coordinates:
ori_and_des_df = trajs.get_start_locations()
ori_and_des_df["origin_lat"] = ori_and_des_df.geometry.y
ori_and_des_df["origin_lon"] = ori_and_des_df.geometry.x

# Get destination coordinates:
_temp_destinations = trajs.get_end_locations()
ori_and_des_df["destination_lat"] = _temp_destinations.geometry.y
ori_and_des_df["destination_lon"] = _temp_destinations.geometry.x

# matrix for clustering:
matrix = ori_and_des_df[
    ["origin_lat", "origin_lon", "destination_lat", "destination_lon"]
].values

epsilon = 5_000  # when both including origin and destination, what does the unit of epsilon become?

db = DBSCAN(eps=epsilon, min_samples=10, algorithm="ball_tree", metric="euclidean").fit(
    matrix
)
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([matrix[cluster_labels == n] for n in range(num_clusters)])
print(f"Number of clusters: {num_clusters}")

ori_and_des_df["cluster"] = cluster_labels
trajs_cluster_id = ori_and_des_df[["traj_id", "cluster"]]

trajs_gdf_clustered = pd.merge(trajs_gdf, trajs_cluster_id, on="traj_id", how="left")
trajs_clustered = mpd.TrajectoryCollection(
    trajs_gdf_clustered.set_index("timestamp"), traj_id_col="traj_id", t="timestamp"
)

# %% inspect a cluster:
selected_clusters = trajs_clustered.filter("cluster", list(range(1, 82 + 1)))
# selected_clusters = trajs_clustered.filter(
#     "cluster", [-1]
# )  # select the non-clustered points
# plot_traj_length_distribution(selected_clusters)

# plot clusters as colors
_, ax = plt.subplots(1, figsize=(12, 12))
selected_clusters.plot(ax=ax, alpha=0.3, linewidth=1, column="cluster")
cx.add_basemap(ax, crs=trajs.trajectories[0].crs, source=cx.providers.CartoDB.Positron)
plt.show()


# %% 2. Create a FlowMap from the TrajectoryCollection
# after clustering, tessellate the clusters and make a flow map
