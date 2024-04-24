# %% imports

import contextily as cx
import movingpandas as mpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from utils.plotting import plot_static, plot_traj_length_distribution, plot_trajs
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
# Get destination coordinates:
_temp_destinations = trajs.get_end_locations()

cluster_df = pd.DataFrame(
    {
        "traj_id": pd.concat(
            [ori_and_des_df["traj_id"], _temp_destinations["traj_id"]]
        ),
        "x_coords": pd.concat(
            [ori_and_des_df.geometry.x, _temp_destinations.geometry.x]
        ),
        "y_coords": pd.concat(
            [ori_and_des_df.geometry.y, _temp_destinations.geometry.y]
        ),
    }
).reset_index(drop=True)

# add start and end labels for good measure:
cluster_df["label"] = ["start"] * len(ori_and_des_df) + ["end"] * len(
    _temp_destinations
)


# Run DBSCA
matrix = cluster_df[["x_coords", "y_coords"]].values


# Elbow plot for epsilon and min_samples
# epsilon = 500  # unit is meters (same as CRS)
epsilon_values = np.append(np.linspace(10, 200, 20), np.linspace(200, 1000, 9))
temp_best_eps = 180
temp_best_min_samples = 5
min_samples_values = np.arange(1, 30, 1)

n_clusters = []
# for eps in tqdm(epsilon_values):
for min in tqdm(min_samples_values):
    # db = DBSCAN(eps=eps, min_samples=10, algorithm="ball_tree", metric="euclidean").fit(
    #     matrix
    # )
    db = DBSCAN(
        eps=10,
        min_samples=12,
        algorithm="ball_tree",
        metric="euclidean",
    ).fit(matrix)
    cluster_labels = db.labels_
    n_clusters.append(len(set(cluster_labels)))
# plt.plot(epsilon_values, n_clusters, "ro-")
plt.plot(min_samples_values, n_clusters, "ro-")
plt.xlabel("Epsilon")
plt.ylabel("Number of clusters")
plt.grid()
plt.show()

# ---- Grid search -----
scores = []
for eps in tqdm(epsilon_values):
    for min in min_samples_values:
        db = DBSCAN(
            eps=eps,
            min_samples=min,
            algorithm="ball_tree",
            metric="euclidean",
        ).fit(matrix)

        _temp_df = pd.DataFrame(
            {"x": matrix[:, 0], "y": matrix[:, 1], "cluster": db.labels_}
        )

        # remove points that are not clustered:
        # _temp_df = _temp_df.query("cluster != -1")

        # calculate silhoutte
        if _temp_df["cluster"].nunique() <= 1:
            sil_score = (-1, None)
        else:
            sil_score = (
                silhouette_score(
                    _temp_df[["x", "y"]], _temp_df["cluster"], metric="euclidean"
                ),
            )

        scores.append((eps, min, sil_score))

unzipped_lists = list(zip(*scores))
max_sil_score = 0.0
best_idx = 0
for idx, score in enumerate(unzipped_lists[2]):
    print(idx)
    print(score)
    if isinstance(score, tuple):
        if score[0] > max_sil_score:
            max_sil_score = score[0]
            print(max_sil_score)
            best_idx = idx
    else:
        continue
scores[best_idx]


# -----

cluster_df["cluster"] = cluster_labels
trajs_gdf["cluster_start"] = pd.merge(
    trajs_gdf, cluster_df.query("label == 'start'"), on="traj_id", how="left"
)["cluster"]
trajs_gdf["cluster_end"] = pd.merge(
    trajs_gdf, cluster_df.query("label == 'end'"), on="traj_id", how="left"
)["cluster"]


# ! TESTING:
plot_df = cluster_df.query("cluster == 20")
plt.scatter(plot_df["x_coords"], plot_df["y_coords"], c=plot_df["cluster"], alpha=0.5)
plt.legend()
plt.show()


# ------ OLD PLOTTING CODE NOT ADADPTED TO NEW APPROACH ------
trajs_gdf_clustered = pd.merge(
    trajs_gdf, cluster_df[["traj_id", "cluster"]], on="traj_id", how="left"
)
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
