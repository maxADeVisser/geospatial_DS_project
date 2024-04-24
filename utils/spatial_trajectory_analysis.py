# %%
from typing import Callable, Literal

import contextily as cx
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.plotting import plot_static, plot_traj_length_distribution, plot_trajs
from utils.postprocessing import create_traj_collection, load_and_parse_gdf_from_file


def get_cluster_df_and_matrix(trajs_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Gets the starting coordinates and ending coordinates of the trajectories and labels them as start and end respectively."""
    # redunant computation below. Can just use timestamps from the trajs_gdf to get start and end locations
    trajs = mpd.TrajectoryCollection(
        trajs_gdf.set_index("timestamp"), traj_id_col="traj_id", t="timestamp"
    )
    ori_and_des_df = trajs.get_start_locations()
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

    cluster_df["label"] = ["start"] * len(ori_and_des_df) + ["end"] * len(
        _temp_destinations
    )
    return cluster_df, cluster_df[["x_coords", "y_coords"]].values


def run_DBSCAN(
    cluster_matrix: np.ndarray,
    eps: int,
    min_samples: int,
    distance_metric: str = "euclidean",
    verbose: bool = False,
) -> np.ndarray:
    """Runs DBSCAN on the cluster matrix and returns the cluster labels for each row in the cluster_matrix"""
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric=distance_metric,
    ).fit(cluster_matrix)

    cluster_labels = db.labels_
    if verbose:
        print(f"Number of clusters: {len(set(cluster_labels))}")
        print(f"Number of noise points: {list(cluster_labels).count(-1)}")
    return cluster_labels


def scale_cluster_matrix(cluster_matrix: np.ndarray) -> np.ndarray:
    """Scales the cluster matrix using the StandardScaler"""
    return StandardScaler().fit_transform(cluster_matrix)


def elbow_plot_DBSCAN(
    cluster_matrix: np.ndarray,
    iterating_param: Literal["eps", "min_samples"],
    varying_param: list[int],
    constant_param: int,
) -> None:
    """TODO"""
    n_clusters = []
    other_param = "min_samples" if iterating_param == "eps" else "eps"
    for param in tqdm(varying_param):
        params_dict = {
            "eps": param if iterating_param == "eps" else constant_param,
            "min_samples": (
                param if iterating_param == "min_samples" else constant_param
            ),
        }
        labels = run_DBSCAN(cluster_matrix, **params_dict)
        n_clusters.append(len(set(labels)))
    plt.plot(varying_param, n_clusters, "ro-")
    plt.xlabel(
        f"{iterating_param} values\nwith {other_param} being constant: {constant_param}"
    )
    plt.ylabel("Number of clusters")
    plt.grid()
    plt.show()


def grid_search_DBSCAN(
    cluster_matrix: np.ndarray,
    eps_values: list[int],
    min_sample_values: list[int],
    eval_func: Callable,
    eval_func_kwargs: dict = {},
    dbscan_metric: str = "euclidean",
) -> tuple[int, int, float]:
    """Grid search implementation. Returns the best epsilon and min_samples values for DBSCAN along with the evaluation score for thse parameters.
    @evalutation_func must take in the following arguments: (X: pd.DataFrame, y: pd.Series) and return a float.
    @eval_kwargs is a dictionary of additional keyword arguments passed to the evaluation function.
    Returns: (best_eps, best_min_samples, best_eval_score)
    """
    all_scores = []
    for eps in tqdm(eps_values):
        for min in min_sample_values:
            cluster_labels = run_DBSCAN(
                cluster_matrix,
                eps,
                min,
                distance_metric=dbscan_metric,
            )
            _temp_array = np.column_stack((cluster_matrix, cluster_labels))
            # calculate evaluation score:
            if np.unique(_temp_array[:, -1]).size <= 1:  # if only one cluster is formed
                eval_score = (-1, None)
            else:
                eval_score = (
                    eval_func(
                        _temp_array[:, :-1], _temp_array[:, -1], **eval_func_kwargs
                    ),
                )
            all_scores.append((eps, min, eval_score))

    # Get best parameters:
    unzipped_lists = list(zip(*all_scores))
    best_score = 0.0
    best_score_idx = 0
    for idx, eval_score in enumerate(unzipped_lists[2]):
        if isinstance(eval_score, tuple):
            if eval_score[0] > best_score:
                best_score = eval_score[0]
                best_score_idx = idx
        else:
            continue

    return all_scores[best_score_idx]


# %%

if __name__ == "__main__":
    trajs_gdf = load_and_parse_gdf_from_file(
        "out/post_processed_ais_data/speed_filtered_trajectories.shp"
    )

    cluster_df, cluster_matrix = get_cluster_df_and_matrix(trajs_gdf)
    # matrix = scale_cluster_matrix(matrix) # scaling ?

    # elbow plots
    eps_vars = np.append(np.linspace(10, 200, 20), np.linspace(200, 1000, 9))
    elbow_plot_DBSCAN(cluster_matrix, "eps", varying_param=eps_vars, constant_param=10)

    min_samples_values = np.arange(1, 30, 1)
    elbow_plot_DBSCAN(
        cluster_matrix,
        "min_samples",
        varying_param=min_samples_values,
        constant_param=1000,
    )

    best_eps, best_min_samples, best_eval_score = grid_search_DBSCAN(
        cluster_matrix,
        [1000, 2000],
        [10, 20],
        eval_func=silhouette_score,  # TODO add a better evaluation function
        eval_func_kwargs=dict(metric="euclidean"),
    )

    cluster_labels = run_DBSCAN(
        cluster_matrix, eps=best_eps, min_samples=best_min_samples, verbose=True
    )
    cluster_df["cluster"] = cluster_labels

    # add start and end cluster labels to original trajs_gdf:
    trajs_gdf["cluster_start"] = pd.merge(
        trajs_gdf, cluster_df.query("label == 'start'"), on="traj_id", how="left"
    )["cluster"]
    trajs_gdf["cluster_end"] = pd.merge(
        trajs_gdf, cluster_df.query("label == 'end'"), on="traj_id", how="left"
    )["cluster"]

# %%
