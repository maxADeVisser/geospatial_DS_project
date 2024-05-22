"""This module contains functions for spatial trajectory analysis"""

# %%
from typing import Callable, Literal

import contextily as cx
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def get_cluster_df_and_matrix(trajs_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Gets the starting coordinates and ending coordinates of the trajectories and labels them as start and end respectively."""
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
    """Runs DBSCAN on the cluster matrix and returns the cluster labels for each row in the cluster_matrix. eps has the unit of the CRS of the data"""

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric=distance_metric,
    ).fit(cluster_matrix)

    cluster_labels = db.labels_
    if verbose:
        print(f"Number of clusters: {len(set(cluster_labels))}")
        print(
            f"Fraction of noise points: {np.sum(cluster_labels == -1) / len(cluster_labels)}"
        )
    return cluster_labels


def elbow_plot_DBSCAN(
    cluster_matrix: np.ndarray,
    iterating_param: Literal["eps", "min_samples"],
    varying_param: list[int],
    constant_param: int,
    ax=None,
    figsize=(12, 8),
) -> None:
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)

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
    ax.plot(varying_param, n_clusters, "ro-")
    ax.set_xlabel(
        f"{iterating_param} values\nwith {other_param} being constant: {constant_param}"
    )
    ax.set_ylabel("Number of clusters")
    ax.grid()
    return ax


def cluster_evalutation(X: np.ndarray, y: np.ndarray, n_outliers: int) -> float:
    """Evaluation function for DBSCAN clustering. Returns the silhouette score minus the fraction of outliers squared. This is to penalize the model for having too many outliers."""
    return silhouette_score(X, y, metric="euclidean") - ((n_outliers / len(y)) ** 2)


def grid_search_DBSCAN(
    cluster_matrix: np.ndarray,
    eps_values: list[int],
    min_sample_values: list[int],
    eval_func: Callable = cluster_evalutation,
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

            # compute evaluation score:
            _temp_array = np.column_stack((cluster_matrix, cluster_labels))
            if np.unique(_temp_array[:, -1]).size <= 1:  # if only one cluster is formed
                eval_score = (-1, None)

            n_outliers = np.sum(cluster_labels == -1)  # n elements in noise cluster
            eval_score = eval_func(_temp_array[:, :-1], _temp_array[:, -1], n_outliers)

            print(
                f"Eps: {eps}, Min sample: {min}, score: {eval_score}, outliers fraction: {n_outliers / len(cluster_labels)}, n clusters: {len(set(cluster_labels))}"
            )
            all_scores.append((eps, min, eval_score))
    return all_scores


def plot_grid_search_results(
    grid_search_results: list[tuple[int, int, float]], save_path: str
) -> None:
    unzipped_lists = list(zip(*grid_search_results))
    df = pd.DataFrame(unzipped_lists).transpose()
    df.columns = ["Epsilon", "Min_sample", "Evaluation Score"]
    heatmap_data = df.pivot(
        index="Epsilon", columns="Min_sample", values="Evaluation Score"
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm")
    plt.title("Heatmap of DBSCAN grid search results")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def _get_best_params(
    all_scores: list[tuple[int, int, float]]
) -> tuple[int, int, float]:
    """Util function for grid_search_DBSCAN(). Returns the best parameters based on the evaluation score."""
    unzipped_lists = list(zip(*all_scores))
    best_score = 0.0
    best_score_idx = 0
    for idx, eval_score in enumerate(unzipped_lists[2]):
        if eval_score > best_score:
            best_score = eval_score
            best_score_idx = idx
    best_results = all_scores[best_score_idx]
    print(
        f"Best is eps {best_results[0]}, min_samples {best_results[1]} with score {best_results[2]}"
    )
    return best_results


def inspect_start_cluster(trajs_gdf: pd.DataFrame, cluster_id: int, a=0.5) -> None:
    plot_df = trajs_gdf.query(f"cluster_start == {cluster_id}")
    unique_end_clusters = plot_df["cluster_end"].unique()
    print(f"n end clusters: {unique_end_clusters.size}")

    colormap = plt.get_cmap("gist_ncar")
    colors = [colormap(i) for i in np.linspace(0, 1, unique_end_clusters.size)]

    _, ax = plt.subplots(1, figsize=(12, 12))

    for i, e in enumerate(unique_end_clusters):
        endpoint_df = plot_df.query(f"cluster_end == {e}")
        trajs = mpd.TrajectoryCollection(
            endpoint_df.set_index("timestamp"), traj_id_col="traj_id", t="timestamp"
        )
        trajs.plot(ax=ax, alpha=a, color=colors[i], linewidth=1)

    cx.add_basemap(
        ax, crs=trajs.trajectories[0].crs, source=cx.providers.CartoDB.Positron
    )
    plt.show()
