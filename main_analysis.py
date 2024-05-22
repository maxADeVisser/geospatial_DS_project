"""This file is the main file for running the spatial clustering analysis"""

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.map_clusters import get_cluster_names
from utils.postprocessing import load_and_parse_gdf_from_file
from utils.spatial_trajectory_analysis import (
    _get_best_params,
    elbow_plot_DBSCAN,
    get_cluster_df_and_matrix,
    grid_search_DBSCAN,
    inspect_start_cluster,
    plot_grid_search_results,
    run_DBSCAN,
)

# %%

if __name__ == "__main__":
    trajs_gdf = load_and_parse_gdf_from_file(
        "out/post_processed_ais_data/all_august/august_speed_filtered_trajectories.shp"
    )

    cluster_df, cluster_matrix = get_cluster_df_and_matrix(trajs_gdf)

    # DBSCAN parameter ranges:
    eps_values = np.concatenate([np.arange(200, 901, 100), np.arange(1000, 5001, 500)])
    min_sample_values = np.arange(5, 21, 3)

    # elbow plots
    _, axs = plt.subplots(1, 2, figsize=(12, 8))
    elbow_plot_DBSCAN(
        cluster_matrix, "eps", varying_param=eps_values, constant_param=10, ax=axs[0]
    )
    elbow_plot_DBSCAN(
        cluster_matrix,
        "min_samples",
        varying_param=min_sample_values,
        constant_param=1000,
        ax=axs[1],
    )
    plt.suptitle("Elbow plots for DBSCAN hyperparameters")
    plt.tight_layout()
    plt.show()

    # Grid search (takes time when running on all august data - 30 minutes)
    grid_search_scores = grid_search_DBSCAN(
        cluster_matrix,
        eps_values=eps_values,
        min_sample_values=min_sample_values,
    )

    # plotting the grid search results:
    plot_grid_search_results(
        grid_search_scores,
        save_path="out/plots/grid_search_results.png",
    )

    best_eps, best_min_samples, best_eval_score = _get_best_params(grid_search_scores)
    # RES: got eps=1500, min_samples=14, outlier fraction=0.127

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

    # get the cluster names (also takes some time):
    cluster_names = get_cluster_names(cluster_df).reset_index()[
        ["cluster", "loc_names"]
    ]

    # add the cluster names to the trajs_gdf
    # start location names:
    trajs_gdf = (
        pd.merge(
            trajs_gdf,
            cluster_names[["cluster", "loc_names"]],
            left_on="cluster_start",
            right_on="cluster",
            how="left",
        )
        .drop(columns="cluster")
        .rename(columns={"loc_names": "start_loc"})
    )
    # end location names:
    trajs_gdf = (
        pd.merge(
            trajs_gdf,
            cluster_names[["cluster", "loc_names"]],
            left_on="cluster_end",
            right_on="cluster",
            how="left",
        )
        .drop(columns="cluster")
        .rename(columns={"loc_names": "end_loc"})
    )

    # Inspect a cluster:
    inspect_start_cluster(trajs_gdf, cluster_id=5, a=0.5)
    # cluster 4 still has outliers??

    # Export the clustered trajectories:
    export_df = trajs_gdf.copy()
    export_df["timestamp"] = export_df["timestamp"].astype(str)
    export_df["stop_split"] = export_df["stop_split"].astype(str)

    export_df.info()
    export_df.to_file("out/clustered/clustered_trajectories.shp")
