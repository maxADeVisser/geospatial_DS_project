"""A file for util functions for the app"""

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import movingpandas as mpd
import numpy as np

from utils.project_types import MapProjection


def plot_destinations_from_start_location(
    trajs_gdf: gpd.GeoDataFrame,
    title: str,
    traj_opacity: float = 0.5,
    mark_centroids: bool = False,
) -> plt.Figure:
    """Function to plot trajectories starting from a given location and color them based on their end location."""
    colormap = plt.get_cmap("gist_ncar")
    # Create a color for each end location:
    colors = [
        colormap(i) for i in np.linspace(0, 1, trajs_gdf["end_loc"].unique().size)
    ]

    fig, ax = plt.subplots(1, figsize=(12, 12))

    for i, (end_location, group) in enumerate(trajs_gdf.groupby("end_loc")):
        trajs = mpd.TrajectoryCollection(
            group.set_index("timestamp"), traj_id_col="traj_id", t="timestamp"
        )
        trajs.plot(
            ax=ax,
            alpha=traj_opacity,
            linewidth=1,
            color=colors[i],
        )

    ax.set_title(title)

    cx.add_basemap(
        ax,
        crs=MapProjection.UTMzone32n.value,
        source=cx.providers.CartoDB.Positron,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if mark_centroids:
        # only do it for the first point in each trajectory
        start_coords = np.array(
            [
                [entry.x, entry.y]
                for entry in trajs_gdf.groupby("traj_id")["geometry"].first()
            ]
        )
        start_center = np.mean(start_coords, axis=0)
        ax.scatter(
            start_center[0], start_center[1], marker="x", color="red", s=100, zorder=10
        )

        # Mark the end points centroids
        end_grouped = trajs_gdf.groupby(["end_loc", "traj_id"]).last()["geometry"]
        for end_location in end_grouped.index.unique(level=0):
            end_coords = np.array(
                [[entry.x, entry.y] for entry in end_grouped[end_location]]
            )
            end_center = np.mean(end_coords, axis=0)
            ax.scatter(
                end_center[0],
                end_center[1],
                marker="x",
                color="green",
                s=100,
                zorder=10,
            )

    return fig


def plot_trajs_start_to_end(
    trajs_gdf: gpd.GeoDataFrame,
    title: str,
    traj_opacity: float = 0.5,
    mark_centroids: bool = False,
) -> plt.Figure:
    """Function to plot trajectories starting from a given location and ending at another location."""
    fig, ax = plt.subplots(1, figsize=(12, 12))

    trajs = mpd.TrajectoryCollection(
        trajs_gdf.set_index("timestamp"), traj_id_col="traj_id", t="timestamp"
    )
    trajs.plot(
        ax=ax,
        alpha=traj_opacity,
        linewidth=1,
        legend=True,
        column="speed (km/",
    )

    ax.set_title(title)

    cx.add_basemap(
        ax,
        crs=MapProjection.UTMzone32n.value,
        source=cx.providers.CartoDB.Positron,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if mark_centroids:
        # only do it for the first point in each trajectory
        start_coords = np.array(
            [
                [entry.x, entry.y]
                for entry in trajs_gdf.groupby("traj_id")["geometry"].first()
            ]
        )
        start_center = np.mean(start_coords, axis=0)
        ax.scatter(
            start_center[0], start_center[1], marker="x", color="red", s=100, zorder=10
        )

        # Mark the end point centroid
        end_coords = np.array(
            [
                [entry.x, entry.y]
                for entry in trajs_gdf.groupby("traj_id")["geometry"].last()
            ]
        )
        end_center = np.mean(end_coords, axis=0)
        ax.scatter(
            end_center[0], end_center[1], marker="x", color="green", s=100, zorder=10
        )

    return fig
