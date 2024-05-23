"""A file for util functions for the app"""

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import movingpandas as mpd
import numpy as np

from utils.project_types import MapProjection


# TODO figure out how to cache this function (maybe not needed?)
def inspect_start_cluster_app(
    trajs_gdf: gpd.GeoDataFrame,
    title: str,
    size: int,
    traj_opacity=0.5,
    mark_centroid: bool = False,
    show_speed: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(1, figsize=(size, size))
    trajs = mpd.TrajectoryCollection(
        trajs_gdf.set_index("timestamp"), traj_id_col="traj_id", t="timestamp"
    )
    trajs.plot(
        ax=ax,
        alpha=traj_opacity,
        linewidth=1,
        column="speed (km/" if show_speed else None,
        legend=True,
    )
    ax.set_title(title)

    cx.add_basemap(
        ax,
        crs=MapProjection.UTMzone32n.value,
        source=cx.providers.CartoDB.Positron,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if mark_centroid:  # TODO make this work properly
        coords = np.array([[entry.x, entry.y] for entry in trajs_gdf["geometry"]])
        center = np.mean(coords, axis=0)
        ax.scatter(center[0], center[1], marker="x", color="red", s=100)

    return fig
