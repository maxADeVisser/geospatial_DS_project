"""A file for util functions for the app"""

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import movingpandas as mpd
import numpy as np

from utils.project_types import MapProjection


def inspect_start_cluster_app(
    trajs_gdf: gpd.GeoDataFrame,
    title: str,
    size: int,
    traj_opacity: float = 0.5,
    mark_centroids: bool = False,
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


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.collections import LineCollection

# # Example DataFrame
# data = {
#     "id": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
#     "timestamp": pd.date_range("2021-01-01", periods=9, freq="H"),
#     "x": np.random.rand(9),
#     "y": np.random.rand(9),
# }

# df = pd.DataFrame(data)


# # Function to create LineCollection for each group
# def create_line_collection(df):
#     fig, ax = plt.subplots()

#     for name, group in df.groupby("end_loc"):
#         group = group.sort_values(by="timestamp")

#         # Create segments from the coordinates
#         points = group[[group["geometry"].x, group["geometry"].y]].values
#         segments = [list(zip(points[:-1], points[1:]))]
#         segments = np.concatenate(segments)

#         # Create a LineCollection from the segments
#         lc = LineCollection(segments, linewidths=2, label=name)

#         # Add the LineCollection to the plot
#         ax.add_collection(lc)

#     # Set plot limits
#     ax.set_xlim(df["x"].min() - 0.1, df["x"].max() + 0.1)
#     ax.set_ylim(df["y"].min() - 0.1, df["y"].max() + 0.1)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.legend()
#     ax.set_title("Line Collection for Each Group of Points")

#     plt.show()


# # Call the function
# create_line_collection(df)
