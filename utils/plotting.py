import contextily as cx
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt


def zoom_center(
    lons: list[float],
    lats: list[float],
) -> tuple[float, dict[str, float]]:
    """
    Copied from https://stackoverflow.com/questions/63787612/plotly-automatic-zooming-for-mapbox-maps
    """
    width_to_height = 0.7

    max_lon, min_lon = max(lons, default=180), min(lons, default=-180)
    max_lat, min_lat = max(lats, default=90), min(lats, default=-90)
    center = {
        "lon": round((max_lon + min_lon) / 2, 6),
        "lat": round((max_lat + min_lat) / 2, 6),
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array(
        [
            0.0007,
            0.0014,
            0.003,
            0.006,
            0.012,
            0.024,
            0.048,
            0.096,
            0.192,
            0.3712,
            0.768,
            1.536,
            3.072,
            6.144,
            11.8784,
            23.7568,
            47.5136,
            98.304,
            190.0544,
            360.0,
        ]
    )

    margin = 1.5
    lat_height = (max_lat - min_lat) * margin * width_to_height
    lon_width = (max_lon - min_lon) * margin / width_to_height
    lon_zoom = np.interp(lon_width, lon_zoom_range, range(20, 0, -1))
    lat_zoom = np.interp(lat_height, lon_zoom_range, range(20, 0, -1))
    zoom = round(min(lon_zoom, lat_zoom), 2)

    return zoom, center


def plot_AIS_trace(ais_df: pd.DataFrame, trace_granularity: int = 5) -> None:
    """Plots the AIS trace on a interactive plotly map."""
    map_objects: list[go.Scattermapbox] = []
    # unwrapping longitudes to avoid international dateline plotting issues
    ais_lats = ais_df["lat"].round(trace_granularity)
    ais_lons = np.unwrap(ais_df["lon"].round(trace_granularity), period=360)
    ais_color = "firebrick"
    ais_trace = go.Scattermapbox(
        mode="lines",
        lon=ais_lons,
        lat=ais_lats,
        name="Actual route",
        marker=go.scattermapbox.Marker(color=ais_color),
    )
    map_objects.append(ais_trace)

    zoom, center = zoom_center(
        lats=ais_lats,
        lons=ais_lons,
    )

    layout = go.Layout(
        margin={"l": 0, "t": 0, "b": 0, "r": 0},
        mapbox={
            "style": "carto-positron",
            "center": center,
            "zoom": zoom,
        },
        legend={"title": "Routes"},
    )

    fig = go.Figure(data=map_objects, layout=layout)
    fig.update_layout(margin={"l": 0, "t": 0, "b": 0, "r": 0})
    return fig


def plot_static(
    trajectory_df: gpd.GeoDataFrame,
    size: int = 12,
    alpha: float = 0.1,
    marker_size: int = 1,
    save_path: str | None = None,
) -> None:
    """Plots the AIS trace on a static map"""
    _, ax = plt.subplots(1, figsize=(size, size))
    trajectory_df.plot(ax=ax, alpha=alpha, color="blue", markersize=marker_size)
    cx.add_basemap(ax, crs=trajectory_df.crs, source=cx.providers.CartoDB.Positron)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_trajs(trajs: mpd.TrajectoryCollection, a: float = 0.3) -> None:
    _, ax = plt.subplots(1, figsize=(12, 12))
    trajs.plot(ax=ax, alpha=a, color="blue", linewidth=1)
    cx.add_basemap(
        ax, crs=trajs.trajectories[0].crs, source=cx.providers.CartoDB.Positron
    )
    plt.show()


def plot_traj_length_distribution(
    trajs: mpd.TrajectoryCollection, bins: int = 50
) -> None:
    """Plot the distribution of the length of the splitted trajectories"""
    lengths = [traj.get_length() / 1_000 for traj in trajs.trajectories]
    sns.histplot(lengths, bins=bins)
    plt.xlabel("Length of Trajectory (km)")
    plt.ylabel("Counts")
    plt.grid(axis="y", alpha=0.75)
    plt.gca().set_axisbelow(True)  # Set grid lines behind bars
    plt.show()


def plot_traj(trajs: mpd.Trajectory) -> None:
    """Plot a single trajectory"""
    return trajs.hvplot(
        title=f"{trajs.id}",
        line_width=3,
        line_color="blue",
        colorbar=True,
        cmap="RdYlGn",
    )


def plot_speed_for_traj(traj: mpd.Trajectory) -> None:
    """Plot the speed of the TrajectoryCollection"""
    traj.plot(column="speed (km/h)", linewidth=5, capstyle="round", legend=True)
