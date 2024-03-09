import contextily as cx
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from utils.preprocessing import to_geodf
from utils.project_types import MapProjection


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
    trajectory_df: gpd.GeoDataFrame
) -> None:
    """Plots the AIS trace on a static map"""
    _, ax = plt.subplots(1)
    ax.set_title(f"Trajectory of MMSI: {trajectory_df['MMSI'].iloc[0]}\n")
    trajectory_df.plot(ax=ax)
    ax.set_axis_off()
    cx.add_basemap(ax, crs=trajectory_df.crs, source=cx.providers.CartoDB.Positron)
