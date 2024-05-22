"""Create a dashboard with streamlit"""

import contextily as cx
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from utils.spatial_trajectory_analysis import inspect_start_cluster

st.set_page_config(page_title="Trajectories", initial_sidebar_state="collapsed")

from utils.postprocessing import load_and_parse_gdf_from_file


def inspect_start_cluster(
    trajs_gdf: gpd.GeoDataFrame, a=0.5, mark_centroid: bool = False
) -> plt.Figure:
    fig, ax = plt.subplots(1, figsize=(12, 12))
    trajs = mpd.TrajectoryCollection(
        trajs_gdf.set_index("timestamp"), traj_id_col="traj_id", t="timestamp"
    )
    trajs.plot(ax=ax, alpha=a, color="blue", linewidth=1)
    cx.add_basemap(
        ax,
        crs="EPSG:25832",
        source=cx.providers.CartoDB.Positron,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if mark_centroid:
        coords = np.array([[entry.x, entry.y] for entry in trajs_gdf["geometry"]])
        center = np.mean(coords, axis=0)
        ax.scatter(center[0], center[1], marker="x", color="red", s=100)
    return fig


@st.cache_data
def load_dataframe():
    return load_and_parse_gdf_from_file("out/clustered/clustered_trajectories.shp")


gdf = load_dataframe()

st.markdown("# Trajectories")
st.info(
    "This dashboard allows you to inspect the trajectories of the vehicles. "
    "You can select a start location and optionally an end location to see the trajectories."
)

st.markdown("## Inputs")
# select start location
all_start_locations = sorted(gdf["start_loc"].dropna().unique().tolist())
_get_number_of_start_locations = lambda x: len(gdf.query(f"start_loc == '{x}'"))
sorted_start_locations = sorted(
    all_start_locations,
    key=_get_number_of_start_locations,
    reverse=True,
)
selected_start_loc = st.selectbox(
    "Select start locations",
    options=sorted_start_locations,
    format_func=lambda x: f"{x} ({_get_number_of_start_locations(x)})",
)

# Filter on start location
if selected_start_loc:
    start_filtered_gdf = gdf.query(f"start_loc == '{selected_start_loc}'")

# get end locations for the selected start location
_get_number_of_end_locations = lambda x: len(
    start_filtered_gdf.query(f"end_loc == '{x}'")
)
matching_end_locations = (
    start_filtered_gdf["end_loc"].dropna().unique().tolist()
)  # end locations with matching locations for the selected start location
sorted_end_locations = sorted(
    matching_end_locations, key=_get_number_of_end_locations, reverse=True
)
# TODO check this code again. When showing also the end locations, it does not seems to work
selected_end_loc = st.selectbox(
    "Select end locations",
    options=sorted_end_locations,
    format_func=lambda x: f"{x} ({_get_number_of_end_locations(x)})",
)
include_end_loc = st.checkbox("Include end locations")
submitted = st.button("Cluster Trajectories!", help="Click to cluster trajectories")


if submitted:
    st.markdown("## Clustered trajectories")
    if not include_end_loc:
        fig = inspect_start_cluster(start_filtered_gdf, a=0.5, mark_centroid=True)
        st.pyplot(fig)

    if selected_start_loc and include_end_loc:
        start_filtered_gdf = start_filtered_gdf.query(
            f"end_loc == '{selected_end_loc}'"
        )

        if len(start_filtered_gdf) == 0:
            st.markdown("No trajectories found for the given locations")
            st.stop()

        fig = inspect_start_cluster(start_filtered_gdf, a=0.5)
        st.pyplot(fig)
