"""Create a dashboard with streamlit"""

import contextily as cx
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

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


@st.cache_data(ttl="30 minutes", show_spinner="Loading data ...")
def load_dataframe(path: str):
    return load_and_parse_gdf_from_file(path)


gdf = load_dataframe(path="out/clustered/clustered_trajectories.shp")

st.markdown("# Density Based Clustering of AIS Trajectories")
st.info(
    "This dashboard allows you to inspect the trajectories of the vehicles. "
    "You can select a start location and optionally an end location to see the trajectories."
)

st.markdown("## Inputs")
# select start location:
start_loc_counts = gdf["start_loc"].value_counts().sort_values(ascending=False)
_get_number_of_start_locations = lambda x: start_loc_counts[x]

sorted_start_locations = sorted(
    start_loc_counts.index.tolist(),
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

# get end locations for the selected start location:
end_loc_counts = (
    start_filtered_gdf["end_loc"].value_counts().sort_values(ascending=False)
)
_get_number_of_end_locations = lambda x: end_loc_counts[x]
sorted_end_locations = sorted(
    end_loc_counts.index.tolist(), key=_get_number_of_end_locations, reverse=True
)
selected_end_loc = st.selectbox(
    "Select end locations",
    options=sorted_end_locations,
    format_func=lambda x: f"{x} ({_get_number_of_end_locations(x)})",
)
include_end_loc = st.checkbox("Include end locations")
submitted = st.button("Cluster Trajectories!", help="Click to cluster trajectories")


# When the button is clicked, show the clustered trajectories
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
