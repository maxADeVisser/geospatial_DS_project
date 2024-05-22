"""Create a dashboard with streamlit"""

import contextily as cx
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from utils.spatial_trajectory_analysis import inspect_start_cluster

st.set_page_config(page_title="Trajectories")

from utils.postprocessing import load_and_parse_gdf_from_file


def inspect_start_cluster(trajs_gdf: gpd.GeoDataFrame, a=0.5) -> plt.Figure:
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
    return fig


@st.cache_data
def load_dataframe():
    return load_and_parse_gdf_from_file("out/clustered/clustered_trajectories.shp")


gdf = load_dataframe()

st.markdown("# Trajectories")
st.markdown("## Inputs")

# TODO include the ones in the end locations
all_locations = sorted(gdf["start_loc"].dropna().unique().tolist())

with st.form("my_form"):
    selected_start_loc = st.selectbox("Select start locations", options=all_locations)
    gdf = gdf.query(f"start_loc == '{selected_start_loc}'")
    end_locations = (
        gdf["end_loc"].dropna().unique().tolist()
    )  # TODO sort them by popularity

    selected_end_loc = st.selectbox(
        "Select end locations", options=end_locations
    )  # TODO only show the ones that have a match from the start location
    include_end_loc = st.checkbox("Include end locations")
    submitted = st.form_submit_button("Submit")


if submitted:
    st.markdown("## Clustered trajectories")
    if selected_start_loc and not include_end_loc:
        gdf = gdf.query(f"start_loc == '{selected_start_loc}'")

        fig = inspect_start_cluster(gdf, a=0.5)
        st.pyplot(fig)

    if selected_start_loc and include_end_loc:
        gdf = gdf.query(
            f"start_loc == '{selected_start_loc}' and end_loc == '{selected_end_loc}'"
        )

        if len(gdf) == 0:
            st.markdown("No trajectories found for the given locations")
            st.stop()

        fig = inspect_start_cluster(gdf, a=0.5)
        st.pyplot(fig)
