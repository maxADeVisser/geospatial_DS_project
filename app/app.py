"""Create a dashboard with streamlit"""

import streamlit as st

# dont move!:
st.set_page_config(page_title="AIS Trajectories", initial_sidebar_state="collapsed")
from app_utils import plot_destinations_from_start_location, plot_trajs_start_to_end

# from app.app_utils import inspect_start_cluster_app
from utils.postprocessing import load_and_parse_gdf_from_file


@st.cache_data(ttl="30 minutes", show_spinner="Loading data ...")
def load_dataframe(path: str):
    return load_and_parse_gdf_from_file(path)


gdf = load_dataframe(path="out/clustered/clustered_trajectories.shp").query(
    "cluster_st != -1.0 and cluster_en != -1.0"
)
# We are removing non-clustered points

st.markdown("# Density Based Clustering of AIS Trajectories")
st.info(
    "This dashboard allows you to inspect the trajectories of the Sailing boats in Danish waters. "
    "You can select a start location and optionally an end location to see the trajectories."
)
with st.expander("About the clustering method", expanded=False):
    st.markdown(
        """
        The trajectories are clustered both on the started and ending coordinates using the Density Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm. The parameters used for clustering are:
        - Epsilon: The maximum distance in meters between two data points for one to be considered as in the neighborhood of the other.
        - Minimum Samples: The number of samples in a neighborhood for a point to be considered as a core point.
        """
    )

st.markdown("# Inputs")
with st.expander("Explain Inputs", expanded=False):
    st.markdown(
        """
        - For the start location selection, the number in the parenthesis indicates the number of trajectories that start from that location.
        - When selecting the end location, the number in the parenthesis indicates the number of trajectories that start from the selected start location and end at the selected end location (i.e. the end locations are filtered based on the selected start location).
            - The `Include end locations` checkbox allows you to include the end locations in the clustering.If this is not checked, only the start locations will be considered."""
    )

# ------ INITIALISE STATE VARIABLES ------
selected_end_loc_var = "selected_end_loc"
if selected_end_loc_var not in st.session_state:
    st.session_state[selected_end_loc_var] = None

selected_start_loc_var = "selected_start_loc"
if selected_start_loc_var not in st.session_state:
    st.session_state[selected_start_loc_var] = None

traj_opacity_var = "trajectory_opacity"
if traj_opacity_var not in st.session_state:
    st.session_state[traj_opacity_var] = 0.5

submitted_var = "submitted"
if submitted_var not in st.session_state:
    st.session_state[submitted_var] = False

mark_start_var = "mark_start_location"
if mark_start_var not in st.session_state:
    st.session_state[mark_start_var] = False

# ------ SELECT START LOCATION ------
start_trajectory_counts = (
    gdf.groupby("start_loc")["traj_id"].nunique().sort_values(ascending=False)
)


if st.session_state[selected_start_loc_var] is None:
    st.session_state[selected_start_loc_var] = start_trajectory_counts.index[0]

_get_number_of_start_locations = lambda x: start_trajectory_counts[x]
sorted_start_locations = sorted(
    start_trajectory_counts.index.tolist(),
    key=_get_number_of_start_locations,
    reverse=True,
)
selected_start_loc = st.selectbox(
    "Select start locations",
    options=sorted_start_locations,
    format_func=lambda x: f"{x} ({_get_number_of_start_locations(x)})",
    key=selected_start_loc_var,
)

# Filter data on start location
if selected_start_loc:
    filtered_gdf = gdf.query(f"start_loc == '{selected_start_loc}'")

    # ------ SELECT END LOCATION ------
    end_trajectory_counts = (
        filtered_gdf.groupby("end_loc")["traj_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    if st.session_state[selected_end_loc_var] is None:
        st.session_state[selected_end_loc_var] = end_trajectory_counts.index[1]

    _get_number_of_end_locations = lambda x: end_trajectory_counts[x]
    sorted_end_locations = sorted(
        end_trajectory_counts.index.tolist(),
        key=_get_number_of_end_locations,
        reverse=True,
    )
    selected_end_loc = st.selectbox(
        "Select end locations (given start location)",
        options=sorted_end_locations,
        format_func=lambda x: f"{x} ({_get_number_of_end_locations(x)})",
        key=selected_end_loc_var,
    )
    include_end_loc = st.checkbox("Include end locations")

# ----- SUBMIT BUTTON -----
go_button = st.button("Cluster Trajectories!", help="Click to cluster trajectories")
if go_button:
    st.session_state[submitted_var] = True


# FOR  DEBUGGING
# for key, var in st.session_state.items():
#     st.write(key, var)

# ------ TRAJECTORY OPACITY SLIDER ------
st.slider(
    "Trajectory Opacity",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    key=traj_opacity_var,
)

mark_start = st.checkbox("Mark Start Location", value=True)

# ------ DISPLAY CLUSTERED TRAJECTORIES ------
if st.session_state[submitted_var]:
    st.markdown("# Clustered Trajectories")
    if not include_end_loc:
        fig = plot_destinations_from_start_location(
            filtered_gdf,
            traj_opacity=st.session_state[traj_opacity_var],
            mark_centroids=mark_start,
            title=f"Trajectories starting from {selected_start_loc}",
        )
        st.pyplot(fig)

    if selected_start_loc and include_end_loc:
        filtered_gdf = filtered_gdf.query(f"end_loc == '{selected_end_loc}'")

        if len(filtered_gdf) == 0:
            st.markdown("No trajectories found between the given locations")
            st.stop()

        fig = plot_trajs_start_to_end(
            filtered_gdf,
            traj_opacity=st.session_state[traj_opacity_var],
            title=f"Trajectories starting from {selected_start_loc} and ending at {selected_end_loc}",
            mark_centroids=mark_start,
        )
        st.pyplot(fig)

    if st.checkbox("Show Filtered Data"):
        df = filtered_gdf.drop(columns=["geometry"]).reset_index(drop=True)
        df["x_coord"] = filtered_gdf["geometry"].x.round(2)
        df["y_coord"] = filtered_gdf["geometry"].y.round(2)
        st.dataframe(df)

    if st.button("Save Plot"):
        if not include_end_loc:
            DOWNLOAD_PATH = f"out/saved_streamlit_plots/{selected_start_loc}.png"
        else:
            DOWNLOAD_PATH = (
                f"out/saved_streamlit_plots/{selected_start_loc}-{selected_end_loc}.png"
            )
        fig.savefig(DOWNLOAD_PATH, dpi=300, bbox_inches="tight")
        st.info(f"Plot saved to {DOWNLOAD_PATH}")
