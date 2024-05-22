"""Create a dashboard with streamlit"""

import folium
import geopandas as gpd
import streamlit as st
from shapely.geometry import Polygon
from streamlit_folium import folium_static

# Finding trajectories in an area of interest:
# area_of_interest = Polygon(
#     [
#         (11.89935, 57.69270),
#         (11.90161, 57.68902),
#         (11.90334, 57.68967),
#         (11.90104, 57.69354),
#         (11.89935, 57.69270),
#     ]
# )


def main():
    st.title("AIS Data Explore")

    # Display map with uploaded image
    st.subheader("Step 2: Select Polygon")
    m = folium.Map(location=[0, 0], zoom_start=2)
    folium_static(m)


if __name__ == "__main__":
    main()
