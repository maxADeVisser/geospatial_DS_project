# %%
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon

from utils.plotting import plot_static
from utils.preprocessing import to_geodf
from utils.project_types import MapProjection

# code is from: https://github.com/uber/h3-py-notebooks/blob/master/notebooks/unified_data_layers.ipynb


def plot_scatter(
    df,
    metric_col,
    x="lon",
    y="lat",
    marker=".",
    alpha=1,
    figsize=(16, 12),
    colormap="viridis",
):
    df.plot.scatter(
        x=x,
        y=y,
        c=metric_col,
        title=metric_col,
        edgecolors="none",
        colormap=colormap,
        marker=marker,
        alpha=alpha,
        figsize=figsize,
    )
    plt.xticks([], [])
    plt.yticks([], [])


def kring_smoothing(df, hex_col, metric_col, k):
    dfk = df[[hex_col]]
    dfk.index = dfk[hex_col]
    dfs = (
        dfk[hex_col]
        .apply(lambda x: pd.Series(list(h3.k_ring(x, k))))
        .stack()
        .to_frame("hexk")
        .reset_index(1, drop=True)
        .reset_index()
        .merge(df[[hex_col, metric_col]])
        .fillna(0)
        .groupby(["hexk"])[[metric_col]]
        .sum()
        .divide((1 + 3 * k * (k + 1)))
        .reset_index()
        .rename(index=str, columns={"hexk": hex_col})
    )
    dfs["lat"] = dfs[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    dfs["lon"] = dfs[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])
    return dfs


df = pd.read_parquet("out/ais_data/traj_first_10_days.parquet")
H3_RESOLUTION = 7
hex_col = "h3_hex" + str(H3_RESOLUTION)

# find hexs containing the points
df[hex_col] = df.apply(lambda x: h3.geo_to_h3(x.lat, x.lon, H3_RESOLUTION), axis=1)

# aggregate the points
# this does the same things as below
hex_counts = df[hex_col].value_counts().to_frame("counts").reset_index()
# hex_counts = (
#     df.groupby([hex_col])[hex_col]
#     .agg("counts")
#     .to_frame("count")
#     .reset_index()
# )

# filter out hexes with less than X AIS counts in it
min_readings = 100
# max_readings = 10000 # TODO add a max reading or do kring smoothing
hex_counts = hex_counts[hex_counts["counts"] > min_readings]

# find center of hex for visualisation
hex_counts["lon"] = hex_counts[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])
hex_counts["lat"] = hex_counts[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])


def add_geometry(row):
    points = h3.h3_to_geo_boundary(row[hex_col], True)
    return Polygon(points)


# Create the actual hexagon geometries:
hex_counts["geometry"] = hex_counts.apply(add_geometry, axis=1)

hex_counts = gpd.GeoDataFrame(hex_counts, geometry="geometry")

hex_counts.crs = MapProjection.WGS84.value  # original angular map projection
epsg_code = int(MapProjection.UTMzone32n.value.split(":")[-1])
hex_counts = hex_counts.to_crs(epsg=epsg_code)

# %%

# hex_counts.explore()

# plot the hexs
plot_scatter(hex_counts, metric_col="counts", marker="o", figsize=(17, 15))
plt.title("hex-grid: noise complaints")


# kring_smoothing
k = 2
df311s = kring_smoothing(hex_counts, hex_col, metric_col="counts", k=k)
print("sum sanity check:", df311s["counts"].sum() / hex_counts["counts"].sum())
plot_scatter(df311s, metric_col="counts", marker="o")

df311s.plot.scatter(
    x="lon",
    y="lat",
    c="counts",
    title="counts",
    edgecolors="none",
    colormap="viridis",
    marker="o",
    alpha=1,
    figsize=(16, 12),
)
plt.xticks([], [])
plt.yticks([], [])
plt.title("noise complaints: 2-ring average")
plt.show()
