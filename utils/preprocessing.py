import datetime as dt
import os
import zipfile

import geopandas as gpd
import movingpandas as mpd
import pandas as pd

from utils.project_types import MapProjection, ShipType, TimeFrequency


def download_ais_data(date: dt.date, out_folder: str) -> str:
    """Downloads the AIS data for a given date from https://web.ais.dk/aisdata/ and saves it to @out_folder"""
    # Check if the folder exists, if not, create it
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Download the file
        
    url = rf"http://web.ais.dk/aisdata/aisdk-{date}.zip"
    print(f"Downloading {url} to {out_folder}")
    file_name = f"aisdk-{date}.zip"
    os.system(f"wget {url} -O {os.path.join(out_folder, file_name)}")
    return os.path.join(out_folder, file_name)


def unzip_ais_data(zip_file_path: str, out_folder: str) -> list:
    """Unzips the AIS data file to @out_folder and returns the list of extracted file names"""
    # Create the output folder if it doesn't exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(out_folder)

    # Get the list of extracted file names
    extracted_files = zip_ref.namelist()

    return extracted_files


def load_csv_file(filepath: str) -> pd.DataFrame:
    """Loads a a full (no chunking) ais*.csv file from https://web.ais.dk/aisdata/,
    filters unused columns and returns a dataframe."""
    assert filepath.endswith(".csv"), "File must be a csv file."
    df = pd.read_csv(
        filepath,
        parse_dates=["# Timestamp"],
        index_col="# Timestamp",
        dayfirst=True,
        usecols=["# Timestamp", "MMSI", "Ship type", "Latitude", "Longitude"],
    ).rename(columns={"Latitude": "lat", "Longitude": "lon"})
    print(f"raw df uses {df.memory_usage().sum() / 1_000_000} Mb")

    assert df["lat"].isnull().sum() == 0, "Latitude column has missing values"
    assert df["lon"].isnull().sum() == 0, "Longitude column has missing values"
    return df


def change_data_frequency(
    ais_df: gpd.GeoDataFrame, data_freq: TimeFrequency
) -> gpd.GeoDataFrame:
    """Changes the data frequency of the dataframe.
    Resample every @data_freq and return the first value of each group."""
    crs = int(str(ais_df.crs).split(":")[-1])
    ais_df = ais_df.resample(
        rule=data_freq.value
    ).first()
    ais_df.crs = crs
    return ais_df  # resample based on first index (timestamp)


def to_geodf(ais_df: pd.DataFrame, projection: MapProjection = MapProjection.UTMzone32n) -> pd.DataFrame:
    """Turns a pd dataframe into geo pandas and project the coordinates
    into @projection (and drops non-used columns as of now)"""
    geo_ais_df = gpd.GeoDataFrame(
        ais_df, geometry=gpd.points_from_xy(ais_df.lon, ais_df.lat)
    )
    geo_ais_df.crs = MapProjection.WGS84.value # original angular map projection
    epsg_code = int(projection.value.split(":")[-1])
    return geo_ais_df.to_crs(epsg=epsg_code).drop(columns=["lat", "lon", "Ship type"])


def change_crs(geo_df: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Changes the coordinate reference system (CRS) of the GeoDataFrame."""
    return geo_df.to_crs(crs)


def check_time_gap_threshold(
    ais_trace: list[int], time_gap_threshold: dt.timedelta
) -> dt.timedelta | None:
    """Check that the AIS has no time gaps strictly larger than @time_gap_threshold"""
    assert all(
        ais_trace[i].datetime <= ais_trace[i + 1].datetime
        for i in range(len(ais_trace) - 1)
    ), "AIS trace is not in chronological order"

    for i in range(0, len(ais_trace)):
        if i == len(ais_trace) - 1:
            return None
        curr_time = ais_trace[i].datetime
        next_time = ais_trace[i + 1].datetime
        time_diff = next_time - curr_time

        if time_diff.total_seconds() > time_gap_threshold.total_seconds():
            return time_diff
    return None


def reduce_data(csv_file_path: str, out_folder_path: str):
    """NOT WORKING PROPERLY YET... data is very messy (surprise, surprise)"""
    ais_df = load_csv_file(csv_file_path)

    # Extract csv file name
    csv_file_name = csv_file_path.split("/")[-1].split(".")[0]

    # Group by MMSI
    grouped = ais_df.groupby("MMSI")

    c = 0
    for MMSI, group in grouped:
        # The data has "time gaps", so when resampling, there will be missing values. We simply remove these:
        updated_frequency = change_data_frequency(group, TimeFrequency.min_10).dropna()

        updated_frequency.drop(columns=["MMSI"]).to_parquet(
            f"{out_folder_path}/{csv_file_name}_{MMSI}.parquet"
        )
        c += 1
    print(f"Number of files created: {c}")


def remove_faulty_ais_readings(ais_df: pd.DataFrame) -> pd.DataFrame:
    return ais_df.loc[(ais_df["lon"] != 0.0)]


if __name__ == "__main__":
    today = load_csv_file("data_files/aisdk-2023-08-01.csv")
    today.memory_usage()  # memory usage of the dataframe in bytes
    print(f"df uses {aug1.memory_usage().sum() / 1_000_000} Mb")
    yesterday = load_csv_file("data_files/aisdk-2024-02-17.csv")
    grouped = today.groupby("MMSI")
    # groups = list(grouped.groups.keys())

    ### Stiching together data from multiple days
    MMSI = 538005405  # MMSI that has a ongoing trajectory
    today_vessel1 = grouped.get_group(MMSI)[1:]  # remove the first row (outlier)
    yesterday_group = yesterday.groupby("MMSI")
    yesterday_vessel1 = yesterday_group.get_group(MMSI)

    # Concatenate the two dataframes
    concatenated = pd.concat([yesterday_vessel1, today_vessel1])
    changed_freq = change_data_frequency(concatenated, TimeFrequency.min_10)
    geo = to_geodf(changed_freq)
    geo.plot()

    # Creating a moving pandas trajectory and plotting it
    # to install do:
    # conda install hvplot
    # conda install -c pyviz geoviews-core
    traj = mpd.Trajectory(
        df=changed_freq, traj_id=str(MMSI), t=changed_freq.index, x="lon", y="lat"
    )
    traj.df  # the dataframe

    # Detect stops in the trajectory
    detector = mpd.TrajectoryStopDetector(traj)
    stops = detector.get_stop_points(
        min_duration=dt.timedelta(seconds=600), max_diameter=50
    )

    import hvplot.pandas

    # plot stops
    stops.hvplot(geo=True, tiles=True, hover_cols="all")
