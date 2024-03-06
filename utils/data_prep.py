import os
from datetime import timedelta

import geopandas as gpd
import movingpandas as mpd
import pandas as pd

from utils.project_types import ShipType, TimeFrequency


def load_csv_file(filepath: str) -> pd.DataFrame:
    """Loads a a full (no chunking) ais*.csv file from https://web.ais.dk/aisdata/,
    filters unused columns and returns a dataframe."""
    assert filepath.endswith(".csv"), "File must be a csv file."
    df =  pd.read_csv(
        filepath,
        parse_dates=["# Timestamp"],
        index_col="# Timestamp",
        dayfirst=True,
        usecols=["# Timestamp", "MMSI", "Latitude", "Longitude"],
    ).rename(columns={"Latitude": "lat", "Longitude": "lon"})

    assert df['lat'].isnull().sum() == 0, "Latitude column has missing values"
    assert df['lon'].isnull().sum() == 0, "Longitude column has missing values"
    return df


def change_data_frequency(
    ais_df: pd.DataFrame, data_freq: TimeFrequency
) -> pd.DataFrame:
    """Changes the data frequency of the dataframe.
    Resample every @data_freq and return the first value of each group."""
    return ais_df.resample(rule=data_freq.value).first() # resample based on index (timestamp)


def preprocessing(ais_df: pd.DataFrame) -> pd.DataFrame:
    # Convert to GeoPandas DataFrame
    geo_ais_df = gpd.GeoDataFrame(
        ais_df, geometry=gpd.points_from_xy(ais_df.lon, ais_df.lat)
    )
    return geo_ais_df


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
    return ais_df.loc[(ais_df['lon'] != 0)]


if __name__ == "__main__":
    # out_folder_path = "test_out/test_chunking"

    # Remove files:
    # folder_files = os.listdir(out_folder_path)
    # for f in folder_files:
    #     os.remove(os.path.join(out_folder_path, f))

    ### Test the reduce_data function
    # reduce_data("data_files/aisdk-2024-02-18.csv", out_folder_path)
    # d = pd.read_parquet("test_out/test_chunking/aisdk-2024-02-18_219011048.parquet")


    ### Other testing:
    today = load_csv_file("data_files/test_aug_1_sailboat.csv")
    today.memory_usage() # memory usage of the dataframe in bytes
    yesterday = load_csv_file("data_files/aisdk-2024-02-17.csv")
    grouped = today.groupby("MMSI")
    #groups = list(grouped.groups.keys())

    ### Stiching together data from multiple days
    MMSI = 538005405 # MMSI that has a ongoing trajectory
    today_vessel1 = grouped.get_group(MMSI)[1:] # remove the first row (outlier)
    yesterday_group = yesterday.groupby("MMSI")
    yesterday_vessel1 = yesterday_group.get_group(MMSI)

    # Concatenate the two dataframes
    concatenated = pd.concat([yesterday_vessel1, today_vessel1])
    changed_freq = change_data_frequency(concatenated, TimeFrequency.min_10)
    geo = preprocessing(changed_freq)
    geo.plot()

    # Creating a moving pandas trajectory and plotting it
    # to install do:
    # conda install hvplot
    # conda install -c pyviz geoviews-core
    traj = mpd.Trajectory(df=changed_freq, traj_id=str(MMSI), t=changed_freq.index, x="lon", y="lat")
    traj.df # the dataframe

    # Detect stops in the trajectory
    detector = mpd.TrajectoryStopDetector(traj)
    stops = detector.get_stop_points(min_duration=timedelta(seconds=600), max_diameter=50)

    import hvplot.pandas

    # plot stops
    stops.hvplot(geo=True, tiles=True, hover_cols="all")
