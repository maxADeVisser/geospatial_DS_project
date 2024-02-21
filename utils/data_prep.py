import os

import geopandas as gpd
import pandas as pd

from utils.project_types import TimeFrequency


def load_csv_file(filepath: str) -> pd.DataFrame:
    """Loads a csv file, filters unused columns and returns a dataframe."""
    return pd.read_csv(
        filepath,
        parse_dates=["# Timestamp"],
        index_col="# Timestamp",
        dayfirst=True,
        usecols=["# Timestamp", "MMSI", "Latitude", "Longitude"],
    ).rename(columns={"Latitude": "lat", "Longitude": "lon"})


def change_data_frequency(
    ais_df: pd.DataFrame, data_freq: TimeFrequency
) -> pd.DataFrame:
    """Changes the data frequency of the dataframe."""
    return ais_df.resample(rule=data_freq.value).mean()  # resample based on index


def preprocessing(ais_df: pd.DataFrame) -> pd.DataFrame:
    # Convert to GeoPandas DataFrame
    geo_ais_df = gpd.GeoDataFrame(
        ais_df, geometry=gpd.points_from_xy(ais_df.Longitude, ais_df.Latitude)
    )
    return geo_ais_df


def reduce_data(csv_file_path: str, out_folder_path: str):
    ais_df = load_csv_file(csv_file_path)

    csv_file_name = csv_file_path.split("/")[-1].split(".")[0]

    grouped = ais_df.groupby("MMSI")

    for MMSI, group in grouped:
        change_data_frequency(group, TimeFrequency.min_15).to_parquet(
            f"{out_folder_path}/{csv_file_name}_{MMSI}.parquet"
        )


if __name__ == "__main__":
    folder_path = "test_out/test_chunking"

    # Remove files:
    folder_files = os.listdir(folder_path)
    for f in folder_files:
        os.remove(os.path.join(folder_path, f))

    # df = load_csv_file("data_files/aisdk-2024-02-18.csv")
    # grouped = df.groupby("MMSI")
    # for i, g in grouped:
    #     print(i)
    #     print(change_data_frequency(g, TimeFrequency.min_15))
    #     break

    reduce_data("data_files/aisdk-2024-02-18.csv", folder_path)

    # pd.read_parquet("test_out/test_chunking/aisdk-2024-02-18_2190069.parquet")
