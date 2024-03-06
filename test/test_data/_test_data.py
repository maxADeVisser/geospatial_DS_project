import pandas as pd


# 50 Sailing boats MMSI from the data
# df.loc[df['Ship type'] == "Sailing"]['MMSI'].unique()[:50]
class TestSailingMMSIs:
    sailing_boats = [
        219029112,
        220092000,
        219028635,
        219016692,
        219025250,
        219002846,
        219019558,
        219021291,
        219026459,
        220276000,
        219003038,
        219023898,
        219006722,
        219028144,
        219029917,
        219002312,
        219029210,
        219024777,
        211878330,
        265591210,
        219000933,
        219003597,
        219417000,
        219016668,
        265050540,
        261002353,
        211798910,
        219531000,
        219023617,
        219031651,
        219003726,
        219029370,
        219020436,
        219026821,
        219002663,
        276014240,
        219501000,
        265798780,
        219023018,
        219013667,
        219025371,
        219006564,
        265814190,
        219029447,
        219003232,
        219025633,
        235008205,
        219015332,
        219028878,
        219032365,
    ]


def _generate_raw_test_df(csv_path: str = "data_files/aisdk-2023-08-01.csv") -> None:
    """Filters for shiptype=Sailboat and writes a raw test dataframe.
    Default is loading the data file from the 1st of August 2023."""
    df = pd.read_csv(csv_path)
    df.loc[df["Ship type"] == "Sailing"] \
        .to_csv("data_files/test_aug_1_sailboat.csv", index=False)


if __name__ == "__main__":
    _generate_raw_test_df()
