from utils._test_data import TestSailingMMSIs
from utils.data_prep import (change_data_frequency, load_csv_file,
                             preprocessing, reduce_data,
                             remove_faulty_ais_readings)
from utils.project_types import ShipType, TimeFrequency

# %% Load the data
testMMSIs = TestSailingMMSIs.sailing_boats
aug1 = load_csv_file("data_files/test_aug_1_sailboat.csv")[1:]
aug1 = remove_faulty_ais_readings(aug1)
aug1
# %%

updated_freq = change_data_frequency(aug1, TimeFrequency.min_15)
grouped = updated_freq.groupby("MMSI")
selected_MMSI = grouped.get_group(testMMSIs[0])

# TODO do outlier removal

#groups = list(grouped.groups.keys())
