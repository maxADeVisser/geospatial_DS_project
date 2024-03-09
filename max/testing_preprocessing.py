# %%
import contextily as cx
import matplotlib.pyplot as plt
from IPython.display import display

from utils.plotting import plot_AIS_trace, plot_static
from utils.preprocessing import (change_data_frequency, load_csv_file,
                                 remove_faulty_ais_readings, to_geodf)
from utils.project_types import TimeFrequency

# %% Load the data
# check in test/test_data/_test_data.py for how to generate the file used here
aug1 = load_csv_file("data_files/test_aug_1_sailboat.csv")
aug1 = remove_faulty_ais_readings(aug1)
aug1

# %%
grouped = aug1.groupby("MMSI")
groups = list(grouped.groups.keys())

# Investigate the number of entries for each MMSI and visualise trajectory
# for i in range(len(groups)):
#     minimum_entries = 3000
#     if len(grouped.get_group(groups[i])) >= minimum_entries:
#         selected_MMSI = groups[i]
#         display(grouped.get_group(selected_MMSI))
#         geo = to_geodf(grouped.get_group(selected_MMSI))
#         geo.plot()
#         break
#     else:
#         continue

# ?  MMSI on a ongoing trip: 211215180
selected_MMSI = 211215180
MMSI = aug1[aug1['MMSI'] == selected_MMSI]
MMSI = to_geodf(MMSI)

# change time frequency of data and plot
plot_AIS_trace(change_data_frequency(MMSI, TimeFrequency.min_15))


plot_static(MMSI)

# %%
