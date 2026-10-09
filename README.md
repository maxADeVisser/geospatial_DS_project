# Geospatial Data Science Exam project

Identifying popular sailing routes in Danish waters by clustering the start and end points of sailboat trajectories derived from open AIS (Automatic Identification System) vessel data. The results can be explored in an interactive Streamlit app.

This was the exam project for the Geospatial Data Science course at the IT University of Copenhagen

**Overview**
Raw AIS data is a stream of timestamped position reports from ships. On its own it says little about where vessels go. This project turns it into trajectories and then into routes:

1. Acquire daily AIS files from the Danish Maritime Authority and filter them down to sailboats inside Danish waters.
2. Build trajectories per vessel (identified by MMSI), split them into separate journeys, and clean out stops and faulty readings.
3. Cluster the start and end points of the journeys with DBSCAN to find the places sailors depart from and arrive at.
4. Explore the resulting routes between clusters in a Streamlit app.


## Data
The data can be found [here](https://web.ais.dk/aisdata/)

## Method
**Data acquisition and preprocessing (main.py)**
- Downloads and unzips one day of AIS data at a time, loaded lazily with [Dask](https://docs.dask.org/en/stable/dataframe.html) because the raw daily files are large.
- Keeps only sailboats and removes readings outside Danish waters (a bounding box on latitude/longitude), which filters out faulty positions.
- Resamples each vessel's track to one position every 15 minutes and appends the result to a single Parquet file.

**Trajectory construction (main.py)**

Built with [MovingPandas](https://movingpandas.org/) and GeoPandas:
- Vessels with fewer than 50 readings (about 12.5 hours of sailing) are dropped.
- Trajectories shorter than 10 km are discarded.
- A trajectory is split wherever there is a gap of more than 10 hours between readings.
- Stop detection splits trajectories where a vessel stays within a 1 km diameter for at least 3 hours, so that harbour stays separate one journey from the next.
- Speed is computed for every reading, and readings above 100 km/h are removed as sensor errors.

**Clustering (main_analysis.py)**
- The start and end point of every trajectory are clustered with DBSCAN.
- Hyperparameters were chosen with elbow plots and a grid search over eps (200m to 5km) and min_samples (5 to 20).
- For August 2023 the best setting was eps = 1500 m and min_samples = 14, which labels about 12.7 % of the points as outliers.
- Clusters are given place names, and each trajectory is labelled with its start and end cluster.

**Application (app/)**
- A Streamlit app for exploring routes between clusters, for example Svendborg to Nyborg or Svendborg to Marstal.


## Instruction to setup appication
- The application requires the processed data file to be on your local computer in order to run. So to run the application, do the following steps in the described order:
- make sure to have a `out` folder in the project root directory where the scripts can store the data files.

1. The acquisition and processing pipeline needs to run (can take up to 3 hours as it needs to download and process the data.)
   - this is done be running the `main.py` file first
2. The clustering pipeline needs to run.
   - this is done by running the `main_analysis.py` file with the input being the output file of the first step.

- Once the two steps above are competed, you can run the application from the root directory of the project using `streamlit run app/app.py`

Here are to examples of the types of plots that the application generates:
![Svendborg-Nyborg](https://github.com/maxADeVisser/geospatial_DS_project/assets/1355181/b67b8d77-01b2-4544-9e6e-38a6195e13fb)
![Svendborg](https://github.com/maxADeVisser/geospatial_DS_project/assets/1355181/71a9121e-36d1-449b-a61f-c8f0a9b058e0)
![Svendborg-Marstal](https://github.com/maxADeVisser/geospatial_DS_project/assets/1355181/b9efd06e-ad01-4a15-93f7-88e04c6d7eb2)
![Vordingborg Kommune](https://github.com/maxADeVisser/geospatial_DS_project/assets/1355181/a9f10278-2fbe-4a4c-a76c-62b0225f0ef7)
