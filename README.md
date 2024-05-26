# Geospatial Data Science Exam project
![Svendborg-Nyborg](https://github.com/maxADeVisser/geospatial_DS_project/assets/1355181/b67b8d77-01b2-4544-9e6e-38a6195e13fb)
![Svendborg](https://github.com/maxADeVisser/geospatial_DS_project/assets/1355181/71a9121e-36d1-449b-a61f-c8f0a9b058e0)

## Data
The data can be found [here](https://web.ais.dk/aisdata/)

## Instruction to setup appication

- The application requires the processed data file to be on your local computer in order to run. So to run the application, do the following steps in the described order:

- make sure to have a `out` folder in the project root directory where the scripts can store the data files.

1. The acquisition and processing pipeline needs to run (can take up to 3 hours as it needs to download and process the data.)
   - this is done be running the main.py file first
2. The clustering pipeline needs to run.
   - this is done by running the main_analysis.py file with the input being the output file of the first step.

- Once the two steps above are competed, you can run the application from the root directory of the project using `streamlit run app/app.py`
