# Problem 2: Weather Prediction via Linear Regression & Random Forest Regression

## Running Instructions

```bash
source ~/venvs/BigData/bin/activate
cd /home/berserker/2_Studies/2/Big_Data/Assignment_2/bda-exercises-2-group-1
python solutions/problem_2/main.py --input "data/problem_2/NOAA-065900/065900*"
```

## Data

- Input path/pattern: data/problem_2/NOAA-065900/065900*
- Station id: 065900
- Files used: 61
- Years covered in parsed rows: 1949-2025
- Raw line count: 1191269
- Parsed row count: 1178233
- Dropped rows: 13036
- Training rows, 1949-2023: 1136215
- Validation rows, 2024: 25208
- Test rows, 2025: 16810

## Parsing and Preprocessing

- Source parser reference: solutions/problem_2/parseNOAA.scala
- Extracted fields: year, month, day, hour, latitude, longitude, elevation, wind_direction, wind_speed, ceiling_height, visibility_distance, dew_point_temperature, air_temperature.
- Scaling: latitude and longitude divided by 1000; wind_speed, dew_point_temperature, and air_temperature divided by 10.
- Missing-value policy: rows with missing air_temperature were dropped; missing feature values were median-imputed from the training split.
- Feature columns: year, month, day, hour, latitude, longitude, elevation, wind_direction, wind_speed, ceiling_height, visibility_distance, dew_point_temperature
- Label column: air_temperature

## Models

- LinearRegression: maxIter=100, regParam=0.001.
- RandomForestRegressor: numTrees=10, featureSubsetStrategy=auto, impurity=variance, seed=42.

## Results

- LinearRegression validation MSE, 2024: 11.187828
- RandomForestRegressor validation MSE, 2024: 8.687501
- LinearRegression test MSE, 2025: 20.728202
- RandomForestRegressor test MSE, 2025: 15.671728
- Best model by 2025 MSE: RandomForestRegressor

## Correlation Analysis

Pearson correlations with air_temperature:

- year: 0.112006
- month: 0.176302
- day: 0.012003
- hour: 0.110126
- latitude: 0.038897
- longitude: -0.008385
- elevation: -0.086348
- wind_direction: 0.032881
- wind_speed: -0.040566
- ceiling_height: 0.091776
- visibility_distance: 0.310516
- dew_point_temperature: 0.830170

Highest absolute non-undefined correlation: dew_point_temperature (0.830170)

## Spark Runtimes

- SparkSession creation: 2.873 s
- Raw line count: 1.260 s
- Parse and cache NOAA RDD: 1.055 s
- Create and cache DataFrame: 1.464 s
- Training split count: 0.275 s
- Validation split count: 0.159 s
- Test split count: 0.105 s
- LinearRegression fit: 2.799 s
- LinearRegression validation MSE: 0.168 s
- LinearRegression test MSE: 0.136 s
- RandomForestRegressor fit: 3.011 s
- RandomForestRegressor validation MSE: 0.222 s
- RandomForestRegressor test MSE: 0.149 s
- Correlation imputer fit: 0.827 s
- Correlation DataFrame cache: 0.169 s
- Correlation computation: 0.934 s
- Total runtime: 16.935 s
