# Problem 1:

- Models used: Gemini Pro and Claude Sonnet 4.6, and OpenCode with the BigPickle model.

## Main prompting:

To Gemini Pro:

```
According to the instructions and solution attached to Problem 1 of my homework, cross-check the solution and highlight any mistakes. If you find any, please explain them and provide the correct solution.
```

Output:

```
Your PySpark implementation successfully performs the necessary data splitting, pipeline construction and model tuning, but deliberately and correctly deviates from the technically flawed API instructions in the assignment. The prompt requests the use of BinaryClassificationMetrics and BinaryClassificationEvaluator to extract scalar accuracy, precision, and recall, which is impossible within PySpark's framework. Your workaround using the Multiclass equivalents is the mathematically sound solution. To avoid losing marks due to a strict marking system, keep your current code but explicitly document these PySpark API limitations in your 'Problem 1.txt' submission to justify your programming choices.
```

To Claude Sonnet 4.6:

```
According to the instructions and solution attached to Problem 1 of my homework, cross-check the solution and highlight any mistakes. If you find any, please explain them and provide the correct solution.
```

Output:

```
Step (a) ✅ Fully Compliant

Schema with all 18 fields, explicit StructType, HeartDisease as target, CSV loaded correctly, null filtering applied. Nothing to fix.
Step (b) ✅ Now Fully Compliant

The justification comment above cv_evaluator is in place and correctly explains the ml vs mllib API constraint. The final test evaluation reports precision, recall, and accuracy per label via MulticlassMetrics, plus areaUnderROC and areaUnderPR via BinaryClassificationMetrics. All 6 sub-requirements satisfied.
Step (c) ✅ Now Fully Compliant

The equivalent justification comment is present above the TrainValidationSplit definition. trainRatio=0.8 is set, same hyperparameter grid is used, model is saved, test metrics are reported, and the CV vs TVS comparison table is printed. All 7 sub-requirements satisfied.
Step (d) ✅ Now Fully Compliant (with one minor code bug to verify)

The DecisionTree baseline is replaced with GBTClassifier wrapped in OneVsRest for multi-class support. Both GBTClassifier and OneVsRest are correctly imported. The RandomForest CV uses the full grid including numTrees, the best model is saved, MulticlassMetrics is used throughout, and the GBT vs RF comparison table is printed at the end.

One thing to verify at runtime: This line uses a property without calling it as a method:

python
best_num_trees = best_rf_stage.getNumTrees  # ← missing ()

getNumTrees is a method, not a property, in PySpark's RandomForestClassificationModel. It should be:

python
best_num_trees = best_rf_stage.getNumTrees()

This will silently print the function object reference instead of the integer value in all the comparison print statements.
Updated Summary
Step	Status	Remaining Issue
(a)	✅	None
(b)	✅	None
(c)	✅	None
(d)	⚠️	getNumTrees missing () — fix before running

One-line fix: best_num_trees = best_rf_stage.getNumTrees().
Prepared using Claude Sonnet 4.6
```

To OpenCode with the BigPickle model:

```
Given the current content of Problem 1.md and the available solutions and run results, expand the report with any additional insights, explanations or conclusions that can be drawn from the data and the model's performance. Ensure the report is comprehensive, providing a clear understanding of the results obtained from the problem-solving process.

Problem 1: Heart Disease Classification with PySpark

Running Instructions
Set up local Spark and Python paths:

export SPARK_HOME=/home/isaac/Apps/spark-4.0.2-bin-hadoop3
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-src.zip:$PYTHONPATH

Run the program:

python solutions/problem_1/main.py

General Strategy
- Load and preprocess 2020 heart disease cleaned CSV (filter invalid HeartDisease values)
- Part B: 80/20 train/test split, tune DecisionTree (impurity, maxDepth, maxBins) via 5-fold CrossValidator for binary HeartDisease target
- Part C: Compare CrossValidator with TrainValidationSplit (trainRatio=0.8) for the same DecisionTree task
- Part D: Switch to multiclass AgeCategory target, train DecisionTree and RandomForest (tune numTrees via TrainValidationSplit), evaluate with MulticlassMetrics

Results
Part B: DecisionTree + CrossValidator
- Best hyperparams: impurity=gini, maxDepth=6, maxBins=100
- Test AUC: 0.5953, Accuracy: 0.9146
  - HeartDisease=Yes: Precision=0.5812, Recall=0.0524
  - HeartDisease=No: Precision=0.9172, Recall=0.9964

Part C: TrainValidationSplit Comparison
- Identical best hyperparams and test metrics to Part B
- Runtime reduced from 389.6s (CrossValidator) to 83.0s

Part D: AgeCategory (Multiclass) Target
- DecisionTree: Accuracy=0.1573, Weighted Precision=0.1342
- RandomForest (best numTrees=20): Accuracy=0.1648, Weighted Precision=0.1230
- RandomForest slightly outperforms DecisionTree; both have low accuracy (AgeCategory is poorly predicted by other features)

Spark Runtimes
Step | Duration (s)
SparkSession creation | 3.424
Load CSV | 0.753
Preprocessing filters | 0.103
Train/Test randomSplit (lazy) | 0.020
Count train/test records | 3.264
CrossValidator fit (5-fold) | 389.592
Training prediction + eval (AUC/PR) | 2.013
Save best DecisionTree model | 0.692
Test prediction + eval (AUC/PR) | 1.389
Test MulticlassMetrics | 2.034
Test BinaryClassificationMetrics | 0.289
TrainValidationSplit fit | 83.043
TVS training prediction + eval | 1.734
Save TVS best model | 0.181
TVS test prediction + eval | 1.282
TVS MulticlassMetrics | 1.130
TVS BinaryClassificationMetrics | 0.260
AgeCategory randomSplit (lazy) | 0.003
Count AgeCategory train/test records | 0.880
DecisionTree (AgeCategory) fit | 10.297
DecisionTree (AgeCategory) transform | 0.130
DecisionTree (AgeCategory) MulticlassMetrics | 1.122
RandomForest TrainValidationSplit fit | 26.549
RandomForest transform | 0.068
RandomForest MulticlassMetrics | 1.015
```

Output: The current version of the report.

# Problem 2:

- Model used: ChatGPT-5.5.

## Main prompting:

ChatGPT conversation excerpts:

Me:

```
For Problem 2 in Exercises2.pdf, what exactly do I need to implement and submit? Ignore Problems 1 and 3.
```

ChatGPT:

```
Problem 2 asks for the NOAA Luxembourg station files starting with 065900. The required parts are: parse the listed NOAA fields, cache the parsed data, train LinearRegression on 1949-2023, validate on 2024, train RandomForestRegressor with n = 10 regression trees and automatic training mode, validate on 2024, test both models on 2025, report mean-squared error, report which attribute has the highest correlation with air_temperature, and put run instructions, results, and runtimes in Problem_2.md.
```

Used in our solution: `solutions/problem_2/main.py`, lines 182-239, writes the `Problem_2.md` report.

Me:

```
The target column is air_temperature. Is this a regression or classification task? Should I report MSE, accuracy, AUC, or something else?
```

ChatGPT:

```
This is regression. The target is continuous temperature, so accuracy and AUC do not fit unless you create a separate classification problem. Problem 2 directly asks for mean-squared error.
```

Used in our solution: `solutions/problem_2/main.py`, lines 282-284.

Code copied from our solution:

```
mse_evaluator = RegressionEvaluator(
    labelCol="air_temperature", predictionCol="prediction", metricName="mse"
)
```

Me:

```
Look at solutions/problem_2/parseNOAA.scala. Which substring positions, scaling factors, and missing-value codes do I need for the NOAA fields listed in Problem 2?
```

ChatGPT:

```
Use the parser offsets from parseNOAA.scala and apply the NOAA scaling factors. Temperature and wind speed values need division by 10. Latitude and longitude need division by 1000. Missing values such as +9999, 9999, 99999, and 999999 should become null or be filtered when the label is missing.
```

Used in our solution: `solutions/problem_2/main.py`, lines 74-135. The assignment text says: "Apply the provided parsing function parseNOAA.scala" and lists the NOAA attributes to extract.

Code copied from our solution:

```
air_temperature = parse_optional_scaled(
    line[87:92], 10.0, {"+9999", "-9999", "99999"}
)

latitude = parse_optional_scaled(
    line[28:34], 1000.0, {"+99999", "-99999", "999999"}
)

longitude = parse_optional_scaled(
    line[34:41], 1000.0, {"+999999", "-999999", "9999999"}
)

wind_speed = parse_optional_scaled(line[65:69], 10.0, {"9999"})
```

Me:

```
How do I parse the NOAA date into year, month, and day, and the NOAA time into hour only?
```

ChatGPT:

```
Take the date/time substrings and convert them to integers: year = line[15:19], month = line[19:21], day = line[21:23], hour = line[23:25].
```

Used in our solution: `solutions/problem_2/main.py`, lines 96-102. The assignment text says: "date (split into three fields: year, month and day)" and "time (extract only the hour without the minutes)."

Code copied from our solution:

```
year = parse_required_int(line[15:19])
month = parse_required_int(line[19:21])
day = parse_required_int(line[21:23])
hour = parse_required_int(line[23:25])
```

Me:

```
Give me a PySpark schema for these parsed NOAA fields: year, month, day, hour, latitude, longitude, elevation, wind_direction, wind_speed, ceiling_height, visibility_distance, dew_point_temperature, and air_temperature.
```

ChatGPT:

```
Use IntegerType for year/month/day/hour and DoubleType for measured weather fields. Keep air_temperature as the label.
```

Used in our solution: `solutions/problem_2/main.py`, lines 22-38.

Code copied from our solution:

```
SCHEMA = T.StructType(
    [
        T.StructField("year", T.IntegerType(), False),
        T.StructField("month", T.IntegerType(), False),
        T.StructField("day", T.IntegerType(), False),
        T.StructField("hour", T.IntegerType(), False),
        T.StructField("latitude", T.DoubleType(), True),
        T.StructField("longitude", T.DoubleType(), True),
        T.StructField("elevation", T.DoubleType(), True),
        T.StructField("wind_direction", T.DoubleType(), True),
        T.StructField("wind_speed", T.DoubleType(), True),
        T.StructField("ceiling_height", T.DoubleType(), True),
        T.StructField("visibility_distance", T.DoubleType(), True),
        T.StructField("dew_point_temperature", T.DoubleType(), True),
        T.StructField("air_temperature", T.DoubleType(), False),
    ]
)
```

Me:

```
How do I split a PySpark DataFrame by year so that 1949-2023 is training data, 2024 is validation data, and 2025 is test data?
```

ChatGPT:

```
Filter by year instead of using a random split.
```

Used in our solution: `solutions/problem_2/main.py`, lines 269-271. The assignment text says: "training set with all records from 1949 until 2023", "validation set with all measurements for 2024", and "whole year of 2025 (so this is your test set)."

Code copied from our solution:

```
train_df = weather.filter((col("year") >= 1949) & (col("year") <= 2023)).cache()
val_df = weather.filter(col("year") == 2024).cache()
test_df = weather.filter(col("year") == 2025).cache()
```

Me:

```
Show me a minimal PySpark ML pipeline for predicting air_temperature with LinearRegression.
```

ChatGPT:

```
Use an imputer, a vector assembler, and LinearRegression with air_temperature as labelCol.
```

Used in our solution: `solutions/problem_2/main.py`, lines 286-295. The assignment text says: "Train a LinearRegression model."

Code copied from our solution:

```
lr_imputer, lr_assembler = make_preprocessing_stages()
lr = LinearRegression(
    labelCol="air_temperature",
    featuresCol="features",
    predictionCol="prediction",
    maxIter=100,
    regParam=0.001,
)
lr_pipeline = Pipeline(stages=[lr_imputer, lr_assembler, lr])
```

Me:

```
Show me a minimal PySpark ML pipeline for RandomForestRegressor with numTrees=10 and featureSubsetStrategy="auto".
```

ChatGPT:

```
Use RandomForestRegressor with labelCol="air_temperature", featuresCol="features", numTrees=10, and featureSubsetStrategy="auto".
```

Used in our solution: `solutions/problem_2/main.py`, lines 307-319. The assignment text says: "Train a RandomForest regressor based again on n = 10 regression trees and automatic training mode."

Code copied from our solution:

```
rf = RandomForestRegressor(
    labelCol="air_temperature",
    featuresCol="features",
    predictionCol="prediction",
    numTrees=10,
    featureSubsetStrategy="auto",
    impurity="variance",
    seed=42,
)
```

Me:

```
How do I handle missing feature values in Spark ML without using information from the 2024 validation set or the 2025 test set?
```

ChatGPT:

```
Put the Imputer inside the pipeline and fit the pipeline only on train_df. Then the imputer statistics are learned from training data only.
```

Used in our solution: `solutions/problem_2/main.py`, lines 147-153, 294-295, and 317-319.

Code copied from our solution:

```
def make_preprocessing_stages() -> tuple[Imputer, VectorAssembler]:
    imputer = Imputer(
        inputCols=NULLABLE_FEATURE_COLS,
        outputCols=IMPUTED_COLS,
    ).setStrategy("median")
    assembler = VectorAssembler(inputCols=ASSEMBLER_INPUT_COLS, outputCol="features")
    return imputer, assembler
```

Me:

```
How do I calculate mean-squared error in PySpark for predictions against air_temperature?
```

ChatGPT:

```
Use RegressionEvaluator with metricName="mse".
```

Used in our solution: `solutions/problem_2/main.py`, lines 163-165, 296-305, and 321-330. The assignment text says: "Calculate the mean-squared error between the predicted temperatures and the actual measurements for 2025."

Code copied from our solution:

```
def evaluate_mse(model, data, evaluator: RegressionEvaluator) -> float:
    predictions = model.transform(data)
    return evaluator.evaluate(predictions)
```

Me:

```
How do I calculate Pearson correlation in Spark between each input attribute and air_temperature?
```

ChatGPT:

```
Use DataFrame.stat.corr(feature_column, "air_temperature") for each feature column.
```

Used in our solution: `solutions/problem_2/main.py`, lines 341-351. The assignment text says: "Which of the above attributes do you think has the highest correlation with our target attribute, i.e., the air temperature?"

Code copied from our solution:

```
for original_column, model_column in pairs:
    output.append(
        (original_column, corr_df.stat.corr(model_column, "air_temperature"))
    )
```

Me:

```
Give me the WSL virtualenv installation commands for these missing modules: ModuleNotFoundError: No module named 'numpy' and ModuleNotFoundError: No module named 'pyspark'.
```

ChatGPT:

```
source ~/venvs/BigData/bin/activate
python -m pip install numpy pyspark
```

Used for running `solutions/problem_2/main.py` from the WSL virtual environment.

Me:

```
How do I measure and print runtime sections in a PySpark script?
```

ChatGPT:

```
Wrap each expensive section with time.perf_counter() and store the duration.
```

Used in our solution: `solutions/problem_2/main.py`, lines 138-144 and 168-169. The assignment text says to report "the most important results (such as number of results, runtimes, etc.)."

Code copied from our solution:

```
def timed(timings: list[tuple[str, float]], name: str, fn):
    start = time.perf_counter()
    result = fn()
    duration = time.perf_counter() - start
    timings.append((name, duration))
    print(f"({name}) Duration: {duration:.3f} seconds")
    return result
```

Me:

```
How should I structure Problem_2.md so it includes run commands, model parameters, MSE values, correlation values, and runtimes? AI usage is documented separately in AI Usage Statement.md.
```

ChatGPT:

```
Use sections like: Running Instructions, Data, Parsing and Preprocessing, Models, Results, Correlation Analysis, and Spark Runtimes.
```

Used in our solution: `solutions/problem_2/main.py`, lines 182-239, and `Problem_2.md`.

Raw results copied from `Problem_2.md`:

```
Results
- LinearRegression validation MSE, 2024: 11.187828
- RandomForestRegressor validation MSE, 2024: 8.687501
- LinearRegression test MSE, 2025: 20.728202
- RandomForestRegressor test MSE, 2025: 15.671728
- Best model by 2025 MSE: RandomForestRegressor
```

# Problem 3:
