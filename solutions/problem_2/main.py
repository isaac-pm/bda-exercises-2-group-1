from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import col


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = "data/problem_2/NOAA-065900/065900*"
REPORT_PATH = REPO_ROOT / "Problem_2.md"

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

BASE_FEATURE_COLS = ["year", "month", "day", "hour"]
NULLABLE_FEATURE_COLS = [
    "latitude",
    "longitude",
    "elevation",
    "wind_direction",
    "wind_speed",
    "ceiling_height",
    "visibility_distance",
    "dew_point_temperature",
]
FEATURE_COLS = BASE_FEATURE_COLS + NULLABLE_FEATURE_COLS
IMPUTED_COLS = [f"{column}_imputed" for column in NULLABLE_FEATURE_COLS]
ASSEMBLER_INPUT_COLS = BASE_FEATURE_COLS + IMPUTED_COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Problem 2 NOAA weather regression solution."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Glob path for NOAA Luxembourg station files beginning with 065900.",
    )
    parser.add_argument("--app-name", default="NOAAWeatherRegressionProblem2")
    parser.add_argument("--master", default=None)
    return parser.parse_args()


def parse_required_int(text: str) -> int:
    return int(text.strip())


def parse_optional_scaled(
    text: str, scale: float, missing_values: set[str]
) -> float | None:
    value = text.strip()
    if value in missing_values:
        return None
    try:
        return float(value) / scale
    except ValueError:
        return None


def parse_noaa_line(line: str) -> tuple | None:
    if len(line) < 98:
        return None

    air_temperature = parse_optional_scaled(
        line[87:92], 10.0, {"+9999", "-9999", "99999"}
    )
    if air_temperature is None:
        return None

    try:
        year = parse_required_int(line[15:19])
        month = parse_required_int(line[19:21])
        day = parse_required_int(line[21:23])
        hour = parse_required_int(line[23:25])
    except ValueError:
        return None

    latitude = parse_optional_scaled(
        line[28:34], 1000.0, {"+99999", "-99999", "999999"}
    )
    longitude = parse_optional_scaled(
        line[34:41], 1000.0, {"+999999", "-999999", "9999999"}
    )
    elevation = parse_optional_scaled(
        line[46:51], 1.0, {"+9999", "-9999", "99999"}
    )
    wind_direction = parse_optional_scaled(line[60:63], 1.0, {"999"})
    wind_speed = parse_optional_scaled(line[65:69], 10.0, {"9999"})
    ceiling_height = parse_optional_scaled(line[70:75], 1.0, {"99999"})
    visibility_distance = parse_optional_scaled(line[78:84], 1.0, {"999999"})
    dew_point_temperature = parse_optional_scaled(
        line[93:98], 10.0, {"+9999", "-9999", "99999"}
    )

    return (
        year,
        month,
        day,
        hour,
        latitude,
        longitude,
        elevation,
        wind_direction,
        wind_speed,
        ceiling_height,
        visibility_distance,
        dew_point_temperature,
        air_temperature,
    )


def timed(timings: list[tuple[str, float]], name: str, fn):
    start = time.perf_counter()
    result = fn()
    duration = time.perf_counter() - start
    timings.append((name, duration))
    print(f"({name}) Duration: {duration:.3f} seconds")
    return result


def make_preprocessing_stages() -> tuple[Imputer, VectorAssembler]:
    imputer = Imputer(
        inputCols=NULLABLE_FEATURE_COLS,
        outputCols=IMPUTED_COLS,
    ).setStrategy("median")
    assembler = VectorAssembler(inputCols=ASSEMBLER_INPUT_COLS, outputCol="features")
    return imputer, assembler


def create_spark(args: argparse.Namespace) -> SparkSession:
    builder = SparkSession.builder.appName(args.app_name)
    if args.master:
        builder = builder.master(args.master)
    return builder.getOrCreate()


def evaluate_mse(model, data, evaluator: RegressionEvaluator) -> float:
    predictions = model.transform(data)
    return evaluator.evaluate(predictions)


def format_timings(timings: list[tuple[str, float]]) -> str:
    return "\n".join(f"- {name}: {duration:.3f} s" for name, duration in timings)


def format_correlations(correlations: list[tuple[str, float | None]]) -> str:
    lines = []
    for name, value in correlations:
        if value is None or math.isnan(value):
            lines.append(f"- {name}: undefined")
        else:
            lines.append(f"- {name}: {value:.6f}")
    return "\n".join(lines)


def write_report(results: dict, timings: list[tuple[str, float]]) -> None:
    report = f"""# Problem 2: Weather Prediction via Linear Regression & Random Forest Regression

## Running Instructions

```bash
source ~/venvs/BigData/bin/activate
cd /home/berserker/2_Studies/2/Big_Data/Assignment_2/bda-exercises-2-group-1
python solutions/problem_2/main.py --input "{results['input_pattern']}"
```

## Data

- Input path/pattern: {results['input_pattern']}
- Station id: 065900
- Files used: {results['file_count']}
- Years covered in parsed rows: {results['min_year']}-{results['max_year']}
- Raw line count: {results['raw_count']}
- Parsed row count: {results['parsed_count']}
- Dropped rows: {results['dropped_count']}
- Training rows, 1949-2023: {results['train_count']}
- Validation rows, 2024: {results['val_count']}
- Test rows, 2025: {results['test_count']}

## Parsing and Preprocessing

- Source parser reference: solutions/problem_2/parseNOAA.scala
- Extracted fields: year, month, day, hour, latitude, longitude, elevation, wind_direction, wind_speed, ceiling_height, visibility_distance, dew_point_temperature, air_temperature.
- Scaling: latitude and longitude divided by 1000; wind_speed, dew_point_temperature, and air_temperature divided by 10.
- Missing-value policy: rows with missing air_temperature were dropped; missing feature values were median-imputed from the training split.
- Feature columns: {', '.join(FEATURE_COLS)}
- Label column: air_temperature

## Models

- LinearRegression: maxIter=100, regParam=0.001.
- RandomForestRegressor: numTrees=10, featureSubsetStrategy=auto, impurity=variance, seed=42.

## Results

- LinearRegression validation MSE, 2024: {results['lr_val_mse']:.6f}
- RandomForestRegressor validation MSE, 2024: {results['rf_val_mse']:.6f}
- LinearRegression test MSE, 2025: {results['lr_test_mse']:.6f}
- RandomForestRegressor test MSE, 2025: {results['rf_test_mse']:.6f}
- Best model by 2025 MSE: {results['best_model']}

## Correlation Analysis

Pearson correlations with air_temperature:

{format_correlations(results['correlations'])}

Highest absolute non-undefined correlation: {results['best_correlation_name']} ({results['best_correlation_value']:.6f})

## Spark Runtimes

{format_timings(timings)}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    total_start = time.perf_counter()
    timings: list[tuple[str, float]] = []
    args = parse_args()

    input_pattern = str((REPO_ROOT / args.input).resolve()) if not "://" in args.input else args.input
    local_files = sorted(glob.glob(input_pattern)) if not "://" in input_pattern else []
    if not local_files and not "://" in input_pattern:
        raise RuntimeError(f"No NOAA input files found for pattern: {args.input}")

    spark = timed(timings, "SparkSession creation", lambda: create_spark(args))
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    raw = sc.textFile(input_pattern)
    raw_count = timed(timings, "Raw line count", raw.count)
    if raw_count == 0:
        raise RuntimeError(f"No NOAA input lines found for pattern: {args.input}")

    parsed_rdd = raw.map(parse_noaa_line).filter(lambda row: row is not None).cache()
    parsed_count = timed(timings, "Parse and cache NOAA RDD", parsed_rdd.count)
    if parsed_count == 0:
        raise RuntimeError("NOAA parser produced zero rows after filtering missing labels.")

    weather = spark.createDataFrame(parsed_rdd, schema=SCHEMA).cache()
    dataframe_count = timed(timings, "Create and cache DataFrame", weather.count)

    year_bounds = weather.agg({"year": "min"}).collect()[0][0], weather.agg(
        {"year": "max"}
    ).collect()[0][0]

    train_df = weather.filter((col("year") >= 1949) & (col("year") <= 2023)).cache()
    val_df = weather.filter(col("year") == 2024).cache()
    test_df = weather.filter(col("year") == 2025).cache()

    train_count = timed(timings, "Training split count", train_df.count)
    val_count = timed(timings, "Validation split count", val_df.count)
    test_count = timed(timings, "Test split count", test_df.count)
    if train_count == 0 or val_count == 0 or test_count == 0:
        raise RuntimeError(
            "One required chronological split is empty: "
            f"train={train_count}, validation={val_count}, test={test_count}"
        )

    mse_evaluator = RegressionEvaluator(
        labelCol="air_temperature", predictionCol="prediction", metricName="mse"
    )

    lr_imputer, lr_assembler = make_preprocessing_stages()
    lr = LinearRegression(
        labelCol="air_temperature",
        featuresCol="features",
        predictionCol="prediction",
        maxIter=100,
        regParam=0.001,
    )
    lr_pipeline = Pipeline(stages=[lr_imputer, lr_assembler, lr])
    lr_model = timed(timings, "LinearRegression fit", lambda: lr_pipeline.fit(train_df))
    lr_val_mse = timed(
        timings,
        "LinearRegression validation MSE",
        lambda: evaluate_mse(lr_model, val_df, mse_evaluator),
    )
    lr_test_mse = timed(
        timings,
        "LinearRegression test MSE",
        lambda: evaluate_mse(lr_model, test_df, mse_evaluator),
    )

    rf_imputer, rf_assembler = make_preprocessing_stages()
    rf = RandomForestRegressor(
        labelCol="air_temperature",
        featuresCol="features",
        predictionCol="prediction",
        numTrees=10,
        featureSubsetStrategy="auto",
        impurity="variance",
        seed=42,
    )
    rf_pipeline = Pipeline(stages=[rf_imputer, rf_assembler, rf])
    rf_model = timed(
        timings, "RandomForestRegressor fit", lambda: rf_pipeline.fit(train_df)
    )
    rf_val_mse = timed(
        timings,
        "RandomForestRegressor validation MSE",
        lambda: evaluate_mse(rf_model, val_df, mse_evaluator),
    )
    rf_test_mse = timed(
        timings,
        "RandomForestRegressor test MSE",
        lambda: evaluate_mse(rf_model, test_df, mse_evaluator),
    )

    corr_imputer = Imputer(
        inputCols=NULLABLE_FEATURE_COLS, outputCols=IMPUTED_COLS
    ).setStrategy("median")
    corr_model = timed(
        timings, "Correlation imputer fit", lambda: corr_imputer.fit(train_df)
    )
    corr_df = corr_model.transform(weather).cache()
    timed(timings, "Correlation DataFrame cache", corr_df.count)

    def compute_correlations():
        pairs = [(column, column) for column in BASE_FEATURE_COLS]
        pairs += list(zip(NULLABLE_FEATURE_COLS, IMPUTED_COLS))
        output = []
        for original_column, model_column in pairs:
            output.append(
                (original_column, corr_df.stat.corr(model_column, "air_temperature"))
            )
        return output

    correlations = timed(timings, "Correlation computation", compute_correlations)
    valid_correlations = [
        (name, value)
        for name, value in correlations
        if value is not None and not math.isnan(value)
    ]
    if not valid_correlations:
        raise RuntimeError("No defined correlations were available.")
    best_correlation_name, best_correlation_value = max(
        valid_correlations, key=lambda item: abs(item[1])
    )

    total_duration = time.perf_counter() - total_start
    timings.append(("Total runtime", total_duration))

    best_model = (
        "LinearRegression"
        if lr_test_mse <= rf_test_mse
        else "RandomForestRegressor"
    )
    results = {
        "input_pattern": args.input,
        "file_count": len(local_files) if local_files else "unknown",
        "raw_count": raw_count,
        "parsed_count": dataframe_count,
        "dropped_count": raw_count - parsed_count,
        "min_year": year_bounds[0],
        "max_year": year_bounds[1],
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "lr_val_mse": lr_val_mse,
        "rf_val_mse": rf_val_mse,
        "lr_test_mse": lr_test_mse,
        "rf_test_mse": rf_test_mse,
        "best_model": best_model,
        "correlations": correlations,
        "best_correlation_name": best_correlation_name,
        "best_correlation_value": best_correlation_value,
    }
    write_report(results, timings)
    print(f"Problem 2 report written to {REPORT_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
