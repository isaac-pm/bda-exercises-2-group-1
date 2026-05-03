from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    OneVsRest,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit


def print_duration(name, start, end):
    print(f"({name}) Duration: {end-start:.3f} seconds")


t0 = time.perf_counter()
spark = (
    SparkSession.builder.appName("HeartDiseaseClassification")
    .master("local[*]")
    .config("spark.driver.memory", "16g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "4g")
    .config("spark.driver.maxResultSize", "2g")
    .getOrCreate()
)
t1 = time.perf_counter()
print_duration("SparkSession creation", t0, t1)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "problem_1" / "heart_2020_cleaned.csv"
MODEL_PATH = REPO_ROOT / "solutions" / "problem_1" / "best_decision_tree_model"
MODEL_PATH_RF = REPO_ROOT / "solutions" / "problem_1" / "best_random_forest_model"
MODEL_PATH_TVS = REPO_ROOT / "solutions" / "problem_1" / "best_decision_tree_model_tvs"

# ----------------------------------------------------------------
# Step A.1: Load and preprocess the data

schema = T.StructType(
    [
        T.StructField("HeartDisease", T.StringType(), True),
        T.StructField("BMI", T.DoubleType(), True),
        T.StructField("Smoking", T.StringType(), True),
        T.StructField("AlcoholDrinking", T.StringType(), True),
        T.StructField("Stroke", T.StringType(), True),
        T.StructField("PhysicalHealth", T.DoubleType(), True),
        T.StructField("MentalHealth", T.DoubleType(), True),
        T.StructField("DiffWalking", T.StringType(), True),
        T.StructField("Sex", T.StringType(), True),
        T.StructField("AgeCategory", T.StringType(), True),
        T.StructField("Race", T.StringType(), True),
        T.StructField("Diabetic", T.StringType(), True),
        T.StructField("PhysicalActivity", T.StringType(), True),
        T.StructField("GenHealth", T.StringType(), True),
        T.StructField("SleepTime", T.DoubleType(), True),
        T.StructField("Asthma", T.StringType(), True),
        T.StructField("KidneyDisease", T.StringType(), True),
        T.StructField("SkinCancer", T.StringType(), True),
    ]
)

t0 = time.perf_counter()
data = spark.read.option("header", True).csv(str(DATA_PATH), schema=schema)
t1 = time.perf_counter()
print_duration("Load CSV", t0, t1)

t0 = time.perf_counter()
data = data.filter(col("HeartDisease").isNotNull())
data = data.filter((col("HeartDisease") == "Yes") | (col("HeartDisease") == "No"))
t1 = time.perf_counter()
print_duration("Preprocessing filters", t0, t1)

# ----------------------------------------------------------------
# Part (A): Load and preprocess data
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PART (A): Load and preprocess data")
print("=" * 60)

# ----------------------------------------------------------------
# Part (B): DecisionTree with CrossValidator
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PART (B): DecisionTree with CrossValidator")
print("=" * 60)

# ----------------------------------------------------------------
# Step B.1: 80%/20% split

t0 = time.perf_counter()
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
t1 = time.perf_counter()
print_duration("Train/Test randomSplit (lazy)", t0, t1)

t0 = time.perf_counter()
train_count = train_data.count()
test_count = test_data.count()
t1 = time.perf_counter()
print_duration("Counting train/test records", t0, t1)

print(f"\n(B.1) Training records: {train_count}")
print(f"(B.1) Testing records: {test_count}")

categorical_columns = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

indexed_columns = [col + "_indexed" for col in categorical_columns]

string_indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="skip")
    for col in categorical_columns
]

label_indexer = StringIndexer(
    inputCol="HeartDisease", outputCol="label", handleInvalid="skip"
)

numeric_columns = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

feature_columns = numeric_columns + indexed_columns

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

pipeline = Pipeline(stages=string_indexers + [label_indexer, assembler, dt])

# ----------------------------------------------------------------
# Step B.2: Hyperparameter tuning with 5-fold cross-validation

paramGrid = (
    ParamGridBuilder()
    .addGrid(dt.impurity, ["entropy", "gini"])
    .addGrid(dt.maxDepth, [6, 12, 24])
    .addGrid(dt.maxBins, [20, 50, 100])
    .build()
)

# CrossValidator requires a pyspark.ml Evaluator. We use
# BinaryClassificationEvaluator(metricName="areaUnderROC"), which is equivalent
# to the areaUnderROC metric from pyspark.mllib.evaluation.BinaryClassificationMetrics.
cv_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=cv_evaluator,
    numFolds=5,
    seed=42,
)

print("\n(B.2) Starting 5-fold cross-validation with hyperparameter search...")
print(
    "(B.2) Parameters: impurity=[entropy, gini], maxDepth=[6, 12, 24], maxBins=[20, 50, 100]"
)
print("(B.2) This may take a while...")

t0 = time.perf_counter()
cv_model = cv.fit(train_data)
t1 = time.perf_counter()
print_duration("CrossValidator fit (5-fold)", t0, t1)

best_model = cv_model.bestModel
best_dt_model = best_model.stages[-1]

print(f"\n(B.2) Best hyperparameters:")
print(f"  impurity: {best_dt_model.getImpurity()}")
print(f"  maxDepth: {best_dt_model.getMaxDepth()}")
print(f"  maxBins: {best_dt_model.getMaxBins()}")

# ----------------------------------------------------------------
# Step B.3: Measure metrics on training data with best model


train_auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
t0 = time.perf_counter()
train_predictions = best_model.transform(train_data)
train_auc = train_auc_evaluator.evaluate(train_predictions)
t1 = time.perf_counter()
print_duration("Training prediction + eval (AUC)", t0, t1)

print(f"\n(B.3) Training metrics (best model):")
print(f"  AUC: {train_auc}")

# Save the best DecisionTree model
t0 = time.perf_counter()
best_dt_model.write().overwrite().save(str(MODEL_PATH))
t1 = time.perf_counter()
print_duration("Save best DecisionTree model", t0, t1)
print(
    "\n(B.3) Best DecisionTree model saved to: solutions/problem_1/best_decision_tree_model"
)

# ----------------------------------------------------------------
# Step B.4: Evaluate best model on 20% test split

t0 = time.perf_counter()
test_predictions = best_model.transform(test_data)
test_auc = train_auc_evaluator.evaluate(test_predictions)
t1 = time.perf_counter()
print_duration("Test prediction + eval (AUC/PR)", t0, t1)

print(f"\n(B.4) Test set metrics (best model, 20% split):")
print(f"  AUC: {test_auc}")

# Compute per-label precision, recall, accuracy using MulticlassMetrics
t0 = time.perf_counter()
prediction_label_rdd_b = test_predictions.select("prediction", "label").rdd.map(
    lambda row: (float(row.prediction), float(row.label))
)
metrics_b = MulticlassMetrics(prediction_label_rdd_b)
accuracy = metrics_b.accuracy
precision_1 = metrics_b.precision(1.0)
recall_1 = metrics_b.recall(1.0)
precision_0 = metrics_b.precision(0.0)
recall_0 = metrics_b.recall(0.0)
t1 = time.perf_counter()
print_duration("Test MulticlassMetrics computation", t0, t1)

print(f"\n(B.4) Per-label metrics on test set (MulticlassMetrics):")
print(f"  Label 1 (HeartDisease=Yes):")
print(f"    Precision: {precision_1:.4f}")
print(f"    Recall: {recall_1:.4f}")
print(f"  Label 0 (HeartDisease=No):")
print(f"    Precision: {precision_0:.4f}")
print(f"    Recall: {recall_0:.4f}")
print(f"  Overall Accuracy: {accuracy:.4f}")

t0 = time.perf_counter()
bc_metrics_b = BinaryClassificationMetrics(prediction_label_rdd_b)
t1 = time.perf_counter()
print_duration("Test BinaryClassificationMetrics computation", t0, t1)
print(f"  AUC (BinaryClassificationMetrics): {bc_metrics_b.areaUnderROC:.4f}")
print(f"  Area under PR (BinaryClassificationMetrics): {bc_metrics_b.areaUnderPR:.4f}")

# ----------------------------------------------------------------
# Part (C): TrainValidationSplit instead of CrossValidator
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PART (C): TrainValidationSplit vs CrossValidator Comparison")
print("=" * 60)

# ----------------------------------------------------------------
# Step C.1: Reuse 80/20 split (train_data, test_data) from Step B.1
print("\n(C.1) Reusing 80/20 split from Part B Step (B.1)")

# ----------------------------------------------------------------
# Step C.2: Replace CrossValidator with TrainValidationSplit (trainRatio=0.8)
# TrainValidationSplit requires a pyspark.ml Evaluator. We use
# BinaryClassificationEvaluator(metricName="areaUnderROC"), which is equivalent
# to the areaUnderROC metric from pyspark.mllib.evaluation.BinaryClassificationMetrics.
tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=cv_evaluator,
    trainRatio=0.8,
    seed=42,
)

print("\n(C.2) Starting TrainValidationSplit with hyperparameter search...")
print(
    "(C.2) Parameters: impurity=[entropy, gini], maxDepth=[6,12,24], maxBins=[20,50,100]"
)
print("(C.2) trainRatio=0.8 (80% of training data for fitting, 20% for validation)")

t0 = time.perf_counter()
tvs_model = tvs.fit(train_data)
t1 = time.perf_counter()
print_duration("TrainValidationSplit fit", t0, t1)

best_model_c = tvs_model.bestModel
best_dt_model_c = best_model_c.stages[-1]

print(f"\n(C.2) Best hyperparameters (TrainValidationSplit):")
print(f"  impurity: {best_dt_model_c.getImpurity()}")
print(f"  maxDepth: {best_dt_model_c.getMaxDepth()}")
print(f"  maxBins: {best_dt_model_c.getMaxBins()}")

# ----------------------------------------------------------------
# Step C.3: Evaluate on training data, save model
t0 = time.perf_counter()
train_predictions_c = best_model_c.transform(train_data)
train_auc_c = train_auc_evaluator.evaluate(train_predictions_c)
t1 = time.perf_counter()
print_duration("TVS training prediction + eval", t0, t1)

print(f"\n(C.3) Training metrics (TrainValidationSplit best model):")
print(f"  AUC: {train_auc_c:.4f}")

# Save best TVS model
t0 = time.perf_counter()
best_dt_model_c.write().overwrite().save(str(MODEL_PATH_TVS))
t1 = time.perf_counter()
print_duration("Save TVS best DecisionTree model", t0, t1)
print(
    "(C.3) Best DecisionTree model (TrainValidationSplit) saved to: best_decision_tree_model_tvs"
)

# ----------------------------------------------------------------
# Step C.4: Evaluate on 20% test split
t0 = time.perf_counter()
test_predictions_c = best_model_c.transform(test_data)
test_auc_c = train_auc_evaluator.evaluate(test_predictions_c)
t1 = time.perf_counter()
print_duration("TVS test prediction + eval", t0, t1)

print(f"\n(C.4) Test set metrics (TrainValidationSplit, 20% split):")
print(f"  AUC: {test_auc_c:.4f}")

# Per-label precision, recall, accuracy using MulticlassMetrics
t0 = time.perf_counter()
prediction_label_rdd_c = test_predictions_c.select("prediction", "label").rdd.map(
    lambda row: (float(row.prediction), float(row.label))
)
metrics_c = MulticlassMetrics(prediction_label_rdd_c)
accuracy_c = metrics_c.accuracy
precision_1_c = metrics_c.precision(1.0)
recall_1_c = metrics_c.recall(1.0)
precision_0_c = metrics_c.precision(0.0)
recall_0_c = metrics_c.recall(0.0)
t1 = time.perf_counter()
print_duration("TVS MulticlassMetrics computation", t0, t1)

print(
    f"\n(C.4) Per-label metrics on test set (TrainValidationSplit, MulticlassMetrics):"
)
print(f"  Label 1 (HeartDisease=Yes):")
print(f"    Precision: {precision_1_c:.4f}")
print(f"    Recall: {recall_1_c:.4f}")
print(f"  Label 0 (HeartDisease=No):")
print(f"    Precision: {precision_0_c:.4f}")
print(f"    Recall: {recall_0_c:.4f}")
print(f"  Overall Accuracy: {accuracy_c:.4f}")

print("\n(C.5) Comparison: CrossValidator (Part B) vs TrainValidationSplit (Part C)")
print(f"  Metric           | CrossValidator | TrainValidationSplit")
print(f"  -----------------|----------------|---------------------")
print(f"  Accuracy         | {accuracy:.4f}         | {accuracy_c:.4f}")
print(f"  Precision (Yes)  | {precision_1:.4f}         | {precision_1_c:.4f}")
print(f"  Recall    (Yes)  | {recall_1:.4f}         | {recall_1_c:.4f}")
print(f"  Precision (No)   | {precision_0:.4f}         | {precision_0_c:.4f}")
print(f"  Recall    (No)   | {recall_0:.4f}         | {recall_0_c:.4f}")
print(f"  Best impurity    | {best_dt_model.getImpurity()}         | {best_dt_model_c.getImpurity()}")
print(f"  Best maxDepth    | {best_dt_model.getMaxDepth()}         | {best_dt_model_c.getMaxDepth()}")
print(f"  Best maxBins     | {best_dt_model.getMaxBins()}         | {best_dt_model_c.getMaxBins()}")
print("  Note: CrossValidator uses 5-fold CV (more reliable, slower).")
print("  TrainValidationSplit uses a single 80/20 split (faster, less stable).")

t0 = time.perf_counter()
bc_metrics_c = BinaryClassificationMetrics(prediction_label_rdd_c)
t1 = time.perf_counter()
print_duration("TVS BinaryClassificationMetrics computation", t0, t1)
print(f"  AUC (BinaryClassificationMetrics): {bc_metrics_c.areaUnderROC:.4f}")
print(f"  Area under PR (BinaryClassificationMetrics): {bc_metrics_c.areaUnderPR:.4f}")

# ----------------------------------------------------------------
# Part (D): Target=AgeCategory, RandomForest, MulticlassMetrics
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PART (D): Target=AgeCategory, RandomForest, MulticlassMetrics")
print("=" * 60)

# ----------------------------------------------------------------
# Step D.1: Create 80/20 split for AgeCategory target
print("\n(D.1) Creating 80/20 split for AgeCategory target...")
data_d = data.filter(col("AgeCategory").isNotNull())
t0 = time.perf_counter()
train_data_d, test_data_d = data_d.randomSplit([0.8, 0.2], seed=42)
t1 = time.perf_counter()
print_duration("AgeCategory Train/Test randomSplit (lazy)", t0, t1)

t0 = time.perf_counter()
train_count_d = train_data_d.count()
test_count_d = test_data_d.count()
t1 = time.perf_counter()
print_duration("Counting AgeCategory train/test records", t0, t1)
print(f"(D.1) Training records: {train_count_d}")
print(f"(D.1) Testing records: {test_count_d}")

# Features: exclude AgeCategory (target), keep all other original features
categorical_columns_d = [
    "HeartDisease",
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

numeric_columns_d = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
indexed_columns_d = [col + "_indexed" for col in categorical_columns_d]

# String indexers for feature columns (excluding AgeCategory)
string_indexers_d = [
    StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="skip")
    for col in categorical_columns_d
]

# Label indexer for AgeCategory (multiclass target)
label_indexer_d = StringIndexer(
    inputCol="AgeCategory", outputCol="label", handleInvalid="skip"
)

# Vector assembler for features
assembler_d = VectorAssembler(
    inputCols=numeric_columns_d + indexed_columns_d, outputCol="features"
)

# ----------------------------------------------------------------
# Step D.2: Train and evaluate GradientBoostedTrees (OvR) for AgeCategory

print("\n(D.2) Training GradientBoostedTrees (OvR) for AgeCategory target...")
dt_d = GBTClassifier(labelCol="label", featuresCol="features", maxDepth=6, maxIter=10)
ovr_gbt = OneVsRest(classifier=dt_d)
pipeline_dt = Pipeline(
    stages=string_indexers_d + [label_indexer_d, assembler_d, ovr_gbt]
)
t0 = time.perf_counter()
model_dt = pipeline_dt.fit(train_data_d)
t1 = time.perf_counter()
print_duration("GradientBoostedTrees (AgeCategory) fit", t0, t1)
t0 = time.perf_counter()
pred_dt = model_dt.transform(test_data_d)
t1 = time.perf_counter()
print_duration("GradientBoostedTrees (AgeCategory) transform", t0, t1)

# Use MulticlassMetrics for evaluation

t0 = time.perf_counter()
prediction_label_rdd_dt = pred_dt.select("prediction", "label").rdd.map(
    lambda row: (float(row.prediction), float(row.label))
)
metrics_dt = MulticlassMetrics(prediction_label_rdd_dt)
acc_dt = metrics_dt.accuracy
prec_dt = metrics_dt.weightedPrecision
rec_dt = metrics_dt.weightedRecall
t1 = time.perf_counter()
print_duration("GradientBoostedTrees (AgeCategory) MulticlassMetrics", t0, t1)

print(f"\n(D.2) GradientBoostedTrees test metrics (AgeCategory, MulticlassMetrics):")
print(f"  Accuracy: {acc_dt:.4f}")
print(f"  Weighted Precision: {prec_dt:.4f}")
print(f"  Weighted Recall: {rec_dt:.4f}")

# ----------------------------------------------------------------
# Step D.3: RandomForest with hyperparameter tuning (numTrees, maxDepth, maxBins, impurity)
print(
    "\n(D.3) Training RandomForest with hyperparameter tuning for AgeCategory target..."
)

rf_d = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
)

pipeline_rf = Pipeline(stages=string_indexers_d + [label_indexer_d, assembler_d, rf_d])

paramGrid_builder_rf = (
    ParamGridBuilder()
    .addGrid(rf_d.numTrees, [5, 10, 20])
    .addGrid(rf_d.maxDepth, [6, 12, 24])
    .addGrid(rf_d.maxBins, [20, 50, 100])
)

# RandomForest in Spark supports impurity=[gini, entropy].
paramGrid_rf = paramGrid_builder_rf.addGrid(rf_d.impurity, ["gini", "entropy"]).build()

# Use MulticlassClassificationEvaluator for tuning
multiclass_cv_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy",
    # Note: equivalent to MulticlassMetrics.accuracy; ml API required by CrossValidator
)

cv_rf = CrossValidator(
    estimator=pipeline_rf,
    estimatorParamMaps=paramGrid_rf,
    evaluator=multiclass_cv_eval,
    numFolds=5,
    seed=42,
)

print(
    "(D.3) Starting CrossValidator with numTrees=[5,10,20], "
    "maxDepth=[6,12,24], maxBins=[20,50,100], impurity=[gini,entropy]..."
)
t0 = time.perf_counter()
cv_rf_model = cv_rf.fit(train_data_d)
t1 = time.perf_counter()
print_duration("RandomForest CrossValidator fit", t0, t1)

best_rf_model = cv_rf_model.bestModel
best_rf_stage = best_rf_model.stages[-1]
best_num_trees = best_rf_stage.getNumTrees
print(f"(D.3) Best numTrees: {best_num_trees}")
print(f"(D.3) Best maxDepth: {best_rf_stage.getMaxDepth()}")
print(f"(D.3) Best maxBins: {best_rf_stage.getMaxBins()}")
print(f"(D.3) Best impurity: {best_rf_stage.getImpurity()}")

t0 = time.perf_counter()
pred_rf = best_rf_model.transform(test_data_d)
t1 = time.perf_counter()
print_duration("RandomForest transform", t0, t1)

# Use MulticlassMetrics for evaluation
t0 = time.perf_counter()
prediction_label_rdd_rf = pred_rf.select("prediction", "label").rdd.map(
    lambda row: (float(row.prediction), float(row.label))
)
metrics_rf = MulticlassMetrics(prediction_label_rdd_rf)
acc_rf = metrics_rf.accuracy
prec_rf = metrics_rf.weightedPrecision
rec_rf = metrics_rf.weightedRecall
t1 = time.perf_counter()
print_duration("RandomForest MulticlassMetrics", t0, t1)

print(f"\n(D.3) RandomForest test metrics (AgeCategory, MulticlassMetrics):")
labels_rf = sorted(
    prediction_label_rdd_rf.map(lambda x: x[1]).distinct().collect()
)
for label in labels_rf:
    print(f"  Label {int(label)}:")
    print(f"    Precision: {metrics_rf.precision(label):.4f}")
    print(f"    Recall: {metrics_rf.recall(label):.4f}")
print(f"  Accuracy: {acc_rf:.4f}")
print(f"  Weighted Precision: {prec_rf:.4f}")
print(f"  Weighted Recall: {rec_rf:.4f}")

# Save best RandomForest model
t0 = time.perf_counter()
best_rf_stage.write().overwrite().save(str(MODEL_PATH_RF))
t1 = time.perf_counter()
print_duration("Save best RandomForest model", t0, t1)
print(
    "(D.3) Best RandomForest model saved to: solutions/problem_1/best_random_forest_model"
)

# ----------------------------------------------------------------
# Step D.4: Model comparison summary
print(f"\n(D.4) Model Comparison Summary (AgeCategory target, full dataset):")
print(
    f"  GradientBoostedTrees (OvR)  - Accuracy: {acc_dt:.4f}, Precision: {prec_dt:.4f}, Recall: {rec_dt:.4f}"
)
print(
    f"  RandomForest               - Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f} (best numTrees={best_num_trees})"
)
print("\n(D.5) Comparison: GBT (OvR) vs RandomForest")
print(f"  Metric           | GBT (OvR)       | RandomForest")
print(f"  -----------------|-----------------|---------------------")
print(f"  Accuracy         | {acc_dt:.4f}         | {acc_rf:.4f}")
print(f"  Weighted Precision | {prec_dt:.4f}         | {prec_rf:.4f}")
print(f"  Weighted Recall    | {rec_dt:.4f}         | {rec_rf:.4f}")

spark.stop()
