# Problem 1: Heart Disease Classification with PySpark — Full Report

## 1. Running Instructions

Set up local Spark and Python paths before execution:

```bash
export SPARK_HOME=/home/isaac/Apps/spark-4.0.2-bin-hadoop3
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-src.zip:$PYTHONPATH
```

Run the program:

```bash
python solutions/problem_1/main.py
```

The script is configured to use all available local cores (`local[*]`), with 16 GB driver memory, 4 GB off-heap memory, and a 2 GB max result size.

---

## 2. Dataset Overview

The dataset used is `heart_2020_cleaned.csv` from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2020 survey. It contains **319,795** records after filtering invalid `HeartDisease` entries (only "Yes"/"No" retained).

**Features (17 total):**

- 4 numeric: `BMI`, `PhysicalHealth`, `MentalHealth`, `SleepTime`
- 13 categorical: `Smoking`, `AlcoholDrinking`, `Stroke`, `DiffWalking`, `Sex`, `AgeCategory`, `Race`, `Diabetic`, `PhysicalActivity`, `GenHealth`, `Asthma`, `KidneyDisease`, `SkinCancer`

**Targets used:**

- Part B/C: Binary `HeartDisease` ("Yes" / "No")
- Part D: Multiclass `AgeCategory` (13 ordinal categories: "18-24" through "80+")

**Train/Test split:** 80%/20% — yielding **255,972** training and **63,823** test records in Part B/C and D (same split reused).

---

## 3. Preprocessing Pipeline

All categorical features are transformed using `StringIndexer` (converted to numeric indices), and then assembled into a feature vector with `VectorAssembler`. The label column (`HeartDisease` or `AgeCategory`) is also indexed separately. This entire pipeline (indexers → label indexer → assembler → classifier) is wrapped in a `Pipeline` for seamless cross-validation.

---

## 4. Part B: DecisionTree with 5-Fold CrossValidator

### 4.1 Hyperparameter Grid

The `ParamGridBuilder` explores **18 combinations**:
| Parameter | Values |
|-----------|--------|
| impurity | entropy, gini |
| maxDepth | 6, 12, 24 |
| maxBins | 20, 50, 100 |

### 4.2 Best Model Found

- **impurity**: gini
- **maxDepth**: 6
- **maxBins**: 100

The model favors a shallow tree (depth 6) with the Gini impurity criterion and the maximum bin count, suggesting the data benefits from fine-grained categorical splitting but not from deep trees (risk of overfitting).

### 4.3 Results on Test Set (20% holdout)

| Metric                         | Value  |
| ------------------------------ | ------ |
| AUC (ROC)                      | 0.5953 |
| Area under PR                  | 0.1244 |
| Overall Accuracy               | 0.9146 |
| **HeartDisease=Yes** Precision | 0.5812 |
| **HeartDisease=Yes** Recall    | 0.0524 |
| **HeartDisease=No** Precision  | 0.9172 |
| **HeartDisease=No** Recall     | 0.9964 |

**Analysis:** The high overall accuracy (91.5%) is misleading — it is driven by the class imbalance. The majority class (No) has near-perfect recall (99.6%), while the minority class (Yes) has very poor recall (5.2%), meaning the model almost never predicts heart disease. The AUC of 0.595 is only marginally better than random (0.5), indicating the selected features have limited discriminative power for heart disease under this model.

### 4.4 Runtime

| Step                                       | Duration (s) |
| ------------------------------------------ | ------------ |
| SparkSession creation                      | 3.424        |
| Load CSV                                   | 0.753        |
| Preprocessing filters                      | 0.103        |
| Train/Test randomSplit (lazy)              | 0.020        |
| Count train/test records                   | 3.264        |
| **CrossValidator fit (5-fold, 18 models)** | **389.592**  |
| Training prediction + eval (AUC/PR)        | 2.013        |
| Save best DecisionTree model               | 0.692        |
| Test prediction + eval (AUC/PR)            | 1.389        |
| Test MulticlassMetrics                     | 2.034        |
| Test BinaryClassificationMetrics           | 0.289        |

The CrossValidator took **389.6 seconds** (~6.5 minutes) because it trains 18 models × 5 folds = **90 full model fits**, each involving string indexing and vector assembly across ~256K records.

---

## 5. Part C: TrainValidationSplit vs CrossValidator

### 5.1 Approach

Replaces the 5-fold CrossValidator with a single `TrainValidationSplit` using `trainRatio=0.8` (80% of training data for fitting, 20% for validation). The same hyperparameter grid (18 combinations) and evaluation metric (accuracy via `MulticlassClassificationEvaluator`) are used.

### 5.2 Best Model Found

Identical to Part B: impurity=gini, maxDepth=6, maxBins=100.

### 5.3 Results on Test Set

Results are **identical** to Part B (within floating-point precision), confirming that the best hyperparameters are robust and that TrainValidationSplit with a proper train/validation split produces comparable model selection quality for this dataset.

| Metric           | Value  |
| ---------------- | ------ |
| AUC (ROC)        | 0.5953 |
| Area under PR    | 0.1244 |
| Overall Accuracy | 0.9146 |

### 5.4 Runtime Comparison

| Step                       | CrossValidator (s) | TrainValidationSplit (s) | Speedup         |
| -------------------------- | ------------------ | ------------------------ | --------------- |
| Hyperparameter search      | 389.592            | 83.043                   | **4.7× faster** |
| Training prediction + eval | 2.013              | 1.734                    | ~same           |
| Test prediction + eval     | 1.389              | 1.282                    | ~same           |
| Test MulticlassMetrics     | 2.034              | 1.130                    | ~same           |

**Why the speedup:** TrainValidationSplit fits only **18 models** (one per hyperparameter combination) vs CrossValidator's 90 models (18 × 5 folds). This is a roughly 5× reduction, closely matching the observed 4.7× speedup.

---

## 6. Part D: Multiclass Target — AgeCategory

### 6.1 Task

Predict the patient's `AgeCategory` (13 classes: "18-24", "25-29", ..., "80+") using all other features. This is a significantly harder task — age is only weakly correlated with the other health indicators.

### 6.2 DecisionTree Baseline

- Fixed hyperparams: impurity=gini, maxDepth=6, maxBins=20
- **Fit time**: 10.297 s
- **Test Accuracy**: 0.1573
- **Weighted Precision**: 0.1342
- **Weighted Recall**: 0.1573

### 6.3 RandomForest with Hyperparameter Tuning

- Tuned `numTrees` ∈ {5, 10, 20} via TrainValidationSplit
- **Best numTrees**: 20
- **Fit time**: 26.549 s
- **Test Accuracy**: 0.1648
- **Weighted Precision**: 0.1230
- **Weighted Recall**: 0.1648

### 6.4 Model Comparison

| Model                   | Accuracy | Weighted Precision | Weighted Recall | Fit Time (s) |
| ----------------------- | -------- | ------------------ | --------------- | ------------ |
| DecisionTree            | 0.1573   | 0.1342             | 0.1573          | 10.297       |
| RandomForest (20 trees) | 0.1648   | 0.1230             | 0.1648          | 26.549       |

**Analysis:** Both models perform poorly (accuracy ~16%), which is expected — with 13 balanced-ish classes, random guessing yields ~7.7%, so the models have learned _something_, but the available features (health indicators, lifestyle) are weak predictors of age category. RandomForest gains a small accuracy improvement at the cost of ~2.6× longer training time.

---

## 7. Summary of All Spark Runtimes

| Phase / Step                           | Duration (s) |
| -------------------------------------- | ------------ |
| **Initialization**                     |              |
| SparkSession creation                  | 3.424        |
| Load CSV                               | 0.753        |
| Preprocessing filters                  | 0.103        |
| **Part B — CrossValidator**            |              |
| Train/Test randomSplit (lazy)          | 0.020        |
| Count train/test records               | 3.264        |
| CrossValidator fit (5-fold, 18 models) | **389.592**  |
| Training prediction + eval (AUC/PR)    | 2.013        |
| Save best DecisionTree model           | 0.692        |
| Test prediction + eval                 | 1.389        |
| Test MulticlassMetrics                 | 2.034        |
| Test BinaryClassificationMetrics       | 0.289        |
| **Part C — TrainValidationSplit**      |              |
| TrainValidationSplit fit (18 models)   | **83.043**   |
| TVS training prediction + eval         | 1.734        |
| Save TVS best model                    | 0.181        |
| TVS test prediction + eval             | 1.282        |
| TVS MulticlassMetrics                  | 1.130        |
| TVS BinaryClassificationMetrics        | 0.260        |
| **Part D — AgeCategory**               |              |
| AgeCategory randomSplit (lazy)         | 0.003        |
| Count AgeCategory train/test records   | 0.880        |
| DecisionTree (AgeCategory) fit         | 10.297       |
| DecisionTree (AgeCategory) transform   | 0.130        |
| DecisionTree MulticlassMetrics         | 1.122        |
| RandomForest TVS fit (3 models)        | **26.549**   |
| RandomForest transform                 | 0.068        |
| RandomForest MulticlassMetrics         | 1.015        |

**Total runtime**: ~514 seconds (~8.6 minutes) for the full script.

---

## 8. Key Takeaways

1. **TrainValidationSplit** is ~4.7× faster than CrossValidator for the same hyperparameter grid, with no loss in model quality for this dataset.
2. **Heart disease prediction** using these features yields poor minority-class recall (5.2%) despite high overall accuracy (91.5%), revealing a class imbalance issue that would benefit from class weighting or different evaluation metrics.
3. **AgeCategory prediction** is a fundamentally difficult task with the given features — both DecisionTree and RandomForest achieve only ~16% accuracy, barely above random chance for 13 classes.
4. **RandomForest** provides marginal gains over DecisionTree for AgeCategory but at 2.6× the training cost.
