# Problem 1: Heart Disease Classification with PySpark

## Running

```bash
cd solutions/problem_1/
source env/bin/activate
export SPARK_HOME=/home/isaac/Apps/spark-4.0.2-bin-hadoop3
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-src.zip:$PYTHONPATH
python main.py
```

## Dataset + Pipeline

- BRFSS 2020 `heart_2020_cleaned.csv`, 319,795 rows after filtering to HeartDisease=Yes/No.
- 17 features: 4 numeric + 13 categorical; StringIndexer + VectorAssembler in a Pipeline.
- 80/20 split reused across Parts B–D (255,972 train / 63,823 test).
- Labels: HeartDisease for Parts B/C (binary), AgeCategory for Part D (13 classes).

## Strategy

- Build a single Spark ML Pipeline (indexers → label indexer → assembler → classifier).
- For HeartDisease, tune a DecisionTree with 5-fold CrossValidator using AUC to pick hyperparameters, then compare against TrainValidationSplit for speed.
- For AgeCategory, run a GBT (OvR) baseline and a tuned RandomForest to assess multiclass performance.

## Part B: DecisionTree + CrossValidator (AUC)

- Grid: impurity=[entropy,gini], maxDepth=[6,12,24], maxBins=[20,50,100].
- Best model: entropy, maxDepth=6, maxBins=20.

Test metrics (20% holdout):

| Metric | Value |
| --- | --- |
| AUC (BinaryClassificationEvaluator) | 0.5912 |
| Accuracy | 0.9142 |
| Precision (Yes) | 0.5619 |
| Recall (Yes) | 0.0459 |
| Precision (No) | 0.9167 |
| Recall (No) | 0.9966 |
| AUC (BinaryClassificationMetrics) | 0.5213 |
| Area under PR (BinaryClassificationMetrics) | 0.3352 |

**Note:** `BinaryClassificationMetrics` uses hard class predictions, so its AUC differs from the evaluator’s rawPrediction-based AUC.

**Quick read:** Accuracy is high due to class imbalance, but recall for HeartDisease=Yes is only 4.6%, so the model rarely flags positives.

## Part C: TrainValidationSplit vs CrossValidator

- Best hyperparameters match Part B (entropy, depth 6, bins 20).
- Test metrics are identical to Part B within precision.
- Runtime: CrossValidator 472.9 s vs TrainValidationSplit 106.7 s (~4.4× faster).

## Runtime Snapshot

| Step | Duration (s) |
| --- | --- |
| SparkSession creation | 4.353 |
| Load CSV | 1.277 |
| Preprocessing filters | 0.152 |
| CrossValidator fit (5-fold) | 472.912 |
| TrainValidationSplit fit | 106.708 |
| RandomForest CV fit (AgeCategory) | 13106.924 |

## Results Summary

- Best HeartDisease tree (entropy, depth 6, bins 20) reaches AUC 0.5912, accuracy 0.9142, but Recall(Yes)=0.0459.
- TrainValidationSplit matches CV quality with much lower tuning time.
- AgeCategory remains weak: GBT (OvR) accuracy 0.1704, RandomForest accuracy 0.1736.

## Part D: Target = AgeCategory (multiclass)

- GBT (OvR) test: accuracy 0.1704, weighted precision 0.1577, weighted recall 0.1704.
- RandomForest (CV) best: numTrees=20, maxDepth=12, maxBins=50, impurity=gini.
- RandomForest test: accuracy 0.1736, weighted precision 0.1616, weighted recall 0.1736.

## Takeaways

1. AUC-optimized trees select a shallow entropy model, but minority-class recall stays very low (4.6%).
2. TrainValidationSplit gives the same model quality as CV with ~4× less tuning time.
3. AgeCategory prediction remains weak (~17% accuracy) even with RandomForest tuning.
