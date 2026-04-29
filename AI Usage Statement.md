# Problem 1:

- Models used: Gemini Pro and OpenCode with the BigPickle model.

## Main prompting:

To Gemini Pro:

```
According to the instructions and solution attached to Problem 1 of my homework, cross-check the solution and highlight any mistakes. If you find any, please explain them and provide the correct solution.
```

Output:

```
Your PySpark implementation successfully performs the necessary data splitting, pipeline construction and model tuning, but deliberately and correctly deviates from the technically flawed API instructions in the assignment. The prompt requests the use of BinaryClassificationMetrics and BinaryClassificationEvaluator to extract scalar accuracy, precision, and recall, which is impossible within PySpark's framework. Your workaround using the Multiclass equivalents is the mathematically sound solution. To avoid losing marks due to a strict marking system, keep your current code but explicitly document these PySpark API limitations in your 'Problem 1.txt' submission to justify your programming choices.
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

# Problem 3:
