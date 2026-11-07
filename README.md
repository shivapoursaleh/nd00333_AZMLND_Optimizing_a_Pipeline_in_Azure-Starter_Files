
# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**Problem statement:** The dataset contains data about a Portuguese bank's direct marketing phone campaigns: information about the customers (age, job, marital status, education, loans), the calls (contact type, month, day, duration), earlier campaigns, and economic indicators at the time of the call. We seek to predict whether a customer will subscribe to a term deposit (column `y`, yes/no), which makes this a binary classification problem.

**Solution:** The best performing model was a **LightGBM** classifier found by AutoML, with an accuracy of **0.9197**. The Scikit-learn Logistic Regression tuned with HyperDrive reached an accuracy of **0.9115**.

## Scikit-learn Pipeline
**Pipeline architecture:**

1. **Data:** `train.py` first tries to load the bank marketing CSV from the provided URL with `TabularDatasetFactory`. This failed on the compute cluster (an unsupported .NET Core dependency on Ubuntu 22.04), and the provided sample-data URL later became inaccessible altogether (HTTP 403). Therefore the same Bank Marketing dataset was downloaded from the UCI Machine Learning Repository (`bank-additional-full.csv`, 41,188 rows, 20 features + target) and saved as `bankmarketing_train.csv` in the project folder; `train.py` falls back to this local copy, which is uploaded to the cluster together with the script.
2. **Cleaning:** The `clean_data` function drops missing values, one-hot encodes `job`, `contact` and `education`, converts yes/no columns (`default`, `housing`, `loan`), `marital` and `poutcome` to 1/0, maps `month` and `day_of_week` to numbers, and separates the target `y` (yes = 1, no = 0).
3. **Split:** The data is split into 80% training and 20% test data (`random_state=42`, so every run uses the same split).
4. **Classification algorithm:** Scikit-learn `LogisticRegression`, a linear model that outputs the probability that a customer subscribes.
5. **Hyperparameter tuning:** HyperDrive runs `train.py` on a 4-node `Standard_D2_V2` compute cluster, tuning two hyperparameters:
   - `--C`: inverse of regularization strength (smaller values mean stronger regularization and a simpler model)
   - `--max_iter`: maximum number of iterations for the solver to converge
6. Each run logs its **Accuracy** on the test set; HyperDrive maximizes this metric over at most 20 runs (4 in parallel), and the best model is saved to `outputs/model.joblib` and registered.

**Best HyperDrive run:** C = **0.9466**, max_iter = **100**, Accuracy = **0.9115** (91.15%). The HyperDrive experiment completed all 20 runs in about 6 minutes; each training run took only about 16–22 seconds. The next best runs reached 0.9109 (C = 0.6621, max_iter = 200) and 0.9103 (C = 3.7755, max_iter = 100).

Tuning C and max_iter had only a small effect on accuracy, because every run used the same data split and the same algorithm. The model's performance is limited mainly by the algorithm itself (a linear model), not by its hyperparameters.


**Benefits of the parameter sampler:**
I used `RandomParameterSampling` with `C ~ uniform(0.01, 10)` and `max_iter ~ choice(50, 100, 200, 300)`. Random sampling supports both continuous and discrete hyperparameters, explores the search space efficiently with far fewer runs than an exhaustive grid search, lets all runs execute in parallel, and supports early termination policies (unlike Bayesian sampling). This makes it a good balance between search quality and the limited time and compute available.

**Benefits of the early stopping policy:**
I used `BanditPolicy(evaluation_interval=2, slack_factor=0.1)`. It stops any run whose primary metric is more than 10% worse than the best run so far, checked every 2 evaluation intervals. This saves compute time and cost by not spending resources on runs that are clearly underperforming, while still allowing reasonably good runs to finish.

## AutoML
AutoML was configured as a classification task with `accuracy` as the primary metric, a 30-minute experiment timeout, `y` as the label column, and 5-fold cross-validation (the data is split into 5 parts; each model is trained on 4 parts and validated on the remaining part, five times, which gives a more reliable estimate than a single train/test split).

The AutoML dataset was created with `TabularDatasetFactory` from the same CSV (uploaded to the workspace's default datastore) and cleaned with the same `clean_data` function as in `train.py`, so both experiments used identical features.

AutoML tried **9** pipelines: LightGBM, XGBoost and Extremely Randomized Trees with different scalers (MaxAbsScaler, SparseNormalizer, StandardScalerWrapper), plus two ensembles (VotingEnsemble and StackEnsemble) built from the best models. The top results were:

| Pipeline | Accuracy |
|---|---|
| MaxAbsScaler, LightGBM | **0.9197** |
| VotingEnsemble | 0.9192 |
| StackEnsemble | 0.9192 |
| SparseNormalizer, XGBoostClassifier | 0.9180 |
| MaxAbsScaler, XGBoostClassifier | 0.9172 |

Gradient-boosted tree models (LightGBM and XGBoost) clearly performed best, while Extremely Randomized Trees reached only 0.7336. Interestingly, the ensembles did not beat the single best LightGBM model.

The best model was a **LightGBM** classifier with a MaxAbsScaler preprocessing step with an accuracy of **0.9197** (91.97%). LightGBM is a gradient-boosting algorithm that builds many small decision trees one after another, each new tree correcting the errors of the previous ones. AutoML used it mostly with default hyperparameters: `min_data_in_leaf=20` (each leaf of a tree must contain at least 20 samples, which limits overfitting), `n_jobs=1` and CPU training. The model is registered in the workspace as `automl_best_model`.

AutoML's data guardrails flagged **class imbalance**: the smallest class (`y = 1`, customers who subscribed) has only **4,108** of **40,124** training samples (about 10.2%). Imbalanced data can make accuracy look better than the model really is, because a model biased toward the majority "no" class still scores high. The other guardrails passed: no missing feature values and no high-cardinality features were detected.

**Note on the data size:** AutoML trained on 40,124 rows, while HyperDrive used all 41,188 rows. When the CSV was loaded with `TabularDatasetFactory`, the `previous` column (number of earlier contacts) was automatically detected as a True/False column, because most of its values are 0 or 1. Rows with 2 or more previous contacts became empty and were removed by `dropna()` in `clean_data`. This could be fixed by setting the column type explicitly (`set_column_types={"previous": DataType.to_long()}`). The difference (about 2.6% of the rows) should be kept in mind when comparing the two accuracies.

## Pipeline comparison
| | HyperDrive | AutoML |
|---|---|---|
| Algorithm | Logistic Regression | LightGBM |
| Accuracy | 0.9115 | 0.9197 |
| Validation | Single 80/20 split | 5-fold cross-validation |

**Accuracy:** AutoML achieved **higher** accuracy, by about **0.8 percentage points** (0.9197 vs. 0.9115). Note that the two accuracies were measured slightly differently: HyperDrive on one 20% test split, AutoML as the average over 5 cross-validation folds.

**Architecture:** The HyperDrive pipeline tunes only two hyperparameters of a single, fixed algorithm (Logistic Regression), which can only learn linear relationships. AutoML automatically tried many different algorithms (including tree-based models such as gradient boosting), different preprocessing and scaling steps, and finally combined the best models into ensembles.

**Why the difference:** Tree-based models such as LightGBM can capture non-linear relationships and interactions between features (for example between call duration and economic indicators), which a linear model like Logistic Regression cannot. They also need no feature scaling and handle features with very different ranges (such as `duration` in seconds and `euribor3m` in percent) well. AutoML also explores a much larger search space. The difference is relatively small because the dataset is imbalanced, so all models reach high accuracy partly by correctly predicting the large "no" class.

## Future work
- **Use a better metric than accuracy:** The dataset is imbalanced (about 11% "yes"), so a model that always predicts "no" already reaches about 89% accuracy. Optimizing `AUC_weighted`, F1 or recall would better reflect how well the model finds customers who will subscribe.
- **Handle the class imbalance:** Use class weights (`class_weight="balanced"`) or resampling techniques such as SMOTE so the model learns the minority class better.
- **Remove `duration`:** Call duration is only known after a call has happened, so it would not be available when deciding whom to call. Removing it gives a more realistic model.
- **Improve the feature cleaning:** One-hot encode `marital`, `default`, `housing`, `loan` and `poutcome` instead of mapping them to 1/0, so values like "unknown", "divorced" or "failure" keep their own meaning.
- **Larger and better hyperparameter search:** Try more runs, a log-scale range for `C` (e.g. `loguniform`), Bayesian sampling, or other algorithms in HyperDrive.
- **Run AutoML longer:** The 30-minute timeout limited the number of models AutoML could try; more time could find a better model.
