from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 11
# LOGISTIC REGRESSION BASELINE
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_FILE = (
    MODEL_DATA_DIR
    / "train_60.csv"
)

VALID_FILE = (
    MODEL_DATA_DIR
    / "validation_20.csv"
)

METADATA_FILE = (
    MODEL_DATA_DIR
    / "modeling_metadata.json"
)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 11: LOGISTIC REGRESSION BASELINE")
print("=" * 90)


for file in [
    TRAIN_FILE,
    VALID_FILE,
    METADATA_FILE,
]:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )


train_df = pd.read_csv(
    TRAIN_FILE
)

valid_df = pd.read_csv(
    VALID_FILE
)


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


TARGET = metadata["target"]

numeric_features = (
    metadata[
        "numeric_features"
    ]
)

categorical_features = (
    metadata[
        "categorical_features"
    ]
)


print(
    f"\nTraining data   : "
    f"{train_df.shape[0]:,} rows x "
    f"{train_df.shape[1]} columns"
)

print(
    f"Validation data : "
    f"{valid_df.shape[0]:,} rows x "
    f"{valid_df.shape[1]} columns"
)

print(
    f"\nNumeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)


# ------------------------------------------------------------
# 3. X / Y
# ------------------------------------------------------------

X_train = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[
    TARGET
].astype(int)


X_valid = valid_df.drop(
    columns=[TARGET]
)

y_valid = valid_df[
    TARGET
].astype(int)


# ------------------------------------------------------------
# 4. NUMERIC PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),
    ]
)


# ------------------------------------------------------------
# 5. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)


# ------------------------------------------------------------
# 6. COLUMN TRANSFORMER
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


# ------------------------------------------------------------
# 7. LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)


# ------------------------------------------------------------
# 8. COMPLETE PIPELINE
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            logistic_model
        ),
    ]
)


# ------------------------------------------------------------
# 9. TRAIN MODEL
# ------------------------------------------------------------

print(
    "\nTraining Logistic Regression..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "[OK] Model training complete."
)


# ------------------------------------------------------------
# 10. PREDICTIONS
# ------------------------------------------------------------

train_predictions = pipeline.predict(
    X_train
)

train_probabilities = (
    pipeline.predict_proba(
        X_train
    )[:, 1]
)


valid_predictions = pipeline.predict(
    X_valid
)

valid_probabilities = (
    pipeline.predict_proba(
        X_valid
    )[:, 1]
)


# ------------------------------------------------------------
# 11. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(
    dataset_name,
    y_true,
    predictions,
    probabilities
):

    return {
        "dataset":
            dataset_name,

        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1_score":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),
    }


# ------------------------------------------------------------
# 12. TRAIN + VALIDATION METRICS
# ------------------------------------------------------------

train_metrics = calculate_metrics(
    "Train",
    y_train,
    train_predictions,
    train_probabilities
)


valid_metrics = calculate_metrics(
    "Validation",
    y_valid,
    valid_predictions,
    valid_probabilities
)


metrics_df = pd.DataFrame(
    [
        train_metrics,
        valid_metrics,
    ]
)


metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]


metrics_df[
    metric_columns
] = (
    metrics_df[
        metric_columns
    ]
    .round(4)
)


# ------------------------------------------------------------
# 13. CONFUSION MATRIX
# ------------------------------------------------------------

tn, fp, fn, tp = (
    confusion_matrix(
        y_valid,
        valid_predictions
    )
    .ravel()
)


confusion_df = pd.DataFrame(
    {
        "actual_negative": [
            tn,
            fn
        ],

        "actual_positive": [
            fp,
            tp
        ],
    },

    index=[
        "predicted_negative",
        "predicted_positive",
    ]
)


# ------------------------------------------------------------
# 14. RECALL AT TOP 10%
# ------------------------------------------------------------

validation_scores = pd.DataFrame(
    {
        "actual_churn":
            y_valid.values,

        "churn_probability":
            valid_probabilities,
    }
)


validation_scores = (
    validation_scores
    .sort_values(
        "churn_probability",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


top_10_count = int(
    np.ceil(
        len(
            validation_scores
        )
        * 0.10
    )
)


top_10 = validation_scores.head(
    top_10_count
)


total_churners = int(
    validation_scores[
        "actual_churn"
    ].sum()
)


top_10_churners = int(
    top_10[
        "actual_churn"
    ].sum()
)


recall_at_top_10 = (
    top_10_churners
    / total_churners
    if total_churners > 0
    else 0
)


# ------------------------------------------------------------
# 15. LIFT AT TOP 10%
# ------------------------------------------------------------

overall_churn_rate = (
    validation_scores[
        "actual_churn"
    ].mean()
)


top_10_churn_rate = (
    top_10[
        "actual_churn"
    ].mean()
)


lift_at_top_10 = (
    top_10_churn_rate
    / overall_churn_rate
    if overall_churn_rate > 0
    else 0
)


# ------------------------------------------------------------
# 16. SAVE VALIDATION PREDICTIONS
# ------------------------------------------------------------

validation_output = X_valid.copy()

validation_output[
    "actual_churn"
] = y_valid.values

validation_output[
    "predicted_churn"
] = valid_predictions

validation_output[
    "churn_probability"
] = valid_probabilities


VALID_PRED_FILE = (
    REPORT_DIR
    / "11_logistic_validation_predictions.csv"
)


validation_output.to_csv(
    VALID_PRED_FILE,
    index=False
)


# ------------------------------------------------------------
# 17. SAVE METRICS
# ------------------------------------------------------------

METRICS_FILE = (
    REPORT_DIR
    / "11_logistic_regression_metrics.csv"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ------------------------------------------------------------
# 18. SAVE MODEL
# ------------------------------------------------------------

MODEL_FILE = (
    MODEL_DIR
    / "logistic_regression_baseline.pkl"
)


joblib.dump(
    pipeline,
    MODEL_FILE
)


# ------------------------------------------------------------
# 19. CLASSIFICATION REPORT
# ------------------------------------------------------------

classification_text = (
    classification_report(
        y_valid,
        valid_predictions,
        digits=4,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 20. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - STEP 11 "
    "LOGISTIC REGRESSION BASELINE"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "Model: Logistic Regression"
)

report.append(
    "Class weighting: balanced"
)

report.append(
    "Decision threshold: 0.50"
)

report.append("")

report.append(
    "PREPROCESSING"
)

report.append(
    "-" * 100
)

report.append(
    "Numeric: median imputation + StandardScaler"
)

report.append(
    "Categorical: most-frequent imputation + one-hot encoding"
)

report.append(
    "All preprocessing is fitted on training data only."
)

report.append("")

report.append(
    "TRAIN / VALIDATION METRICS"
)

report.append(
    "-" * 100
)

report.append(
    metrics_df.to_string(
        index=False
    )
)

report.append("")

report.append(
    "VALIDATION CONFUSION MATRIX"
)

report.append(
    "-" * 100
)

report.append(
    f"True Negatives  : {tn:,}"
)

report.append(
    f"False Positives : {fp:,}"
)

report.append(
    f"False Negatives : {fn:,}"
)

report.append(
    f"True Positives  : {tp:,}"
)

report.append("")

report.append(
    "VALIDATION CLASSIFICATION REPORT"
)

report.append(
    "-" * 100
)

report.append(
    classification_text
)

report.append("")

report.append(
    "BUSINESS RANKING METRICS"
)

report.append(
    "-" * 100
)

report.append(
    f"Validation churners: "
    f"{total_churners:,}"
)

report.append(
    f"Top 10% customers scored: "
    f"{top_10_count:,}"
)

report.append(
    f"Churners captured in top 10%: "
    f"{top_10_churners:,}"
)

report.append(
    f"Recall@Top10%: "
    f"{recall_at_top_10:.4f}"
)

report.append(
    f"Lift@Top10%: "
    f"{lift_at_top_10:.2f}x"
)

report.append("")

report.append(
    "IMPORTANT"
)

report.append(
    "-" * 100
)

report.append(
    "The final test set was NOT used in this step."
)

report.append(
    "Model comparison and threshold decisions "
    "must be based on validation data first."
)


REPORT_FILE = (
    REPORT_DIR
    / "11_logistic_regression_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 21. TERMINAL OUTPUT
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "TRAIN / VALIDATION METRICS"
)

print(
    "=" * 90
)

print(
    metrics_df.to_string(
        index=False
    )
)


print(
    "\nVALIDATION CONFUSION MATRIX"
)

print(
    f"TN: {tn:,}"
)

print(
    f"FP: {fp:,}"
)

print(
    f"FN: {fn:,}"
)

print(
    f"TP: {tp:,}"
)


print(
    "\nBUSINESS RANKING METRICS"
)

print(
    f"Recall@Top10% : "
    f"{recall_at_top_10:.4f}"
)

print(
    f"Lift@Top10%   : "
    f"{lift_at_top_10:.2f}x"
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 11 COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nGenerated files:"
)

print(
    f"- "
    f"{MODEL_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{METRICS_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{VALID_PRED_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{REPORT_FILE.relative_to(PROJECT_ROOT)}"
)


print(
    "\nThe TEST set remains untouched."
)