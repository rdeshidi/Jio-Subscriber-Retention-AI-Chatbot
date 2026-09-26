from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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

from xgboost import XGBClassifier


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 13
# XGBOOST MODEL
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DATA_DIR = PROJECT_ROOT / "data" / "modeling"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = MODEL_DATA_DIR / "train_60.csv"
VALID_FILE = MODEL_DATA_DIR / "validation_20.csv"
METADATA_FILE = MODEL_DATA_DIR / "modeling_metadata.json"


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 13: XGBOOST")
print("=" * 90)

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

TARGET = metadata["target"]

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]

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
# 4. CLASS IMBALANCE RATIO
# ------------------------------------------------------------

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative_count
    / positive_count
)

print(
    f"\nTraining non-churners : "
    f"{negative_count:,}"
)

print(
    f"Training churners     : "
    f"{positive_count:,}"
)

print(
    f"scale_pos_weight      : "
    f"{scale_pos_weight:.2f}"
)


# ------------------------------------------------------------
# 5. PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
    ]
)

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
# 6. XGBOOST MODEL
# ------------------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


# ------------------------------------------------------------
# 7. PIPELINE
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            xgb_model
        ),
    ]
)


# ------------------------------------------------------------
# 8. TRAIN MODEL
# ------------------------------------------------------------

print(
    "\nTraining XGBoost..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "[OK] XGBoost training complete."
)


# ------------------------------------------------------------
# 9. PREDICTIONS
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
# 10. METRICS
# ------------------------------------------------------------

def calculate_metrics(
    dataset_name,
    y_true,
    predictions,
    probabilities
):

    return {
        "dataset": dataset_name,

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
] = metrics_df[
    metric_columns
].round(4)


# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

tn, fp, fn, tp = (
    confusion_matrix(
        y_valid,
        valid_predictions
    )
    .ravel()
)


# ------------------------------------------------------------
# 12. TOP-10% BUSINESS METRICS
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
    .reset_index(drop=True)
)


top_10_count = int(
    np.ceil(
        len(validation_scores)
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
)

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
)


# ------------------------------------------------------------
# 13. FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_names = (
    pipeline.named_steps[
        "preprocessor"
    ]
    .get_feature_names_out()
)

feature_importances = (
    pipeline.named_steps[
        "model"
    ]
    .feature_importances_
)

importance_df = pd.DataFrame(
    {
        "feature":
            feature_names,

        "importance":
            feature_importances,
    }
)

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 14. SAVE OUTPUTS
# ------------------------------------------------------------

METRICS_FILE = (
    REPORT_DIR
    / "13_xgboost_metrics.csv"
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


IMPORTANCE_FILE = (
    REPORT_DIR
    / "13_xgboost_feature_importance.csv"
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


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
    / "13_xgboost_validation_predictions.csv"
)

validation_output.to_csv(
    VALID_PRED_FILE,
    index=False
)


MODEL_FILE = (
    MODEL_DIR
    / "xgboost_model.pkl"
)

joblib.dump(
    pipeline,
    MODEL_FILE
)


# ------------------------------------------------------------
# 15. REPORT
# ------------------------------------------------------------

classification_text = (
    classification_report(
        y_valid,
        valid_predictions,
        digits=4,
        zero_division=0
    )
)


report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - STEP 13 XGBOOST MODEL"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    f"scale_pos_weight = "
    f"{scale_pos_weight:.2f}"
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
    "CLASSIFICATION REPORT"
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
    f"Recall@Top10%: "
    f"{recall_at_top_10:.4f}"
)

report.append(
    f"Lift@Top10%: "
    f"{lift_at_top_10:.2f}x"
)

report.append("")

report.append(
    "TOP 15 FEATURE IMPORTANCES"
)

report.append(
    "-" * 100
)

report.append(
    importance_df.head(15).to_string(
        index=False
    )
)

report.append("")

report.append(
    "The final test set was NOT used."
)


REPORT_FILE = (
    REPORT_DIR
    / "13_xgboost_report.txt"
)

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 16. TERMINAL OUTPUT
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
    "\nTOP 10 FEATURE IMPORTANCES"
)

print(
    importance_df.head(10).to_string(
        index=False
    )
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 13 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nGenerated files:"
)

print(
    f"- {MODEL_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {METRICS_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {VALID_PRED_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {IMPORTANCE_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {REPORT_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    "\nThe TEST set remains untouched."
)