from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

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
# DAY 2 - STEP 14
# CATBOOST MODEL
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
print("DAY 2 - STEP 14: CATBOOST")
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
).copy()

y_train = train_df[
    TARGET
].astype(int)


X_valid = valid_df.drop(
    columns=[TARGET]
).copy()

y_valid = valid_df[
    TARGET
].astype(int)


# ------------------------------------------------------------
# 4. PREPARE CATEGORICAL COLUMNS
# ------------------------------------------------------------

# CatBoost can directly handle categorical features.
# Missing categorical values are converted to a label.

for col in categorical_features:

    X_train[col] = (
        X_train[col]
        .fillna("Missing")
        .astype(str)
    )

    X_valid[col] = (
        X_valid[col]
        .fillna("Missing")
        .astype(str)
    )


# ------------------------------------------------------------
# 5. NUMERIC MEDIAN IMPUTATION
# ------------------------------------------------------------

# Medians are learned ONLY from training data.

numeric_medians = {}

for col in numeric_features:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_valid[col] = pd.to_numeric(
        X_valid[col],
        errors="coerce"
    )

    median_value = X_train[col].median()

    numeric_medians[col] = (
        float(median_value)
        if pd.notna(median_value)
        else 0.0
    )

    X_train[col] = X_train[col].fillna(
        numeric_medians[col]
    )

    X_valid[col] = X_valid[col].fillna(
        numeric_medians[col]
    )


# ------------------------------------------------------------
# 6. MODEL
# ------------------------------------------------------------

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=100,
    allow_writing_files=False,
)


# ------------------------------------------------------------
# 7. TRAIN
# ------------------------------------------------------------

print(
    "\nTraining CatBoost..."
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=75,
)


print(
    "[OK] CatBoost training complete."
)

print(
    f"Best iteration: "
    f"{cat_model.get_best_iteration()}"
)


# ------------------------------------------------------------
# 8. PREDICTIONS
# ------------------------------------------------------------

train_predictions = cat_model.predict(
    X_train
).astype(int).ravel()

train_probabilities = (
    cat_model.predict_proba(
        X_train
    )[:, 1]
)


valid_predictions = cat_model.predict(
    X_valid
).astype(int).ravel()

valid_probabilities = (
    cat_model.predict_proba(
        X_valid
    )[:, 1]
)


# ------------------------------------------------------------
# 9. METRIC FUNCTION
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


# ------------------------------------------------------------
# 10. TRAIN + VALIDATION METRICS
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
# 12. TOP-10% RANKING METRICS
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
    if total_churners > 0
    else 0
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
    if overall_churn_rate > 0
    else 0
)


# ------------------------------------------------------------
# 13. FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance = (
    cat_model.get_feature_importance()
)


importance_df = pd.DataFrame(
    {
        "feature":
            X_train.columns,

        "importance":
            feature_importance,
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
# 14. SAVE VALIDATION PREDICTIONS
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
    / "14_catboost_validation_predictions.csv"
)


validation_output.to_csv(
    VALID_PRED_FILE,
    index=False
)


# ------------------------------------------------------------
# 15. SAVE METRICS
# ------------------------------------------------------------

METRICS_FILE = (
    REPORT_DIR
    / "14_catboost_metrics.csv"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ------------------------------------------------------------
# 16. SAVE FEATURE IMPORTANCE
# ------------------------------------------------------------

IMPORTANCE_FILE = (
    REPORT_DIR
    / "14_catboost_feature_importance.csv"
)


importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ------------------------------------------------------------
# 17. SAVE MODEL + PREPROCESSING INFORMATION
# ------------------------------------------------------------

MODEL_FILE = (
    MODEL_DIR
    / "catboost_model_package.pkl"
)


model_package = {
    "model": cat_model,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "numeric_medians": numeric_medians,
    "target": TARGET,
}


joblib.dump(
    model_package,
    MODEL_FILE
)


# ------------------------------------------------------------
# 18. CLASSIFICATION REPORT
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
# 19. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - STEP 14 CATBOOST MODEL"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "Model: CatBoostClassifier"
)

report.append(
    "Categorical features handled natively."
)

report.append(
    "Class imbalance: auto_class_weights='Balanced'"
)

report.append(
    f"Best iteration: "
    f"{cat_model.get_best_iteration()}"
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
    "IMPORTANT"
)

report.append(
    "-" * 100
)

report.append(
    "The final test set was NOT used."
)

report.append(
    "CatBoost used validation data only for "
    "early stopping and model evaluation."
)


REPORT_FILE = (
    REPORT_DIR
    / "14_catboost_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 20. TERMINAL OUTPUT
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
    "STEP 14 COMPLETE"
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