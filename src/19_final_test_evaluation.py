from pathlib import Path
import joblib
import numpy as np
import pandas as pd

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
# DAY 2 - STEP 19
# FINAL UNTOUCHED TEST EVALUATION
# ============================================================


# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

FINAL_THRESHOLD = 0.49


# ------------------------------------------------------------
# 2. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DATA_DIR = PROJECT_ROOT / "data" / "modeling"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_FILE = (
    MODEL_DATA_DIR
    / "test_20.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "catboost_model_package.pkl"
)


# ------------------------------------------------------------
# 3. START
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 19: FINAL TEST EVALUATION")
print("=" * 90)


if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"Test file not found:\n"
        f"{TEST_FILE}"
    )


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Model file not found:\n"
        f"{MODEL_FILE}"
    )


# ------------------------------------------------------------
# 4. LOAD FINAL MODEL PACKAGE
# ------------------------------------------------------------

model_package = joblib.load(
    MODEL_FILE
)


model = model_package[
    "model"
]

numeric_features = model_package[
    "numeric_features"
]

categorical_features = model_package[
    "categorical_features"
]

numeric_medians = model_package[
    "numeric_medians"
]

TARGET = model_package[
    "target"
]


print(
    "\nFinal model     : "
    "Original CatBoost"
)

print(
    f"Final threshold : "
    f"{FINAL_THRESHOLD:.2f}"
)


# ------------------------------------------------------------
# 5. LOAD TEST DATA
# ------------------------------------------------------------

test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"\nTest rows       : "
    f"{len(test_df):,}"
)

print(
    f"Test columns    : "
    f"{test_df.shape[1]}"
)


X_test = test_df.drop(
    columns=[TARGET]
).copy()

y_test = (
    test_df[
        TARGET
    ]
    .astype(int)
)


# ------------------------------------------------------------
# 6. APPLY TRAINING-DERIVED PREPROCESSING
# ------------------------------------------------------------

# IMPORTANT:
# No values are learned from the test data.
# We reuse medians learned during training.

for col in categorical_features:

    X_test[col] = (
        X_test[col]
        .fillna("Missing")
        .astype(str)
    )


for col in numeric_features:

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )

    X_test[col] = (
        X_test[col]
        .fillna(
            numeric_medians[
                col
            ]
        )
    )


# ------------------------------------------------------------
# 7. PREDICT PROBABILITIES
# ------------------------------------------------------------

test_probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)


# ------------------------------------------------------------
# 8. APPLY FROZEN THRESHOLD
# ------------------------------------------------------------

test_predictions = (
    test_probabilities
    >= FINAL_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 9. METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


# ------------------------------------------------------------
# 10. CONFUSION MATRIX
# ------------------------------------------------------------

tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )
    .ravel()
)


# ------------------------------------------------------------
# 11. TOP-10% BUSINESS METRICS
# ------------------------------------------------------------

scores = pd.DataFrame(
    {
        "actual_churn":
            y_test.values,

        "churn_probability":
            test_probabilities,
    }
)


scores = (
    scores
    .sort_values(
        "churn_probability",
        ascending=False
    )
    .reset_index(drop=True)
)


top_10_count = int(
    np.ceil(
        len(scores)
        * 0.10
    )
)


top_10 = scores.head(
    top_10_count
)


total_churners = int(
    scores[
        "actual_churn"
    ].sum()
)


top_10_churners = int(
    top_10[
        "actual_churn"
    ].sum()
)


recall_at_top10 = (
    top_10_churners
    / total_churners
    if total_churners > 0
    else 0
)


overall_churn_rate = (
    scores[
        "actual_churn"
    ].mean()
)


top10_churn_rate = (
    top_10[
        "actual_churn"
    ].mean()
)


lift_at_top10 = (
    top10_churn_rate
    / overall_churn_rate
    if overall_churn_rate > 0
    else 0
)


# ------------------------------------------------------------
# 12. SUMMARY TABLE
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    [
        {
            "model":
                "Original CatBoost",

            "threshold":
                FINAL_THRESHOLD,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "recall_at_top10":
                recall_at_top10,

            "lift_at_top10":
                lift_at_top10,

            "true_negatives":
                tn,

            "false_positives":
                fp,

            "false_negatives":
                fn,

            "true_positives":
                tp,
        }
    ]
)


metric_columns = [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "recall_at_top10",
    "lift_at_top10",
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
# 13. SAVE METRICS
# ------------------------------------------------------------

METRICS_FILE = (
    REPORT_DIR
    / "19_final_test_metrics.csv"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ------------------------------------------------------------
# 14. SAVE TEST PREDICTIONS
# ------------------------------------------------------------

prediction_output = X_test.copy()

prediction_output[
    "actual_churn"
] = y_test.values

prediction_output[
    "churn_probability"
] = test_probabilities

prediction_output[
    "predicted_churn"
] = test_predictions


PREDICTION_FILE = (
    REPORT_DIR
    / "19_final_test_predictions.csv"
)


prediction_output.to_csv(
    PREDICTION_FILE,
    index=False
)


# ------------------------------------------------------------
# 15. CLASSIFICATION REPORT
# ------------------------------------------------------------

classification_text = (
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 16. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - STEP 19 FINAL TEST EVALUATION"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "FINAL FROZEN CONFIGURATION"
)

report.append(
    "-" * 100
)

report.append(
    "Model: Original CatBoost"
)

report.append(
    f"Threshold: "
    f"{FINAL_THRESHOLD:.2f}"
)

report.append(
    "Model selected using validation PR-AUC."
)

report.append(
    "Threshold selected using validation data "
    "with the business constraint of maintaining "
    "at least 80% recall."
)

report.append("")

report.append(
    "FINAL TEST METRICS"
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
    "CONFUSION MATRIX"
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
    "BUSINESS RANKING PERFORMANCE"
)

report.append(
    "-" * 100
)

report.append(
    f"Test customers: "
    f"{len(y_test):,}"
)

report.append(
    f"Actual churners: "
    f"{total_churners:,}"
)

report.append(
    f"Top 10% customers: "
    f"{top_10_count:,}"
)

report.append(
    f"Churners captured in top 10%: "
    f"{top_10_churners:,}"
)

report.append(
    f"Recall@Top10%: "
    f"{recall_at_top10:.4f}"
)

report.append(
    f"Lift@Top10%: "
    f"{lift_at_top10:.2f}x"
)

report.append("")

report.append(
    "IMPORTANT"
)

report.append(
    "-" * 100
)

report.append(
    "This is the first and final evaluation "
    "of the frozen champion configuration "
    "on the untouched test set."
)

report.append(
    "The test results must not be used to "
    "retune the model or classification threshold."
)


REPORT_FILE = (
    REPORT_DIR
    / "19_final_test_evaluation_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 17. TERMINAL OUTPUT
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "FINAL TEST RESULTS"
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
    "\nCONFUSION MATRIX"
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
    f"{recall_at_top10:.4f}"
)

print(
    f"Lift@Top10%   : "
    f"{lift_at_top10:.2f}x"
)


print(
    "\nGenerated files:"
)

print(
    f"- "
    f"{METRICS_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{PREDICTION_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{REPORT_FILE.relative_to(PROJECT_ROOT)}"
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 19 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nDo NOT tune the model using these "
    "test results."
)