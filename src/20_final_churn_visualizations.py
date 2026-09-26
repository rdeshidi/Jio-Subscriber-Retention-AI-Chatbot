from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 20
# FINAL CHURN MODEL VISUALISATIONS
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DATA_DIR = PROJECT_ROOT / "data" / "modeling"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TEST_PRED_FILE = (
    REPORT_DIR
    / "19_final_test_predictions.csv"
)

TEST_METRICS_FILE = (
    REPORT_DIR
    / "19_final_test_metrics.csv"
)

VALIDATION_METRICS_FILE = (
    REPORT_DIR
    / "14_catboost_metrics.csv"
)

THRESHOLD_FILE = (
    REPORT_DIR
    / "16_threshold_optimization_summary.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "catboost_model_package.pkl"
)

TEST_FILE = (
    MODEL_DATA_DIR
    / "test_20.csv"
)


# ------------------------------------------------------------
# 2. START
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 20: FINAL CHURN VISUALISATIONS")
print("=" * 90)


required_files = [
    TEST_PRED_FILE,
    TEST_METRICS_FILE,
    VALIDATION_METRICS_FILE,
    THRESHOLD_FILE,
    MODEL_FILE,
    TEST_FILE,
]


for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )


# ------------------------------------------------------------
# 3. LOAD FINAL TEST RESULTS
# ------------------------------------------------------------

pred_df = pd.read_csv(
    TEST_PRED_FILE
)

test_metrics = pd.read_csv(
    TEST_METRICS_FILE
).iloc[0]


y_test = (
    pred_df[
        "actual_churn"
    ]
    .astype(int)
    .values
)

test_probability = (
    pred_df[
        "churn_probability"
    ]
    .values
)

test_prediction = (
    pred_df[
        "predicted_churn"
    ]
    .astype(int)
    .values
)


# ------------------------------------------------------------
# 4. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp.plot(
    ax=ax,
    values_format=",d"
)

ax.set_title(
    "Final CatBoost - Test Confusion Matrix"
)

plt.tight_layout()


CONFUSION_FILE = (
    CHART_DIR
    / "17_final_confusion_matrix.png"
)


plt.savefig(
    CONFUSION_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 5. ROC CURVE
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(
    y_test,
    test_probability
)


test_roc_auc = roc_auc_score(
    y_test,
    test_probability
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    fpr,
    tpr,
    label=(
        f"CatBoost "
        f"(AUC = {test_roc_auc:.3f})"
    )
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Final CatBoost - ROC Curve"
)

plt.legend()

plt.tight_layout()


ROC_FILE = (
    CHART_DIR
    / "18_final_roc_curve.png"
)


plt.savefig(
    ROC_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 6. PRECISION-RECALL CURVE
# ------------------------------------------------------------

precision_values, recall_values, _ = (
    precision_recall_curve(
        y_test,
        test_probability
    )
)


test_pr_auc = (
    average_precision_score(
        y_test,
        test_probability
    )
)


baseline = (
    np.mean(
        y_test
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    recall_values,
    precision_values,
    label=(
        f"CatBoost "
        f"(PR-AUC = {test_pr_auc:.3f})"
    )
)


plt.axhline(
    baseline,
    linestyle="--",
    label=(
        f"Baseline = {baseline:.3f}"
    )
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Final CatBoost - Precision-Recall Curve"
)

plt.legend()

plt.tight_layout()


PR_FILE = (
    CHART_DIR
    / "19_final_precision_recall_curve.png"
)


plt.savefig(
    PR_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 7. DECILE / LIFT ANALYSIS
# ------------------------------------------------------------

risk_df = pd.DataFrame(
    {
        "actual_churn":
            y_test,

        "churn_probability":
            test_probability,
    }
)


risk_df = (
    risk_df
    .sort_values(
        "churn_probability",
        ascending=False
    )
    .reset_index(drop=True)
)


n = len(
    risk_df
)


risk_df[
    "risk_decile"
] = np.minimum(
    (
        np.floor(
            np.arange(n)
            * 10
            / n
        )
        + 1
    ).astype(int),
    10
)


overall_churn_rate = (
    risk_df[
        "actual_churn"
    ]
    .mean()
)


decile_df = (
    risk_df
    .groupby(
        "risk_decile",
        as_index=False
    )
    .agg(
        customers=(
            "actual_churn",
            "size"
        ),

        churners=(
            "actual_churn",
            "sum"
        ),

        churn_rate=(
            "actual_churn",
            "mean"
        ),

        average_score=(
            "churn_probability",
            "mean"
        ),
    )
)


decile_df[
    "lift"
] = (
    decile_df[
        "churn_rate"
    ]
    / overall_churn_rate
)


total_churners = (
    decile_df[
        "churners"
    ]
    .sum()
)


decile_df[
    "cumulative_churners"
] = (
    decile_df[
        "churners"
    ]
    .cumsum()
)


decile_df[
    "cumulative_capture"
] = (
    decile_df[
        "cumulative_churners"
    ]
    / total_churners
)


DECILE_FILE = (
    REPORT_DIR
    / "20_final_test_decile_analysis.csv"
)


decile_df.to_csv(
    DECILE_FILE,
    index=False
)


# ------------------------------------------------------------
# 8. LIFT BY RISK DECILE CHART
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.bar(
    decile_df[
        "risk_decile"
    ],
    decile_df[
        "lift"
    ]
)


plt.axhline(
    1.0,
    linestyle="--"
)


plt.xlabel(
    "Risk Decile (1 = Highest Risk)"
)

plt.ylabel(
    "Lift"
)

plt.title(
    "Final CatBoost - Lift by Risk Decile"
)

plt.xticks(
    range(1, 11)
)

plt.tight_layout()


LIFT_FILE = (
    CHART_DIR
    / "20_final_lift_by_decile.png"
)


plt.savefig(
    LIFT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 9. CUMULATIVE CHURN CAPTURE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    decile_df[
        "risk_decile"
    ]
    * 10,

    decile_df[
        "cumulative_capture"
    ]
    * 100,

    marker="o",
    label="Model"
)


plt.plot(
    [10, 100],
    [10, 100],
    linestyle="--",
    label="Random targeting"
)


plt.xlabel(
    "Percentage of Customers Targeted"
)

plt.ylabel(
    "Percentage of Churners Captured"
)

plt.title(
    "Final CatBoost - Cumulative Churn Capture"
)

plt.legend()

plt.tight_layout()


CAPTURE_FILE = (
    CHART_DIR
    / "21_final_cumulative_churn_capture.png"
)


plt.savefig(
    CAPTURE_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. LOAD FINAL CATBOOST MODEL
# ------------------------------------------------------------

model_package = joblib.load(
    MODEL_FILE
)


model = model_package[
    "model"
]


test_original = pd.read_csv(
    TEST_FILE
)


TARGET = model_package[
    "target"
]


X_test = test_original.drop(
    columns=[TARGET]
)


# ------------------------------------------------------------
# 11. FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance = (
    model.get_feature_importance()
)


feature_names = getattr(
    model,
    "feature_names_",
    None
)


if (
    feature_names is None
    or len(feature_names)
    != len(feature_importance)
):

    feature_names = (
        X_test.columns
        .tolist()
    )


importance_df = pd.DataFrame(
    {
        "feature":
            feature_names,

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


IMPORTANCE_FILE = (
    REPORT_DIR
    / "20_final_catboost_feature_importance.csv"
)


importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


top_15 = (
    importance_df
    .head(15)
    .sort_values(
        "importance",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top_15[
        "feature"
    ],
    top_15[
        "importance"
    ]
)


plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Final CatBoost - Top 15 Feature Importances"
)

plt.tight_layout()


IMPORTANCE_CHART = (
    CHART_DIR
    / "22_final_catboost_feature_importance.png"
)


plt.savefig(
    IMPORTANCE_CHART,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 12. VALIDATION METRICS
# ------------------------------------------------------------

validation_metrics_all = pd.read_csv(
    VALIDATION_METRICS_FILE
)


validation_rank_metrics = (
    validation_metrics_all[
        validation_metrics_all[
            "dataset"
        ] == "Validation"
    ]
    .iloc[0]
)


threshold_summary = pd.read_csv(
    THRESHOLD_FILE
)


validation_threshold_row = (
    threshold_summary[
        (
            threshold_summary[
                "model"
            ] == "CatBoost"
        )
        &
        (
            threshold_summary[
                "strategy"
            ] == "Recall >= 80%"
        )
    ]
    .iloc[0]
)


# ------------------------------------------------------------
# 13. VALIDATION VS TEST COMPARISON
# ------------------------------------------------------------

comparison_df = pd.DataFrame(
    [
        {
            "dataset":
                "Validation",

            "precision":
                validation_threshold_row[
                    "precision"
                ],

            "recall":
                validation_threshold_row[
                    "recall"
                ],

            "f1":
                validation_threshold_row[
                    "f1_score"
                ],

            "roc_auc":
                validation_rank_metrics[
                    "roc_auc"
                ],

            "pr_auc":
                validation_rank_metrics[
                    "pr_auc"
                ],

            "recall_at_top10":
                0.7378,

            "lift_at_top10":
                7.3766,
        },

        {
            "dataset":
                "Test",

            "precision":
                test_metrics[
                    "precision"
                ],

            "recall":
                test_metrics[
                    "recall"
                ],

            "f1":
                test_metrics[
                    "f1"
                ],

            "roc_auc":
                test_metrics[
                    "roc_auc"
                ],

            "pr_auc":
                test_metrics[
                    "pr_auc"
                ],

            "recall_at_top10":
                test_metrics[
                    "recall_at_top10"
                ],

            "lift_at_top10":
                test_metrics[
                    "lift_at_top10"
                ],
        },
    ]
)


COMPARISON_FILE = (
    REPORT_DIR
    / "20_validation_vs_test_comparison.csv"
)


comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)


chart_metrics = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "recall_at_top10",
]


comparison_plot = (
    comparison_df
    .set_index(
        "dataset"
    )[
        chart_metrics
    ]
)


ax = comparison_plot.plot(
    kind="bar",
    figsize=(11, 7)
)


ax.set_title(
    "Final CatBoost - Validation vs Test Performance"
)

ax.set_xlabel(
    "Dataset"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.tight_layout()


COMPARISON_CHART = (
    CHART_DIR
    / "23_validation_vs_test_comparison.png"
)


plt.savefig(
    COMPARISON_CHART,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 14. FINAL HUMAN-READABLE SUMMARY
# ------------------------------------------------------------

tn, fp, fn, tp = (
    cm.ravel()
)


top_decile = (
    decile_df[
        decile_df[
            "risk_decile"
        ] == 1
    ]
    .iloc[0]
)


report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - FINAL 30-DAY CHURN MODEL SUMMARY"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "FINAL MODEL"
)

report.append(
    "-" * 100
)

report.append(
    "Model: CatBoostClassifier"
)

report.append(
    "Target: churn_flag_30d"
)

report.append(
    "Operational threshold: 0.49"
)

report.append(
    "Model selected using validation PR-AUC."
)

report.append(
    "Threshold selected on validation data with "
    "a retention-oriented >=80% recall constraint."
)

report.append("")

report.append(
    "FINAL UNTOUCHED TEST PERFORMANCE"
)

report.append(
    "-" * 100
)

report.append(
    f"Accuracy: "
    f"{test_metrics['accuracy']:.4f}"
)

report.append(
    f"Precision: "
    f"{test_metrics['precision']:.4f}"
)

report.append(
    f"Recall: "
    f"{test_metrics['recall']:.4f}"
)

report.append(
    f"F1: "
    f"{test_metrics['f1']:.4f}"
)

report.append(
    f"ROC-AUC: "
    f"{test_metrics['roc_auc']:.4f}"
)

report.append(
    f"PR-AUC: "
    f"{test_metrics['pr_auc']:.4f}"
)

report.append(
    f"Recall@Top10%: "
    f"{test_metrics['recall_at_top10']:.4f}"
)

report.append(
    f"Lift@Top10%: "
    f"{test_metrics['lift_at_top10']:.2f}x"
)

report.append("")

report.append(
    "CONFUSION MATRIX"
)

report.append(
    "-" * 100
)

report.append(
    f"True negatives: "
    f"{tn:,}"
)

report.append(
    f"False positives: "
    f"{fp:,}"
)

report.append(
    f"False negatives: "
    f"{fn:,}"
)

report.append(
    f"True positives: "
    f"{tp:,}"
)

report.append("")

report.append(
    "BUSINESS INTERPRETATION"
)

report.append(
    "-" * 100
)

report.append(
    f"The highest-risk 10% of test customers "
    f"contained {int(top_decile['churners']):,} "
    f"actual churners."
)

report.append(
    f"Top-decile churn rate: "
    f"{top_decile['churn_rate'] * 100:.2f}%."
)

report.append(
    f"Overall test churn rate: "
    f"{overall_churn_rate * 100:.2f}%."
)

report.append(
    f"Top-decile lift: "
    f"{top_decile['lift']:.2f}x."
)

report.append(
    "The model therefore concentrates a large "
    "share of future churn into a much smaller "
    "customer group for targeted retention action."
)

report.append("")

report.append(
    "IMPORTANT MODEL DRIVERS"
)

report.append(
    "-" * 100
)

report.append(
    importance_df
    .head(10)
    .to_string(
        index=False
    )
)

report.append("")

report.append(
    "Feature importance shows which variables "
    "the model relied on most strongly. "
    "It does not establish that those variables "
    "cause churn."
)

report.append("")

report.append(
    "WORKFLOW COMPLETED"
)

report.append(
    "-" * 100
)

report.append(
    "1. Prepared a leakage-controlled modelling dataset."
)

report.append(
    "2. Created 60% train / 20% validation / "
    "20% untouched test splits."
)

report.append(
    "3. Trained Logistic Regression baseline."
)

report.append(
    "4. Trained Gradient Boosting, XGBoost "
    "and CatBoost candidates."
)

report.append(
    "5. Compared Precision, Recall, F1, "
    "ROC-AUC, PR-AUC, Recall@Top10% and Lift."
)

report.append(
    "6. Reviewed train-validation gaps "
    "for overfitting."
)

report.append(
    "7. Optimised classification thresholds "
    "using validation data only."
)

report.append(
    "8. Performed targeted XGBoost and "
    "CatBoost tuning."
)

report.append(
    "9. Selected the final CatBoost model "
    "using validation evidence."
)

report.append(
    "10. Froze the 0.49 threshold before "
    "opening the test set."
)

report.append(
    "11. Evaluated the frozen configuration "
    "once on the untouched test set."
)

report.append("")

report.append(
    "The test set was not used for model or "
    "threshold retuning."
)


REPORT_FILE = (
    REPORT_DIR
    / "20_final_churn_model_summary.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 15. COMPLETE
# ------------------------------------------------------------

print(
    "\nGenerated charts:"
)

for file in [
    CONFUSION_FILE,
    ROC_FILE,
    PR_FILE,
    LIFT_FILE,
    CAPTURE_FILE,
    IMPORTANCE_CHART,
    COMPARISON_CHART,
]:

    print(
        f"- {file.relative_to(PROJECT_ROOT)}"
    )


print(
    "\nGenerated reports:"
)

for file in [
    DECILE_FILE,
    IMPORTANCE_FILE,
    COMPARISON_FILE,
    REPORT_FILE,
]:

    print(
        f"- {file.relative_to(PROJECT_ROOT)}"
    )


print(
    "\n" + "=" * 90
)

print(
    "STEP 20 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\n30-DAY CHURN MODELLING WORKFLOW COMPLETE."
)