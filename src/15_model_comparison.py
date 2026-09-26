from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 15
# MODEL COMPARISON
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. MODEL FILE INFORMATION
# ------------------------------------------------------------

models = {
    "Logistic Regression": {
        "metrics":
            REPORT_DIR
            / "11_logistic_regression_metrics.csv",

        "predictions":
            REPORT_DIR
            / "11_logistic_validation_predictions.csv",
    },

    "Gradient Boosting": {
        "metrics":
            REPORT_DIR
            / "12_gradient_boosting_metrics.csv",

        "predictions":
            REPORT_DIR
            / "12_gradient_boosting_validation_predictions.csv",
    },

    "XGBoost": {
        "metrics":
            REPORT_DIR
            / "13_xgboost_metrics.csv",

        "predictions":
            REPORT_DIR
            / "13_xgboost_validation_predictions.csv",
    },

    "CatBoost": {
        "metrics":
            REPORT_DIR
            / "14_catboost_metrics.csv",

        "predictions":
            REPORT_DIR
            / "14_catboost_validation_predictions.csv",
    },
}


# ------------------------------------------------------------
# 3. CHECK FILES
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 15: MODEL COMPARISON")
print("=" * 90)


for model_name, paths in models.items():

    for file_type, file_path in paths.items():

        if not file_path.exists():

            raise FileNotFoundError(
                f"{model_name} "
                f"{file_type} file not found:\n"
                f"{file_path}"
            )


# ------------------------------------------------------------
# 4. CALCULATE TOP-10% METRICS
# ------------------------------------------------------------

def calculate_top10_metrics(
    prediction_file
):

    df = pd.read_csv(
        prediction_file
    )

    df = (
        df
        .sort_values(
            "churn_probability",
            ascending=False
        )
        .reset_index(drop=True)
    )

    total_rows = len(df)

    top_n = int(
        np.ceil(
            total_rows * 0.10
        )
    )

    top_df = df.head(
        top_n
    )

    total_churners = int(
        df[
            "actual_churn"
        ].sum()
    )

    top_churners = int(
        top_df[
            "actual_churn"
        ].sum()
    )

    recall_top10 = (
        top_churners
        / total_churners
        if total_churners > 0
        else 0
    )

    overall_rate = (
        df[
            "actual_churn"
        ].mean()
    )

    top_rate = (
        top_df[
            "actual_churn"
        ].mean()
    )

    lift_top10 = (
        top_rate
        / overall_rate
        if overall_rate > 0
        else 0
    )

    return {
        "top10_customers":
            top_n,

        "top10_churners":
            top_churners,

        "recall_at_top10":
            recall_top10,

        "lift_at_top10":
            lift_top10,
    }


# ------------------------------------------------------------
# 5. BUILD COMPARISON TABLE
# ------------------------------------------------------------

comparison_rows = []


for model_name, paths in models.items():

    metrics_df = pd.read_csv(
        paths["metrics"]
    )

    train_row = (
        metrics_df[
            metrics_df[
                "dataset"
            ] == "Train"
        ]
        .iloc[0]
    )

    valid_row = (
        metrics_df[
            metrics_df[
                "dataset"
            ] == "Validation"
        ]
        .iloc[0]
    )


    top10 = calculate_top10_metrics(
        paths[
            "predictions"
        ]
    )


    comparison_rows.append(
        {
            "model":
                model_name,

            "validation_accuracy":
                valid_row[
                    "accuracy"
                ],

            "validation_precision":
                valid_row[
                    "precision"
                ],

            "validation_recall":
                valid_row[
                    "recall"
                ],

            "validation_f1":
                valid_row[
                    "f1_score"
                ],

            "validation_roc_auc":
                valid_row[
                    "roc_auc"
                ],

            "validation_pr_auc":
                valid_row[
                    "pr_auc"
                ],

            "recall_at_top10":
                top10[
                    "recall_at_top10"
                ],

            "lift_at_top10":
                top10[
                    "lift_at_top10"
                ],

            "top10_churners":
                top10[
                    "top10_churners"
                ],

            "roc_auc_train_gap":
                train_row[
                    "roc_auc"
                ]
                -
                valid_row[
                    "roc_auc"
                ],

            "pr_auc_train_gap":
                train_row[
                    "pr_auc"
                ]
                -
                valid_row[
                    "pr_auc"
                ],

            "recall_train_gap":
                train_row[
                    "recall"
                ]
                -
                valid_row[
                    "recall"
                ],
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


numeric_cols = (
    comparison_df
    .select_dtypes(
        include="number"
    )
    .columns
)


comparison_df[
    numeric_cols
] = (
    comparison_df[
        numeric_cols
    ]
    .round(4)
)


# ------------------------------------------------------------
# 6. SAVE COMPARISON CSV
# ------------------------------------------------------------

COMPARISON_FILE = (
    REPORT_DIR
    / "15_model_comparison.csv"
)


comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)


# ------------------------------------------------------------
# 7. PRINT COMPARISON
# ------------------------------------------------------------

display_columns = [
    "model",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_roc_auc",
    "validation_pr_auc",
    "recall_at_top10",
    "lift_at_top10",
]


print(
    "\n" + "=" * 90
)

print(
    "VALIDATION MODEL COMPARISON"
)

print(
    "=" * 90
)

print(
    comparison_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 8. OVERFITTING REVIEW
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "TRAIN - VALIDATION PERFORMANCE GAPS"
)

print(
    "=" * 90
)


gap_columns = [
    "model",
    "roc_auc_train_gap",
    "pr_auc_train_gap",
    "recall_train_gap",
]


print(
    comparison_df[
        gap_columns
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 9. CHART 1
# VALIDATION QUALITY METRICS
# ------------------------------------------------------------

chart_metrics = [
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_roc_auc",
    "validation_pr_auc",
]


chart_df = (
    comparison_df
    .set_index(
        "model"
    )[
        chart_metrics
    ]
)


ax = chart_df.plot(
    kind="bar",
    figsize=(12, 7)
)

ax.set_title(
    "Model Comparison - Validation Metrics"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()


CHART_1 = (
    CHART_DIR
    / "12_model_validation_metrics.png"
)


plt.savefig(
    CHART_1,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. CHART 2
# BUSINESS RANKING METRICS
# ------------------------------------------------------------

ranking_df = (
    comparison_df
    .set_index(
        "model"
    )[
        [
            "recall_at_top10",
            "lift_at_top10",
        ]
    ]
)


fig, ax1 = plt.subplots(
    figsize=(11, 7)
)


x = np.arange(
    len(
        ranking_df
    )
)


bars = ax1.bar(
    x,
    ranking_df[
        "recall_at_top10"
    ]
)


ax1.set_ylabel(
    "Recall @ Top 10%"
)

ax1.set_ylim(
    0,
    1
)

ax1.set_xticks(
    x
)

ax1.set_xticklabels(
    ranking_df.index,
    rotation=20,
    ha="right"
)


ax2 = ax1.twinx()


ax2.plot(
    x,
    ranking_df[
        "lift_at_top10"
    ],
    marker="o"
)


ax2.set_ylabel(
    "Lift @ Top 10x"
)


ax1.set_title(
    "Business Ranking Performance"
)


plt.tight_layout()


CHART_2 = (
    CHART_DIR
    / "13_model_business_ranking.png"
)


plt.savefig(
    CHART_2,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 11. CHART 3
# ROC-AUC TRAIN/VALIDATION GAP
# ------------------------------------------------------------

gap_chart = (
    comparison_df[
        [
            "model",
            "roc_auc_train_gap",
        ]
    ]
    .set_index(
        "model"
    )
)


ax = gap_chart.plot(
    kind="bar",
    legend=False,
    figsize=(10, 6)
)


ax.set_title(
    "ROC-AUC Train vs Validation Gap"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Train ROC-AUC - Validation ROC-AUC"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()


CHART_3 = (
    CHART_DIR
    / "14_model_overfitting_gap.png"
)


plt.savefig(
    CHART_3,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 12. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - STEP 15 MODEL COMPARISON"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "MODELS COMPARED"
)

report.append(
    "-" * 100
)

report.append(
    "1. Logistic Regression"
)

report.append(
    "2. Gradient Boosting"
)

report.append(
    "3. XGBoost"
)

report.append(
    "4. CatBoost"
)

report.append("")

report.append(
    "VALIDATION RESULTS"
)

report.append(
    "-" * 100
)

report.append(
    comparison_df[
        display_columns
    ].to_string(
        index=False
    )
)

report.append("")

report.append(
    "OVERFITTING REVIEW"
)

report.append(
    "-" * 100
)

report.append(
    comparison_df[
        gap_columns
    ].to_string(
        index=False
    )
)

report.append("")

report.append(
    "INTERPRETATION"
)

report.append(
    "-" * 100
)

report.append(
    "Logistic Regression provides strong recall, "
    "ROC-AUC and top-decile capture."
)

report.append(
    "Gradient Boosting improves precision and F1 "
    "relative to Logistic Regression, but shows "
    "a larger train-validation gap."
)

report.append(
    "XGBoost gives the highest precision and F1 "
    "at the default 0.50 threshold, but also shows "
    "the strongest evidence of overfitting among "
    "the initial models."
)

report.append(
    "CatBoost provides the strongest validation "
    "PR-AUC among the four initial models and "
    "handles categorical variables natively."
)

report.append("")

report.append(
    "No final champion is selected in this step."
)

report.append(
    "The next stage should tune promising models "
    "and optimise the classification threshold "
    "using validation data only."
)

report.append(
    "The final test set remains untouched."
)


REPORT_FILE = (
    REPORT_DIR
    / "15_model_comparison_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 13. FINAL OUTPUT
# ------------------------------------------------------------

print(
    "\nGenerated files:"
)

print(
    f"- {COMPARISON_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {REPORT_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {CHART_1.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {CHART_2.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {CHART_3.relative_to(PROJECT_ROOT)}"
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 15 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nThe TEST set remains untouched."
)