from pathlib import Path
import pandas as pd


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 18
# FINAL VALIDATION COMPARISON
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 18: FINAL VALIDATION COMPARISON")
print("=" * 90)


# ------------------------------------------------------------
# 1. LOAD ORIGINAL MODEL RESULTS
# ------------------------------------------------------------

original_files = {
    "Logistic Regression":
        REPORT_DIR
        / "11_logistic_regression_metrics.csv",

    "Gradient Boosting":
        REPORT_DIR
        / "12_gradient_boosting_metrics.csv",

    "Original XGBoost":
        REPORT_DIR
        / "13_xgboost_metrics.csv",

    "Original CatBoost":
        REPORT_DIR
        / "14_catboost_metrics.csv",
}


rows = []


for model_name, file_path in original_files.items():

    metrics = pd.read_csv(
        file_path
    )

    train = (
        metrics[
            metrics["dataset"] == "Train"
        ]
        .iloc[0]
    )

    valid = (
        metrics[
            metrics["dataset"] == "Validation"
        ]
        .iloc[0]
    )

    rows.append(
        {
            "model":
                model_name,

            "source":
                "Original",

            "precision":
                valid["precision"],

            "recall":
                valid["recall"],

            "f1":
                valid["f1_score"],

            "roc_auc":
                valid["roc_auc"],

            "pr_auc":
                valid["pr_auc"],

            "roc_auc_gap":
                (
                    train["roc_auc"]
                    -
                    valid["roc_auc"]
                ),

            "pr_auc_gap":
                (
                    train["pr_auc"]
                    -
                    valid["pr_auc"]
                ),
        }
    )


# ------------------------------------------------------------
# 2. LOAD TUNING RESULTS
# ------------------------------------------------------------

TUNING_FILE = (
    REPORT_DIR
    / "17_targeted_model_tuning_results.csv"
)


tuning = pd.read_csv(
    TUNING_FILE
)


for _, row in tuning.iterrows():

    rows.append(
        {
            "model":
                row["configuration"],

            "source":
                "Tuned",

            "precision":
                row["validation_precision"],

            "recall":
                row["validation_recall"],

            "f1":
                row["validation_f1"],

            "roc_auc":
                row["validation_roc_auc"],

            "pr_auc":
                row["validation_pr_auc"],

            "roc_auc_gap":
                row["roc_auc_gap"],

            "pr_auc_gap":
                row["pr_auc_gap"],
        }
    )


comparison = pd.DataFrame(
    rows
)


# ------------------------------------------------------------
# 3. ADD BUSINESS RANKING RESULTS
# ------------------------------------------------------------

comparison_15 = pd.read_csv(
    REPORT_DIR
    / "15_model_comparison.csv"
)


ranking_lookup = {

    "Logistic Regression":
        "Logistic Regression",

    "Gradient Boosting":
        "Gradient Boosting",

    "Original XGBoost":
        "XGBoost",

    "Original CatBoost":
        "CatBoost",
}


comparison[
    "recall_at_top10"
] = pd.NA

comparison[
    "lift_at_top10"
] = pd.NA


for i, row in comparison.iterrows():

    model_name = row["model"]

    if model_name in ranking_lookup:

        source_name = (
            ranking_lookup[
                model_name
            ]
        )

        source_row = (
            comparison_15[
                comparison_15[
                    "model"
                ] == source_name
            ]
            .iloc[0]
        )

        comparison.loc[
            i,
            "recall_at_top10"
        ] = source_row[
            "recall_at_top10"
        ]

        comparison.loc[
            i,
            "lift_at_top10"
        ] = source_row[
            "lift_at_top10"
        ]


# Add ranking metrics for tuned models

for i, row in comparison.iterrows():

    if row["source"] == "Tuned":

        tune_row = (
            tuning[
                tuning[
                    "configuration"
                ] == row["model"]
            ]
            .iloc[0]
        )

        comparison.loc[
            i,
            "recall_at_top10"
        ] = tune_row[
            "recall_at_top10"
        ]

        comparison.loc[
            i,
            "lift_at_top10"
        ] = tune_row[
            "lift_at_top10"
        ]


comparison[
    "recall_at_top10"
] = pd.to_numeric(
    comparison[
        "recall_at_top10"
    ]
)

comparison[
    "lift_at_top10"
] = pd.to_numeric(
    comparison[
        "lift_at_top10"
    ]
)


# ------------------------------------------------------------
# 4. SORT BY PR-AUC
# ------------------------------------------------------------

comparison = (
    comparison
    .sort_values(
        [
            "pr_auc",
            "roc_auc",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .reset_index(
        drop=True
    )
)


numeric_columns = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "recall_at_top10",
    "lift_at_top10",
    "roc_auc_gap",
    "pr_auc_gap",
]


comparison[
    numeric_columns
] = (
    comparison[
        numeric_columns
    ]
    .round(4)
)


# ------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------

OUTPUT_FILE = (
    REPORT_DIR
    / "18_final_validation_comparison.csv"
)


comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 6. SHORTLIST
# ------------------------------------------------------------

shortlist_names = [

    "Logistic Regression",

    "Original CatBoost",

    "CAT_StrongReg_HalfWeight",

    "XGB_Regularized_FullWeight",
]


shortlist = (
    comparison[
        comparison[
            "model"
        ]
        .isin(
            shortlist_names
        )
    ]
    .copy()
)


SHORTLIST_FILE = (
    REPORT_DIR
    / "18_model_shortlist.csv"
)


shortlist.to_csv(
    SHORTLIST_FILE,
    index=False
)


# ------------------------------------------------------------
# 7. TERMINAL OUTPUT
# ------------------------------------------------------------

print(
    "\nALL ORIGINAL + TUNED MODELS"
)

print(
    "-" * 90
)


display_cols = [
    "model",
    "source",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "recall_at_top10",
    "lift_at_top10",
    "roc_auc_gap",
]


print(
    comparison[
        display_cols
    ].to_string(
        index=False
    )
)


print(
    "\n" + "=" * 90
)

print(
    "FINAL VALIDATION SHORTLIST"
)

print(
    "=" * 90
)


print(
    shortlist[
        display_cols
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 8. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - STEP 18 FINAL VALIDATION COMPARISON"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "ALL MODEL RESULTS"
)

report.append(
    "-" * 100
)

report.append(
    comparison[
        display_cols
    ].to_string(
        index=False
    )
)

report.append("")

report.append(
    "FINAL SHORTLIST"
)

report.append(
    "-" * 100
)

report.append(
    shortlist[
        display_cols
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
    "Logistic Regression remains the most stable "
    "baseline and provides strong ROC-AUC and "
    "top-decile churn capture."
)

report.append(
    "Original CatBoost retains the strongest "
    "validation PR-AUC among the initial and "
    "tuned candidates."
)

report.append(
    "CAT_StrongReg_HalfWeight provides a useful "
    "balance of ranking performance, F1 and "
    "reduced overfitting."
)

report.append(
    "Regularized XGBoost materially improves "
    "on the original XGBoost by reducing the "
    "train-validation gap and increasing PR-AUC."
)

report.append("")

report.append(
    "No test data has been used."
)

report.append(
    "Final model selection and threshold choice "
    "must be completed before evaluating the "
    "untouched test set."
)


REPORT_FILE = (
    REPORT_DIR
    / "18_final_validation_comparison_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


print(
    "\nGenerated files:"
)

print(
    f"- "
    f"{OUTPUT_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{SHORTLIST_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{REPORT_FILE.relative_to(PROJECT_ROOT)}"
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 18 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nThe TEST set remains untouched."
)