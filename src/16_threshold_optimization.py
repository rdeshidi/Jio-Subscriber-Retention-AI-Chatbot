from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 16
# CLASSIFICATION THRESHOLD OPTIMIZATION
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
# 2. MODEL PREDICTION FILES
# ------------------------------------------------------------

models = {

    "Logistic Regression":
        REPORT_DIR
        / "11_logistic_validation_predictions.csv",

    "Gradient Boosting":
        REPORT_DIR
        / "12_gradient_boosting_validation_predictions.csv",

    "XGBoost":
        REPORT_DIR
        / "13_xgboost_validation_predictions.csv",

    "CatBoost":
        REPORT_DIR
        / "14_catboost_validation_predictions.csv",
}


# ------------------------------------------------------------
# 3. SETTINGS
# ------------------------------------------------------------

thresholds = np.arange(
    0.05,
    0.951,
    0.01
)

HIGH_RECALL_TARGET = 0.80


# ------------------------------------------------------------
# 4. START
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 16: THRESHOLD OPTIMIZATION")
print("=" * 90)


# ------------------------------------------------------------
# 5. EVALUATE THRESHOLDS
# ------------------------------------------------------------

all_threshold_results = []


for model_name, prediction_file in models.items():

    if not prediction_file.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{prediction_file}"
        )


    df = pd.read_csv(
        prediction_file
    )


    y_true = (
        df[
            "actual_churn"
        ]
        .astype(int)
        .values
    )


    probabilities = (
        df[
            "churn_probability"
        ]
        .values
    )


    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)


        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1]
            )
            .ravel()
        )


        all_threshold_results.append(
            {
                "model":
                    model_name,

                "threshold":
                    round(
                        float(threshold),
                        2
                    ),

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

                "tn":
                    int(tn),

                "fp":
                    int(fp),

                "fn":
                    int(fn),

                "tp":
                    int(tp),
            }
        )


threshold_df = pd.DataFrame(
    all_threshold_results
)


# ------------------------------------------------------------
# 6. SAVE FULL THRESHOLD SEARCH
# ------------------------------------------------------------

THRESHOLD_FILE = (
    REPORT_DIR
    / "16_all_threshold_results.csv"
)


threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False
)


# ------------------------------------------------------------
# 7. BEST F1 THRESHOLD FOR EACH MODEL
# ------------------------------------------------------------

best_f1_rows = []


for model_name in models:

    model_results = (
        threshold_df[
            threshold_df[
                "model"
            ] == model_name
        ]
        .copy()
    )


    best_index = (
        model_results[
            "f1_score"
        ]
        .idxmax()
    )


    best_row = (
        model_results
        .loc[
            best_index
        ]
        .copy()
    )


    best_row[
        "strategy"
    ] = "Maximum F1"


    best_f1_rows.append(
        best_row
    )


best_f1_df = pd.DataFrame(
    best_f1_rows
)


# ------------------------------------------------------------
# 8. HIGH-RECALL THRESHOLD
# ------------------------------------------------------------

high_recall_rows = []


for model_name in models:

    model_results = (
        threshold_df[
            threshold_df[
                "model"
            ] == model_name
        ]
        .copy()
    )


    eligible = (
        model_results[
            model_results[
                "recall"
            ] >= HIGH_RECALL_TARGET
        ]
        .copy()
    )


    if len(eligible) > 0:

        # Among thresholds that maintain at least
        # 80% recall, choose the one with
        # highest precision.

        best_index = (
            eligible[
                "precision"
            ]
            .idxmax()
        )


        best_row = (
            eligible
            .loc[
                best_index
            ]
            .copy()
        )


        best_row[
            "strategy"
        ] = "Recall >= 80%"


        high_recall_rows.append(
            best_row
        )


high_recall_df = pd.DataFrame(
    high_recall_rows
)


# ------------------------------------------------------------
# 9. COMBINE SUMMARY
# ------------------------------------------------------------

summary_df = pd.concat(
    [
        best_f1_df,
        high_recall_df,
    ],
    ignore_index=True
)


summary_columns = [
    "model",
    "strategy",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "tn",
    "fp",
    "fn",
    "tp",
]


summary_df = summary_df[
    summary_columns
]


for col in [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]:

    summary_df[col] = (
        summary_df[col]
        .round(4)
    )


SUMMARY_FILE = (
    REPORT_DIR
    / "16_threshold_optimization_summary.csv"
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ------------------------------------------------------------
# 10. PRINT MAXIMUM-F1 RESULTS
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "BEST THRESHOLD BY F1 SCORE"
)

print(
    "=" * 90
)


print(
    best_f1_df[
        [
            "model",
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "fp",
            "fn",
            "tp",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 11. PRINT HIGH-RECALL RESULTS
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "BEST PRECISION WHILE MAINTAINING "
    "AT LEAST 80% RECALL"
)

print(
    "=" * 90
)


if len(high_recall_df) > 0:

    print(
        high_recall_df[
            [
                "model",
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "fp",
                "fn",
                "tp",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No model reached the required "
        "80% recall."
    )


# ------------------------------------------------------------
# 12. F1 VS THRESHOLD CHART
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 7)
)


for model_name in models:

    model_results = (
        threshold_df[
            threshold_df[
                "model"
            ] == model_name
        ]
    )


    plt.plot(
        model_results[
            "threshold"
        ],

        model_results[
            "f1_score"
        ],

        label=model_name
    )


plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "F1 Score vs Classification Threshold"
)

plt.legend()

plt.tight_layout()


F1_CHART = (
    CHART_DIR
    / "15_f1_vs_threshold.png"
)


plt.savefig(
    F1_CHART,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 13. PRECISION / RECALL CURVES
# ------------------------------------------------------------

for model_name in models:

    model_results = (
        threshold_df[
            threshold_df[
                "model"
            ] == model_name
        ]
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        model_results[
            "threshold"
        ],

        model_results[
            "precision"
        ],

        label="Precision"
    )


    plt.plot(
        model_results[
            "threshold"
        ],

        model_results[
            "recall"
        ],

        label="Recall"
    )


    plt.xlabel(
        "Classification Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        f"{model_name} - "
        f"Precision and Recall vs Threshold"
    )

    plt.legend()

    plt.tight_layout()


    safe_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    chart_file = (
        CHART_DIR
        / (
            f"16_threshold_"
            f"{safe_name}.png"
        )
    )


    plt.savefig(
        chart_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 14. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - STEP 16 "
    "THRESHOLD OPTIMIZATION"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "PURPOSE"
)

report.append(
    "-" * 100
)

report.append(
    "The default classification threshold "
    "of 0.50 may not provide the best "
    "precision-recall trade-off for an "
    "imbalanced churn dataset."
)

report.append("")

report.append(
    "Thresholds from 0.05 to 0.95 were "
    "evaluated using VALIDATION data only."
)

report.append("")

report.append(
    "No test data was used."
)

report.append("")

report.append(
    "BEST F1 THRESHOLDS"
)

report.append(
    "-" * 100
)

report.append(
    best_f1_df[
        [
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "fp",
            "fn",
            "tp",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)

report.append("")

report.append(
    "HIGH-RECALL THRESHOLDS"
)

report.append(
    "-" * 100
)

if len(high_recall_df) > 0:

    report.append(
        high_recall_df[
            [
                "model",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "fp",
                "fn",
                "tp",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

else:

    report.append(
        "No qualifying threshold."
    )


report.append("")

report.append(
    "INTERPRETATION"
)

report.append(
    "-" * 100
)

report.append(
    "Changing the classification threshold "
    "changes precision, recall, F1 and the "
    "confusion matrix."
)

report.append(
    "ROC-AUC and PR-AUC do not change because "
    "they evaluate ranking performance across "
    "thresholds."
)

report.append(
    "The maximum-F1 threshold provides one "
    "statistical operating point."
)

report.append(
    "The high-recall threshold represents a "
    "retention-oriented scenario in which the "
    "business wants to identify at least 80% "
    "of churners while reducing false positives "
    "as much as possible."
)

report.append("")

report.append(
    "Probability values from class-balanced "
    "models should currently be interpreted "
    "primarily as risk scores rather than "
    "perfectly calibrated real-world churn "
    "probabilities."
)

report.append("")

report.append(
    "The final test set remains untouched."
)


REPORT_FILE = (
    REPORT_DIR
    / "16_threshold_optimization_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 15. COMPLETE
# ------------------------------------------------------------

print(
    "\nGenerated files:"
)

print(
    f"- "
    f"{THRESHOLD_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{SUMMARY_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{REPORT_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{F1_CHART.relative_to(PROJECT_ROOT)}"
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 16 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nNo model was retrained."
)

print(
    "The TEST set remains untouched."
)