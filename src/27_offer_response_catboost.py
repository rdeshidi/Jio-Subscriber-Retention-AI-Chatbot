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
)


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 27
# OFFER RESPONSE CATBOOST MODEL
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_FILE = (
    MODEL_DATA_DIR
    / "offer_response_train_60.csv"
)

VALIDATION_FILE = (
    MODEL_DATA_DIR
    / "offer_response_validation_20.csv"
)

METADATA_FILE = (
    MODEL_DATA_DIR
    / "offer_response_metadata.json"
)


# ------------------------------------------------------------
# 2. START
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 27: OFFER RESPONSE CATBOOST")
print("=" * 90)


for file in [
    TRAIN_FILE,
    VALIDATION_FILE,
    METADATA_FILE,
]:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


TARGET = metadata[
    "target"
]

feature_columns = metadata[
    "feature_columns"
]

numeric_features = metadata[
    "numeric_features"
]

categorical_features = metadata[
    "categorical_features"
]


X_train = train_df[
    feature_columns
].copy()

y_train = (
    train_df[
        TARGET
    ]
    .astype(int)
)


X_validation = validation_df[
    feature_columns
].copy()

y_validation = (
    validation_df[
        TARGET
    ]
    .astype(int)
)


print(
    f"\nTraining rows   : "
    f"{len(X_train):,}"
)

print(
    f"Validation rows : "
    f"{len(X_validation):,}"
)

print(
    f"Numeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)


# ------------------------------------------------------------
# 4. PREPARE CATEGORICAL FEATURES
# ------------------------------------------------------------

# CatBoost handles categoricals natively.
#
# Missing categorical values are represented
# explicitly as the string "Missing".

for col in categorical_features:

    X_train[col] = (
        X_train[col]
        .fillna("Missing")
        .astype(str)
    )

    X_validation[col] = (
        X_validation[col]
        .fillna("Missing")
        .astype(str)
    )


# ------------------------------------------------------------
# 5. PREPARE NUMERIC FEATURES
# ------------------------------------------------------------

# Learn median values ONLY from training data.

numeric_medians = {}


for col in numeric_features:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_validation[col] = pd.to_numeric(
        X_validation[col],
        errors="coerce"
    )


    median_value = (
        X_train[
            col
        ].median()
    )


    if pd.isna(
        median_value
    ):

        median_value = 0.0


    numeric_medians[
        col
    ] = float(
        median_value
    )


    X_train[col] = (
        X_train[col]
        .fillna(
            median_value
        )
    )


    X_validation[col] = (
        X_validation[col]
        .fillna(
            median_value
        )
    )


# ------------------------------------------------------------
# 6. CATBOOST MODEL
# ------------------------------------------------------------

# No class weighting is used.
#
# Approximately 37% of exposed subscribers
# redeemed an offer, so this is not a severe
# imbalance problem.

model = CatBoostClassifier(

    iterations=500,

    learning_rate=0.03,

    depth=6,

    l2_leaf_reg=5,

    random_strength=1.0,

    loss_function="Logloss",

    eval_metric="AUC",

    random_seed=42,

    verbose=False,

    allow_writing_files=False,
)


# ------------------------------------------------------------
# 7. TRAIN
# ------------------------------------------------------------

print(
    "\nTraining CatBoost..."
)


model.fit(

    X_train,

    y_train,

    cat_features=
        categorical_features,

    eval_set=(
        X_validation,
        y_validation
    ),

    use_best_model=True,

    early_stopping_rounds=75,
)


best_iteration = (
    model.get_best_iteration()
)


print(
    "[OK] Training complete."
)

print(
    f"Best iteration : "
    f"{best_iteration}"
)


# ------------------------------------------------------------
# 8. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(
    y_true,
    probability,
    threshold=0.50
):

    prediction = (
        probability
        >= threshold
    ).astype(int)


    return {

        "accuracy":
            accuracy_score(
                y_true,
                prediction
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probability
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probability
            ),
    }


# ------------------------------------------------------------
# 9. PREDICTIONS
# ------------------------------------------------------------

train_probability = (
    model.predict_proba(
        X_train
    )[:, 1]
)


validation_probability = (
    model.predict_proba(
        X_validation
    )[:, 1]
)


train_metrics = calculate_metrics(
    y_train,
    train_probability
)


validation_metrics = calculate_metrics(
    y_validation,
    validation_probability
)


validation_prediction = (
    validation_probability
    >= 0.50
).astype(int)


# ------------------------------------------------------------
# 10. CONFUSION MATRIX
# ------------------------------------------------------------

tn, fp, fn, tp = (
    confusion_matrix(
        y_validation,
        validation_prediction,
        labels=[0, 1]
    )
    .ravel()
)


# ------------------------------------------------------------
# 11. TOP-10% RESPONSE TARGETING
# ------------------------------------------------------------

ranking_df = pd.DataFrame(
    {
        "actual_response":
            y_validation.values,

        "response_probability":
            validation_probability,
    }
)


ranking_df = (
    ranking_df
    .sort_values(
        "response_probability",
        ascending=False
    )
    .reset_index(drop=True)
)


top_n = int(
    np.ceil(
        len(ranking_df)
        * 0.10
    )
)


top10 = ranking_df.head(
    top_n
)


total_responders = int(
    ranking_df[
        "actual_response"
    ].sum()
)


top10_responders = int(
    top10[
        "actual_response"
    ].sum()
)


recall_at_top10 = (
    top10_responders
    / total_responders
)


overall_response_rate = (
    ranking_df[
        "actual_response"
    ].mean()
)


top10_response_rate = (
    top10[
        "actual_response"
    ].mean()
)


lift_at_top10 = (
    top10_response_rate
    / overall_response_rate
)


# ------------------------------------------------------------
# 12. RANDOM / NO-SKILL REFERENCES
# ------------------------------------------------------------

random_roc_reference = 0.50

random_pr_reference = (
    overall_response_rate
)

random_top10_recall = 0.10

random_lift = 1.00


# ------------------------------------------------------------
# 13. TRAIN-VALIDATION GAPS
# ------------------------------------------------------------

roc_auc_gap = (
    train_metrics[
        "roc_auc"
    ]
    -
    validation_metrics[
        "roc_auc"
    ]
)


pr_auc_gap = (
    train_metrics[
        "pr_auc"
    ]
    -
    validation_metrics[
        "pr_auc"
    ]
)


# ------------------------------------------------------------
# 14. METRICS TABLE
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    [
        {
            "dataset":
                "Train",

            **train_metrics,
        },

        {
            "dataset":
                "Validation",

            **validation_metrics,
        },
    ]
)


METRICS_FILE = (
    REPORT_DIR
    / "27_offer_response_catboost_metrics.csv"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ------------------------------------------------------------
# 15. VALIDATION PREDICTIONS
# ------------------------------------------------------------

validation_output = pd.DataFrame(
    {
        "actual_response":
            y_validation.values,

        "response_probability":
            validation_probability,

        "predicted_response":
            validation_prediction,
    }
)


PREDICTION_FILE = (
    REPORT_DIR
    / "27_offer_response_catboost_validation_predictions.csv"
)


validation_output.to_csv(
    PREDICTION_FILE,
    index=False
)


# ------------------------------------------------------------
# 16. FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance = (
    model.get_feature_importance()
)


importance_df = pd.DataFrame(
    {
        "feature":
            model.feature_names_,

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
    / "27_offer_response_catboost_feature_importance.csv"
)


importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ------------------------------------------------------------
# 17. SAVE MODEL
# ------------------------------------------------------------

MODEL_FILE = (
    MODEL_DIR
    / "offer_response_catboost.pkl"
)


joblib.dump(
    {
        "model":
            model,

        "target":
            TARGET,

        "feature_columns":
            feature_columns,

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "numeric_medians":
            numeric_medians,

        "threshold":
            0.50,

        "best_iteration":
            best_iteration,
    },

    MODEL_FILE
)


# ------------------------------------------------------------
# 18. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "STEP 27 - OFFER RESPONSE CATBOOST MODEL"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "OBJECTIVE"
)

report.append(
    "-" * 100
)

report.append(
    "Predict offer redemption among historically "
    "offer-exposed subscribers."
)

report.append(
    "This is a propensity model, not causal uplift."
)

report.append("")

report.append(
    "PERFORMANCE"
)

report.append(
    "-" * 100
)

report.append(
    metrics_df
    .round(4)
    .to_string(
        index=False
    )
)

report.append("")

report.append(
    f"Best iteration: "
    f"{best_iteration}"
)

report.append(
    f"Train-validation ROC-AUC gap: "
    f"{roc_auc_gap:.4f}"
)

report.append(
    f"Train-validation PR-AUC gap: "
    f"{pr_auc_gap:.4f}"
)

report.append("")

report.append(
    "VALIDATION CONFUSION MATRIX"
)

report.append(
    "-" * 100
)

report.append(
    f"TN: {tn:,}"
)

report.append(
    f"FP: {fp:,}"
)

report.append(
    f"FN: {fn:,}"
)

report.append(
    f"TP: {tp:,}"
)

report.append("")

report.append(
    "TARGETING PERFORMANCE"
)

report.append(
    "-" * 100
)

report.append(
    f"Responders captured in top 10%: "
    f"{top10_responders:,} / "
    f"{total_responders:,}"
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
    "NO-SKILL REFERENCES"
)

report.append(
    "-" * 100
)

report.append(
    f"Random ROC-AUC reference: "
    f"{random_roc_reference:.2f}"
)

report.append(
    f"PR-AUC prevalence baseline: "
    f"{random_pr_reference:.4f}"
)

report.append(
    f"Random Recall@Top10%: "
    f"{random_top10_recall:.2f}"
)

report.append(
    f"Random Lift@Top10%: "
    f"{random_lift:.2f}x"
)

report.append("")

report.append(
    "TOP FEATURE IMPORTANCES"
)

report.append(
    "-" * 100
)

report.append(
    importance_df
    .head(15)
    .round(4)
    .to_string(
        index=False
    )
)

report.append("")

report.append(
    "IMPORTANT INTERPRETATION"
)

report.append(
    "-" * 100
)

report.append(
    "Feature importance should only be interpreted "
    "if the model demonstrates meaningful "
    "out-of-sample predictive discrimination."
)

report.append(
    "If validation ROC-AUC remains near 0.50, "
    "PR-AUC remains near the response prevalence, "
    "and lift remains near 1.0, the available "
    "features do not provide useful offer-response "
    "prediction signal."
)

report.append(
    "In that case, further tuning should not be "
    "used to manufacture an apparently strong model."
)

report.append(
    "The offer-response test set remains untouched."
)


REPORT_FILE = (
    REPORT_DIR
    / "27_offer_response_catboost_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 19. TERMINAL OUTPUT
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "CATBOOST OFFER RESPONSE RESULTS"
)

print(
    "=" * 90
)


print(
    metrics_df
    .round(4)
    .to_string(
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
    "\nTOP-10% TARGETING PERFORMANCE"
)

print(
    f"Responders captured : "
    f"{top10_responders:,} / "
    f"{total_responders:,}"
)

print(
    f"Recall@Top10%       : "
    f"{recall_at_top10:.4f}"
)

print(
    f"Lift@Top10%         : "
    f"{lift_at_top10:.2f}x"
)


print(
    "\nGENERALISATION"
)

print(
    f"ROC-AUC gap         : "
    f"{roc_auc_gap:.4f}"
)

print(
    f"PR-AUC gap          : "
    f"{pr_auc_gap:.4f}"
)


print(
    "\nNO-SKILL REFERENCES"
)

print(
    f"ROC-AUC             : 0.5000"
)

print(
    f"PR-AUC baseline     : "
    f"{overall_response_rate:.4f}"
)

print(
    f"Recall@Top10%       : 0.1000"
)

print(
    f"Lift@Top10%         : 1.00x"
)


print(
    "\nTOP 10 FEATURE IMPORTANCES"
)

print(
    importance_df
    .head(10)
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nGenerated files:"
)


for file in [
    METRICS_FILE,
    PREDICTION_FILE,
    IMPORTANCE_FILE,
    MODEL_FILE,
    REPORT_FILE,
]:

    print(
        f"- "
        f"{file.relative_to(PROJECT_ROOT)}"
    )


print(
    "\n" + "=" * 90
)

print(
    "STEP 27 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nOffer-response TEST set remains untouched."
)