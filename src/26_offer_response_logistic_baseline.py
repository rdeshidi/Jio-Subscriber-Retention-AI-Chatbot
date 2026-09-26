from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

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
)


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 26
# OFFER RESPONSE LOGISTIC REGRESSION BASELINE
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
print("DAY 2 - STEP 26: OFFER RESPONSE LOGISTIC BASELINE")
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
# 4. PREPROCESSING
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


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
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
# 5. LOGISTIC REGRESSION
# ------------------------------------------------------------

model = LogisticRegression(

    max_iter=2000,

    random_state=42,

    # No class weighting needed here.
    # Response prevalence is approximately 37%.
    class_weight=None,
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        ),
    ]
)


# ------------------------------------------------------------
# 6. TRAIN
# ------------------------------------------------------------

print(
    "\nTraining Logistic Regression..."
)


pipeline.fit(
    X_train,
    y_train
)


print(
    "[OK] Training complete."
)


# ------------------------------------------------------------
# 7. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50
):

    predictions = (
        probabilities
        >= threshold
    ).astype(int)


    return {

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

        "f1":
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
# 8. PREDICTIONS
# ------------------------------------------------------------

train_probability = (
    pipeline.predict_proba(
        X_train
    )[:, 1]
)


validation_probability = (
    pipeline.predict_proba(
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


# ------------------------------------------------------------
# 9. CONFUSION MATRIX
# ------------------------------------------------------------

validation_prediction = (
    validation_probability
    >= 0.50
).astype(int)


tn, fp, fn, tp = (
    confusion_matrix(
        y_validation,
        validation_prediction,
        labels=[0, 1]
    )
    .ravel()
)


# ------------------------------------------------------------
# 10. TOP-10% RESPONSE CAPTURE
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
# 11. SAVE METRICS
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
    / "26_offer_response_logistic_metrics.csv"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ------------------------------------------------------------
# 12. SAVE VALIDATION PREDICTIONS
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
    / "26_offer_response_logistic_validation_predictions.csv"
)


validation_output.to_csv(
    PREDICTION_FILE,
    index=False
)


# ------------------------------------------------------------
# 13. COEFFICIENT INTERPRETATION
# ------------------------------------------------------------

fitted_preprocessor = (
    pipeline
    .named_steps[
        "preprocessor"
    ]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


coefficients = (
    pipeline
    .named_steps[
        "model"
    ]
    .coef_[0]
)


coefficient_df = pd.DataFrame(
    {
        "feature":
            feature_names,

        "coefficient":
            coefficients,

        "absolute_coefficient":
            np.abs(
                coefficients
            ),
    }
)


coefficient_df = (
    coefficient_df
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
    .reset_index(drop=True)
)


COEFFICIENT_FILE = (
    REPORT_DIR
    / "26_offer_response_logistic_coefficients.csv"
)


coefficient_df.to_csv(
    COEFFICIENT_FILE,
    index=False
)


# ------------------------------------------------------------
# 14. SAVE MODEL
# ------------------------------------------------------------

MODEL_FILE = (
    MODEL_DIR
    / "offer_response_logistic_baseline.pkl"
)


joblib.dump(
    {
        "pipeline":
            pipeline,

        "target":
            TARGET,

        "feature_columns":
            feature_columns,

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "threshold":
            0.50,
    },

    MODEL_FILE
)


# ------------------------------------------------------------
# 15. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "STEP 26 - OFFER RESPONSE LOGISTIC REGRESSION BASELINE"
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
    "Predict offer redemption among customers "
    "who were historically exposed to an offer."
)

report.append("")

report.append(
    "This is a response-propensity model, "
    "not a causal uplift model."
)

report.append("")

report.append(
    "TRAIN / VALIDATION PERFORMANCE"
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
    "VALIDATION CONFUSION MATRIX"
)

report.append(
    "-" * 100
)

report.append(
    f"True negatives  : {tn:,}"
)

report.append(
    f"False positives : {fp:,}"
)

report.append(
    f"False negatives : {fn:,}"
)

report.append(
    f"True positives  : {tp:,}"
)

report.append("")

report.append(
    "TARGETING PERFORMANCE"
)

report.append(
    "-" * 100
)

report.append(
    f"Validation responders: "
    f"{total_responders:,}"
)

report.append(
    f"Customers in top 10%: "
    f"{top_n:,}"
)

report.append(
    f"Responders captured in top 10%: "
    f"{top10_responders:,}"
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
    "TOP COEFFICIENTS BY ABSOLUTE MAGNITUDE"
)

report.append(
    "-" * 100
)

report.append(
    coefficient_df
    .head(20)
    .round(4)
    .to_string(
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
    "Positive coefficients are associated with "
    "higher predicted redemption propensity; "
    "negative coefficients are associated with "
    "lower predicted propensity, holding other "
    "model inputs constant."
)

report.append(
    "These coefficients describe model "
    "associations and should not be interpreted "
    "as causal effects of customer attributes."
)

report.append(
    "The offer-response test set remains untouched."
)


REPORT_FILE = (
    REPORT_DIR
    / "26_offer_response_logistic_report.txt"
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
    "LOGISTIC REGRESSION RESULTS"
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
    "\nTOP 10 MODEL COEFFICIENTS"
)

print(
    coefficient_df[
        [
            "feature",
            "coefficient",
        ]
    ]
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
    COEFFICIENT_FILE,
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
    "STEP 26 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nOffer-response TEST set remains untouched."
)