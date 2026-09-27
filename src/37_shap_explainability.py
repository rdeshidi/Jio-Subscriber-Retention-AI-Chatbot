"""
Step 37 - SHAP Explainability
Jio Subscriber Retention & AI Chatbot

Purpose:
1. Explain the existing final CatBoost churn model.
2. Produce global SHAP feature importance using a sample of the
   untouched test set.
3. Produce individual subscriber explanations for the highest-risk
   subscribers from the cleaned subscriber dataset.
4. Save CSV reports, charts, and a text summary.

Important:
- The existing CatBoost model is NOT retrained.
- The test set is NOT used for threshold tuning or model selection.
- SHAP is an explainability layer only.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import Pool


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "catboost_model_package.pkl"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "test_20.csv"
)

SUBSCRIBER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "subscribers_clean.csv"
)

OUTPUT_REPORTS = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

OUTPUT_CHARTS = (
    PROJECT_ROOT
    / "outputs"
    / "charts"
)


OUTPUT_REPORTS.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CHARTS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Keep SHAP computation practical on the user's laptop.
GLOBAL_SAMPLE_SIZE = 3000

# Number of high-risk subscribers to explain individually.
TOP_RISK_SUBSCRIBERS = 20

# Number of SHAP reasons shown per subscriber.
TOP_REASONS_PER_SUBSCRIBER = 5

# Human-facing explanation cleanup:
# Do not show both a business label and its encoded duplicate
# as separate reasons.
DISPLAY_DUPLICATE_FEATURES = {
    "circle_code",
}


# ============================================================
# LOAD MODEL PACKAGE
# ============================================================

print("=" * 78)
print("JIO RETENTION - STEP 37 SHAP EXPLAINABILITY")
print("=" * 78)

print("\n[1/7] Loading existing CatBoost model package...")

with open(
    MODEL_PATH,
    "rb",
) as f:
    package = pickle.load(f)

model = package["model"]
numeric_features = list(
    package["numeric_features"]
)
categorical_features = list(
    package["categorical_features"]
)
numeric_medians = package[
    "numeric_medians"
]
target_name = package["target"]

feature_columns = (
    numeric_features
    + categorical_features
)

print(
    f"[OK] Model class: {type(model)}"
)

print(
    f"[OK] Numeric features: "
    f"{len(numeric_features)}"
)

print(
    f"[OK] Categorical features: "
    f"{len(categorical_features)}"
)

print(
    f"[OK] Total features: "
    f"{len(feature_columns)}"
)

print(
    f"[OK] Target: {target_name}"
)


# ============================================================
# PREPROCESSING HELPER
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recreate the feature representation required by the
    saved CatBoost model.

    Numeric:
      - convert to numeric
      - fill missing values using training medians

    Categorical:
      - fill missing values
      - convert to string
    """

    missing_features = [
        col
        for col in feature_columns
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )

    X = df[
        feature_columns
    ].copy()

    # Numeric preprocessing
    for col in numeric_features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

        median_value = numeric_medians.get(
            col,
            0,
        )

        X[col] = X[col].fillna(
            median_value
        )

    # Categorical preprocessing
    for col in categorical_features:

        X[col] = X[col].fillna(
            "__MISSING__"
        )

        X[col] = (
            X[col]
            .astype(str)
        )

    return X


# ============================================================
# CATBOOST POOL HELPER
# ============================================================

cat_feature_indices = [
    feature_columns.index(col)
    for col in categorical_features
]


def make_pool(
    X: pd.DataFrame,
) -> Pool:

    return Pool(
        X,
        cat_features=cat_feature_indices,
    )


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n[2/7] Loading untouched test dataset...")

test_df = pd.read_csv(
    TEST_PATH
)

print(
    f"[OK] Test rows: {len(test_df):,}"
)

X_test = prepare_features(
    test_df
)

print(
    f"[OK] Test feature matrix: "
    f"{X_test.shape}"
)


# ============================================================
# 3. GLOBAL SHAP SAMPLE
# ============================================================

print("\n[3/7] Preparing global SHAP sample...")

sample_size = min(
    GLOBAL_SAMPLE_SIZE,
    len(X_test),
)

rng = np.random.default_rng(
    RANDOM_STATE
)

sample_indices = rng.choice(
    len(X_test),
    size=sample_size,
    replace=False,
)

X_test_sample = (
    X_test
    .iloc[sample_indices]
    .reset_index(drop=True)
)

print(
    f"[OK] SHAP test sample: "
    f"{len(X_test_sample):,} rows"
)


# ============================================================
# 4. CALCULATE GLOBAL SHAP
# ============================================================

print("\n[4/7] Calculating SHAP values...")

explainer = shap.TreeExplainer(
    model
)

test_pool = make_pool(
    X_test_sample
)

shap_values_raw = (
    explainer.shap_values(
        test_pool
    )
)

# SHAP may return a list for some model types.
if isinstance(
    shap_values_raw,
    list,
):

    shap_values = (
        np.asarray(
            shap_values_raw[-1]
        )
    )

else:

    shap_values = (
        np.asarray(
            shap_values_raw
        )
    )

# Some SHAP/model combinations can
# return an extra base-value column.
if (
    shap_values.ndim == 2
    and shap_values.shape[1]
    == len(feature_columns) + 1
):

    shap_values = (
        shap_values[:, :-1]
    )

if (
    shap_values.ndim != 2
    or shap_values.shape[1]
    != len(feature_columns)
):

    raise ValueError(
        "Unexpected SHAP matrix shape: "
        f"{shap_values.shape}; expected "
        f"(rows, {len(feature_columns)})"
    )

print(
    f"[OK] SHAP matrix: "
    f"{shap_values.shape}"
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = (
    np.abs(shap_values)
    .mean(axis=0)
)

mean_signed_shap = (
    shap_values
    .mean(axis=0)
)

global_shap = pd.DataFrame(
    {
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap,
    }
)

global_shap = (
    global_shap
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

global_shap[
    "rank"
] = (
    np.arange(
        1,
        len(global_shap) + 1,
    )
)

global_shap_path = (
    OUTPUT_REPORTS
    / "37_shap_global_feature_importance.csv"
)

global_shap.to_csv(
    global_shap_path,
    index=False,
)

print(
    f"[OK] Saved: "
    f"{global_shap_path}"
)


# ============================================================
# GLOBAL SHAP BAR CHART
# ============================================================

print(
    "\n[5/7] Generating global SHAP chart..."
)

top_global = (
    global_shap
    .head(15)
    .sort_values(
        "mean_abs_shap",
        ascending=True,
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_global["feature"],
    top_global["mean_abs_shap"],
)

plt.xlabel(
    "Mean absolute SHAP value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top SHAP Features - 30-Day Churn Model"
)

plt.tight_layout()

global_chart_path = (
    OUTPUT_CHARTS
    / "25_shap_global_feature_importance.png"
)

plt.savefig(
    global_chart_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close()

print(
    f"[OK] Saved: "
    f"{global_chart_path}"
)


# ============================================================
# 6. INDIVIDUAL HIGH-RISK EXPLANATIONS
# ============================================================

print(
    "\n[6/7] Generating individual subscriber explanations..."
)

subscriber_df = pd.read_csv(
    SUBSCRIBER_PATH
)

if "subscriber_id" not in subscriber_df.columns:

    raise ValueError(
        "subscriber_id not found in "
        "subscribers_clean.csv"
    )

X_all = prepare_features(
    subscriber_df
)

all_pool = make_pool(
    X_all
)

risk_probability = (
    model.predict_proba(
        all_pool
    )[:, 1]
)

subscriber_risk = pd.DataFrame(
    {
        "subscriber_id":
            subscriber_df[
                "subscriber_id"
            ],
        "churn_probability_30d":
            risk_probability,
    }
)

subscriber_risk = (
    subscriber_risk
    .sort_values(
        "churn_probability_30d",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

top_risk = subscriber_risk.head(
    TOP_RISK_SUBSCRIBERS
).copy()

print(
    f"[OK] Selected top "
    f"{len(top_risk)} high-risk subscribers"
)

top_indices = (
    top_risk.index
)

# The top_risk DataFrame was reset, so
# get the corresponding original row
# positions by subscriber ID.
top_ids = set(
    top_risk[
        "subscriber_id"
    ]
)

id_to_position = {
    sid: idx
    for idx, sid in enumerate(
        subscriber_df[
            "subscriber_id"
        ]
    )
}

selected_positions = [
    id_to_position[sid]
    for sid in top_risk[
        "subscriber_id"
    ]
]

X_high_risk = (
    X_all
    .iloc[selected_positions]
    .reset_index(drop=True)
)

high_risk_pool = make_pool(
    X_high_risk
)

individual_shap_raw = (
    explainer.shap_values(
        high_risk_pool
    )
)

if isinstance(
    individual_shap_raw,
    list,
):

    individual_shap = np.asarray(
        individual_shap_raw[-1]
    )

else:

    individual_shap = np.asarray(
        individual_shap_raw
    )

if (
    individual_shap.ndim == 2
    and individual_shap.shape[1]
    == len(feature_columns) + 1
):

    individual_shap = (
        individual_shap[:, :-1]
    )


# ============================================================
# LONG-FORM SUBSCRIBER EXPLANATION
# ============================================================

explanation_rows = []

for row_number, subscriber_id in enumerate(
    top_risk[
        "subscriber_id"
    ]
):

    row_values = (
        X_high_risk
        .iloc[row_number]
    )

    row_shap = (
        individual_shap[
            row_number
        ]
    )

    reason_df = pd.DataFrame(
        {
            "feature":
                feature_columns,
            "shap_value":
                row_shap,
        }
    )

    reason_df = reason_df[
    ~reason_df["feature"].isin(
        DISPLAY_DUPLICATE_FEATURES
    )
].copy()

    reason_df[
        "abs_shap_value"
    ] = np.abs(
        reason_df[
            "shap_value"
        ]
    )

    reason_df = (
        reason_df
        .sort_values(
            "abs_shap_value",
            ascending=False,
        )
        .head(
            TOP_REASONS_PER_SUBSCRIBER
        )
    )

    risk_probability_value = float(
        top_risk
        .iloc[row_number][
            "churn_probability_30d"
        ]
    )

    for reason_rank, (
        _, reason_row
    ) in enumerate(
        reason_df.iterrows(),
        start=1,
    ):

        feature_name = (
            reason_row[
                "feature"
            ]
        )

        shap_value = float(
            reason_row[
                "shap_value"
            ]
        )

        feature_value = (
            row_values[
                feature_name
            ]
        )

        direction = (
            "increases churn risk"
            if shap_value > 0
            else "decreases churn risk"
        )

        explanation_rows.append(
    {
        "explanation_id":
            f"HIGH_RISK_{row_number + 1:03d}",
        "churn_probability_30d":
            risk_probability_value,
                "reason_rank":
                    reason_rank,
                "feature":
                    feature_name,
                "feature_value":
                    feature_value,
                "shap_value":
                    shap_value,
                "abs_shap_value":
                    abs(
                        shap_value
                    ),
                "direction":
                    direction,
            }
        )


individual_explanations = pd.DataFrame(
    explanation_rows
)

individual_path = (
    OUTPUT_REPORTS
    / "37_shap_individual_subscriber_explanations.csv"
)

individual_explanations.to_csv(
    individual_path,
    index=False,
)

print(
    f"[OK] Saved: "
    f"{individual_path}"
)


# ============================================================
# HIGH-RISK SUBSCRIBER SUMMARY
# ============================================================

risk_summary_path = (
    OUTPUT_REPORTS
    / "37_high_risk_subscribers_shap_summary.csv"
)

safe_risk_summary = top_risk.copy()

safe_risk_summary.insert(
    0,
    "explanation_id",
    [
        f"HIGH_RISK_{i:03d}"
        for i in range(
            1,
            len(safe_risk_summary) + 1,
        )
    ],
)

safe_risk_summary = safe_risk_summary.drop(
    columns=["subscriber_id"]
)

safe_risk_summary.to_csv(
    risk_summary_path,
    index=False,
)

print(
    f"[OK] Saved: "
    f"{risk_summary_path}"
)


# ============================================================
# TEXT REPORT
# ============================================================

print(
    "\n[7/7] Writing SHAP report..."
)

report_path = (
    OUTPUT_REPORTS
    / "37_shap_explainability_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "JIO SUBSCRIBER RETENTION\n"
    )

    f.write(
        "STEP 37 - SHAP EXPLAINABILITY REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Purpose\n"
    )

    f.write(
        "Explain the existing final CatBoost 30-day churn model "
        "without retraining it.\n\n"
    )

    f.write(
        "Global SHAP analysis\n"
    )

    f.write(
        f"Test-set rows available: {len(test_df):,}\n"
    )

    f.write(
        f"SHAP explanation sample: {len(X_test_sample):,}\n"
    )

    f.write(
        "Global feature importance is based on mean absolute SHAP value.\n\n"
    )

    f.write(
        "Top 15 SHAP features\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for _, row in global_shap.head(15).iterrows():

        f.write(
            f"{int(row['rank']):2d}. "
            f"{row['feature']}: "
            f"{row['mean_abs_shap']:.6f}\n"
        )

    f.write(
        "\nIndividual subscriber explanations\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"High-risk subscribers explained: "
        f"{len(top_risk)}\n\n"
    )

    for row_number, (_, risk_row) in enumerate(
        top_risk.iterrows(),
        start=1,
    ):

        probability = (
            risk_row[
                "churn_probability_30d"
            ]
        )

        f.write(
            f"Explanation {row_number:03d} | "
            f"30-day churn probability: "
            f"{probability:.4f}\n"
        )

        explanation_id = (
            f"HIGH_RISK_{row_number:03d}"
        )

        subscriber_reasons = (
            individual_explanations[
                individual_explanations[
                    "explanation_id"
                ] == explanation_id
            ]
            .sort_values(
                "reason_rank"
            )
        )

        for _, reason in subscriber_reasons.iterrows():

            f.write(
                f"  {int(reason['reason_rank'])}. "
                f"{reason['feature']} = "
                f"{reason['feature_value']} | "
                f"SHAP={reason['shap_value']:.6f} | "
                f"{reason['direction']}\n"
            )

        f.write("\n")

    f.write(
        "Interpretation note\n"
    )

    f.write(
        "A positive SHAP value indicates that the feature pushes the "
        "model prediction toward higher churn risk for that observation. "
        "A negative SHAP value pushes the prediction toward lower churn "
        "risk. SHAP explains the model prediction; it does not prove that "
        "changing a feature will causally change churn.\n"
    )

print(
    f"[OK] Saved: {report_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("STEP 37 SHAP SUMMARY")
print("=" * 78)

print(
    f"Global SHAP sample: "
    f"{len(X_test_sample):,}"
)

print(
    f"Individual subscribers explained: "
    f"{len(top_risk):,}"
)

print(
    "\nTop 10 SHAP features:"
)

for _, row in global_shap.head(10).iterrows():

    print(
        f"{int(row['rank']):2d}. "
        f"{row['feature']}: "
        f"{row['mean_abs_shap']:.6f}"
    )

print(
    "\nCreated artifacts:"
)

print(
    global_shap_path
)

print(
    global_chart_path
)

print(
    individual_path
)

print(
    risk_summary_path
)

print(
    report_path
)

print(
    "\n[OK] Step 37 SHAP explainability completed."
)