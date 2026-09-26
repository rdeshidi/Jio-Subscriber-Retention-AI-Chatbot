from pathlib import Path
import joblib
import numpy as np
import pandas as pd


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 22
# CHURN RISK + CUSTOMER VALUE PRIORITISATION
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSCRIBER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "subscribers_clean.csv"
)

CLV_FILE = (
    PROJECT_ROOT
    / "data"
    / "clv"
    / "clv_value_prepared.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "catboost_model_package.pkl"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

CLV_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "clv"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLV_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. START
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 22: RISK-VALUE PRIORITISATION")
print("=" * 90)


for file in [
    SUBSCRIBER_FILE,
    CLV_FILE,
    MODEL_FILE,
]:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

subscribers = pd.read_csv(
    SUBSCRIBER_FILE
)

clv = pd.read_csv(
    CLV_FILE
)


print(
    f"\nSubscribers loaded : "
    f"{len(subscribers):,}"
)

print(
    f"CLV rows loaded     : "
    f"{len(clv):,}"
)


# ------------------------------------------------------------
# 4. LOAD FROZEN CATBOOST MODEL
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


# ------------------------------------------------------------
# 5. PREPARE MODEL FEATURES
# ------------------------------------------------------------

# IMPORTANT:
# CatBoost must receive columns in exactly the same
# order used during model training.
#
# Do NOT rebuild the order as:
# numeric_features + categorical_features
#
# The trained CatBoost model already stores the
# original feature order.

feature_columns = list(
    model.feature_names_
)


print(
    f"\nModel expects "
    f"{len(feature_columns)} features."
)


# Validate that every trained feature is available.

missing_model_features = [
    col
    for col in feature_columns
    if col not in subscribers.columns
]


if missing_model_features:

    raise ValueError(
        "The following model features are "
        "missing from subscribers_clean.csv:\n"
        + "\n".join(
            missing_model_features
        )
    )


X = subscribers[
    feature_columns
].copy()


print(
    "[OK] Original CatBoost feature "
    "order restored."
)

for col in categorical_features:

    X[col] = (
        X[col]
        .fillna("Missing")
        .astype(str)
    )


for col in numeric_features:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X[col] = (
        X[col]
        .fillna(
            numeric_medians[
                col
            ]
        )
    )

# ------------------------------------------------------------
# 5A. FINAL MODEL INPUT VALIDATION
# ------------------------------------------------------------

if list(X.columns) != feature_columns:

    raise ValueError(
        "Feature order does not match "
        "the trained CatBoost model."
    )


for col in categorical_features:

    if X[col].isna().any():

        raise ValueError(
            f"Categorical column still "
            f"contains missing values: {col}"
        )


print(
    "[OK] CatBoost scoring input validated."
)


# ------------------------------------------------------------
# 6. SCORE ALL SUBSCRIBERS
# ------------------------------------------------------------

print(
    "\nScoring subscribers using "
    "the frozen CatBoost model..."
)


churn_risk_score = (
    model.predict_proba(
        X
    )[:, 1]
)


print(
    "[OK] Scoring complete."
)


# ------------------------------------------------------------
# 7. CREATE BASE PRIORITY DATASET
# ------------------------------------------------------------

priority_df = pd.DataFrame(
    {
        "subscriber_id":
            subscribers[
                "subscriber_id"
            ],

        "circle":
            subscribers[
                "circle"
            ],

        "churn_risk_score":
            churn_risk_score,
    }
)


priority_df = (
    priority_df
    .merge(
        clv[
            [
                "subscriber_id",
                "monthly_value_basis_inr",
                "revenue_proxy_12m_inr",
                "customer_value_band",
            ]
        ],
        on="subscriber_id",
        how="left",
        validate="one_to_one",
    )
)


# ------------------------------------------------------------
# 8. RISK PERCENTILE
# ------------------------------------------------------------

priority_df[
    "risk_percentile"
] = (
    priority_df[
        "churn_risk_score"
    ]
    .rank(
        pct=True,
        method="average"
    )
)


# ------------------------------------------------------------
# 9. VALUE PERCENTILE
# ------------------------------------------------------------

priority_df[
    "value_percentile"
] = (
    priority_df[
        "revenue_proxy_12m_inr"
    ]
    .rank(
        pct=True,
        method="average"
    )
)


# ------------------------------------------------------------
# 10. PRIORITY INDEX
# ------------------------------------------------------------

# Unitless rank-based prioritisation score.
#
# Multiplication ensures that a subscriber needs
# both meaningful risk AND meaningful value to
# receive a very high score.
#
# This is NOT a monetary expected-loss estimate.

priority_df[
    "risk_value_priority_score"
] = (
    priority_df[
        "risk_percentile"
    ]
    *
    priority_df[
        "value_percentile"
    ]
    *
    100
)


# ------------------------------------------------------------
# 11. RISK DECILES
# ------------------------------------------------------------

priority_df[
    "risk_rank"
] = (
    priority_df[
        "churn_risk_score"
    ]
    .rank(
        method="first",
        ascending=False
    )
)


total_customers = len(
    priority_df
)


priority_df[
    "risk_decile"
] = np.minimum(
    np.ceil(
        priority_df[
            "risk_rank"
        ]
        / total_customers
        * 10
    ),
    10
).astype(int)


# Decile 1 = highest risk.


# ------------------------------------------------------------
# 12. RETENTION PRIORITY SEGMENTS
# ------------------------------------------------------------

def assign_priority_segment(row):

    risk_decile = (
        row[
            "risk_decile"
        ]
    )

    value_band = (
        row[
            "customer_value_band"
        ]
    )


    if (
        risk_decile == 1
        and value_band == "High Value"
    ):

        return (
            "Priority 1 - High Risk / High Value"
        )


    elif (
        risk_decile == 1
        and value_band
        == "Upper-Mid Value"
    ):

        return (
            "Priority 2 - High Risk / Upper-Mid Value"
        )


    elif (
        risk_decile == 1
    ):

        return (
            "Priority 3 - High Risk / Lower Value"
        )


    elif (
        risk_decile <= 3
        and value_band == "High Value"
    ):

        return (
            "Priority 4 - High Value Watchlist"
        )


    else:

        return (
            "Standard Monitoring"
        )


priority_df[
    "retention_priority_segment"
] = (
    priority_df.apply(
        assign_priority_segment,
        axis=1
    )
)


# ------------------------------------------------------------
# 13. ADD ACTUAL CHURN ONLY FOR ANALYSIS
# ------------------------------------------------------------

# These labels are included only to evaluate
# historical/sandbox segmentation.
#
# They must NOT be used when deciding a live
# future customer's priority.

for target in [
    "churn_flag_30d",
    "churn_flag_90d",
]:

    if target in subscribers.columns:

        converted = (
            subscribers[target]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                    "yes": 1,
                    "no": 0,
                }
            )
        )

        priority_df[
            target
        ] = converted


# ------------------------------------------------------------
# 14. PRIORITY SEGMENT SUMMARY
# ------------------------------------------------------------

summary_agg = {
    "subscriber_id":
        "count",

    "monthly_value_basis_inr":
        "mean",

    "revenue_proxy_12m_inr":
        "mean",

    "churn_risk_score":
        "mean",

    "risk_value_priority_score":
        "mean",
}


if "churn_flag_30d" in priority_df.columns:

    summary_agg[
        "churn_flag_30d"
    ] = "mean"


segment_summary = (
    priority_df
    .groupby(
        "retention_priority_segment"
    )
    .agg(
        summary_agg
    )
    .rename(
        columns={
            "subscriber_id":
                "customers",

            "monthly_value_basis_inr":
                "avg_monthly_value_inr",

            "revenue_proxy_12m_inr":
                "avg_12m_revenue_proxy_inr",

            "churn_risk_score":
                "avg_churn_risk_score",

            "risk_value_priority_score":
                "avg_priority_score",

            "churn_flag_30d":
                "actual_30d_churn_rate",
        }
    )
    .reset_index()
)


if (
    "actual_30d_churn_rate"
    in segment_summary.columns
):

    segment_summary[
        "actual_30d_churn_pct"
    ] = (
        segment_summary[
            "actual_30d_churn_rate"
        ]
        * 100
    )

    segment_summary = (
        segment_summary
        .drop(
            columns=[
                "actual_30d_churn_rate"
            ]
        )
    )


# ------------------------------------------------------------
# 15. SORT PRIORITY DATASET
# ------------------------------------------------------------

priority_df = (
    priority_df
    .sort_values(
        "risk_value_priority_score",
        ascending=False
    )
    .reset_index(drop=True)
)


priority_df[
    "overall_priority_rank"
] = (
    np.arange(
        1,
        len(priority_df) + 1
    )
)


# ------------------------------------------------------------
# 16. TOP PRIORITY SUBSCRIBERS
# ------------------------------------------------------------

top_priority = (
    priority_df
    .head(1000)
    .copy()
)


# ------------------------------------------------------------
# 17. SAVE OUTPUTS
# ------------------------------------------------------------

PRIORITY_FILE = (
    CLV_DATA_DIR
    / "customer_risk_value_priority.csv"
)


priority_df.to_csv(
    PRIORITY_FILE,
    index=False
)


SUMMARY_FILE = (
    REPORT_DIR
    / "22_risk_value_segment_summary.csv"
)


segment_summary.to_csv(
    SUMMARY_FILE,
    index=False
)


TOP_FILE = (
    REPORT_DIR
    / "22_top1000_retention_priority.csv"
)


top_priority.to_csv(
    TOP_FILE,
    index=False
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
    "DAY 2 - STEP 22 RISK-VALUE PRIORITISATION"
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
    "The CLV assessment found no observed "
    "future-value target."
)

report.append(
    "Therefore the project uses a transparent "
    "12-month ARPU-based revenue proxy together "
    "with the frozen churn model's risk ranking."
)

report.append("")

report.append(
    "RISK-VALUE PRIORITY INDEX"
)

report.append(
    "-" * 100
)

report.append(
    "risk percentile x value percentile x 100"
)

report.append("")

report.append(
    "The index is unitless."
)

report.append(
    "It is NOT an estimated rupee loss and "
    "should not be interpreted as a calibrated "
    "financial expectation."
)

report.append("")

report.append(
    "RETENTION PRIORITY SEGMENTS"
)

report.append(
    "-" * 100
)

report.append(
    segment_summary
    .round(4)
    .to_string(
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
    "Priority 1 identifies subscribers who "
    "are simultaneously in the highest churn-risk "
    "decile and highest customer-value quartile."
)

report.append(
    "Priority 2 captures highest-risk subscribers "
    "with upper-mid customer value."
)

report.append(
    "Priority 3 contains other subscribers in "
    "the highest-risk decile."
)

report.append(
    "Priority 4 creates a watchlist of high-value "
    "customers within the top three risk deciles."
)

report.append(
    "All other customers remain under standard "
    "monitoring."
)

report.append("")

report.append(
    "IMPORTANT LIMITATION"
)

report.append(
    "-" * 100
)

report.append(
    "The CatBoost model was trained with class "
    "weighting, so churn scores are used here "
    "primarily for ranking rather than interpreted "
    "as perfectly calibrated real-world probabilities."
)

report.append(
    "Similarly, the 12-month revenue measure is "
    "an ARPU-based proxy, not observed lifetime "
    "contribution or profit."
)


REPORT_FILE = (
    REPORT_DIR
    / "22_risk_value_prioritization_report.txt"
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
    "RETENTION PRIORITY SEGMENTS"
)

print(
    "=" * 90
)


print(
    segment_summary
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nTOP 10 PRIORITY CUSTOMERS"
)

print(
    "-" * 90
)


display_cols = [
    "overall_priority_rank",
    "subscriber_id",
    "circle",
    "churn_risk_score",
    "revenue_proxy_12m_inr",
    "customer_value_band",
    "risk_decile",
    "risk_value_priority_score",
    "retention_priority_segment",
]


print(
    priority_df[
        display_cols
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
    PRIORITY_FILE,
    SUMMARY_FILE,
    TOP_FILE,
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
    "STEP 22 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nCLV/value prioritisation workflow complete."
)