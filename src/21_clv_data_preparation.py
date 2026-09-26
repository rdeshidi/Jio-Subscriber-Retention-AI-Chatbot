from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 21
# CLV TARGET ASSESSMENT & DATA PREPARATION
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "subscribers_clean.csv"
)

CLV_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "clv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

CLV_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. START
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 21: CLV TARGET ASSESSMENT")
print("=" * 90)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nLoaded dataset: "
    f"{df.shape[0]:,} rows x "
    f"{df.shape[1]} columns"
)


# ------------------------------------------------------------
# 4. SEARCH FOR POSSIBLE TRUE CLV TARGETS
# ------------------------------------------------------------

target_keywords = [
    "clv",
    "lifetime_value",
    "future_revenue",
    "future_value",
    "future_margin",
    "contribution",
    "profit",
    "margin",
    "revenue_12m",
    "revenue_next",
    "value_12m",
]


possible_true_targets = []


for col in df.columns:

    col_lower = col.lower()

    if any(
        keyword in col_lower
        for keyword in target_keywords
    ):

        possible_true_targets.append(
            col
        )


print(
    "\nPossible genuine future-value "
    "target columns found:"
)


if possible_true_targets:

    for col in possible_true_targets:
        print(f"- {col}")

else:

    print(
        "- None found"
    )


# ------------------------------------------------------------
# 5. REVIEW VALUE-RELATED INPUT FIELDS
# ------------------------------------------------------------

candidate_value_columns = [
    "plan_price_inr",
    "arpu_last_month_inr",
    "arpu_3m_avg_inr",
    "arpu_6m_avg_inr",
    "tenure_months",
    "recharge_count_6m",
    "avg_recharge_gap_days",
    "days_since_last_recharge",
    "payment_failures_6m",
    "offer_exposed_90d",
    "offer_redeemed_90d",
    "churn_flag_30d",
    "churn_flag_90d",
]


available_value_columns = [
    col
    for col in candidate_value_columns
    if col in df.columns
]


print(
    "\nAvailable CLV/value-related columns:"
)


for col in available_value_columns:

    print(
        f"- {col}"
    )


# ------------------------------------------------------------
# 6. NUMERIC CONVERSION
# ------------------------------------------------------------

numeric_value_columns = [
    "plan_price_inr",
    "arpu_last_month_inr",
    "arpu_3m_avg_inr",
    "arpu_6m_avg_inr",
    "tenure_months",
    "recharge_count_6m",
    "avg_recharge_gap_days",
    "days_since_last_recharge",
    "payment_failures_6m",
]


for col in numeric_value_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ------------------------------------------------------------
# 7. SELECT MONTHLY VALUE BASIS
# ------------------------------------------------------------

# Preference:
# 1. 6-month average ARPU
# 2. 3-month average ARPU
# 3. Last-month ARPU
#
# 6-month ARPU is preferred because it is less
# sensitive to one unusually high or low month.

monthly_value = pd.Series(
    np.nan,
    index=df.index,
    dtype=float
)


value_source = pd.Series(
    "Unavailable",
    index=df.index,
    dtype=object
)


if "arpu_6m_avg_inr" in df.columns:

    mask = (
        monthly_value.isna()
        &
        df[
            "arpu_6m_avg_inr"
        ].notna()
    )

    monthly_value.loc[
        mask
    ] = (
        df.loc[
            mask,
            "arpu_6m_avg_inr"
        ]
    )

    value_source.loc[
        mask
    ] = "6M Average ARPU"


if "arpu_3m_avg_inr" in df.columns:

    mask = (
        monthly_value.isna()
        &
        df[
            "arpu_3m_avg_inr"
        ].notna()
    )

    monthly_value.loc[
        mask
    ] = (
        df.loc[
            mask,
            "arpu_3m_avg_inr"
        ]
    )

    value_source.loc[
        mask
    ] = "3M Average ARPU"


if "arpu_last_month_inr" in df.columns:

    mask = (
        monthly_value.isna()
        &
        df[
            "arpu_last_month_inr"
        ].notna()
    )

    monthly_value.loc[
        mask
    ] = (
        df.loc[
            mask,
            "arpu_last_month_inr"
        ]
    )

    value_source.loc[
        mask
    ] = "Last-Month ARPU"


df[
    "monthly_value_basis_inr"
] = monthly_value


df[
    "monthly_value_source"
] = value_source


# ------------------------------------------------------------
# 8. CREATE 12-MONTH REVENUE PROXY
# ------------------------------------------------------------

# IMPORTANT:
# This is NOT observed CLV.
#
# It represents an annualized revenue proxy:
#
# monthly ARPU basis x 12 months
#
# It does NOT account for:
# - actual future churn
# - contribution margin
# - acquisition cost
# - retention cost
# - discounting
# - future plan changes

df[
    "revenue_proxy_12m_inr"
] = (
    df[
        "monthly_value_basis_inr"
    ]
    * 12
)


# ------------------------------------------------------------
# 9. CREATE VALUE BANDS
# ------------------------------------------------------------

valid_proxy = (
    df[
        "revenue_proxy_12m_inr"
    ]
    .dropna()
)


if len(valid_proxy) > 0:

    q25 = valid_proxy.quantile(
        0.25
    )

    q50 = valid_proxy.quantile(
        0.50
    )

    q75 = valid_proxy.quantile(
        0.75
    )


    def value_band(value):

        if pd.isna(value):

            return "Unknown"

        if value <= q25:

            return "Low Value"

        elif value <= q50:

            return "Lower-Mid Value"

        elif value <= q75:

            return "Upper-Mid Value"

        else:

            return "High Value"


    df[
        "customer_value_band"
    ] = (
        df[
            "revenue_proxy_12m_inr"
        ]
        .apply(
            value_band
        )
    )

else:

    q25 = np.nan
    q50 = np.nan
    q75 = np.nan

    df[
        "customer_value_band"
    ] = "Unknown"


# ------------------------------------------------------------
# 10. DATA QUALITY CHECKS
# ------------------------------------------------------------

missing_monthly_value = int(
    df[
        "monthly_value_basis_inr"
    ]
    .isna()
    .sum()
)


negative_monthly_value = int(
    (
        df[
            "monthly_value_basis_inr"
        ] < 0
    )
    .sum()
)


zero_monthly_value = int(
    (
        df[
            "monthly_value_basis_inr"
        ] == 0
    )
    .sum()
)


print(
    "\nCLV proxy data-quality review:"
)

print(
    f"Missing monthly value basis : "
    f"{missing_monthly_value:,}"
)

print(
    f"Negative monthly values     : "
    f"{negative_monthly_value:,}"
)

print(
    f"Zero monthly values         : "
    f"{zero_monthly_value:,}"
)


# ------------------------------------------------------------
# 11. VALUE SOURCE SUMMARY
# ------------------------------------------------------------

value_source_summary = (
    df[
        "monthly_value_source"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "value_source"
    )
    .reset_index(
        name="customers"
    )
)


VALUE_SOURCE_FILE = (
    REPORT_DIR
    / "21_clv_value_source_summary.csv"
)


value_source_summary.to_csv(
    VALUE_SOURCE_FILE,
    index=False
)


# ------------------------------------------------------------
# 12. VALUE-BAND SUMMARY
# ------------------------------------------------------------

value_band_summary = (
    df
    .groupby(
        "customer_value_band",
        dropna=False
    )
    .agg(
        customers=(
            "customer_value_band",
            "size"
        ),

        avg_monthly_value_inr=(
            "monthly_value_basis_inr",
            "mean"
        ),

        avg_revenue_proxy_12m_inr=(
            "revenue_proxy_12m_inr",
            "mean"
        ),

        median_revenue_proxy_12m_inr=(
            "revenue_proxy_12m_inr",
            "median"
        ),
    )
    .reset_index()
)


# Add churn rates only as descriptive context.
# Churn is NOT used to construct the value proxy.

if "churn_flag_30d" in df.columns:

    churn30 = (
        df[
            "churn_flag_30d"
        ]
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


    df[
        "_temp_churn30"
    ] = churn30


    churn_by_value = (
        df
        .groupby(
            "customer_value_band"
        )[
            "_temp_churn30"
        ]
        .mean()
        .mul(100)
        .rename(
            "churn_30d_pct"
        )
        .reset_index()
    )


    value_band_summary = (
        value_band_summary
        .merge(
            churn_by_value,
            on="customer_value_band",
            how="left"
        )
    )


    df = df.drop(
        columns=[
            "_temp_churn30"
        ]
    )


VALUE_BAND_FILE = (
    REPORT_DIR
    / "21_clv_value_band_summary.csv"
)


value_band_summary.to_csv(
    VALUE_BAND_FILE,
    index=False
)


# ------------------------------------------------------------
# 13. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

clv_stats = (
    df[
        [
            "monthly_value_basis_inr",
            "revenue_proxy_12m_inr",
        ]
    ]
    .describe()
    .round(2)
)


STATS_FILE = (
    REPORT_DIR
    / "21_clv_proxy_statistics.csv"
)


clv_stats.to_csv(
    STATS_FILE
)


# ------------------------------------------------------------
# 14. SAVE PREPARED CLV DATASET
# ------------------------------------------------------------

OUTPUT_COLUMNS = [

    "subscriber_id",

    "circle",

    "tenure_months",

    "plan_type",

    "plan_price_inr",

    "arpu_last_month_inr",

    "arpu_3m_avg_inr",

    "arpu_6m_avg_inr",

    "monthly_value_basis_inr",

    "monthly_value_source",

    "revenue_proxy_12m_inr",

    "customer_value_band",

    "recharge_count_6m",

    "avg_recharge_gap_days",

    "days_since_last_recharge",

    "payment_failures_6m",

    "churn_flag_30d",

    "churn_flag_90d",
]


OUTPUT_COLUMNS = [
    col
    for col in OUTPUT_COLUMNS
    if col in df.columns
]


clv_df = df[
    OUTPUT_COLUMNS
].copy()


CLV_FILE = (
    CLV_DATA_DIR
    / "clv_value_prepared.csv"
)


clv_df.to_csv(
    CLV_FILE,
    index=False
)


# ------------------------------------------------------------
# 15. METADATA
# ------------------------------------------------------------

metadata = {

    "true_clv_target_found":
        len(
            possible_true_targets
        ) > 0,

    "possible_true_target_columns":
        possible_true_targets,

    "proxy_name":
        "revenue_proxy_12m_inr",

    "proxy_formula":
        (
            "monthly_value_basis_inr * 12"
        ),

    "monthly_value_preference":
        [
            "arpu_6m_avg_inr",
            "arpu_3m_avg_inr",
            "arpu_last_month_inr",
        ],

    "proxy_limitations":
        [
            "Not observed future revenue",
            "Does not include contribution margin",
            "Does not include acquisition cost",
            "Does not include retention offer cost",
            "Does not explicitly model future churn",
            "Does not discount future cash flows",
        ],

    "value_band_quartiles": {
        "q25":
            (
                None
                if pd.isna(q25)
                else float(q25)
            ),

        "q50":
            (
                None
                if pd.isna(q50)
                else float(q50)
            ),

        "q75":
            (
                None
                if pd.isna(q75)
                else float(q75)
            ),
    },
}


METADATA_FILE = (
    CLV_DATA_DIR
    / "clv_metadata.json"
)


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
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
    "DAY 2 - STEP 21 CLV TARGET ASSESSMENT"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "CLV TARGET ASSESSMENT"
)

report.append(
    "-" * 100
)


if possible_true_targets:

    report.append(
        "Potential future-value target fields "
        "were detected and require manual review:"
    )

    for col in possible_true_targets:

        report.append(
            f"- {col}"
        )

else:

    report.append(
        "No genuine observed future CLV, "
        "future contribution, future margin, "
        "or next-12-month revenue target was "
        "found in the subscriber dataset."
    )


report.append("")

report.append(
    "DECISION"
)

report.append(
    "-" * 100
)

report.append(
    "A supervised CLV regression target will "
    "NOT be invented."
)

report.append(
    "Instead, this sandbox stage creates a "
    "transparent 12-month revenue proxy for "
    "customer-value prioritisation."
)

report.append("")

report.append(
    "PROXY DEFINITION"
)

report.append(
    "-" * 100
)

report.append(
    "Preferred monthly value basis:"
)

report.append(
    "1. 6-month average ARPU"
)

report.append(
    "2. 3-month average ARPU when 6-month "
    "ARPU is unavailable"
)

report.append(
    "3. Last-month ARPU when both averages "
    "are unavailable"
)

report.append("")

report.append(
    "12-month revenue proxy = "
    "monthly value basis x 12"
)

report.append("")

report.append(
    "This is an annualized revenue proxy, "
    "not realized customer lifetime value."
)

report.append("")

report.append(
    "VALUE-BAND SUMMARY"
)

report.append(
    "-" * 100
)

report.append(
    value_band_summary
    .round(2)
    .to_string(
        index=False
    )
)

report.append("")

report.append(
    "PROXY STATISTICS"
)

report.append(
    "-" * 100
)

report.append(
    clv_stats.to_string()
)

report.append("")

report.append(
    "LIMITATIONS"
)

report.append(
    "-" * 100
)

report.append(
    "The supplied dataset does not contain "
    "observed future 12-month revenue or "
    "contribution margin."
)

report.append(
    "Therefore the current proxy should be "
    "used for value segmentation and "
    "prioritisation, not presented as an "
    "observed financial CLV outcome."
)

report.append(
    "Future churn risk and retention economics "
    "can be combined with this value proxy in "
    "a later risk-value prioritisation stage."
)


REPORT_FILE = (
    REPORT_DIR
    / "21_clv_target_assessment_report.txt"
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
    "CLV TARGET ASSESSMENT"
)

print(
    "=" * 90
)


if possible_true_targets:

    print(
        "Potential future-value targets "
        "were found:"
    )

    for col in possible_true_targets:
        print(f"- {col}")

else:

    print(
        "No genuine future-value / CLV "
        "target was found."
    )

    print(
        "Using a transparent annualized "
        "revenue proxy instead."
    )


print(
    "\nVALUE SOURCE SUMMARY"
)

print(
    value_source_summary.to_string(
        index=False
    )
)


print(
    "\nVALUE BAND SUMMARY"
)

print(
    value_band_summary
    .round(2)
    .to_string(
        index=False
    )
)


print(
    "\nGenerated files:"
)

for file in [
    CLV_FILE,
    METADATA_FILE,
    VALUE_SOURCE_FILE,
    VALUE_BAND_FILE,
    STATS_FILE,
    REPORT_FILE,
]:

    print(
        f"- {file.relative_to(PROJECT_ROOT)}"
    )


print(
    "\n" + "=" * 90
)

print(
    "STEP 21 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nNo CLV regression model has been "
    "trained yet."
)