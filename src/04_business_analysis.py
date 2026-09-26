from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# STEP 04 - BUSINESS / CHURN ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "subscribers":
        PROCESSED_DIR / "subscribers_clean.csv",

    "service_requests":
        PROCESSED_DIR / "service_requests_clean.csv",

    "network_sites":
        PROCESSED_DIR / "network_sites_clean.csv",

    "circle_monthly_kpi":
        PROCESSED_DIR / "circle_monthly_kpi_clean.csv",

    "circle_targets":
        PROCESSED_DIR / "circle_targets_clean.csv",
}


# ------------------------------------------------------------
# 2. CHECK REQUIRED FILES
# ------------------------------------------------------------

for name, path in FILES.items():

    if not path.exists():

        raise FileNotFoundError(
            f"Required processed file not found:\n{path}"
        )


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

subs = pd.read_csv(
    FILES["subscribers"]
)

sr = pd.read_csv(
    FILES["service_requests"]
)

sites = pd.read_csv(
    FILES["network_sites"]
)

kpi = pd.read_csv(
    FILES["circle_monthly_kpi"]
)

targets = pd.read_csv(
    FILES["circle_targets"]
)


# ------------------------------------------------------------
# 4. DATE CONVERSION
# ------------------------------------------------------------

for col in [
    "join_date",
    "churn_date",
]:

    if col in subs.columns:

        subs[col] = pd.to_datetime(
            subs[col],
            errors="coerce"
        )


for col in [
    "raised_date",
    "resolved_date",
]:

    if col in sr.columns:

        sr[col] = pd.to_datetime(
            sr[col],
            errors="coerce"
        )


if "month_end" in kpi.columns:

    kpi["month_end"] = pd.to_datetime(
        kpi["month_end"],
        errors="coerce"
    )


# ------------------------------------------------------------
# 5. PREPARE 30-DAY CHURN TARGET
# ------------------------------------------------------------

TARGET = "churn_flag_30d"


if TARGET not in subs.columns:

    raise ValueError(
        f"{TARGET} not found in subscriber dataset."
    )


if subs[TARGET].dtype == "bool":

    subs["_churn30"] = (
        subs[TARGET]
        .astype(int)
    )

else:

    subs["_churn30"] = (
        subs[TARGET]
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


if subs["_churn30"].isna().any():

    raise ValueError(
        "Could not convert all churn_flag_30d "
        "values to 0/1."
    )


# ------------------------------------------------------------
# 6. HELPER FUNCTION
# ------------------------------------------------------------

def churn_summary(
    df,
    group_col,
    label_name=None
):

    result = (
        df.groupby(
            group_col,
            dropna=False,
            observed=False
        )
        .agg(
            subscribers=(
                "subscriber_id",
                "size"
            ),

            churners_30d=(
                "_churn30",
                "sum"
            ),

            churn_rate_30d_pct=(
                "_churn30",
                "mean"
            ),
        )
        .reset_index()
    )

    result[
        "churn_rate_30d_pct"
    ] = (
        result[
            "churn_rate_30d_pct"
        ]
        * 100
    ).round(2)

    if (
        label_name
        and
        group_col != label_name
    ):

        result = result.rename(
            columns={
                group_col: label_name
            }
        )

    return result


# ------------------------------------------------------------
# 7. PROJECT HEADER
# ------------------------------------------------------------

print("=" * 90)
print(
    "JIO SUBSCRIBER RETENTION PROJECT"
)
print(
    "STEP 04 - BUSINESS / CHURN ANALYSIS"
)
print("=" * 90)


overall_churners = int(
    subs["_churn30"].sum()
)

overall_rate = (
    subs["_churn30"].mean()
    * 100
)


# ============================================================
# PART 1
# PROBLEMATIC REGIONS / CIRCLES
# ============================================================

region = (
    subs.groupby(
        [
            "circle",
            "circle_code",
            "zone",
        ],
        dropna=False
    )
    .agg(
        subscribers=(
            "subscriber_id",
            "size"
        ),

        churners_30d=(
            "_churn30",
            "sum"
        ),

        churn_rate_30d_pct=(
            "_churn30",
            "mean"
        ),

        avg_arpu_inr=(
            "arpu_last_month_inr",
            "mean"
        ),

        avg_complaints_6m=(
            "complaints_6m",
            "mean"
        ),

        avg_unresolved_complaints=(
            "unresolved_complaints",
            "mean"
        ),

        avg_sinr_db=(
            "avg_sinr_db",
            "mean"
        ),

        avg_congestion_score=(
            "site_congestion_score",
            "mean"
        ),
    )
    .reset_index()
)


region[
    "churn_rate_30d_pct"
] = (
    region[
        "churn_rate_30d_pct"
    ]
    * 100
).round(2)


for col in [
    "avg_arpu_inr",
    "avg_complaints_6m",
    "avg_unresolved_complaints",
    "avg_sinr_db",
    "avg_congestion_score",
]:

    region[col] = (
        region[col]
        .round(2)
    )


# ------------------------------------------------------------
# Latest circle KPI
# ------------------------------------------------------------

latest_month = (
    kpi["month_end"]
    .max()
)


latest_kpi = (
    kpi.loc[
        kpi["month_end"].eq(
            latest_month
        ),
        [
            "circle",
            "closing_base",
            "churned_subscribers",
            "monthly_churn_pct",
            "arpu_inr",
            "port_out_requests",
            "complaints_logged",
        ]
    ]
    .copy()
)


latest_kpi = latest_kpi.rename(
    columns={
        "monthly_churn_pct":
            "latest_kpi_churn_pct",

        "arpu_inr":
            "latest_kpi_arpu_inr",
    }
)


region = region.merge(
    latest_kpi,
    on="circle",
    how="left"
)


# ------------------------------------------------------------
# Add FY27 churn target information
# ------------------------------------------------------------

if "churn_ceiling_fy27_pct" in targets.columns:

    region = region.merge(
        targets[
            [
                "circle",
                "churn_ceiling_fy27_pct",
                "retention_budget_lakh",
                "budget_utilised_lakh",
                "review_status",
            ]
        ],
        on="circle",
        how="left"
    )

    region[
        "above_churn_ceiling_pct_points"
    ] = (
        region[
            "latest_kpi_churn_pct"
        ]
        -
        region[
            "churn_ceiling_fy27_pct"
        ]
    ).round(2)


region = (
    region.sort_values(
        [
            "churn_rate_30d_pct",
            "latest_kpi_churn_pct",
        ],
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


region.insert(
    0,
    "sample_churn_rank",
    np.arange(
        1,
        len(region) + 1
    )
)


REGION_FILE = (
    REPORT_DIR
    / "04_region_churn_analysis.csv"
)

region.to_csv(
    REGION_FILE,
    index=False
)


# ============================================================
# PART 2
# MONTHLY CHURN TREND
# ============================================================

trend = (
    kpi.groupby(
        "month_end",
        as_index=False
    )
    .agg(
        closing_base=(
            "closing_base",
            "sum"
        ),

        gross_adds=(
            "gross_adds",
            "sum"
        ),

        churned_subscribers=(
            "churned_subscribers",
            "sum"
        ),

        net_adds=(
            "net_adds",
            "sum"
        ),

        port_in_requests=(
            "port_in_requests",
            "sum"
        ),

        port_out_requests=(
            "port_out_requests",
            "sum"
        ),

        complaints_logged=(
            "complaints_logged",
            "sum"
        ),
    )
)


trend[
    "weighted_churn_pct"
] = (
    trend[
        "churned_subscribers"
    ]
    /
    trend[
        "closing_base"
    ]
    * 100
).round(2)


# ------------------------------------------------------------
# Weighted ARPU
# ------------------------------------------------------------

kpi["_arpu_x_base"] = (
    kpi["arpu_inr"]
    *
    kpi["closing_base"]
)


weighted_arpu = (
    kpi.groupby(
        "month_end",
        as_index=False
    )
    .agg(
        arpu_x_base=(
            "_arpu_x_base",
            "sum"
        ),

        base=(
            "closing_base",
            "sum"
        ),
    )
)


weighted_arpu[
    "weighted_arpu_inr"
] = (
    weighted_arpu[
        "arpu_x_base"
    ]
    /
    weighted_arpu[
        "base"
    ]
).round(2)


trend = trend.merge(
    weighted_arpu[
        [
            "month_end",
            "weighted_arpu_inr",
        ]
    ],
    on="month_end",
    how="left"
)


trend[
    "net_porting"
] = (
    trend[
        "port_in_requests"
    ]
    -
    trend[
        "port_out_requests"
    ]
)


TREND_FILE = (
    REPORT_DIR
    / "04_monthly_churn_trend.csv"
)

trend.to_csv(
    TREND_FILE,
    index=False
)


# ============================================================
# PART 3
# TENURE / CHURN TIMELINE
# ============================================================

tenure_bins = [
    -1,
    3,
    6,
    12,
    24,
    36,
    60,
    np.inf,
]


tenure_labels = [
    "0-3 months",
    "4-6 months",
    "7-12 months",
    "13-24 months",
    "25-36 months",
    "37-60 months",
    "61+ months",
]


subs[
    "tenure_band"
] = pd.cut(
    subs[
        "tenure_months"
    ],
    bins=tenure_bins,
    labels=tenure_labels
)


tenure = churn_summary(
    subs,
    "tenure_band"
)


TENURE_FILE = (
    REPORT_DIR
    / "04_tenure_churn_analysis.csv"
)

tenure.to_csv(
    TENURE_FILE,
    index=False
)


# ============================================================
# PART 4
# COMPLAINT ANALYSIS
# ============================================================

subs[
    "complaint_band"
] = np.select(
    [
        subs[
            "complaints_6m"
        ].eq(0),

        subs[
            "complaints_6m"
        ].eq(1),

        subs[
            "complaints_6m"
        ].ge(2),
    ],

    [
        "0 complaints",
        "1 complaint",
        "2+ complaints",
    ],

    default="Unknown"
)


subs[
    "unresolved_band"
] = np.select(
    [
        subs[
            "unresolved_complaints"
        ].eq(0),

        subs[
            "unresolved_complaints"
        ].eq(1),

        subs[
            "unresolved_complaints"
        ].ge(2),
    ],

    [
        "0 unresolved",
        "1 unresolved",
        "2+ unresolved",
    ],

    default="Unknown"
)


complaint_parts = []


temp = churn_summary(
    subs,
    "complaint_band",
    "segment"
)

temp.insert(
    0,
    "analysis",
    "Complaints in last 6 months"
)

complaint_parts.append(
    temp
)


temp = churn_summary(
    subs,
    "unresolved_band",
    "segment"
)

temp.insert(
    0,
    "analysis",
    "Unresolved complaints"
)

complaint_parts.append(
    temp
)


complaint_analysis = pd.concat(
    complaint_parts,
    ignore_index=True
)


COMPLAINT_FILE = (
    REPORT_DIR
    / "04_complaint_churn_analysis.csv"
)

complaint_analysis.to_csv(
    COMPLAINT_FILE,
    index=False
)


# ============================================================
# PART 5
# SERVICE REQUEST / SLA ANALYSIS
# ============================================================

def bool_to_int(series):

    if series.dtype == "bool":

        return series.astype(
            int
        )

    return (
        series.astype(str)
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
        .fillna(0)
        .astype(int)
    )


for col in [
    "sla_breach_flag",
    "reopened_flag",
]:

    sr[
        f"_{col}"
    ] = bool_to_int(
        sr[col]
    )


sr[
    "_open_flag"
] = (
    sr["status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .ne("closed")
    .astype(int)
)


sr_agg = (
    sr.groupby(
        "subscriber_id",
        as_index=False
    )
    .agg(
        service_request_count=(
            "sr_id",
            "count"
        ),

        sla_breach_count=(
            "_sla_breach_flag",
            "sum"
        ),

        reopened_count=(
            "_reopened_flag",
            "sum"
        ),

        open_request_count=(
            "_open_flag",
            "sum"
        ),

        avg_csat_score=(
            "csat_score",
            "mean"
        ),
    )
)


subs_sr = subs.merge(
    sr_agg,
    on="subscriber_id",
    how="left"
)


for col in [
    "service_request_count",
    "sla_breach_count",
    "reopened_count",
    "open_request_count",
]:

    subs_sr[col] = (
        subs_sr[col]
        .fillna(0)
        .astype(int)
    )


subs_sr[
    "sr_count_band"
] = np.select(
    [
        subs_sr[
            "service_request_count"
        ].eq(0),

        subs_sr[
            "service_request_count"
        ].eq(1),

        subs_sr[
            "service_request_count"
        ].ge(2),
    ],

    [
        "0 requests",
        "1 request",
        "2+ requests",
    ],

    default="Unknown"
)


subs_sr[
    "sla_breach_band"
] = np.where(
    subs_sr[
        "sla_breach_count"
    ].ge(1),

    "1+ SLA breach",
    "No SLA breach"
)


subs_sr[
    "open_request_band"
] = np.where(
    subs_sr[
        "open_request_count"
    ].ge(1),

    "1+ open request",
    "No open request"
)


subs_sr[
    "reopened_band"
] = np.where(
    subs_sr[
        "reopened_count"
    ].ge(1),

    "1+ reopened request",
    "No reopened request"
)


service_parts = []


for source_col, label in [
    (
        "sr_count_band",
        "Service request count"
    ),

    (
        "sla_breach_band",
        "SLA breach"
    ),

    (
        "open_request_band",
        "Open service request"
    ),

    (
        "reopened_band",
        "Reopened service request"
    ),
]:

    temp = churn_summary(
        subs_sr,
        source_col,
        "segment"
    )

    temp.insert(
        0,
        "analysis",
        label
    )

    service_parts.append(
        temp
    )


service_analysis = pd.concat(
    service_parts,
    ignore_index=True
)


SERVICE_FILE = (
    REPORT_DIR
    / "04_service_request_churn_analysis.csv"
)

service_analysis.to_csv(
    SERVICE_FILE,
    index=False
)


# ------------------------------------------------------------
# Service request category analysis
# ------------------------------------------------------------

sr_category = (
    sr[
        [
            "subscriber_id",
            "sr_category",
        ]
    ]
    .drop_duplicates()
    .merge(
        subs[
            [
                "subscriber_id",
                "_churn30",
            ]
        ],
        on="subscriber_id",
        how="left"
    )
)


sr_category_analysis = (
    sr_category.groupby(
        "sr_category",
        dropna=False
    )
    .agg(
        subscribers=(
            "subscriber_id",
            "nunique"
        ),

        churners_30d=(
            "_churn30",
            "sum"
        ),

        churn_rate_30d_pct=(
            "_churn30",
            "mean"
        ),
    )
    .reset_index()
)


sr_category_analysis[
    "churn_rate_30d_pct"
] = (
    sr_category_analysis[
        "churn_rate_30d_pct"
    ]
    * 100
).round(2)


SR_CATEGORY_FILE = (
    REPORT_DIR
    / "04_service_request_category_analysis.csv"
)

sr_category_analysis.to_csv(
    SR_CATEGORY_FILE,
    index=False
)


# ============================================================
# PART 6
# NETWORK QUALITY VS CHURN
# ============================================================

subs[
    "sinr_band"
] = pd.cut(
    subs[
        "avg_sinr_db"
    ],

    bins=[
        -np.inf,
        5,
        15,
        np.inf,
    ],

    labels=[
        "<5 dB",
        "5-14.9 dB",
        "15+ dB",
    ],

    right=False
)


subs[
    "congestion_band"
] = pd.cut(
    subs[
        "site_congestion_score"
    ],

    bins=[
        -np.inf,
        30,
        70,
        np.inf,
    ],

    labels=[
        "<30",
        "30-69.9",
        "70+",
    ],

    right=False
)


subs[
    "drop_call_band"
] = pd.cut(
    subs[
        "drop_call_rate_pct"
    ],

    bins=[
        -np.inf,
        0.5,
        1.0,
        np.inf,
    ],

    labels=[
        "<0.5%",
        "0.5-0.99%",
        "1.0%+",
    ],

    right=False
)


network_parts = []


for source_col, label in [
    (
        "sinr_band",
        "SINR quality"
    ),

    (
        "congestion_band",
        "Congestion score"
    ),

    (
        "drop_call_band",
        "Drop-call rate"
    ),
]:

    temp = churn_summary(
        subs,
        source_col,
        "segment"
    )

    temp.insert(
        0,
        "analysis",
        label
    )

    network_parts.append(
        temp
    )


network_analysis = pd.concat(
    network_parts,
    ignore_index=True
)


NETWORK_FILE = (
    REPORT_DIR
    / "04_network_churn_analysis.csv"
)

network_analysis.to_csv(
    NETWORK_FILE,
    index=False
)


# ------------------------------------------------------------
# Circle-level network infrastructure
# ------------------------------------------------------------

site_circle = (
    sites.groupby(
        "circle",
        as_index=False
    )
    .agg(
        site_count=(
            "site_id",
            "count"
        ),

        subscribers_served=(
            "subscribers_served",
            "sum"
        ),

        avg_prb_utilisation_pct=(
            "prb_utilisation_pct",
            "mean"
        ),

        avg_site_sinr_db=(
            "avg_sinr_db",
            "mean"
        ),

        avg_site_drop_call_pct=(
            "drop_call_rate_pct",
            "mean"
        ),

        avg_throughput_mbps=(
            "avg_throughput_mbps",
            "mean"
        ),

        congested_sites=(
            "congestion_flag",
            "sum"
        ),
    )
)


site_circle[
    "congested_site_pct"
] = (
    site_circle[
        "congested_sites"
    ]
    /
    site_circle[
        "site_count"
    ]
    * 100
).round(2)


site_circle = site_circle.merge(
    latest_kpi[
        [
            "circle",
            "latest_kpi_churn_pct",
        ]
    ],
    on="circle",
    how="left"
)


for col in [
    "avg_prb_utilisation_pct",
    "avg_site_sinr_db",
    "avg_site_drop_call_pct",
    "avg_throughput_mbps",
]:

    site_circle[col] = (
        site_circle[col]
        .round(2)
    )


SITE_FILE = (
    REPORT_DIR
    / "04_circle_network_infrastructure.csv"
)

site_circle.to_csv(
    SITE_FILE,
    index=False
)


# ============================================================
# PART 7
# IMPORTANT CUSTOMER PATTERNS
# ============================================================

# ARPU
subs[
    "arpu_band"
] = pd.cut(
    subs[
        "arpu_last_month_inr"
    ],

    bins=[
        -np.inf,
        150,
        250,
        np.inf,
    ],

    labels=[
        "<₹150",
        "₹150-₹249.99",
        "₹250+",
    ],

    right=False
)


# Days since recharge
subs[
    "days_since_recharge_band"
] = pd.cut(
    subs[
        "days_since_last_recharge"
    ],

    bins=[
        -1,
        7,
        30,
        60,
        np.inf,
    ],

    labels=[
        "0-7 days",
        "8-30 days",
        "31-60 days",
        "61+ days",
    ]
)


# Average recharge gap
subs[
    "recharge_gap_band"
] = pd.cut(
    subs[
        "avg_recharge_gap_days"
    ],

    bins=[
        -np.inf,
        30,
        60,
        90,
        np.inf,
    ],

    labels=[
        "<30 days",
        "30-59 days",
        "60-89 days",
        "90+ days",
    ],

    right=False
)


# Payment failures
subs[
    "payment_failure_band"
] = np.select(
    [
        subs[
            "payment_failures_6m"
        ].eq(0),

        subs[
            "payment_failures_6m"
        ].eq(1),

        subs[
            "payment_failures_6m"
        ].ge(2),
    ],

    [
        "0 failures",
        "1 failure",
        "2+ failures",
    ],

    default="Unknown"
)


pattern_parts = []


for source_col, label in [
    (
        "arpu_band",
        "ARPU band"
    ),

    (
        "days_since_recharge_band",
        "Days since last recharge"
    ),

    (
        "recharge_gap_band",
        "Average recharge gap"
    ),

    (
        "payment_failure_band",
        "Payment failures"
    ),

    (
        "plan_type",
        "Plan type"
    ),

    (
        "is_5g_active",
        "5G active"
    ),

    (
        "autopay_enabled",
        "Autopay enabled"
    ),

    (
        "family_plan_flag",
        "Family plan"
    ),

    (
        "num_services",
        "Number of services"
    ),
]:

    if source_col not in subs.columns:
        continue

    temp = churn_summary(
        subs,
        source_col,
        "segment"
    )

    temp.insert(
        0,
        "analysis",
        label
    )

    pattern_parts.append(
        temp
    )


customer_patterns = pd.concat(
    pattern_parts,
    ignore_index=True
)


PATTERN_FILE = (
    REPORT_DIR
    / "04_customer_pattern_analysis.csv"
)

customer_patterns.to_csv(
    PATTERN_FILE,
    index=False
)


# ============================================================
# PART 8
# CHURN REASONS
# DESCRIPTIVE ONLY - NOT FOR MODELLING
# ============================================================

churn_reason_analysis = (
    subs[
        subs[
            "churn_reason"
        ].notna()
    ]
    .groupby(
        "churn_reason",
        as_index=False
    )
    .agg(
        customers=(
            "subscriber_id",
            "size"
        )
    )
    .sort_values(
        "customers",
        ascending=False
    )
)


total_reasons = (
    churn_reason_analysis[
        "customers"
    ]
    .sum()
)


if total_reasons > 0:

    churn_reason_analysis[
        "share_pct"
    ] = (
        churn_reason_analysis[
            "customers"
        ]
        /
        total_reasons
        * 100
    ).round(2)


CHURN_REASON_FILE = (
    REPORT_DIR
    / "04_churn_reason_analysis.csv"
)

churn_reason_analysis.to_csv(
    CHURN_REASON_FILE,
    index=False
)


# ============================================================
# PART 9
# NUMERIC CORRELATION WITH 30-DAY CHURN
# ============================================================

correlation_fields = [
    "tenure_months",
    "plan_price_inr",
    "arpu_last_month_inr",
    "arpu_3m_avg_inr",
    "recharge_count_6m",
    "avg_recharge_gap_days",
    "days_since_last_recharge",
    "payment_failures_6m",
    "data_gb_last_month",
    "voice_minutes_last_month",
    "avg_sinr_db",
    "drop_call_rate_pct",
    "site_congestion_score",
    "complaints_6m",
    "unresolved_complaints",
    "avg_resolution_days",
    "app_logins_30d",
    "outgoing_to_competitor_pct",
]

correlation_rows = []

for col in correlation_fields:

    if col not in subs.columns:
        continue

    pair = subs[
        [
            col,
            "_churn30",
        ]
    ].copy()

    # Defensive numeric conversion
    pair[col] = pd.to_numeric(
        pair[col],
        errors="coerce"
    )

    pair["_churn30"] = pd.to_numeric(
        pair["_churn30"],
        errors="coerce"
    )

    pair = pair.dropna()

    if (
        len(pair) == 0
        or pair[col].nunique() <= 1
    ):
        continue

    correlation = pair[col].corr(
        pair["_churn30"]
    )

    correlation_rows.append(
        {
            "feature": col,
            "correlation_with_churn_30d": round(
                float(correlation),
                4
            ),
            "absolute_correlation": round(
                abs(float(correlation)),
                4
            ),
            "non_missing_rows": len(pair),
        }
    )


correlations = pd.DataFrame(
    correlation_rows
)

if not correlations.empty:

    correlations = (
        correlations
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .reset_index(drop=True)
    )


CORRELATION_FILE = (
    REPORT_DIR
    / "04_churn_correlations.csv"
)

correlations.to_csv(
    CORRELATION_FILE,
    index=False
)


# ============================================================
# PART 10
# BUILD BUSINESS REPORT
# ============================================================

top_regions = (
    region.head(5)
)


report = []


report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "STEP 04 - BUSINESS / CHURN ANALYSIS REPORT"
)

report.append(
    "=" * 100
)


report.append("")

report.append(
    f"Subscriber sample: "
    f"{len(subs):,}"
)

report.append(
    f"30-day churners: "
    f"{overall_churners:,}"
)

report.append(
    f"30-day churn rate: "
    f"{overall_rate:.2f}%"
)

report.append(
    f"Latest circle KPI month: "
    f"{latest_month.date()}"
)


# ------------------------------------------------------------
# Regions
# ------------------------------------------------------------

report.append("")

report.append(
    "=" * 100
)

report.append(
    "1. PROBLEMATIC REGIONS / CIRCLES"
)

report.append(
    "=" * 100
)


report.append(
    top_regions[
        [
            "sample_churn_rank",
            "circle",
            "subscribers",
            "churners_30d",
            "churn_rate_30d_pct",
            "latest_kpi_churn_pct",
            "port_out_requests",
        ]
    ].to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Tenure
# ------------------------------------------------------------

report.append("")

report.append(
    "=" * 100
)

report.append(
    "2. CHURN TIMELINE / TENURE"
)

report.append(
    "=" * 100
)

report.append(
    tenure.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Complaints
# ------------------------------------------------------------

report.append("")

report.append(
    "=" * 100
)

report.append(
    "3. COMPLAINT PATTERNS"
)

report.append(
    "=" * 100
)

report.append(
    complaint_analysis.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Service Requests
# ------------------------------------------------------------

report.append("")

report.append(
    "=" * 100
)

report.append(
    "4. SERVICE REQUEST / SLA PATTERNS"
)

report.append(
    "=" * 100
)

report.append(
    service_analysis.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Network
# ------------------------------------------------------------

report.append("")

report.append(
    "=" * 100
)

report.append(
    "5. NETWORK QUALITY PATTERNS"
)

report.append(
    "=" * 100
)

report.append(
    network_analysis.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Customer patterns
# ------------------------------------------------------------

report.append("")

report.append(
    "=" * 100
)

report.append(
    "6. IMPORTANT CUSTOMER PATTERNS"
)

report.append(
    "=" * 100
)

report.append(
    customer_patterns.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Correlations
# ------------------------------------------------------------

report.append("")

report.append(
    "=" * 100
)

report.append(
    "7. TOP NUMERIC CORRELATIONS WITH 30-DAY CHURN"
)

report.append(
    "=" * 100
)

report.append(
    correlations.head(
        15
    ).to_string(
        index=False
    )
)


report.append("")

report.append(
    "Important: correlation shows association, "
    "not proof of causation."
)


# ------------------------------------------------------------
# Leakage note
# ------------------------------------------------------------

report.append("")

report.append(
    "=" * 100
)

report.append(
    "8. MODELLING CAUTION"
)

report.append(
    "=" * 100
)


report.append(
    "mnp_enquiry_flag, churn_reason and churn_date "
    "were NOT included in the churn correlation "
    "analysis because they represent target-leakage risk."
)


report.append(
    "churn_reason is used only for descriptive "
    "analysis of customers already known to churn."
)


REPORT_FILE = (
    REPORT_DIR
    / "04_business_analysis_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


# ============================================================
# PART 11
# TERMINAL SUMMARY
# ============================================================

print(
    f"\nOverall 30-day churn: "
    f"{overall_churners:,} / "
    f"{len(subs):,} "
    f"({overall_rate:.2f}%)"
)


print(
    "\nTOP 5 PROBLEMATIC CIRCLES"
)

print(
    top_regions[
        [
            "sample_churn_rank",
            "circle",
            "subscribers",
            "churn_rate_30d_pct",
            "latest_kpi_churn_pct",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nTENURE VS CHURN"
)

print(
    tenure.to_string(
        index=False
    )
)


print(
    "\nCOMPLAINTS VS CHURN"
)

print(
    complaint_analysis.to_string(
        index=False
    )
)


print(
    "\nNETWORK QUALITY VS CHURN"
)

print(
    network_analysis.to_string(
        index=False
    )
)


print(
    "\nTOP 10 NUMERIC CORRELATIONS "
    "WITH 30-DAY CHURN"
)

print(
    correlations[
        [
            "feature",
            "correlation_with_churn_30d",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 04 COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nGenerated analysis files:"
)


generated_files = [

    REGION_FILE,

    TREND_FILE,

    TENURE_FILE,

    COMPLAINT_FILE,

    SERVICE_FILE,

    SR_CATEGORY_FILE,

    NETWORK_FILE,

    SITE_FILE,

    PATTERN_FILE,

    CHURN_REASON_FILE,

    CORRELATION_FILE,

    REPORT_FILE,
]


for file in generated_files:

    print(
        f"- "
        f"{file.relative_to(PROJECT_ROOT)}"
    )


print(
    "\nNo predictive model "
    "has been trained in Step 04."
)