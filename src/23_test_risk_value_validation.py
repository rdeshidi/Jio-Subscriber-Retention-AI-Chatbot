from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 23
# OUT-OF-SAMPLE RISK-VALUE VALIDATION
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
CLV_DIR = PROJECT_ROOT / "data" / "clv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


PREDICTION_FILE = (
    REPORT_DIR
    / "19_final_test_predictions.csv"
)

CLV_METADATA_FILE = (
    CLV_DIR
    / "clv_metadata.json"
)


print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 23: TEST RISK-VALUE VALIDATION")
print("=" * 90)


# ------------------------------------------------------------
# 1. LOAD UNTOUCHED TEST PREDICTIONS
# ------------------------------------------------------------

test = pd.read_csv(
    PREDICTION_FILE
)


with open(
    CLV_METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)


print(
    f"\nTest customers : {len(test):,}"
)


# ------------------------------------------------------------
# 2. CREATE SAME 12-MONTH VALUE PROXY
# ------------------------------------------------------------

test[
    "monthly_value_basis_inr"
] = pd.to_numeric(
    test[
        "arpu_6m_avg_inr"
    ],
    errors="coerce"
)


test[
    "revenue_proxy_12m_inr"
] = (
    test[
        "monthly_value_basis_inr"
    ]
    * 12
)


# ------------------------------------------------------------
# 3. APPLY ORIGINAL FULL-DATA VALUE BAND CUTS
# ------------------------------------------------------------

q25 = (
    metadata[
        "value_band_quartiles"
    ][
        "q25"
    ]
)

q50 = (
    metadata[
        "value_band_quartiles"
    ][
        "q50"
    ]
)

q75 = (
    metadata[
        "value_band_quartiles"
    ][
        "q75"
    ]
)


def assign_value_band(value):

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


test[
    "customer_value_band"
] = (
    test[
        "revenue_proxy_12m_inr"
    ]
    .apply(
        assign_value_band
    )
)


# ------------------------------------------------------------
# 4. TEST-SET RISK DECILES
# ------------------------------------------------------------

test[
    "risk_rank"
] = (
    test[
        "churn_probability"
    ]
    .rank(
        ascending=False,
        method="first"
    )
)


test[
    "risk_decile"
] = np.minimum(
    np.ceil(
        test[
            "risk_rank"
        ]
        / len(test)
        * 10
    ),
    10
).astype(int)


# ------------------------------------------------------------
# 5. ASSIGN SAME RETENTION SEGMENTS
# ------------------------------------------------------------

def assign_priority(row):

    risk_decile = row[
        "risk_decile"
    ]

    value_band = row[
        "customer_value_band"
    ]


    if (
        risk_decile == 1
        and value_band == "High Value"
    ):

        return (
            "Priority 1 - High Risk / High Value"
        )


    elif (
        risk_decile == 1
        and value_band == "Upper-Mid Value"
    ):

        return (
            "Priority 2 - High Risk / Upper-Mid Value"
        )


    elif risk_decile == 1:

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

        return "Standard Monitoring"


test[
    "retention_priority_segment"
] = test.apply(
    assign_priority,
    axis=1
)


# ------------------------------------------------------------
# 6. SEGMENT SUMMARY
# ------------------------------------------------------------

segment_summary = (
    test
    .groupby(
        "retention_priority_segment"
    )
    .agg(
        customers=(
            "actual_churn",
            "size"
        ),

        actual_churners=(
            "actual_churn",
            "sum"
        ),

        actual_churn_rate=(
            "actual_churn",
            "mean"
        ),

        avg_risk_score=(
            "churn_probability",
            "mean"
        ),

        avg_monthly_value_inr=(
            "monthly_value_basis_inr",
            "mean"
        ),

        avg_12m_revenue_proxy_inr=(
            "revenue_proxy_12m_inr",
            "mean"
        ),
    )
    .reset_index()
)


segment_summary[
    "actual_churn_pct"
] = (
    segment_summary[
        "actual_churn_rate"
    ]
    * 100
)


segment_summary = (
    segment_summary
    .drop(
        columns=[
            "actual_churn_rate"
        ]
    )
)


# ------------------------------------------------------------
# 7. TOP-RISK DECILE PERFORMANCE
# ------------------------------------------------------------

top_decile = test[
    test[
        "risk_decile"
    ] == 1
]


total_churners = int(
    test[
        "actual_churn"
    ].sum()
)


top_decile_churners = int(
    top_decile[
        "actual_churn"
    ].sum()
)


top_decile_recall = (
    top_decile_churners
    / total_churners
)


overall_churn_rate = (
    test[
        "actual_churn"
    ].mean()
)


top_decile_churn_rate = (
    top_decile[
        "actual_churn"
    ].mean()
)


top_decile_lift = (
    top_decile_churn_rate
    / overall_churn_rate
)


# ------------------------------------------------------------
# 8. PRIORITY 1 SPECIFIC RESULTS
# ------------------------------------------------------------

priority1 = test[
    test[
        "retention_priority_segment"
    ]
    ==
    "Priority 1 - High Risk / High Value"
]


priority1_customers = len(
    priority1
)


priority1_churners = int(
    priority1[
        "actual_churn"
    ].sum()
)


priority1_churn_rate = (
    priority1[
        "actual_churn"
    ].mean()
    if priority1_customers > 0
    else 0
)


# ------------------------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------------------------

SUMMARY_FILE = (
    REPORT_DIR
    / "23_test_risk_value_segment_summary.csv"
)


segment_summary.to_csv(
    SUMMARY_FILE,
    index=False
)


DETAIL_FILE = (
    REPORT_DIR
    / "23_test_risk_value_details.csv"
)


test.to_csv(
    DETAIL_FILE,
    index=False
)


# ------------------------------------------------------------
# 10. CHART - CHURN RATE BY SEGMENT
# ------------------------------------------------------------

chart_df = (
    segment_summary
    .sort_values(
        "actual_churn_pct",
        ascending=False
    )
)


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    chart_df[
        "retention_priority_segment"
    ],
    chart_df[
        "actual_churn_pct"
    ]
)


plt.ylabel(
    "Actual 30-Day Churn Rate (%)"
)

plt.xlabel(
    "Retention Priority Segment"
)

plt.title(
    "Out-of-Sample Churn Rate by Risk-Value Segment"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()


CHART_FILE = (
    CHART_DIR
    / "24_test_risk_value_segment_churn.png"
)


plt.savefig(
    CHART_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 11. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "STEP 23 - OUT-OF-SAMPLE RISK-VALUE VALIDATION"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "This analysis uses only the previously "
    "untouched 20% test population."
)

report.append("")

report.append(
    "SEGMENT RESULTS"
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
    "TOP RISK DECILE"
)

report.append(
    "-" * 100
)

report.append(
    f"Customers: {len(top_decile):,}"
)

report.append(
    f"Actual churners captured: "
    f"{top_decile_churners:,} "
    f"of {total_churners:,}"
)

report.append(
    f"Recall@Top10%: "
    f"{top_decile_recall:.4f}"
)

report.append(
    f"Lift@Top10%: "
    f"{top_decile_lift:.2f}x"
)

report.append("")

report.append(
    "PRIORITY 1 - HIGH RISK / HIGH VALUE"
)

report.append(
    "-" * 100
)

report.append(
    f"Customers: "
    f"{priority1_customers:,}"
)

report.append(
    f"Actual churners: "
    f"{priority1_churners:,}"
)

report.append(
    f"Actual churn rate: "
    f"{priority1_churn_rate * 100:.2f}%"
)

report.append("")

report.append(
    "INTERPRETATION"
)

report.append(
    "-" * 100
)

report.append(
    "This test-only analysis provides the cleanest "
    "estimate of how the risk-value segmentation "
    "could generalize to unseen subscribers."
)

report.append(
    "The value measure remains an annualized "
    "ARPU proxy rather than observed CLV."
)


REPORT_FILE = (
    REPORT_DIR
    / "23_test_risk_value_validation_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 12. TERMINAL OUTPUT
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "OUT-OF-SAMPLE RISK-VALUE SEGMENTS"
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
    "\nTOP RISK DECILE"
)

print(
    f"Customers              : "
    f"{len(top_decile):,}"
)

print(
    f"Churners captured      : "
    f"{top_decile_churners:,} / "
    f"{total_churners:,}"
)

print(
    f"Recall@Top10%          : "
    f"{top_decile_recall:.4f}"
)

print(
    f"Lift@Top10%            : "
    f"{top_decile_lift:.2f}x"
)


print(
    "\nPRIORITY 1 - HIGH RISK / HIGH VALUE"
)

print(
    f"Customers              : "
    f"{priority1_customers:,}"
)

print(
    f"Actual churners        : "
    f"{priority1_churners:,}"
)

print(
    f"Actual churn rate      : "
    f"{priority1_churn_rate * 100:.2f}%"
)


print(
    "\nGenerated files:"
)

for file in [
    SUMMARY_FILE,
    DETAIL_FILE,
    REPORT_FILE,
    CHART_FILE,
]:

    print(
        f"- {file.relative_to(PROJECT_ROOT)}"
    )


print(
    "\n" + "=" * 90
)

print(
    "STEP 23 COMPLETE"
)

print(
    "=" * 90
)