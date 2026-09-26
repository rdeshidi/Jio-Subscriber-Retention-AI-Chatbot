from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# STEP 05 - BUSINESS VISUALIZATIONS
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CHART_DIR = PROJECT_ROOT / "outputs" / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. INPUT FILES
# ------------------------------------------------------------

FILES = {
    "region":
        REPORT_DIR / "04_region_churn_analysis.csv",

    "trend":
        REPORT_DIR / "04_monthly_churn_trend.csv",

    "tenure":
        REPORT_DIR / "04_tenure_churn_analysis.csv",

    "complaints":
        REPORT_DIR / "04_complaint_churn_analysis.csv",

    "network":
        REPORT_DIR / "04_network_churn_analysis.csv",

    "patterns":
        REPORT_DIR / "04_customer_pattern_analysis.csv",

    "correlations":
        REPORT_DIR / "04_churn_correlations.csv",
}


for name, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Required analysis file missing:\n{path}"
        )


# ------------------------------------------------------------
# 3. LOAD ANALYSIS TABLES
# ------------------------------------------------------------

region = pd.read_csv(FILES["region"])
trend = pd.read_csv(FILES["trend"])
tenure = pd.read_csv(FILES["tenure"])
complaints = pd.read_csv(FILES["complaints"])
network = pd.read_csv(FILES["network"])
patterns = pd.read_csv(FILES["patterns"])
correlations = pd.read_csv(FILES["correlations"])


if "month_end" in trend.columns:

    trend["month_end"] = pd.to_datetime(
        trend["month_end"],
        errors="coerce"
    )


print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("STEP 05 - BUSINESS VISUALIZATIONS")
print("=" * 90)


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------

generated_charts = []


def save_chart(filename):

    output_path = CHART_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    generated_charts.append(
        output_path
    )

    print(
        f"[SAVED] "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )


def add_bar_labels(ax, decimals=2):

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt=f"%.{decimals}f",
            padding=3,
            fontsize=8
        )


# ============================================================
# CHART 1
# TOP 10 PROBLEMATIC CIRCLES
# ============================================================

top_circles = (
    region
    .sort_values(
        "churn_rate_30d_pct",
        ascending=False
    )
    .head(10)
    .sort_values(
        "churn_rate_30d_pct",
        ascending=True
    )
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_circles["circle"],
    top_circles["churn_rate_30d_pct"]
)

plt.xlabel("30-Day Churn Rate (%)")
plt.ylabel("Circle")

plt.title(
    "Top 10 Circles by 30-Day Subscriber Churn"
)

plt.grid(
    axis="x",
    alpha=0.25
)

save_chart(
    "01_top10_problematic_circles.png"
)


# ============================================================
# CHART 2
# MONTHLY CHURN TREND
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    trend["month_end"],
    trend["weighted_churn_pct"],
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Weighted Monthly Churn (%)")

plt.title(
    "Jio Monthly Churn Trend"
)

plt.xticks(
    rotation=45
)

plt.grid(
    alpha=0.25
)

save_chart(
    "02_monthly_churn_trend.png"
)


# ============================================================
# CHART 3
# TENURE VS CHURN
# ============================================================

tenure_order = [
    "0-3 months",
    "4-6 months",
    "7-12 months",
    "13-24 months",
    "25-36 months",
    "37-60 months",
    "61+ months",
]

tenure_plot = tenure.copy()

tenure_plot["tenure_band"] = pd.Categorical(
    tenure_plot["tenure_band"],
    categories=tenure_order,
    ordered=True
)

tenure_plot = tenure_plot.sort_values(
    "tenure_band"
)

plt.figure(figsize=(10, 6))

ax = plt.bar(
    tenure_plot["tenure_band"],
    tenure_plot["churn_rate_30d_pct"]
)

plt.xlabel("Subscriber Tenure")
plt.ylabel("30-Day Churn Rate (%)")

plt.title(
    "30-Day Churn Rate by Subscriber Tenure"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.25
)

add_bar_labels(
    plt.gca()
)

save_chart(
    "03_tenure_vs_churn.png"
)


# ============================================================
# CHART 4
# COMPLAINT COUNT VS CHURN
# ============================================================

complaint_plot = complaints[
    complaints["analysis"]
    ==
    "Complaints in last 6 months"
].copy()

complaint_order = [
    "0 complaints",
    "1 complaint",
    "2+ complaints",
]

complaint_plot["segment"] = pd.Categorical(
    complaint_plot["segment"],
    categories=complaint_order,
    ordered=True
)

complaint_plot = complaint_plot.sort_values(
    "segment"
)

plt.figure(figsize=(8, 6))

plt.bar(
    complaint_plot["segment"],
    complaint_plot["churn_rate_30d_pct"]
)

plt.xlabel("Complaint Frequency")
plt.ylabel("30-Day Churn Rate (%)")

plt.title(
    "Customer Complaints vs 30-Day Churn"
)

plt.grid(
    axis="y",
    alpha=0.25
)

add_bar_labels(
    plt.gca()
)

save_chart(
    "04_complaints_vs_churn.png"
)


# ============================================================
# CHART 5
# UNRESOLVED COMPLAINTS VS CHURN
# ============================================================

unresolved_plot = complaints[
    complaints["analysis"]
    ==
    "Unresolved complaints"
].copy()

unresolved_order = [
    "0 unresolved",
    "1 unresolved",
    "2+ unresolved",
]

unresolved_plot["segment"] = pd.Categorical(
    unresolved_plot["segment"],
    categories=unresolved_order,
    ordered=True
)

unresolved_plot = unresolved_plot.sort_values(
    "segment"
)

plt.figure(figsize=(8, 6))

plt.bar(
    unresolved_plot["segment"],
    unresolved_plot["churn_rate_30d_pct"]
)

plt.xlabel("Unresolved Complaints")
plt.ylabel("30-Day Churn Rate (%)")

plt.title(
    "Unresolved Complaints vs 30-Day Churn"
)

plt.grid(
    axis="y",
    alpha=0.25
)

add_bar_labels(
    plt.gca()
)

save_chart(
    "05_unresolved_complaints_vs_churn.png"
)


# ============================================================
# CHART 6
# SINR QUALITY VS CHURN
# ============================================================

sinr_plot = network[
    network["analysis"]
    ==
    "SINR quality"
].copy()

sinr_order = [
    "<5 dB",
    "5-14.9 dB",
    "15+ dB",
]

sinr_plot["segment"] = pd.Categorical(
    sinr_plot["segment"],
    categories=sinr_order,
    ordered=True
)

sinr_plot = sinr_plot.sort_values(
    "segment"
)

plt.figure(figsize=(8, 6))

plt.bar(
    sinr_plot["segment"],
    sinr_plot["churn_rate_30d_pct"]
)

plt.xlabel("Average SINR")
plt.ylabel("30-Day Churn Rate (%)")

plt.title(
    "Network Signal Quality vs Subscriber Churn"
)

plt.grid(
    axis="y",
    alpha=0.25
)

add_bar_labels(
    plt.gca()
)

save_chart(
    "06_sinr_vs_churn.png"
)


# ============================================================
# CHART 7
# NETWORK CONGESTION VS CHURN
# ============================================================

congestion_plot = network[
    network["analysis"]
    ==
    "Congestion score"
].copy()

congestion_order = [
    "<30",
    "30-69.9",
    "70+",
]

congestion_plot["segment"] = pd.Categorical(
    congestion_plot["segment"],
    categories=congestion_order,
    ordered=True
)

congestion_plot = congestion_plot.sort_values(
    "segment"
)

plt.figure(figsize=(8, 6))

plt.bar(
    congestion_plot["segment"],
    congestion_plot["churn_rate_30d_pct"]
)

plt.xlabel("Site Congestion Score")
plt.ylabel("30-Day Churn Rate (%)")

plt.title(
    "Network Congestion vs Subscriber Churn"
)

plt.grid(
    axis="y",
    alpha=0.25
)

add_bar_labels(
    plt.gca()
)

save_chart(
    "07_congestion_vs_churn.png"
)


# ============================================================
# CHART 8
# DROP CALL RATE VS CHURN
# ============================================================

drop_plot = network[
    network["analysis"]
    ==
    "Drop-call rate"
].copy()

drop_order = [
    "<0.5%",
    "0.5-0.99%",
    "1.0%+",
]

drop_plot["segment"] = pd.Categorical(
    drop_plot["segment"],
    categories=drop_order,
    ordered=True
)

drop_plot = drop_plot.sort_values(
    "segment"
)

plt.figure(figsize=(8, 6))

plt.bar(
    drop_plot["segment"],
    drop_plot["churn_rate_30d_pct"]
)

plt.xlabel("Drop-Call Rate")
plt.ylabel("30-Day Churn Rate (%)")

plt.title(
    "Drop-Call Rate vs Subscriber Churn"
)

plt.grid(
    axis="y",
    alpha=0.25
)

add_bar_labels(
    plt.gca()
)

save_chart(
    "08_drop_call_rate_vs_churn.png"
)


# ============================================================
# CHART 9
# ARPU VS CHURN
# ============================================================

arpu_plot = patterns[
    patterns["analysis"]
    ==
    "ARPU band"
].copy()

arpu_order = [
    "<₹150",
    "₹150-₹249.99",
    "₹250+",
]

arpu_plot["segment"] = pd.Categorical(
    arpu_plot["segment"],
    categories=arpu_order,
    ordered=True
)

arpu_plot = arpu_plot.sort_values(
    "segment"
)

plt.figure(figsize=(8, 6))

plt.bar(
    arpu_plot["segment"],
    arpu_plot["churn_rate_30d_pct"]
)

plt.xlabel("Monthly ARPU Band")
plt.ylabel("30-Day Churn Rate (%)")

plt.title(
    "ARPU Segment vs Subscriber Churn"
)

plt.grid(
    axis="y",
    alpha=0.25
)

add_bar_labels(
    plt.gca()
)

save_chart(
    "09_arpu_vs_churn.png"
)


# ============================================================
# CHART 10
# DAYS SINCE LAST RECHARGE VS CHURN
# ============================================================

recharge_plot = patterns[
    patterns["analysis"]
    ==
    "Days since last recharge"
].copy()

recharge_order = [
    "0-7 days",
    "8-30 days",
    "31-60 days",
    "61+ days",
]

recharge_plot["segment"] = pd.Categorical(
    recharge_plot["segment"],
    categories=recharge_order,
    ordered=True
)

recharge_plot = recharge_plot.sort_values(
    "segment"
)

plt.figure(figsize=(9, 6))

plt.bar(
    recharge_plot["segment"],
    recharge_plot["churn_rate_30d_pct"]
)

plt.xlabel("Days Since Last Recharge")
plt.ylabel("30-Day Churn Rate (%)")

plt.title(
    "Recharge Inactivity vs Subscriber Churn"
)

plt.grid(
    axis="y",
    alpha=0.25
)

add_bar_labels(
    plt.gca()
)

save_chart(
    "10_days_since_recharge_vs_churn.png"
)


# ============================================================
# CHART 11
# TOP NUMERIC CORRELATIONS
# ============================================================

top_corr = (
    correlations
    .head(10)
    .sort_values(
        "absolute_correlation",
        ascending=True
    )
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_corr["feature"],
    top_corr["correlation_with_churn_30d"]
)

plt.xlabel(
    "Correlation with 30-Day Churn"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top Numeric Associations with 30-Day Churn"
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.grid(
    axis="x",
    alpha=0.25
)

save_chart(
    "11_top_churn_correlations.png"
)


# ------------------------------------------------------------
# 5. CHART MANIFEST
# ------------------------------------------------------------

manifest = pd.DataFrame(
    {
        "chart_number":
            range(
                1,
                len(generated_charts) + 1
            ),

        "file_name":
            [
                path.name
                for path in generated_charts
            ],

        "relative_path":
            [
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                )
                for path in generated_charts
            ],
    }
)


MANIFEST_FILE = (
    REPORT_DIR
    / "05_chart_manifest.csv"
)

manifest.to_csv(
    MANIFEST_FILE,
    index=False
)


# ------------------------------------------------------------
# 6. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("STEP 05 COMPLETE")
print("=" * 90)

print(
    f"\nCharts generated: "
    f"{len(generated_charts)}"
)

for chart in generated_charts:

    print(
        f"- "
        f"{chart.relative_to(PROJECT_ROOT)}"
    )

print(
    "\nChart manifest:"
)

print(
    f"- "
    f"{MANIFEST_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    "\nThese charts are descriptive Day-1 "
    "business-analysis outputs."
)

print(
    "No predictive model has been trained."
)