from pathlib import Path
import pandas as pd


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# STEP 09 - FINAL DAY 1 SUMMARY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
ARCH_DIR = PROJECT_ROOT / "architecture"
SQL_DIR = PROJECT_ROOT / "sql"

FINAL_MD = PROJECT_ROOT / "DAY1_SUBMISSION_SUMMARY.md"
FINAL_TXT = REPORT_DIR / "09_day1_final_summary.txt"


# ------------------------------------------------------------
# 1. LOAD ANALYSIS OUTPUTS
# ------------------------------------------------------------

region = pd.read_csv(
    REPORT_DIR / "04_region_churn_analysis.csv"
)

tenure = pd.read_csv(
    REPORT_DIR / "04_tenure_churn_analysis.csv"
)

complaints = pd.read_csv(
    REPORT_DIR / "04_complaint_churn_analysis.csv"
)

network = pd.read_csv(
    REPORT_DIR / "04_network_churn_analysis.csv"
)

patterns = pd.read_csv(
    REPORT_DIR / "04_customer_pattern_analysis.csv"
)

correlations = pd.read_csv(
    REPORT_DIR / "04_churn_correlations.csv"
)

subscribers = pd.read_csv(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "subscribers_clean.csv"
)


# ------------------------------------------------------------
# 2. OVERALL CHURN
# ------------------------------------------------------------

target = (
    subscribers["churn_flag_30d"]
    .astype(str)
    .str.lower()
    .map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }
    )
)

total_subscribers = len(subscribers)
churners_30d = int(target.sum())

churn_rate = (
    churners_30d
    / total_subscribers
    * 100
)


# ------------------------------------------------------------
# 3. KEY RESULTS
# ------------------------------------------------------------

top5 = (
    region
    .sort_values(
        "churn_rate_30d_pct",
        ascending=False
    )
    .head(5)
)


tenure_lookup = tenure.set_index(
    "tenure_band"
)["churn_rate_30d_pct"]


complaint_lookup = (
    complaints[
        complaints["analysis"]
        == "Complaints in last 6 months"
    ]
    .set_index("segment")[
        "churn_rate_30d_pct"
    ]
)


unresolved_lookup = (
    complaints[
        complaints["analysis"]
        == "Unresolved complaints"
    ]
    .set_index("segment")[
        "churn_rate_30d_pct"
    ]
)


congestion_lookup = (
    network[
        network["analysis"]
        == "Congestion score"
    ]
    .set_index("segment")[
        "churn_rate_30d_pct"
    ]
)


sinr_lookup = (
    network[
        network["analysis"]
        == "SINR quality"
    ]
    .set_index("segment")[
        "churn_rate_30d_pct"
    ]
)


arpu_lookup = (
    patterns[
        patterns["analysis"]
        == "ARPU band"
    ]
    .set_index("segment")[
        "churn_rate_30d_pct"
    ]
)


top_corr = (
    correlations
    .head(5)[
        [
            "feature",
            "correlation_with_churn_30d",
        ]
    ]
)


# ------------------------------------------------------------
# 4. FILE CHECKS
# ------------------------------------------------------------

required_files = [
    PROJECT_ROOT / "src" / "01_data_exploration.py",
    PROJECT_ROOT / "src" / "02_data_quality_validation.py",
    PROJECT_ROOT / "src" / "03_data_cleaning.py",
    PROJECT_ROOT / "src" / "04_business_analysis.py",
    PROJECT_ROOT / "src" / "05_visualizations.py",
    PROJECT_ROOT / "src" / "06_prepare_sql_exports.py",
    PROJECT_ROOT / "src" / "08_generate_architecture.py",

    SQL_DIR / "01_create_tables.sql",
    SQL_DIR / "02_import_data.sql",
    SQL_DIR / "03_validation_queries.sql",

    ARCH_DIR / "Jio_End_to_End_Architecture.png",
    ARCH_DIR / "Jio_End_to_End_Architecture.pdf",
    ARCH_DIR / "Jio_Architecture_Plan.txt",
]


missing_files = [
    str(
        file.relative_to(PROJECT_ROOT)
    )
    for file in required_files
    if not file.exists()
]


# ------------------------------------------------------------
# 5. BUILD MARKDOWN SUMMARY
# ------------------------------------------------------------

lines = []

lines.append(
    "# Jio Subscriber Retention & AI Chatbot Project"
)

lines.append(
    "## Day 1 Submission Summary"
)

lines.append("")

lines.append(
    "### 1. Day 1 Objective"
)

lines.append("")

lines.append(
    "The Day 1 milestone focused on completing "
    "the subscriber-retention data analysis, "
    "preparing the supplied sandbox data in SQL, "
    "and designing the end-to-end architecture "
    "for predictive modelling and the employee "
    "AI chatbot."
)

lines.append("")

lines.append(
    "No predictive model was trained during "
    "Day 1. Predictive modelling is reserved "
    "for the Day 2 milestone."
)

lines.append("")

lines.append(
    "### 2. Dataset Overview"
)

lines.append("")

lines.append(
    f"- Subscriber records: **{total_subscribers:,}**"
)

lines.append(
    "- Service requests: **14,206**"
)

lines.append(
    "- Network sites: **3,847**"
)

lines.append(
    "- Circle-month KPI records: **396**"
)

lines.append(
    "- Circle targets: **22**"
)

lines.append(
    "- Retention offers: **20**"
)

lines.append("")

lines.append(
    "### 3. Overall Churn"
)

lines.append("")

lines.append(
    f"- 30-day churners: "
    f"**{churners_30d:,}**"
)

lines.append(
    f"- 30-day churn rate: "
    f"**{churn_rate:.2f}%**"
)

lines.append(
    "- 90-day churners: **3,241**"
)

lines.append(
    "- 90-day churn rate: **5.01%**"
)

lines.append("")

lines.append(
    "### 4. Problematic Regions"
)

lines.append("")

for _, row in top5.iterrows():

    lines.append(
        f"- **{row['circle']}** — "
        f"{row['churn_rate_30d_pct']:.2f}% "
        f"30-day churn"
    )


lines.append("")

lines.append(
    "### 5. Churn Timeline / Tenure"
)

lines.append("")

lines.append(
    f"- 0–3 months: "
    f"**{tenure_lookup['0-3 months']:.2f}%**"
)

lines.append(
    f"- 4–6 months: "
    f"**{tenure_lookup['4-6 months']:.2f}%**"
)

lines.append(
    f"- 7–12 months: "
    f"**{tenure_lookup['7-12 months']:.2f}%**"
)

lines.append(
    f"- 61+ months: "
    f"**{tenure_lookup['61+ months']:.2f}%**"
)

lines.append("")

lines.append(
    "The highest churn occurs during the first "
    "three months of subscriber tenure and "
    "declines steadily as tenure increases."
)

lines.append("")

lines.append(
    "### 6. Complaints and Service Issues"
)

lines.append("")

lines.append(
    f"- No complaints: "
    f"**{complaint_lookup['0 complaints']:.2f}%** churn"
)

lines.append(
    f"- One complaint: "
    f"**{complaint_lookup['1 complaint']:.2f}%** churn"
)

lines.append(
    f"- Two or more complaints: "
    f"**{complaint_lookup['2+ complaints']:.2f}%** churn"
)

lines.append(
    f"- Two or more unresolved complaints: "
    f"**{unresolved_lookup['2+ unresolved']:.2f}%** churn"
)

lines.append("")

lines.append(
    "Unresolved complaints show a particularly "
    "strong association with subscriber churn."
)

lines.append("")

lines.append(
    "### 7. Network Quality"
)

lines.append("")

lines.append(
    f"- Low congestion (<30): "
    f"**{congestion_lookup['<30']:.2f}%** churn"
)

lines.append(
    f"- High congestion (70+): "
    f"**{congestion_lookup['70+']:.2f}%** churn"
)

lines.append(
    f"- SINR below 5 dB: "
    f"**{sinr_lookup['<5 dB']:.2f}%** churn"
)

lines.append(
    f"- SINR 15+ dB: "
    f"**{sinr_lookup['15+ dB']:.2f}%** churn"
)

lines.append("")

lines.append(
    "Poor network conditions are associated "
    "with materially higher subscriber churn."
)

lines.append("")

lines.append(
    "### 8. ARPU / Price Sensitivity"
)

lines.append("")

lines.append(
    f"- ARPU below ₹150: "
    f"**{arpu_lookup['<₹150']:.2f}%** churn"
)

lines.append(
    f"- ARPU ₹150–₹249.99: "
    f"**{arpu_lookup['₹150-₹249.99']:.2f}%** churn"
)

lines.append(
    f"- ARPU ₹250+: "
    f"**{arpu_lookup['₹250+']:.2f}%** churn"
)

lines.append("")

lines.append(
    "Lower-ARPU subscribers show substantially "
    "higher churn, supporting the project's "
    "price-sensitivity hypothesis."
)

lines.append("")

lines.append(
    "### 9. Strongest Numeric Associations"
)

lines.append("")

for _, row in top_corr.iterrows():

    lines.append(
        f"- `{row['feature']}`: "
        f"{row['correlation_with_churn_30d']:.4f}"
    )

lines.append("")

lines.append(
    "These are associations and should not be "
    "interpreted as proof of causation."
)

lines.append("")

lines.append(
    "### 10. Data Quality"
)

lines.append("")

lines.append(
    "- Subscriber IDs are unique."
)

lines.append(
    "- No exact duplicate subscriber rows were found."
)

lines.append(
    "- Service-request subscriber references matched "
    "the subscriber table."
)

lines.append(
    "- No rows were deleted during cleaning."
)

lines.append(
    "- No missing values were artificially imputed."
)

lines.append(
    "- The supplied `churn_flag_30d` target was preserved."
)

lines.append(
    "- 739 synthetic records have a known mismatch "
    "between the supplied 30-day churn label and "
    "`churn_date`; this was documented rather than "
    "rewriting the source target."
)

lines.append("")

lines.append(
    "Potential leakage fields identified for "
    "Day 2 modelling:"
)

lines.append("")

lines.append(
    "- `mnp_enquiry_flag`"
)

lines.append(
    "- `churn_reason`"
)

lines.append(
    "- `churn_date`"
)

lines.append(
    "- `subscriber_id` should also be excluded "
    "as an identifier during model training."
)

lines.append("")

lines.append(
    "### 11. SQL Database"
)

lines.append("")

lines.append(
    "Database: `jio_retention_db`"
)

lines.append("")

lines.append(
    "Tables:"
)

lines.append("")

lines.append(
    "- `subscribers` — 64,738 rows"
)

lines.append(
    "- `service_requests` — 14,206 rows"
)

lines.append(
    "- `network_sites` — 3,847 rows"
)

lines.append(
    "- `circle_monthly_kpi` — 396 rows"
)

lines.append(
    "- `circle_targets` — 22 rows"
)

lines.append(
    "- `offer_catalogue` — 20 rows"
)

lines.append("")

lines.append(
    "SQL validation reproduced the same core "
    "business findings as the Python analysis."
)

lines.append("")

lines.append(
    "### 12. Architecture Plan"
)

lines.append("")

lines.append(
    "The proposed architecture includes:"
)

lines.append("")

lines.append(
    "- MySQL analytical database"
)

lines.append(
    "- Python / Pandas processing"
)

lines.append(
    "- Power BI management dashboards"
)

lines.append(
    "- 30-day churn prediction"
)

lines.append(
    "- Customer Lifetime Value modelling"
)

lines.append(
    "- Uplift / offer-response modelling"
)

lines.append(
    "- Customer score store"
)

lines.append(
    "- Text-to-SQL agent"
)

lines.append(
    "- SQLAlchemy"
)

lines.append(
    "- Database schema metadata"
)

lines.append(
    "- RAG over Excel/business files"
)

lines.append(
    "- LLM response layer"
)

lines.append(
    "- Timestamped backend audit logs"
)

lines.append(
    "- Read-only SQL and governance controls"
)

lines.append("")

lines.append(
    "### 13. Day 2 Plan"
)

lines.append("")

lines.append(
    "Day 2 will focus on:"
)

lines.append("")

lines.append(
    "1. Logistic Regression baseline for "
    "30-day churn."
)

lines.append(
    "2. Gradient Boosting / XGBoost / CatBoost "
    "champion-model comparison."
)

lines.append(
    "3. Precision, Recall, F1, ROC-AUC, PR-AUC, "
    "Lift and Recall@Top-Decile."
)

lines.append(
    "4. CLV modelling design."
)

lines.append(
    "5. Discount-response / uplift modelling design."
)

lines.append("")

lines.append(
    "### 14. Main Day 1 Deliverables"
)

lines.append("")

lines.append(
    "- Python exploration, validation, cleaning, "
    "analysis and visualization scripts"
)

lines.append(
    "- 11 business-analysis charts"
)

lines.append(
    "- MySQL database with six imported tables"
)

lines.append(
    "- SQL table-creation, import and validation scripts"
)

lines.append(
    "- End-to-end architecture PNG"
)

lines.append(
    "- End-to-end architecture PDF"
)

lines.append(
    "- Detailed architecture plan"
)

lines.append(
    "- Final Day 1 findings summary"
)


# ------------------------------------------------------------
# 6. SAVE FILES
# ------------------------------------------------------------

summary_text = "\n".join(lines)

FINAL_MD.write_text(
    summary_text,
    encoding="utf-8"
)

FINAL_TXT.write_text(
    summary_text.replace("#", ""),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 7. TERMINAL SUMMARY
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("STEP 09 - FINAL DAY 1 SUMMARY")
print("=" * 90)

print(
    f"\n30-day churn: "
    f"{churners_30d:,} / "
    f"{total_subscribers:,} "
    f"({churn_rate:.2f}%)"
)

print("\nTop 5 circles:")

print(
    top5[
        [
            "circle",
            "churn_rate_30d_pct",
        ]
    ].to_string(
        index=False
    )
)

print("\nGenerated files:")

print(
    f"- {FINAL_MD.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {FINAL_TXT.relative_to(PROJECT_ROOT)}"
)

if missing_files:

    print(
        "\nWARNING - Missing expected files:"
    )

    for file in missing_files:
        print(f"- {file}")

else:

    print(
        "\nAll expected Day 1 project files "
        "were found."
    )

print("\nDAY 1 COMPLETE")