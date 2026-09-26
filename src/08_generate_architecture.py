from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# STEP 08 - END-TO-END ARCHITECTURE PLAN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCH_DIR = PROJECT_ROOT / "architecture"
ARCH_DIR.mkdir(parents=True, exist_ok=True)

PNG_FILE = ARCH_DIR / "Jio_End_to_End_Architecture.png"
PDF_FILE = ARCH_DIR / "Jio_End_to_End_Architecture.pdf"
PLAN_FILE = ARCH_DIR / "Jio_Architecture_Plan.txt"


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    body="",
    fontsize=9,
):

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=fontsize + 1,
        fontweight="bold",
    )

    if body:

        ax.text(
            x + width / 2,
            y + height * 0.38,
            body,
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True,
        )


def add_arrow(
    ax,
    start,
    end,
):

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
    )

    ax.add_patch(arrow)


# ============================================================
# 2. CREATE ARCHITECTURE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 10)
)

ax.set_xlim(0, 16)
ax.set_ylim(0, 11)

ax.axis("off")

ax.set_title(
    "Jio Subscriber Retention & AI Chatbot — End-to-End Architecture",
    fontsize=18,
    fontweight="bold",
    pad=20,
)


# ============================================================
# LAYER 1 - DATA SOURCES
# ============================================================

add_box(
    ax,
    0.4,
    9.1,
    2.3,
    1.1,
    "Subscriber / CRM",
    "Profile • Circle • Plan\nTenure • Device",
)

add_box(
    ax,
    3.0,
    9.1,
    2.3,
    1.1,
    "Billing / Recharge",
    "ARPU • Recharge Gaps\nPayments • Plan History",
)

add_box(
    ax,
    5.6,
    9.1,
    2.3,
    1.1,
    "Service Requests",
    "Complaints • SLA\nReopen • Resolution",
)

add_box(
    ax,
    8.2,
    9.1,
    2.3,
    1.1,
    "Network Data",
    "SINR • Drop Calls\nCongestion • Throughput",
)

add_box(
    ax,
    10.8,
    9.1,
    2.3,
    1.1,
    "Circle KPIs",
    "Churn • ARPU\nPort-in / Port-out",
)

add_box(
    ax,
    13.4,
    9.1,
    2.2,
    1.1,
    "Excel / Business Files",
    "Offers • Targets\nPrice Sheets",
)


# ============================================================
# LAYER 2 - STORAGE / DATABASE
# ============================================================

add_box(
    ax,
    4.8,
    7.45,
    6.4,
    1.0,
    "SQL Database / Analytical Warehouse",
    "MySQL • subscribers • service_requests • network_sites • circle_monthly_kpi • circle_targets • offer_catalogue",
    fontsize=8,
)


# Source arrows
for x in [
    1.55,
    4.15,
    6.75,
    9.35,
    11.95,
    14.5,
]:
    add_arrow(
        ax,
        (x, 9.1),
        (8.0, 8.45),
    )


# ============================================================
# LAYER 3 - PROCESSING / ANALYTICS
# ============================================================

add_box(
    ax,
    0.6,
    5.65,
    4.2,
    1.15,
    "Data Processing & Feature Engineering",
    "Python • Pandas\nValidation • Cleaning • Feature Engineering",
)

add_box(
    ax,
    5.9,
    5.65,
    4.2,
    1.15,
    "Business Analytics / BI",
    "Circle Churn • Tenure • Complaints\nNetwork • ARPU • Recharge Patterns",
)

add_box(
    ax,
    11.2,
    5.65,
    4.2,
    1.15,
    "Power BI / Management Dashboard",
    "Executive KPIs • Trends\nRegional Drill-downs",
)


add_arrow(
    ax,
    (8.0, 7.45),
    (2.7, 6.8),
)

add_arrow(
    ax,
    (8.0, 7.45),
    (8.0, 6.8),
)

add_arrow(
    ax,
    (8.0, 7.45),
    (13.3, 6.8),
)


# ============================================================
# LAYER 4 - PREDICTIVE MODELLING
# ============================================================

add_box(
    ax,
    0.5,
    3.65,
    3.2,
    1.25,
    "30-Day Churn Model",
    "Baseline: Logistic Regression\nCandidates: GBT • XGBoost • CatBoost",
)

add_box(
    ax,
    4.1,
    3.65,
    3.2,
    1.25,
    "Customer Lifetime Value",
    "Estimate future customer value\nPrioritise valuable at-risk users",
)

add_box(
    ax,
    7.7,
    3.65,
    3.2,
    1.25,
    "Uplift / Offer Response",
    "Who changes behaviour\nbecause of an offer?",
)

add_box(
    ax,
    11.3,
    3.65,
    4.0,
    1.25,
    "Customer Score Store",
    "Churn Probability • CLV\nOffer Response • SHAP / Reason Codes",
)


add_arrow(
    ax,
    (2.7, 5.65),
    (2.1, 4.9),
)

add_arrow(
    ax,
    (2.7, 5.65),
    (5.7, 4.9),
)

add_arrow(
    ax,
    (2.7, 5.65),
    (9.3, 4.9),
)

add_arrow(
    ax,
    (3.7, 4.25),
    (11.3, 4.25),
)

add_arrow(
    ax,
    (7.3, 4.25),
    (11.3, 4.25),
)

add_arrow(
    ax,
    (10.9, 4.25),
    (11.3, 4.25),
)


# ============================================================
# LAYER 5 - GENAI / CHATBOT
# ============================================================

add_box(
    ax,
    0.7,
    1.55,
    3.5,
    1.25,
    "Employee Question",
    "\"Which 5 circles lost the most customers?\"\n\"Why is this subscriber high risk?\"",
)

add_box(
    ax,
    4.7,
    1.55,
    3.2,
    1.25,
    "Text-to-SQL Agent",
    "SQL Agent • SQLAlchemy\nSchema Metadata • Read-only SQL",
)

add_box(
    ax,
    8.4,
    1.55,
    3.0,
    1.25,
    "RAG / Excel Retrieval",
    "Offer Catalogue • Targets\nPrice / Policy Sheets",
)

add_box(
    ax,
    11.9,
    1.55,
    3.4,
    1.25,
    "LLM Response Layer",
    "Answer • Explanation\nCharts • Recommended Offer",
)


add_arrow(
    ax,
    (4.2, 2.17),
    (4.7, 2.17),
)

add_arrow(
    ax,
    (7.9, 2.17),
    (11.9, 2.17),
)

add_arrow(
    ax,
    (11.4, 2.17),
    (11.9, 2.17),
)

add_arrow(
    ax,
    (13.3, 3.65),
    (6.3, 2.8),
)

add_arrow(
    ax,
    (14.5, 9.1),
    (9.9, 2.8),
)


# ============================================================
# LAYER 6 - LOGGING / GOVERNANCE
# ============================================================

add_box(
    ax,
    3.7,
    0.15,
    8.6,
    0.85,
    "Backend Audit & Safety Layer",
    "Timestamp • User Question • Generated SQL • Execution Status • Row Count • Output • Error • Latency • Read-only Controls",
    fontsize=8,
)

add_arrow(
    ax,
    (6.3, 1.55),
    (6.5, 1.0),
)

add_arrow(
    ax,
    (13.6, 1.55),
    (10.5, 1.0),
)


# ============================================================
# 3. SAVE ARCHITECTURE IMAGE
# ============================================================

plt.tight_layout()

plt.savefig(
    PNG_FILE,
    dpi=220,
    bbox_inches="tight",
)

plt.savefig(
    PDF_FILE,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 4. WRITE ARCHITECTURE PLAN
# ============================================================

architecture_plan = """
JIO SUBSCRIBER RETENTION & AI CHATBOT
END-TO-END ARCHITECTURE PLAN
============================================================

1. BUSINESS OBJECTIVE
------------------------------------------------------------
The solution is designed to reduce customer churn by detecting
high-risk subscribers before they leave, identifying customers
worth retaining, and selecting customers who are likely to
respond positively to retention offers.

The analytical system also supports an employee-facing GenAI
chatbot so managers can obtain answers from structured customer
data and business Excel files without waiting for manual analyst
requests.


2. DATA SOURCES
------------------------------------------------------------
Primary data sources include:

• Subscriber / CRM
  - Subscriber profile
  - Circle and zone
  - Plan type
  - Tenure
  - Device information

• Billing / Recharge
  - ARPU
  - Recharge frequency
  - Recharge gaps
  - Payment failures
  - Plan history

• Service Requests
  - Complaints
  - SLA breaches
  - Resolution time
  - Reopened tickets
  - CSAT

• Network Data
  - SINR
  - Drop-call rate
  - Congestion
  - Throughput

• Circle KPI Data
  - Subscriber base
  - Monthly churn
  - Port-in / port-out
  - ARPU
  - Complaints

• Excel / Business Files
  - Retention offer catalogue
  - Circle targets
  - Pricing information
  - Business planning sheets


3. STORAGE LAYER
------------------------------------------------------------
The Day-1 implementation uses MySQL.

Database:
    jio_retention_db

Main tables:
    subscribers
    service_requests
    network_sites
    circle_monthly_kpi
    circle_targets
    offer_catalogue

The database becomes the structured source for analytics,
predictive modelling, dashboards and the Text-to-SQL chatbot.


4. DATA PROCESSING
------------------------------------------------------------
Python and Pandas are used for:

• Data exploration
• Data-quality validation
• Cleaning and standardisation
• Date handling
• Feature engineering
• Business analysis
• Model preparation

Raw data is preserved separately from processed datasets.


5. BUSINESS ANALYTICS
------------------------------------------------------------
The analytical layer answers questions including:

• Which circles have the highest churn?
• At what subscriber tenure is churn highest?
• Do complaints increase churn?
• Do unresolved complaints increase churn?
• Is poor network quality associated with churn?
• Are low-ARPU customers more price sensitive?
• Does recharge inactivity signal churn?

Outputs can be consumed in charts, reports and Power BI.


6. PREDICTIVE MODELLING
------------------------------------------------------------

A. 30-Day Churn Prediction
   Baseline:
       Logistic Regression

   Champion-model candidates:
       Gradient Boosted Trees
       XGBoost
       CatBoost

   Important evaluation metrics:
       Precision
       Recall
       F1 Score
       ROC-AUC
       PR-AUC
       Recall at Top Decile
       Lift

B. Customer Lifetime Value
   Estimate expected future value so retention teams can
   prioritise high-value customers instead of treating all
   churn-risk subscribers equally.

C. Uplift / Offer Response
   Estimate which customers are likely to remain specifically
   because they received a retention offer.

   This avoids discounting customers who would have stayed
   without an incentive.


7. MODEL OUTPUT / CUSTOMER SCORE STORE
------------------------------------------------------------
Model outputs should be written back to SQL tables.

Example fields:

subscriber_id
churn_probability_30d
risk_decile
predicted_clv
offer_response_score
recommended_offer
model_version
score_timestamp

This score table becomes available to dashboards, agents and
retention teams.


8. TEXT-TO-SQL CHATBOT
------------------------------------------------------------
The employee chatbot accepts normal English questions.

Example:
    "Which five circles had the highest churn last month?"

Flow:

Employee Question
        ↓
LLM / Intent Router
        ↓
SQL Agent
        ↓
SQLAlchemy
        ↓
Database Schema Metadata
        ↓
Generated SQL
        ↓
SQL Validation
        ↓
Read-only Database Execution
        ↓
Result
        ↓
LLM Answer


9. RAG / EXCEL RETRIEVAL
------------------------------------------------------------
Some commercial information is stored outside the SQL database.

Examples:
    Offer catalogues
    Circle targets
    Price sheets
    Policy documents

RAG retrieval allows the chatbot to combine SQL facts with
information contained in these business files.

Example:
    "Why is this customer high risk and what approved offer
     should I give?"

Possible response flow:

SQL:
    Customer churn risk + customer value

Model:
    SHAP / risk explanation

Excel / RAG:
    Approved offers

LLM:
    Final explanation and recommended offer


10. BACKEND LOGGING
------------------------------------------------------------
The instructor requires timestamped backend logs.

Each chatbot request should capture:

timestamp
user_id
question
generated_sql
execution_status
execution_time
row_count
result_summary
error_message
model_name
session_id

This provides traceability, debugging and governance.


11. SECURITY / GOVERNANCE
------------------------------------------------------------
The chatbot database connection should use:

• Read-only SQL credentials
• Table allow-list
• Row limits
• SQL parser validation
• Blocked DDL commands
• Blocked DML commands
• Prompt-injection filtering
• Explicit failure / abstention when confidence is low
• Full audit logging

The chatbot should never be allowed to modify production data.


12. BUSINESS OUTPUT
------------------------------------------------------------
The final architecture supports:

• Churn-risk identification
• High-value customer prioritisation
• Targeted retention offers
• Reduced discount wastage
• Faster management decision-making
• Self-service business analytics
• Explainable subscriber-level risk
• Auditable GenAI chatbot responses


13. PROJECT PHASING
------------------------------------------------------------

DAY 1
    Data Analysis
    Data Validation
    SQL Database Setup
    Visualisations
    End-to-End Architecture Plan

DAY 2
    30-Day Churn Prediction
    CLV Planning / Modelling
    Offer Response / Uplift Planning

LATER PHASE
    Text-to-SQL Agent
    RAG over Excel files
    LLM Integration
    Backend Logs
    Testing and Evaluation
"""


PLAN_FILE.write_text(
    architecture_plan.strip(),
    encoding="utf-8"
)


# ============================================================
# 5. FINAL OUTPUT
# ============================================================

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("STEP 08 - END-TO-END ARCHITECTURE")
print("=" * 90)

print("\nArchitecture files generated:")

print(
    f"- {PNG_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {PDF_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- {PLAN_FILE.relative_to(PROJECT_ROOT)}"
)

print("\nSTEP 08 COMPLETE")