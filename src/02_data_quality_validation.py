from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# STEP 02 - DATA QUALITY & VALIDATION
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

EXCEL_FILE = RAW_DATA_DIR / "Jio_Retention_Dataset.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ISSUES_FILE = OUTPUT_DIR / "02_data_quality_issues.csv"
CHECKS_FILE = OUTPUT_DIR / "02_validation_checks.csv"
MISSING_FILE = OUTPUT_DIR / "02_missingness_profile.csv"
REPORT_FILE = OUTPUT_DIR / "02_data_quality_report.txt"


# Dataset snapshot date from the supplied sandbox documentation
SNAPSHOT_DATE = pd.Timestamp("2026-08-31")


print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("STEP 02 - DATA QUALITY & VALIDATION")
print("=" * 90)


# ------------------------------------------------------------
# 2. LOAD WORKBOOK
# ------------------------------------------------------------

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"Excel file not found:\n{EXCEL_FILE}"
    )

excel = pd.ExcelFile(EXCEL_FILE)

dataframes = {
    sheet: pd.read_excel(EXCEL_FILE, sheet_name=sheet)
    for sheet in excel.sheet_names
}

print("\nWorkbook loaded successfully.")

for sheet, df in dataframes.items():
    print(
        f"{sheet:<22} "
        f"{df.shape[0]:>7,} rows x "
        f"{df.shape[1]:>3} columns"
    )


# ------------------------------------------------------------
# 3. VALIDATION STORAGE
# ------------------------------------------------------------

checks = []
issues = []


def add_check(
    table,
    check_name,
    status,
    affected_rows=0,
    details=""
):
    checks.append(
        {
            "table": table,
            "check_name": check_name,
            "status": status,
            "affected_rows": int(affected_rows),
            "details": details,
        }
    )


def add_issue(
    table,
    check_name,
    severity,
    affected_rows,
    details
):
    issues.append(
        {
            "table": table,
            "check_name": check_name,
            "severity": severity,
            "affected_rows": int(affected_rows),
            "details": details,
        }
    )


def evaluate_check(
    table,
    check_name,
    affected_rows,
    severity="WARNING",
    details_if_problem="",
    details_if_pass="No issue found."
):
    affected_rows = int(affected_rows)

    if affected_rows == 0:
        add_check(
            table,
            check_name,
            "PASS",
            0,
            details_if_pass
        )
    else:
        add_check(
            table,
            check_name,
            "REVIEW",
            affected_rows,
            details_if_problem
        )

        add_issue(
            table,
            check_name,
            severity,
            affected_rows,
            details_if_problem
        )


# ------------------------------------------------------------
# 4. BASIC TABLE CHECKS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("BASIC TABLE VALIDATION")
print("=" * 90)

for table_name, df in dataframes.items():

    duplicate_rows = int(df.duplicated().sum())

    evaluate_check(
        table_name,
        "Exact duplicate rows",
        duplicate_rows,
        severity="HIGH",
        details_if_problem="Exact duplicate rows detected.",
        details_if_pass="No exact duplicate rows."
    )

    empty_columns = [
        col for col in df.columns
        if df[col].isna().all()
    ]

    evaluate_check(
        table_name,
        "Completely empty columns",
        len(empty_columns),
        severity="MEDIUM",
        details_if_problem=(
            f"Completely empty columns: {empty_columns}"
        ),
        details_if_pass="No completely empty columns."
    )


# ------------------------------------------------------------
# 5. MISSINGNESS PROFILE
# ------------------------------------------------------------

missing_records = []

for table_name, df in dataframes.items():

    for col in df.columns:

        missing_count = int(df[col].isna().sum())

        missing_pct = (
            missing_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        missing_records.append(
            {
                "table": table_name,
                "column": col,
                "rows": len(df),
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
            }
        )

missing_df = pd.DataFrame(missing_records)

missing_df.to_csv(
    MISSING_FILE,
    index=False
)


# ------------------------------------------------------------
# 6. MAIN SUBSCRIBER TABLE
# ------------------------------------------------------------

if "subscribers" not in dataframes:
    raise ValueError(
        "'subscribers' sheet was not found."
    )

subscribers = dataframes["subscribers"].copy()


print("\n" + "=" * 90)
print("SUBSCRIBER TABLE VALIDATION")
print("=" * 90)


# ------------------------------------------------------------
# 7. SUBSCRIBER ID
# ------------------------------------------------------------

if "subscriber_id" in subscribers.columns:

    missing_ids = int(
        subscribers["subscriber_id"].isna().sum()
    )

    duplicate_ids = int(
        subscribers["subscriber_id"]
        .duplicated()
        .sum()
    )

    evaluate_check(
        "subscribers",
        "Missing subscriber_id",
        missing_ids,
        severity="CRITICAL",
        details_if_problem="Missing subscriber identifiers found.",
        details_if_pass="All subscriber IDs are present."
    )

    evaluate_check(
        "subscribers",
        "Duplicate subscriber_id",
        duplicate_ids,
        severity="CRITICAL",
        details_if_problem="Duplicate subscriber identifiers found.",
        details_if_pass="Subscriber IDs are unique."
    )


# ------------------------------------------------------------
# 8. TARGET LOGIC
# ------------------------------------------------------------

if {
    "churn_flag_30d",
    "churn_flag_90d"
}.issubset(subscribers.columns):

    invalid_target_logic = (
        (subscribers["churn_flag_30d"] == True)
        &
        (subscribers["churn_flag_90d"] != True)
    )

    evaluate_check(
        "subscribers",
        "30-day churn must also be 90-day churn",
        invalid_target_logic.sum(),
        severity="CRITICAL",
        details_if_problem=(
            "Some 30-day churners are not marked as "
            "90-day churners."
        ),
        details_if_pass=(
            "Every 30-day churner is also marked "
            "as a 90-day churner."
        )
    )


# ------------------------------------------------------------
# 9. CHURN DATE CONSISTENCY
# ------------------------------------------------------------

if "churn_date" in subscribers.columns:

    subscribers["_parsed_churn_date"] = pd.to_datetime(
        subscribers["churn_date"],
        errors="coerce"
    )

    if "churn_flag_90d" in subscribers.columns:

        churn90_missing_date = (
            (subscribers["churn_flag_90d"] == True)
            &
            subscribers["_parsed_churn_date"].isna()
        )

        evaluate_check(
            "subscribers",
            "90-day churners missing churn_date",
            churn90_missing_date.sum(),
            severity="HIGH",
            details_if_problem=(
                "Customers marked as 90-day churners "
                "have no churn date."
            ),
            details_if_pass=(
                "All 90-day churners have a churn date."
            )
        )

        non_churn_with_date = (
            (subscribers["churn_flag_90d"] == False)
            &
            subscribers["_parsed_churn_date"].notna()
        )

        evaluate_check(
            "subscribers",
            "Non-churners with churn_date",
            non_churn_with_date.sum(),
            severity="HIGH",
            details_if_problem=(
                "Customers not marked as 90-day churners "
                "have a churn date."
            ),
            details_if_pass=(
                "Non-churners correctly have no churn date."
            )
        )


    # Churn dates should be after snapshot date
    before_snapshot = (
        subscribers["_parsed_churn_date"].notna()
        &
        (
            subscribers["_parsed_churn_date"]
            <= SNAPSHOT_DATE
        )
    )

    evaluate_check(
        "subscribers",
        "Churn dates before/on snapshot date",
        before_snapshot.sum(),
        severity="HIGH",
        details_if_problem=(
            "Some churn dates occur before or on "
            "the dataset snapshot date."
        ),
        details_if_pass=(
            "All churn dates occur after the snapshot date."
        )
    )


    # 90-day horizon
    max_90d_date = SNAPSHOT_DATE + pd.Timedelta(days=90)

    beyond_90d = (
        subscribers["_parsed_churn_date"].notna()
        &
        (
            subscribers["_parsed_churn_date"]
            > max_90d_date
        )
    )

    evaluate_check(
        "subscribers",
        "Churn dates beyond 90-day prediction horizon",
        beyond_90d.sum(),
        severity="HIGH",
        details_if_problem=(
            f"Churn dates found after {max_90d_date.date()}."
        ),
        details_if_pass=(
            "All churn dates fall within the 90-day horizon."
        )
    )


    # 30-day target should have churn within first 30 days
    if "churn_flag_30d" in subscribers.columns:

        max_30d_date = (
            SNAPSHOT_DATE
            + pd.Timedelta(days=30)
        )

        invalid_30d_date = (
            (subscribers["churn_flag_30d"] == True)
            &
            (
                subscribers["_parsed_churn_date"]
                > max_30d_date
            )
        )

        evaluate_check(
            "subscribers",
            "30-day churners with churn_date after 30 days",
            invalid_30d_date.sum(),
            severity="CRITICAL",
            details_if_problem=(
                "Some customers marked as 30-day churners "
                "have churn dates outside the 30-day horizon."
            ),
            details_if_pass=(
                "30-day churn dates correctly fall "
                "within the first 30 days."
            )
        )


# ------------------------------------------------------------
# 10. CHURN REASON CONSISTENCY
# ------------------------------------------------------------

if {
    "churn_reason",
    "churn_flag_90d"
}.issubset(subscribers.columns):

    missing_reason_for_churner = (
        (subscribers["churn_flag_90d"] == True)
        &
        subscribers["churn_reason"].isna()
    )

    evaluate_check(
        "subscribers",
        "90-day churners missing churn_reason",
        missing_reason_for_churner.sum(),
        severity="MEDIUM",
        details_if_problem=(
            "Some churners do not have a churn reason."
        ),
        details_if_pass=(
            "All 90-day churners have a churn reason."
        )
    )

    reason_for_non_churner = (
        (subscribers["churn_flag_90d"] == False)
        &
        subscribers["churn_reason"].notna()
    )

    evaluate_check(
        "subscribers",
        "Non-churners with churn_reason",
        reason_for_non_churner.sum(),
        severity="MEDIUM",
        details_if_problem=(
            "Some non-churners have a churn reason."
        ),
        details_if_pass=(
            "Non-churners correctly have no churn reason."
        )
    )


# ------------------------------------------------------------
# 11. JOIN DATE & TENURE
# ------------------------------------------------------------

if "join_date" in subscribers.columns:

    subscribers["_parsed_join_date"] = pd.to_datetime(
        subscribers["join_date"],
        errors="coerce"
    )

    invalid_join_dates = (
        subscribers["_parsed_join_date"].isna()
    )

    evaluate_check(
        "subscribers",
        "Invalid join_date",
        invalid_join_dates.sum(),
        severity="HIGH",
        details_if_problem="Invalid/unparseable join dates found.",
        details_if_pass="All join dates are valid."
    )

    future_join_dates = (
        subscribers["_parsed_join_date"]
        > SNAPSHOT_DATE
    )

    evaluate_check(
        "subscribers",
        "Join dates after snapshot date",
        future_join_dates.sum(),
        severity="CRITICAL",
        details_if_problem=(
            "Subscribers have join dates after "
            "the snapshot date."
        ),
        details_if_pass=(
            "All join dates occur on/before snapshot date."
        )
    )


if {
    "join_date",
    "tenure_months"
}.issubset(subscribers.columns):

    calculated_tenure = (
        (
            SNAPSHOT_DATE
            - subscribers["_parsed_join_date"]
        ).dt.days / 30.4375
    )

    tenure_difference = (
        calculated_tenure
        - subscribers["tenure_months"]
    ).abs()

    # Allow approximately one month due to month-length conventions
    tenure_mismatch = tenure_difference > 1.5

    evaluate_check(
        "subscribers",
        "Tenure inconsistent with join_date",
        tenure_mismatch.sum(),
        severity="MEDIUM",
        details_if_problem=(
            "Reported tenure differs from join-date-derived "
            "tenure by more than 1.5 months."
        ),
        details_if_pass=(
            "Tenure is consistent with join dates."
        )
    )


# ------------------------------------------------------------
# 12. NUMERIC RANGE CHECKS
# ------------------------------------------------------------

nonnegative_columns = [
    "tenure_months",
    "plan_price_inr",
    "plan_validity_days",
    "arpu_last_month_inr",
    "arpu_3m_avg_inr",
    "arpu_6m_avg_inr",
    "recharge_count_6m",
    "avg_recharge_gap_days",
    "days_since_last_recharge",
    "payment_failures_6m",
    "data_gb_last_month",
    "data_gb_3m_avg",
    "voice_minutes_last_month",
    "sms_count_last_month",
    "complaints_6m",
    "unresolved_complaints",
    "avg_resolution_days",
    "num_services",
    "app_logins_30d",
]

for col in nonnegative_columns:

    if col in subscribers.columns:

        numeric = pd.to_numeric(
            subscribers[col],
            errors="coerce"
        )

        negative_values = (
            numeric.notna()
            &
            (numeric < 0)
        )

        evaluate_check(
            "subscribers",
            f"Negative values in {col}",
            negative_values.sum(),
            severity="HIGH",
            details_if_problem=(
                f"Negative values detected in {col}."
            ),
            details_if_pass=(
                f"No negative values in {col}."
            )
        )


# ------------------------------------------------------------
# 13. PERCENTAGE / SCORE RANGE CHECKS
# ------------------------------------------------------------

range_0_100_columns = [
    "drop_call_rate_pct",
    "site_congestion_score",
    "outgoing_to_competitor_pct",
]

for col in range_0_100_columns:

    if col in subscribers.columns:

        numeric = pd.to_numeric(
            subscribers[col],
            errors="coerce"
        )

        invalid = (
            numeric.notna()
            &
            (
                (numeric < 0)
                |
                (numeric > 100)
            )
        )

        evaluate_check(
            "subscribers",
            f"{col} outside 0-100",
            invalid.sum(),
            severity="HIGH",
            details_if_problem=(
                f"{col} contains values outside 0-100."
            ),
            details_if_pass=(
                f"{col} values are within 0-100."
            )
        )


# ------------------------------------------------------------
# 14. SINR PLAUSIBILITY
# ------------------------------------------------------------

if "avg_sinr_db" in subscribers.columns:

    sinr = pd.to_numeric(
        subscribers["avg_sinr_db"],
        errors="coerce"
    )

    implausible_sinr = (
        sinr.notna()
        &
        (
            (sinr < -30)
            |
            (sinr > 50)
        )
    )

    evaluate_check(
        "subscribers",
        "Implausible avg_sinr_db",
        implausible_sinr.sum(),
        severity="MEDIUM",
        details_if_problem=(
            "SINR values outside the broad plausibility "
            "range -30 to 50 dB were found."
        ),
        details_if_pass=(
            "SINR values fall within a broad plausible range."
        )
    )


# ------------------------------------------------------------
# 15. COMPLAINT LOGIC
# ------------------------------------------------------------

if {
    "complaints_6m",
    "unresolved_complaints"
}.issubset(subscribers.columns):

    invalid_complaints = (
        subscribers["unresolved_complaints"]
        >
        subscribers["complaints_6m"]
    )

    evaluate_check(
        "subscribers",
        "Unresolved complaints exceed total complaints",
        invalid_complaints.sum(),
        severity="CRITICAL",
        details_if_problem=(
            "Some customers have more unresolved complaints "
            "than total complaints."
        ),
        details_if_pass=(
            "Unresolved complaint count never exceeds "
            "total complaints."
        )
    )


# ------------------------------------------------------------
# 16. 5G LOGIC
# ------------------------------------------------------------

if {
    "is_5g_device",
    "is_5g_active"
}.issubset(subscribers.columns):

    invalid_5g = (
        (subscribers["is_5g_active"] == True)
        &
        (subscribers["is_5g_device"] != True)
    )

    evaluate_check(
        "subscribers",
        "5G active but device not 5G-capable",
        invalid_5g.sum(),
        severity="HIGH",
        details_if_problem=(
            "Subscribers marked as active on 5G "
            "but not using a 5G device."
        ),
        details_if_pass=(
            "Every active 5G subscriber has a 5G device."
        )
    )


# ------------------------------------------------------------
# 17. OFFER LOGIC
# ------------------------------------------------------------

if {
    "offer_exposed_90d",
    "offer_redeemed_90d"
}.issubset(subscribers.columns):

    invalid_redemption = (
        (subscribers["offer_redeemed_90d"] == True)
        &
        (subscribers["offer_exposed_90d"] != True)
    )

    evaluate_check(
        "subscribers",
        "Offer redeemed without offer exposure",
        invalid_redemption.sum(),
        severity="CRITICAL",
        details_if_problem=(
            "Some customers redeemed an offer despite "
            "not being marked as exposed."
        ),
        details_if_pass=(
            "All offer redemptions have corresponding exposure."
        )
    )


# ------------------------------------------------------------
# 18. BOOLEAN COLUMN CHECK
# ------------------------------------------------------------

boolean_like_columns = [
    "autopay_enabled",
    "is_5g_device",
    "is_5g_active",
    "family_plan_flag",
    "roaming_user_flag",
    "offer_exposed_90d",
    "offer_redeemed_90d",
    "mnp_enquiry_flag",
    "churn_flag_30d",
    "churn_flag_90d",
]

for col in boolean_like_columns:

    if col in subscribers.columns:

        non_null_values = set(
            subscribers[col]
            .dropna()
            .unique()
            .tolist()
        )

        valid_values = {
            True,
            False,
            1,
            0,
            "True",
            "False",
            "TRUE",
            "FALSE",
            "Yes",
            "No",
            "YES",
            "NO",
            "Y",
            "N",
        }

        invalid_values = (
            non_null_values
            - valid_values
        )

        evaluate_check(
            "subscribers",
            f"Unexpected values in {col}",
            len(invalid_values),
            severity="HIGH",
            details_if_problem=(
                f"Unexpected categorical values: "
                f"{sorted(map(str, invalid_values))}"
            ),
            details_if_pass=(
                f"{col} contains valid boolean-style values."
            )
        )


# ------------------------------------------------------------
# 19. SERVICE REQUEST REFERENTIAL INTEGRITY
# ------------------------------------------------------------

if "service_requests" in dataframes:

    service_requests = dataframes[
        "service_requests"
    ].copy()

    if (
        "subscriber_id" in service_requests.columns
        and
        "subscriber_id" in subscribers.columns
    ):

        subscriber_id_set = set(
            subscribers["subscriber_id"]
            .dropna()
            .astype(str)
        )

        service_subscriber_ids = (
            service_requests["subscriber_id"]
            .dropna()
            .astype(str)
        )

        unmatched = ~service_subscriber_ids.isin(
            subscriber_id_set
        )

        evaluate_check(
            "service_requests",
            "Subscriber FK not found in subscribers",
            unmatched.sum(),
            severity="CRITICAL",
            details_if_problem=(
                "Some service requests reference "
                "subscriber IDs that do not exist."
            ),
            details_if_pass=(
                "Every service request references "
                "a valid subscriber."
            )
        )


# ------------------------------------------------------------
# 20. DETECT POSSIBLE PRIMARY KEY COLUMNS
# ------------------------------------------------------------

possible_id_names = [
    "service_request_id",
    "request_id",
    "ticket_id",
    "site_id",
    "offer_id",
]

for table_name, df in dataframes.items():

    for id_col in possible_id_names:

        if id_col in df.columns:

            missing_pk = int(
                df[id_col].isna().sum()
            )

            duplicate_pk = int(
                df[id_col]
                .dropna()
                .duplicated()
                .sum()
            )

            evaluate_check(
                table_name,
                f"Missing primary-key candidate {id_col}",
                missing_pk,
                severity="CRITICAL",
                details_if_problem=(
                    f"{id_col} contains missing values."
                ),
                details_if_pass=(
                    f"No missing values in {id_col}."
                )
            )

            evaluate_check(
                table_name,
                f"Duplicate primary-key candidate {id_col}",
                duplicate_pk,
                severity="CRITICAL",
                details_if_problem=(
                    f"{id_col} contains duplicate values."
                ),
                details_if_pass=(
                    f"{id_col} values are unique."
                )
            )


# ------------------------------------------------------------
# 21. CIRCLE CONSISTENCY
# ------------------------------------------------------------

tables_with_circle = {}

for table_name, df in dataframes.items():

    if "circle" in df.columns:

        tables_with_circle[table_name] = set(
            df["circle"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

if "subscribers" in tables_with_circle:

    subscriber_circles = tables_with_circle[
        "subscribers"
    ]

    for table_name, circles in tables_with_circle.items():

        if table_name == "subscribers":
            continue

        missing_from_other = (
            subscriber_circles - circles
        )

        evaluate_check(
            table_name,
            "Subscriber circles missing from table",
            len(missing_from_other),
            severity="MEDIUM",
            details_if_problem=(
                f"Circles present in subscribers but absent "
                f"from {table_name}: "
                f"{sorted(missing_from_other)}"
            ),
            details_if_pass=(
                "All subscriber circles are represented."
            )
        )


# ------------------------------------------------------------
# 22. LEAKAGE REVIEW
# ------------------------------------------------------------

leakage_columns = [
    "mnp_enquiry_flag",
    "churn_reason",
    "churn_date",
]

for col in leakage_columns:

    if col in subscribers.columns:

        add_check(
            "subscribers",
            f"Target leakage review: {col}",
            "REVIEW",
            int(subscribers[col].notna().sum()),
            (
                "Retain in raw/business-analysis data, "
                "but review/exclude before predictive modelling."
            )
        )


# ------------------------------------------------------------
# 23. TARGET DISTRIBUTION
# ------------------------------------------------------------

for target in [
    "churn_flag_30d",
    "churn_flag_90d"
]:

    if target in subscribers.columns:

        churn_count = int(
            (subscribers[target] == True).sum()
        )

        churn_rate = (
            churn_count
            / len(subscribers)
            * 100
        )

        add_check(
            "subscribers",
            f"{target} class balance",
            "INFO",
            churn_count,
            (
                f"Positive class: {churn_count:,} "
                f"({churn_rate:.2f}%). "
                f"Class imbalance must be considered "
                f"during modelling."
            )
        )


# ------------------------------------------------------------
# 24. REMOVE TEMPORARY VALIDATION COLUMNS
# ------------------------------------------------------------

temporary_columns = [
    "_parsed_churn_date",
    "_parsed_join_date",
]

for col in temporary_columns:

    if col in subscribers.columns:
        subscribers.drop(
            columns=col,
            inplace=True
        )


# ------------------------------------------------------------
# 25. SAVE VALIDATION OUTPUTS
# ------------------------------------------------------------

checks_df = pd.DataFrame(checks)

issues_df = pd.DataFrame(issues)

if issues_df.empty:

    issues_df = pd.DataFrame(
        columns=[
            "table",
            "check_name",
            "severity",
            "affected_rows",
            "details",
        ]
    )

checks_df.to_csv(
    CHECKS_FILE,
    index=False
)

issues_df.to_csv(
    ISSUES_FILE,
    index=False
)


# ------------------------------------------------------------
# 26. BUILD TEXT REPORT
# ------------------------------------------------------------

report = []

report.append("=" * 100)
report.append("JIO SUBSCRIBER RETENTION PROJECT")
report.append("STEP 02 - DATA QUALITY & VALIDATION REPORT")
report.append("=" * 100)

report.append("")
report.append(
    f"Dataset snapshot date: "
    f"{SNAPSHOT_DATE.date()}"
)

report.append("")
report.append("VALIDATION SUMMARY")
report.append("-" * 100)

status_counts = (
    checks_df["status"]
    .value_counts()
)

for status, count in status_counts.items():

    report.append(
        f"{status:<10}: {count}"
    )

report.append("")
report.append(
    f"Total checks performed: "
    f"{len(checks_df)}"
)

report.append(
    f"Total identified issues: "
    f"{len(issues_df)}"
)

report.append("")
report.append("=" * 100)
report.append("CHECK RESULTS")
report.append("=" * 100)

report.append(
    checks_df.to_string(index=False)
)

report.append("")
report.append("=" * 100)
report.append("ISSUES REQUIRING REVIEW")
report.append("=" * 100)

if len(issues_df) == 0:

    report.append(
        "No genuine data-quality issues detected."
    )

else:

    report.append(
        issues_df.to_string(index=False)
    )

report.append("")
report.append("=" * 100)
report.append("MISSINGNESS - TOP COLUMNS")
report.append("=" * 100)

top_missing = (
    missing_df[
        missing_df["missing_count"] > 0
    ]
    .sort_values(
        "missing_pct",
        ascending=False
    )
    .head(30)
)

report.append(
    top_missing.to_string(index=False)
)

report.append("")
report.append("=" * 100)
report.append("MODELLING CAUTION")
report.append("=" * 100)

report.append(
    "mnp_enquiry_flag, churn_reason and churn_date "
    "must be reviewed as potential target-leakage "
    "variables before predictive modelling."
)

report.append(
    "Step 02 performs validation only. "
    "No raw values have been modified."
)

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 27. TERMINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("VALIDATION SUMMARY")
print("=" * 90)

print(
    checks_df["status"]
    .value_counts()
    .to_string()
)

print(
    f"\nTotal checks performed : "
    f"{len(checks_df)}"
)

print(
    f"Issues requiring review: "
    f"{len(issues_df)}"
)

if len(issues_df) > 0:

    print("\nIssues:")

    display_cols = [
        "table",
        "check_name",
        "severity",
        "affected_rows",
    ]

    print(
        issues_df[
            display_cols
        ].to_string(index=False)
    )

else:

    print(
        "\nNo genuine data-quality "
        "problems were detected."
    )


print("\n" + "=" * 90)
print("STEP 02 COMPLETE")
print("=" * 90)

print("\nGenerated files:")

print(
    "1. outputs/reports/"
    "02_validation_checks.csv"
)

print(
    "2. outputs/reports/"
    "02_data_quality_issues.csv"
)

print(
    "3. outputs/reports/"
    "02_missingness_profile.csv"
)

print(
    "4. outputs/reports/"
    "02_data_quality_report.txt"
)

print(
    "\nNo source data was modified."
)