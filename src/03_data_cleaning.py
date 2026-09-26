from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# STEP 03 - DATA CLEANING / ANALYSIS-READY PREPARATION
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

EXCEL_FILE = RAW_DATA_DIR / "Jio_Retention_Dataset.xlsx"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLEANING_LOG_FILE = REPORT_DIR / "03_cleaning_log.csv"
REPORT_FILE = REPORT_DIR / "03_cleaning_report.txt"
MISMATCH_FILE = REPORT_DIR / "03_churn_30d_date_mismatch.csv"

SNAPSHOT_DATE = pd.Timestamp("2026-08-31")


print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("STEP 03 - DATA CLEANING / ANALYSIS-READY PREPARATION")
print("=" * 90)


# ------------------------------------------------------------
# 2. LOAD WORKBOOK
# ------------------------------------------------------------

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"Excel workbook not found:\n{EXCEL_FILE}"
    )

excel = pd.ExcelFile(EXCEL_FILE)

analysis_tables = [
    "subscribers",
    "service_requests",
    "network_sites",
    "circle_monthly_kpi",
    "circle_targets",
    "offer_catalogue",
]

dataframes = {}

for table in analysis_tables:

    if table in excel.sheet_names:

        df = pd.read_excel(
            EXCEL_FILE,
            sheet_name=table
        )

        dataframes[table] = df

        print(
            f"[LOADED] {table:<22} "
            f"{df.shape[0]:>7,} rows x "
            f"{df.shape[1]:>3} columns"
        )


# ------------------------------------------------------------
# 3. CLEANING LOG
# ------------------------------------------------------------

cleaning_log = []


def log_action(
    table,
    action,
    affected_rows,
    details
):
    cleaning_log.append(
        {
            "table": table,
            "action": action,
            "affected_rows": int(affected_rows),
            "details": details,
        }
    )


# ------------------------------------------------------------
# 4. STANDARDISE COLUMN NAMES
# ------------------------------------------------------------

for table_name, df in dataframes.items():

    old_columns = list(df.columns)

    new_columns = [
        str(col).strip()
        for col in df.columns
    ]

    changed = sum(
        old != new
        for old, new in zip(
            old_columns,
            new_columns
        )
    )

    df.columns = new_columns

    log_action(
        table_name,
        "Strip whitespace from column names",
        changed,
        (
            "Leading/trailing whitespace removed "
            "from column names where present."
        ),
    )


# ------------------------------------------------------------
# 5. STANDARDISE STRING VALUES
# ------------------------------------------------------------

for table_name, df in dataframes.items():

    changed_cells = 0

    string_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in string_columns:

        original = df[col].copy()

        # Preserve missing values while trimming text
        df[col] = df[col].apply(
            lambda x:
            x.strip()
            if isinstance(x, str)
            else x
        )

        changed_cells += int(
            (
                original.fillna("__MISSING__")
                !=
                df[col].fillna("__MISSING__")
            ).sum()
        )

    log_action(
        table_name,
        "Trim string values",
        changed_cells,
        (
            "Leading/trailing whitespace removed "
            "from text values. Missing values preserved."
        ),
    )


# ------------------------------------------------------------
# 6. CONVERT EMPTY STRINGS TO MISSING
# ------------------------------------------------------------

for table_name, df in dataframes.items():

    empty_count = 0

    string_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in string_columns:

        empty_mask = (
            df[col]
            .astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
        )

        empty_count += int(
            empty_mask.sum()
        )

        if empty_mask.any():
            df.loc[
                empty_mask,
                col
            ] = pd.NA

    log_action(
        table_name,
        "Convert empty strings to missing values",
        empty_count,
        (
            "Blank strings converted to NA. "
            "Existing legitimate missing values retained."
        ),
    )


# ------------------------------------------------------------
# 7. SAFE DATE STANDARDISATION
# ------------------------------------------------------------

def looks_like_date_column(column_name):

    name = column_name.lower()

    exact_date_names = {
        "month",
        "month_end",
        "snapshot_date",
        "report_month",
        "join_date",
        "churn_date",
        "raised_date",
        "resolved_date",
    }

    return (
        name in exact_date_names
        or name.endswith("_date")
    )


for table_name, df in dataframes.items():

    for col in df.columns:

        if not looks_like_date_column(col):
            continue

        # Skip columns that are already datetime
        if pd.api.types.is_datetime64_any_dtype(
            df[col]
        ):
            continue

        non_missing = df[col].notna().sum()

        if non_missing == 0:
            continue

        parsed = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        parse_success = (
            parsed.notna().sum()
            /
            non_missing
        )

        # Only convert automatically if at least
        # 95% of existing values are valid dates.
        if parse_success >= 0.95:

            converted_count = int(
                parsed.notna().sum()
            )

            df[col] = parsed

            log_action(
                table_name,
                f"Parse date column: {col}",
                converted_count,
                (
                    f"{col} converted to datetime. "
                    f"Parse success: "
                    f"{parse_success * 100:.2f}%."
                ),
            )


# ------------------------------------------------------------
# 8. SUBSCRIBER-SPECIFIC PREPARATION
# ------------------------------------------------------------

subscribers = dataframes["subscribers"].copy()


# Explicitly parse the two important dates
for col in [
    "join_date",
    "churn_date",
]:

    if col in subscribers.columns:

        subscribers[col] = pd.to_datetime(
            subscribers[col],
            errors="coerce"
        )


# ------------------------------------------------------------
# 9. CHECK KNOWN 30-DAY LABEL / DATE MISMATCH
# ------------------------------------------------------------

if {
    "churn_flag_30d",
    "churn_date",
}.issubset(subscribers.columns):

    max_30d_date = (
        SNAPSHOT_DATE
        + pd.Timedelta(days=30)
    )

    mismatch_mask = (
        (subscribers["churn_flag_30d"] == True)
        &
        subscribers["churn_date"].notna()
        &
        (
            subscribers["churn_date"]
            > max_30d_date
        )
    )

    mismatch_rows = subscribers.loc[
        mismatch_mask
    ].copy()

    if not mismatch_rows.empty:

        mismatch_rows[
            "days_after_snapshot"
        ] = (
            mismatch_rows["churn_date"]
            - SNAPSHOT_DATE
        ).dt.days

        mismatch_rows.to_csv(
            MISMATCH_FILE,
            index=False
        )

    log_action(
        "subscribers",
        "Document 30-day churn/date mismatch",
        mismatch_mask.sum(),
        (
            "Rows were documented only. "
            "churn_flag_30d was NOT altered and "
            "churn_date was NOT altered."
        ),
    )


# ------------------------------------------------------------
# 10. PRESERVE AUTHORITATIVE TARGET
# ------------------------------------------------------------

log_action(
    "subscribers",
    "Preserve churn_flag_30d",
    0,
    (
        "The supplied churn_flag_30d remains "
        "the authoritative modelling target. "
        "It was not regenerated from churn_date."
    ),
)


# ------------------------------------------------------------
# 11. EXPECTED MISSINGNESS
# ------------------------------------------------------------

# We deliberately do NOT fill churn_reason/churn_date
# because these should normally be missing for non-churners.

for col in [
    "churn_reason",
    "churn_date",
]:

    if col in subscribers.columns:

        missing_count = int(
            subscribers[col]
            .isna()
            .sum()
        )

        log_action(
            "subscribers",
            f"Preserve expected missingness: {col}",
            missing_count,
            (
                f"Missing values in {col} retained. "
                "No imputation performed."
            ),
        )


# ------------------------------------------------------------
# 12. LEAKAGE FIELDS
# ------------------------------------------------------------

leakage_fields = [
    "mnp_enquiry_flag",
    "churn_reason",
    "churn_date",
]

existing_leakage_fields = [
    col
    for col in leakage_fields
    if col in subscribers.columns
]

log_action(
    "subscribers",
    "Preserve leakage fields for Day-1 analysis",
    0,
    (
        "Fields retained in processed data: "
        f"{existing_leakage_fields}. "
        "They must be excluded/reviewed during "
        "predictive modelling."
    ),
)


# Replace working subscriber table
dataframes[
    "subscribers"
] = subscribers


# ------------------------------------------------------------
# 13. FINAL DUPLICATE CHECK
# ------------------------------------------------------------

for table_name, df in dataframes.items():

    duplicate_count = int(
        df.duplicated().sum()
    )

    log_action(
        table_name,
        "Final duplicate-row check",
        duplicate_count,
        (
            "Duplicate rows detected after preparation."
            if duplicate_count > 0
            else
            "No duplicate rows detected."
        ),
    )


# ------------------------------------------------------------
# 14. SAVE PROCESSED CSV FILES
# ------------------------------------------------------------

saved_files = []

for table_name, df in dataframes.items():

    output_file = (
        PROCESSED_DATA_DIR
        /
        f"{table_name}_clean.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    saved_files.append(
        output_file
    )

    print(
        f"[SAVED] {output_file.name:<35} "
        f"{df.shape[0]:>7,} rows x "
        f"{df.shape[1]:>3} columns"
    )


# ------------------------------------------------------------
# 15. VALIDATE SUBSCRIBER SHAPE
# ------------------------------------------------------------

raw_subscribers = pd.read_excel(
    EXCEL_FILE,
    sheet_name="subscribers"
)

processed_subscribers = dataframes[
    "subscribers"
]

same_rows = (
    len(raw_subscribers)
    ==
    len(processed_subscribers)
)

same_columns = (
    len(raw_subscribers.columns)
    ==
    len(processed_subscribers.columns)
)

same_ids = True

if "subscriber_id" in processed_subscribers.columns:

    same_ids = (
        set(
            raw_subscribers[
                "subscriber_id"
            ].astype(str)
        )
        ==
        set(
            processed_subscribers[
                "subscriber_id"
            ].astype(str)
        )
    )


# ------------------------------------------------------------
# 16. SAVE CLEANING LOG
# ------------------------------------------------------------

cleaning_log_df = pd.DataFrame(
    cleaning_log
)

cleaning_log_df.to_csv(
    CLEANING_LOG_FILE,
    index=False
)


# ------------------------------------------------------------
# 17. BUILD CLEANING REPORT
# ------------------------------------------------------------

report = []

report.append("=" * 100)
report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)
report.append(
    "STEP 03 - DATA CLEANING / "
    "ANALYSIS-READY PREPARATION REPORT"
)
report.append("=" * 100)

report.append("")
report.append(
    "Cleaning strategy: conservative."
)

report.append(
    "The supplied synthetic sandbox was already "
    "structurally clean, so no unnecessary "
    "imputation, row deletion, target rewriting, "
    "or outlier removal was performed."
)

report.append("")
report.append("=" * 100)
report.append("PROCESSED TABLES")
report.append("=" * 100)

for table_name, df in dataframes.items():

    report.append(
        f"{table_name:<25} "
        f"{df.shape[0]:>7,} rows x "
        f"{df.shape[1]:>3} columns"
    )

report.append("")
report.append("=" * 100)
report.append("SUBSCRIBER INTEGRITY")
report.append("=" * 100)

report.append(
    f"Raw subscriber rows       : "
    f"{len(raw_subscribers):,}"
)

report.append(
    f"Processed subscriber rows : "
    f"{len(processed_subscribers):,}"
)

report.append(
    f"Same row count            : "
    f"{same_rows}"
)

report.append(
    f"Same column count         : "
    f"{same_columns}"
)

report.append(
    f"Same subscriber ID set    : "
    f"{same_ids}"
)

if {
    "churn_flag_30d",
    "churn_date",
}.issubset(
    processed_subscribers.columns
):

    mismatch_count = int(
        (
            (processed_subscribers[
                "churn_flag_30d"
            ] == True)
            &
            processed_subscribers[
                "churn_date"
            ].notna()
            &
            (
                processed_subscribers[
                    "churn_date"
                ]
                >
                (
                    SNAPSHOT_DATE
                    + pd.Timedelta(
                        days=30
                    )
                )
            )
        ).sum()
    )

    report.append("")
    report.append(
        f"Known 30-day label/date mismatch: "
        f"{mismatch_count:,} rows"
    )

    report.append(
        "Action: documented only. "
        "The supplied target was preserved."
    )

report.append("")
report.append("=" * 100)
report.append("IMPORTANT MODELLING NOTE")
report.append("=" * 100)

report.append(
    "mnp_enquiry_flag, churn_reason and "
    "churn_date remain available for Day-1 "
    "business analysis, but must be reviewed/"
    "excluded before predictive modelling "
    "because of target leakage."
)

report.append("")
report.append("=" * 100)
report.append("CLEANING LOG")
report.append("=" * 100)

report.append(
    cleaning_log_df.to_string(
        index=False
    )
)

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 18. TERMINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("CLEANING SUMMARY")
print("=" * 90)

print(
    f"Subscriber row count preserved : "
    f"{same_rows}"
)

print(
    f"Subscriber column count preserved: "
    f"{same_columns}"
)

print(
    f"Subscriber ID set preserved    : "
    f"{same_ids}"
)

if MISMATCH_FILE.exists():

    print(
        "\nKnown 30-day churn/date "
        "mismatch documented:"
    )

    print(
        f"{MISMATCH_FILE.relative_to(PROJECT_ROOT)}"
    )


print("\n" + "=" * 90)
print("STEP 03 COMPLETE")
print("=" * 90)

print("\nProcessed datasets:")

for file in saved_files:

    print(
        f"- {file.relative_to(PROJECT_ROOT)}"
    )

print("\nReports:")

print(
    "- outputs/reports/"
    "03_cleaning_log.csv"
)

print(
    "- outputs/reports/"
    "03_cleaning_report.txt"
)

print(
    "- outputs/reports/"
    "03_churn_30d_date_mismatch.csv"
)

print(
    "\nIMPORTANT:"
)

print(
    "No rows were deleted."
)

print(
    "No missing values were imputed."
)

print(
    "No churn targets were changed."
)

print(
    "No leakage columns were removed yet."
)