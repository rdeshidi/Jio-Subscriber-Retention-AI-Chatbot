from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# STEP 01 - RAW DATA EXPLORATION
# ============================================================

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

EXCEL_FILE = RAW_DATA_DIR / "Jio_Retention_Dataset.xlsx"
SUBSCRIBERS_CSV = RAW_DATA_DIR / "subscribers.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "01_data_exploration_report.txt"
SHEET_SUMMARY_FILE = OUTPUT_DIR / "01_sheet_summary.csv"
MISSING_SUMMARY_FILE = OUTPUT_DIR / "01_missing_values_summary.csv"

print("=" * 80)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("STEP 01 - DATA EXPLORATION")
print("=" * 80)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Excel file   : {EXCEL_FILE}")
print(f"CSV file     : {SUBSCRIBERS_CSV}")


# ------------------------------------------------------------
# 2. CHECK FILES EXIST
# ------------------------------------------------------------

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"\nExcel file not found:\n{EXCEL_FILE}"
    )

if not SUBSCRIBERS_CSV.exists():
    raise FileNotFoundError(
        f"\nSubscribers CSV not found:\n{SUBSCRIBERS_CSV}"
    )

print("\n[OK] Required raw data files found.")


# ------------------------------------------------------------
# 3. OPEN EXCEL WORKBOOK
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("EXCEL WORKBOOK INSPECTION")
print("=" * 80)

excel = pd.ExcelFile(EXCEL_FILE)

print("\nSheets found:")
for i, sheet in enumerate(excel.sheet_names, start=1):
    print(f"{i:>2}. {sheet}")


# ------------------------------------------------------------
# 4. LOAD ALL EXCEL SHEETS
# ------------------------------------------------------------

dataframes = {}

print("\nLoading sheets...")

for sheet_name in excel.sheet_names:
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
    dataframes[sheet_name] = df

    print(
        f"[LOADED] {sheet_name:<22} "
        f"Rows: {len(df):>7,} | Columns: {len(df.columns):>3}"
    )


# ------------------------------------------------------------
# 5. LOAD SEPARATE SUBSCRIBERS CSV
# ------------------------------------------------------------

print("\nLoading subscribers.csv...")

subscribers_csv = pd.read_csv(SUBSCRIBERS_CSV)

print(
    f"[LOADED] subscribers.csv       "
    f"Rows: {len(subscribers_csv):,} | "
    f"Columns: {len(subscribers_csv.columns)}"
)


# ------------------------------------------------------------
# 6. CREATE SHEET-LEVEL SUMMARY
# ------------------------------------------------------------

sheet_summary = []

for sheet_name, df in dataframes.items():

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())

    missing_pct = (
        (missing_cells / total_cells) * 100
        if total_cells > 0
        else 0
    )

    sheet_summary.append(
        {
            "sheet_name": sheet_name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_cells": missing_cells,
            "missing_pct": round(missing_pct, 2),
        }
    )

sheet_summary_df = pd.DataFrame(sheet_summary)

sheet_summary_df.to_csv(
    SHEET_SUMMARY_FILE,
    index=False
)

print("\n" + "=" * 80)
print("SHEET SUMMARY")
print("=" * 80)

print(sheet_summary_df.to_string(index=False))


# ------------------------------------------------------------
# 7. COLUMN-BY-COLUMN MISSING VALUE SUMMARY
# ------------------------------------------------------------

missing_records = []

for sheet_name, df in dataframes.items():

    for column in df.columns:

        missing_count = int(df[column].isna().sum())

        missing_pct = (
            missing_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        missing_records.append(
            {
                "sheet_name": sheet_name,
                "column_name": column,
                "dtype": str(df[column].dtype),
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
                "unique_non_null": int(
                    df[column].nunique(dropna=True)
                ),
            }
        )

missing_summary_df = pd.DataFrame(missing_records)

missing_summary_df.to_csv(
    MISSING_SUMMARY_FILE,
    index=False
)


# ------------------------------------------------------------
# 8. BUILD TEXT REPORT
# ------------------------------------------------------------

report_lines = []


def add_report(text=""):
    report_lines.append(str(text))


add_report("=" * 100)
add_report("JIO SUBSCRIBER RETENTION PROJECT")
add_report("STEP 01 - RAW DATA EXPLORATION REPORT")
add_report("=" * 100)

add_report()
add_report(f"Excel source: {EXCEL_FILE.name}")
add_report(f"CSV source  : {SUBSCRIBERS_CSV.name}")

add_report()
add_report("=" * 100)
add_report("1. EXCEL SHEETS")
add_report("=" * 100)

for sheet_name in excel.sheet_names:
    add_report(sheet_name)


# ------------------------------------------------------------
# 9. DETAILED SHEET INSPECTION
# ------------------------------------------------------------

for sheet_name, df in dataframes.items():

    add_report()
    add_report("=" * 100)
    add_report(f"SHEET: {sheet_name}")
    add_report("=" * 100)

    add_report(f"Rows       : {df.shape[0]:,}")
    add_report(f"Columns    : {df.shape[1]:,}")
    add_report(f"Duplicates : {df.duplicated().sum():,}")
    add_report(
        f"Missing cells: "
        f"{df.isna().sum().sum():,}"
    )

    add_report()
    add_report("COLUMN INFORMATION")
    add_report("-" * 100)

    column_info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing": df.isna().sum().values,
            "missing_pct": (
                df.isna().mean().values * 100
            ).round(2),
            "unique": [
                df[col].nunique(dropna=True)
                for col in df.columns
            ],
        }
    )

    add_report(column_info.to_string(index=False))

    add_report()
    add_report("FIRST 5 ROWS")
    add_report("-" * 100)
    add_report(df.head().to_string(index=False))


# ------------------------------------------------------------
# 10. SUBSCRIBER DATASET DEEPER INSPECTION
# ------------------------------------------------------------

if "subscribers" in dataframes:

    subscribers = dataframes["subscribers"]

    print("\n" + "=" * 80)
    print("MAIN SUBSCRIBER DATASET")
    print("=" * 80)

    print(
        f"\nShape: "
        f"{subscribers.shape[0]:,} rows x "
        f"{subscribers.shape[1]} columns"
    )

    print("\nColumns:")

    for i, col in enumerate(subscribers.columns, start=1):
        print(f"{i:>2}. {col}")

    add_report()
    add_report("=" * 100)
    add_report("MAIN SUBSCRIBER DATASET")
    add_report("=" * 100)

    add_report(
        f"Shape: {subscribers.shape[0]:,} rows x "
        f"{subscribers.shape[1]} columns"
    )

    add_report()
    add_report("SUBSCRIBER COLUMNS")
    add_report("-" * 100)

    for i, col in enumerate(subscribers.columns, start=1):
        add_report(f"{i:>2}. {col}")


    # --------------------------------------------------------
    # 11. SUBSCRIBER ID CHECK
    # --------------------------------------------------------

    if "subscriber_id" in subscribers.columns:

        total_rows = len(subscribers)
        unique_ids = subscribers["subscriber_id"].nunique(
            dropna=True
        )
        missing_ids = subscribers["subscriber_id"].isna().sum()
        duplicate_ids = subscribers["subscriber_id"].duplicated().sum()

        print("\nSubscriber ID check:")
        print(f"Total rows       : {total_rows:,}")
        print(f"Unique IDs       : {unique_ids:,}")
        print(f"Missing IDs      : {missing_ids:,}")
        print(f"Duplicate IDs    : {duplicate_ids:,}")

        add_report()
        add_report("SUBSCRIBER ID CHECK")
        add_report("-" * 100)
        add_report(f"Total rows    : {total_rows:,}")
        add_report(f"Unique IDs    : {unique_ids:,}")
        add_report(f"Missing IDs   : {missing_ids:,}")
        add_report(f"Duplicate IDs : {duplicate_ids:,}")


    # --------------------------------------------------------
    # 12. CHURN TARGET DISTRIBUTION
    # --------------------------------------------------------

    target_columns = [
        "churn_flag_30d",
        "churn_flag_90d",
    ]

    print("\n" + "=" * 80)
    print("CHURN TARGET DISTRIBUTION")
    print("=" * 80)

    add_report()
    add_report("=" * 100)
    add_report("CHURN TARGET DISTRIBUTION")
    add_report("=" * 100)

    for target in target_columns:

        if target in subscribers.columns:

            counts = (
                subscribers[target]
                .value_counts(dropna=False)
                .sort_index()
            )

            percentages = (
                subscribers[target]
                .value_counts(
                    normalize=True,
                    dropna=False
                )
                .sort_index()
                * 100
            )

            target_summary = pd.DataFrame(
                {
                    "count": counts,
                    "percentage": percentages.round(2),
                }
            )

            print(f"\n{target}:")
            print(target_summary)

            add_report()
            add_report(target)
            add_report("-" * 60)
            add_report(target_summary.to_string())


    # --------------------------------------------------------
    # 13. CATEGORICAL COLUMN OVERVIEW
    # --------------------------------------------------------

    categorical_columns = subscribers.select_dtypes(
    include=["object", "str", "category"]
).columns

    add_report()
    add_report("=" * 100)
    add_report("CATEGORICAL COLUMN OVERVIEW")
    add_report("=" * 100)

    for col in categorical_columns:

        add_report()
        add_report(f"COLUMN: {col}")
        add_report("-" * 60)

        add_report(
            subscribers[col]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )


    # --------------------------------------------------------
    # 14. NUMERICAL SUMMARY
    # --------------------------------------------------------

    numeric_df = subscribers.select_dtypes(
        include=[np.number]
    )

    add_report()
    add_report("=" * 100)
    add_report("NUMERICAL SUMMARY - SUBSCRIBERS")
    add_report("=" * 100)

    if not numeric_df.empty:

        numerical_summary = (
            numeric_df.describe()
            .T
            .round(2)
        )

        add_report(
            numerical_summary.to_string()
        )


    # --------------------------------------------------------
    # 15. DATE RANGE CHECKS
    # --------------------------------------------------------

    possible_date_columns = [
        "join_date",
        "churn_date",
    ]

    print("\n" + "=" * 80)
    print("DATE RANGE CHECK")
    print("=" * 80)

    add_report()
    add_report("=" * 100)
    add_report("DATE RANGE CHECK")
    add_report("=" * 100)

    for col in possible_date_columns:

        if col in subscribers.columns:

            parsed = pd.to_datetime(
                subscribers[col],
                errors="coerce"
            )

            valid_dates = parsed.notna().sum()

            if valid_dates > 0:

                print(
                    f"{col:<20} "
                    f"Min: {parsed.min()} | "
                    f"Max: {parsed.max()} | "
                    f"Valid: {valid_dates:,}"
                )

                add_report(
                    f"{col}: "
                    f"min={parsed.min()}, "
                    f"max={parsed.max()}, "
                    f"valid={valid_dates:,}"
                )


    # --------------------------------------------------------
    # 16. IMPORTANT BUSINESS FIELD CHECK
    # --------------------------------------------------------

    expected_business_fields = [
        "subscriber_id",
        "circle",
        "join_date",
        "tenure_months",
        "plan_type",
        "plan_price_inr",
        "arpu_last_month_inr",
        "arpu_3m_avg_inr",
        "recharge_count_6m",
        "avg_recharge_gap_days",
        "days_since_last_recharge",
        "payment_failures_6m",
        "data_gb_last_month",
        "is_5g_active",
        "avg_sinr_db",
        "drop_call_rate_pct",
        "site_congestion_score",
        "complaints_6m",
        "unresolved_complaints",
        "avg_resolution_days",
        "offer_exposed_90d",
        "offer_redeemed_90d",
        "mnp_enquiry_flag",
        "churn_flag_30d",
        "churn_flag_90d",
        "churn_reason",
        "churn_date",
    ]

    add_report()
    add_report("=" * 100)
    add_report("IMPORTANT BUSINESS FIELD CHECK")
    add_report("=" * 100)

    print("\nImportant business fields:")

    for col in expected_business_fields:

        status = (
            "FOUND"
            if col in subscribers.columns
            else "MISSING"
        )

        print(f"{col:<35} {status}")

        add_report(
            f"{col:<35} {status}"
        )


    # --------------------------------------------------------
    # 17. POTENTIAL TARGET LEAKAGE FIELDS
    # --------------------------------------------------------

    leakage_columns = [
        "mnp_enquiry_flag",
        "churn_reason",
        "churn_date",
    ]

    print("\n" + "=" * 80)
    print("POTENTIAL TARGET LEAKAGE COLUMNS")
    print("=" * 80)

    add_report()
    add_report("=" * 100)
    add_report("POTENTIAL TARGET LEAKAGE COLUMNS")
    add_report("=" * 100)

    for col in leakage_columns:

        if col in subscribers.columns:

            print(
                f"[REVIEW LATER] {col}"
            )

            add_report(
                f"[REVIEW LATER] {col}"
            )

    add_report()
    add_report(
        "NOTE: No columns are being removed in Step 01. "
        "These columns are only flagged for later modelling review."
    )


# ------------------------------------------------------------
# 18. COMPARE EXCEL SUBSCRIBERS VS CSV
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("EXCEL SUBSCRIBERS VS subscribers.csv")
print("=" * 80)

add_report()
add_report("=" * 100)
add_report("EXCEL SUBSCRIBERS VS subscribers.csv")
add_report("=" * 100)

if "subscribers" in dataframes:

    excel_subscribers = dataframes["subscribers"]

    print(
        f"Excel shape : {excel_subscribers.shape}"
    )

    print(
        f"CSV shape   : {subscribers_csv.shape}"
    )

    same_rows = (
        excel_subscribers.shape[0]
        == subscribers_csv.shape[0]
    )

    same_columns_count = (
        excel_subscribers.shape[1]
        == subscribers_csv.shape[1]
    )

    same_column_names = (
        list(excel_subscribers.columns)
        == list(subscribers_csv.columns)
    )

    print(f"Same row count        : {same_rows}")
    print(f"Same column count     : {same_columns_count}")
    print(f"Same column names     : {same_column_names}")

    add_report(
        f"Excel shape: {excel_subscribers.shape}"
    )

    add_report(
        f"CSV shape  : {subscribers_csv.shape}"
    )

    add_report(
        f"Same row count    : {same_rows}"
    )

    add_report(
        f"Same column count : {same_columns_count}"
    )

    add_report(
        f"Same column names : {same_column_names}"
    )

    if (
        "subscriber_id" in excel_subscribers.columns
        and
        "subscriber_id" in subscribers_csv.columns
    ):

        excel_ids = set(
            excel_subscribers["subscriber_id"]
            .dropna()
            .astype(str)
        )

        csv_ids = set(
            subscribers_csv["subscriber_id"]
            .dropna()
            .astype(str)
        )

        same_ids = excel_ids == csv_ids

        print(
            f"Same subscriber ID set: {same_ids}"
        )

        add_report(
            f"Same subscriber ID set: {same_ids}"
        )


# ------------------------------------------------------------
# 19. CHECK RELATIONSHIP BETWEEN SUBSCRIBERS AND SERVICE REQUESTS
# ------------------------------------------------------------

if (
    "subscribers" in dataframes
    and
    "service_requests" in dataframes
):

    subscribers = dataframes["subscribers"]
    service_requests = dataframes["service_requests"]

    if (
        "subscriber_id" in subscribers.columns
        and
        "subscriber_id" in service_requests.columns
    ):

        subscriber_ids = set(
            subscribers["subscriber_id"]
            .dropna()
            .astype(str)
        )

        sr_ids = service_requests[
            "subscriber_id"
        ].dropna().astype(str)

        matched = sr_ids.isin(
            subscriber_ids
        ).sum()

        unmatched = (
            len(sr_ids) - matched
        )

        print("\n" + "=" * 80)
        print("SUBSCRIBERS <-> SERVICE REQUESTS RELATIONSHIP")
        print("=" * 80)

        print(
            f"Service request rows       : "
            f"{len(service_requests):,}"
        )

        print(
            f"Matched subscriber IDs     : "
            f"{matched:,}"
        )

        print(
            f"Unmatched subscriber IDs   : "
            f"{unmatched:,}"
        )

        add_report()
        add_report("=" * 100)
        add_report(
            "SUBSCRIBERS <-> SERVICE REQUESTS RELATIONSHIP"
        )
        add_report("=" * 100)

        add_report(
            f"Service request rows     : "
            f"{len(service_requests):,}"
        )

        add_report(
            f"Matched subscriber IDs   : "
            f"{matched:,}"
        )

        add_report(
            f"Unmatched subscriber IDs : "
            f"{unmatched:,}"
        )


# ------------------------------------------------------------
# 20. SAVE TEXT REPORT
# ------------------------------------------------------------

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 21. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 01 COMPLETE")
print("=" * 80)

print("\nGenerated files:")

print(
    f"1. {REPORT_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"2. {SHEET_SUMMARY_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"3. {MISSING_SUMMARY_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    "\nIMPORTANT: Step 01 only explored the raw data."
)

print(
    "No cleaning, deletion, imputation, encoding, "
    "or modelling has been performed."
)