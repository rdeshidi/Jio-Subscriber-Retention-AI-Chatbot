from pathlib import Path
import pandas as pd


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# STEP 06C - PREPARE SQL-SAFE CSV EXPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SQL_EXPORT_DIR = PROJECT_ROOT / "outputs" / "sql_exports"

SQL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


TABLES = {
    "subscribers": "subscribers_clean.csv",
    "service_requests": "service_requests_clean.csv",
    "network_sites": "network_sites_clean.csv",
    "circle_monthly_kpi": "circle_monthly_kpi_clean.csv",
    "circle_targets": "circle_targets_clean.csv",
    "offer_catalogue": "offer_catalogue_clean.csv",
}


BOOLEAN_COLUMNS = {
    "subscribers": [
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
    ],

    "service_requests": [
        "sla_breach_flag",
        "reopened_flag",
    ],

    "network_sites": [
        "congestion_flag",
    ],
}


DATE_COLUMNS = {
    "subscribers": {
        "join_date": "%Y-%m-%d",
        "churn_date": "%Y-%m-%d",
    },

    "service_requests": {
        "raised_date": "%Y-%m-%d",
        "resolved_date": "%Y-%m-%d %H:%M:%S",
    },

    "circle_monthly_kpi": {
        "month_end": "%Y-%m-%d",
    },

    "offer_catalogue": {
        "valid_from": "%Y-%m-%d",
        "valid_to": "%Y-%m-%d",
    },
}


print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("STEP 06C - PREPARE SQL-SAFE EXPORTS")
print("=" * 90)


manifest = []


def convert_boolean(series):

    mapping = {
        "true": "1",
        "false": "0",
        "1": "1",
        "0": "0",
        "yes": "1",
        "no": "0",
        "y": "1",
        "n": "0",
    }

    result = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    return result


for table_name, filename in TABLES.items():

    input_file = PROCESSED_DIR / filename

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing processed file:\n{input_file}"
        )

    # Read as strings so IDs/codes such as PIN codes
    # are not accidentally reformatted.
    df = pd.read_csv(
        input_file,
        dtype="string",
        keep_default_na=True
    )

    original_rows = len(df)

    print(
        f"\nProcessing {table_name}: "
        f"{df.shape[0]:,} rows x "
        f"{df.shape[1]} columns"
    )


    # --------------------------------------------------------
    # BOOLEAN -> 1 / 0
    # --------------------------------------------------------

    for col in BOOLEAN_COLUMNS.get(
        table_name,
        []
    ):

        if col not in df.columns:
            continue

        original_non_null = (
            df[col].notna().sum()
        )

        converted = convert_boolean(
            df[col]
        )

        failed = (
            df[col].notna()
            &
            converted.isna()
        )

        if failed.any():

            unexpected = (
                df.loc[
                    failed,
                    col
                ]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                f"{table_name}.{col} contains "
                f"unexpected boolean values: "
                f"{unexpected}"
            )

        df[col] = converted

        print(
            f"  [OK] Boolean: {col:<30} "
            f"{original_non_null:,} values converted"
        )


    # --------------------------------------------------------
    # STANDARDISE DATES
    # --------------------------------------------------------

    for col, date_format in DATE_COLUMNS.get(
        table_name,
        {}
    ).items():

        if col not in df.columns:
            continue

        original_non_null = (
            df[col].notna().sum()
        )

        parsed = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        failed = (
            df[col].notna()
            &
            parsed.isna()
        )

        if failed.any():

            bad_values = (
                df.loc[
                    failed,
                    col
                ]
                .head(10)
                .tolist()
            )

            raise ValueError(
                f"{table_name}.{col} contains "
                f"unparseable dates: {bad_values}"
            )

        formatted = parsed.dt.strftime(
            date_format
        )

        df[col] = formatted

        print(
            f"  [OK] Date:    {col:<30} "
            f"{original_non_null:,} values formatted"
        )


    # --------------------------------------------------------
    # REMOVE ACCIDENTAL WHITESPACE
    # --------------------------------------------------------

    for col in df.columns:

        if pd.api.types.is_string_dtype(
            df[col]
        ):

            df[col] = (
                df[col]
                .str.strip()
            )


    # --------------------------------------------------------
    # SAVE SQL-SAFE CSV
    # --------------------------------------------------------

    output_file = (
        SQL_EXPORT_DIR
        /
        f"{table_name}_sql.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
        na_rep=""
    )

    if len(df) != original_rows:
        raise ValueError(
            f"Row count changed for {table_name}."
        )

    manifest.append(
        {
            "table_name": table_name,
            "rows": len(df),
            "columns": len(df.columns),
            "sql_file": output_file.name,
        }
    )

    print(
        f"  [SAVED] {output_file.name}"
    )


# ------------------------------------------------------------
# SAVE MANIFEST
# ------------------------------------------------------------

manifest_df = pd.DataFrame(
    manifest
)

manifest_file = (
    SQL_EXPORT_DIR
    / "sql_import_manifest.csv"
)

manifest_df.to_csv(
    manifest_file,
    index=False
)


print("\n" + "=" * 90)
print("SQL EXPORT SUMMARY")
print("=" * 90)

print(
    manifest_df.to_string(
        index=False
    )
)

print("\n" + "=" * 90)
print("STEP 06C COMPLETE")
print("=" * 90)

print(
    "\nSQL-ready files are in:"
)

print(
    "outputs/sql_exports/"
)