from pathlib import Path
import importlib.util
import json
import os

import pandas as pd

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    inspect,
    text,
)

from sqlalchemy.engine import URL


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# TEXT-TO-SQL CHATBOT
# STEP 30 - DATABASE CONNECTION + SAFE SCHEMA DISCOVERY
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
)

ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)

GITIGNORE_FILE = (
    PROJECT_ROOT
    / ".gitignore"
)

SAFETY_FILE = (
    SRC_DIR
    / "29_chatbot_privacy_and_sql_safety.py"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("STEP 30: CHATBOT DATABASE SETUP")
print("=" * 90)


# ------------------------------------------------------------
# 2. LOAD STEP 29 SAFETY MODULE
# ------------------------------------------------------------

if not SAFETY_FILE.exists():

    raise FileNotFoundError(
        f"Step 29 safety file not found:\n"
        f"{SAFETY_FILE}"
    )


spec = importlib.util.spec_from_file_location(
    "jio_chatbot_safety",
    SAFETY_FILE,
)


if (
    spec is None
    or spec.loader is None
):

    raise ImportError(
        "Unable to load Step 29 "
        "privacy and SQL safety module."
    )


safety = (
    importlib.util
    .module_from_spec(
        spec
    )
)


spec.loader.exec_module(
    safety
)


ALLOWED_TABLES = (
    safety.ALLOWED_TABLES
)

validate_sql = (
    safety.validate_sql
)

enforce_row_limit = (
    safety.enforce_row_limit
)

sanitize_results = (
    safety.sanitize_results
)

is_sensitive_column = (
    safety.is_sensitive_column
)


print(
    "\n[OK] Step 29 privacy and "
    "SQL safety rules loaded."
)


# ------------------------------------------------------------
# 3. CHECK .ENV SECURITY
# ------------------------------------------------------------

if not ENV_FILE.exists():

    raise FileNotFoundError(
        f".env file not found:\n"
        f"{ENV_FILE}"
    )


print(
    "[OK] .env file found."
)


# Check whether .env is excluded from Git.

gitignore_protected = False


if GITIGNORE_FILE.exists():

    gitignore_lines = [
        line.strip()
        for line in (
            GITIGNORE_FILE
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
    ]


    gitignore_protected = (
        ".env"
        in gitignore_lines
    )


if gitignore_protected:

    print(
        "[OK] .env is protected "
        "by .gitignore."
    )

else:

    print(
        "[WARNING] .env was not found "
        "in .gitignore."
    )

    print(
        "Add this line to .gitignore:"
    )

    print(
        ".env"
    )


# ------------------------------------------------------------
# 4. LOAD DATABASE SETTINGS
# ------------------------------------------------------------

load_dotenv(
    ENV_FILE
)


DB_HOST = os.getenv(
    "JIO_DB_HOST"
)

DB_PORT = os.getenv(
    "JIO_DB_PORT",
    "3306"
)

DB_NAME = os.getenv(
    "JIO_DB_NAME"
)

DB_USER = os.getenv(
    "JIO_DB_USER"
)

DB_PASSWORD = os.getenv(
    "JIO_DB_PASSWORD"
)


required_settings = {
    "JIO_DB_HOST":
        DB_HOST,

    "JIO_DB_NAME":
        DB_NAME,

    "JIO_DB_USER":
        DB_USER,

    "JIO_DB_PASSWORD":
        DB_PASSWORD,
}


missing_settings = [
    name
    for name, value
    in required_settings.items()
    if not value
]


if missing_settings:

    raise ValueError(
        "Missing settings in .env:\n"
        + "\n".join(
            missing_settings
        )
    )


# ------------------------------------------------------------
# 5. REQUIRE READ-ONLY CHATBOT USER
# ------------------------------------------------------------

# Do not allow this chatbot setup to run using
# the MySQL root account.

if DB_USER.lower() == "root":

    raise ValueError(
        "The chatbot must not connect using "
        "the MySQL root account.\n"
        "Use the read-only 'jio_chatbot' user."
    )


print(
    "\nDATABASE CONFIGURATION"
)

print(
    "-" * 50
)

print(
    f"Host     : {DB_HOST}"
)

print(
    f"Port     : {DB_PORT}"
)

print(
    f"Database : {DB_NAME}"
)

print(
    f"User     : {DB_USER}"
)

print(
    "Password : [HIDDEN]"
)


# ------------------------------------------------------------
# 6. CREATE SQLALCHEMY URL
# ------------------------------------------------------------

database_url = URL.create(

    drivername=
        "mysql+pymysql",

    username=
        DB_USER,

    password=
        DB_PASSWORD,

    host=
        DB_HOST,

    port=
        int(DB_PORT),

    database=
        DB_NAME,
)


# ------------------------------------------------------------
# 7. CREATE DATABASE ENGINE
# ------------------------------------------------------------

engine = create_engine(

    database_url,

    pool_pre_ping=True,

    pool_recycle=1800,

    future=True,
)


# ------------------------------------------------------------
# 8. TEST DATABASE CONNECTION
# ------------------------------------------------------------

print(
    "\nTesting read-only MySQL connection..."
)


try:

    with engine.connect() as connection:

        database_name = (
            connection
            .execute(
                text(
                    "SELECT DATABASE()"
                )
            )
            .scalar()
        )


        current_user = (
            connection
            .execute(
                text(
                    "SELECT CURRENT_USER()"
                )
            )
            .scalar()
        )


    print(
        "[OK] MySQL connection successful."
    )

    print(
        f"Connected database : "
        f"{database_name}"
    )

    print(
        f"Authenticated user : "
        f"{current_user}"
    )


except Exception as error:

    print(
        "\n[ERROR] MySQL connection failed."
    )

    print(
        f"{type(error).__name__}: "
        f"{error}"
    )

    raise


# ------------------------------------------------------------
# 9. DATABASE INSPECTOR
# ------------------------------------------------------------

inspector = inspect(
    engine
)


database_tables = set(
    inspector.get_table_names()
)


print(
    f"\nDatabase tables discovered : "
    f"{len(database_tables)}"
)


for table in sorted(
    database_tables
):

    print(
        f"- {table}"
    )


# ------------------------------------------------------------
# 10. APPROVED TABLE CHECK
# ------------------------------------------------------------

approved_tables_present = sorted(
    database_tables
    &
    ALLOWED_TABLES
)


missing_expected_tables = sorted(
    ALLOWED_TABLES
    -
    database_tables
)


unapproved_tables = sorted(
    database_tables
    -
    ALLOWED_TABLES
)


print(
    "\nCHATBOT TABLE ACCESS"
)

print(
    "-" * 50
)


for table in approved_tables_present:

    print(
        f"[ALLOWED] {table}"
    )


if missing_expected_tables:

    print(
        "\nExpected tables not found:"
    )

    for table in missing_expected_tables:

        print(
            f"[MISSING] {table}"
        )


if unapproved_tables:

    print(
        "\nTables excluded from chatbot schema:"
    )

    for table in unapproved_tables:

        print(
            f"[BLOCKED] {table}"
        )


if len(
    approved_tables_present
) != 6:

    print(
        "\n[WARNING] Expected 6 approved "
        "Jio project tables."
    )

else:

    print(
        "\n[OK] All 6 approved project "
        "tables are available."
    )


# ------------------------------------------------------------
# 11. BUILD PRIVACY-SAFE SCHEMA
# ------------------------------------------------------------

safe_schema = {}


print(
    "\nPRIVACY-SAFE SCHEMA DISCOVERY"
)

print(
    "-" * 50
)


for table in approved_tables_present:

    columns = (
        inspector
        .get_columns(
            table
        )
    )


    safe_columns = []

    blocked_columns = []


    for column in columns:

        column_name = (
            column[
                "name"
            ]
        )


        column_type = str(
            column[
                "type"
            ]
        )


        if is_sensitive_column(
            column_name
        ):

            blocked_columns.append(
                column_name
            )

            continue


        safe_columns.append(
            {
                "name":
                    column_name,

                "type":
                    column_type,

                "nullable":
                    bool(
                        column[
                            "nullable"
                        ]
                    ),
            }
        )


    safe_schema[
        table
    ] = {
        "safe_columns":
            safe_columns,

        "blocked_columns":
            blocked_columns,
    }


    print(
        f"\n{table}"
    )

    print(
        f"  Safe columns    : "
        f"{len(safe_columns)}"
    )

    print(
        f"  Blocked columns : "
        f"{len(blocked_columns)}"
    )


    if blocked_columns:

        for column in blocked_columns:

            print(
                f"    [BLOCKED] "
                f"{column}"
            )


# ------------------------------------------------------------
# 12. SAVE SAFE SCHEMA JSON
# ------------------------------------------------------------

SAFE_SCHEMA_FILE = (
    MODEL_DATA_DIR
    / "chatbot_safe_schema.json"
)


safe_schema_metadata = {

    "database":
        DB_NAME,

    "database_user":
        DB_USER,

    "allowed_tables":
        approved_tables_present,

    "schema":
        safe_schema,

    "privacy_policy":
        (
            "Only privacy-safe business "
            "analysis columns are exposed "
            "to the chatbot."
        ),
}


with open(
    SAFE_SCHEMA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        safe_schema_metadata,
        file,
        indent=4
    )


# ------------------------------------------------------------
# 13. ROW COUNTS THROUGH SQL SAFETY LAYER
# ------------------------------------------------------------

row_counts = {}


print(
    "\nDATABASE ROW COUNTS"
)

print(
    "-" * 50
)


with engine.connect() as connection:

    for table in approved_tables_present:

        sql = (
            f"SELECT "
            f"COUNT(*) AS row_count "
            f"FROM `{table}`"
        )


        allowed, reason = (
            validate_sql(
                sql
            )
        )


        if not allowed:

            raise RuntimeError(
                f"Safety layer rejected "
                f"internal query:\n"
                f"{reason}"
            )


        result = (
            connection
            .execute(
                text(
                    sql
                )
            )
            .scalar()
        )


        row_counts[
            table
        ] = int(
            result
        )


        print(
            f"{table:<25} "
            f"{int(result):>10,}"
        )


# ------------------------------------------------------------
# 14. TEST SAFE ANALYTICAL QUERY
# ------------------------------------------------------------

sample_query = """
SELECT
    circle,
    COUNT(*) AS subscriber_count,
    ROUND(
        AVG(arpu_last_month_inr),
        2
    ) AS avg_arpu
FROM subscribers
GROUP BY circle
ORDER BY subscriber_count DESC
LIMIT 5
"""


allowed, reason = validate_sql(
    sample_query
)


if not allowed:

    raise RuntimeError(
        "Sample query was rejected:\n"
        + reason
    )


safe_query = enforce_row_limit(
    sample_query,
    max_rows=200,
)


print(
    "\n[OK] Sample query passed "
    "Step 29 SQL validation."
)


# ------------------------------------------------------------
# 15. EXECUTE SAFE QUERY
# ------------------------------------------------------------

with engine.connect() as connection:

    result = connection.execute(
        text(
            safe_query
        )
    )


    rows = (
        result.fetchall()
    )


    column_names = (
        result.keys()
    )


sample_df = pd.DataFrame(
    rows,
    columns=column_names,
)


# ------------------------------------------------------------
# 16. SANITIZE QUERY RESULTS
# ------------------------------------------------------------

sample_df = sanitize_results(
    sample_df
)


print(
    "\nTOP 5 CIRCLES BY SUBSCRIBER COUNT"
)

print(
    "-" * 70
)


print(
    sample_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 17. CREATE HUMAN-READABLE SAFE SCHEMA REPORT
# ------------------------------------------------------------

report = []


report.append(
    "=" * 100
)

report.append(
    "JIO CHATBOT - PRIVACY-SAFE DATABASE SCHEMA"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    f"Database: {DB_NAME}"
)

report.append(
    f"Database user: {DB_USER}"
)

report.append("")

report.append(
    "The chatbot is restricted to approved "
    "Jio analytical tables."
)

report.append(
    "Sensitive or confidential columns are "
    "excluded from the chatbot-visible schema."
)

report.append("")


for table in approved_tables_present:

    report.append(
        f"TABLE: {table}"
    )

    report.append(
        "-" * 100
    )


    for column in (
        safe_schema[
            table
        ][
            "safe_columns"
        ]
    ):

        report.append(
            f"{column['name']:<35} "
            f"{column['type']}"
        )


    blocked = (
        safe_schema[
            table
        ][
            "blocked_columns"
        ]
    )


    if blocked:

        report.append("")

        report.append(
            "PRIVACY-BLOCKED COLUMNS:"
        )


        for column in blocked:

            report.append(
                f"[BLOCKED] {column}"
            )


    report.append("")


report.append(
    "ROW COUNTS"
)

report.append(
    "-" * 100
)


for table, count in row_counts.items():

    report.append(
        f"{table:<30} "
        f"{count:>10,}"
    )


SAFE_SCHEMA_REPORT = (
    REPORT_DIR
    / "30_chatbot_safe_database_schema.txt"
)


SAFE_SCHEMA_REPORT.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 18. FINAL SUMMARY
# ------------------------------------------------------------

print(
    "\nGenerated files:"
)


print(
    f"- "
    f"{SAFE_SCHEMA_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{SAFE_SCHEMA_REPORT.relative_to(PROJECT_ROOT)}"
)


print(
    "\nSECURITY STATUS"
)

print(
    "-" * 50
)

print(
    "[OK] Read-only chatbot database user"
)

print(
    "[OK] Approved-table allowlist"
)

print(
    "[OK] Sensitive-column filtering"
)

print(
    "[OK] SQL safety validation"
)

print(
    "[OK] Result sanitization"
)

print(
    "[OK] Password hidden from output"
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 30 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nSQLAlchemy -> MySQL connection "
    "is working securely."
)

print(
    "No LLM has been connected yet."
)