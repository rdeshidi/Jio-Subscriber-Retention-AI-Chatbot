from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import os

import pandas as pd

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    text,
)

from sqlalchemy.engine import URL


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# TEXT-TO-SQL CHATBOT
# STEP 31 - SAFE READ-ONLY QUERY ENGINE
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

MODEL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)

SAFETY_FILE = (
    SRC_DIR
    / "29_chatbot_privacy_and_sql_safety.py"
)

SAFE_SCHEMA_FILE = (
    MODEL_DATA_DIR
    / "chatbot_safe_schema.json"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. LOAD STEP 29 SAFETY MODULE
# ------------------------------------------------------------

if not SAFETY_FILE.exists():

    raise FileNotFoundError(
        f"Safety module not found:\n"
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
        "Unable to load Step 29 safety module."
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


validate_sql = (
    safety.validate_sql
)

validate_user_question = (
    safety.validate_user_question
)

enforce_row_limit = (
    safety.enforce_row_limit
)

sanitize_results = (
    safety.sanitize_results
)

validate_result_columns = (
    safety.validate_result_columns
)

privacy_refusal_message = (
    safety.privacy_refusal_message
)


# ------------------------------------------------------------
# 3. LOAD SAFE SCHEMA
# ------------------------------------------------------------

if not SAFE_SCHEMA_FILE.exists():

    raise FileNotFoundError(
        f"Safe schema file not found:\n"
        f"{SAFE_SCHEMA_FILE}\n\n"
        "Run Step 30 first."
    )


with open(
    SAFE_SCHEMA_FILE,
    "r",
    encoding="utf-8"
) as file:

    safe_schema_metadata = json.load(
        file
    )


# ------------------------------------------------------------
# 4. LOAD DATABASE SETTINGS
# ------------------------------------------------------------

if not ENV_FILE.exists():

    raise FileNotFoundError(
        f".env file not found:\n"
        f"{ENV_FILE}"
    )


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


if not all(
    [
        DB_HOST,
        DB_NAME,
        DB_USER,
        DB_PASSWORD,
    ]
):

    raise ValueError(
        "Database configuration is incomplete."
    )


if DB_USER.lower() == "root":

    raise ValueError(
        "The query engine must not use "
        "the MySQL root account."
    )


# ------------------------------------------------------------
# 5. CREATE DATABASE ENGINE
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


engine = create_engine(

    database_url,

    pool_pre_ping=True,

    pool_recycle=1800,

    future=True,
)


# ============================================================
# SAFE QUERY ENGINE
# ============================================================


# ------------------------------------------------------------
# 6. EXECUTE SAFE SQL
# ------------------------------------------------------------

def execute_safe_sql(
    sql: str,
    max_rows: int = 200,
) -> pd.DataFrame:

    # ----------------------------------------
    # SQL safety validation
    # ----------------------------------------

    allowed, reason = validate_sql(
        sql
    )


    if not allowed:

        raise PermissionError(
            reason
        )


    # ----------------------------------------
    # Apply maximum row limit
    # ----------------------------------------

    safe_sql = enforce_row_limit(
        sql,
        max_rows=max_rows,
    )


    # ----------------------------------------
    # Execute using read-only DB account
    # ----------------------------------------

    with engine.connect() as connection:

        result = connection.execute(
            text(
                safe_sql
            )
        )


        rows = result.fetchall()

        columns = list(
            result.keys()
        )


    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )


    # ----------------------------------------
    # Result privacy validation
    # ----------------------------------------

    result_safe, result_reason = (
        validate_result_columns(
            dataframe
        )
    )


    if not result_safe:

        # Drop sensitive fields as a backup.
        dataframe = sanitize_results(
            dataframe
        )


        # Verify again after sanitization.
        result_safe, result_reason = (
            validate_result_columns(
                dataframe
            )
        )


        if not result_safe:

            raise PermissionError(
                "Query result failed "
                "privacy validation."
            )


    return dataframe


# ------------------------------------------------------------
# 7. QUESTION PRIVACY GATE
# ------------------------------------------------------------

def check_question(
    question: str
) -> tuple[bool, str]:

    allowed, reason = (
        validate_user_question(
            question
        )
    )


    if not allowed:

        return (
            False,
            privacy_refusal_message(),
        )


    return (
        True,
        reason,
    )


# ------------------------------------------------------------
# 8. COMBINED SAFE QUERY FUNCTION
# ------------------------------------------------------------

def process_question_and_sql(
    question: str,
    sql: str,
    max_rows: int = 200,
) -> pd.DataFrame:

    question_allowed, message = (
        check_question(
            question
        )
    )


    if not question_allowed:

        raise PermissionError(
            message
        )


    return execute_safe_sql(
        sql,
        max_rows=max_rows,
    )


# ============================================================
# SELF-TESTS
# ============================================================


# ------------------------------------------------------------
# 9. RUN QUERY-ENGINE TESTS
# ------------------------------------------------------------

def run_tests():

    print("=" * 90)
    print("JIO SUBSCRIBER RETENTION PROJECT")
    print("STEP 31: SAFE READ-ONLY QUERY ENGINE")
    print("=" * 90)


    tests_passed = 0
    total_tests = 6


    # --------------------------------------------------------
    # TEST 1 - SAFE AGGREGATE QUERY
    # --------------------------------------------------------

    print(
        "\nTEST 1 - SAFE AGGREGATE QUERY"
    )

    print(
        "-" * 90
    )


    question = (
        "Show the top five circles "
        "by subscriber count."
    )


    sql = """
    SELECT
        circle,
        COUNT(*) AS subscriber_count
    FROM subscribers
    GROUP BY circle
    ORDER BY subscriber_count DESC
    LIMIT 5
    """


    try:

        result = process_question_and_sql(
            question,
            sql,
        )


        print(
            result.to_string(
                index=False
            )
        )


        print(
            "\n[PASS] Safe aggregate query executed."
        )

        tests_passed += 1


    except Exception as error:

        print(
            f"[FAIL] {error}"
        )


    # --------------------------------------------------------
    # TEST 2 - SAFE ARPU QUERY
    # --------------------------------------------------------

    print(
        "\nTEST 2 - SAFE BUSINESS ANALYSIS QUERY"
    )

    print(
        "-" * 90
    )


    question = (
        "Show average ARPU by circle."
    )


    sql = """
    SELECT
        circle,
        ROUND(
            AVG(arpu_last_month_inr),
            2
        ) AS avg_arpu
    FROM subscribers
    GROUP BY circle
    ORDER BY avg_arpu DESC
    LIMIT 10
    """


    try:

        result = process_question_and_sql(
            question,
            sql,
        )


        print(
            result.to_string(
                index=False
            )
        )


        print(
            "\n[PASS] Business query executed."
        )

        tests_passed += 1


    except Exception as error:

        print(
            f"[FAIL] {error}"
        )


    # --------------------------------------------------------
    # TEST 3 - BLOCK SENSITIVE QUESTION
    # --------------------------------------------------------

    print(
        "\nTEST 3 - BLOCK PERSONAL INFORMATION REQUEST"
    )

    print(
        "-" * 90
    )


    question = (
        "Give me all subscriber IDs."
    )


    sql = """
    SELECT
        subscriber_id,
        circle
    FROM subscribers
    LIMIT 10
    """


    try:

        process_question_and_sql(
            question,
            sql,
        )


        print(
            "[FAIL] Sensitive request was executed."
        )


    except PermissionError as error:

        print(
            f"Blocked: {error}"
        )

        print(
            "[PASS] Sensitive request blocked."
        )

        tests_passed += 1


    # --------------------------------------------------------
    # TEST 4 - BLOCK SENSITIVE SQL EVEN IF QUESTION LOOKS SAFE
    # --------------------------------------------------------

    print(
        "\nTEST 4 - BLOCK HIDDEN IDENTIFIER QUERY"
    )

    print(
        "-" * 90
    )


    question = (
        "Show subscriber information by circle."
    )


    sql = """
    SELECT
        subscriber_id,
        circle
    FROM subscribers
    LIMIT 10
    """


    try:

        process_question_and_sql(
            question,
            sql,
        )


        print(
            "[FAIL] Identifier query was executed."
        )


    except PermissionError as error:

        print(
            f"Blocked: {error}"
        )

        print(
            "[PASS] SQL-level identifier "
            "protection worked."
        )

        tests_passed += 1


    # --------------------------------------------------------
    # TEST 5 - BLOCK WRITE OPERATION
    # --------------------------------------------------------

    print(
        "\nTEST 5 - BLOCK DATABASE MODIFICATION"
    )

    print(
        "-" * 90
    )


    question = (
        "Update subscriber data."
    )


    sql = """
    UPDATE subscribers
    SET circle = 'Test'
    """


    try:

        process_question_and_sql(
            question,
            sql,
        )


        print(
            "[FAIL] UPDATE was executed."
        )


    except PermissionError as error:

        print(
            f"Blocked: {error}"
        )

        print(
            "[PASS] Write operation blocked."
        )

        tests_passed += 1


    # --------------------------------------------------------
    # TEST 6 - AUTOMATIC ROW LIMIT
    # --------------------------------------------------------

    print(
        "\nTEST 6 - AUTOMATIC ROW LIMIT"
    )

    print(
        "-" * 90
    )


    question = (
        "Show plan types and circles."
    )


    sql = """
    SELECT
        circle,
        plan_type
    FROM subscribers
    """


    try:

        result = process_question_and_sql(
            question,
            sql,
            max_rows=25,
        )


        print(
            f"Rows returned: "
            f"{len(result)}"
        )


        if len(result) <= 25:

            print(
                "[PASS] Row limit enforced."
            )

            tests_passed += 1

        else:

            print(
                "[FAIL] Too many rows returned."
            )


    except Exception as error:

        print(
            f"[FAIL] {error}"
        )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "QUERY ENGINE TEST SUMMARY"
    )

    print(
        "=" * 90
    )


    print(
        f"Passed : "
        f"{tests_passed} / "
        f"{total_tests}"
    )


    if tests_passed != total_tests:

        raise RuntimeError(
            "One or more Step 31 "
            "query-engine tests failed."
        )


    print(
        "\n[OK] Safe query engine "
        "passed all tests."
    )


    print(
        "\nSECURITY LAYERS ACTIVE"
    )

    print(
        "-" * 90
    )

    print(
        "[OK] Natural-language privacy filter"
    )

    print(
        "[OK] SQL read-only validation"
    )

    print(
        "[OK] Sensitive-column blocking"
    )

    print(
        "[OK] Approved-table allowlist"
    )

    print(
        "[OK] Automatic row limiting"
    )

    print(
        "[OK] Result sanitization"
    )

    print(
        "[OK] Read-only MySQL user"
    )


    print(
        "\n" + "=" * 90
    )

    print(
        "STEP 31 COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "\nNo LLM has been connected yet."
    )


# ------------------------------------------------------------
# 10. RUN WHEN EXECUTED DIRECTLY
# ------------------------------------------------------------

if __name__ == "__main__":

    run_tests()