from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# TEXT-TO-SQL CHATBOT
# STEP 29 - PRIVACY & SQL SAFETY LAYER
# ============================================================


# ------------------------------------------------------------
# 1. APPROVED DATABASE TABLES
# ------------------------------------------------------------

# The employee chatbot is restricted to these
# project-analysis tables only.

ALLOWED_TABLES = {
    "subscribers",
    "service_requests",
    "network_sites",
    "circle_monthly_kpi",
    "circle_targets",
    "offer_catalogue",
}


# ------------------------------------------------------------
# 2. BLOCKED DATABASES / SYSTEM SCHEMAS
# ------------------------------------------------------------

BLOCKED_SCHEMAS = {
    "mysql",
    "sys",
    "information_schema",
    "performance_schema",
}


# ------------------------------------------------------------
# 3. SENSITIVE / CONFIDENTIAL COLUMNS
# ------------------------------------------------------------

# These are always blocked from chatbot output.
#
# subscriber_id is included even though the current
# project dataset is synthetic. We still treat it as
# an individual identifier.

BLOCKED_COLUMNS = {
    "subscriber_id",

    # Employee identifiers
    "employee_id",
    "emp_id",
    "staff_id",
    "worker_id",

    # Contact information
    "phone",
    "phone_number",
    "mobile",
    "mobile_number",
    "contact_number",
    "email",
    "email_address",

    # Address / location of individuals
    "address",
    "home_address",
    "residential_address",
    "postal_address",

    # Government / financial identifiers
    "aadhaar",
    "aadhaar_number",
    "pan",
    "pan_number",
    "passport_number",
    "bank_account",
    "bank_account_number",

    # Personal demographic identifiers
    "date_of_birth",
    "dob",

    # Authentication / secrets
    "password",
    "password_hash",
    "pin",
    "otp",
    "secret",
    "token",
    "api_key",

    # HR / confidential employment information
    "salary",
    "salary_inr",
    "compensation",
    "medical_information",
    "medical_record",
    "disciplinary_record",
}


# ------------------------------------------------------------
# 4. SENSITIVE COLUMN KEYWORDS
# ------------------------------------------------------------

# Used as a second layer in case future datasets use
# slightly different column names.

SENSITIVE_COLUMN_PATTERNS = [
    r"(^|_)employee_?id($|_)",
    r"(^|_)emp_?id($|_)",
    r"(^|_)subscriber_?id($|_)",

    r"(^|_)phone($|_)",
    r"(^|_)mobile($|_)",
    r"(^|_)contact_?number($|_)",

    r"(^|_)email($|_)",

    r"(^|_)address($|_)",

    r"(^|_)aadhaar($|_)",
    r"(^|_)pan_?number($|_)",
    r"(^|_)passport_?number($|_)",

    r"(^|_)bank_?account($|_)",

    r"(^|_)password($|_)",
    r"(^|_)secret($|_)",
    r"(^|_)token($|_)",
    r"(^|_)api_?key($|_)",

    r"(^|_)salary($|_)",
    r"(^|_)compensation($|_)",
]


# ------------------------------------------------------------
# 5. PROHIBITED SQL OPERATIONS
# ------------------------------------------------------------

# The chatbot is analytical and READ ONLY.

PROHIBITED_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "RENAME",

    "GRANT",
    "REVOKE",

    "CALL",
    "EXECUTE",
    "EXEC",

    "LOAD",
    "OUTFILE",
    "DUMPFILE",

    "SET",
    "USE",

    "LOCK",
    "UNLOCK",
}


# ------------------------------------------------------------
# 6. PRIVACY-SENSITIVE USER REQUEST PATTERNS
# ------------------------------------------------------------

SENSITIVE_REQUEST_PATTERNS = [

    # --------------------------------------------------------
    # Contact information
    # --------------------------------------------------------

    r"\b(phone|phones|phone number|phone numbers)\b",

    r"\b(mobile|mobiles|mobile number|mobile numbers)\b",

    r"\b(contact number|contact numbers|contact details)\b",

    r"\b(email|emails|email address|email addresses)\b",


    # --------------------------------------------------------
    # Individual identifiers
    # --------------------------------------------------------

    r"\b(employee ids?|employee identifiers?)\b",

    r"\b(emp ids?|emp identifiers?)\b",

    r"\b(subscriber ids?|subscriber identifiers?)\b",

    r"\b(staff ids?|staff identifiers?)\b",

    r"\b(worker ids?|worker identifiers?)\b",


    # --------------------------------------------------------
    # Personal addresses
    # --------------------------------------------------------

    r"\b(home address|home addresses)\b",

    r"\b(residential address|residential addresses)\b",

    r"\b(personal address|personal addresses)\b",


    # --------------------------------------------------------
    # Government / financial identifiers
    # --------------------------------------------------------

    r"\b(aadhaar|aadhaar number|aadhaar numbers)\b",

    r"\b(pan number|pan numbers)\b",

    r"\b(passport number|passport numbers)\b",

    r"\b(bank account|bank accounts|bank account number|bank account numbers)\b",


    # --------------------------------------------------------
    # Authentication / credentials
    # --------------------------------------------------------

    r"\b(password|passwords)\b",

    r"\b(pin|pins)\b",

    r"\b(otp|otps)\b",

    r"\b(api key|api keys)\b",

    r"\b(secret token|secret tokens)\b",


    # --------------------------------------------------------
# HR-sensitive / confidential employment information
# --------------------------------------------------------

r"\b(salary|salaries)\b",

r"\b(compensation|compensation details)\b",

r"\b(payroll|payroll information|payroll details)\b",

r"\b(wage|wages)\b",

r"\b(bonus|bonuses)\b",

r"\b(medical record|medical records)\b",

r"\b(disciplinary record|disciplinary records)\b",

r"\b(performance review|performance reviews)\b",

r"\b(employee evaluation|employee evaluations)\b",

    # --------------------------------------------------------
    # Generic personal / confidential information
    # --------------------------------------------------------

    r"\b(personal details|personal information)\b",

    r"\b(private information|confidential information)\b",

    r"\b(personally identifiable information|pii)\b",
]


# ------------------------------------------------------------
# 7. GENERAL SETTINGS
# ------------------------------------------------------------

DEFAULT_MAX_ROWS = 200


# ============================================================
# PRIVACY FUNCTIONS
# ============================================================


# ------------------------------------------------------------
# 8. NORMALIZE COLUMN NAME
# ------------------------------------------------------------

def normalize_name(name: str) -> str:

    cleaned = str(name).strip().lower()

    cleaned = re.sub(
        r"[^a-z0-9]+",
        "_",
        cleaned,
    )

    cleaned = cleaned.strip("_")

    return cleaned


# ------------------------------------------------------------
# 9. CHECK WHETHER COLUMN IS SENSITIVE
# ------------------------------------------------------------

def is_sensitive_column(
    column_name: str
) -> bool:

    normalized = normalize_name(
        column_name
    )


    if normalized in BLOCKED_COLUMNS:

        return True


    for pattern in SENSITIVE_COLUMN_PATTERNS:

        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        ):

            return True


    return False


# ------------------------------------------------------------
# 10. FIND SENSITIVE COLUMNS
# ------------------------------------------------------------

def find_sensitive_columns(
    columns: Iterable[str]
) -> list[str]:

    return [
        str(column)
        for column in columns
        if is_sensitive_column(
            str(column)
        )
    ]


# ------------------------------------------------------------
# 11. VALIDATE USER QUESTION
# ------------------------------------------------------------

def validate_user_question(
    question: str
) -> tuple[bool, str]:

    if not question:

        return (
            False,
            "The question is empty."
        )


    question_lower = (
        question
        .strip()
        .lower()
    )


    for pattern in SENSITIVE_REQUEST_PATTERNS:

        if re.search(
            pattern,
            question_lower,
            flags=re.IGNORECASE,
        ):

            return (
                False,
                (
                    "This request asks for personal, "
                    "identifying, confidential, or "
                    "sensitive information. "
                    "The chatbot may provide only "
                    "privacy-safe analytical or "
                    "aggregated information."
                ),
            )


    return (
        True,
        "Question passed privacy validation."
    )


# ============================================================
# SQL SAFETY FUNCTIONS
# ============================================================


# ------------------------------------------------------------
# 12. REMOVE STRING LITERALS FOR SAFETY SCANNING
# ------------------------------------------------------------

def strip_string_literals(
    sql: str
) -> str:

    # We remove quoted content before keyword scanning
    # so text values such as 'DROP' do not incorrectly
    # trigger a prohibited-operation rule.

    sql = re.sub(
        r"'(?:''|[^'])*'",
        "''",
        sql,
    )

    sql = re.sub(
        r'"(?:""|[^"])*"',
        '""',
        sql,
    )

    return sql


# ------------------------------------------------------------
# 13. REMOVE SQL COMMENTS
# ------------------------------------------------------------

def remove_sql_comments(
    sql: str
) -> str:

    sql = re.sub(
        r"--.*?$",
        "",
        sql,
        flags=re.MULTILINE,
    )

    sql = re.sub(
        r"/\*.*?\*/",
        "",
        sql,
        flags=re.DOTALL,
    )

    return sql


# ------------------------------------------------------------
# 14. VALIDATE SQL
# ------------------------------------------------------------

def validate_sql(
    sql: str
) -> tuple[bool, str]:

    if not sql or not sql.strip():

        return (
            False,
            "SQL query is empty."
        )


    original_sql = sql.strip()


    # ----------------------------------------
    # Remove trailing semicolon
    # ----------------------------------------

    clean_sql = original_sql.rstrip()

    if clean_sql.endswith(";"):

        clean_sql = clean_sql[:-1].rstrip()


    # ----------------------------------------
    # Reject multiple statements
    # ----------------------------------------

    if ";" in clean_sql:

        return (
            False,
            "Multiple SQL statements are not allowed."
        )


    # ----------------------------------------
    # Reject SQL comments
    # ----------------------------------------

    if (
        "--" in clean_sql
        or "/*" in clean_sql
        or "*/" in clean_sql
    ):

        return (
            False,
            "SQL comments are not allowed."
        )


    scan_sql = remove_sql_comments(
        clean_sql
    )

    scan_sql = strip_string_literals(
        scan_sql
    )

    scan_upper = scan_sql.upper()


    # ----------------------------------------
    # Only SELECT / WITH queries allowed
    # ----------------------------------------

    first_keyword_match = re.match(
        r"^\s*([A-Za-z]+)",
        scan_sql,
    )


    if not first_keyword_match:

        return (
            False,
            "Unable to determine SQL statement type."
        )


    first_keyword = (
        first_keyword_match
        .group(1)
        .upper()
    )


    if first_keyword not in {
        "SELECT",
        "WITH",
    }:

        return (
            False,
            (
                "Only read-only SELECT queries "
                "are allowed."
            ),
        )


    # ----------------------------------------
    # Block dangerous SQL operations
    # ----------------------------------------

    for keyword in PROHIBITED_SQL_KEYWORDS:

        if re.search(
            rf"\b{re.escape(keyword)}\b",
            scan_upper,
        ):

            return (
                False,
                (
                    f"Prohibited SQL operation "
                    f"detected: {keyword}"
                ),
            )


    # ----------------------------------------
    # Block system schemas
    # ----------------------------------------

    for schema in BLOCKED_SCHEMAS:

        if re.search(
            rf"\b{re.escape(schema)}\s*\.",
            scan_sql,
            flags=re.IGNORECASE,
        ):

            return (
                False,
                (
                    f"Access to system schema "
                    f"'{schema}' is not allowed."
                ),
            )


    # ----------------------------------------
    # Block SELECT *
    # ----------------------------------------

    # We allow COUNT(*), but do not allow returning
    # every field because that could expose identifiers.

    select_star_pattern = (
        r"\bSELECT\s+"
        r"(?:DISTINCT\s+)?"
        r"\*"
    )


    if re.search(
        select_star_pattern,
        scan_sql,
        flags=re.IGNORECASE,
    ):

        return (
            False,
            (
                "SELECT * is not allowed. "
                "Queries must explicitly name "
                "privacy-safe columns."
            ),
        )


    # Block aliases such as s.*

    if re.search(
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*\*",
        scan_sql,
        flags=re.IGNORECASE,
    ):

        return (
            False,
            (
                "Selecting all columns from a table "
                "or alias is not allowed."
            ),
        )


    # ----------------------------------------
    # Block sensitive columns
    # ----------------------------------------

    for blocked_column in BLOCKED_COLUMNS:

        pattern = (
            rf"(?<![A-Za-z0-9_])"
            rf"`?{re.escape(blocked_column)}`?"
            rf"(?![A-Za-z0-9_])"
        )


        if re.search(
            pattern,
            scan_sql,
            flags=re.IGNORECASE,
        ):

            return (
                False,
                (
                    "Sensitive or confidential "
                    f"column detected: "
                    f"{blocked_column}"
                ),
            )


    # ----------------------------------------
    # Block unapproved tables
    # ----------------------------------------

    table_matches = re.findall(
        r"\b(?:FROM|JOIN)\s+"
        r"`?([A-Za-z_][A-Za-z0-9_]*)`?",
        scan_sql,
        flags=re.IGNORECASE,
    )


    for table_name in table_matches:

        normalized_table = (
            table_name.lower()
        )


        if normalized_table not in ALLOWED_TABLES:

            return (
                False,
                (
                    "Query references an "
                    "unapproved table: "
                    f"{table_name}"
                ),
            )


    return (
        True,
        "SQL passed read-only and privacy validation."
    )


# ------------------------------------------------------------
# 15. ENFORCE RESULT ROW LIMIT
# ------------------------------------------------------------

def enforce_row_limit(
    sql: str,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> str:

    clean_sql = sql.strip().rstrip(";").strip()


    if max_rows <= 0:

        raise ValueError(
            "max_rows must be greater than zero."
        )


    # If a LIMIT already exists, reduce it if needed.

    limit_match = re.search(
        r"\bLIMIT\s+(\d+)\s*$",
        clean_sql,
        flags=re.IGNORECASE,
    )


    if limit_match:

        requested_limit = int(
            limit_match.group(1)
        )


        if requested_limit <= max_rows:

            return (
                clean_sql
                + ";"
            )


        clean_sql = re.sub(
            r"\bLIMIT\s+\d+\s*$",
            f"LIMIT {max_rows}",
            clean_sql,
            flags=re.IGNORECASE,
        )


        return (
            clean_sql
            + ";"
        )


    # Aggregate-only queries often return a tiny
    # result, but adding LIMIT remains harmless.

    return (
        f"{clean_sql}\n"
        f"LIMIT {max_rows};"
    )


# ============================================================
# RESULT-SANITIZATION FUNCTIONS
# ============================================================


# ------------------------------------------------------------
# 16. SANITIZE DATAFRAME RESULTS
# ------------------------------------------------------------

def sanitize_results(
    dataframe: pd.DataFrame
) -> pd.DataFrame:

    safe_df = dataframe.copy()


    sensitive_columns = (
        find_sensitive_columns(
            safe_df.columns
        )
    )


    if sensitive_columns:

        safe_df = safe_df.drop(
            columns=sensitive_columns,
            errors="ignore",
        )


    return safe_df


# ------------------------------------------------------------
# 17. VERIFY RESULT PRIVACY
# ------------------------------------------------------------

def validate_result_columns(
    dataframe: pd.DataFrame
) -> tuple[bool, str]:

    sensitive_columns = (
        find_sensitive_columns(
            dataframe.columns
        )
    )


    if sensitive_columns:

        return (
            False,
            (
                "Sensitive result columns detected: "
                + ", ".join(
                    sensitive_columns
                )
            ),
        )


    return (
        True,
        "Result columns passed privacy validation."
    )


# ============================================================
# CHATBOT REFUSAL MESSAGE
# ============================================================


# ------------------------------------------------------------
# 18. STANDARD PRIVACY RESPONSE
# ------------------------------------------------------------

def privacy_refusal_message() -> str:

    return (
        "I can't provide personal, identifying, "
        "confidential, or sensitive information "
        "about employees, customers, or other "
        "individuals. I can help with aggregated "
        "and privacy-safe business analysis instead."
    )


# ============================================================
# SELF-TESTS
# ============================================================


# ------------------------------------------------------------
# 19. RUN SAFETY TESTS
# ------------------------------------------------------------

def run_self_tests():

    print("=" * 90)
    print("JIO SUBSCRIBER RETENTION PROJECT")
    print("STEP 29: CHATBOT PRIVACY & SQL SAFETY")
    print("=" * 90)


    # ----------------------------------------
    # User-question tests
    # ----------------------------------------

    question_tests = [

        (
            "Which circle has the highest churn rate?",
            True,
        ),

        (
            "Show average ARPU by circle.",
            True,
        ),

        (
            "Give me all subscriber IDs.",
            False,
        ),

        (
            "Show employee phone numbers.",
            False,
        ),

        (
            "Give me personal information about employees.",
            False,
        ),

        (
    "List all subscriber email addresses.",
    False,
),

(
    "Give me employee IDs and salaries.",
    False,
),

(
    "Show subscriber contact details.",
    False,
),

(
    "What is the average salary by department?",
    False,
),

(
    "Show churn rate by circle.",
    True,
),
    ]


    print(
        "\nUSER QUESTION PRIVACY TESTS"
    )

    print(
        "-" * 90
    )


    question_passed = 0


    for question, expected in question_tests:

        allowed, message = (
            validate_user_question(
                question
            )
        )


        test_ok = (
            allowed == expected
        )


        if test_ok:
            question_passed += 1


        status = (
            "PASS"
            if test_ok
            else "FAIL"
        )


        print(
            f"[{status}] "
            f"{question}"
        )

        print(
            f"       Allowed: {allowed}"
        )


    # ----------------------------------------
    # SQL tests
    # ----------------------------------------

    sql_tests = [

        (
            """
            SELECT
                circle,
                COUNT(*) AS subscribers
            FROM subscribers
            GROUP BY circle
            """,
            True,
        ),

        (
            """
            SELECT
                circle,
                AVG(arpu_last_month_inr) AS avg_arpu
            FROM subscribers
            GROUP BY circle
            """,
            True,
        ),

        (
            """
            SELECT *
            FROM subscribers
            """,
            False,
        ),

        (
            """
            SELECT
                subscriber_id,
                circle
            FROM subscribers
            """,
            False,
        ),

        (
            """
            DELETE
            FROM subscribers
            """,
            False,
        ),

        (
            """
            DROP TABLE subscribers
            """,
            False,
        ),

        (
            """
            SELECT
                user,
                host
            FROM mysql.user
            """,
            False,
        ),

        (
            """
            SELECT
                circle
            FROM secret_employee_table
            """,
            False,
        ),
    ]


    print(
        "\nSQL SAFETY TESTS"
    )

    print(
        "-" * 90
    )


    sql_passed = 0


    for sql, expected in sql_tests:

        allowed, message = (
            validate_sql(
                sql
            )
        )


        test_ok = (
            allowed == expected
        )


        if test_ok:
            sql_passed += 1


        status = (
            "PASS"
            if test_ok
            else "FAIL"
        )


        one_line_sql = (
            " ".join(
                sql.split()
            )
        )


        print(
            f"[{status}] "
            f"{one_line_sql}"
        )

        print(
            f"       Allowed: {allowed}"
        )

        print(
            f"       Reason : {message}"
        )


    # ----------------------------------------
    # Result sanitization test
    # ----------------------------------------

    print(
        "\nRESULT SANITIZATION TEST"
    )

    print(
        "-" * 90
    )


    sample_result = pd.DataFrame(
        {
            "subscriber_id": [
                "JIO10000001",
                "JIO10000002",
            ],

            "circle": [
                "Telangana",
                "Maharashtra & Goa",
            ],

            "arpu": [
                250.0,
                310.0,
            ],

            "email": [
                "person1@example.com",
                "person2@example.com",
            ],
        }
    )


    sanitized = sanitize_results(
        sample_result
    )


    sensitive_remaining = (
        find_sensitive_columns(
            sanitized.columns
        )
    )


    result_test_passed = (
        len(
            sensitive_remaining
        ) == 0
    )


    print(
        "Original columns : "
        f"{sample_result.columns.tolist()}"
    )

    print(
        "Sanitized columns: "
        f"{sanitized.columns.tolist()}"
    )


    if result_test_passed:

        print(
            "[PASS] Sensitive columns removed."
        )

    else:

        print(
            "[FAIL] Sensitive columns remain."
        )


    # ----------------------------------------
    # Row limit test
    # ----------------------------------------

    print(
        "\nROW LIMIT TEST"
    )

    print(
        "-" * 90
    )


    sample_sql = """
    SELECT
        circle,
        plan_type
    FROM subscribers
    """


    limited_sql = enforce_row_limit(
        sample_sql,
        max_rows=200,
    )


    print(
        limited_sql
    )


    row_limit_passed = (
        "LIMIT 200" in
        limited_sql.upper()
    )


    if row_limit_passed:

        print(
            "[PASS] Maximum row limit enforced."
        )

    else:

        print(
            "[FAIL] Row limit not enforced."
        )


    # ----------------------------------------
    # Final test summary
    # ----------------------------------------

    total_tests = (
        len(question_tests)
        +
        len(sql_tests)
        +
        2
    )


    passed_tests = (
        question_passed
        +
        sql_passed
        +
        int(
            result_test_passed
        )
        +
        int(
            row_limit_passed
        )
    )


    print(
        "\n" + "=" * 90
    )

    print(
        "SAFETY TEST SUMMARY"
    )

    print(
        "=" * 90
    )


    print(
        f"Passed : "
        f"{passed_tests} / {total_tests}"
    )


    if passed_tests == total_tests:

        print(
            "\n[OK] Privacy and SQL safety "
            "layer passed all tests."
        )

    else:

        raise RuntimeError(
            "One or more privacy / SQL "
            "safety tests failed."
        )


    print(
        "\n" + "=" * 90
    )

    print(
        "STEP 29 COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "\nNo database connection was used."
    )

    print(
        "No LLM was used."
    )

    print(
        "Privacy controls are ready for "
        "the next chatbot stage."
    )


# ------------------------------------------------------------
# 20. RUN WHEN EXECUTED DIRECTLY
# ------------------------------------------------------------

if __name__ == "__main__":

    run_self_tests()