"""
Step 32 - Local LLM SQL Agent
Jio Subscriber Retention & AI Chatbot

Flow:
Employee question
    -> privacy check
    -> local Qwen 2.5 3B
    -> SQL extraction
    -> Step 31 safety engine
    -> read-only MySQL
    -> sanitized result

No OpenAI API key.
No paid API calls.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP31_PATH = PROJECT_ROOT / "src" / "31_chatbot_readonly_query_engine.py"

SAFE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "chatbot_safe_schema.json"
)

OLLAMA_MODEL = "qwen2.5:3b"
MAX_ROWS = 200


# ============================================================
# LOAD STEP 31
# ============================================================

def load_step31():
    """Load the existing privacy-safe SQL engine."""

    if not STEP31_PATH.exists():
        raise FileNotFoundError(
            f"Step 31 not found:\n{STEP31_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "chatbot_readonly_query_engine",
        STEP31_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Step 31.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "check_question",
        "process_question_and_sql",
    ]

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise AttributeError(
            "Step 31 is missing: " + ", ".join(missing)
        )

    return module


# ============================================================
# OLLAMA EXECUTABLE
# ============================================================

def find_ollama() -> str:
    """
    Find Ollama either through PATH or its normal Windows
    installation location.
    """

    path_ollama = shutil.which("ollama")

    if path_ollama:
        return path_ollama

    local_app_data = os.environ.get("LOCALAPPDATA")

    if local_app_data:
        direct_path = (
            Path(local_app_data)
            / "Programs"
            / "Ollama"
            / "ollama.exe"
        )

        if direct_path.exists():
            return str(direct_path)

    raise FileNotFoundError(
        "Ollama executable could not be found."
    )


def check_ollama() -> str:
    """Verify Ollama is available."""

    ollama_exe = find_ollama()

    result = subprocess.run(
        [ollama_exe, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Ollama version check failed:\n"
            + result.stderr.strip()
        )

    print("[OK] Ollama detected:")
    print(result.stdout.strip())

    return ollama_exe


# ============================================================
# SAFE SCHEMA
# ============================================================

def load_safe_schema() -> dict[str, Any]:
    """Load Step 30 privacy-safe schema."""

    if not SAFE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Safe schema not found:\n{SAFE_SCHEMA_PATH}"
        )

    with open(
        SAFE_SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def build_schema_prompt(schema: dict[str, Any]) -> str:
    """
    Convert the actual Step 30 schema structure into
    a clean prompt for Qwen.
    """

    database = schema.get(
        "database",
        "jio_retention_db",
    )

    allowed_tables = schema.get(
        "allowed_tables",
        [],
    )

    schema_dict = schema.get(
        "schema",
        {},
    )

    lines: list[str] = []

    lines.append(
        f"DATABASE: {database}"
    )

    lines.append(
        "\nAPPROVED TABLES:"
    )

    for table in allowed_tables:
        lines.append(f"- {table}")

    lines.append(
        "\nAPPROVED TABLE SCHEMAS:"
    )

    for table in allowed_tables:

        table_info = schema_dict.get(
            table,
            {},
        )

        safe_columns = table_info.get(
            "safe_columns",
            [],
        )

        blocked_columns = table_info.get(
            "blocked_columns",
            [],
        )

        lines.append(f"\nTABLE: {table}")

        for column in safe_columns:

            if isinstance(column, dict):
                name = column.get("name")
                col_type = column.get("type", "")

                if name:
                    lines.append(
                        f"  {name} : {col_type}"
                    )

        if blocked_columns:
            lines.append(
                "  BLOCKED COLUMNS: "
                + ", ".join(blocked_columns)
            )

    # Important business semantics.
    lines.append(
        """
IMPORTANT BUSINESS DEFINITIONS:

1. For 30-day churn analysis, use subscribers.churn_flag_30d.
2. churn_flag_30d = 1 means churned and 0 means not churned.
3. Do not use churn_date, churn_reason,
   mnp_enquiry_flag, or churn_flag_90d for 30-day churn analysis.
4. For subscriber-level average ARPU, use
   subscribers.arpu_last_month_inr.
5. circle_monthly_kpi is a monthly aggregated circle KPI table.
6. circle_monthly_kpi.arpu_inr is the circle-level monthly ARPU KPI.
7. circle_monthly_kpi.monthly_churn_pct is the already-calculated
   monthly circle churn KPI.
8. circle_targets contains planning/target information.
9. Never invent tables.
10. Never invent columns.
11. Never use subscriber_id.
12. Never use SELECT *.
"""
    )

    return "\n".join(lines)


# ============================================================
# LOCAL LLM
# ============================================================

def ask_local_llm(
    ollama_exe: str,
    question: str,
    schema_text: str,
) -> str:
    """
    Ask Qwen for a single MySQL SELECT query.
    """

    prompt = f"""
You are the SQL-generation component of a telecom
business analytics chatbot.

Convert the employee's question into ONE MySQL SELECT query.

STRICT RULES:

- Output ONE SELECT query only.
- Do not output explanations.
- Do not output multiple queries.
- Never use INSERT.
- Never use UPDATE.
- Never use DELETE.
- Never use DROP.
- Never use ALTER.
- Never use CREATE.
- Never use TRUNCATE.
- Never use GRANT.
- Never use REVOKE.
- Never use CALL.
- Never use SET.
- Use ONLY approved tables.
- Use ONLY approved columns.
- Never use subscriber_id.
- Never use employee personal information.
- Never use employee identifiers.
- Never use salaries or compensation.
- Never use customer contact information.
- Never use SELECT *.
- Use LIMIT {MAX_ROWS} for row-producing queries.
- Do not invent tables.
- Do not invent columns.

{schema_text}

FEW-SHOT SQL EXAMPLES:

Example 1:
Question:
Which 5 circles have the highest 30-day churn rate?

SQL:
SELECT
    circle,
    COUNT(*) AS total_subscribers,
    SUM(churn_flag_30d) AS churned_subscribers,
    ROUND(AVG(churn_flag_30d) * 100, 2)
        AS churn_rate_percentage
FROM subscribers
GROUP BY circle
ORDER BY churn_rate_percentage DESC
LIMIT 5;

Example 2:
Question:
What is the average ARPU by circle?

SQL:
SELECT
    circle,
    ROUND(AVG(arpu_last_month_inr), 2)
        AS average_arpu_inr
FROM subscribers
GROUP BY circle
ORDER BY average_arpu_inr DESC
LIMIT {MAX_ROWS};

Now answer this employee question:

{question}

Return ONLY the SQL query inside one ```sql``` block.
""".strip()

    result = subprocess.run(
        [
            ollama_exe,
            "run",
            OLLAMA_MODEL,
            prompt,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Ollama failed:\n"
            + result.stderr.strip()
        )

    response = result.stdout.strip()

    if not response:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return response


# ============================================================
# SQL EXTRACTION
# ============================================================

def extract_sql(response: str) -> str:
    """Extract SQL from the Qwen response."""

    match = re.search(
        r"```sql\s*(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        sql = match.group(1).strip()

    else:
        match = re.search(
            r"```\s*(.*?)```",
            response,
            flags=re.DOTALL,
        )

        if match:
            sql = match.group(1).strip()
        else:
            sql = response.strip()

    # Remove accidental labels.
    sql = re.sub(
        r"^(SQL Query|SQL|Query)\s*:\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    ).strip()

    # If the model returned explanatory text before SELECT,
    # keep only the SQL beginning at SELECT.
    select_match = re.search(
        r"\bSELECT\b",
        sql,
        flags=re.IGNORECASE,
    )

    if select_match:
        sql = sql[select_match.start():].strip()

    # Remove trailing markdown fences.
    sql = sql.replace("```", "").strip()

    return sql


# ============================================================
# QUESTION PROCESSING
# ============================================================

def process_employee_question(
    step31,
    ollama_exe: str,
    question: str,
) -> bool:

    print("\n" + "=" * 78)
    print("EMPLOYEE QUESTION")
    print("=" * 78)
    print(question)

    # --------------------------------------------------------
    # 1. Natural-language privacy check
    # --------------------------------------------------------

    print("\n[1/4] Privacy check...")

    question_ok, privacy_message = (
        step31.check_question(question)
    )

    print(
        f"{question_ok}: {privacy_message}"
    )

    if not question_ok:
        print(
            "[BLOCKED] Question rejected before LLM execution."
        )
        return True

    # --------------------------------------------------------
    # 2. Load schema
    # --------------------------------------------------------

    print(
        "\n[2/4] Loading privacy-safe schema..."
    )

    schema = load_safe_schema()

    schema_text = build_schema_prompt(
        schema
    )

    print(
        "[OK] Privacy-safe schema loaded."
    )

    # --------------------------------------------------------
    # 3. Generate SQL
    # --------------------------------------------------------

    print(
        "\n[3/4] Asking Qwen to generate SQL..."
    )

    raw_response = ask_local_llm(
        ollama_exe,
        question,
        schema_text,
    )

    print(
        "\n--- RAW LLM RESPONSE ---"
    )
    print(raw_response)

    sql = extract_sql(
        raw_response
    )

    print(
        "\n--- EXTRACTED SQL ---"
    )
    print(sql)

    if not sql:
        print(
            "[FAIL] No SQL extracted."
        )
        return False

    # --------------------------------------------------------
    # 4. Step 31 security + database
    # --------------------------------------------------------

    print(
        "\n[4/4] Sending SQL through Step 31..."
    )

    try:

        result = step31.process_question_and_sql(
            question=question,
            sql=sql,
            max_rows=MAX_ROWS,
        )

        print(
            "\n--- STEP 31 RESULT ---"
        )

        if isinstance(result, dict):
            print(
                json.dumps(
                    result,
                    indent=2,
                    default=str,
                )
            )
        else:
            print(result)

        print(
            "\n[PASS] End-to-end request completed."
        )

        return True

    except Exception as exc:

        print(
            "\n[FAIL] Step 31 rejected or could not execute SQL."
        )
        print(
            type(exc).__name__
            + ": "
            + str(exc)
        )

        return False


# ============================================================
# TEST SUITE
# ============================================================

def run_tests(
    step31,
    ollama_exe: str,
) -> None:

    passed = 0
    total = 5

    print("\n" + "=" * 78)
    print("STEP 32 TEST SUITE")
    print("=" * 78)

    # --------------------------------------------------------
    # Test 1 - safe business question
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("TEST 1 - SAFE BUSINESS QUESTION")
    print("-" * 78)

    if process_employee_question(
        step31,
        ollama_exe,
        "Which 5 circles have the highest 30-day churn rate?",
    ):
        passed += 1

    # --------------------------------------------------------
    # Test 2 - safe aggregate
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("TEST 2 - SAFE AGGREGATE QUESTION")
    print("-" * 78)

    if process_employee_question(
        step31,
        ollama_exe,
        "What is the average ARPU by circle?",
    ):
        passed += 1

    # --------------------------------------------------------
    # Test 3 - customer identifier
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("TEST 3 - CUSTOMER PRIVACY")
    print("-" * 78)

    ok, message = step31.check_question(
        "Give me all subscriber IDs."
    )

    print(ok, message)

    if not ok:
        print(
            "[PASS] Subscriber identifiers blocked."
        )
        passed += 1
    else:
        print(
            "[FAIL] Subscriber identifier request allowed."
        )

    # --------------------------------------------------------
    # Test 4 - employee sensitive information
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("TEST 4 - EMPLOYEE PRIVACY")
    print("-" * 78)

    ok, message = step31.check_question(
        "What is the average salary by department?"
    )

    print(ok, message)

    if not ok:
        print(
            "[PASS] Employee sensitive information blocked."
        )
        passed += 1
    else:
        print(
            "[FAIL] Employee sensitive information allowed."
        )

    # --------------------------------------------------------
    # Test 5 - dangerous SQL
    # --------------------------------------------------------

    print("\n" + "-" * 78)
    print("TEST 5 - DATABASE MODIFICATION SAFETY")
    print("-" * 78)

    dangerous_sql = """
DELETE FROM subscribers
WHERE complaints_6m = 0;
""".strip()

    try:

        step31.process_question_and_sql(
            question="Delete subscribers with zero complaints.",
            sql=dangerous_sql,
            max_rows=MAX_ROWS,
        )

        print(
            "[FAIL] Dangerous SQL was not rejected."
        )

    except Exception as exc:

        print(
            type(exc).__name__
            + ": "
            + str(exc)
        )

        print(
            "[PASS] Database modification was blocked."
        )

        passed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 78)

    if passed == total:
        print(
            f"STEP 32 TEST RESULT: {passed}/{total} PASSED"
        )
        print(
            "[OK] Local LLM SQL agent is functioning."
        )
    else:
        print(
            f"STEP 32 TEST RESULT: {passed}/{total} PASSED"
        )
        print(
            "[REVIEW] Some Step 32 tests still need attention."
        )

    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("JIO CHATBOT - STEP 32 LOCAL LLM SQL AGENT")
    print("=" * 78)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Model: {OLLAMA_MODEL}"
    )

    if not SAFE_SCHEMA_PATH.exists():
        print(
            "\n[ERROR] Safe schema missing:"
        )
        print(
            SAFE_SCHEMA_PATH
        )
        sys.exit(1)

    try:

        ollama_exe = check_ollama()

        print(
            "\n[OK] Loading Step 31..."
        )

        step31 = load_step31()

        print(
            "[OK] Step 31 loaded."
        )

        run_tests(
            step31,
            ollama_exe,
        )

    except Exception as exc:

        print(
            "\n[ERROR]"
        )
        print(
            type(exc).__name__
            + ": "
            + str(exc)
        )

        sys.exit(1)


if __name__ == "__main__":
    main()