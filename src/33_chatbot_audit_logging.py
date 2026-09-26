"""
Step 33 - Chatbot Audit Logging
Jio Subscriber Retention & AI Chatbot

Logs:
- timestamp
- employee question
- generated SQL
- privacy status
- validation/execution status
- runtime
- row count
- output/error
- Ollama version
- model version

Privacy:
- blocked sensitive questions are logged as REDACTED
- no subscriber IDs or employee personal information are written
  to the audit log
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP31_PATH = (
    PROJECT_ROOT
    / "src"
    / "31_chatbot_readonly_query_engine.py"
)

STEP32_PATH = (
    PROJECT_ROOT
    / "src"
    / "32_local_llm_sql_agent.py"
)

SAFE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "chatbot_safe_schema.json"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

AUDIT_CSV = (
    AUDIT_DIR
    / "33_chatbot_audit_log.csv"
)

AUDIT_JSONL = (
    AUDIT_DIR
    / "33_chatbot_audit_log.jsonl"
)

OLLAMA_MODEL = "qwen2.5:3b"


# ============================================================
# LOAD MODULES
# ============================================================

def load_module(path: Path, module_name: str):
    """Load a Python module from a file path."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# OLLAMA
# ============================================================

def find_ollama() -> str:
    """Find Ollama through PATH or normal Windows location."""

    found = shutil.which("ollama")

    if found:
        return found

    local_app_data = os.environ.get(
        "LOCALAPPDATA"
    )

    if local_app_data:

        candidate = (
            Path(local_app_data)
            / "Programs"
            / "Ollama"
            / "ollama.exe"
        )

        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Ollama executable not found."
    )


def get_ollama_version(
    ollama_exe: str,
) -> str:

    result = subprocess.run(
        [
            ollama_exe,
            "--version",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip()


# ============================================================
# AUDIT HELPERS
# ============================================================

AUDIT_FIELDS = [
    "timestamp_utc",
    "question_hash",
    "question",
    "privacy_status",
    "sql_generated",
    "validation_status",
    "execution_status",
    "runtime_seconds",
    "row_count",
    "result_status",
    "error",
    "model",
    "ollama_version",
]


def ensure_audit_files() -> None:
    """Create audit files with headers if they do not exist."""

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not AUDIT_CSV.exists():

        with open(
            AUDIT_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=AUDIT_FIELDS,
            )

            writer.writeheader()


def question_hash(question: str) -> str:
    """Create a non-reversible audit identifier for the question."""

    return hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()[:16]


def write_audit_record(
    record: dict[str, Any],
) -> None:

    ensure_audit_files()

    # CSV
    with open(
        AUDIT_CSV,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=AUDIT_FIELDS,
        )

        writer.writerow(
            {
                field: record.get(
                    field,
                    "",
                )
                for field in AUDIT_FIELDS
            }
        )

    # JSONL
    with open(
        AUDIT_JSONL,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                default=str,
                ensure_ascii=False,
            )
            + "\n"
        )


def count_result_rows(result: Any) -> int:
    """Determine row count from common Step 31 result formats."""

    if result is None:
        return 0

    if hasattr(result, "shape"):
        try:
            return int(result.shape[0])
        except Exception:
            pass

    if isinstance(result, list):
        return len(result)

    if isinstance(result, dict):

        for key in [
            "rows",
            "data",
            "results",
        ]:

            value = result.get(key)

            if isinstance(value, list):
                return len(value)

    return 0


# ============================================================
# END-TO-END AUDITED REQUEST
# ============================================================

def run_audited_request(
    step31,
    step32,
    ollama_exe: str,
    question: str,
) -> dict[str, Any]:

    started = time.perf_counter()

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "question_hash": question_hash(
            question
        ),

        "question": question,

        "privacy_status": "",

        "sql_generated": "",

        "validation_status": "",

        "execution_status": "",

        "runtime_seconds": 0,

        "row_count": 0,

        "result_status": "",

        "error": "",

        "model": OLLAMA_MODEL,

        "ollama_version": get_ollama_version(
            ollama_exe
        ),
    }

    print("\n" + "=" * 78)
    print("AUDITED CHATBOT REQUEST")
    print("=" * 78)
    print(question)

    try:

        # ----------------------------------------------------
        # Privacy check
        # ----------------------------------------------------

        allowed, privacy_message = (
            step31.check_question(question)
        )

        record["privacy_status"] = (
            "PASSED"
            if allowed
            else "BLOCKED"
        )

        print(
            "\nPrivacy:",
            record["privacy_status"],
        )

        if not allowed:

            # Do not persist potentially sensitive
            # question text.
            record["question"] = (
                "[REDACTED: privacy blocked]"
            )

            record["validation_status"] = (
                "NOT_RUN"
            )

            record["execution_status"] = (
                "NOT_RUN"
            )

            record["result_status"] = (
                "BLOCKED"
            )

            record["error"] = privacy_message

            return record

        # ----------------------------------------------------
        # Load schema
        # ----------------------------------------------------

        schema = step32.load_safe_schema()

        schema_text = (
            step32.build_schema_prompt(
                schema
            )
        )

        # ----------------------------------------------------
        # Generate SQL
        # ----------------------------------------------------

        print(
            "\nGenerating SQL with Qwen..."
        )

        raw_response = step32.ask_local_llm(
            ollama_exe,
            question,
            schema_text,
        )

        sql = step32.extract_sql(
            raw_response
        )

        record["sql_generated"] = sql

        print(
            "\nGenerated SQL:"
        )
        print(sql)

        # ----------------------------------------------------
        # Validate + execute through Step 31
        # ----------------------------------------------------

        print(
            "\nRunning Step 31 validation..."
        )

        validation_started = time.perf_counter()

        result = step31.process_question_and_sql(
            question=question,
            sql=sql,
            max_rows=200,
        )

        validation_runtime = (
            time.perf_counter()
            - validation_started
        )

        record["validation_status"] = (
            "PASSED"
        )

        record["execution_status"] = (
            "PASSED"
        )

        record["row_count"] = (
            count_result_rows(result)
        )

        record["result_status"] = (
            "SUCCESS"
        )

        record["runtime_seconds"] = round(
            time.perf_counter()
            - started,
            4,
        )

        print(
            "\nExecution: PASSED"
        )

        print(
            "Rows returned:",
            record["row_count"],
        )

        print(
            "Runtime:",
            record["runtime_seconds"],
            "seconds",
        )

        # Keep console output concise.
        print(
            "\nResult preview:"
        )

        if hasattr(result, "head"):
            print(
                result.head(5).to_string(
                    index=False
                )
            )

        else:
            print(
                str(result)[:2000]
            )

        return record

    except PermissionError as exc:

        record["validation_status"] = (
            "BLOCKED"
        )

        record["execution_status"] = (
            "NOT_RUN"
        )

        record["result_status"] = (
            "BLOCKED"
        )

        record["error"] = str(exc)

        record["runtime_seconds"] = round(
            time.perf_counter()
            - started,
            4,
        )

        print(
            "\nValidation: BLOCKED"
        )
        print(
            "Reason:",
            str(exc),
        )

        return record

    except Exception as exc:

        record["validation_status"] = (
            record["validation_status"]
            or "PASSED"
        )

        record["execution_status"] = (
            "FAILED"
        )

        record["result_status"] = (
            "ERROR"
        )

        record["error"] = (
            type(exc).__name__
            + ": "
            + str(exc)
        )

        record["runtime_seconds"] = round(
            time.perf_counter()
            - started,
            4,
        )

        print(
            "\nExecution: FAILED"
        )
        print(
            "Reason:",
            record["error"],
        )

        return record

    finally:

        # Audit record is always written.
        write_audit_record(record)


# ============================================================
# AUDIT SECURITY TEST
# ============================================================

def run_audit_test() -> None:

    ensure_audit_files()

    test_record = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "question_hash": question_hash(
            "AUDIT TEST"
        ),

        "question": "AUDIT TEST",

        "privacy_status": "TEST",

        "sql_generated": (
            "SELECT circle, COUNT(*) "
            "FROM subscribers "
            "GROUP BY circle "
            "LIMIT 5"
        ),

        "validation_status": "PASSED",

        "execution_status": "PASSED",

        "runtime_seconds": 0.1234,

        "row_count": 5,

        "result_status": "SUCCESS",

        "error": "",

        "model": OLLAMA_MODEL,

        "ollama_version": "test",
    }

    write_audit_record(
        test_record
    )

    print(
        "[OK] Audit write test passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "JIO CHATBOT - STEP 33 AUDIT LOGGING"
    )
    print("=" * 78)

    ensure_audit_files()

    print(
        "\nAudit CSV:"
    )
    print(
        AUDIT_CSV
    )

    print(
        "\nAudit JSONL:"
    )
    print(
        AUDIT_JSONL
    )

    # --------------------------------------------------------
    # Load components
    # --------------------------------------------------------

    print(
        "\n[1/3] Loading Step 31..."
    )

    step31 = load_module(
        STEP31_PATH,
        "step31_audit_module",
    )

    print(
        "[OK] Step 31 loaded."
    )

    print(
        "\n[2/3] Loading Step 32..."
    )

    step32 = load_module(
        STEP32_PATH,
        "step32_audit_module",
    )

    print(
        "[OK] Step 32 loaded."
    )

    ollama_exe = find_ollama()

    print(
        "\nOllama:"
    )
    print(
        get_ollama_version(
            ollama_exe
        )
    )

    # --------------------------------------------------------
    # Audit file write test
    # --------------------------------------------------------

    print(
        "\n[3/3] Testing audit logging..."
    )

    run_audit_test()

    # --------------------------------------------------------
    # Real end-to-end audited requests
    # --------------------------------------------------------

    print(
        "\n" + "=" * 78
    )
    print(
        "END-TO-END AUDIT TESTS"
    )
    print(
        "=" * 78
    )

    requests = [
        "Which 5 circles have the highest 30-day churn rate?",
        "What is the average ARPU by circle?",
        "Give me all subscriber IDs.",
    ]

    records = []

    for question in requests:

        record = run_audited_request(
            step31=step31,
            step32=step32,
            ollama_exe=ollama_exe,
            question=question,
        )

        records.append(record)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successful = sum(
        1
        for r in records
        if r["result_status"]
        == "SUCCESS"
    )

    blocked = sum(
        1
        for r in records
        if r["result_status"]
        == "BLOCKED"
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "AUDIT TEST SUMMARY"
    )

    print(
        f"Requests: {len(records)}"
    )

    print(
        f"Successful requests: {successful}"
    )

    print(
        f"Blocked requests: {blocked}"
    )

    print(
        "\nAudit files created:"
    )

    print(
        AUDIT_CSV
    )

    print(
        AUDIT_JSONL
    )

    print(
        "\n[OK] Step 33 complete."
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()