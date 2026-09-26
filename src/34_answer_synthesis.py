"""
Step 34 - Verified Answer Synthesis
Jio Subscriber Retention & AI Chatbot

Important design:
- Step 32 uses Qwen for SQL generation.
- Step 31 validates and executes the SQL.
- Step 34 uses the returned database result as the source of truth.
- Numeric rankings and min/max facts are calculated in Python.
- The LLM is NOT allowed to invent or change numeric facts.

This prevents a language model from producing a plausible but
factually incorrect summary of a correct SQL result.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# CONFIGURATION
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

DEMO_REPORT = (
    OUTPUT_DIR
    / "34_answer_synthesis_demo.txt"
)

OLLAMA_MODEL = "qwen2.5:3b"

MAX_ROWS = 200


# ============================================================
# MODULE LOADER
# ============================================================

def load_module(
    path: Path,
    module_name: str,
):
    """Load a Python module from file."""

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

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# OLLAMA
# ============================================================

def find_ollama() -> str:
    """Find Ollama on Windows."""

    found = shutil.which(
        "ollama"
    )

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
        "Ollama executable was not found."
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
# RESULT CONVERSION
# ============================================================

def to_dataframe(
    result: Any,
) -> pd.DataFrame | None:
    """
    Convert common Step 31 results into a DataFrame.
    """

    if isinstance(
        result,
        pd.DataFrame,
    ):
        return result.copy()

    if isinstance(
        result,
        list,
    ):

        try:
            return pd.DataFrame(
                result
            )
        except Exception:
            return None

    if isinstance(
        result,
        dict,
    ):

        # Try common result containers.
        for key in [
            "rows",
            "data",
            "results",
        ]:

            value = result.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                try:
                    return pd.DataFrame(
                        value
                    )
                except Exception:
                    return None

    return None


# ============================================================
# VERIFIED FACT EXTRACTION
# ============================================================

def find_column(
    df: pd.DataFrame,
    exact_names: list[str],
    contains_terms: list[str],
) -> str | None:
    """
    Find the most appropriate column from the returned data.
    """

    for name in exact_names:

        if name in df.columns:
            return name

    for column in df.columns:

        lower = str(
            column
        ).lower()

        if all(
            term.lower() in lower
            for term in contains_terms
        ):
            return column

    return None


def extract_top_n_from_question(
    question: str,
    default: int = 5,
) -> int:

    match = re.search(
        r"\btop\s+(\d+)\b",
        question,
        flags=re.IGNORECASE,
    )

    if match:
        return int(
            match.group(1)
        )

    match = re.search(
        r"\b(\d+)\s+(?:highest|lowest)\b",
        question,
        flags=re.IGNORECASE,
    )

    if match:
        return int(
            match.group(1)
        )

    return default


def verified_churn_summary(
    question: str,
    df: pd.DataFrame,
) -> str | None:
    """
    Deterministically answer circle churn-ranking questions.
    """

    q = question.lower()

    if (
        "churn" not in q
        or "circle" not in q
    ):
        return None

    rate_column = find_column(
        df,
        exact_names=[
            "churn_rate_percentage",
            "monthly_churn_pct",
        ],
        contains_terms=[
            "churn",
            "rate",
        ],
    )

    circle_column = find_column(
        df,
        exact_names=[
            "circle",
        ],
        contains_terms=[
            "circle",
        ],
    )

    if not rate_column or not circle_column:
        return None

    working = df.copy()

    working[rate_column] = pd.to_numeric(
        working[rate_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            rate_column
        ]
    )

    if working.empty:
        return None

    n = extract_top_n_from_question(
        question,
        default=5,
    )

    top = (
        working
        .sort_values(
            rate_column,
            ascending=False,
        )
        .head(n)
    )

    lines = [
        f"Top {len(top)} circles by 30-day churn rate:"
    ]

    for _, row in top.iterrows():

        circle = str(
            row[circle_column]
        )

        rate = float(
            row[rate_column]
        )

        lines.append(
            f"- {circle}: {rate:.2f}%"
        )

    return "\n".join(
        lines
    )


def verified_arpu_summary(
    question: str,
    df: pd.DataFrame,
) -> str | None:
    """
    Deterministically answer average-ARPU-by-circle questions.
    """

    q = question.lower()

    if (
        "arpu" not in q
        or "circle" not in q
    ):
        return None

    arpu_column = find_column(
        df,
        exact_names=[
            "average_arpu_inr",
            "arpu_last_month_inr",
            "arpu_inr",
        ],
        contains_terms=[
            "arpu",
        ],
    )

    circle_column = find_column(
        df,
        exact_names=[
            "circle",
        ],
        contains_terms=[
            "circle",
        ],
    )

    if not arpu_column or not circle_column:
        return None

    working = df.copy()

    working[arpu_column] = pd.to_numeric(
        working[arpu_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            arpu_column
        ]
    )

    if working.empty:
        return None

    highest_row = working.loc[
        working[arpu_column].idxmax()
    ]

    lowest_row = working.loc[
        working[arpu_column].idxmin()
    ]

    highest_circle = str(
        highest_row[circle_column]
    )

    highest_value = float(
        highest_row[arpu_column]
    )

    lowest_circle = str(
        lowest_row[circle_column]
    )

    lowest_value = float(
        lowest_row[arpu_column]
    )

    return (
        "Average ARPU by circle ranges from "
        f"₹{lowest_value:.2f} in {lowest_circle} "
        f"to ₹{highest_value:.2f} in {highest_circle}.\n"
        f"- Highest average ARPU: "
        f"{highest_circle} — ₹{highest_value:.2f}\n"
        f"- Lowest average ARPU: "
        f"{lowest_circle} — ₹{lowest_value:.2f}"
    )


# ============================================================
# GENERIC VERIFIED SUMMARY
# ============================================================

def generic_verified_summary(
    question: str,
    df: pd.DataFrame,
) -> str:
    """
    Safe fallback for questions we haven't specifically templated.

    We deliberately avoid making inferred claims.
    """

    if df.empty:
        return (
            "The query returned no rows."
        )

    row_count = len(
        df
    )

    preview = df.head(
        5
    )

    return (
        f"The query returned {row_count} rows. "
        "The first five returned rows are shown below.\n\n"
        + preview.to_markdown(
            index=False
        )
    )


# ============================================================
# VERIFIED ANSWER
# ============================================================

def build_verified_answer(
    question: str,
    result: Any,
) -> str:
    """
    Build a factually controlled answer directly from the
    database result.

    Numeric claims come from Python calculations over the
    returned data, not from the LLM.
    """

    df = to_dataframe(
        result
    )

    if df is None:
        return (
            "The database returned a result, but it could not "
            "be converted into a tabular format for verified "
            "summarization."
        )

    if df.empty:
        return (
            "The query returned no data."
        )

    # --------------------------------------------------------
    # Known churn ranking
    # --------------------------------------------------------

    churn_answer = verified_churn_summary(
        question,
        df,
    )

    if churn_answer:
        return churn_answer

    # --------------------------------------------------------
    # Known ARPU analysis
    # --------------------------------------------------------

    arpu_answer = verified_arpu_summary(
        question,
        df,
    )

    if arpu_answer:
        return arpu_answer

    # --------------------------------------------------------
    # Generic safe fallback
    # --------------------------------------------------------

    return generic_verified_summary(
        question,
        df,
    )


# ============================================================
# TRACEABILITY
# ============================================================

def add_sql_trace(
    answer: str,
    sql: str,
) -> str:
    """
    Append the exact validated SQL generated by the application.
    """

    return (
        answer.strip()
        + "\n\nSQL used:\n"
        + sql.strip()
    )


# ============================================================
# END-TO-END
# ============================================================

def run_chatbot_question(
    step31,
    step32,
    ollama_exe: str,
    question: str,
) -> str:

    print("\n" + "=" * 78)
    print("CHATBOT QUESTION")
    print("=" * 78)

    print(question)

    # --------------------------------------------------------
    # Privacy
    # --------------------------------------------------------

    allowed, message = (
        step31.check_question(
            question
        )
    )

    print(
        "\nPrivacy:",
        "PASSED" if allowed else "BLOCKED",
    )

    if not allowed:

        print(
            "\n--- FINAL CHATBOT ANSWER ---"
        )

        print(
            message
        )

        return message

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    schema = (
        step32.load_safe_schema()
    )

    schema_text = (
        step32.build_schema_prompt(
            schema
        )
    )

    # --------------------------------------------------------
    # SQL generation
    # --------------------------------------------------------

    print(
        "\nGenerating SQL..."
    )

    raw_response = (
        step32.ask_local_llm(
            ollama_exe,
            question,
            schema_text,
        )
    )

    sql = (
        step32.extract_sql(
            raw_response
        )
    )

    print(
        "\nGenerated SQL:"
    )

    print(
        sql
    )

    # --------------------------------------------------------
    # Step 31 validation/execution
    # --------------------------------------------------------

    print(
        "\nExecuting through Step 31..."
    )

    result = (
        step31.process_question_and_sql(
            question=question,
            sql=sql,
            max_rows=MAX_ROWS,
        )
    )

    print(
        "[OK] Query executed successfully."
    )

    # --------------------------------------------------------
    # Verified answer
    # --------------------------------------------------------

    print(
        "\nBuilding verified answer..."
    )

    verified_answer = (
        build_verified_answer(
            question,
            result,
        )
    )

    final_answer = (
        add_sql_trace(
            verified_answer,
            sql,
        )
    )

    print(
        "\n--- FINAL CHATBOT ANSWER ---"
    )

    print(
        final_answer
    )

    return final_answer


# ============================================================
# DEMO REPORT
# ============================================================

def append_demo_report(
    question: str,
    answer: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DEMO_REPORT,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n" + "=" * 78 + "\n"
        )

        f.write(
            "QUESTION\n"
        )

        f.write(
            question + "\n\n"
        )

        f.write(
            "VERIFIED ANSWER\n"
        )

        f.write(
            answer + "\n"
        )


# ============================================================
# TESTS
# ============================================================

def run_tests(
    step31,
    step32,
    ollama_exe: str,
) -> None:

    questions = [
        "Which 5 circles have the highest 30-day churn rate?",
        "What is the average ARPU by circle?",
    ]

    passed = 0

    for question in questions:

        try:

            answer = run_chatbot_question(
                step31,
                step32,
                ollama_exe,
                question,
            )

            append_demo_report(
                question,
                answer,
            )

            passed += 1

            print(
                "\n[PASS] Verified answer generated."
            )

        except Exception as exc:

            print(
                "\n[FAIL]"
            )

            print(
                type(exc).__name__
                + ": "
                + str(exc)
            )

    print(
        "\n" + "=" * 78
    )

    print(
        f"STEP 34 TEST RESULT: "
        f"{passed}/{len(questions)} PASSED"
    )

    print(
        "\nDemo report:"
    )

    print(
        DEMO_REPORT
    )

    if passed == len(questions):

        print(
            "\n[OK] Step 34 verified answer synthesis is functioning."
        )

    else:

        print(
            "\n[REVIEW] Some tests failed."
        )

    print(
        "=" * 78
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "JIO CHATBOT - STEP 34 VERIFIED ANSWER SYNTHESIS"
    )
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Fresh report.
    with open(
        DEMO_REPORT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "JIO CHATBOT - STEP 34 VERIFIED ANSWER SYNTHESIS\n"
        )

    print(
        "\n[1/3] Finding Ollama..."
    )

    ollama_exe = find_ollama()

    print(
        "[OK]",
        ollama_exe,
    )

    print(
        get_ollama_version(
            ollama_exe
        )
    )

    print(
        "\n[2/3] Loading Step 31..."
    )

    step31 = load_module(
        STEP31_PATH,
        "step31_verified_module",
    )

    print(
        "[OK] Step 31 loaded."
    )

    print(
        "\n[3/3] Loading Step 32..."
    )

    step32 = load_module(
        STEP32_PATH,
        "step32_verified_module",
    )

    print(
        "[OK] Step 32 loaded."
    )

    run_tests(
        step31,
        step32,
        ollama_exe,
    )


if __name__ == "__main__":
    main()