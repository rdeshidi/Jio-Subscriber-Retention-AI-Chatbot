"""
Step 35 - Jio Retention Streamlit Chatbot

Local/free architecture:
Streamlit
    -> Privacy filter
    -> Qwen 2.5 3B via Ollama
    -> Step 31 SQL safety
    -> Read-only MySQL
    -> Answer synthesis

No OpenAI API key is required.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st


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

STEP34_PATH = (
    PROJECT_ROOT
    / "src"
    / "34_answer_synthesis.py"
)

STEP38_PATH = (
    PROJECT_ROOT
    / "src"
    / "38_shap_chatbot_integration.py"
)

STEP33_PATH = (
    PROJECT_ROOT
    / "src"
    / "33_chatbot_audit_logging.py"
)

OLLAMA_MODEL = "qwen2.5:3b"

MAX_ROWS = 200


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Jio Retention AI Chatbot",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# MODULE LOADER
# ============================================================

@st.cache_resource
def load_module(
    path_string: str,
    module_name: str,
):
    """Load an existing project module once."""

    path = Path(path_string)

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

    spec.loader.exec_module(module)

    return module


# ============================================================
# OLLAMA
# ============================================================

def find_ollama() -> str:
    """Find Ollama on Windows."""

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
        "Ollama was not found."
    )


# ============================================================
# AUDIT LOGGING
# ============================================================

def write_audit_record(
    record: dict,
) -> None:
    """
    Write a privacy-safe audit record.

    Uses the same audit-file locations established in Step 33.
    """

    audit_module = load_module(
        str(STEP33_PATH),
        "step33_streamlit_module",
    )

    audit_module.write_audit_record(
        record
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# LOAD PROJECT COMPONENTS
# ============================================================

try:

    step31 = load_module(
        str(STEP31_PATH),
        "step31_streamlit_module",
    )

    step32 = load_module(
        str(STEP32_PATH),
        "step32_streamlit_module",
    )

    step34 = load_module(
        str(STEP34_PATH),
        "step34_streamlit_module",
    )

    step38 = load_module(
    str(STEP38_PATH),
    "step38_streamlit_module",
)

    ollama_exe = find_ollama()

except Exception as exc:

    st.error(
        "Chatbot initialization failed."
    )

    st.code(
        f"{type(exc).__name__}: {exc}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Jio Retention AI")

    st.write(
        "Local analytics assistant for "
        "privacy-safe business questions."
    )

    st.divider()

    st.subheader("System")

    st.success(
        "Ollama connected"
    )

    st.write(
        f"Model: `{OLLAMA_MODEL}`"
    )

    st.write(
        "Database: `jio_retention_db`"
    )

    st.write(
        "Access: Read-only"
    )

    st.divider()

    st.subheader("Security")

    st.write("✅ Privacy filtering")
    st.write("✅ Approved-table allow-list")
    st.write("✅ SELECT-only SQL")
    st.write("✅ Sensitive-column blocking")
    st.write("✅ Row-limit guard")
    st.write("✅ Result sanitization")
    st.write("✅ Query audit logging")

    st.divider()

    st.caption(
        "Customer and employee personal information "
        "is not provided by this assistant."
    )

    if st.button(
        "Clear conversation",
        width="stretch",
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Jio Subscriber Retention AI Chatbot"
)

st.caption(
    "Ask privacy-safe business questions about "
    "subscriber retention, churn, ARPU and related KPIs."
)

st.info(
    "Examples: "
    "Which 5 circles have the highest 30-day churn rate?  "
    "What is the average ARPU by circle?"
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        if message["role"] == "user":

            st.write(
                message["content"]
            )

        else:

            st.markdown(
                message["content"]
            )

            if message.get("sql"):

                with st.expander(
                    "View validated SQL"
                ):

                    st.code(
                        message["sql"],
                        language="sql",
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a Jio business analytics question..."
)


if question:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.write(question)

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        start_time = __import__(
            "time"
        ).perf_counter()

        try:

            # ------------------------------------------------
            # 1. Privacy check
            # ------------------------------------------------

            allowed, privacy_message = (
                step31.check_question(
                    question
                )
            )

            if not allowed:

                answer = (
                    privacy_message
                )

                st.warning(
                    answer
                )

                runtime = round(
                    __import__("time").perf_counter()
                    - start_time,
                    4,
                )

                # Sensitive question itself is not stored.
                write_audit_record(
                    {
                        "timestamp_utc": (
                            __import__("datetime")
                            .datetime
                            .now(
                                __import__("datetime")
                                .timezone.utc
                            )
                            .isoformat()
                        ),
                        "question_hash": __import__(
                            "hashlib"
                        ).sha256(
                            question.encode(
                                "utf-8"
                            )
                        ).hexdigest()[:16],
                        "question": (
                            "[REDACTED: privacy blocked]"
                        ),
                        "privacy_status": "BLOCKED",
                        "sql_generated": "",
                        "validation_status": "NOT_RUN",
                        "execution_status": "NOT_RUN",
                        "runtime_seconds": runtime,
                        "row_count": 0,
                        "result_status": "BLOCKED",
                        "error": privacy_message,
                        "model": OLLAMA_MODEL,
                        "ollama_version": (
                            "local Ollama"
                        ),
                    }
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            else:
                # --------------------------------------------
                # 1A. SHAP explainability route
                # --------------------------------------------

                shap_answer = (
                    step38.answer_shap_question(
                        question
                    )
                )

                shap_question = any(
                    phrase in question.lower()
                    for phrase in [
                        "shap",
                        "model explanation",
                        "main churn factors",
                        "important churn factors",
                        "what drives churn",
                        "what is driving churn",
                        "what are the main factors driving churn",
                        "factors driving churn",
                    ]
                )

                if shap_question:

                    answer = shap_answer

                    st.markdown(
                        answer
                    )

                    runtime = round(
                        __import__("time").perf_counter()
                        - start_time,
                        4,
                    )

                    write_audit_record(
                        {
                            "timestamp_utc": (
                                __import__(
                                    "datetime"
                                )
                                .datetime
                                .now(
                                    __import__(
                                        "datetime"
                                    )
                                    .timezone.utc
                                )
                                .isoformat()
                            ),
                            "question_hash": __import__(
                                "hashlib"
                            ).sha256(
                                question.encode(
                                    "utf-8"
                                )
                            ).hexdigest()[:16],
                            "question": question,
                            "privacy_status": "PASSED",
                            "sql_generated": "",
                            "validation_status": "NOT_APPLICABLE",
                            "execution_status": "NOT_RUN",
                            "runtime_seconds": runtime,
                            "row_count": 0,
                            "result_status": "SUCCESS",
                            "error": "",
                            "model": "SHAP + "
                                    + OLLAMA_MODEL,
                            "ollama_version": "local Ollama",
                        }
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                else:

                # --------------------------------------------
                # 2. Load privacy-safe schema
                # --------------------------------------------

                    schema = (
                        step32.load_safe_schema()
                    )

                    schema_text = (
                        step32.build_schema_prompt(
                            schema
                        )
                    )

                    # --------------------------------------------
                    # 3. Generate SQL
                    # --------------------------------------------

                    with st.spinner(
                        "Generating SQL with local Qwen model..."
                    ):

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

                    # --------------------------------------------
                    # 4. Execute through Step 31
                    # --------------------------------------------

                    with st.spinner(
                        "Validating and querying Jio database..."
                    ):

                        db_result = (
                            step31.process_question_and_sql(
                                question=question,
                                sql=sql,
                                max_rows=MAX_ROWS,
                            )
                        )

                    # --------------------------------------------
                    # 5. Synthesize answer
                    # --------------------------------------------

                    with st.spinner(
                        "Preparing verified business answer..."
                    ):

                        answer = (
                            step34.build_verified_answer(
                                question,
                                db_result,
                            )
                        )

                        answer = (
                            step34.add_sql_trace(
                                answer,
                                sql,
                            )
                        )

                    runtime = round(
                        __import__("time").perf_counter()
                        - start_time,
                        4,
                    )

                    # --------------------------------------------
                    # 6. Display answer
                    # --------------------------------------------

                    st.markdown(
                        answer
                    )

                    # --------------------------------------------
                    # 7. Optional data preview
                    # --------------------------------------------

                    if isinstance(
                        db_result,
                        pd.DataFrame,
                    ):

                        with st.expander(
                            "View returned data"
                        ):

                            st.dataframe(
                                db_result,
                            width="stretch",
                                hide_index=True,
                            )

                    elif isinstance(
                        db_result,
                        list,
                    ):

                        with st.expander(
                            "View returned data"
                        ):

                            st.write(
                                db_result
                            )

                    # --------------------------------------------
                    # 8. Extract trace SQL for UI storage
                    # --------------------------------------------

                    answer_for_history = answer

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_for_history,
                            "sql": sql,
                        }
                    )

                    # --------------------------------------------
                    # 9. Audit log
                    # --------------------------------------------

                    row_count = 0

                    if hasattr(
                        db_result,
                        "shape",
                    ):

                        row_count = int(
                            db_result.shape[0]
                        )

                    elif isinstance(
                        db_result,
                        list,
                    ):

                        row_count = len(
                            db_result
                        )

                    write_audit_record(
                        {
                            "timestamp_utc": (
                                __import__(
                                    "datetime"
                                )
                                .datetime
                                .now(
                                    __import__(
                                        "datetime"
                                    )
                                    .timezone.utc
                                )
                                .isoformat()
                            ),
                            "question_hash": __import__(
                                "hashlib"
                            ).sha256(
                                question.encode(
                                    "utf-8"
                                )
                            ).hexdigest()[:16],
                            "question": question,
                            "privacy_status": "PASSED",
                            "sql_generated": sql,
                            "validation_status": "PASSED",
                            "execution_status": "PASSED",
                            "runtime_seconds": runtime,
                            "row_count": row_count,
                            "result_status": "SUCCESS",
                            "error": "",
                            "model": OLLAMA_MODEL,
                            "ollama_version": "local Ollama",
                        }
                    )

        except PermissionError as exc:

            runtime = round(
                __import__("time").perf_counter()
                - start_time,
                4,
            )

            st.error(
                "The request was blocked by the SQL security layer."
            )

            st.code(
                str(exc)
            )

            write_audit_record(
                {
                    "timestamp_utc": (
                        __import__(
                            "datetime"
                        )
                        .datetime
                        .now(
                            __import__(
                                "datetime"
                            )
                            .timezone.utc
                        )
                        .isoformat()
                    ),
                    "question_hash": __import__(
                        "hashlib"
                    ).sha256(
                        question.encode(
                            "utf-8"
                        )
                    ).hexdigest()[:16],
                    "question": question,
                    "privacy_status": "PASSED",
                    "sql_generated": "",
                    "validation_status": "BLOCKED",
                    "execution_status": "NOT_RUN",
                    "runtime_seconds": runtime,
                    "row_count": 0,
                    "result_status": "BLOCKED",
                    "error": str(exc),
                    "model": OLLAMA_MODEL,
                    "ollama_version": "local Ollama",
                }
            )

        except Exception as exc:

            runtime = round(
                __import__("time").perf_counter()
                - start_time,
                4,
            )

            st.error(
                "The chatbot encountered an execution error."
            )

            st.code(
                f"{type(exc).__name__}: {exc}"
            )

            write_audit_record(
                {
                    "timestamp_utc": (
                        __import__(
                            "datetime"
                        )
                        .datetime
                        .now(
                            __import__(
                                "datetime"
                            )
                            .timezone.utc
                        )
                        .isoformat()
                    ),
                    "question_hash": __import__(
                        "hashlib"
                    ).sha256(
                        question.encode(
                            "utf-8"
                        )
                    ).hexdigest()[:16],
                    "question": question,
                    "privacy_status": "PASSED",
                    "sql_generated": "",
                    "validation_status": "UNKNOWN",
                    "execution_status": "FAILED",
                    "runtime_seconds": runtime,
                    "row_count": 0,
                    "result_status": "ERROR",
                    "error": (
                        type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                    "model": OLLAMA_MODEL,
                    "ollama_version": "local Ollama",
                }
            )