"""
Step 36 - Final Project Validation
Jio Subscriber Retention & AI Chatbot

Purpose:
- Validate project structure
- Validate important deliverables
- Compile all Python scripts
- Check chatbot artifacts
- Check security/configuration hygiene
- Produce a final validation report

This script does NOT:
- retrain ML models
- execute chatbot LLM calls
- modify the database
"""

from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_PATH = OUTPUT_DIR / "36_final_project_validation_report.txt"


# ============================================================
# HELPERS
# ============================================================

passed = []
failed = []
warnings = []


def check(
    name: str,
    condition: bool,
    detail: str = "",
) -> None:

    if condition:

        passed.append(name)

        print(f"[PASS] {name}")

        if detail:
            print(f"       {detail}")

    else:

        failed.append(name)

        print(f"[FAIL] {name}")

        if detail:
            print(f"       {detail}")


def warning(
    name: str,
    detail: str,
) -> None:

    warnings.append(name)

    print(f"[WARN] {name}")
    print(f"       {detail}")


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("JIO RETENTION PROJECT - FINAL VALIDATION")
print("=" * 78)

print(
    f"Project root: {PROJECT_ROOT}"
)


# ============================================================
# 1. CORE DIRECTORIES
# ============================================================

print("\n" + "-" * 78)
print("1. PROJECT STRUCTURE")
print("-" * 78)

required_directories = [
    "architecture",
    "data",
    "data/raw",
    "data/processed",
    "data/modeling",
    "models",
    "outputs",
    "outputs/charts",
    "outputs/reports",
    "outputs/sql_exports",
    "sql",
    "src",
]

for directory in required_directories:

    path = PROJECT_ROOT / directory

    check(
        f"Directory exists: {directory}",
        path.is_dir(),
    )


# ============================================================
# 2. CORE SOURCE SCRIPTS
# ============================================================

print("\n" + "-" * 78)
print("2. CORE SOURCE SCRIPTS")
print("-" * 78)

required_scripts = [
    "01_data_exploration.py",
    "02_data_quality_validation.py",
    "03_data_cleaning.py",
    "04_business_analysis.py",
    "05_visualizations.py",
    "06_prepare_sql_exports.py",
    "08_generate_architecture.py",
    "09_finalize_day1.py",
    "10_model_data_preparation.py",
    "11_logistic_regression_baseline.py",
    "12_gradient_boosting_model.py",
    "13_xgboost_model.py",
    "14_catboost_model.py",
    "15_model_comparison.py",
    "16_threshold_optimization.py",
    "17_targeted_model_tuning.py",
    "18_final_validation_comparison.py",
    "19_final_test_evaluation.py",
    "20_final_churn_visualizations.py",
    "21_clv_data_preparation.py",
    "22_risk_value_prioritization.py",
    "23_test_risk_value_validation.py",
    "24_offer_uplift_feasibility.py",
    "25_offer_response_data_preparation.py",
    "26_offer_response_logistic_baseline.py",
    "27_offer_response_catboost.py",
    "28_offer_response_final_summary.py",
    "29_chatbot_privacy_and_sql_safety.py",
    "30_chatbot_database_setup.py",
    "31_chatbot_readonly_query_engine.py",
    "32_local_llm_sql_agent.py",
    "33_chatbot_audit_logging.py",
    "34_answer_synthesis.py",
    "35_jio_chatbot_app.py",
]

for script in required_scripts:

    path = (
        PROJECT_ROOT
        / "src"
        / script
    )

    check(
        f"Source exists: {script}",
        path.is_file(),
    )


# ============================================================
# 3. PYTHON COMPILATION
# ============================================================

print("\n" + "-" * 78)
print("3. PYTHON SYNTAX VALIDATION")
print("-" * 78)

source_files = sorted(
    (
        PROJECT_ROOT
        / "src"
    ).glob("*.py")
)

compile_failures = []

for path in source_files:

    try:

        py_compile.compile(
            str(path),
            doraise=True,
        )

        print(
            f"[PASS] Compiles: {path.name}"
        )

    except Exception as exc:

        compile_failures.append(
            (
                path.name,
                str(exc),
            )
        )

        print(
            f"[FAIL] Compiles: {path.name}"
        )

        print(
            f"       {exc}"
        )


check(
    "All Python source files compile",
    len(compile_failures) == 0,
    f"{len(source_files)} Python files checked",
)


# ============================================================
# 4. MODEL ARTIFACTS
# ============================================================

print("\n" + "-" * 78)
print("4. MODEL ARTIFACTS")
print("-" * 78)

model_files = [
    "models/logistic_regression_baseline.pkl",
    "models/gradient_boosting_model.pkl",
    "models/xgboost_model.pkl",
    "models/catboost_model_package.pkl",
    "models/tuned_xgboost_best.pkl",
    "models/tuned_catboost_best.pkl",
]

for relative_path in model_files:

    path = PROJECT_ROOT / relative_path

    check(
        f"Model artifact: {relative_path}",
        path.is_file(),
    )

# Original final-model outputs may be stored under different
# filenames, so only test the known tuned artifacts above
# and report directory contents separately.


# ============================================================
# 5. MODEL REPORTS
# ============================================================

print("\n" + "-" * 78)
print("5. MODELING REPORTS")
print("-" * 78)

model_reports = [
    "15_model_comparison.csv",
    "16_threshold_optimization_summary.csv",
    "17_targeted_model_tuning_results.csv",
    "18_final_validation_comparison.csv",
    "18_model_shortlist.csv",
    "19_final_test_metrics.csv",
    "19_final_test_predictions.csv",
    "20_final_test_decile_analysis.csv",
    "20_final_catboost_feature_importance.csv",
    "20_validation_vs_test_comparison.csv",
    "20_final_churn_model_summary.txt",
]

for filename in model_reports:

    path = (
        OUTPUT_DIR
        / filename
    )

    check(
        f"Report exists: {filename}",
        path.is_file(),
    )


# ============================================================
# 6. CLV / OFFER ARTIFACTS
# ============================================================

print("\n" + "-" * 78)
print("6. CLV / OFFER ARTIFACTS")
print("-" * 78)

clv_files = [
    "data/clv/clv_value_prepared.csv",
    "data/clv/clv_metadata.json",
    "data/clv/customer_risk_value_priority.csv",
]

for relative_path in clv_files:

    check(
        f"Artifact exists: {relative_path}",
        (
            PROJECT_ROOT
            / relative_path
        ).is_file(),
    )


offer_reports = [
    "24_offer_uplift_feasibility_report.txt",
    "25_offer_response_data_preparation_report.txt",
    "26_offer_response_logistic_report.txt",
    "27_offer_response_catboost_report.txt",
    "28_offer_response_final_summary.txt",
]

for filename in offer_reports:

    path = (
        OUTPUT_DIR
        / filename
    )

    check(
        f"Offer report exists: {filename}",
        path.is_file(),
    )


# ============================================================
# 7. CHATBOT ARTIFACTS
# ============================================================

print("\n" + "-" * 78)
print("7. CHATBOT ARTIFACTS")
print("-" * 78)

chatbot_files = [
    "data/modeling/chatbot_safe_schema.json",
    "outputs/reports/30_chatbot_safe_database_schema.txt",
    "outputs/reports/33_chatbot_audit_log.csv",
    "outputs/reports/33_chatbot_audit_log.jsonl",
    "outputs/reports/34_answer_synthesis_demo.txt",
]

for relative_path in chatbot_files:

    check(
        f"Chatbot artifact exists: {relative_path}",
        (
            PROJECT_ROOT
            / relative_path
        ).is_file(),
    )


# ============================================================
# 8. SQL FILES
# ============================================================

print("\n" + "-" * 78)
print("8. SQL ARTIFACTS")
print("-" * 78)

sql_files = [
    "sql/01_create_tables.sql",
    "sql/02_import_data.sql",
    "sql/03_validation_queries.sql",
]

for relative_path in sql_files:

    check(
        f"SQL file exists: {relative_path}",
        (
            PROJECT_ROOT
            / relative_path
        ).is_file(),
    )


# ============================================================
# 9. DOCUMENTATION
# ============================================================

print("\n" + "-" * 78)
print("9. DOCUMENTATION")
print("-" * 78)

documentation_files = [
    "Jio_Retention_Brief.pdf",
    "DAY1_SUBMISSION_SUMMARY.md",
]

for filename in documentation_files:

    check(
        f"Documentation exists: {filename}",
        (
            PROJECT_ROOT
            / filename
        ).is_file(),
    )


# ============================================================
# 10. SECURITY / CONFIGURATION
# ============================================================

print("\n" + "-" * 78)
print("10. SECURITY / CONFIGURATION")
print("-" * 78)

env_file = PROJECT_ROOT / ".env"
gitignore_file = PROJECT_ROOT / ".gitignore"

check(
    ".env exists",
    env_file.is_file(),
)

check(
    ".gitignore exists",
    gitignore_file.is_file(),
)

if gitignore_file.is_file():

    gitignore_text = (
        gitignore_file
        .read_text(
            encoding="utf-8",
            errors="replace",
        )
        .lower()
    )

    env_ignored = (
        ".env" in gitignore_text
        or "*.env" in gitignore_text
    )

    check(
        ".env is protected by .gitignore",
        env_ignored,
    )

# Ensure raw credentials are not accidentally present in
# Python source files.

credential_hits = []

# This validation script contains security-pattern text itself,
# so exclude it from the source-value scan.
files_to_scan = [
    path
    for path in source_files
    if path.name != "36_final_project_validation.py"
]

for path in files_to_scan:

    try:

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception:
        continue

    # Look for actual hard-coded assignments, not merely the
    # presence of variable names or security-pattern strings.
    assignment_patterns = [
        r'(?i)JIO_DB_PASSWORD\s*=\s*["\']([^"\']+)["\']',
        r'(?i)OPENAI_API_KEY\s*=\s*["\']([^"\']+)["\']',
    ]

    for pattern in assignment_patterns:

        matches = re.findall(
            pattern,
            text,
        )

        for value in matches:

            # Ignore obvious placeholders.
            if value.strip().lower() in {
                "",
                "your_password",
                "your_api_key",
                "password",
                "changeme",
            }:
                continue

            credential_hits.append(
                (
                    path.name,
                    "credential assignment detected",
                )
            )

    # Detect actual OpenAI-style secret values if present.
    if re.search(
        r"\bsk-[A-Za-z0-9_-]{20,}\b",
        text,
    ):
        credential_hits.append(
            (
                path.name,
                "API-key-like value detected",
            )
        )


check(
    "No obvious hard-coded credential values found in Python source",
    len(credential_hits) == 0,
)


# ============================================================
# 11. STREAMLIT DEPRECATION CHECK
# ============================================================

print("\n" + "-" * 78)
print("11. STREAMLIT CODE CHECK")
print("-" * 78)

streamlit_path = (
    PROJECT_ROOT
    / "src"
    / "35_jio_chatbot_app.py"
)

if streamlit_path.is_file():

    text = streamlit_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    check(
        "Streamlit app does not use deprecated "
        "use_container_width parameter",
        "use_container_width" not in text,
    )


# ============================================================
# 12. KEY PROJECT OUTPUT COUNTS
# ============================================================

print("\n" + "-" * 78)
print("12. OUTPUT INVENTORY")
print("-" * 78)

chart_dir = (
    PROJECT_ROOT
    / "outputs"
    / "charts"
)

report_dir = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

chart_count = (
    len(list(chart_dir.glob("*")))
    if chart_dir.is_dir()
    else 0
)

report_count = (
    len(list(report_dir.glob("*")))
    if report_dir.is_dir()
    else 0
)

print(
    f"Charts/files in outputs/charts: {chart_count}"
)

print(
    f"Reports/files in outputs/reports: {report_count}"
)

check(
    "At least one chart artifact exists",
    chart_count > 0,
)

check(
    "At least one report artifact exists",
    report_count > 0,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("FINAL VALIDATION SUMMARY")
print("=" * 78)

print(
    f"Passed:   {len(passed)}"
)

print(
    f"Failed:   {len(failed)}"
)

print(
    f"Warnings: {len(warnings)}"
)

if failed:

    print(
        "\nFAILED CHECKS:"
    )

    for item in failed:

        print(
            f"- {item}"
        )

if warnings:

    print(
        "\nWARNINGS:"
    )

    for item in warnings:

        print(
            f"- {item}"
        )

# ------------------------------------------------------------
# Write report
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "JIO SUBSCRIBER RETENTION PROJECT\n"
    )

    f.write(
        "FINAL PROJECT VALIDATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Project root: {PROJECT_ROOT}\n\n"
    )

    f.write(
        f"Passed checks: {len(passed)}\n"
    )

    f.write(
        f"Failed checks: {len(failed)}\n"
    )

    f.write(
        f"Warnings: {len(warnings)}\n\n"
    )

    f.write(
        "PASSED\n"
    )

    for item in passed:
        f.write(
            f"- {item}\n"
        )

    f.write(
        "\nFAILED\n"
    )

    for item in failed:
        f.write(
            f"- {item}\n"
        )

    f.write(
        "\nWARNINGS\n"
    )

    for item in warnings:
        f.write(
            f"- {item}\n"
        )

    if not failed:

        f.write(
            "\nFINAL STATUS: PASS\n"
        )

    else:

        f.write(
            "\nFINAL STATUS: REVIEW REQUIRED\n"
        )

print(
    "\nValidation report:"
)

print(
    REPORT_PATH
)

if failed:

    print(
        "\n[REVIEW] Some checks failed."
    )

    sys.exit(1)

else:

    print(
        "\n[OK] Final project validation passed."
    )