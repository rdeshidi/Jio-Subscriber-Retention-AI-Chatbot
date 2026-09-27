"""
Step 38 - SHAP Chatbot Integration Layer

Purpose:
- Load the privacy-safe SHAP artifacts from Step 37.
- Provide business-friendly explanations for chatbot use.
- Never expose subscriber IDs.
- Never expose internal explanation IDs to the user.
- Keep the existing Streamlit app unchanged for now.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GLOBAL_SHAP_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "37_shap_global_feature_importance.csv"
)

INDIVIDUAL_SHAP_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "37_shap_individual_subscriber_explanations.csv"
)


# ------------------------------------------------------------
# Human-friendly feature names
# ------------------------------------------------------------

FEATURE_LABELS = {
    "device_brand": "device brand",
    "voice_minutes_last_month": "voice usage",
    "family_plan_flag": "family-plan status",
    "arpu_last_month_inr": "recent ARPU",
    "circle": "circle",
    "avg_recharge_gap_days": "average recharge gap",
    "plan_type": "plan type",
    "home_product": "home product",
    "recharge_count_6m": "recharge frequency",
    "outgoing_to_competitor_pct": "outgoing activity toward competitors",
    "data_gb_3m_avg": "average data usage",
    "avg_sinr_db": "signal quality",
    "zone": "zone",
    "days_since_last_recharge": "days since last recharge",
    "is_5g_device": "5G device status",
    "payment_failures_6m": "payment failures",
    "unresolved_complaints": "unresolved complaints",
    "complaints_6m": "complaint volume",
    "site_congestion_score": "site congestion",
    "drop_call_rate_pct": "drop-call rate",
    "app_logins_30d": "app engagement",
    "tenure_months": "customer tenure",
}


# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------

def load_global_shap() -> pd.DataFrame:
    if not GLOBAL_SHAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing SHAP global report: {GLOBAL_SHAP_PATH}"
        )

    return pd.read_csv(GLOBAL_SHAP_PATH)


def load_individual_shap() -> pd.DataFrame:
    if not INDIVIDUAL_SHAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing SHAP individual report: {INDIVIDUAL_SHAP_PATH}"
        )

    return pd.read_csv(INDIVIDUAL_SHAP_PATH)


# ------------------------------------------------------------
# Global explanation
# ------------------------------------------------------------

def summarize_global_shap(top_n: int = 8) -> str:
    """
    Return a privacy-safe explanation of the main model drivers.
    """

    df = load_global_shap()

    df = df.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).head(top_n)

    reasons = []

    for _, row in df.iterrows():

        feature = row["feature"]

        label = FEATURE_LABELS.get(
            feature,
            feature.replace("_", " "),
        )

        reasons.append(
            f"- {label}"
        )

    return (
        "The main factors contributing to the "
        "30-day churn model are:\n"
        + "\n".join(reasons)
        + "\n\n"
        "These are model explanations, not proof of "
        "causation."
    )


# ------------------------------------------------------------
# Individual explanation
# ------------------------------------------------------------

def explain_high_risk_case(
    explanation_number: int = 1,
    top_n: int = 5,
) -> str:
    """
    Return a privacy-safe explanation for an internally
    selected high-risk case.

    The internal explanation ID and subscriber ID are never
    returned.
    """

    df = load_individual_shap()

    explanation_id = (
        f"HIGH_RISK_{explanation_number:03d}"
    )

    case = df[
        df["explanation_id"]
        == explanation_id
    ].copy()

    if case.empty:
        return (
            "No SHAP explanation was found for "
            "the requested case."
        )

    probability = float(
        case["churn_probability_30d"].iloc[0]
    )

    case = (
        case.sort_values(
            "reason_rank"
        )
        .head(top_n)
    )

    increasing = []
    decreasing = []

    for _, row in case.iterrows():

        feature = row["feature"]

        label = FEATURE_LABELS.get(
            feature,
            feature.replace("_", " "),
        )

        value = row["feature_value"]

        text = (
            f"{label} ({value})"
        )

        if row["shap_value"] > 0:

            increasing.append(text)

        else:

            decreasing.append(text)

    lines = [
        "Model-based churn explanation:",
        f"Estimated 30-day churn risk: "
        f"{probability:.1%}",
    ]

    if increasing:

        lines.append(
            "\nFactors increasing modelled risk:"
        )

        for item in increasing:
            lines.append(
                f"- {item}"
            )

    if decreasing:

        lines.append(
            "\nFactors decreasing modelled risk:"
        )

        for item in decreasing:
            lines.append(
                f"- {item}"
            )

    lines.append(
        "\nSHAP explains why the model produced "
        "this prediction; it does not establish "
        "a causal relationship."
    )

    return "\n".join(lines)


# ------------------------------------------------------------
# Simple routing for chatbot integration
# ------------------------------------------------------------

def answer_shap_question(
    question: str,
) -> str:
    """
    Route safe explainability questions to SHAP.

    This first integration deliberately supports
    explainability questions without allowing users to
    request subscriber identifiers.
    """

    q = question.lower().strip()

    blocked_terms = [
        "subscriber id",
        "subscriber ids",
        "phone number",
        "contact number",
        "email",
        "employee id",
        "salary",
    ]

    if any(
        term in q
        for term in blocked_terms
    ):
        return (
            "I can't provide personal, identifying, "
            "confidential, or sensitive information. "
            "I can provide aggregated or privacy-safe "
            "model explanations instead."
        )

    global_terms = [
    "main churn factors",
    "important churn factors",
    "what drives churn",
    "what is driving churn",
    "what are the main factors driving churn",
    "factors driving churn",
    "why churn",
    "shap",
    "model explanation",
    "model factors",
]

    if any(
        term in q
        for term in global_terms
    ):

        return summarize_global_shap(
            top_n=8
        )

    return (
        "I can explain the model's main churn-risk "
        "factors using SHAP."
    )


# ------------------------------------------------------------
# Test suite
# ------------------------------------------------------------

def main() -> None:

    print("=" * 78)
    print("JIO RETENTION - STEP 38 SHAP CHATBOT INTEGRATION")
    print("=" * 78)

    print("\n[1/4] Loading SHAP artifacts...")

    global_df = load_global_shap()
    individual_df = load_individual_shap()

    print(
        f"[OK] Global SHAP rows: "
        f"{len(global_df):,}"
    )

    print(
        f"[OK] Individual SHAP rows: "
        f"{len(individual_df):,}"
    )

    print(
        "\n[2/4] Testing global SHAP explanation..."
    )

    global_answer = answer_shap_question(
        "What are the main factors driving churn?"
    )

    print(global_answer)

    if "device brand" not in global_answer:
        raise AssertionError(
            "Global SHAP explanation did not return "
            "expected feature information."
        )

    print(
        "\n[PASS] Global explanation test"
    )

    print(
        "\n[3/4] Testing individual SHAP explanation..."
    )

    individual_answer = explain_high_risk_case(
        explanation_number=1,
        top_n=5,
    )

    print(individual_answer)

    if "subscriber_id" in individual_answer.lower():
        raise AssertionError(
            "Subscriber ID leaked into explanation."
        )

    if "HIGH_RISK_001" in individual_answer:
        raise AssertionError(
            "Internal explanation ID leaked "
            "into explanation."
        )

    if "Estimated 30-day churn risk" not in individual_answer:
        raise AssertionError(
            "Individual SHAP explanation did not "
            "return risk information."
        )

    print(
        "\n[PASS] Individual explanation privacy test"
    )

    print(
        "\n[4/4] Testing privacy blocking..."
    )

    blocked_answer = answer_shap_question(
        "Give me the subscriber IDs."
    )

    print(blocked_answer)

    if "I can't provide" not in blocked_answer:
        raise AssertionError(
            "Sensitive request was not blocked."
        )

    print(
        "\n[PASS] Privacy blocking test"
    )

    print("\n" + "=" * 78)
    print("STEP 38 TEST RESULT: 3/3 PASSED")
    print("[OK] SHAP chatbot integration layer is ready.")
    print("=" * 78)


if __name__ == "__main__":
    main()