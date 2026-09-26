from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 24
# OFFER RESPONSE & UPLIFT FEASIBILITY AUDIT
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSCRIBER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "subscribers_clean.csv"
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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. FIND OFFER CATALOGUE
# ------------------------------------------------------------

catalogue_candidates = [

    PROJECT_ROOT
    / "data"
    / "processed"
    / "offer_catalogue_clean.csv",

    PROJECT_ROOT
    / "data"
    / "processed"
    / "offer_catalogue.csv",

    PROJECT_ROOT
    / "data"
    / "raw"
    / "offer_catalogue.csv",
]


OFFER_CATALOGUE_FILE = None


for candidate in catalogue_candidates:

    if candidate.exists():

        OFFER_CATALOGUE_FILE = candidate
        break


# ------------------------------------------------------------
# 3. START
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 24: OFFER RESPONSE & UPLIFT FEASIBILITY")
print("=" * 90)


if not SUBSCRIBER_FILE.exists():

    raise FileNotFoundError(
        f"Subscriber file not found:\n"
        f"{SUBSCRIBER_FILE}"
    )


# ------------------------------------------------------------
# 4. LOAD SUBSCRIBERS
# ------------------------------------------------------------

df = pd.read_csv(
    SUBSCRIBER_FILE
)


print(
    f"\nSubscribers loaded : "
    f"{len(df):,}"
)

print(
    f"Columns            : "
    f"{df.shape[1]}"
)


# ------------------------------------------------------------
# 5. BINARY CONVERSION FUNCTION
# ------------------------------------------------------------

def normalize_binary(series):

    text = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


    mapped = text.map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "1": 1,
            "0": 0,
        }
    )


    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )


    numeric = numeric.where(
        numeric.isin(
            [0, 1]
        )
    )


    mapped = mapped.fillna(
        numeric
    )


    return mapped.astype(
        "Int64"
    )


# ------------------------------------------------------------
# 6. REQUIRED FIELD CHECK
# ------------------------------------------------------------

required_fields = [
    "offer_exposed_90d",
    "offer_redeemed_90d",
    "churn_flag_30d",
]


missing_fields = [
    col
    for col in required_fields
    if col not in df.columns
]


if missing_fields:

    raise ValueError(
        "Missing required offer-analysis fields:\n"
        + "\n".join(
            missing_fields
        )
    )


# ------------------------------------------------------------
# 7. NORMALIZE FLAGS
# ------------------------------------------------------------

df[
    "_offer_exposed"
] = normalize_binary(
    df[
        "offer_exposed_90d"
    ]
)


df[
    "_offer_redeemed"
] = normalize_binary(
    df[
        "offer_redeemed_90d"
    ]
)


df[
    "_churn30"
] = normalize_binary(
    df[
        "churn_flag_30d"
    ]
)


# ------------------------------------------------------------
# 8. BASIC DATA QUALITY
# ------------------------------------------------------------

missing_exposure = int(
    df[
        "_offer_exposed"
    ]
    .isna()
    .sum()
)


missing_redemption = int(
    df[
        "_offer_redeemed"
    ]
    .isna()
    .sum()
)


missing_churn = int(
    df[
        "_churn30"
    ]
    .isna()
    .sum()
)


redeemed_without_exposure = int(
    (
        (df["_offer_redeemed"] == 1)
        &
        (df["_offer_exposed"] == 0)
    )
    .sum()
)


print(
    "\nDATA QUALITY"
)

print(
    f"Missing exposure flags     : "
    f"{missing_exposure:,}"
)

print(
    f"Missing redemption flags   : "
    f"{missing_redemption:,}"
)

print(
    f"Missing 30d churn labels   : "
    f"{missing_churn:,}"
)

print(
    f"Redeemed without exposure  : "
    f"{redeemed_without_exposure:,}"
)


# ------------------------------------------------------------
# 9. EXPOSURE SUMMARY
# ------------------------------------------------------------

valid_exposure = df[
    df[
        "_offer_exposed"
    ].notna()
]


total_valid = len(
    valid_exposure
)


exposed_count = int(
    (
        valid_exposure[
            "_offer_exposed"
        ]
        == 1
    )
    .sum()
)


unexposed_count = int(
    (
        valid_exposure[
            "_offer_exposed"
        ]
        == 0
    )
    .sum()
)


exposure_rate = (
    exposed_count
    / total_valid
    if total_valid > 0
    else 0
)


# ------------------------------------------------------------
# 10. REDEMPTION AMONG EXPOSED
# ------------------------------------------------------------

exposed = df[
    df[
        "_offer_exposed"
    ] == 1
].copy()


exposed_valid_response = exposed[
    exposed[
        "_offer_redeemed"
    ].notna()
]


redeemed_count = int(
    (
        exposed_valid_response[
            "_offer_redeemed"
        ]
        == 1
    )
    .sum()
)


not_redeemed_count = int(
    (
        exposed_valid_response[
            "_offer_redeemed"
        ]
        == 0
    )
    .sum()
)


redemption_rate = (
    redeemed_count
    / len(
        exposed_valid_response
    )
    if len(
        exposed_valid_response
    ) > 0
    else 0
)


print(
    "\nOFFER EXPOSURE / RESPONSE"
)

print(
    f"Exposed customers          : "
    f"{exposed_count:,}"
)

print(
    f"Unexposed customers        : "
    f"{unexposed_count:,}"
)

print(
    f"Exposure rate              : "
    f"{exposure_rate * 100:.2f}%"
)

print(
    f"Redeemed among exposed     : "
    f"{redeemed_count:,}"
)

print(
    f"Did not redeem             : "
    f"{not_redeemed_count:,}"
)

print(
    f"Redemption rate            : "
    f"{redemption_rate * 100:.2f}%"
)


# ------------------------------------------------------------
# 11. EXPOSURE / REDEMPTION SUMMARY CSV
# ------------------------------------------------------------

offer_summary = pd.DataFrame(
    [
        {
            "metric":
                "Total valid subscribers",

            "value":
                total_valid,
        },

        {
            "metric":
                "Offer exposed",

            "value":
                exposed_count,
        },

        {
            "metric":
                "Offer not exposed",

            "value":
                unexposed_count,
        },

        {
            "metric":
                "Exposure rate",

            "value":
                exposure_rate,
        },

        {
            "metric":
                "Redeemed among exposed",

            "value":
                redeemed_count,
        },

        {
            "metric":
                "Not redeemed among exposed",

            "value":
                not_redeemed_count,
        },

        {
            "metric":
                "Redemption rate among exposed",

            "value":
                redemption_rate,
        },

        {
            "metric":
                "Redeemed without exposure",

            "value":
                redeemed_without_exposure,
        },
    ]
)


SUMMARY_FILE = (
    REPORT_DIR
    / "24_offer_exposure_redemption_summary.csv"
)


offer_summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ------------------------------------------------------------
# 12. DESCRIPTIVE CHURN ASSOCIATION
# ------------------------------------------------------------

association_rows = []


for exposure_value, label in [
    (0, "Not Exposed"),
    (1, "Exposed"),
]:

    subset = df[
        df[
            "_offer_exposed"
        ] == exposure_value
    ]


    subset = subset[
        subset[
            "_churn30"
        ].notna()
    ]


    customers = len(
        subset
    )


    churners = int(
        subset[
            "_churn30"
        ].sum()
    )


    churn_rate = (
        churners / customers
        if customers > 0
        else np.nan
    )


    association_rows.append(
        {
            "analysis":
                "Offer Exposure",

            "segment":
                label,

            "customers":
                customers,

            "churners":
                churners,

            "churn_rate":
                churn_rate,
        }
    )


for redemption_value, label in [
    (0, "Exposed - Not Redeemed"),
    (1, "Exposed - Redeemed"),
]:

    subset = df[
        (
            df[
                "_offer_exposed"
            ] == 1
        )
        &
        (
            df[
                "_offer_redeemed"
            ] == redemption_value
        )
    ]


    subset = subset[
        subset[
            "_churn30"
        ].notna()
    ]


    customers = len(
        subset
    )


    churners = int(
        subset[
            "_churn30"
        ].sum()
    )


    churn_rate = (
        churners / customers
        if customers > 0
        else np.nan
    )


    association_rows.append(
        {
            "analysis":
                "Offer Redemption",

            "segment":
                label,

            "customers":
                customers,

            "churners":
                churners,

            "churn_rate":
                churn_rate,
        }
    )


association_df = pd.DataFrame(
    association_rows
)


association_df[
    "churn_pct"
] = (
    association_df[
        "churn_rate"
    ]
    * 100
)


ASSOCIATION_FILE = (
    REPORT_DIR
    / "24_offer_churn_association.csv"
)


association_df.to_csv(
    ASSOCIATION_FILE,
    index=False
)


# ------------------------------------------------------------
# 13. SEARCH FOR CAUSAL / TREATMENT FIELDS
# ------------------------------------------------------------

column_names_lower = {
    col:
        col.lower()
    for col in df.columns
}


randomization_keywords = [
    "random",
    "control",
    "experiment",
    "ab_test",
    "a_b",
    "holdout",
]


timing_keywords = [
    "offer_date",
    "exposure_date",
    "assigned_date",
    "assignment_date",
    "redeem_date",
    "redemption_date",
    "treatment_date",
]


offer_link_keywords = [
    "offer_id",
    "campaign_id",
    "offer_code",
    "campaign_code",
]


randomization_fields = [
    col
    for col, lower in column_names_lower.items()
    if any(
        keyword in lower
        for keyword in randomization_keywords
    )
]


timing_fields = [
    col
    for col, lower in column_names_lower.items()
    if any(
        keyword in lower
        for keyword in timing_keywords
    )
]


subscriber_offer_link_fields = [
    col
    for col, lower in column_names_lower.items()
    if any(
        keyword in lower
        for keyword in offer_link_keywords
    )
]


# ------------------------------------------------------------
# 14. OFFER CATALOGUE AUDIT
# ------------------------------------------------------------

catalogue_exists = (
    OFFER_CATALOGUE_FILE
    is not None
)


catalogue_columns = []


if catalogue_exists:

    offer_catalogue = pd.read_csv(
        OFFER_CATALOGUE_FILE
    )

    catalogue_columns = (
        offer_catalogue
        .columns
        .tolist()
    )


    print(
        f"\nOffer catalogue found:"
    )

    print(
        f"{OFFER_CATALOGUE_FILE.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"Catalogue rows: "
        f"{len(offer_catalogue):,}"
    )


else:

    print(
        "\nOffer catalogue CSV not found."
    )


# ------------------------------------------------------------
# 15. FEASIBILITY DECISIONS
# ------------------------------------------------------------

response_model_feasible = (
    exposed_count >= 100
    and redeemed_count >= 30
    and not_redeemed_count >= 30
)


has_randomization_evidence = (
    len(
        randomization_fields
    ) > 0
)


has_treatment_timing = (
    len(
        timing_fields
    ) > 0
)


has_offer_level_link = (
    len(
        subscriber_offer_link_fields
    ) > 0
)


causal_uplift_ready = (
    has_randomization_evidence
    and has_treatment_timing
)


# ------------------------------------------------------------
# 16. FEASIBILITY METADATA
# ------------------------------------------------------------

metadata = {

    "response_model_feasible":
        bool(
            response_model_feasible
        ),

    "causal_uplift_ready":
        bool(
            causal_uplift_ready
        ),

    "exposed_customers":
        exposed_count,

    "redeemed_customers":
        redeemed_count,

    "non_redeemed_customers":
        not_redeemed_count,

    "redemption_rate":
        float(
            redemption_rate
        ),

    "randomization_fields":
        randomization_fields,

    "treatment_timing_fields":
        timing_fields,

    "subscriber_offer_link_fields":
        subscriber_offer_link_fields,

    "offer_catalogue_found":
        catalogue_exists,

    "offer_catalogue_columns":
        catalogue_columns,

    "important_limitations": [

        (
            "Exposure is observational unless "
            "random assignment is documented."
        ),

        (
            "Predictor timestamps relative to "
            "offer exposure are not established."
        ),

        (
            "Exposure/churn differences are "
            "associations, not causal effects."
        ),

        (
            "Offer-level uplift cannot be "
            "estimated without a subscriber-to-"
            "offer treatment link."
        ),
    ],
}


METADATA_FILE = (
    MODEL_DATA_DIR
    / "offer_uplift_feasibility.json"
)


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ------------------------------------------------------------
# 17. SAVE EXPOSED RESPONSE POPULATION
# ------------------------------------------------------------

# This file is saved for the NEXT step.
#
# It is NOT automatically considered model-ready,
# because timing of predictor variables relative
# to the historical offer is still ambiguous.

response_population = df[
    (
        df[
            "_offer_exposed"
        ] == 1
    )
    &
    (
        df[
            "_offer_redeemed"
        ].notna()
    )
].copy()


response_population[
    "offer_response_target"
] = (
    response_population[
        "_offer_redeemed"
    ]
    .astype(int)
)


response_population = (
    response_population
    .drop(
        columns=[
            "_offer_exposed",
            "_offer_redeemed",
            "_churn30",
        ],
        errors="ignore"
    )
)


RESPONSE_POPULATION_FILE = (
    MODEL_DATA_DIR
    / "offer_response_population.csv"
)


response_population.to_csv(
    RESPONSE_POPULATION_FILE,
    index=False
)


# ------------------------------------------------------------
# 18. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "STEP 24 - OFFER RESPONSE & UPLIFT FEASIBILITY AUDIT"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "OFFER EXPOSURE / RESPONSE"
)

report.append(
    "-" * 100
)

report.append(
    f"Subscribers: "
    f"{len(df):,}"
)

report.append(
    f"Exposed customers: "
    f"{exposed_count:,}"
)

report.append(
    f"Exposure rate: "
    f"{exposure_rate * 100:.2f}%"
)

report.append(
    f"Redeemed among exposed: "
    f"{redeemed_count:,}"
)

report.append(
    f"Redemption rate among exposed: "
    f"{redemption_rate * 100:.2f}%"
)

report.append(
    f"Redeemed without exposure: "
    f"{redeemed_without_exposure:,}"
)

report.append("")

report.append(
    "DESCRIPTIVE CHURN ASSOCIATION"
)

report.append(
    "-" * 100
)

report.append(
    association_df[
        [
            "analysis",
            "segment",
            "customers",
            "churners",
            "churn_pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)

report.append("")

report.append(
    "CAUSAL-UPLIFT AUDIT"
)

report.append(
    "-" * 100
)

report.append(
    f"Randomization/control fields: "
    f"{randomization_fields}"
)

report.append(
    f"Treatment timing fields: "
    f"{timing_fields}"
)

report.append(
    f"Subscriber-offer linkage fields: "
    f"{subscriber_offer_link_fields}"
)

report.append(
    f"Offer catalogue found: "
    f"{catalogue_exists}"
)

report.append("")

report.append(
    f"Offer-response model feasible: "
    f"{response_model_feasible}"
)

report.append(
    f"Causal uplift model ready: "
    f"{causal_uplift_ready}"
)

report.append("")

report.append(
    "INTERPRETATION"
)

report.append(
    "-" * 100
)

report.append(
    "A redemption-response propensity model "
    "may be feasible if there are sufficient "
    "redeemers and non-redeemers among exposed "
    "customers."
)

report.append(
    "However, a proper causal uplift model "
    "requires defensible treatment/control "
    "assignment and treatment timing."
)

report.append(
    "Simple churn differences between exposed "
    "and unexposed customers must not be "
    "reported as the causal effect of discounts."
)

report.append(
    "The next modelling step should be chosen "
    "from the actual feasibility results rather "
    "than forcing uplift modelling when the "
    "required experimental structure is absent."
)


REPORT_FILE = (
    REPORT_DIR
    / "24_offer_uplift_feasibility_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 19. TERMINAL OUTPUT
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "DESCRIPTIVE CHURN ASSOCIATION"
)

print(
    "=" * 90
)


print(
    association_df[
        [
            "analysis",
            "segment",
            "customers",
            "churners",
            "churn_pct",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 90
)

print(
    "UPLIFT FEASIBILITY"
)

print(
    "=" * 90
)


print(
    f"Randomization/control fields : "
    f"{randomization_fields}"
)

print(
    f"Treatment timing fields      : "
    f"{timing_fields}"
)

print(
    f"Subscriber-offer link fields : "
    f"{subscriber_offer_link_fields}"
)


print(
    f"\nResponse model feasible      : "
    f"{response_model_feasible}"
)

print(
    f"Causal uplift ready          : "
    f"{causal_uplift_ready}"
)


print(
    "\nGenerated files:"
)


for file in [
    SUMMARY_FILE,
    ASSOCIATION_FILE,
    METADATA_FILE,
    RESPONSE_POPULATION_FILE,
    REPORT_FILE,
]:

    print(
        f"- "
        f"{file.relative_to(PROJECT_ROOT)}"
    )


print(
    "\n" + "=" * 90
)

print(
    "STEP 24 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nNo causal uplift model has "
    "been trained yet."
)