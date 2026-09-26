from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 25
# OFFER RESPONSE MODEL DATA PREPARATION
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

MODEL_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_FILE = (
    MODEL_DATA_DIR
    / "offer_response_population.csv"
)


# ------------------------------------------------------------
# 2. START
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 25: OFFER RESPONSE DATA PREPARATION")
print("=" * 90)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Required file not found:\n"
        f"{INPUT_FILE}"
    )


# ------------------------------------------------------------
# 3. LOAD RESPONSE POPULATION
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nLoaded exposed population : "
    f"{len(df):,} rows"
)

print(
    f"Columns                   : "
    f"{df.shape[1]}"
)


TARGET = (
    "offer_response_target"
)


if TARGET not in df.columns:

    raise ValueError(
        f"Target column missing: "
        f"{TARGET}"
    )


# ------------------------------------------------------------
# 4. TARGET VALIDATION
# ------------------------------------------------------------

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


if df[TARGET].isna().any():

    raise ValueError(
        "Offer-response target contains "
        "missing or invalid values."
    )


df[TARGET] = (
    df[TARGET]
    .astype(int)
)


invalid_target = (
    ~df[TARGET].isin(
        [0, 1]
    )
)


if invalid_target.any():

    raise ValueError(
        "Target must contain only 0 and 1."
    )


responders = int(
    df[TARGET].sum()
)

nonresponders = int(
    len(df)
    - responders
)

response_rate = (
    responders
    / len(df)
)


print(
    "\nOFFER RESPONSE TARGET"
)

print(
    f"Responders      : "
    f"{responders:,}"
)

print(
    f"Non-responders  : "
    f"{nonresponders:,}"
)

print(
    f"Response rate   : "
    f"{response_rate * 100:.2f}%"
)


# ------------------------------------------------------------
# 5. DEFINE EXCLUSIONS
# ------------------------------------------------------------

# These fields should NOT be used as model predictors.
#
# Reasons:
# - subscriber_id: identifier
# - offer_response_target: target
# - offer_redeemed_90d: direct target duplicate
# - offer_exposed_90d: constant because everyone here
#   was exposed
# - churn labels / reason / date: future or outcome fields
# - mnp_enquiry_flag: late-stage churn behaviour and
#   timing relative to offer is uncertain
# - join_date: tenure_months already captures customer age

exclusion_reasons = {

    "subscriber_id":
        "Unique identifier",

    "offer_response_target":
        "Prediction target",

    "offer_redeemed_90d":
        "Direct target leakage",

    "offer_exposed_90d":
        "Constant in exposed-only population",

    "churn_flag_30d":
        "Outcome / future churn field",

    "churn_flag_90d":
        "Outcome / future churn field",

    "churn_reason":
        "Post-outcome field",

    "churn_date":
        "Post-outcome field",

    "mnp_enquiry_flag":
        (
            "Late-stage behaviour with uncertain "
            "timing relative to offer exposure"
        ),

    "join_date":
        (
            "Raw date excluded; tenure_months "
            "already captures tenure"
        ),
}


excluded_columns = [
    col
    for col in exclusion_reasons
    if col in df.columns
]


feature_columns = [
    col
    for col in df.columns
    if col not in excluded_columns
]


# ------------------------------------------------------------
# 6. FEATURE MATRIX
# ------------------------------------------------------------

X = df[
    feature_columns
].copy()


y = df[
    TARGET
].copy()


print(
    f"\nFeatures retained : "
    f"{len(feature_columns)}"
)

print(
    f"Columns excluded  : "
    f"{len(excluded_columns)}"
)


print(
    "\nEXCLUDED COLUMNS"
)


for col in excluded_columns:

    print(
        f"- {col}: "
        f"{exclusion_reasons[col]}"
    )


# ------------------------------------------------------------
# 7. IDENTIFY NUMERIC / CATEGORICAL FEATURES
# ------------------------------------------------------------

numeric_features = (
    X
    .select_dtypes(
        include=[
            np.number
        ]
    )
    .columns
    .tolist()
)


categorical_features = [
    col
    for col in X.columns
    if col not in numeric_features
]


print(
    f"\nNumeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)


# ------------------------------------------------------------
# 8. BASIC MISSINGNESS AUDIT
# ------------------------------------------------------------

missing_summary = (
    X
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)


missing_summary = (
    missing_summary[
        missing_summary > 0
    ]
)


print(
    "\nMISSING VALUES"
)


if len(
    missing_summary
) == 0:

    print(
        "No missing feature values."
    )

else:

    print(
        missing_summary
        .head(15)
        .to_string()
    )


# ------------------------------------------------------------
# 9. STRATIFIED 60 / 20 / 20 SPLIT
# ------------------------------------------------------------

# First:
# 60% train
# 40% temporary
#
# Then temporary:
# 20% validation
# 20% test

train_df, temp_df = train_test_split(

    df,

    test_size=0.40,

    random_state=42,

    stratify=df[
        TARGET
    ],
)


validation_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=42,

    stratify=temp_df[
        TARGET
    ],
)


# ------------------------------------------------------------
# 10. SPLIT VALIDATION FUNCTION
# ------------------------------------------------------------

def summarize_split(
    name,
    split_df
):

    rows = len(
        split_df
    )

    positive = int(
        split_df[
            TARGET
        ].sum()
    )

    negative = int(
        rows
        - positive
    )

    rate = (
        positive
        / rows
    )

    return {
        "dataset":
            name,

        "rows":
            rows,

        "responders":
            positive,

        "nonresponders":
            negative,

        "response_rate":
            rate,
    }


split_summary = pd.DataFrame(
    [
        summarize_split(
            "Train",
            train_df
        ),

        summarize_split(
            "Validation",
            validation_df
        ),

        summarize_split(
            "Test",
            test_df
        ),
    ]
)


# ------------------------------------------------------------
# 11. VERIFY NO OVERLAP
# ------------------------------------------------------------

if "subscriber_id" in df.columns:

    train_ids = set(
        train_df[
            "subscriber_id"
        ]
    )

    validation_ids = set(
        validation_df[
            "subscriber_id"
        ]
    )

    test_ids = set(
        test_df[
            "subscriber_id"
        ]
    )


    overlap_train_val = (
        train_ids
        &
        validation_ids
    )


    overlap_train_test = (
        train_ids
        &
        test_ids
    )


    overlap_val_test = (
        validation_ids
        &
        test_ids
    )


    if (
        overlap_train_val
        or overlap_train_test
        or overlap_val_test
    ):

        raise ValueError(
            "Subscriber overlap detected "
            "between modelling splits."
        )


    print(
        "\n[OK] No subscriber overlap "
        "between train, validation and test."
    )


# ------------------------------------------------------------
# 12. SAVE SPLITS
# ------------------------------------------------------------

TRAIN_FILE = (
    MODEL_DATA_DIR
    / "offer_response_train_60.csv"
)

VALIDATION_FILE = (
    MODEL_DATA_DIR
    / "offer_response_validation_20.csv"
)

TEST_FILE = (
    MODEL_DATA_DIR
    / "offer_response_test_20.csv"
)


train_df.to_csv(
    TRAIN_FILE,
    index=False
)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ------------------------------------------------------------
# 13. SAVE SPLIT SUMMARY
# ------------------------------------------------------------

SPLIT_SUMMARY_FILE = (
    REPORT_DIR
    / "25_offer_response_split_summary.csv"
)


split_summary.to_csv(
    SPLIT_SUMMARY_FILE,
    index=False
)


# ------------------------------------------------------------
# 14. SAVE METADATA
# ------------------------------------------------------------

metadata = {

    "target":
        TARGET,

    "population":
        (
            "Subscribers with offer_exposed_90d = 1 "
            "and valid redemption outcome"
        ),

    "rows":
        int(
            len(df)
        ),

    "responders":
        responders,

    "nonresponders":
        nonresponders,

    "response_rate":
        float(
            response_rate
        ),

    "feature_columns":
        feature_columns,

    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    "excluded_columns":
        excluded_columns,

    "exclusion_reasons":
        {
            col:
                exclusion_reasons[
                    col
                ]
            for col in excluded_columns
        },

    "split":
        {
            "train":
                0.60,

            "validation":
                0.20,

            "test":
                0.20,

            "random_state":
                42,

            "stratified":
                True,
        },

    "important_limitations": [

        (
            "Offer exposure was not documented "
            "as randomized."
        ),

        (
            "Predictor timing relative to offer "
            "exposure is not available."
        ),

        (
            "The model should be described as an "
            "offer-redemption propensity model, "
            "not a causal uplift model."
        ),

        (
            "The offer catalogue cannot be linked "
            "to individual subscribers because "
            "no subscriber-level offer ID exists."
        ),
    ],
}


METADATA_FILE = (
    MODEL_DATA_DIR
    / "offer_response_metadata.json"
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
# 15. HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "STEP 25 - OFFER RESPONSE MODEL DATA PREPARATION"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "MODELLING OBJECTIVE"
)

report.append(
    "-" * 100
)

report.append(
    "Predict whether an offer-exposed customer "
    "will redeem the offer."
)

report.append("")

report.append(
    "This is a response-propensity model."
)

report.append(
    "It is NOT a causal uplift model."
)

report.append("")

report.append(
    "TARGET DISTRIBUTION"
)

report.append(
    "-" * 100
)

report.append(
    f"Responders: "
    f"{responders:,}"
)

report.append(
    f"Non-responders: "
    f"{nonresponders:,}"
)

report.append(
    f"Response rate: "
    f"{response_rate * 100:.2f}%"
)

report.append("")

report.append(
    "SPLIT SUMMARY"
)

report.append(
    "-" * 100
)

report.append(
    split_summary
    .round(4)
    .to_string(
        index=False
    )
)

report.append("")

report.append(
    "FEATURES"
)

report.append(
    "-" * 100
)

report.append(
    f"Features retained: "
    f"{len(feature_columns)}"
)

report.append(
    f"Numeric features: "
    f"{len(numeric_features)}"
)

report.append(
    f"Categorical features: "
    f"{len(categorical_features)}"
)

report.append("")

report.append(
    "EXCLUSIONS"
)

report.append(
    "-" * 100
)


for col in excluded_columns:

    report.append(
        f"{col}: "
        f"{exclusion_reasons[col]}"
    )


report.append("")

report.append(
    "IMPORTANT LIMITATION"
)

report.append(
    "-" * 100
)

report.append(
    "The dataset does not provide timestamps "
    "showing whether every predictor was measured "
    "before the historical offer exposure."
)

report.append(
    "Therefore the response model is suitable "
    "for sandbox analysis and propensity "
    "demonstration, but not yet a production "
    "causal targeting model."
)


REPORT_FILE = (
    REPORT_DIR
    / "25_offer_response_data_preparation_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 16. TERMINAL OUTPUT
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "60 / 20 / 20 OFFER RESPONSE SPLIT"
)

print(
    "=" * 90
)


display_summary = (
    split_summary
    .copy()
)


display_summary[
    "response_rate_pct"
] = (
    display_summary[
        "response_rate"
    ]
    * 100
)


print(
    display_summary[
        [
            "dataset",
            "rows",
            "responders",
            "nonresponders",
            "response_rate_pct",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


print(
    "\nGenerated files:"
)


for file in [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    METADATA_FILE,
    SPLIT_SUMMARY_FILE,
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
    "STEP 25 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nOffer-response TEST set remains untouched."
)