from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 10
# MODELLING DATA PREPARATION
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "subscribers_clean.csv"
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

MODEL_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------

TARGET = "churn_flag_30d"

RANDOM_STATE = 42

# 60% train / 20% validation / 20% test
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 10: MODELLING DATA PREPARATION")
print("=" * 90)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nLoaded dataset: "
    f"{df.shape[0]:,} rows x "
    f"{df.shape[1]} columns"
)


# ------------------------------------------------------------
# 4. CONVERT TARGET TO 0 / 1
# ------------------------------------------------------------

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


def convert_binary_target(series):

    if series.dtype == bool:
        return series.astype(int)

    converted = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
            }
        )
    )

    return converted


df[TARGET] = convert_binary_target(
    df[TARGET]
)


if df[TARGET].isna().any():

    bad_count = int(
        df[TARGET]
        .isna()
        .sum()
    )

    raise ValueError(
        f"{bad_count} target values could "
        f"not be converted to 0/1."
    )


df[TARGET] = df[TARGET].astype(int)


# ------------------------------------------------------------
# 5. DEFINE EXCLUDED COLUMNS
# ------------------------------------------------------------

# These fields must NOT be used for the 30-day churn model.

excluded_columns = {

    # Identifier
    "subscriber_id":
        "Unique customer identifier; no predictive meaning.",

    # Direct leakage / post-outcome information
    "mnp_enquiry_flag":
        "Strong target-leakage risk and occurs too late "
        "for the intended early-retention use case.",

    "churn_reason":
        "Only known after churn / outcome.",

    "churn_date":
        "Direct future outcome information.",

    # Future target
    "churn_flag_90d":
        "Future churn outcome and therefore leakage "
        "for a 30-day churn prediction model.",

    # Raw join date
    "join_date":
        "Tenure_months already represents customer age "
        "at snapshot and is safer for the baseline model.",
}


# ------------------------------------------------------------
# 6. CHECK EXCLUDED COLUMNS
# ------------------------------------------------------------

existing_exclusions = [
    col
    for col in excluded_columns
    if col in df.columns
]


missing_exclusions = [
    col
    for col in excluded_columns
    if col not in df.columns
]


print("\nColumns excluded from modelling:")

for col in existing_exclusions:
    print(
        f"- {col}: "
        f"{excluded_columns[col]}"
    )


# ------------------------------------------------------------
# 7. CREATE FEATURE MATRIX
# ------------------------------------------------------------

X = df.drop(
    columns=[
        TARGET,
        *existing_exclusions,
    ]
)

y = df[TARGET].copy()


print(
    f"\nFeatures retained: "
    f"{X.shape[1]}"
)

print(
    f"Target: {TARGET}"
)


# ------------------------------------------------------------
# 8. BASIC FEATURE TYPE REVIEW
# ------------------------------------------------------------

numeric_columns = (
    X.select_dtypes(
        include="number"
    )
    .columns
    .tolist()
)

categorical_columns = [
    col
    for col in X.columns
    if col not in numeric_columns
]


print(
    f"\nNumeric features     : "
    f"{len(numeric_columns)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_columns)}"
)


# ------------------------------------------------------------
# 9. MISSING VALUE PROFILE
# ------------------------------------------------------------

missing_profile = (
    X.isna()
    .sum()
    .to_frame(
        "missing_count"
    )
)

missing_profile[
    "missing_pct"
] = (
    missing_profile[
        "missing_count"
    ]
    / len(X)
    * 100
).round(2)


missing_profile = (
    missing_profile[
        missing_profile[
            "missing_count"
        ] > 0
    ]
    .sort_values(
        "missing_pct",
        ascending=False
    )
)


MISSING_FILE = (
    REPORT_DIR
    / "10_model_feature_missingness.csv"
)

missing_profile.to_csv(
    MISSING_FILE
)


# ------------------------------------------------------------
# 10. TARGET DISTRIBUTION
# ------------------------------------------------------------

target_summary = (
    y.value_counts()
    .sort_index()
    .to_frame(
        "count"
    )
)

target_summary[
    "percentage"
] = (
    target_summary[
        "count"
    ]
    / len(y)
    * 100
).round(2)


print("\nTarget distribution:")

print(
    target_summary.to_string()
)


# ------------------------------------------------------------
# 11. FIRST SPLIT
# 60% TRAIN / 40% TEMP
# ------------------------------------------------------------

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)


# ------------------------------------------------------------
# 12. SECOND SPLIT
# 20% VALIDATION / 20% TEST
# ------------------------------------------------------------

X_valid, X_test, y_valid, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
)


# ------------------------------------------------------------
# 13. COMBINE FEATURES + TARGET
# ------------------------------------------------------------

train_df = X_train.copy()
train_df[TARGET] = y_train

valid_df = X_valid.copy()
valid_df[TARGET] = y_valid

test_df = X_test.copy()
test_df[TARGET] = y_test


# ------------------------------------------------------------
# 14. SORT INDEX FOR CLEAN OUTPUT
# ------------------------------------------------------------

train_df = train_df.sort_index()
valid_df = valid_df.sort_index()
test_df = test_df.sort_index()


# ------------------------------------------------------------
# 15. SAVE SPLITS
# ------------------------------------------------------------

TRAIN_FILE = (
    MODEL_DATA_DIR
    / "train_60.csv"
)

VALID_FILE = (
    MODEL_DATA_DIR
    / "validation_20.csv"
)

TEST_FILE = (
    MODEL_DATA_DIR
    / "test_20.csv"
)


train_df.to_csv(
    TRAIN_FILE,
    index=False
)

valid_df.to_csv(
    VALID_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ------------------------------------------------------------
# 16. SPLIT VALIDATION
# ------------------------------------------------------------

def split_summary(
    name,
    split_df
):

    positive = int(
        split_df[TARGET]
        .sum()
    )

    total = len(
        split_df
    )

    rate = (
        positive
        / total
        * 100
    )

    return {
        "split": name,
        "rows": total,
        "churners_30d": positive,
        "churn_rate_pct": round(
            rate,
            2
        ),
    }


split_summary_df = pd.DataFrame(
    [
        split_summary(
            "Train",
            train_df
        ),

        split_summary(
            "Validation",
            valid_df
        ),

        split_summary(
            "Test",
            test_df
        ),
    ]
)


SPLIT_SUMMARY_FILE = (
    REPORT_DIR
    / "10_split_summary.csv"
)

split_summary_df.to_csv(
    SPLIT_SUMMARY_FILE,
    index=False
)


# ------------------------------------------------------------
# 17. SAVE FEATURE LISTS
# ------------------------------------------------------------

metadata = {

    "target":
        TARGET,

    "random_state":
        RANDOM_STATE,

    "split":
        {
            "train": TRAIN_SIZE,
            "validation":
                VALIDATION_SIZE,
            "test": TEST_SIZE,
        },

    "excluded_columns":
        excluded_columns,

    "numeric_features":
        numeric_columns,

    "categorical_features":
        categorical_columns,

    "feature_count":
        len(X.columns),
}


METADATA_FILE = (
    MODEL_DATA_DIR
    / "modeling_metadata.json"
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
# 18. CREATE HUMAN-READABLE REPORT
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "JIO SUBSCRIBER RETENTION PROJECT"
)

report.append(
    "DAY 2 - STEP 10 "
    "MODELLING DATA PREPARATION"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    f"Original rows: "
    f"{len(df):,}"
)

report.append(
    f"Original columns: "
    f"{len(df.columns)}"
)

report.append(
    f"Target: {TARGET}"
)

report.append(
    f"Features retained: "
    f"{len(X.columns)}"
)

report.append(
    f"Numeric features: "
    f"{len(numeric_columns)}"
)

report.append(
    f"Categorical features: "
    f"{len(categorical_columns)}"
)

report.append("")

report.append(
    "EXCLUDED FROM MODEL"
)

report.append(
    "-" * 100
)

for col in existing_exclusions:

    report.append(
        f"{col}: "
        f"{excluded_columns[col]}"
    )


report.append("")

report.append(
    "TARGET DISTRIBUTION"
)

report.append(
    "-" * 100
)

report.append(
    target_summary.to_string()
)


report.append("")

report.append(
    "TRAIN / VALIDATION / TEST"
)

report.append(
    "-" * 100
)

report.append(
    split_summary_df.to_string(
        index=False
    )
)


report.append("")

report.append(
    "IMPORTANT MODELLING NOTE"
)

report.append(
    "-" * 100
)

report.append(
    "No imputation, one-hot encoding, scaling, "
    "class weighting or resampling has been "
    "performed in Step 10."
)

report.append(
    "Those transformations will be placed inside "
    "the modelling pipeline so they are fitted "
    "only on training data and do not leak "
    "validation/test information."
)


REPORT_FILE = (
    REPORT_DIR
    / "10_model_data_preparation_report.txt"
)

REPORT_FILE.write_text(
    "\n".join(
        report
    ),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 19. FINAL TERMINAL OUTPUT
# ------------------------------------------------------------

print(
    "\n" + "=" * 90
)

print(
    "TRAIN / VALIDATION / TEST SPLIT"
)

print(
    "=" * 90
)

print(
    split_summary_df.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 10 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nGenerated files:"
)

print(
    f"- "
    f"{TRAIN_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{VALID_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{TEST_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{METADATA_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{SPLIT_SUMMARY_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{MISSING_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{REPORT_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    "\nNo model has been trained yet."
)