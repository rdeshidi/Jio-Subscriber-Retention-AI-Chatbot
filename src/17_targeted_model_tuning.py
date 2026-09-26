from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ============================================================
# JIO SUBSCRIBER RETENTION PROJECT
# DAY 2 - STEP 17
# TARGETED MODEL TUNING
# XGBOOST + CATBOOST
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DATA_DIR = PROJECT_ROOT / "data" / "modeling"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = MODEL_DATA_DIR / "train_60.csv"
VALID_FILE = MODEL_DATA_DIR / "validation_20.csv"
METADATA_FILE = MODEL_DATA_DIR / "modeling_metadata.json"


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("=" * 90)
print("JIO SUBSCRIBER RETENTION PROJECT")
print("DAY 2 - STEP 17: TARGETED MODEL TUNING")
print("=" * 90)

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)


TARGET = metadata["target"]

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]


X_train = train_df.drop(
    columns=[TARGET]
).copy()

y_train = train_df[
    TARGET
].astype(int)


X_valid = valid_df.drop(
    columns=[TARGET]
).copy()

y_valid = valid_df[
    TARGET
].astype(int)


print(
    f"\nTraining rows   : {len(X_train):,}"
)

print(
    f"Validation rows : {len(X_valid):,}"
)


# ------------------------------------------------------------
# 3. CLASS IMBALANCE
# ------------------------------------------------------------

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

imbalance_ratio = (
    negative_count
    / positive_count
)

half_weight = (
    imbalance_ratio / 2
)

sqrt_weight = (
    np.sqrt(
        imbalance_ratio
    )
)


print(
    f"\nClass imbalance ratio : "
    f"{imbalance_ratio:.2f}"
)

print(
    f"Half weight            : "
    f"{half_weight:.2f}"
)

print(
    f"Square-root weight     : "
    f"{sqrt_weight:.2f}"
)


# ------------------------------------------------------------
# 4. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(
    y_true,
    predictions,
    probabilities
):

    return {

        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),
    }


# ------------------------------------------------------------
# 5. TOP-10% BUSINESS METRICS
# ------------------------------------------------------------

def calculate_top10(
    y_true,
    probabilities
):

    scores = pd.DataFrame(
        {
            "actual":
                np.asarray(
                    y_true
                ),

            "probability":
                probabilities,
        }
    )

    scores = (
        scores
        .sort_values(
            "probability",
            ascending=False
        )
        .reset_index(drop=True)
    )


    top_n = int(
        np.ceil(
            len(scores)
            * 0.10
        )
    )

    top = scores.head(
        top_n
    )


    total_churners = int(
        scores[
            "actual"
        ].sum()
    )

    top_churners = int(
        top[
            "actual"
        ].sum()
    )


    recall_top10 = (
        top_churners
        / total_churners
        if total_churners > 0
        else 0
    )


    overall_rate = (
        scores[
            "actual"
        ].mean()
    )

    top_rate = (
        top[
            "actual"
        ].mean()
    )


    lift_top10 = (
        top_rate
        / overall_rate
        if overall_rate > 0
        else 0
    )


    return (
        recall_top10,
        lift_top10
    )


# ============================================================
# PART A - XGBOOST TUNING
# ============================================================


# ------------------------------------------------------------
# 6. XGBOOST PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)


xgb_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


print(
    "\nPreparing XGBoost features..."
)


X_train_xgb = (
    xgb_preprocessor
    .fit_transform(
        X_train
    )
)

X_valid_xgb = (
    xgb_preprocessor
    .transform(
        X_valid
    )
)


print(
    "[OK] XGBoost preprocessing complete."
)


# ------------------------------------------------------------
# 7. XGBOOST CONFIGURATIONS
# ------------------------------------------------------------

xgb_configs = [

    {
        "name":
            "XGB_Regularized_FullWeight",

        "params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 10,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "gamma": 0.10,
            "reg_alpha": 0.50,
            "reg_lambda": 5.0,
            "scale_pos_weight":
                imbalance_ratio,
        },
    },

    {
        "name":
            "XGB_StrongRegularization_HalfWeight",

        "params": {
            "n_estimators": 250,
            "learning_rate": 0.04,
            "max_depth": 2,
            "min_child_weight": 15,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "gamma": 0.20,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
            "scale_pos_weight":
                half_weight,
        },
    },

    {
        "name":
            "XGB_Conservative_SqrtWeight",

        "params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 12,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "gamma": 0.10,
            "reg_alpha": 0.75,
            "reg_lambda": 8.0,
            "scale_pos_weight":
                sqrt_weight,
        },
    },
]


# ------------------------------------------------------------
# 8. TRAIN XGBOOST VARIANTS
# ------------------------------------------------------------

results = []

xgb_models = {}


print(
    "\n" + "=" * 90
)

print(
    "XGBOOST TUNING"
)

print(
    "=" * 90
)


for config in xgb_configs:

    print(
        f"\nTraining "
        f"{config['name']}..."
    )


    model = XGBClassifier(

        **config["params"],

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        n_jobs=-1,
    )


    model.fit(
        X_train_xgb,
        y_train
    )


    train_prob = (
        model.predict_proba(
            X_train_xgb
        )[:, 1]
    )

    valid_prob = (
        model.predict_proba(
            X_valid_xgb
        )[:, 1]
    )


    train_pred = (
        train_prob >= 0.50
    ).astype(int)

    valid_pred = (
        valid_prob >= 0.50
    ).astype(int)


    train_metrics = (
        calculate_metrics(
            y_train,
            train_pred,
            train_prob
        )
    )


    valid_metrics = (
        calculate_metrics(
            y_valid,
            valid_pred,
            valid_prob
        )
    )


    recall10, lift10 = (
        calculate_top10(
            y_valid,
            valid_prob
        )
    )


    results.append(
        {
            "family":
                "XGBoost",

            "configuration":
                config["name"],

            "validation_accuracy":
                valid_metrics[
                    "accuracy"
                ],

            "validation_precision":
                valid_metrics[
                    "precision"
                ],

            "validation_recall":
                valid_metrics[
                    "recall"
                ],

            "validation_f1":
                valid_metrics[
                    "f1"
                ],

            "validation_roc_auc":
                valid_metrics[
                    "roc_auc"
                ],

            "validation_pr_auc":
                valid_metrics[
                    "pr_auc"
                ],

            "recall_at_top10":
                recall10,

            "lift_at_top10":
                lift10,

            "roc_auc_gap":
                (
                    train_metrics[
                        "roc_auc"
                    ]
                    -
                    valid_metrics[
                        "roc_auc"
                    ]
                ),

            "pr_auc_gap":
                (
                    train_metrics[
                        "pr_auc"
                    ]
                    -
                    valid_metrics[
                        "pr_auc"
                    ]
                ),

            "train_roc_auc":
                train_metrics[
                    "roc_auc"
                ],

            "train_pr_auc":
                train_metrics[
                    "pr_auc"
                ],
        }
    )


    xgb_models[
        config["name"]
    ] = model


    print(
        f"Validation PR-AUC : "
        f"{valid_metrics['pr_auc']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{valid_metrics['roc_auc']:.4f}"
    )

    print(
        f"ROC-AUC gap       : "
        f"{train_metrics['roc_auc'] - valid_metrics['roc_auc']:.4f}"
    )


# ============================================================
# PART B - CATBOOST TUNING
# ============================================================


# ------------------------------------------------------------
# 9. PREPARE CATBOOST DATA
# ------------------------------------------------------------

X_train_cat = X_train.copy()
X_valid_cat = X_valid.copy()


for col in categorical_features:

    X_train_cat[col] = (
        X_train_cat[col]
        .fillna("Missing")
        .astype(str)
    )

    X_valid_cat[col] = (
        X_valid_cat[col]
        .fillna("Missing")
        .astype(str)
    )


numeric_medians = {}


for col in numeric_features:

    X_train_cat[col] = pd.to_numeric(
        X_train_cat[col],
        errors="coerce"
    )

    X_valid_cat[col] = pd.to_numeric(
        X_valid_cat[col],
        errors="coerce"
    )


    median_value = (
        X_train_cat[
            col
        ].median()
    )


    numeric_medians[col] = (
        float(median_value)
        if pd.notna(
            median_value
        )
        else 0.0
    )


    X_train_cat[col] = (
        X_train_cat[col]
        .fillna(
            numeric_medians[
                col
            ]
        )
    )


    X_valid_cat[col] = (
        X_valid_cat[col]
        .fillna(
            numeric_medians[
                col
            ]
        )
    )


# ------------------------------------------------------------
# 10. CATBOOST CONFIGURATIONS
# ------------------------------------------------------------

cat_configs = [

    {
        "name":
            "CAT_Regularized_Balanced",

        "params": {
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 5,
            "l2_leaf_reg": 5,
            "random_strength": 1.0,
            "auto_class_weights":
                "Balanced",
        },
    },

    {
        "name":
            "CAT_StrongReg_HalfWeight",

        "params": {
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 4,
            "l2_leaf_reg": 8,
            "random_strength": 1.5,
            "class_weights":
                [
                    1.0,
                    half_weight
                ],
        },
    },

    {
        "name":
            "CAT_Conservative_SqrtWeight",

        "params": {
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 5,
            "l2_leaf_reg": 10,
            "random_strength": 2.0,
            "class_weights":
                [
                    1.0,
                    sqrt_weight
                ],
        },
    },
]


# ------------------------------------------------------------
# 11. TRAIN CATBOOST VARIANTS
# ------------------------------------------------------------

cat_models = {}


print(
    "\n" + "=" * 90
)

print(
    "CATBOOST TUNING"
)

print(
    "=" * 90
)


for config in cat_configs:

    print(
        f"\nTraining "
        f"{config['name']}..."
    )


    model = CatBoostClassifier(

        **config["params"],

        loss_function="Logloss",

        eval_metric="AUC",

        random_seed=42,

        verbose=False,

        allow_writing_files=False,
    )


    model.fit(

        X_train_cat,

        y_train,

        cat_features=
            categorical_features,

        eval_set=(
            X_valid_cat,
            y_valid
        ),

        use_best_model=True,

        early_stopping_rounds=75,
    )


    train_prob = (
        model.predict_proba(
            X_train_cat
        )[:, 1]
    )

    valid_prob = (
        model.predict_proba(
            X_valid_cat
        )[:, 1]
    )


    train_pred = (
        train_prob >= 0.50
    ).astype(int)

    valid_pred = (
        valid_prob >= 0.50
    ).astype(int)


    train_metrics = (
        calculate_metrics(
            y_train,
            train_pred,
            train_prob
        )
    )


    valid_metrics = (
        calculate_metrics(
            y_valid,
            valid_pred,
            valid_prob
        )
    )


    recall10, lift10 = (
        calculate_top10(
            y_valid,
            valid_prob
        )
    )


    results.append(
        {
            "family":
                "CatBoost",

            "configuration":
                config["name"],

            "validation_accuracy":
                valid_metrics[
                    "accuracy"
                ],

            "validation_precision":
                valid_metrics[
                    "precision"
                ],

            "validation_recall":
                valid_metrics[
                    "recall"
                ],

            "validation_f1":
                valid_metrics[
                    "f1"
                ],

            "validation_roc_auc":
                valid_metrics[
                    "roc_auc"
                ],

            "validation_pr_auc":
                valid_metrics[
                    "pr_auc"
                ],

            "recall_at_top10":
                recall10,

            "lift_at_top10":
                lift10,

            "roc_auc_gap":
                (
                    train_metrics[
                        "roc_auc"
                    ]
                    -
                    valid_metrics[
                        "roc_auc"
                    ]
                ),

            "pr_auc_gap":
                (
                    train_metrics[
                        "pr_auc"
                    ]
                    -
                    valid_metrics[
                        "pr_auc"
                    ]
                ),

            "train_roc_auc":
                train_metrics[
                    "roc_auc"
                ],

            "train_pr_auc":
                train_metrics[
                    "pr_auc"
                ],

            "best_iteration":
                model.get_best_iteration(),
        }
    )


    cat_models[
        config["name"]
    ] = model


    print(
        f"Best iteration     : "
        f"{model.get_best_iteration()}"
    )

    print(
        f"Validation PR-AUC  : "
        f"{valid_metrics['pr_auc']:.4f}"
    )

    print(
        f"Validation ROC-AUC : "
        f"{valid_metrics['roc_auc']:.4f}"
    )

    print(
        f"ROC-AUC gap        : "
        f"{train_metrics['roc_auc'] - valid_metrics['roc_auc']:.4f}"
    )


# ------------------------------------------------------------
# 12. RESULTS TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        [
            "validation_pr_auc",
            "validation_roc_auc",
            "roc_auc_gap",
        ],

        ascending=[
            False,
            False,
            True,
        ]
    )
    .reset_index(drop=True)
)


numeric_cols = (
    results_df
    .select_dtypes(
        include="number"
    )
    .columns
)


results_df[
    numeric_cols
] = (
    results_df[
        numeric_cols
    ]
    .round(4)
)


# ------------------------------------------------------------
# 13. SELECT BEST XGBOOST
# ------------------------------------------------------------

best_xgb_row = (
    results_df[
        results_df[
            "family"
        ] == "XGBoost"
    ]
    .iloc[0]
)


best_xgb_name = (
    best_xgb_row[
        "configuration"
    ]
)


best_xgb_model = (
    xgb_models[
        best_xgb_name
    ]
)


# ------------------------------------------------------------
# 14. SELECT BEST CATBOOST
# ------------------------------------------------------------

best_cat_row = (
    results_df[
        results_df[
            "family"
        ] == "CatBoost"
    ]
    .iloc[0]
)


best_cat_name = (
    best_cat_row[
        "configuration"
    ]
)


best_cat_model = (
    cat_models[
        best_cat_name
    ]
)


# ------------------------------------------------------------
# 15. SAVE BEST XGBOOST PACKAGE
# ------------------------------------------------------------

XGB_MODEL_FILE = (
    MODEL_DIR
    / "tuned_xgboost_best.pkl"
)


joblib.dump(
    {
        "model":
            best_xgb_model,

        "preprocessor":
            xgb_preprocessor,

        "configuration":
            best_xgb_name,

        "target":
            TARGET,
    },

    XGB_MODEL_FILE
)


# ------------------------------------------------------------
# 16. SAVE BEST CATBOOST PACKAGE
# ------------------------------------------------------------

CAT_MODEL_FILE = (
    MODEL_DIR
    / "tuned_catboost_best.pkl"
)


joblib.dump(
    {
        "model":
            best_cat_model,

        "configuration":
            best_cat_name,

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "numeric_medians":
            numeric_medians,

        "target":
            TARGET,
    },

    CAT_MODEL_FILE
)


# ------------------------------------------------------------
# 17. SAVE RESULTS CSV
# ------------------------------------------------------------

RESULTS_FILE = (
    REPORT_DIR
    / "17_targeted_model_tuning_results.csv"
)


results_df.to_csv(
    RESULTS_FILE,
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
    "DAY 2 - STEP 17 TARGETED MODEL TUNING"
)

report.append(
    "=" * 100
)

report.append("")

report.append(
    "PURPOSE"
)

report.append(
    "-" * 100
)

report.append(
    "Targeted tuning was performed on XGBoost "
    "and CatBoost because the initial comparison "
    "showed stronger train-validation gaps than "
    "Logistic Regression."
)

report.append("")

report.append(
    "Validation PR-AUC was used as the primary "
    "selection metric because 30-day churn is "
    "highly imbalanced."
)

report.append("")

report.append(
    "ROC-AUC, Recall@Top10%, Lift and "
    "train-validation gaps were retained as "
    "supporting diagnostics."
)

report.append("")

report.append(
    "TUNING RESULTS"
)

report.append(
    "-" * 100
)

report.append(
    results_df.to_string(
        index=False
    )
)

report.append("")

report.append(
    "BEST TUNED XGBOOST"
)

report.append(
    "-" * 100
)

report.append(
    f"Configuration: "
    f"{best_xgb_name}"
)

report.append(
    f"Validation PR-AUC: "
    f"{best_xgb_row['validation_pr_auc']:.4f}"
)

report.append(
    f"Validation ROC-AUC: "
    f"{best_xgb_row['validation_roc_auc']:.4f}"
)

report.append(
    f"Recall@Top10%: "
    f"{best_xgb_row['recall_at_top10']:.4f}"
)

report.append("")

report.append(
    "BEST TUNED CATBOOST"
)

report.append(
    "-" * 100
)

report.append(
    f"Configuration: "
    f"{best_cat_name}"
)

report.append(
    f"Validation PR-AUC: "
    f"{best_cat_row['validation_pr_auc']:.4f}"
)

report.append(
    f"Validation ROC-AUC: "
    f"{best_cat_row['validation_roc_auc']:.4f}"
)

report.append(
    f"Recall@Top10%: "
    f"{best_cat_row['recall_at_top10']:.4f}"
)

report.append("")

report.append(
    "IMPORTANT"
)

report.append(
    "-" * 100
)

report.append(
    "No test data was used during tuning."
)

report.append(
    "The final 20% test set remains untouched."
)

report.append(
    "Threshold optimization for the tuned models "
    "should be performed only after this tuning "
    "stage."
)


REPORT_FILE = (
    REPORT_DIR
    / "17_targeted_model_tuning_report.txt"
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
    "TARGETED TUNING RESULTS"
)

print(
    "=" * 90
)


display_columns = [
    "family",
    "configuration",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_roc_auc",
    "validation_pr_auc",
    "recall_at_top10",
    "lift_at_top10",
    "roc_auc_gap",
]


print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


print(
    "\nBEST TUNED XGBOOST"
)

print(
    f"Configuration : "
    f"{best_xgb_name}"
)

print(
    f"PR-AUC        : "
    f"{best_xgb_row['validation_pr_auc']:.4f}"
)

print(
    f"ROC-AUC       : "
    f"{best_xgb_row['validation_roc_auc']:.4f}"
)


print(
    "\nBEST TUNED CATBOOST"
)

print(
    f"Configuration : "
    f"{best_cat_name}"
)

print(
    f"PR-AUC        : "
    f"{best_cat_row['validation_pr_auc']:.4f}"
)

print(
    f"ROC-AUC       : "
    f"{best_cat_row['validation_roc_auc']:.4f}"
)


print(
    "\nGenerated files:"
)

print(
    f"- "
    f"{RESULTS_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{REPORT_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{XGB_MODEL_FILE.relative_to(PROJECT_ROOT)}"
)

print(
    f"- "
    f"{CAT_MODEL_FILE.relative_to(PROJECT_ROOT)}"
)


print(
    "\n" + "=" * 90
)

print(
    "STEP 17 COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nThe TEST set remains untouched."
)