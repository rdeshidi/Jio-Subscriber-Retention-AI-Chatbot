# Jio Subscriber Retention & AI Chatbot

## Project Overview

This capstone project analyzes subscriber churn and retention opportunities using data analysis, SQL, machine learning, customer value prioritization, offer-response analysis, and a local LLM-powered conversational analytics assistant.

The project covers an end-to-end analytics workflow:

- Exploratory data analysis
- Data quality validation and cleaning
- SQL database implementation
- 30-day churn prediction
- Model comparison and threshold optimization
- CLV/value proxy analysis
- Risk x value prioritization
- Offer-response feasibility analysis
- Privacy-safe text-to-SQL chatbot
- Local LLM integration using Ollama + Qwen
- Read-only SQL execution
- Answer verification
- Audit logging
- Streamlit chatbot interface

---

## Project Architecture

```text
Employee
   |
   v
Streamlit Chatbot
   |
   v
Natural-Language Privacy Filter
   |
   v
Qwen 2.5 3B (Ollama, Local)
   |
   v
SQL Generation
   |
   v
SQL Validation + Table Allow-List
   |
   v
Read-Only MySQL User
   |
   v
Privacy-Safe Result
   |
   v
Verified Answer Synthesis
   |
   v
Audit Log
```

---

## Main Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- CatBoost
- MySQL
- SQLAlchemy
- PyMySQL
- Streamlit
- Ollama
- Qwen 2.5 3B

---

## Project Structure

```text
bootcamp_proj_05_Jio_Retention/
|
├── architecture/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── modeling/
│   └── clv/
├── models/
├── outputs/
│   ├── charts/
│   ├── reports/
│   └── sql_exports/
├── sql/
├── src/
│   ├── 01_data_exploration.py
│   ├── 02_data_quality_validation.py
│   ├── 03_data_cleaning.py
│   ├── 04_business_analysis.py
│   ├── 05_visualizations.py
│   ├── 06_prepare_sql_exports.py
│   ├── 08_generate_architecture.py
│   ├── 09_finalize_day1.py
│   ├── 10_model_data_preparation.py
│   ├── 11_logistic_regression_baseline.py
│   ├── 12_gradient_boosting_model.py
│   ├── 13_xgboost_model.py
│   ├── 14_catboost_model.py
│   ├── 15_model_comparison.py
│   ├── 16_threshold_optimization.py
│   ├── 17_targeted_model_tuning.py
│   ├── 18_final_validation_comparison.py
│   ├── 19_final_test_evaluation.py
│   ├── 20_final_churn_visualizations.py
│   ├── 21_clv_data_preparation.py
│   ├── 22_risk_value_prioritization.py
│   ├── 23_test_risk_value_validation.py
│   ├── 24_offer_uplift_feasibility.py
│   ├── 25_offer_response_data_preparation.py
│   ├── 26_offer_response_logistic_baseline.py
│   ├── 27_offer_response_catboost.py
│   ├── 28_offer_response_final_summary.py
│   ├── 29_chatbot_privacy_and_sql_safety.py
│   ├── 30_chatbot_database_setup.py
│   ├── 31_chatbot_readonly_query_engine.py
│   ├── 32_local_llm_sql_agent.py
│   ├── 33_chatbot_audit_logging.py
│   ├── 34_answer_synthesis.py
│   ├── 35_jio_chatbot_app.py
│   └── 36_final_project_validation.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Environment Setup

### 1. Create a Python virtual environment

```powershell
python -m venv jio_env
```

Activate it:

```powershell
.\jio_env\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure database connection

Copy `.env.example` to `.env`, then add the local MySQL password for the read-only chatbot database user.

The `.env` file is intentionally excluded from GitHub.

---

## Database

Database:

```text
jio_retention_db
```

Chatbot database user:

```text
jio_chatbot
```

The chatbot uses a read-only database account.

SQL scripts:

```text
sql/
├── 01_create_tables.sql
├── 02_import_data.sql
└── 03_validation_queries.sql
```

---

## Churn Modeling

The primary modeling target is:

```text
churn_flag_30d
```

The workflow includes:

- Logistic Regression baseline
- Gradient Boosting
- XGBoost
- CatBoost
- Model comparison
- Threshold optimization
- Targeted tuning
- Final validation
- Untouched test-set evaluation

Evaluation emphasizes:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Recall at top decile
- Lift at top decile

---

## Customer Value and Prioritization

The project includes a transparent annualized revenue/value proxy based on subscriber ARPU and combines risk with customer value for retention prioritization.

Outputs include:

```text
data/clv/clv_value_prepared.csv
data/clv/customer_risk_value_priority.csv
```

---

## Offer Response Analysis

Historical offer-response data is evaluated for predictive feasibility.

The analysis explicitly distinguishes:

- Response prediction
- Causal uplift modeling

Because the available historical data does not provide the required randomized treatment/control structure, causal uplift modeling is not claimed as completed.

---

## AI Chatbot

The chatbot uses:

```text
Ollama
Qwen 2.5 3B
```

The model runs locally, so an OpenAI API key is not required.

Example privacy-safe questions:

```text
Which 5 circles have the highest 30-day churn rate?

What is the average ARPU by circle?
```

---

## Chatbot Security Controls

The chatbot includes:

- Natural-language privacy filtering
- Approved table allow-list
- SELECT-only SQL validation
- Sensitive-column blocking
- Automatic row limits
- Result sanitization
- Read-only MySQL access
- Audit logging
- Verified numerical answer synthesis

Requests for subscriber identifiers, employee personal information, salary information, or database modifications are blocked.

---

## Running the Chatbot

Start the Streamlit application:

```powershell
python -m streamlit run src/35_jio_chatbot_app.py
```

Then open:

```text
http://localhost:8501
```

---

## Validation

Run final project validation:

```powershell
python src/36_final_project_validation.py
```

The validation checks project structure, source compilation, model artifacts, reports, chatbot artifacts, SQL files, documentation, configuration hygiene, and Streamlit code.

Final validation result:

```text
Passed:   89
Failed:   0
Warnings: 0

[OK] Final project validation passed.
```

---

## Key Deliverables

Important output folders:

```text
outputs/charts/
outputs/reports/
outputs/sql_exports/
models/
data/clv/
architecture/
```

The project also contains final chatbot audit logs and answer-synthesis demonstration artifacts.

---

## GitHub Submission

Before pushing the project to GitHub:

1. Make sure `.env` is not included.
2. Make sure `jio_env/` is not included.
3. Keep `.env.example` with placeholder credentials only.
4. Review `git status`.
5. Review staged files with `git diff --cached --name-only`.
6. Commit the project.
7. Push it to the GitHub repository.

The `.gitignore` file excludes environment secrets, the local virtual environment, Python cache files, local database files, and other local-only files.

If the source project brief contains confidentiality restrictions, use a private GitHub repository unless the bootcamp explicitly requires a public one.

---

## Notes

This repository is a capstone exercise and should be treated as a project/demo environment rather than a production telecommunications system.

The local LLM is used for SQL generation, while database access remains behind application-level privacy and SQL validation controls.

The project uses a local Ollama/Qwen setup and does not require a paid OpenAI API key.
