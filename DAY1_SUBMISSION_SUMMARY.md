# Jio Subscriber Retention & AI Chatbot Project
## Day 1 Submission Summary

### 1. Day 1 Objective

The Day 1 milestone focused on completing the subscriber-retention data analysis, preparing the supplied sandbox data in SQL, and designing the end-to-end architecture for predictive modelling and the employee AI chatbot.

No predictive model was trained during Day 1. Predictive modelling is reserved for the Day 2 milestone.

### 2. Dataset Overview

- Subscriber records: **64,738**
- Service requests: **14,206**
- Network sites: **3,847**
- Circle-month KPI records: **396**
- Circle targets: **22**
- Retention offers: **20**

### 3. Overall Churn

- 30-day churners: **1,122**
- 30-day churn rate: **1.73%**
- 90-day churners: **3,241**
- 90-day churn rate: **5.01%**

### 4. Problematic Regions

- **Uttar Pradesh East** — 2.77% 30-day churn
- **Madhya Pradesh & Chhattisgarh** — 2.49% 30-day churn
- **Bihar & Jharkhand** — 2.44% 30-day churn
- **Uttar Pradesh West** — 2.40% 30-day churn
- **Assam** — 2.37% 30-day churn

### 5. Churn Timeline / Tenure

- 0–3 months: **5.82%**
- 4–6 months: **3.20%**
- 7–12 months: **2.36%**
- 61+ months: **0.94%**

The highest churn occurs during the first three months of subscriber tenure and declines steadily as tenure increases.

### 6. Complaints and Service Issues

- No complaints: **1.12%** churn
- One complaint: **2.03%** churn
- Two or more complaints: **3.32%** churn
- Two or more unresolved complaints: **7.95%** churn

Unresolved complaints show a particularly strong association with subscriber churn.

### 7. Network Quality

- Low congestion (<30): **1.18%** churn
- High congestion (70+): **3.87%** churn
- SINR below 5 dB: **2.56%** churn
- SINR 15+ dB: **1.29%** churn

Poor network conditions are associated with materially higher subscriber churn.

### 8. ARPU / Price Sensitivity

- ARPU below ₹150: **3.22%** churn
- ARPU ₹150–₹249.99: **1.43%** churn
- ARPU ₹250+: **0.51%** churn

Lower-ARPU subscribers show substantially higher churn, supporting the project's price-sensitivity hypothesis.

### 9. Strongest Numeric Associations

- `days_since_last_recharge`: 0.2244
- `avg_recharge_gap_days`: 0.2163
- `recharge_count_6m`: -0.1369
- `payment_failures_6m`: 0.0941
- `unresolved_complaints`: 0.0806

These are associations and should not be interpreted as proof of causation.

### 10. Data Quality

- Subscriber IDs are unique.
- No exact duplicate subscriber rows were found.
- Service-request subscriber references matched the subscriber table.
- No rows were deleted during cleaning.
- No missing values were artificially imputed.
- The supplied `churn_flag_30d` target was preserved.
- 739 synthetic records have a known mismatch between the supplied 30-day churn label and `churn_date`; this was documented rather than rewriting the source target.

Potential leakage fields identified for Day 2 modelling:

- `mnp_enquiry_flag`
- `churn_reason`
- `churn_date`
- `subscriber_id` should also be excluded as an identifier during model training.

### 11. SQL Database

Database: `jio_retention_db`

Tables:

- `subscribers` — 64,738 rows
- `service_requests` — 14,206 rows
- `network_sites` — 3,847 rows
- `circle_monthly_kpi` — 396 rows
- `circle_targets` — 22 rows
- `offer_catalogue` — 20 rows

SQL validation reproduced the same core business findings as the Python analysis.

### 12. Architecture Plan

The proposed architecture includes:

- MySQL analytical database
- Python / Pandas processing
- Power BI management dashboards
- 30-day churn prediction
- Customer Lifetime Value modelling
- Uplift / offer-response modelling
- Customer score store
- Text-to-SQL agent
- SQLAlchemy
- Database schema metadata
- RAG over Excel/business files
- LLM response layer
- Timestamped backend audit logs
- Read-only SQL and governance controls

### 13. Day 2 Plan

Day 2 will focus on:

1. Logistic Regression baseline for 30-day churn.
2. Gradient Boosting / XGBoost / CatBoost champion-model comparison.
3. Precision, Recall, F1, ROC-AUC, PR-AUC, Lift and Recall@Top-Decile.
4. CLV modelling design.
5. Discount-response / uplift modelling design.

### 14. Main Day 1 Deliverables

- Python exploration, validation, cleaning, analysis and visualization scripts
- 11 business-analysis charts
- MySQL database with six imported tables
- SQL table-creation, import and validation scripts
- End-to-end architecture PNG
- End-to-end architecture PDF
- Detailed architecture plan
- Final Day 1 findings summary