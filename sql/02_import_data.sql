-- ==========================================================
-- JIO SUBSCRIBER RETENTION PROJECT
-- STEP 06D - BATCH IMPORT SQL DATA
-- ==========================================================

USE jio_retention_db;


-- ==========================================================
-- 0. CHECK LOCAL FILE IMPORT SETTING
-- ==========================================================

SHOW VARIABLES LIKE 'local_infile';


-- ==========================================================
-- 1. CLEAR THE FIVE TABLES BEFORE IMPORT
--    circle_targets is NOT cleared because it was already
--    imported successfully through Workbench.
-- ==========================================================

TRUNCATE TABLE offer_catalogue;
TRUNCATE TABLE circle_monthly_kpi;
TRUNCATE TABLE network_sites;
TRUNCATE TABLE service_requests;
TRUNCATE TABLE subscribers;


-- ==========================================================
-- 2. OFFER CATALOGUE
-- Expected rows: 20
-- ==========================================================

LOAD DATA LOCAL INFILE
'C:/Users/rkred/Downloads/bootcamp_proj_05_Jio_Retention/outputs/sql_exports/offer_catalogue_sql.csv'

INTO TABLE offer_catalogue

CHARACTER SET utf8mb4

FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'

LINES TERMINATED BY '\r\n'

IGNORE 1 LINES

(
    offer_code,
    offer_name,
    target_segment,
    arpu_band_inr,
    discount_pct,
    bonus_data_gb,
    validity_days,
    cost_per_sub_inr,
    approval_status,
    approved_circles,
    valid_from,
    valid_to,
    owner,
    expected_uplift_pct
);


-- ==========================================================
-- 3. CIRCLE MONTHLY KPI
-- Expected rows: 396
-- ==========================================================

LOAD DATA LOCAL INFILE
'C:/Users/rkred/Downloads/bootcamp_proj_05_Jio_Retention/outputs/sql_exports/circle_monthly_kpi_sql.csv'

INTO TABLE circle_monthly_kpi

CHARACTER SET utf8mb4

FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'

LINES TERMINATED BY '\r\n'

IGNORE 1 LINES

(
    month_end,
    circle,
    circle_code,
    zone,
    closing_base,
    gross_adds,
    churned_subscribers,
    net_adds,
    monthly_churn_pct,
    arpu_inr,
    data_traffic_pb,
    port_in_requests,
    port_out_requests,
    complaints_logged
);


-- ==========================================================
-- 4. NETWORK SITES
-- Expected rows: 3,847
-- ==========================================================

LOAD DATA LOCAL INFILE
'C:/Users/rkred/Downloads/bootcamp_proj_05_Jio_Retention/outputs/sql_exports/network_sites_sql.csv'

INTO TABLE network_sites

CHARACTER SET utf8mb4

FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'

LINES TERMINATED BY '\r\n'

IGNORE 1 LINES

(
    site_id,
    circle,
    circle_code,
    zone,
    pin_code,
    technology,
    subscribers_served,
    prb_utilisation_pct,
    avg_sinr_db,
    drop_call_rate_pct,
    avg_throughput_mbps,
    congestion_flag,
    backhaul
);


-- ==========================================================
-- 5. SERVICE REQUESTS
-- Expected rows: 14,206
-- Some fields legitimately contain NULL values.
-- ==========================================================

LOAD DATA LOCAL INFILE
'C:/Users/rkred/Downloads/bootcamp_proj_05_Jio_Retention/outputs/sql_exports/service_requests_sql.csv'

INTO TABLE service_requests

CHARACTER SET utf8mb4

FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'

LINES TERMINATED BY '\r\n'

IGNORE 1 LINES

(
    sr_id,
    subscriber_id,
    circle,
    circle_code,
    product,
    sr_category,
    sr_subcategory,
    channel,
    raised_date,
    sla_days,
    @resolution_days,
    @resolved_date,
    sla_breach_flag,
    reopened_flag,
    status,
    @csat_score
)

SET
    resolution_days =
        NULLIF(@resolution_days, ''),

    resolved_date =
        STR_TO_DATE(
            NULLIF(@resolved_date, ''),
            '%Y-%m-%d %H:%i:%s'
        ),

    csat_score =
        NULLIF(@csat_score, '');


-- ==========================================================
-- 6. SUBSCRIBERS
-- Expected rows: 64,738
-- home_product, churn_reason and churn_date may be NULL.
-- ==========================================================

LOAD DATA LOCAL INFILE
'C:/Users/rkred/Downloads/bootcamp_proj_05_Jio_Retention/outputs/sql_exports/subscribers_sql.csv'

INTO TABLE subscribers

CHARACTER SET utf8mb4

FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'

LINES TERMINATED BY '\r\n'

IGNORE 1 LINES

(
    subscriber_id,
    circle,
    circle_code,
    zone,
    join_date,
    tenure_months,
    plan_type,
    plan_price_inr,
    plan_validity_days,
    arpu_last_month_inr,
    arpu_3m_avg_inr,
    arpu_6m_avg_inr,
    recharge_count_6m,
    avg_recharge_gap_days,
    days_since_last_recharge,
    payment_failures_6m,
    autopay_enabled,
    data_gb_last_month,
    data_gb_3m_avg,
    voice_minutes_last_month,
    sms_count_last_month,
    device_brand,
    is_5g_device,
    is_5g_active,
    avg_sinr_db,
    drop_call_rate_pct,
    site_congestion_score,
    complaints_6m,
    unresolved_complaints,
    avg_resolution_days,
    @home_product,
    num_services,
    family_plan_flag,
    app_logins_30d,
    outgoing_to_competitor_pct,
    roaming_user_flag,
    offer_exposed_90d,
    offer_redeemed_90d,
    mnp_enquiry_flag,
    churn_flag_30d,
    churn_flag_90d,
    @churn_reason,
    @churn_date
)

SET
    home_product =
        NULLIF(@home_product, ''),

    churn_reason =
        NULLIF(@churn_reason, ''),

    churn_date =
        STR_TO_DATE(
            NULLIF(@churn_date, ''),
            '%Y-%m-%d'
        );


-- ==========================================================
-- 7. VALIDATE ALL SIX TABLE ROW COUNTS
-- ==========================================================

SELECT
    'subscribers' AS table_name,
    COUNT(*) AS row_count
FROM subscribers

UNION ALL

SELECT
    'service_requests',
    COUNT(*)
FROM service_requests

UNION ALL

SELECT
    'network_sites',
    COUNT(*)
FROM network_sites

UNION ALL

SELECT
    'circle_monthly_kpi',
    COUNT(*)
FROM circle_monthly_kpi

UNION ALL

SELECT
    'circle_targets',
    COUNT(*)
FROM circle_targets

UNION ALL

SELECT
    'offer_catalogue',
    COUNT(*)
FROM offer_catalogue;


-- ==========================================================
-- 8. PRIMARY KEY VALIDATION
-- ==========================================================

SELECT
    COUNT(*) AS subscriber_rows,
    COUNT(DISTINCT subscriber_id) AS unique_subscriber_ids
FROM subscribers;


SELECT
    COUNT(*) AS service_request_rows,
    COUNT(DISTINCT sr_id) AS unique_service_request_ids
FROM service_requests;


SELECT
    COUNT(*) AS network_site_rows,
    COUNT(DISTINCT site_id) AS unique_site_ids
FROM network_sites;


SELECT
    COUNT(*) AS circle_target_rows,
    COUNT(DISTINCT circle_code) AS unique_circle_codes
FROM circle_targets;


-- ==========================================================
-- 9. TARGET VALIDATION
-- ==========================================================

SELECT
    COUNT(*) AS total_subscribers,

    SUM(churn_flag_30d) AS churners_30d,

    ROUND(
        100.0 * SUM(churn_flag_30d) / COUNT(*),
        2
    ) AS churn_rate_30d_pct,

    SUM(churn_flag_90d) AS churners_90d,

    ROUND(
        100.0 * SUM(churn_flag_90d) / COUNT(*),
        2
    ) AS churn_rate_90d_pct

FROM subscribers;


-- ==========================================================
-- 10. SERVICE REQUEST FOREIGN-KEY STYLE CHECK
-- ==========================================================

SELECT
    COUNT(*) AS unmatched_service_requests

FROM service_requests sr

LEFT JOIN subscribers s
    ON sr.subscriber_id = s.subscriber_id

WHERE s.subscriber_id IS NULL;