-- ==========================================================
-- JIO SUBSCRIBER RETENTION PROJECT
-- STEP 06B - CREATE SQL TABLES
-- ==========================================================

USE jio_retention_db;


-- ==========================================================
-- 1. SUBSCRIBERS
-- ==========================================================

DROP TABLE IF EXISTS subscribers;

CREATE TABLE subscribers (
    subscriber_id VARCHAR(11) NOT NULL,
    circle VARCHAR(50) NOT NULL,
    circle_code VARCHAR(3) NOT NULL,
    zone VARCHAR(10) NOT NULL,
    join_date DATE NOT NULL,
    tenure_months SMALLINT UNSIGNED NOT NULL,

    plan_type VARCHAR(20) NOT NULL,
    plan_price_inr DECIMAL(10,2) NOT NULL,
    plan_validity_days SMALLINT UNSIGNED NOT NULL,

    arpu_last_month_inr DECIMAL(10,2) NOT NULL,
    arpu_3m_avg_inr DECIMAL(10,2) NOT NULL,
    arpu_6m_avg_inr DECIMAL(10,2) NOT NULL,

    recharge_count_6m TINYINT UNSIGNED NOT NULL,
    avg_recharge_gap_days DECIMAL(8,2) NOT NULL,
    days_since_last_recharge SMALLINT UNSIGNED NOT NULL,
    payment_failures_6m TINYINT UNSIGNED NOT NULL,

    autopay_enabled TINYINT(1) NOT NULL,

    data_gb_last_month DECIMAL(10,2) NOT NULL,
    data_gb_3m_avg DECIMAL(10,2) NOT NULL,
    voice_minutes_last_month INT UNSIGNED NOT NULL,
    sms_count_last_month SMALLINT UNSIGNED NOT NULL,

    device_brand VARCHAR(50) NOT NULL,

    is_5g_device TINYINT(1) NOT NULL,
    is_5g_active TINYINT(1) NOT NULL,

    avg_sinr_db DECIMAL(6,2) NOT NULL,
    drop_call_rate_pct DECIMAL(6,2) NOT NULL,
    site_congestion_score DECIMAL(6,2) NOT NULL,

    complaints_6m TINYINT UNSIGNED NOT NULL,
    unresolved_complaints TINYINT UNSIGNED NOT NULL,
    avg_resolution_days DECIMAL(8,2) NOT NULL,

    home_product VARCHAR(30) NULL,
    num_services TINYINT UNSIGNED NOT NULL,
    family_plan_flag TINYINT(1) NOT NULL,

    app_logins_30d SMALLINT UNSIGNED NOT NULL,
    outgoing_to_competitor_pct DECIMAL(6,2) NOT NULL,

    roaming_user_flag TINYINT(1) NOT NULL,

    offer_exposed_90d TINYINT(1) NOT NULL,
    offer_redeemed_90d TINYINT(1) NOT NULL,

    mnp_enquiry_flag TINYINT(1) NOT NULL,

    churn_flag_30d TINYINT(1) NOT NULL,
    churn_flag_90d TINYINT(1) NOT NULL,

    churn_reason VARCHAR(100) NULL,
    churn_date DATE NULL,

    PRIMARY KEY (subscriber_id),

    INDEX idx_subscribers_circle (circle_code),
    INDEX idx_subscribers_churn30 (churn_flag_30d),
    INDEX idx_subscribers_churn90 (churn_flag_90d)
);


-- ==========================================================
-- 2. SERVICE REQUESTS
-- ==========================================================

DROP TABLE IF EXISTS service_requests;

CREATE TABLE service_requests (
    sr_id VARCHAR(11) NOT NULL,
    subscriber_id VARCHAR(11) NOT NULL,

    circle VARCHAR(50) NOT NULL,
    circle_code VARCHAR(3) NOT NULL,

    product VARCHAR(30) NOT NULL,
    sr_category VARCHAR(40) NOT NULL,
    sr_subcategory VARCHAR(80) NOT NULL,
    channel VARCHAR(30) NOT NULL,

    raised_date DATE NOT NULL,

    sla_days TINYINT UNSIGNED NOT NULL,
    resolution_days DECIMAL(8,2) NULL,
    resolved_date DATETIME NULL,

    sla_breach_flag TINYINT(1) NOT NULL,
    reopened_flag TINYINT(1) NOT NULL,

    status VARCHAR(20) NOT NULL,
    csat_score DECIMAL(3,1) NULL,

    PRIMARY KEY (sr_id),

    INDEX idx_sr_subscriber (subscriber_id),
    INDEX idx_sr_circle (circle_code)
);


-- ==========================================================
-- 3. NETWORK SITES
-- ==========================================================

DROP TABLE IF EXISTS network_sites;

CREATE TABLE network_sites (
    site_id VARCHAR(11) NOT NULL,

    circle VARCHAR(50) NOT NULL,
    circle_code VARCHAR(3) NOT NULL,
    zone VARCHAR(10) NOT NULL,

    pin_code CHAR(6) NOT NULL,
    technology VARCHAR(20) NOT NULL,

    subscribers_served INT UNSIGNED NOT NULL,

    prb_utilisation_pct DECIMAL(6,2) NOT NULL,
    avg_sinr_db DECIMAL(6,2) NOT NULL,
    drop_call_rate_pct DECIMAL(6,2) NOT NULL,
    avg_throughput_mbps DECIMAL(8,2) NOT NULL,

    congestion_flag TINYINT(1) NOT NULL,

    backhaul VARCHAR(20) NOT NULL,

    PRIMARY KEY (site_id),

    INDEX idx_network_circle (circle_code)
);


-- ==========================================================
-- 4. CIRCLE MONTHLY KPI
-- ==========================================================

DROP TABLE IF EXISTS circle_monthly_kpi;

CREATE TABLE circle_monthly_kpi (
    month_end DATE NOT NULL,

    circle VARCHAR(50) NOT NULL,
    circle_code VARCHAR(3) NOT NULL,
    zone VARCHAR(10) NOT NULL,

    closing_base BIGINT UNSIGNED NOT NULL,
    gross_adds BIGINT UNSIGNED NOT NULL,
    churned_subscribers BIGINT UNSIGNED NOT NULL,

    net_adds BIGINT NOT NULL,

    monthly_churn_pct DECIMAL(6,2) NOT NULL,
    arpu_inr DECIMAL(10,2) NOT NULL,
    data_traffic_pb DECIMAL(12,2) NOT NULL,

    port_in_requests BIGINT UNSIGNED NOT NULL,
    port_out_requests BIGINT UNSIGNED NOT NULL,

    complaints_logged INT UNSIGNED NOT NULL,

    PRIMARY KEY (
        month_end,
        circle_code
    ),

    INDEX idx_kpi_circle (circle_code)
);


-- ==========================================================
-- 5. CIRCLE TARGETS
-- ==========================================================

DROP TABLE IF EXISTS circle_targets;

CREATE TABLE circle_targets (
    circle VARCHAR(50) NOT NULL,
    circle_code VARCHAR(3) NOT NULL,
    zone VARCHAR(10) NOT NULL,

    base_aug_2026 BIGINT UNSIGNED NOT NULL,

    churn_actual_pct DECIMAL(6,2) NOT NULL,
    arpu_actual_inr DECIMAL(10,2) NOT NULL,

    churn_ceiling_fy27_pct DECIMAL(6,2) NOT NULL,
    arpu_target_fy27_inr DECIMAL(10,2) NOT NULL,

    retention_budget_lakh DECIMAL(12,2) NOT NULL,
    budget_utilised_lakh DECIMAL(12,2) NOT NULL,

    circle_owner VARCHAR(50) NOT NULL,
    review_status VARCHAR(30) NOT NULL,

    PRIMARY KEY (circle_code)
);


-- ==========================================================
-- 6. OFFER CATALOGUE
-- ==========================================================

DROP TABLE IF EXISTS offer_catalogue;

CREATE TABLE offer_catalogue (
    offer_code VARCHAR(12) NOT NULL,

    offer_name VARCHAR(100) NOT NULL,
    target_segment VARCHAR(50) NOT NULL,
    arpu_band_inr VARCHAR(30) NOT NULL,

    discount_pct TINYINT UNSIGNED NOT NULL,
    bonus_data_gb SMALLINT UNSIGNED NOT NULL,
    validity_days SMALLINT UNSIGNED NOT NULL,

    cost_per_sub_inr DECIMAL(10,2) NOT NULL,

    approval_status VARCHAR(30) NOT NULL,
    approved_circles VARCHAR(255) NOT NULL,

    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL,

    owner VARCHAR(50) NOT NULL,

    expected_uplift_pct DECIMAL(6,2) NOT NULL,

    PRIMARY KEY (offer_code)
);


-- ==========================================================
-- VERIFY TABLE CREATION
-- ==========================================================

SHOW TABLES;