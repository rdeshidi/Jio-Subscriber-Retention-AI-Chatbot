-- ==========================================================
-- JIO SUBSCRIBER RETENTION PROJECT
-- STEP 07 - SQL VALIDATION / BUSINESS CHECKS
-- ==========================================================

USE jio_retention_db;


-- ==========================================================
-- 1. TABLE ROW COUNTS
-- ==========================================================

SELECT 'subscribers' AS table_name, COUNT(*) AS row_count
FROM subscribers

UNION ALL

SELECT 'service_requests', COUNT(*)
FROM service_requests

UNION ALL

SELECT 'network_sites', COUNT(*)
FROM network_sites

UNION ALL

SELECT 'circle_monthly_kpi', COUNT(*)
FROM circle_monthly_kpi

UNION ALL

SELECT 'circle_targets', COUNT(*)
FROM circle_targets

UNION ALL

SELECT 'offer_catalogue', COUNT(*)
FROM offer_catalogue;


-- ==========================================================
-- 2. OVERALL CHURN
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
-- 3. TOP 10 PROBLEMATIC CIRCLES
-- ==========================================================

SELECT
    circle,
    COUNT(*) AS subscribers,
    SUM(churn_flag_30d) AS churners_30d,
    ROUND(
        100.0 * SUM(churn_flag_30d) / COUNT(*),
        2
    ) AS churn_rate_30d_pct
FROM subscribers
GROUP BY circle
ORDER BY churn_rate_30d_pct DESC
LIMIT 10;


-- ==========================================================
-- 4. TENURE VS CHURN
-- ==========================================================

SELECT
    CASE
        WHEN tenure_months <= 3 THEN '0-3 months'
        WHEN tenure_months <= 6 THEN '4-6 months'
        WHEN tenure_months <= 12 THEN '7-12 months'
        WHEN tenure_months <= 24 THEN '13-24 months'
        WHEN tenure_months <= 36 THEN '25-36 months'
        WHEN tenure_months <= 60 THEN '37-60 months'
        ELSE '61+ months'
    END AS tenure_band,

    COUNT(*) AS subscribers,
    SUM(churn_flag_30d) AS churners_30d,

    ROUND(
        100.0 * SUM(churn_flag_30d) / COUNT(*),
        2
    ) AS churn_rate_30d_pct

FROM subscribers

GROUP BY tenure_band

ORDER BY
    MIN(tenure_months);


-- ==========================================================
-- 5. COMPLAINT COUNT VS CHURN
-- ==========================================================

SELECT
    CASE
        WHEN complaints_6m = 0 THEN '0 complaints'
        WHEN complaints_6m = 1 THEN '1 complaint'
        ELSE '2+ complaints'
    END AS complaint_band,

    COUNT(*) AS subscribers,
    SUM(churn_flag_30d) AS churners_30d,

    ROUND(
        100.0 * SUM(churn_flag_30d) / COUNT(*),
        2
    ) AS churn_rate_30d_pct

FROM subscribers

GROUP BY complaint_band

ORDER BY
    MIN(complaints_6m);


-- ==========================================================
-- 6. UNRESOLVED COMPLAINTS VS CHURN
-- ==========================================================

SELECT
    CASE
        WHEN unresolved_complaints = 0
            THEN '0 unresolved'

        WHEN unresolved_complaints = 1
            THEN '1 unresolved'

        ELSE '2+ unresolved'
    END AS unresolved_band,

    COUNT(*) AS subscribers,
    SUM(churn_flag_30d) AS churners_30d,

    ROUND(
        100.0 * SUM(churn_flag_30d) / COUNT(*),
        2
    ) AS churn_rate_30d_pct

FROM subscribers

GROUP BY unresolved_band

ORDER BY
    MIN(unresolved_complaints);


-- ==========================================================
-- 7. NETWORK CONGESTION VS CHURN
-- ==========================================================

SELECT
    CASE
        WHEN site_congestion_score < 30
            THEN '<30'

        WHEN site_congestion_score < 70
            THEN '30-69.9'

        ELSE '70+'
    END AS congestion_band,

    COUNT(*) AS subscribers,
    SUM(churn_flag_30d) AS churners_30d,

    ROUND(
        100.0 * SUM(churn_flag_30d) / COUNT(*),
        2
    ) AS churn_rate_30d_pct

FROM subscribers

GROUP BY congestion_band

ORDER BY
    MIN(site_congestion_score);


-- ==========================================================
-- 8. SINR VS CHURN
-- ==========================================================

SELECT
    CASE
        WHEN avg_sinr_db < 5
            THEN '<5 dB'

        WHEN avg_sinr_db < 15
            THEN '5-14.9 dB'

        ELSE '15+ dB'
    END AS sinr_band,

    COUNT(*) AS subscribers,
    SUM(churn_flag_30d) AS churners_30d,

    ROUND(
        100.0 * SUM(churn_flag_30d) / COUNT(*),
        2
    ) AS churn_rate_30d_pct

FROM subscribers

GROUP BY sinr_band

ORDER BY
    MIN(avg_sinr_db);


-- ==========================================================
-- 9. ARPU VS CHURN
-- ==========================================================

SELECT
    CASE
        WHEN arpu_last_month_inr < 150
            THEN '<150'

        WHEN arpu_last_month_inr < 250
            THEN '150-249.99'

        ELSE '250+'
    END AS arpu_band,

    COUNT(*) AS subscribers,
    SUM(churn_flag_30d) AS churners_30d,

    ROUND(
        100.0 * SUM(churn_flag_30d) / COUNT(*),
        2
    ) AS churn_rate_30d_pct

FROM subscribers

GROUP BY arpu_band

ORDER BY
    MIN(arpu_last_month_inr);


-- ==========================================================
-- 10. SERVICE REQUEST REFERENTIAL INTEGRITY
-- ==========================================================

SELECT
    COUNT(*) AS unmatched_service_requests

FROM service_requests sr

LEFT JOIN subscribers s
    ON sr.subscriber_id = s.subscriber_id

WHERE s.subscriber_id IS NULL;


-- ==========================================================
-- 11. SERVICE REQUESTS + CHURN
-- ==========================================================

SELECT
    CASE
        WHEN sr_counts.request_count IS NULL
            THEN '0 requests'

        WHEN sr_counts.request_count = 1
            THEN '1 request'

        ELSE '2+ requests'
    END AS request_band,

    COUNT(*) AS subscribers,

    SUM(s.churn_flag_30d)
        AS churners_30d,

    ROUND(
        100.0
        * SUM(s.churn_flag_30d)
        / COUNT(*),
        2
    ) AS churn_rate_30d_pct

FROM subscribers s

LEFT JOIN
(
    SELECT
        subscriber_id,
        COUNT(*) AS request_count
    FROM service_requests
    GROUP BY subscriber_id
) sr_counts

ON s.subscriber_id =
   sr_counts.subscriber_id

GROUP BY request_band

ORDER BY
    MIN(
        COALESCE(
            sr_counts.request_count,
            0
        )
    );


-- ==========================================================
-- 12. LATEST CIRCLE KPI
-- ==========================================================

SELECT
    circle,
    month_end,
    monthly_churn_pct,
    arpu_inr,
    port_out_requests,
    complaints_logged
FROM circle_monthly_kpi

WHERE month_end = (
    SELECT MAX(month_end)
    FROM circle_monthly_kpi
)

ORDER BY monthly_churn_pct DESC;


-- ==========================================================
-- 13. TARGET VS ACTUAL CHURN
-- ==========================================================

SELECT
    circle,
    churn_actual_pct,
    churn_ceiling_fy27_pct,

    ROUND(
        churn_actual_pct
        - churn_ceiling_fy27_pct,
        2
    ) AS gap_vs_ceiling_pct_points,

    review_status

FROM circle_targets

ORDER BY
    gap_vs_ceiling_pct_points DESC;


-- ==========================================================
-- 14. OFFER CATALOGUE CHECK
-- ==========================================================

SELECT
    offer_code,
    offer_name,
    target_segment,
    discount_pct,
    cost_per_sub_inr,
    approval_status,
    expected_uplift_pct
FROM offer_catalogue
ORDER BY expected_uplift_pct DESC;