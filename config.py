"""
Config for Revenue Forecasting - Databricks Unity Catalog
"""

# =============================================================================
# UNITY CATALOG SETTINGS
# =============================================================================
CATALOG_NAME = "gt_ai_rni_ci_databricks_workspace"  # Your catalog
SCHEMA_NAME = "ml_models"                            # Your schema
MODEL_NAME = "revenue_forecasting_model"             # Model name

# Full Unity Catalog path
FULL_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"

# =============================================================================
# MODEL CONFIG
# =============================================================================
TARGET = "actual_revenue"
RANDOM_STATE = 42
CV_SPLITS = 3

# =============================================================================
# FINAL FEATURES (15 selected)
# =============================================================================
FINAL_FEATURES = [
    'ytd_revenue',
    'remaining_months', 
    'quarter_cumulative',
    'perf_vs_ytd',
    'is_quarter_start',
    'revenue_lag_1',
    'revenue_velocity',
    'revenue_lag_2',
    'revenue_acceleration',
    'signed_per_month',
    'qoq_change',
    'is_q2',
    'trend_direction',
    'revenue_lag_3',
    'is_q4'
]

# =============================================================================
# ALL FEATURES (31 total)
# =============================================================================
ALL_FEATURES = [
    'remaining_months', 'quarter', 'is_q4', 'is_q2', 
    'is_end_of_quarter', 'is_quarter_start', 'quarter_position',
    'ly_same_month_revenue', 'ly_same_qtr_avg',
    'revenue_lag_1', 'revenue_lag_2', 'revenue_lag_3', 
    'revenue_3mo_avg', 'revenue_velocity', 'revenue_acceleration',
    'ytd_revenue', 'ytd_avg', 'perf_vs_ytd',
    'quarter_cumulative', 'prev_quarter_avg', 'last_quarter_end_rev',
    'qoq_change', 'trend_direction',
    'fcst_total_rem', 'fcst_signed_rem', 'fcst_unsigned_rem',
    'fcst_pipeline_rem', 'signed_per_month',
    'committed_ratio', 'unsigned_ratio', 'pipeline_quality'
]
