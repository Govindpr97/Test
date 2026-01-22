"""
Config for Revenue Forecasting Pipeline
"""

import os
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
MODEL_DIR = BASE_DIR / "model_artifacts"
MODEL_DIR.mkdir(exist_ok=True)

# Data file
DATA_FILE = "final_imputed.csv"

# Target column
TARGET = "actual_revenue"

# Model config
MODEL_NAME = "revenue_forecasting_model"
RANDOM_STATE = 42
CV_SPLITS = 3
CORRELATION_THRESHOLD = 0.85
N_TOP_FEATURES = 15

# Alpha ranges for regularization
ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

# Retraining frequency (months)
RETRAIN_FREQUENCY = 6

# =============================================================================
# FINAL FEATURES - Selected via LassoCV with correlation filtering
# =============================================================================
# These are the 15 features selected for the model

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
# ALL FEATURES - Complete list of engineered features (31 total)
# =============================================================================
# Organized by category based on imputation strategy

ALL_FEATURES = [
    # Static/Calendar Features (no imputation needed)
    'remaining_months', 'quarter', 'is_q4', 'is_q2', 
    'is_end_of_quarter', 'is_quarter_start', 'quarter_position',
    
    # Past Year Features (forward fill during prediction)
    'ly_same_month_revenue', 'ly_same_qtr_avg',
    
    # Lag Features (use predictions during rolling forecast)
    'revenue_lag_1', 'revenue_lag_2', 'revenue_lag_3', 
    'revenue_3mo_avg', 'revenue_velocity', 'revenue_acceleration',
    'ytd_revenue', 'ytd_avg', 'perf_vs_ytd',
    'quarter_cumulative', 'prev_quarter_avg', 'last_quarter_end_rev',
    'qoq_change', 'trend_direction',
    
    # Forecast Remaining Features (burn down during prediction)
    'fcst_total_rem', 'fcst_signed_rem', 'fcst_unsigned_rem',
    'fcst_pipeline_rem', 'signed_per_month',
    
    # Trend/Ratio Features (increase with random % during prediction)
    'committed_ratio', 'unsigned_ratio', 'pipeline_quality'
]

# Required raw columns
RAW_COLUMNS = [
    'year', 'month', 'month_num', 'actual_revenue',
    'wtd_pipeline_revenue', 'committed_unsig_revenue', 
    'committed_sign_revenue', 'avg_prob_pct', 'date'
]

# MLflow settings (for when Unity Catalog is ready)
MLFLOW_TRACKING_URI = ""  # Set when UC is provisioned
MLFLOW_EXPERIMENT = "revenue_forecasting"

# Blob storage (current approach)
BLOB_CONNECTION_STRING = os.environ.get("BLOB_CONNECTION_STRING", "")
BLOB_CONTAINER = "models"
