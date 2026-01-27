"""
Model Training Script for Revenue Forecasting - Databricks Unity Catalog
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for revenue prediction"""
    
    def __init__(self):
        self.feature_list = config.ALL_FEATURES
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        logger.info(f"[FEATURE] Creating features for {len(df)} rows")
        
        try:
            df_feat = df.copy().sort_values(['year', 'month_num']).reset_index(drop=True)
            
            if 'month_id' not in df_feat.columns:
                df_feat['month_id'] = df_feat['year'] * 100 + df_feat['month_num']
            
            # Static/Calendar
            df_feat['remaining_months'] = 13 - df_feat['month_num']
            df_feat['quarter'] = ((df_feat['month_num'] - 1) // 3) + 1
            df_feat['is_q4'] = (df_feat['quarter'] == 4).astype(int)
            df_feat['is_q2'] = (df_feat['quarter'] == 2).astype(int)
            df_feat['is_end_of_quarter'] = df_feat['month_num'].isin([3, 6, 9, 12]).astype(int)
            df_feat['is_quarter_start'] = df_feat['month_num'].isin([1, 4, 7, 10]).astype(int)
            df_feat['quarter_position'] = ((df_feat['month_num'] - 1) % 3) + 1
            
            # Past Year
            df_feat['ly_same_month_revenue'] = df_feat.groupby('month_num')['actual_revenue'].shift(1)
            df_feat['ly_same_qtr_avg'] = df_feat.groupby(['quarter'])['actual_revenue'].transform(
                lambda x: x.shift(3).rolling(3, min_periods=1).mean()
            )
            
            # Lag Features
            df_feat['revenue_lag_1'] = df_feat['actual_revenue'].shift(1)
            df_feat['revenue_lag_2'] = df_feat['actual_revenue'].shift(2)
            df_feat['revenue_lag_3'] = df_feat['actual_revenue'].shift(3)
            df_feat['revenue_3mo_avg'] = df_feat['actual_revenue'].shift(1).rolling(3, min_periods=1).mean()
            df_feat['revenue_velocity'] = df_feat['revenue_lag_1'] - df_feat['revenue_lag_2']
            df_feat['revenue_acceleration'] = df_feat['revenue_velocity'] - df_feat['revenue_velocity'].shift(1)
            
            # YTD
            df_feat['ytd_revenue'] = df_feat.groupby('year')['actual_revenue'].cumsum().shift(1)
            df_feat['ytd_avg'] = df_feat['ytd_revenue'] / (df_feat['month_num'] - 1).replace(0, 1)
            df_feat['perf_vs_ytd'] = ((df_feat['revenue_lag_1'] - df_feat['ytd_avg']) / 
                                      (df_feat['ytd_avg'] + 1e-10)).clip(-0.5, 0.5)
            
            # Quarter
            df_feat['quarter_cumulative'] = df_feat.groupby(['year', 'quarter'])['actual_revenue'].cumsum().shift(1)
            df_feat['prev_quarter_avg'] = df_feat['actual_revenue'].shift(1).rolling(3, min_periods=1).mean().shift(2)
            df_feat['last_quarter_end_rev'] = df_feat['actual_revenue'].shift(1).where(
                df_feat['month_num'].shift(1).isin([3, 6, 9, 12])
            ).ffill()
            df_feat['qoq_change'] = ((df_feat['revenue_3mo_avg'] - df_feat['prev_quarter_avg']) / 
                                    (df_feat['prev_quarter_avg'] + 1e-10)).clip(-0.5, 0.5)
            
            # Trend
            df_feat['revenue_6mo_avg'] = df_feat['actual_revenue'].shift(1).rolling(6, min_periods=1).mean()
            df_feat['trend_direction'] = np.sign(df_feat['revenue_3mo_avg'] - df_feat['revenue_6mo_avg'])
            
            # Forecast Remaining
            df_feat['fcst_total_rem'] = (df_feat['committed_sign_revenue'] + 
                                          df_feat['committed_unsig_revenue'] + 
                                          df_feat['wtd_pipeline_revenue'])
            df_feat['fcst_signed_rem'] = df_feat['committed_sign_revenue']
            df_feat['fcst_unsigned_rem'] = df_feat['committed_unsig_revenue']
            df_feat['fcst_pipeline_rem'] = df_feat['wtd_pipeline_revenue']
            df_feat['signed_per_month'] = df_feat['fcst_signed_rem'] / df_feat['remaining_months'].replace(0, 1)
            
            # Ratios
            df_feat['committed_ratio'] = df_feat['fcst_signed_rem'] / (df_feat['fcst_total_rem'] + 1e-10)
            df_feat['unsigned_ratio'] = df_feat['fcst_unsigned_rem'] / (df_feat['fcst_total_rem'] + 1e-10)
            df_feat['pipeline_quality'] = (
                df_feat['fcst_signed_rem'] * 1.0 + 
                df_feat['fcst_unsigned_rem'] * 0.7 +
                df_feat['fcst_pipeline_rem'] * 0.3
            ) / (df_feat['fcst_total_rem'] + 1e-10)
            
            # Handle infinities
            for col in df_feat.columns:
                if df_feat[col].dtype in [np.float64, np.float32]:
                    df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"[FEATURE] Created {len(self.feature_list)} features")
            return df_feat, self.feature_list
            
        except Exception as e:
            logger.error(f"[FEATURE] Error: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class RevenueModelTrainer:
    """Revenue Forecasting Model Trainer"""
    
    def __init__(self, request_id: str = None):
        self.request_id = request_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.scaler = None
        self.selected_features = config.FINAL_FEATURES
        self.feature_medians = None
        self.metrics = {}
        self.X_train = None
        self.run_id = None
        
        logger.info(f"[{self.request_id}] Trainer initialized")
    
    def load_data(self, df: pd.DataFrame = None, file_path: str = None) -> pd.DataFrame:
        logger.info(f"[{self.request_id}] Loading data")
        
        if df is not None:
            data = df.copy()
        elif file_path:
            data = pd.read_csv(file_path, index_col=0)
            logger.info(f"[{self.request_id}] File: {file_path}")
        else:
            raise ValueError("Provide df or file_path")
        
        data = data.sort_values(['year', 'month_num']).reset_index(drop=True)
        data['month_id'] = data['year'] * 100 + data['month_num']
        
        logger.info(f"[{self.request_id}] Rows: {len(data)}, Years: {sorted(data['year'].unique())}")
        return data
    
    def prepare_data(self, df: pd.DataFrame, train_years: List[int] = None, test_year: int = None):
        logger.info(f"[{self.request_id}] Preparing data")
        
        df_feat, _ = self.feature_engineer.create_features(df)
        
        years = sorted(df_feat['year'].unique())
        train_years = train_years or years[:-1]
        test_year = test_year or years[-1]
        
        train_df = df_feat[df_feat['year'].isin(train_years)].dropna(subset=[config.TARGET])
        test_df = df_feat[df_feat['year'] == test_year].copy()
        
        logger.info(f"[{self.request_id}] Train: {len(train_df)} ({train_years}), Test: {len(test_df)} ({test_year})")
        return df_feat, train_df, test_df
    
    def train(self, df: pd.DataFrame = None, file_path: str = None) -> Dict[str, float]:
        """Train the model"""
        logger.info(f"[{self.request_id}] ========== TRAINING STARTED ==========")
        
        try:
            data = self.load_data(df, file_path)
            df_feat, train_df, test_df = self.prepare_data(data)
            
            X_train = train_df[self.selected_features].copy()
            y_train = train_df[config.TARGET].copy()
            
            self.feature_medians = X_train.median()
            X_train = X_train.fillna(self.feature_medians)
            self.X_train = X_train.copy()
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
            alphas = np.logspace(-1, 4, 50)
            
            self.model = RidgeCV(alphas=alphas, cv=tscv)
            self.model.fit(X_train_scaled, y_train)
            
            logger.info(f"[{self.request_id}] Model trained - Alpha: {self.model.alpha_:.2f}")
            
            # Metrics
            train_pred = self.model.predict(X_train_scaled)
            self.metrics = {
                'mape': mean_absolute_percentage_error(y_train, train_pred) * 100,
                'mae': mean_absolute_error(y_train, train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, train_pred))
            }
            
            logger.info(f"[{self.request_id}] MAPE: {self.metrics['mape']:.2f}%")
            logger.info(f"[{self.request_id}] ========== TRAINING COMPLETED ==========")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"[{self.request_id}] Training error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def register_to_unity_catalog(self) -> str:
        """
        Register model to Unity Catalog
        
        Uses: mlflow.set_registry_uri("databricks-uc")
        Model path: {catalog}.{schema}.{model_name}
        """
        logger.info(f"[{self.request_id}] ========== REGISTERING TO UNITY CATALOG ==========")
        logger.info(f"[{self.request_id}] Model: {config.FULL_MODEL_NAME}")
        
        try:
            # Set Unity Catalog registry
            mlflow.set_registry_uri("databricks-uc")
            
            with mlflow.start_run() as run:
                # Create signature
                signature = infer_signature(self.X_train, self.model.predict(self.scaler.transform(self.X_train)))
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    signature=signature,
                    input_example=self.X_train.head(5)
                )
                
                # Log scaler
                mlflow.sklearn.log_model(
                    sk_model=self.scaler,
                    artifact_path="scaler"
                )
                
                # Log metrics
                for key, val in self.metrics.items():
                    mlflow.log_metric(key, val)
                
                # Log params
                mlflow.log_param("alpha", self.model.alpha_)
                mlflow.log_param("n_features", len(self.selected_features))
                mlflow.log_param("features", str(self.selected_features))
                
                # Log artifacts
                mlflow.log_dict({"features": self.selected_features}, "feature_names.json")
                mlflow.log_dict(self.feature_medians.to_dict(), "feature_medians.json")
                
                # Register to Unity Catalog
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri=model_uri, name=config.FULL_MODEL_NAME)
                
                self.run_id = run.info.run_id
            
            logger.info(f"[{self.request_id}] Run ID: {self.run_id}")
            logger.info(f"[{self.request_id}] Registered to: {config.FULL_MODEL_NAME}")
            logger.info(f"[{self.request_id}] ========== REGISTRATION COMPLETED ==========")
            
            return self.run_id
            
        except Exception as e:
            logger.error(f"[{self.request_id}] Registration error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def train_and_register(self, df: pd.DataFrame = None, file_path: str = None) -> Dict:
        """Train and register in one step"""
        self.train(df=df, file_path=file_path)
        run_id = self.register_to_unity_catalog()
        
        return {
            'run_id': run_id,
            'model_name': config.FULL_MODEL_NAME,
            'metrics': self.metrics
        }


# =============================================================================
# DATABRICKS NOTEBOOK USAGE
# =============================================================================
"""
# In Databricks notebook:

from model_training import RevenueModelTrainer
import config

# Option 1: Train and register in one step
trainer = RevenueModelTrainer(request_id="training_001")
result = trainer.train_and_register(file_path="/dbfs/path/to/data.csv")

print(f"Run ID: {result['run_id']}")
print(f"Model: {result['model_name']}")
print(f"MAPE: {result['metrics']['mape']:.2f}%")


# Option 2: Step by step
trainer = RevenueModelTrainer()
trainer.train(file_path="/dbfs/path/to/data.csv")
trainer.register_to_unity_catalog()

# Model is now at: gt_ai_rni_ci_databricks_workspace.ml_models.revenue_forecasting_model
"""
