"""
Model Training and Retraining Script for Revenue Forecasting
Version: 1.0.0
"""

import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for revenue prediction"""
    
    def __init__(self):
        self.feature_list = config.ALL_FEATURES
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
       
        df_feat = df.copy().sort_values(['year', 'month_num']).reset_index(drop=True)
        
        # month_id for unique identification
        if 'month_id' not in df_feat.columns:
            df_feat['month_id'] = df_feat['year'] * 100 + df_feat['month_num']
        
        df_feat['remaining_months'] = 13 - df_feat['month_num']
        df_feat['quarter'] = ((df_feat['month_num'] - 1) // 3) + 1
        df_feat['is_q4'] = (df_feat['quarter'] == 4).astype(int)
        df_feat['is_q2'] = (df_feat['quarter'] == 2).astype(int)
        df_feat['is_end_of_quarter'] = df_feat['month_num'].isin([3, 6, 9, 12]).astype(int)
        df_feat['is_quarter_start'] = df_feat['month_num'].isin([1, 4, 7, 10]).astype(int)
        df_feat['quarter_position'] = ((df_feat['month_num'] - 1) % 3) + 1
        
        df_feat['ly_same_month_revenue'] = df_feat.groupby('month_num')['actual_revenue'].shift(1)
        df_feat['ly_same_qtr_avg'] = df_feat.groupby(['quarter'])['actual_revenue'].transform(
            lambda x: x.shift(3).rolling(3, min_periods=1).mean()
        )
        
        df_feat['revenue_lag_1'] = df_feat['actual_revenue'].shift(1)
        df_feat['revenue_lag_2'] = df_feat['actual_revenue'].shift(2)
        df_feat['revenue_lag_3'] = df_feat['actual_revenue'].shift(3)
        df_feat['revenue_3mo_avg'] = df_feat['actual_revenue'].shift(1).rolling(3, min_periods=1).mean()
        df_feat['revenue_velocity'] = df_feat['revenue_lag_1'] - df_feat['revenue_lag_2']
        df_feat['revenue_acceleration'] = df_feat['revenue_velocity'] - df_feat['revenue_velocity'].shift(1)
        
        # YTD features
        df_feat['ytd_revenue'] = df_feat.groupby('year')['actual_revenue'].cumsum().shift(1)
        df_feat['ytd_avg'] = df_feat['ytd_revenue'] / (df_feat['month_num'] - 1).replace(0, 1)
        df_feat['perf_vs_ytd'] = ((df_feat['revenue_lag_1'] - df_feat['ytd_avg']) / 
                                  (df_feat['ytd_avg'] + 1e-10)).clip(-0.5, 0.5)
        
        # Quarter features
        df_feat['quarter_cumulative'] = df_feat.groupby(['year', 'quarter'])['actual_revenue'].cumsum().shift(1)
        df_feat['prev_quarter_avg'] = df_feat['actual_revenue'].shift(1).rolling(3, min_periods=1).mean().shift(2)
        df_feat['last_quarter_end_rev'] = df_feat['actual_revenue'].shift(1).where(
            df_feat['month_num'].shift(1).isin([3, 6, 9, 12])
        ).ffill()
        df_feat['qoq_change'] = ((df_feat['revenue_3mo_avg'] - df_feat['prev_quarter_avg']) / 
                                (df_feat['prev_quarter_avg'] + 1e-10)).clip(-0.5, 0.5)
        
        # Trend direction
        df_feat['revenue_6mo_avg'] = df_feat['actual_revenue'].shift(1).rolling(6, min_periods=1).mean()
        df_feat['trend_direction'] = np.sign(df_feat['revenue_3mo_avg'] - df_feat['revenue_6mo_avg'])
        
        df_feat['fcst_total_rem'] = (df_feat['committed_sign_revenue'] + 
                                      df_feat['committed_unsig_revenue'] + 
                                      df_feat['wtd_pipeline_revenue'])
        df_feat['fcst_signed_rem'] = df_feat['committed_sign_revenue']
        df_feat['fcst_unsigned_rem'] = df_feat['committed_unsig_revenue']
        df_feat['fcst_pipeline_rem'] = df_feat['wtd_pipeline_revenue']
        df_feat['signed_per_month'] = df_feat['fcst_signed_rem'] / df_feat['remaining_months'].replace(0, 1)
        
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
        
        logger.info(f"Created {len(self.feature_list)} features")
        
        return df_feat, self.feature_list


class FeatureSelector:
    """Selects top features using LassoCV with correlation filtering"""
    
    def __init__(self, n_features: int = 15, corr_threshold: float = 0.85):
        self.n_features = n_features
        self.corr_threshold = corr_threshold
        self.selected_features = []
        self.feature_importance = None
        self.optimal_alpha = None
    
    def _remove_correlated(self, X: pd.DataFrame) -> List[str]:
        """Remove features with correlation > threshold"""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        return to_drop
    
    def select_features(self, df: pd.DataFrame, feature_list: List[str]) -> List[str]:
        """
        Select top N features using LassoCV with correlation filtering.
        """
        # Training data for feature selection
        df_train = df[df['year'].isin([2023, 2024])].copy().dropna(subset=[config.TARGET])
        
        # Filter valid features (< 30% NaN)
        valid_features = [f for f in feature_list 
                        if f in df_train.columns and df_train[f].isna().mean() < 0.3]
        
        X = df_train[valid_features].fillna(df_train[valid_features].median())
        y = df_train[config.TARGET]
        
        # Remove correlated features
        corr_to_drop = self._remove_correlated(X)
        X_filtered = X.drop(columns=corr_to_drop, errors='ignore')
        filtered_features = X_filtered.columns.tolist()
        
        logger.info(f"Original features: {len(valid_features)}")
        logger.info(f"After correlation filter: {len(filtered_features)}")
        logger.info(f"Removed {len(corr_to_drop)} correlated features")
        
        # Scale and run LassoCV
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        alphas = np.logspace(-1, 4, 50)
        
        lasso_cv = LassoCV(alphas=alphas, cv=tscv, random_state=config.RANDOM_STATE, max_iter=10000)
        lasso_cv.fit(X_scaled, y)
        
        self.optimal_alpha = lasso_cv.alpha_
        logger.info(f"Optimal Alpha: {self.optimal_alpha:.2f}")
        
        # Rank features by coefficient magnitude
        self.feature_importance = pd.DataFrame({
            'feature': filtered_features,
            'coefficient': lasso_cv.coef_,
            'abs_coef': np.abs(lasso_cv.coef_)
        }).sort_values('abs_coef', ascending=False)
        
        # Select top N
        self.selected_features = self.feature_importance[
            self.feature_importance['abs_coef'] > 0
        ].head(self.n_features)['feature'].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features:")
        for i, row in self.feature_importance.head(self.n_features).iterrows():
            sign = '+' if row['coefficient'] > 0 else '-'
            logger.info(f"  {row['feature']:25} | Coef: {sign}{row['abs_coef']:>12,.0f}")
        
        return self.selected_features


class RevenueModelTrainer:
    """
    Main trainer for revenue forecasting model.
    - Batch training (initial training on historical data)
    - Retraining (every 6 months or yearly with new data)
    - Model comparison (RidgeCV, LassoCV, ElasticNetCV)
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector(
            n_features=config.N_TOP_FEATURES,
            corr_threshold=config.CORRELATION_THRESHOLD
        )
        
        self.model = None
        self.scaler = None
        self.selected_features = config.FINAL_FEATURES  # Use predefined features by default
        self.feature_medians = None
        self.metrics = {}
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load training data"""
        if file_path is None:
            file_path = config.DATA_DIR / config.DATA_FILE
        
        df = pd.read_csv(file_path, index_col=0)
        df = df.sort_values(['year', 'month_num']).reset_index(drop=True)
        df['month_id'] = df['year'] * 100 + df['month_num']
        
        logger.info(f"Loaded data: {df.shape[0]} rows")
        logger.info(f"Years: {sorted(df['year'].unique())}")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, train_years: List[int] = None, test_year: int = None):
        """Prepare data with features and train/test split"""
        
        # Create features
        df_feat, all_features = self.feature_engineer.create_features(df)
        
        # Determine years
        years = sorted(df_feat['year'].unique())
        if train_years is None:
            train_years = years[:-1] if len(years) > 1 else years
        if test_year is None:
            test_year = years[-1]
        
        train_df = df_feat[df_feat['year'].isin(train_years)].copy().dropna(subset=[config.TARGET])
        test_df = df_feat[df_feat['year'] == test_year].copy()
        
        logger.info(f"Training: {len(train_df)} samples ({train_years})")
        logger.info(f"Test: {len(test_df)} samples ({test_year})")
        
        return df_feat, train_df, test_df
    
    def _calc_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def train(self, df: pd.DataFrame = None, select_features: bool = False) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            df: Data to train on (loads default if None)
            select_features: If True, run feature selection. If False, use FINAL_FEATURES from config.
        
        Returns:
            Dict with training metrics (MAPE, MAE, RMSE)
        """
        if df is None:
            df = self.load_data()
        
        # Prepare data
        df_feat, train_df, test_df = self.prepare_data(df)
        
        # Feature selection (optional - use predefined by default)
        if select_features:
            self.selected_features = self.feature_selector.select_features(df_feat, config.ALL_FEATURES)
        else:
            self.selected_features = config.FINAL_FEATURES
            logger.info(f"Using predefined {len(self.selected_features)} features")
        
        # Prepare feature matrices
        X_train = train_df[self.selected_features].copy()
        y_train = train_df[config.TARGET].copy()
        
        # Store medians for imputation
        self.feature_medians = X_train.median()
        X_train = X_train.fillna(self.feature_medians)
        
        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train RidgeCV (best model from experiments)
        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        alphas = np.logspace(-1, 4, 50)
        
        self.model = RidgeCV(alphas=alphas, cv=tscv)
        self.model.fit(X_train_scaled, y_train)
        
        logger.info(f"Model trained - Optimal Alpha: {self.model.alpha_:.2f}")
        
        # Training metrics
        train_pred = self.model.predict(X_train_scaled)
        self.metrics = self._calc_metrics(y_train, train_pred)
        
        logger.info(f"Training Metrics:")
        logger.info(f"  MAPE: {self.metrics['mape']:.2f}%")
        logger.info(f"  MAE:  {self.metrics['mae']:,.0f}")
        logger.info(f"  RMSE: {self.metrics['rmse']:,.0f}")
        
        # Test metrics (if test data has actuals)
        test_df_with_actual = test_df.dropna(subset=[config.TARGET])
        if len(test_df_with_actual) > 0:
            X_test = test_df_with_actual[self.selected_features].fillna(self.feature_medians)
            X_test_scaled = self.scaler.transform(X_test)
            y_test = test_df_with_actual[config.TARGET]
            
            test_pred = self.model.predict(X_test_scaled)
            test_metrics = self._calc_metrics(y_test, test_pred)
            
            self.metrics['test_mape'] = test_metrics['mape']
            self.metrics['test_mae'] = test_metrics['mae']
            self.metrics['test_rmse'] = test_metrics['rmse']
            
            logger.info(f"Test Metrics:")
            logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
            logger.info(f"  MAE:  {test_metrics['mae']:,.0f}")
            logger.info(f"  RMSE: {test_metrics['rmse']:,.0f}")
        
        return self.metrics
    
    def retrain(self, new_data: pd.DataFrame) -> Dict[str, float]:
        """
        Retrain model with new data.used for periodic retraining (every 6 months or yearly).
        """
        logger.info("Starting model retraining...")
        return self.train(new_data, select_features=False)
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get all model artifacts for saving"""
        
        if self.model is None:
            raise ValueError("No trained model. Call train() first.")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.selected_features,
            'feature_medians': self.feature_medians.to_dict(),
            'metrics': self.metrics,
            'model_alpha': self.model.alpha_,
            'model_coef': self.model.coef_.tolist(),
            'training_date': datetime.now().isoformat()
        }
    
    def save(self, path: str = None) -> str:
        """Save model to pickle file"""
        
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = config.MODEL_DIR / f"{config.MODEL_NAME}_{timestamp}.pkl"
        
        artifacts = self.get_artifacts()
        
        with open(path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Model saved to: {path}")
        
        # Also save a summary
        summary_path = str(path).replace('.pkl', '_summary.json')
        summary = {
            'feature_names': artifacts['feature_names'],
            'metrics': artifacts['metrics'],
            'model_alpha': artifacts['model_alpha'],
            'training_date': artifacts['training_date']
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(path)


def main():
    """Run model training"""
    
    print("\n" + "=" * 60)
    print("REVENUE FORECASTING - MODEL TRAINING")
    print("=" * 60)
    
    trainer = RevenueModelTrainer()
    
    # Train model
    metrics = trainer.train()
    
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Model: RidgeCV (alpha={trainer.model.alpha_:.2f})")
    print(f"Features: {len(trainer.selected_features)}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"MAE:  {metrics['mae']:,.0f}")
    print(f"RMSE: {metrics['rmse']:,.0f}")
    
    # Save
    model_path = trainer.save()
    print(f"\nSaved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("FEATURES USED")
    print("=" * 60)
    for i, feat in enumerate(trainer.selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
