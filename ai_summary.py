"""
AI Summary Module - Executive Summary Generation using SHAP, LIME, and LLM

This module provides functionality to:
1. Fetch historical data from Azure Blob Storage
2. Extract feature importance from trained models
3. Perform SHAP and LIME interpretability analysis
4. Generate executive summaries using LLM

Author: AI Assistant
Date: 2026-01-23
"""

import os
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class BlobStorageConfig:
    """Configuration for Azure Blob Storage connection."""
    connection_string: str = ""
    account_url: str = ""
    container_name: str = "fpna-data"
    credential: Optional[str] = None  # SAS token or account key
    
    def __post_init__(self):
        if not self.connection_string and not self.account_url:
            # Try to get from environment variables
            self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
            self.account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "")


@dataclass
class InterpretabilityConfig:
    """Configuration for SHAP and LIME analysis."""
    shap_max_samples: int = 100  # Max samples for SHAP background
    shap_explainer_type: str = "auto"  # auto, tree, kernel, linear
    lime_num_features: int = 10
    lime_num_samples: int = 5000
    top_n_features: int = 10  # Top N features to include in summary


@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    provider: str = "openai"  # openai, azure_openai, anthropic
    model: str = "gpt-4"
    api_key: str = ""
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")


@dataclass
class SummaryConfig:
    """Main configuration for executive summary generation."""
    blob_config: BlobStorageConfig = field(default_factory=BlobStorageConfig)
    interpretability_config: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    months_lookback: int = 3  # How many months of historical data
    prompts_dir: str = "prompts"  # Directory to store prompt files
    output_dir: str = "outputs"  # Directory for generated summaries


# =============================================================================
# Data Fetching Module
# =============================================================================

class DataFetcherBase(ABC):
    """Abstract base class for data fetching."""
    
    @abstractmethod
    def fetch_data(self, months: int = 3) -> pd.DataFrame:
        """Fetch historical data."""
        pass
    
    @abstractmethod
    def get_recent_rows(self, df: pd.DataFrame, months: int) -> pd.DataFrame:
        """Filter to recent rows based on date column."""
        pass


class BlobDataFetcher(DataFetcherBase):
    """Fetches data from Azure Blob Storage."""
    
    def __init__(self, config: BlobStorageConfig, date_column: str = "date"):
        self.config = config
        self.date_column = date_column
        self._client = None
        
    def _get_client(self):
        """Lazy initialization of blob client."""
        if self._client is None:
            try:
                from azure.storage.blob import BlobServiceClient
                
                if self.config.connection_string:
                    self._client = BlobServiceClient.from_connection_string(
                        self.config.connection_string
                    )
                elif self.config.account_url:
                    self._client = BlobServiceClient(
                        account_url=self.config.account_url,
                        credential=self.config.credential
                    )
                else:
                    raise ValueError(
                        "No Azure Blob Storage credentials provided. "
                        "Set connection_string or account_url in config."
                    )
            except ImportError:
                raise ImportError(
                    "azure-storage-blob package not installed. "
                    "Install with: pip install azure-storage-blob"
                )
        return self._client
    
    def list_blobs(self, prefix: str = "") -> List[str]:
        """List blobs in the container."""
        client = self._get_client()
        container = client.get_container_client(self.config.container_name)
        return [blob.name for blob in container.list_blobs(name_starts_with=prefix)]
    
    def fetch_data(
        self, 
        blob_name: str = "historical_data.csv",
        months: int = 3
    ) -> pd.DataFrame:
        """
        Fetch data from blob storage.
        
        Args:
            blob_name: Name of the blob file to fetch
            months: Number of months of historical data to return
            
        Returns:
            DataFrame with historical data
        """
        client = self._get_client()
        container = client.get_container_client(self.config.container_name)
        blob_client = container.get_blob_client(blob_name)
        
        logger.info(f"Fetching data from blob: {blob_name}")
        
        # Download blob content
        stream = io.BytesIO()
        blob_client.download_blob().readinto(stream)
        stream.seek(0)
        
        # Determine file type and read accordingly
        if blob_name.endswith('.csv'):
            df = pd.read_csv(stream)
        elif blob_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(stream)
        elif blob_name.endswith('.parquet'):
            df = pd.read_parquet(stream)
        else:
            df = pd.read_csv(stream)  # Default to CSV
        
        logger.info(f"Loaded {len(df)} rows from blob storage")
        
        # Filter to recent months
        return self.get_recent_rows(df, months)
    
    def get_recent_rows(self, df: pd.DataFrame, months: int) -> pd.DataFrame:
        """Filter DataFrame to rows from the last N months."""
        if self.date_column not in df.columns:
            logger.warning(
                f"Date column '{self.date_column}' not found. "
                f"Returning last {months * 30} rows as approximation."
            )
            return df.tail(months * 30)
        
        # Convert to datetime if needed
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
        
        # Calculate cutoff date
        cutoff = datetime.now() - timedelta(days=months * 30)
        mask = df[self.date_column] >= cutoff
        
        filtered = df[mask]
        logger.info(f"Filtered to {len(filtered)} rows from last {months} months")
        
        return filtered


class LocalDataFetcher(DataFetcherBase):
    """Fetches data from local filesystem (fallback/testing)."""
    
    def __init__(self, data_path: str, date_column: str = "date"):
        self.data_path = Path(data_path)
        self.date_column = date_column
    
    def fetch_data(self, months: int = 3) -> pd.DataFrame:
        """Fetch data from local file."""
        logger.info(f"Loading data from: {self.data_path}")
        
        if self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(self.data_path)
        elif self.data_path.suffix == '.parquet':
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        return self.get_recent_rows(df, months)
    
    def get_recent_rows(self, df: pd.DataFrame, months: int) -> pd.DataFrame:
        """Filter to recent rows."""
        if self.date_column not in df.columns:
            return df.tail(months * 30)
        
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
        cutoff = datetime.now() - timedelta(days=months * 30)
        
        return df[df[self.date_column] >= cutoff]


# =============================================================================
# Feature Importance Module
# =============================================================================

class FeatureImportanceExtractor:
    """Extracts feature importance from trained models."""
    
    def __init__(self, model_artifact: Dict[str, Any]):
        """
        Initialize with loaded model artifact.
        
        Args:
            model_artifact: Dictionary containing model, preprocessor, etc.
        """
        self.model = model_artifact.get('model')
        self.preprocessor = model_artifact.get('preprocessor')
        self.feature_selector = model_artifact.get('feature_selector')
        self.feature_names = model_artifact.get('feature_names', [])
        self.config = model_artifact.get('config')
        
        if self.model is None:
            raise ValueError("Model not found in artifact")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from the model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        importance_values = self._extract_importance()
        
        if importance_values is None:
            logger.warning("Could not extract feature importance from model")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names[:len(importance_values)],
            'importance': importance_values
        })
        
        # Sort by importance and add rank
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _extract_importance(self) -> Optional[np.ndarray]:
        """Extract importance values based on model type."""
        model = self.model
        
        # Tree-based models (RF, GBR, XGBoost, LightGBM, CatBoost)
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        
        # Linear models (coefficients)
        if hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        
        # Try to get from nested estimator
        if hasattr(model, 'estimator_') and hasattr(model.estimator_, 'feature_importances_'):
            return model.estimator_.feature_importances_
        
        return None
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        importance_df = self.get_feature_importance()
        
        if importance_df.empty:
            return []
        
        top = importance_df.head(n)
        return list(zip(top['feature'], top['importance']))
    
    def importance_summary(self) -> Dict[str, Any]:
        """Generate a summary of feature importance."""
        importance_df = self.get_feature_importance()
        
        if importance_df.empty:
            return {"error": "Could not extract feature importance"}
        
        return {
            "total_features": len(importance_df),
            "top_10_features": self.get_top_features(10),
            "importance_concentration": {
                "top_5_pct": importance_df.head(5)['importance'].sum() / importance_df['importance'].sum() * 100,
                "top_10_pct": importance_df.head(10)['importance'].sum() / importance_df['importance'].sum() * 100,
            },
            "full_importance_table": importance_df.to_dict('records')
        }


# =============================================================================
# SHAP Analysis Module
# =============================================================================

class SHAPAnalyzer:
    """Performs SHAP (SHapley Additive exPlanations) analysis."""
    
    def __init__(
        self, 
        model: Any, 
        feature_names: List[str],
        config: InterpretabilityConfig
    ):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self._explainer = None
        self._shap_values = None
        self._background_data = None
    
    def _create_explainer(self, X_background: np.ndarray):
        """Create appropriate SHAP explainer based on model type."""
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap package not installed. Install with: pip install shap"
            )
        
        model_name = type(self.model).__name__
        explainer_type = self.config.shap_explainer_type
        
        # Sample background data if too large
        if len(X_background) > self.config.shap_max_samples:
            indices = np.random.choice(
                len(X_background), 
                self.config.shap_max_samples, 
                replace=False
            )
            X_background = X_background[indices]
        
        self._background_data = X_background
        
        logger.info(f"Creating SHAP explainer for {model_name}")
        
        # Auto-select explainer type
        if explainer_type == "auto":
            # Tree-based models
            if model_name in ['RandomForestRegressor', 'GradientBoostingRegressor', 
                             'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor',
                             'ExtraTreesRegressor', 'DecisionTreeRegressor']:
                explainer_type = "tree"
            # Linear models
            elif model_name in ['LinearRegression', 'Ridge', 'Lasso', 
                               'ElasticNet', 'BayesianRidge', 'HuberRegressor']:
                explainer_type = "linear"
            else:
                explainer_type = "kernel"
        
        # Create explainer
        if explainer_type == "tree":
            self._explainer = shap.TreeExplainer(self.model)
        elif explainer_type == "linear":
            self._explainer = shap.LinearExplainer(self.model, X_background)
        else:
            # Kernel explainer as fallback (slower but universal)
            self._explainer = shap.KernelExplainer(
                self.model.predict, 
                shap.sample(X_background, min(100, len(X_background)))
            )
        
        return self._explainer
    
    def compute_shap_values(
        self, 
        X_train: np.ndarray, 
        X_explain: np.ndarray
    ) -> np.ndarray:
        """
        Compute SHAP values for explanation data.
        
        Args:
            X_train: Training data for background
            X_explain: Data to explain
            
        Returns:
            SHAP values array
        """
        if self._explainer is None:
            self._create_explainer(X_train)
        
        logger.info(f"Computing SHAP values for {len(X_explain)} samples")
        
        # Compute SHAP values
        self._shap_values = self._explainer.shap_values(X_explain)
        
        # Handle different SHAP value formats
        if isinstance(self._shap_values, list):
            self._shap_values = self._shap_values[0]
        
        return self._shap_values
    
    def get_global_importance(self) -> pd.DataFrame:
        """Get global feature importance from SHAP values."""
        if self._shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(self._shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names[:len(mean_abs_shap)],
            'shap_importance': mean_abs_shap
        })
        
        return df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
    
    def get_feature_interactions(self, top_n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Identify top feature interactions."""
        if self._shap_values is None or self._background_data is None:
            raise ValueError("SHAP values not computed.")
        
        # Simplified interaction detection via correlation of SHAP values
        interactions = {}
        shap_df = pd.DataFrame(
            self._shap_values, 
            columns=self.feature_names[:self._shap_values.shape[1]]
        )
        
        # Get top features
        importance = self.get_global_importance()
        top_features = importance.head(top_n)['feature'].tolist()
        
        for feature in top_features:
            if feature not in shap_df.columns:
                continue
            # Find features with highest absolute correlation in SHAP values
            correlations = shap_df.corr()[feature].abs().drop(feature).sort_values(ascending=False)
            interactions[feature] = list(correlations.head(3).items())
        
        return interactions
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive SHAP analysis summary."""
        global_importance = self.get_global_importance()
        
        return {
            "method": "SHAP",
            "samples_analyzed": len(self._shap_values) if self._shap_values is not None else 0,
            "top_features": global_importance.head(self.config.top_n_features).to_dict('records'),
            "feature_importance_distribution": {
                "mean": float(global_importance['shap_importance'].mean()),
                "std": float(global_importance['shap_importance'].std()),
                "max": float(global_importance['shap_importance'].max()),
                "min": float(global_importance['shap_importance'].min()),
            },
            "interpretation": self._generate_interpretation(global_importance)
        }
    
    def _generate_interpretation(self, importance_df: pd.DataFrame) -> str:
        """Generate human-readable interpretation."""
        top = importance_df.head(3)
        features = top['feature'].tolist()
        
        interpretation = f"The top 3 most influential features are: {', '.join(features)}. "
        
        # Check for feature concentration
        top_5_pct = importance_df.head(5)['shap_importance'].sum() / importance_df['shap_importance'].sum() * 100
        
        if top_5_pct > 70:
            interpretation += f"These features show high concentration ({top_5_pct:.1f}% in top 5), suggesting the model relies heavily on a few key predictors."
        else:
            interpretation += f"Feature importance is distributed across multiple features ({top_5_pct:.1f}% in top 5)."
        
        return interpretation


# =============================================================================
# LIME Analysis Module
# =============================================================================

class LIMEAnalyzer:
    """Performs LIME (Local Interpretable Model-agnostic Explanations) analysis."""
    
    def __init__(
        self, 
        model: Any, 
        feature_names: List[str],
        config: InterpretabilityConfig
    ):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self._explainer = None
        self._explanations = []
    
    def _create_explainer(self, X_train: np.ndarray):
        """Create LIME explainer."""
        try:
            from lime import lime_tabular
        except ImportError:
            raise ImportError(
                "lime package not installed. Install with: pip install lime"
            )
        
        self._explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            mode='regression',
            verbose=False
        )
        
        return self._explainer
    
    def explain_instances(
        self, 
        X_train: np.ndarray,
        X_explain: np.ndarray,
        num_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations for multiple instances.
        
        Args:
            X_train: Training data
            X_explain: Instances to explain
            num_samples: Number of instances to explain (default: all)
            
        Returns:
            List of explanation dictionaries
        """
        if self._explainer is None:
            self._create_explainer(X_train)
        
        if num_samples is None or num_samples > len(X_explain):
            num_samples = min(len(X_explain), 50)  # Limit for performance
        
        logger.info(f"Generating LIME explanations for {num_samples} instances")
        
        self._explanations = []
        
        for i in range(num_samples):
            try:
                exp = self._explainer.explain_instance(
                    X_explain[i],
                    self.model.predict,
                    num_features=self.config.lime_num_features,
                    num_samples=self.config.lime_num_samples
                )
                
                self._explanations.append({
                    'instance_idx': i,
                    'prediction': self.model.predict(X_explain[i:i+1])[0],
                    'local_explanation': dict(exp.as_list()),
                    'intercept': exp.intercept[0] if hasattr(exp, 'intercept') else None,
                    'score': exp.score if hasattr(exp, 'score') else None
                })
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {e}")
        
        return self._explanations
    
    def aggregate_explanations(self) -> pd.DataFrame:
        """Aggregate local explanations to get global feature importance."""
        if not self._explanations:
            raise ValueError("No explanations computed. Call explain_instances first.")
        
        # Collect all feature contributions
        all_contributions = {}
        
        for exp in self._explanations:
            for feature, contribution in exp['local_explanation'].items():
                if feature not in all_contributions:
                    all_contributions[feature] = []
                all_contributions[feature].append(abs(contribution))
        
        # Calculate mean absolute contribution
        aggregated = {
            'feature': list(all_contributions.keys()),
            'lime_importance': [np.mean(v) for v in all_contributions.values()],
            'lime_std': [np.std(v) for v in all_contributions.values()]
        }
        
        df = pd.DataFrame(aggregated)
        return df.sort_values('lime_importance', ascending=False).reset_index(drop=True)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive LIME analysis summary."""
        aggregated = self.aggregate_explanations()
        
        # Identify consistent vs variable features
        consistent_features = aggregated[
            aggregated['lime_std'] / aggregated['lime_importance'] < 0.5
        ]['feature'].tolist()[:5]
        
        variable_features = aggregated[
            aggregated['lime_std'] / aggregated['lime_importance'] >= 0.5
        ]['feature'].tolist()[:5]
        
        return {
            "method": "LIME",
            "instances_explained": len(self._explanations),
            "top_features": aggregated.head(self.config.top_n_features).to_dict('records'),
            "consistent_features": consistent_features,
            "variable_features": variable_features,
            "model_fidelity": {
                "mean_score": np.mean([e.get('score', 0) for e in self._explanations if e.get('score')]),
            },
            "interpretation": self._generate_interpretation(aggregated)
        }
    
    def _generate_interpretation(self, aggregated: pd.DataFrame) -> str:
        """Generate human-readable interpretation."""
        top = aggregated.head(3)
        
        interpretation = f"LIME analysis reveals that {top['feature'].iloc[0]} has the highest local importance "
        interpretation += f"(avg contribution: {top['lime_importance'].iloc[0]:.4f}). "
        
        # Check consistency
        if len(aggregated) > 0:
            high_variability = aggregated[
                aggregated['lime_std'] > aggregated['lime_importance'] * 0.5
            ]
            if len(high_variability) > 0:
                interpretation += f"Features with high variability across instances: {', '.join(high_variability['feature'].head(3).tolist())}."
        
        return interpretation


# =============================================================================
# Prompt Management Module
# =============================================================================

class PromptTemplate:
    """Represents a single prompt template."""
    
    def __init__(self, name: str, content: str, description: str = ""):
        self.name = name
        self.content = content
        self.description = description
    
    def format(self, **kwargs) -> str:
        """Format the prompt with provided variables."""
        return self.content.format(**kwargs)
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "content": self.content
        }
    
    def save(self, path: Path):
        """Save prompt to file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, path: Path) -> 'PromptTemplate':
        """Load prompt from file."""
        data = json.loads(path.read_text())
        return cls(**data)


class PromptManager:
    """Manages prompt templates for LLM interactions."""
    
    # Default prompt templates
    DEFAULT_LOGIC_PROMPT = """
You are an expert financial analyst AI assistant. Your task is to analyze model predictions 
and interpretability results to generate insights about revenue forecasting.

## Analysis Context
- Model Type: {model_type}
- Target Variable: {target_variable}
- Analysis Period: Last {months} months
- Total Records Analyzed: {total_records}

## Feature Importance Summary
The model uses {num_features} features. Top contributing features:
{feature_importance_text}

## SHAP Analysis Results
{shap_summary}

## LIME Analysis Results  
{lime_summary}

## Key Metrics
- Model Performance (RMSE): {rmse}
- R-squared: {r2}
- Mean Absolute Percentage Error: {mape}%

## Data Patterns
{data_patterns}

Based on this analysis, identify:
1. The primary drivers of revenue predictions
2. Feature interactions and their business implications
3. Areas of model confidence vs uncertainty
4. Actionable recommendations for stakeholders
"""

    DEFAULT_FORMAT_PROMPT = """
## Output Format Requirements

Structure your executive summary as follows:

### 1. Executive Overview (2-3 sentences)
High-level summary of model performance and key findings.

### 2. Key Drivers Analysis
- List top 5 drivers with their business interpretation
- Explain directional impact (positive/negative influence)
- Quantify relative importance percentages

### 3. Insights & Patterns
- Identify seasonal/temporal patterns if any
- Note any unusual relationships or interactions
- Highlight data quality considerations

### 4. Recommendations
- List 3-5 actionable recommendations
- Prioritize by potential impact
- Include specific metrics to monitor

### 5. Risk Factors & Caveats
- Model limitations
- Data gaps or quality issues
- External factors not captured

Use clear, professional language suitable for C-level executives.
Avoid technical jargon unless necessary; explain terms when used.
Include specific numbers and percentages to support claims.
"""

    DEFAULT_INSTRUCTIONS_PROMPT = """
## Instructions and Constraints

### DO:
- Provide specific, quantified insights (e.g., "Feature X contributes 23% to predictions")
- Connect technical findings to business implications
- Acknowledge uncertainty where model confidence is low
- Use professional, executive-appropriate language
- Structure information hierarchically from most to least important
- Include both positive findings and areas of concern

### DO NOT:
- Use overly technical ML terminology without explanation
- Make predictions beyond the model's scope
- Ignore contradictions between SHAP and LIME results
- Provide recommendations without supporting evidence
- Use vague language like "might" or "could be" excessively
- Exceed 800 words for the main summary

### SPECIAL HANDLING:
- If SHAP and LIME show conflicting results, note this and explain potential reasons
- If a feature's importance seems counterintuitive, investigate and explain
- If data quality issues are detected, flag them prominently
- Format numbers consistently (2 decimal places for percentages, appropriate rounding)

### BUSINESS CONTEXT:
This analysis supports FP&A (Financial Planning & Analysis) processes.
Revenue forecasting accuracy directly impacts budgeting, resource allocation, 
and strategic planning decisions.
"""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts: Dict[str, PromptTemplate] = {}
        
        # Initialize with defaults
        self._init_default_prompts()
    
    def _init_default_prompts(self):
        """Initialize default prompt templates."""
        self.prompts = {
            'logic': PromptTemplate(
                name='logic',
                content=self.DEFAULT_LOGIC_PROMPT,
                description='Main analysis logic and context'
            ),
            'format': PromptTemplate(
                name='format',
                content=self.DEFAULT_FORMAT_PROMPT,
                description='Output structure and formatting requirements'
            ),
            'instructions': PromptTemplate(
                name='instructions',
                content=self.DEFAULT_INSTRUCTIONS_PROMPT,
                description='Constraints and special handling rules'
            )
        }
    
    def get_prompt(self, name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found. Available: {list(self.prompts.keys())}")
        return self.prompts[name]
    
    def set_prompt(self, name: str, content: str, description: str = ""):
        """Set or update a prompt template."""
        self.prompts[name] = PromptTemplate(name, content, description)
    
    def save_prompts(self):
        """Save all prompts to files."""
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        for name, prompt in self.prompts.items():
            filepath = self.prompts_dir / f"{name}_prompt.json"
            prompt.save(filepath)
            logger.info(f"Saved prompt: {filepath}")
    
    def load_prompts(self):
        """Load prompts from files."""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return
        
        for filepath in self.prompts_dir.glob("*_prompt.json"):
            try:
                prompt = PromptTemplate.load(filepath)
                self.prompts[prompt.name] = prompt
                logger.info(f"Loaded prompt: {filepath}")
            except Exception as e:
                logger.error(f"Failed to load prompt {filepath}: {e}")
    
    def build_full_prompt(self, **kwargs) -> str:
        """Build the complete prompt combining all templates."""
        # Format each prompt template
        logic = self.prompts['logic'].format(**kwargs)
        format_guide = self.prompts['format'].content
        instructions = self.prompts['instructions'].content
        
        return f"{logic}\n\n{format_guide}\n\n{instructions}"


# =============================================================================
# LLM Interface Module
# =============================================================================

class LLMInterface(ABC):
    """Abstract base class for LLM interactions."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        pass


class OpenAIInterface(LLMInterface):
    """Interface for OpenAI API."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI, AzureOpenAI
                
                if self.config.provider == "azure_openai":
                    self._client = AzureOpenAI(
                        api_key=self.config.api_key,
                        api_version=self.config.api_version or "2024-02-01",
                        azure_endpoint=self.config.api_base
                    )
                else:
                    self._client = OpenAI(api_key=self.config.api_key)
                    
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
        return self._client
    
    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        client = self._get_client()
        
        logger.info("Generating executive summary via LLM...")
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are an expert financial analyst creating executive summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing without API calls."""
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response."""
        return """
# Executive Summary - Revenue Forecast Analysis

## Executive Overview
The revenue forecasting model demonstrates strong predictive performance with key drivers 
identified through comprehensive SHAP and LIME analysis. The model shows particular strength 
in capturing historical revenue patterns and pipeline metrics.

## Key Drivers Analysis
1. **Historical Revenue (LM1, LY1)** - 35% importance: Past month and prior year actuals 
   are strongest predictors
2. **Pipeline Metrics** - 25% importance: Committed revenue ratios provide forward-looking signals
3. **YTD Performance** - 20% importance: Year-to-date trends capture momentum
4. **Forecast Accuracy** - 15% importance: Previous forecast gaps inform adjustments
5. **Seasonality Indicators** - 5% importance: Year-end patterns affect predictions

## Recommendations
1. Focus on improving pipeline data quality for better forward visibility
2. Monitor forecast gap metrics closely for early warning signals
3. Consider seasonal adjustments in Q4 predictions
4. Review anomalies in historical comparisons for data quality

## Risk Factors
- Model may underperform during unusual market conditions
- Pipeline data quality directly impacts prediction accuracy
- Historical patterns may not capture rapid market changes

*This is a mock summary for testing purposes.*
"""


# =============================================================================
# Executive Summary Generator (Main Orchestrator)
# =============================================================================

class ExecutiveSummaryGenerator:
    """
    Main orchestrator class that combines all components to generate
    executive summaries from model predictions.
    """
    
    def __init__(self, config: SummaryConfig):
        self.config = config
        self.prompt_manager = PromptManager(config.prompts_dir)
        self._model_artifact = None
        self._data = None
        self._feature_extractor = None
        self._shap_analyzer = None
        self._lime_analyzer = None
        self._llm_interface = None
        
        # Results storage
        self.shap_results = None
        self.lime_results = None
        self.feature_importance = None
        self.executive_summary = None
    
    def load_model(self, model_path: str) -> 'ExecutiveSummaryGenerator':
        """
        Load the trained model artifact.
        
        Args:
            model_path: Path to the joblib model file
            
        Returns:
            self for method chaining
        """
        logger.info(f"Loading model from: {model_path}")
        self._model_artifact = joblib.load(model_path)
        
        self._feature_extractor = FeatureImportanceExtractor(self._model_artifact)
        
        # Initialize analyzers
        model = self._model_artifact['model']
        feature_names = self._model_artifact.get('feature_names', [])
        
        self._shap_analyzer = SHAPAnalyzer(
            model, feature_names, self.config.interpretability_config
        )
        self._lime_analyzer = LIMEAnalyzer(
            model, feature_names, self.config.interpretability_config
        )
        
        return self
    
    def fetch_data(
        self, 
        source: str = "blob",
        blob_name: str = "historical_data.csv",
        local_path: str = None,
        date_column: str = "date"
    ) -> 'ExecutiveSummaryGenerator':
        """
        Fetch historical data from specified source.
        
        Args:
            source: "blob" or "local"
            blob_name: Name of blob file (if source="blob")
            local_path: Path to local file (if source="local")
            date_column: Name of date column for filtering
            
        Returns:
            self for method chaining
        """
        if source == "blob":
            fetcher = BlobDataFetcher(self.config.blob_config, date_column)
            self._data = fetcher.fetch_data(blob_name, self.config.months_lookback)
        elif source == "local":
            if local_path is None:
                raise ValueError("local_path required when source='local'")
            fetcher = LocalDataFetcher(local_path, date_column)
            self._data = fetcher.fetch_data(self.config.months_lookback)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'blob' or 'local'")
        
        return self
    
    def set_data(self, df: pd.DataFrame) -> 'ExecutiveSummaryGenerator':
        """
        Directly set the data DataFrame.
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            self for method chaining
        """
        self._data = df.copy()
        return self
    
    def extract_feature_importance(self) -> Dict[str, Any]:
        """Extract and return feature importance from the model."""
        if self._feature_extractor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.feature_importance = self._feature_extractor.importance_summary()
        return self.feature_importance
    
    def run_shap_analysis(self, X_train: np.ndarray, X_explain: np.ndarray) -> Dict[str, Any]:
        """
        Run SHAP analysis.
        
        Args:
            X_train: Training data for background
            X_explain: Data to explain
            
        Returns:
            SHAP analysis summary
        """
        if self._shap_analyzer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self._shap_analyzer.compute_shap_values(X_train, X_explain)
        self.shap_results = self._shap_analyzer.generate_summary()
        
        return self.shap_results
    
    def run_lime_analysis(self, X_train: np.ndarray, X_explain: np.ndarray) -> Dict[str, Any]:
        """
        Run LIME analysis.
        
        Args:
            X_train: Training data
            X_explain: Data to explain
            
        Returns:
            LIME analysis summary
        """
        if self._lime_analyzer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self._lime_analyzer.explain_instances(X_train, X_explain)
        self.lime_results = self._lime_analyzer.generate_summary()
        
        return self.lime_results
    
    def prepare_data_for_analysis(
        self, 
        df: pd.DataFrame,
        target_col: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for SHAP/LIME analysis using the model's preprocessor.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (X_processed, y)
        """
        if self._model_artifact is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        preprocessor = self._model_artifact.get('preprocessor')
        feature_selector = self._model_artifact.get('feature_selector')
        config = self._model_artifact.get('config')
        
        # Separate features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col].values
        else:
            X = df
            y = None
        
        # Remove date column if present
        if config and hasattr(config, 'date_col') and config.date_col in X.columns:
            X = X.drop(columns=[config.date_col])
        
        # Apply preprocessing
        if preprocessor is not None:
            X_processed = preprocessor.transform(X)
        else:
            X_processed = X.values
        
        # Apply feature selection
        if feature_selector is not None:
            X_processed = feature_selector.transform(X_processed)
        
        return X_processed, y
    
    def _format_feature_importance_text(self) -> str:
        """Format feature importance for prompt."""
        if self.feature_importance is None:
            return "Feature importance not available."
        
        top_features = self.feature_importance.get('top_10_features', [])
        lines = []
        for i, (feature, importance) in enumerate(top_features, 1):
            lines.append(f"{i}. {feature}: {importance:.4f}")
        
        return "\n".join(lines)
    
    def _format_shap_summary(self) -> str:
        """Format SHAP results for prompt."""
        if self.shap_results is None:
            return "SHAP analysis not performed."
        
        lines = [
            f"Samples analyzed: {self.shap_results.get('samples_analyzed', 0)}",
            f"Interpretation: {self.shap_results.get('interpretation', 'N/A')}",
            "\nTop SHAP features:"
        ]
        
        for feat in self.shap_results.get('top_features', [])[:5]:
            lines.append(f"  - {feat['feature']}: {feat['shap_importance']:.4f}")
        
        return "\n".join(lines)
    
    def _format_lime_summary(self) -> str:
        """Format LIME results for prompt."""
        if self.lime_results is None:
            return "LIME analysis not performed."
        
        lines = [
            f"Instances explained: {self.lime_results.get('instances_explained', 0)}",
            f"Interpretation: {self.lime_results.get('interpretation', 'N/A')}",
            "\nTop LIME features:"
        ]
        
        for feat in self.lime_results.get('top_features', [])[:5]:
            lines.append(f"  - {feat['feature']}: {feat['lime_importance']:.4f}")
        
        return "\n".join(lines)
    
    def _analyze_data_patterns(self, df: pd.DataFrame, target_col: str) -> str:
        """Analyze patterns in the data."""
        if df is None or df.empty:
            return "No data available for pattern analysis."
        
        patterns = []
        
        if target_col in df.columns:
            target = df[target_col]
            patterns.append(f"Target variable statistics:")
            patterns.append(f"  - Mean: {target.mean():,.2f}")
            patterns.append(f"  - Std: {target.std():,.2f}")
            patterns.append(f"  - Min: {target.min():,.2f}")
            patterns.append(f"  - Max: {target.max():,.2f}")
            patterns.append(f"  - Trend (recent vs overall): {'Increasing' if target.tail(10).mean() > target.mean() else 'Decreasing'}")
        
        patterns.append(f"\nData characteristics:")
        patterns.append(f"  - Total records: {len(df)}")
        patterns.append(f"  - Features: {len(df.columns)}")
        patterns.append(f"  - Missing values: {df.isnull().sum().sum()}")
        
        return "\n".join(patterns)
    
    def generate_summary(
        self,
        model_metrics: Optional[Dict[str, float]] = None,
        use_mock_llm: bool = False
    ) -> str:
        """
        Generate the executive summary using LLM.
        
        Args:
            model_metrics: Optional dict with 'rmse', 'r2', 'mape' keys
            use_mock_llm: If True, use mock LLM (for testing)
            
        Returns:
            Generated executive summary text
        """
        # Initialize LLM interface
        if use_mock_llm:
            self._llm_interface = MockLLMInterface()
        else:
            self._llm_interface = OpenAIInterface(self.config.llm_config)
        
        # Prepare prompt variables
        model_type = type(self._model_artifact['model']).__name__ if self._model_artifact else "Unknown"
        config = self._model_artifact.get('config') if self._model_artifact else None
        target_var = config.target if config and hasattr(config, 'target') else "revenue"
        
        metrics = model_metrics or {'rmse': 0, 'r2': 0, 'mape': 0}
        
        prompt_vars = {
            'model_type': model_type,
            'target_variable': target_var,
            'months': self.config.months_lookback,
            'total_records': len(self._data) if self._data is not None else 0,
            'num_features': len(self._model_artifact.get('feature_names', [])) if self._model_artifact else 0,
            'feature_importance_text': self._format_feature_importance_text(),
            'shap_summary': self._format_shap_summary(),
            'lime_summary': self._format_lime_summary(),
            'rmse': metrics.get('rmse', 0),
            'r2': metrics.get('r2', 0),
            'mape': metrics.get('mape', 0),
            'data_patterns': self._analyze_data_patterns(self._data, target_var) if self._data is not None else "N/A"
        }
        
        # Build full prompt
        full_prompt = self.prompt_manager.build_full_prompt(**prompt_vars)
        
        # Generate summary
        self.executive_summary = self._llm_interface.generate(full_prompt)
        
        return self.executive_summary
    
    def save_prompts(self):
        """Save prompt templates to files."""
        self.prompt_manager.save_prompts()
    
    def save_summary(self, filename: str = "executive_summary.md"):
        """Save the generated summary to file."""
        if self.executive_summary is None:
            raise ValueError("No summary generated. Call generate_summary() first.")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        filepath.write_text(self.executive_summary)
        
        logger.info(f"Summary saved to: {filepath}")
        return str(filepath)
    
    def save_analysis_results(self, filename: str = "analysis_results.json"):
        """Save all analysis results to JSON."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'feature_importance': self.feature_importance,
            'shap_results': self.shap_results,
            'lime_results': self.lime_results,
            'config': {
                'months_lookback': self.config.months_lookback,
                'shap_max_samples': self.config.interpretability_config.shap_max_samples,
                'lime_num_features': self.config.interpretability_config.lime_num_features
            }
        }
        
        filepath = output_dir / filename
        filepath.write_text(json.dumps(results, indent=2, default=str))
        
        logger.info(f"Analysis results saved to: {filepath}")
        return str(filepath)
    
    def run_full_pipeline(
        self,
        model_path: str,
        data_source: str = "local",
        data_path: str = None,
        target_col: str = "revenue",
        model_metrics: Optional[Dict[str, float]] = None,
        use_mock_llm: bool = False
    ) -> str:
        """
        Run the complete analysis pipeline.
        
        Args:
            model_path: Path to model artifact
            data_source: "blob" or "local"
            data_path: Path to data file (for local) or blob name (for blob)
            target_col: Name of target column
            model_metrics: Model performance metrics
            use_mock_llm: Use mock LLM for testing
            
        Returns:
            Generated executive summary
        """
        logger.info("Starting full analysis pipeline...")
        
        # Step 1: Load model
        self.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Step 2: Fetch/load data
        if data_source == "local" and data_path:
            self.fetch_data(source="local", local_path=data_path)
        elif data_source == "blob" and data_path:
            self.fetch_data(source="blob", blob_name=data_path)
        else:
            logger.warning("No data source specified, skipping data loading")
        
        # Step 3: Extract feature importance
        self.extract_feature_importance()
        logger.info("Feature importance extracted")
        
        # Step 4: Prepare data for analysis
        if self._data is not None and len(self._data) > 0:
            X_processed, _ = self.prepare_data_for_analysis(self._data, target_col)
            
            # Split for train/explain (use first 80% as train background)
            n_train = int(len(X_processed) * 0.8)
            X_train = X_processed[:n_train]
            X_explain = X_processed[n_train:]
            
            if len(X_explain) == 0:
                X_explain = X_processed[-10:]  # Use last 10 if no split possible
            
            # Step 5: Run SHAP analysis
            try:
                self.run_shap_analysis(X_train, X_explain)
                logger.info("SHAP analysis completed")
            except Exception as e:
                logger.error(f"SHAP analysis failed: {e}")
            
            # Step 6: Run LIME analysis
            try:
                self.run_lime_analysis(X_train, X_explain)
                logger.info("LIME analysis completed")
            except Exception as e:
                logger.error(f"LIME analysis failed: {e}")
        
        # Step 7: Generate summary
        summary = self.generate_summary(model_metrics, use_mock_llm)
        logger.info("Executive summary generated")
        
        # Step 8: Save outputs
        self.save_prompts()
        self.save_summary()
        self.save_analysis_results()
        
        return summary


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_config(
    azure_connection_string: str = "",
    openai_api_key: str = "",
    months_lookback: int = 3
) -> SummaryConfig:
    """Create a default configuration."""
    return SummaryConfig(
        blob_config=BlobStorageConfig(connection_string=azure_connection_string),
        interpretability_config=InterpretabilityConfig(),
        llm_config=LLMConfig(api_key=openai_api_key),
        months_lookback=months_lookback
    )


def quick_summary(
    model_path: str,
    data_path: str,
    target_col: str = "revenue",
    use_mock_llm: bool = True
) -> str:
    """
    Quick function to generate summary with minimal configuration.
    
    Args:
        model_path: Path to model artifact
        data_path: Path to data file
        target_col: Target column name
        use_mock_llm: Use mock LLM (default True for testing)
        
    Returns:
        Executive summary text
    """
    config = create_default_config()
    generator = ExecutiveSummaryGenerator(config)
    
    return generator.run_full_pipeline(
        model_path=model_path,
        data_source="local",
        data_path=data_path,
        target_col=target_col,
        use_mock_llm=use_mock_llm
    )


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("AI Summary Module - Executive Summary Generation")
    print("=" * 50)
    
    # Configuration
    config = SummaryConfig(
        blob_config=BlobStorageConfig(
            # Set via environment variables or directly
            connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        ),
        interpretability_config=InterpretabilityConfig(
            shap_max_samples=100,
            lime_num_features=10,
            top_n_features=10
        ),
        llm_config=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "")
        ),
        months_lookback=3,
        prompts_dir="prompts",
        output_dir="outputs"
    )
    
    # Initialize generator
    generator = ExecutiveSummaryGenerator(config)
    
    # Example: Run with local data and mock LLM (for testing)
    model_path = "model/reg_revenue_sklearn.joblib"
    
    # Check if model exists
    if Path(model_path).exists():
        print(f"\nLoading model from: {model_path}")
        generator.load_model(model_path)
        
        # Extract feature importance
        importance = generator.extract_feature_importance()
        print("\nFeature Importance Summary:")
        print(f"  Total features: {importance.get('total_features', 0)}")
        print("  Top 5 features:")
        for feat, imp in importance.get('top_10_features', [])[:5]:
            print(f"    - {feat}: {imp:.4f}")
        
        # Save prompts
        generator.save_prompts()
        print(f"\nPrompts saved to: {config.prompts_dir}/")
        
        # Generate summary with mock LLM (for testing without API)
        print("\nGenerating executive summary (mock LLM)...")
        summary = generator.generate_summary(
            model_metrics={'rmse': 1500.0, 'r2': 0.85, 'mape': 5.2},
            use_mock_llm=True
        )
        
        print("\n" + "=" * 50)
        print("EXECUTIVE SUMMARY")
        print("=" * 50)
        print(summary)
        
        # Save outputs
        generator.save_summary()
        generator.save_analysis_results()
        
    else:
        print(f"\nModel not found at: {model_path}")
        print("Please train the model first using training.py")
        
        # Still save prompts for reference
        generator.save_prompts()
        print(f"\nPrompt templates saved to: {config.prompts_dir}/")
