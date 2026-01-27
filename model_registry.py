"""
Model Loading Utilities for Inference - Unity Catalog
"""

import logging
import json

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_model(version: str = None):
    """
    Load model from Unity Catalog
    
    Args:
        version: Model version (latest if None)
    
    Returns:
        Loaded sklearn model
    """
    mlflow.set_registry_uri("databricks-uc")
    
    if version:
        model_uri = f"models:/{config.FULL_MODEL_NAME}/{version}"
    else:
        model_uri = f"models:/{config.FULL_MODEL_NAME}@latest"
    
    logger.info(f"Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    return model


def load_scaler(run_id: str):
    """Load scaler from run"""
    scaler_uri = f"runs:/{run_id}/scaler"
    
    logger.info(f"Loading scaler from run: {run_id}")
    scaler = mlflow.sklearn.load_model(scaler_uri)
    
    return scaler


def load_artifacts(run_id: str):
    """Load feature names and medians"""
    client = MlflowClient()
    
    path = client.download_artifacts(run_id, "feature_names.json")
    with open(path, 'r') as f:
        features = json.load(f)['features']
    
    path = client.download_artifacts(run_id, "feature_medians.json")
    with open(path, 'r') as f:
        medians = json.load(f)
    
    logger.info(f"Loaded {len(features)} features")
    return features, medians


def list_versions():
    """List all model versions"""
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    
    versions = client.search_model_versions(f"name='{config.FULL_MODEL_NAME}'")
    
    print(f"\nModel: {config.FULL_MODEL_NAME}")
    print("-" * 50)
    for v in versions:
        print(f"Version {v.version}: Run={v.run_id[:8]}...")
    
    return versions


# =============================================================================
# DATABRICKS NOTEBOOK USAGE (FOR INFERENCE)
# =============================================================================
"""
# In Databricks notebook:

from model_registry import load_model, load_scaler, load_artifacts
import config

# Load model (latest version)
model = load_model()

# Or load specific version
model = load_model(version="1")

# For full inference pipeline:
# 1. Load model and scaler
model = load_model()
scaler = load_scaler(run_id="your_run_id")  # Get run_id from training
features, medians = load_artifacts(run_id="your_run_id")

# 2. Prepare data and predict
X = df[features].fillna(medians)
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
"""
