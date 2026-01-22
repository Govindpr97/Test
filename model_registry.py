"""
Model Hosting and Registration Script for Revenue Forecasting
Version: 1.0.0
"""

import os
import json
import pickle
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalStorage:
    """
    Local file storage for models.
    
    Current approach - stores pickle files locally.
    Will be replaced by MLflow when Unity Catalog is ready.
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else config.MODEL_DIR
        self.registry_file = self.base_path / "registry.json"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': {}}
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def save(self, artifacts: Dict, name: str, version: str, description: str = "") -> str:
        """Save model artifacts"""
        
        model_dir = self.base_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        # Save metadata
        metadata = {
            'name': name,
            'version': version,
            'description': description,
            'path': str(model_path),
            'created': datetime.now().isoformat(),
            'metrics': artifacts.get('metrics', {}),
            'features': artifacts.get('feature_names', [])
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        if name not in self.registry['models']:
            self.registry['models'][name] = {'versions': {}}
        
        self.registry['models'][name]['versions'][version] = {
            'path': str(model_path),
            'created': metadata['created']
        }
        self.registry['models'][name]['latest'] = version
        
        self._save_registry()
        
        logger.info(f"Saved: {model_path}")
        return str(model_path)
    
    def load(self, name: str, version: str = None) -> Dict:
        """Load model artifacts"""
        
        if name not in self.registry['models']:
            raise ValueError(f"Model '{name}' not found")
        
        if version is None:
            version = self.registry['models'][name].get('latest')
        
        if version not in self.registry['models'][name]['versions']:
            raise ValueError(f"Version '{version}' not found")
        
        model_path = self.registry['models'][name]['versions'][version]['path']
        
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        logger.info(f"Loaded: {model_path}")
        return artifacts
    
    def list_models(self) -> List[Dict]:
        """List all models"""
        models = []
        
        for name, info in self.registry['models'].items():
            for version, version_info in info['versions'].items():
                models.append({
                    'name': name,
                    'version': version,
                    'path': version_info['path'],
                    'created': version_info['created'],
                    'is_latest': version == info.get('latest')
                })
        
        return models
    
    def delete(self, name: str, version: str = None) -> bool:
        """Delete a model"""
        
        if name not in self.registry['models']:
            return False
        
        if version is None:
            # Delete all versions
            model_dir = self.base_path / name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            del self.registry['models'][name]
        else:
            # Delete specific version
            if version not in self.registry['models'][name]['versions']:
                return False
            
            version_dir = self.base_path / name / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            del self.registry['models'][name]['versions'][version]
            
            # Update latest
            if self.registry['models'][name]['versions']:
                versions = list(self.registry['models'][name]['versions'].keys())
                self.registry['models'][name]['latest'] = versions[-1]
            else:
                del self.registry['models'][name]
        
        self._save_registry()
        logger.info(f"Deleted: {name}" + (f" v{version}" if version else ""))
        return True
    
    def get_latest(self, name: str) -> Optional[str]:
        """Get latest version"""
        if name not in self.registry['models']:
            return None
        return self.registry['models'][name].get('latest')


class MLflowStorage:
    """
    MLflow storage for Unity Catalog integration.
    
    Will be used when Unity Catalog is provisioned.
    Currently placeholder implementation.
    """
    
    def __init__(self, tracking_uri: str = None, experiment: str = None):
        self.tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
        self.experiment = experiment or config.MLFLOW_EXPERIMENT
        self._available = False
        
        self._init_mlflow()
    
    def _init_mlflow(self):
        """Initialize MLflow"""
        if not self.tracking_uri:
            logger.warning("MLflow URI not set - using local storage")
            return
        
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            
            try:
                mlflow.set_experiment(self.experiment)
            except:
                mlflow.create_experiment(self.experiment)
                mlflow.set_experiment(self.experiment)
            
            self._available = True
            logger.info(f"MLflow initialized: {self.tracking_uri}")
            
        except ImportError:
            logger.warning("MLflow not installed")
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}")
    
    def save(self, artifacts: Dict, name: str, version: str, description: str = "") -> str:
        """Save to MLflow"""
        
        if not self._available:
            raise RuntimeError("MLflow not available")
        
        import mlflow
        import mlflow.sklearn
        
        with mlflow.start_run(run_name=f"{name}_{version}"):
            # Log metrics
            for key, val in artifacts.get('metrics', {}).items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, val)
            
            # Log model
            model = artifacts.get('model')
            if model:
                mlflow.sklearn.log_model(model, "model", registered_model_name=name)
            
            # Log scaler
            scaler = artifacts.get('scaler')
            if scaler:
                mlflow.sklearn.log_model(scaler, "scaler")
            
            # Log features
            mlflow.log_dict({'features': artifacts.get('feature_names', [])}, "features.json")
            mlflow.log_dict({'medians': artifacts.get('feature_medians', {})}, "medians.json")
            
            mlflow.set_tag("version", version)
            mlflow.set_tag("description", description)
            
            model_uri = f"models:/{name}/latest"
            logger.info(f"Saved to MLflow: {model_uri}")
            return model_uri
    
    def load(self, name: str, version: str = None) -> Dict:
        """Load from MLflow"""
        
        if not self._available:
            raise RuntimeError("MLflow not available")
        
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        
        if version is None:
            versions = client.get_latest_versions(name)
            if not versions:
                raise ValueError(f"No versions for '{name}'")
            version = versions[0].version
        
        model_uri = f"models:/{name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        return {'model': model, 'model_name': name, 'version': version}
    
    def list_models(self) -> List[Dict]:
        """List MLflow models"""
        
        if not self._available:
            return []
        
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        models = []
        
        try:
            for rm in client.search_registered_models():
                for v in rm.latest_versions:
                    models.append({
                        'name': rm.name,
                        'version': v.version,
                        'created': str(v.creation_timestamp)
                    })
        except Exception as e:
            logger.error(f"Error listing: {e}")
        
        return models


class ModelRegistry:
    """
    Unified model registry.
    
    Supports:
    - Local storage (current - pickle files)
    - MLflow (future - when Unity Catalog ready)
    """
    
    def __init__(self):
        self.local = LocalStorage()
        self.mlflow = MLflowStorage() if config.MLFLOW_TRACKING_URI else None
        self._use_mlflow = False
    
    def use_mlflow(self, enable: bool = True):
        """Switch to MLflow storage"""
        if enable and self.mlflow and self.mlflow._available:
            self._use_mlflow = True
            logger.info("Using MLflow storage")
        else:
            self._use_mlflow = False
            logger.info("Using local storage")
    
    @property
    def storage(self):
        """Get active storage backend"""
        if self._use_mlflow and self.mlflow:
            return self.mlflow
        return self.local
    
    def register(
        self, 
        artifacts: Dict, 
        name: str = None, 
        version: str = None, 
        description: str = ""
    ) -> str:
        """
        Register a model.
        
        Args:
            artifacts: Model artifacts from trainer.get_artifacts()
            name: Model name (default from config)
            version: Version string (auto-generated if None)
            description: Optional description
        
        Returns:
            Path/URI where saved
        """
        name = name or config.MODEL_NAME
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Registering: {name} v{version}")
        
        return self.storage.save(artifacts, name, version, description)
    
    def load(self, name: str = None, version: str = None) -> Dict:
        """Load a model"""
        name = name or config.MODEL_NAME
        return self.storage.load(name, version)
    
    def list(self) -> List[Dict]:
        """List all models"""
        return self.storage.list_models()
    
    def delete(self, name: str = None, version: str = None) -> bool:
        """Delete a model"""
        name = name or config.MODEL_NAME
        return self.local.delete(name, version)
    
    def get_latest(self, name: str = None) -> Optional[str]:
        """Get latest version"""
        name = name or config.MODEL_NAME
        return self.local.get_latest(name)


def main():
    """Demo model registry"""
    
    print("\n" + "=" * 60)
    print("MODEL REGISTRY DEMO")
    print("=" * 60)
    
    registry = ModelRegistry()
    
    # Sample artifacts (normally from trainer)
    sample = {
        'model': None,
        'scaler': None,
        'feature_names': config.FINAL_FEATURES,
        'feature_medians': {'ytd_revenue': 500000000, 'remaining_months': 6},
        'metrics': {'mape': 1.98, 'mae': 3498976, 'rmse': 4005432}
    }
    
    # Register
    print("\nRegistering sample model...")
    path = registry.register(sample, description="Sample model")
    print(f"Saved to: {path}")
    
    # List
    print("\nRegistered models:")
    for m in registry.list():
        status = " (latest)" if m.get('is_latest') else ""
        print(f"  - {m['name']} v{m['version']}{status}")
    
    # Get latest
    latest = registry.get_latest()
    print(f"\nLatest version: {latest}")
    
    print("\n" + "=" * 60)
    
    return registry


if __name__ == "__main__":
    registry = main()
