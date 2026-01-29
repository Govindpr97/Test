# Inputs Required for SHAP and LIME Analysis in Databricks

This document outlines all the inputs required for the `databricks_shap_lime_analysis.py` script.

## 1. Function Parameters

When calling `run_shap_lime_analysis()`, you need to provide:

### Required Parameters:
- **`year`** (int): Year for analysis (e.g., 2024)
- **`month`** (int): Month for analysis (1-12)

### Optional Parameters:
- **`shap_container`** (str): Azure Blob Storage container name for SHAP results (default: "shap-results")
- **`lime_container`** (str): Azure Blob Storage container name for LIME results (default: "lime-results")
- **`storage_account`** (str): Azure Storage account name (optional if using dbutils)
- **`account_key`** (str): Azure Storage account key (optional if using dbutils)
- **`config`** (InterpretabilityConfig): Configuration object (uses defaults if None)

## 2. Model and Data Requirements (from `load_model_and_data()`)

You must implement `load_model_and_data()` to return the following **13 items**:

### Model Objects:
1. **`model`** (Any): Your trained ML model object
   - Must have `.predict()` method
   - Should have `.feature_importances_` attribute (for tree-based models) or `.coef_` (for linear models)
   - Examples: RandomForestRegressor, XGBRegressor, LinearRegression, etc.

2. **`model_type`** (str): String name of the model class
   - Example: `"RandomForestRegressor"`, `"XGBRegressor"`, `"LinearRegression"`
   - Can be obtained with: `type(model).__name__`

3. **`feature_names`** (List[str]): List of feature names in order
   - Must match the order of features in your training data
   - Example: `["feature1", "feature2", "feature3", ...]`
   - Can be obtained from: `model.feature_names_in_` (if available) or from your DataFrame columns

4. **`preprocessor`** (Any, optional): Preprocessing pipeline/transformer
   - Can be `None` if no preprocessing needed
   - Must have `.transform()` method if provided
   - Examples: StandardScaler, Pipeline, ColumnTransformer, etc.

5. **`feature_selector`** (Any, optional): Feature selection transformer
   - Can be `None` if no feature selection used
   - Must have `.transform()` method if provided
   - Examples: SelectKBest, RFE, etc.

### Data Objects:
6. **`X_train`** (numpy.ndarray or pd.DataFrame): Training features
   - Used as background data for SHAP/LIME explainers
   - Should be representative of your training distribution
   - Shape: `(n_samples, n_features)`
   - Will be converted to numpy array if DataFrame

7. **`X_explain`** (numpy.ndarray or pd.DataFrame): Features to explain
   - Data points for which SHAP/LIME values will be computed
   - Typically a subset of recent/important data points
   - Shape: `(n_samples, n_features)`
   - Will be converted to numpy array if DataFrame

8. **`y_train`** (numpy.ndarray, optional): Training target values
   - Not directly used in SHAP/LIME analysis but may be needed for statistics
   - Can be `None`

9. **`y_explain`** (numpy.ndarray, optional): Target values for explanation data
   - Not directly used in SHAP/LIME analysis but may be needed for statistics
   - Can be `None`

10. **`df_data`** (pd.DataFrame): Full DataFrame with all data
    - Used for calculating data statistics (mean, std, min, max, trend)
    - Must include the target column
    - Should contain all features and target variable

11. **`target_col`** (str): Name of the target column in `df_data`
    - Example: `"revenue"`, `"target"`, `"y"`
    - Used for calculating target variable statistics

### Metadata:
12. **`model_metrics`** (Dict[str, float]): Model performance metrics
    - Required keys: `"rmse"`, `"r2"`, `"mape"`
    - Example: `{"rmse": 1500.0, "r2": 0.85, "mape": 5.2}`
    - These are included in the output JSON files

13. **`months_lookback`** (int): Number of months of data analyzed
    - Example: `3` for last 3 months
    - Used in metadata for context

## 3. Model Requirements

Your model must support:

### For SHAP Analysis:
- **`.predict()` method**: Must be able to make predictions
  ```python
  predictions = model.predict(X)
  ```

- **Feature importance** (for tree-based models):
  - `.feature_importances_` attribute (numpy array)
  - Examples: RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting

- **Coefficients** (for linear models):
  - `.coef_` attribute (numpy array)
  - Examples: LinearRegression, Ridge, Lasso, ElasticNet

### For LIME Analysis:
- **`.predict()` method**: Must be able to make predictions
  - LIME will call this method many times during explanation

### Model Types Supported:
- ✅ Tree-based: RandomForest, XGBoost, LightGBM, CatBoost, GradientBoosting, ExtraTrees
- ✅ Linear: LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
- ✅ Other: Any model with `.predict()` method (uses kernel explainer)

## 4. Data Requirements

### Training Data (`X_train`):
- **Purpose**: Background/reference data for SHAP explainer
- **Size**: Should be representative (typically 100-1000 samples)
- **Format**: numpy array or pandas DataFrame
- **Shape**: `(n_samples, n_features)`
- **Note**: If too large (>100 samples), will be automatically sampled

### Explanation Data (`X_explain`):
- **Purpose**: Data points to explain with SHAP/LIME
- **Size**: Typically 10-100 samples (more = slower)
- **Format**: numpy array or pandas DataFrame
- **Shape**: `(n_samples, n_features)`
- **Note**: LIME limits to 50 instances by default

### Full DataFrame (`df_data`):
- **Purpose**: Calculate statistics for metadata
- **Must include**:
  - All feature columns
  - Target column (specified by `target_col`)
- **Used for**:
  - Target variable statistics (mean, std, min, max, trend)
  - Data characteristics (total records, num features, missing values)

## 5. Configuration Options

You can customize the analysis behavior with `InterpretabilityConfig`:

```python
config = InterpretabilityConfig(
    shap_max_samples=100,      # Max samples for SHAP background (default: 100)
    shap_explainer_type="auto", # "auto", "tree", "kernel", "linear" (default: "auto")
    lime_num_features=10,       # Number of top features for LIME (default: 10)
    lime_num_samples=5000,      # Number of samples for LIME (default: 5000)
    top_n_features=10           # Top N features in summary (default: 10)
)
```

## 6. Example Implementation

Here's a template for implementing `load_model_and_data()`:

```python
def load_model_and_data():
    """Load model and data for SHAP/LIME analysis."""
    import joblib
    
    # Load model
    model = joblib.load("/dbfs/path/to/model.joblib")
    model_type = type(model).__name__
    
    # Load or prepare data
    df = spark.sql("SELECT * FROM your_table WHERE ...").toPandas()
    
    # Separate features and target
    target_col = "revenue"
    feature_cols = [col for col in df.columns if col != target_col]
    feature_names = feature_cols
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split into train and explain sets
    # Use 80% for training background, 20% for explanation
    n_train = int(len(df) * 0.8)
    X_train = X.iloc[:n_train]
    X_explain = X.iloc[n_train:n_train+50]  # Explain last 50 samples
    y_train = y.iloc[:n_train]
    y_explain = y.iloc[n_train:n_train+50]
    
    # Preprocessor and feature selector (if you have them)
    preprocessor = None  # Or load from model artifact
    feature_selector = None  # Or load from model artifact
    
    # Model metrics (calculate or load from evaluation)
    model_metrics = {
        "rmse": 1500.0,  # Calculate from your evaluation
        "r2": 0.85,      # Calculate from your evaluation
        "mape": 5.2      # Calculate from your evaluation
    }
    
    months_lookback = 3  # How many months of data you're analyzing
    
    return (
        model, model_type, feature_names, preprocessor, feature_selector,
        X_train, X_explain, y_train, y_explain,
        df, target_col, model_metrics, months_lookback
    )
```

## 7. Summary Checklist

Before running the script, ensure you have:

- [ ] Trained ML model with `.predict()` method
- [ ] Model type name (string)
- [ ] List of feature names (matching data order)
- [ ] Training data (`X_train`) - background for explainers
- [ ] Explanation data (`X_explain`) - data to explain
- [ ] Full DataFrame (`df_data`) with target column
- [ ] Target column name
- [ ] Model performance metrics (rmse, r2, mape)
- [ ] Number of months analyzed
- [ ] (Optional) Preprocessor pipeline
- [ ] (Optional) Feature selector
- [ ] Azure Blob Storage container names
- [ ] Azure Storage credentials (or dbutils configured)

## 8. Notes

- **Preprocessing**: If your model was trained with preprocessing, you must apply the same preprocessing to `X_train` and `X_explain` before passing them, OR provide the preprocessor and it will be applied automatically.

- **Feature Selection**: If your model was trained with feature selection, you must apply the same feature selection, OR provide the feature selector and it will be applied automatically.

- **Data Size**: For performance:
  - SHAP background: 100-1000 samples recommended
  - SHAP explanation: 10-100 samples recommended
  - LIME explanation: Limited to 50 instances by default

- **Model Compatibility**: The script automatically detects model type and selects the appropriate SHAP explainer (tree, linear, or kernel).
