# ============================================================
# src/__init__.py
# ============================================================
"""Milan mobile network traffic forecasting package.

Modules
-------
config          : paths, dtypes, split dates
data_loader     : file discovery, Parquet caching, series extraction
preprocessing   : windowing, normalisation, feature engineering
metrics         : MAE, RMSE, MAPE
evaluation      : walk-forward evaluation loops
models.sarima_model  : SARIMA forecaster + grid search
models.lstm_model    : LSTM forecaster
models.xgboost_model : XGBoost forecaster
"""
__version__ = "0.1.0"