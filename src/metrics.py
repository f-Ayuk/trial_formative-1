# ============================================================
# src/metrics.py — Evaluation metrics
# ============================================================
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred):
    return {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }