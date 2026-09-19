# ============================================================
# src/evaluation.py — Walk-forward evaluation for all models
# ============================================================
import time
import numpy as np
import pandas as pd
from .metrics import evaluate


def walk_forward_sarima(sarima, history_series, test_series):
    """One-step-ahead walk-forward using SARIMA refit each step."""
    preds = []
    history = history_series.copy()
    t0 = time.time()
    for t in range(len(test_series)):
        p = sarima.predict_one_step(history, steps=1)
        preds.append(p)
        history = pd.concat([history,
                            pd.Series([test_series.iloc[t]],
                                    index=[test_series.index[t]])])
    exec_time = time.time() - t0
    return np.array(preds), exec_time


def walk_forward_dl(model, history_values, test_values, seq_len):
    """One-step-ahead walk-forward for LSTM: append predictions to history."""
    preds = []
    buf = list(history_values[-seq_len:])
    t0 = time.time()
    for i in range(len(test_values)):
        x = np.asarray(buf[-seq_len:], dtype="float32")[np.newaxis, :, np.newaxis]
        p = float(model.predict(x)[0])
        preds.append(p)
        buf.append(test_values[i])   # true value feeds next input
    exec_time = time.time() - t0
    return np.array(preds), exec_time


def walk_forward_xgb(model, history_series, test_series,
                    feature_builder, target="y"):
    """One-step-ahead walk-forward for XGBoost."""
    preds = []
    history = history_series.copy()
    t0 = time.time()
    for t in range(len(test_series)):
        combined = pd.concat([history,
                                pd.Series([np.nan], index=[test_series.index[t]])])
        feats = feature_builder(combined).dropna()
        if len(feats) == 0:
            preds.append(float(history.iloc[-1]))
            continue
        last_row = feats.iloc[[-1]].drop(columns=[target], errors="ignore")
        p = float(model.model.predict(last_row.values)[0])
        preds.append(p)
        history = pd.concat([history,
                            pd.Series([test_series.iloc[t]],
                                        index=[test_series.index[t]])])
    exec_time = time.time() - t0
    return np.array(preds), exec_time