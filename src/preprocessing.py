# ============================================================
# src/preprocessing.py — Windowing and feature engineering
# ============================================================
import numpy as np
import pandas as pd
from .config import SEQ_LEN


def make_windows(series, seq_len=SEQ_LEN, horizon=1):
    """Return (X, y) for one-step-ahead forecasting.
    X: (N, seq_len, 1)  y: (N,)
    """
    values = series.values.astype("float32")
    X, y = [], []
    for i in range(len(values) - seq_len - horizon + 1):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len + horizon - 1])
    X = np.asarray(X, dtype="float32")[..., np.newaxis]
    y = np.asarray(y, dtype="float32")
    return X, y


def train_val_test_split(series, train_end, val_start, val_end,
                        test_start, test_end):
    """Chronological split (no shuffling)."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)

    train = s[s.index <= train_end]
    val   = s[(s.index >= val_start) & (s.index <= val_end)]
    test  = s[(s.index >= test_start) & (s.index <= test_end)]
    return train, val, test


def normalise(train, val, test):
    """Z-score normalisation computed on train only."""
    mu, sd = train.mean(), train.std()
    sd = sd if sd > 0 else 1.0
    return ((train - mu) / sd, (val - mu) / sd, (test - mu) / sd), (mu, sd)


def add_time_features(df):
    """Add hour-of-day, day-of-week, and cyclic encodings to a DataFrame."""
    df = df.copy()
    df["hour"]     = df.index.hour
    df["dow"]      = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def make_xgb_features(series):
    """Engineered features for XGBoost."""
    df = series.to_frame("y").copy()
    df = add_time_features(df)
    for lag in [1, 2, 3, 6, 12, 24, 144, 288]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in [6, 24, 144]:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["y"].shift(1).rolling(w).std()
    df = df.dropna()
    return df