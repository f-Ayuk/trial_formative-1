# ============================================================
# src/models/sarima_model.py — SARIMA wrapper with grid search
# ============================================================
import time, warnings
import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..metrics import mae

warnings.filterwarnings("ignore")


class SarimaForecaster:
    """One-step-ahead SARIMA forecaster with walk-forward evaluation."""

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 1, 144),
                trend="n"):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model_fit = None
        self.train_time_ = None

    def fit(self, y_train):
        t0 = time.time()
        self.model_fit = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        self.train_time_ = time.time() - t0
        return self

    def predict_one_step(self, history, steps=1):
        """Refit on the full history each step (walk-forward)."""
        try:
            res = SARIMAX(
                history,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            return float(res.forecast(steps=steps).iloc[-1])
        except Exception:
            # fallback: naive persistence
            return float(history.iloc[-1])

    @staticmethod
    def grid_search(y_train, y_val,
                    p_range=(0, 1, 2), d_range=(0, 1),
                    q_range=(0, 1, 2),
                    P_range=(0, 1), D_range=(0, 1), Q_range=(0, 1),
                    s=144, max_evals=20):
        """Light grid search using validation MAE."""
        results = []
        combos = list(product(p_range, d_range, q_range,
                                P_range, D_range, Q_range))
        # Prioritise non-seasonal-only and small orders first
        combos = sorted(combos, key=lambda c: (sum(c), c))
        combos = combos[:max_evals]

        for c in combos:
            p, d, q, P, D, Q = c
            try:
                res = SARIMAX(
                    y_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                val_pred = res.forecast(steps=len(y_val))
                score = mae(y_val.values, np.asarray(val_pred))
                results.append({"order": (p, d, q),
                                "seasonal": (P, D, Q, s),
                                "val_MAE": score})
            except Exception:
                continue
        return pd.DataFrame(results).sort_values("val_MAE").reset_index(drop=True)