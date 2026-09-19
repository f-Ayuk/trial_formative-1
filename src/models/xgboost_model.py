# ============================================================
# src/models/xgboost_model.py — XGBoost forecaster
# ============================================================
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from ..config import RANDOM_STATE


class XGBoostForecaster:
    """Gradient-boosted trees with engineered lag/rolling/time features."""

    def __init__(self, n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, reg_alpha=0.0, reg_lambda=1.0,
                random_state=RANDOM_STATE):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        )
        self.model = None
        self.feature_cols = None
        self.train_time_ = None

    def fit(self, df_train, target="y", val_df=None):
        self.feature_cols = [c for c in df_train.columns if c != target]
        X = df_train[self.feature_cols].values
        y = df_train[target].values

        self.model = xgb.XGBRegressor(**self.params)
        t0 = time.time()
        if val_df is not None:
            eval_set = [(val_df[self.feature_cols].values,
                        val_df[target].values)]
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y, verbose=False)
        self.train_time_ = time.time() - t0
        return self

    def predict(self, df):
        return self.model.predict(df[self.feature_cols].values)

    def feature_importance(self):
        return pd.Series(self.model.feature_importances_,
                        index=self.feature_cols).sort_values(ascending=False)