# ============================================================
# src/models/lstm_model.py — LSTM forecaster
# ============================================================
import time, os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from ..config import SEQ_LEN, RANDOM_STATE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.random.set_seed(RANDOM_STATE)


class LSTMForecaster:
    """Single-layer or stacked LSTM for one-step-ahead prediction."""

    def __init__(self, seq_len=SEQ_LEN, hidden_units=64, n_layers=1,
                dropout=0.2, learning_rate=1e-3, batch_size=64,
                epochs=30, patience=5):
        self.seq_len = seq_len
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model = None
        self.train_time_ = None

    def _build(self, input_shape):
        inp = layers.Input(shape=input_shape)
        x = inp
        for i in range(self.n_layers):
            return_seq = (i < self.n_layers - 1)
            x = layers.LSTM(self.hidden_units,
                            return_sequences=return_seq)(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1)(x)
        m = models.Model(inp, out)
        m.compile(optimizer=optimizers.Adam(self.learning_rate),
                    loss="mse", metrics=["mae"])
        return m

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=0):
        self.model = self._build(X_train.shape[1:])
        cbs = [callbacks.EarlyStopping(patience=self.patience,
                                        restore_best_weights=True,
                                        monitor="val_loss" if X_val is not None else "loss")]
        t0 = time.time()
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=cbs, verbose=verbose,
        )
        self.train_time_ = time.time() - t0
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0).ravel()

    @staticmethod
    def grid_search(X_train, y_train, X_val, y_val, configs):
        """Run a manual grid over the given config dicts."""
        results = []
        for cfg in configs:
            m = LSTMForecaster(**cfg)
            m.fit(X_train, y_train, X_val, y_val)
            yp = m.predict(X_val)
            from ..metrics import mae, rmse
            results.append({
                **cfg,
                "val_MAE":  mae(y_val, yp),
                "val_RMSE": rmse(y_val, yp),
                "train_time_s": m.train_time_,
            })
        import pandas as pd
        return pd.DataFrame(results).sort_values("val_MAE").reset_index(drop=True)