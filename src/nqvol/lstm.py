"""LSTM benchmark.

Changes relative to the submitted code (see docs/CHANGES.md):
  * the input window for target day t covers days t-4 ... t, so the LSTM sees
    the same same-day predictors (Asia/London session, overnight gap, news,
    lagged ranges) as every other model; previously the window stopped at t-1;
  * sequences are built on the full chronological series before splitting,
    so no window spans the gap between two non-adjacent sample blocks and the
    test period loses no observations;
  * the early-stopping validation set is the last 20% of the training
    sequences in time order;
  * results are reported as the average forecast of an ensemble of models
    trained with different seeds (mean +/- sd across seeds is reported too),
    with TensorFlow op-level determinism enabled.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from . import config as C

tf.config.experimental.enable_op_determinism()


def _build(seq_len, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


class LSTMEnsemble:
    def __init__(self, features, seeds=C.LSTM_SEEDS_MAIN, seq_len=C.LSTM_SEQUENCE_LENGTH,
                 epochs=200, batch_size=32, patience=20, val_share=0.2):
        self.features = list(features)
        self.seeds = list(seeds)
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_share = val_share

    # -- sequence construction ------------------------------------------------
    def _windows(self, X_scaled, idx):
        L = self.seq_len
        return np.stack([X_scaled[i - L + 1:i + 1] for i in idx])

    def _target_index(self, mask):
        idx = np.flatnonzero(mask)
        return idx[idx >= self.seq_len - 1]

    # -- fitting --------------------------------------------------------------
    def fit(self, df, train_mask, target_col):
        X = df[self.features].to_numpy(float)
        y = df[target_col].to_numpy(float)
        self.scaler_x_ = StandardScaler().fit(X[train_mask])
        tr_idx = self._target_index(train_mask)
        self.scaler_y_ = StandardScaler().fit(y[tr_idx].reshape(-1, 1))
        Xs = self.scaler_x_.transform(X)
        X_tr = self._windows(Xs, tr_idx)
        y_tr = self.scaler_y_.transform(y[tr_idx].reshape(-1, 1)).ravel()
        n_fit = int(len(tr_idx) * (1 - self.val_share))
        self.train_index_ = tr_idx
        self.models_, self.best_epochs_ = [], []
        for seed in self.seeds:
            tf.keras.utils.set_random_seed(seed)
            m = _build(self.seq_len, len(self.features))
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                                  restore_best_weights=True)
            hist = m.fit(X_tr[:n_fit], y_tr[:n_fit],
                         validation_data=(X_tr[n_fit:], y_tr[n_fit:]),
                         epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=[es], verbose=0)
            self.models_.append(m)
            self.best_epochs_.append(int(np.argmin(hist.history["val_loss"])) + 1)
        return self

    # -- prediction -----------------------------------------------------------
    def predict_by_seed(self, df, mask):
        """Array (n_seeds, n_targets) of forecasts for the rows selected by mask."""
        Xs = self.scaler_x_.transform(df[self.features].to_numpy(float))
        idx = self._target_index(mask)
        W = self._windows(Xs, idx)
        out = []
        for m in self.models_:
            p = m(W, training=False).numpy().reshape(-1, 1)
            out.append(self.scaler_y_.inverse_transform(p).ravel())
        return np.vstack(out)

    def predict(self, df, mask):
        return self.predict_by_seed(df, mask).mean(axis=0)
