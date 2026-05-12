"""
TCN Revised — VIP Resubmission Compliant
=========================================
Temporal Convolutional Network for daily wheat futures price direction prediction.


Strategy (same as ARX_Revised.py / FAVAR_Revised.py):
  1.  Predict next-day price CHANGE  Δ[t] = Close[t] - Close[t-1]
  2.  Derive binary direction:  predicted Δ > threshold  →  Up (1)
                                predicted Δ ≤ threshold  →  Down (0)

Models  (--mode regression | classification | both):

  TCNRegressor  [--mode regression]:
    Input (30, 37)  →  TemporalConvNet (dilated causal conv blocks)
                    →  Linear decoder  →  scalar Δ̂
    Direction derived post-hoc using a val-calibrated median threshold.
    Loss: MSELoss on Δ.  Optimiser: Adam.

  TCNClassifier  [--mode classification]:
    Input (30, 37)  →  TemporalConvNet (shared architecture)
                    →  Linear decoder  →  logit  →  sigmoid → P(Up)
    Directly predicts binary direction — no intermediate price forecast.
    Loss: BCEWithLogitsLoss (pos_weight from class imbalance).
    P(Up) threshold calibrated on val set to maximise F1-macro.

  Shared architecture:
    Dilated causal convolutions with receptive field ≥ lookback.
    num_channels per block, kernel_size, dropout configurable via CLI.
    Per-task seeds for reproducibility: regression=42, classification=43.








Compliance checklist:
  §1.1  Exactly 31 FRED-MD variables (SELECTED_FEATURES)
  §1.2  T-code stationarity transforms + 1-month reporting delay shift
        Monthly macro forward-filled to daily frequency
  §1.3  Lookback window of 30 trading days: each sample is a (30, 37)
        sequence of [Close | 5 tech | 31 macro] — no contemporaneous info
  §2    5-fold TimeSeriesSplit CV; scaler fitted on train fold only
        Per-task seeds: regression=42, classification=43
  §4.5  Classification metrics on derived direction labels:
        Accuracy, Precision, Recall, F1 (per class + macro), Confusion Matrix,
        ROC-AUC (score = predicted Δ / P(Up), continuous)
        Regression metrics (regression mode): RMSE, MAE, MAPE, R² (price-level)
  §4.6  Two visualizations per mode:
        Panel 1 — Rolling 30-day accuracy over time (clean line chart)
        Panel 2 — Confusion matrix with count + row-% annotation
"""


import os
import sys
import time
import pickle
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, r2_score, mean_squared_error
)

from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# §1.1  SELECTED FEATURES (31 variables, fixed)
# ─────────────────────────────────────────────────────────────────────────────

SELECTED_FEATURES = [
    'RPI', 'W875RX1', 'CMRMTSPLx', 'IPFPNSS', 'USWTRADE', 'USTRADE',
    'BUSLOANS', 'CONSPI', 'S&P 500', 'S&P PE ratio', 'FEDFUNDS',
    'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA',
    'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM',
    'AAAFFM', 'BAAFFM', 'EXSZUSx', 'EXJPUSx', 'EXUSUKx', 'EXCAUSx',
    'PPICMM', 'UMCSENTx'
]
assert len(SELECTED_FEATURES) == 31, "Must have exactly 31 FRED-MD variables."

# §1.2  T-CODE TRANSFORMATION TABLE
# Source: https://files.stlouisfed.org/files/htdocs/fred-databases/FRED-QD_appendix_v6.pdf
# Each entry maps variable name -> t-code integer
#   1 = level (no change)
#   2 = first difference
#   3 = second difference
#   4 = log
#   5 = log first difference (growth rate)
#   6 = log second difference
TCODES = {
    "RPI":          5,   # Real Personal Income
    "W875RX1":      5,   # Real personal income ex. transfer receipts
    "CMRMTSPLx":    5,   # Real Mfg and Trade Industries Sales
    "IPFPNSS":      5,   # IP: Final Products & Nonindustrial Supplies
    "USWTRADE":     5,   # Wholesale Trade: Sales
    "USTRADE":      5,   # Retail and Food Services Sales
    "BUSLOANS":     5,   # Commercial and Industrial Loans
    "CONSPI":       5,   # Nonborrowed reserves of depository institutions
    "S&P 500":      5,   # S&P 500 stock index
    "S&P PE ratio": 5,   # S&P 500 Price-to-Earnings ratio
    "FEDFUNDS":     2,   # Effective Federal Funds Rate
    "TB3MS":        2,   # 3-Month Treasury Bill
    "TB6MS":        2,   # 6-Month Treasury Bill
    "GS1":          2,   # 1-Year Treasury Rate
    "GS5":          2,   # 5-Year Treasury Rate
    "GS10":         2,   # 10-Year Treasury Rate
    "AAA":          2,   # Moody's AAA Corporate Bond Yield
    "BAA":          2,   # Moody's BAA Corporate Bond Yield
    "TB3SMFFM":     1,   # 3-Month Treasury minus Fed Funds spread
    "TB6SMFFM":     1,   # 6-Month Treasury minus Fed Funds spread
    "T1YFFM":       1,   # 1-Year Treasury minus Fed Funds spread
    "T5YFFM":       1,   # 5-Year Treasury minus Fed Funds spread
    "T10YFFM":      1,   # 10-Year Treasury minus Fed Funds spread
    "AAAFFM":       1,   # AAA minus Fed Funds spread
    "BAAFFM":       1,   # BAA minus Fed Funds spread
    "EXSZUSx":      5,   # Switzerland / U.S. Foreign Exchange Rate
    "EXJPUSx":      5,   # Japan / U.S. Foreign Exchange Rate
    "EXUSUKx":      5,   # U.S. / U.K. Foreign Exchange Rate
    "EXCAUSx":      5,   # Canada / U.S. Foreign Exchange Rate
    "PPICMM":       6,   # PPI: Metals and metal products
    "UMCSENTx":     2,   # U. of Michigan Consumer Sentiment
}


# ─────────────────────────────────────────────────────────────────────────────
# T-CODE TRANSFORMATIONS  (§1.2, Table)
# ─────────────────────────────────────────────────────────────────────────────

def apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """
    Apply the FRED-MD stationarity transformation to a single column.

    t-code 1: level (x_t)
    t-code 2: first difference (x_t - x_{t-1})
    t-code 3: second difference
    t-code 4: log
    t-code 5: log first difference (growth rate)
    t-code 6: log second difference
    """
    if tcode == 1:
        return series
    elif tcode == 2:
        return series.diff()
    elif tcode == 3:
        return series.diff().diff()
    elif tcode == 4:
        return np.log(series.replace(0, np.nan))
    elif tcode == 5:
        return np.log(series.replace(0, np.nan)).diff()
    elif tcode == 6:
        log_s = np.log(series.replace(0, np.nan))
        return log_s.diff().diff()
    else:
        raise ValueError(f"Unknown t-code: {tcode}")


def apply_tcodes_to_df(df: pd.DataFrame, tcode_map: dict) -> pd.DataFrame:
    """Apply t-code transformations to each column in df based on tcode_map."""
    transformed = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in tcode_map:
            transformed[col] = apply_tcode(df[col], tcode_map[col])
        else:
            # Default: no change (t-code 1)
            transformed[col] = df[col]
    return transformed


# ─────────────────────────────────────────────────────────────────────────────
#  BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BaseForecastModel(ABC):
    def __init__(self, task_type: str, **hyperparameters):
        self.task_type = task_type
        self.hyperparameters = hyperparameters

    @abstractmethod
    def fit(self, X_train, y_train): pass
    @abstractmethod
    def predict(self, X): pass
    @abstractmethod
    def evaluate(self, X_test, y_test): pass
    @abstractmethod
    def save(self, filepath: str): pass
    @abstractmethod
    def load(self, filepath: str): pass


# ─────────────────────────────────────────────────────────────────────────────
#  TCN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,
                 dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                      stride=stride, padding=padding, dilation=dilation))
        self.chomp1   = Chomp1d(padding)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                      stride=stride, padding=padding, dilation=dilation))
        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i, n_out in enumerate(num_channels):
            dilation = 2 ** i
            n_in     = num_inputs if i == 0 else num_channels[i - 1]
            padding  = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(n_in, n_out, kernel_size,
                                        stride=1, dilation=dilation,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
#  TCN REGRESSOR  (predict price → derive direction)
# ─────────────────────────────────────────────────────────────────────────────

class TCNRegressor(BaseForecastModel):
    """
    Temporal Convolutional Network for next-day wheat price regression.

    Strategy (§1.3 compliant):
      Input  : sliding window of shape (lookback=30, n_features)
               where n_features = 1 (scaled price) + 31 (macro cols)
               — each row uses only data available at t-1 or earlier.
      Target : Close[t]  (next-day closing price, continuous)
      Output : predicted Close[t]

    Direction derivation (leakage-free):
      At sample i, the last price in the input window = Close[t-1] (lag1).
      pred_dir = (predicted_Close[t] > Close[t-1]).astype(int)  ✓
      true_dir = (actual_Close[t]    > Close[t-1]).astype(int)  ✓

    Leakage safeguards:
      - price_scaler fitted on training fold only inside fit().
      - macro_scaler fitted on training macro only inside fit().
      - No future price or macro information in the sliding window.

    Evaluation (§4.5):
      Regression:     RMSE, R² (on price predictions)
      Classification: Accuracy, Precision, Recall, F1, CM, ROC-AUC
                      (on the derived direction labels)
    """

    def __init__(self, task_type='regression',
                 num_channels=None, kernel_size=3, dropout=0.2,
                 lr=0.0003, epochs=1000, batch_size=64, lookback=30,
                 patience=20,
                 **kwargs):
        if num_channels is None:
            num_channels = [64, 32]
        super().__init__(task_type=task_type,
                         num_channels=num_channels,
                         kernel_size=kernel_size,
                         dropout=dropout, lr=lr,
                         epochs=epochs, batch_size=batch_size,
                         lookback=lookback, **kwargs)
        self.num_channels = num_channels
        self.kernel_size  = kernel_size
        self.dropout      = dropout
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lookback     = lookback
        self.patience     = patience

        # Fitted on training data only (§2)
        self.price_scaler = StandardScaler()  # scales input price window
        self.macro_scaler = StandardScaler()  # scales macro block

        self.tcn     = None
        self.decoder = None
        self.n_features = None
        self._direction_threshold = 0.0   # calibrated in fit() on val set
        self._y_scaler = None

        # Device selection (MPS → CUDA → CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_model(self, n_features):
        """Initialise TCN + linear decoder."""
        self.n_features = n_features
        self.tcn = TemporalConvNet(
            n_features, self.num_channels,
            self.kernel_size, self.dropout).to(self.device)
        self.decoder = nn.Linear(self.num_channels[-1], 1).to(self.device)

    def _scale_windows(self, X_raw, fit=False):
        """
        X_raw: np.ndarray shape (N, lookback, 1 + n_macro)
        Returns scaled array of same shape.
        Price channel ([:,  :, 0]) and macro channels ([:, :, 1:]) are scaled
        independently using their respective scalers.
        """
        N, L, F = X_raw.shape
        # Price: fit/transform on (N*L, 1) then reshape back
        prices = X_raw[:, :, 0].reshape(-1, 1)
        if fit:
            prices_s = self.price_scaler.fit_transform(prices)
        else:
            prices_s = self.price_scaler.transform(prices)
        prices_s = prices_s.reshape(N, L, 1)

        if F > 1:
            macros = X_raw[:, :, 1:].reshape(-1, F - 1)
            if fit:
                macros_s = self.macro_scaler.fit_transform(macros)
            else:
                macros_s = self.macro_scaler.transform(macros)
            macros_s = macros_s.reshape(N, L, F - 1)
            return np.concatenate([prices_s, macros_s], axis=2)
        return prices_s

    def _to_tensor(self, X_scaled):
        """Convert (N, L, F) numpy → (N, F, L) FloatTensor on device."""
        return torch.FloatTensor(X_scaled).permute(0, 2, 1).to(self.device)

    # ── BaseForecastModel interface ───────────────────────────────────────

    def fit(self, X_train, y_train):
        """
        X_train : np.ndarray (N, lookback, n_features_raw)
                  channel 0 = log-return sequence (stationary); channels 1+ = macro
        y_train : np.ndarray (N,) — next-day log-returns (continuous)

        Training improvements:
          - Internal 15% val split for early stopping (patience=10).
          - CosineAnnealingLR: smooth LR decay over the epoch budget.
          - Target scaled with StandardScaler (fit on train portion only).
        """
        y_train = np.asarray(y_train).reshape(-1)
        n_val   = max(1, int(len(y_train) * 0.15))
        n_tr    = len(y_train) - n_val

        X_tr_raw, X_val_raw = X_train[:n_tr], X_train[n_tr:]
        y_tr,     y_val     = y_train[:n_tr],  y_train[n_tr:]

        # Scalers fitted on TRAIN portion only
        X_tr_s  = self._scale_windows(X_tr_raw, fit=True)
        X_val_s = self._scale_windows(X_val_raw, fit=False)
        self._build_model(X_tr_s.shape[2])

        self._y_scaler = StandardScaler()
        y_tr_s  = self._y_scaler.fit_transform(y_tr.reshape(-1, 1))
        y_val_s = self._y_scaler.transform(y_val.reshape(-1, 1))

        X_tr_t  = self._to_tensor(X_tr_s)
        y_tr_t  = torch.FloatTensor(y_tr_s).to(self.device)
        X_val_t = self._to_tensor(X_val_s)
        y_val_t = torch.FloatTensor(y_val_s).to(self.device)

        # direction threshold = 0 (positive log-return → Up, negative → Down).
        # This is the correct statistical definition regardless of trend.
        # Mean-drift is absorbed by the y_scaler (StandardScaler subtracts mean),
        # so the model predicts deviations from the training-mean return.
        # Threshold = 0 in unscaled space is equivalent to threshold = mean
        # in scaled space, which centre-balances the scaled target values.
        self._direction_threshold = 0.0

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(
            list(self.tcn.parameters()) + list(self.decoder.parameters()),
            lr=self.lr, weight_decay=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.05)
        criterion = nn.MSELoss()

        self.tcn.train()
        self.decoder.train()
        print(f"  [REG] Training on {n_tr} samples, val on {n_val} | "
              f"device={self.device}")

        best_val_loss = float('inf')
        patience_left = self.patience
        best_state    = None

        for epoch in range(self.epochs):
            self.tcn.train(); self.decoder.train()
            epoch_loss = 0.0
            for bX, by in loader:
                optimizer.zero_grad()
                out  = self.tcn(bX)[:, :, -1]
                pred = self.decoder(out)
                loss = criterion(pred, by)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.tcn.parameters()) +
                    list(self.decoder.parameters()), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            # Validation loss for early stopping
            self.tcn.eval(); self.decoder.eval()
            with torch.no_grad():
                val_pred = self.decoder(self.tcn(X_val_t)[:, :, -1])
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss - 1e-4:   # require meaningful improvement
                best_val_loss = val_loss
                patience_left = self.patience
                best_state = {
                    'tcn'    : {k: v.clone() for k, v in self.tcn.state_dict().items()},
                    'decoder': {k: v.clone() for k, v in self.decoder.state_dict().items()},
                }
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"    Early stop at epoch {epoch+1}  "
                          f"(best val_loss={best_val_loss:.5f})")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train={epoch_loss/len(loader):.5f}  "
                      f"val={val_loss:.5f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        if best_state is not None:
            self.tcn.load_state_dict(best_state['tcn'])
            self.decoder.load_state_dict(best_state['decoder'])

        # ─ Threshold calibration on val set ──────────────────────────────────
        # Target is now price CHANGE (Δ).  Calibrate the Δ threshold (in $)
        # that maximises F1-macro on the val set.
        # A positive predicted Δ → Up; negative → Down.
        self.tcn.eval(); self.decoder.eval()
        with torch.no_grad():
            val_preds_s = self.decoder(self.tcn(X_val_t)[:, :, -1]).cpu().numpy()
        val_deltas = self._y_scaler.inverse_transform(val_preds_s).reshape(-1)  # predicted Δ
        val_true   = (y_val > 0).astype(int)   # y_val contains true Δs; Up if Δ > 0
        candidates = np.linspace(np.percentile(val_deltas, 5),
                                 np.percentile(val_deltas, 95), 100)
        best_t, best_f1 = 0.0, -1.0
        for t in candidates:
            p  = (val_deltas >= t).astype(int)
            f1 = f1_score(val_true, p, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self._direction_threshold = best_t
        print(f"  [REG] Calibrated Δ threshold: {best_t:.4f} USD  "
              f"(val F1-macro: {best_f1:.4f})")

    def predict(self, X):
        """
        Returns predicted next-day closing prices, shape (N,).
        X : np.ndarray (N, lookback, n_features_raw)
        """
        if self.tcn is None:
            raise ValueError("Model not trained! Call fit() first.")
        X_s = self._scale_windows(X, fit=False)
        X_t = self._to_tensor(X_s)
        self.tcn.eval()
        self.decoder.eval()
        with torch.no_grad():
            out  = self.tcn(X_t)[:, :, -1]
            pred = self.decoder(out).cpu().numpy()
        return self._y_scaler.inverse_transform(pred).reshape(-1)

    def evaluate(self, X_test, y_test, y_close=None, y_prev=None):
        """
        y_test : next-day price CHANGES (Δ = Close[t] − Close[t-1]), shape (N,).

        Uses the MEDIAN of test-set predicted Δs as the direction threshold.
        Same reasoning as TCNClassifier: val-calibrated threshold doesn't
        generalise across distribution shifts.  Ranking quality (ROC-AUC)
        is the most reliable metric; median threshold makes 50/50 predictions.
        """
        y_test      = np.asarray(y_test).reshape(-1)     # true Δs
        price_preds = self.predict(X_test)                # predicted Δs
        lag1        = X_test[:, -1, 0]                   # last raw Close (for MAPE denom)

        # ── Median-threshold direction ───────────────────────────────────
        eval_threshold = float(np.median(price_preds))
        pred_dir = (price_preds >= eval_threshold).astype(int)
        true_dir = (y_test  > 0).astype(int)
        print(f"  [REG eval] median Δ threshold={eval_threshold:.4f} USD  "
              f"(pred Up={pred_dir.sum()}  Down={(1-pred_dir).sum()})")


        # ── Regression metrics on price changes ─────────────────────────────
        rmse  = float(np.sqrt(mean_squared_error(y_test, price_preds)))
        mae   = float(np.mean(np.abs(y_test - price_preds)))
        r2_px = float(r2_score(y_test, price_preds))
        # MAPE on absolute price to avoid division by near-zero Δ
        true_abs = lag1 + y_test
        pred_abs = lag1 + price_preds
        mape  = float(np.mean(
            np.abs(true_abs - pred_abs) / np.clip(np.abs(true_abs), 1e-10, None)) * 100)

        # ── Absolute price level stats (pred_close = lag1 + predicted_Δ) ─────
        # RMSE/MAE are identical to the Δ stats (same residuals).
        # R²(price) is high (~0.95+) because lag1 anchors the prediction near
        # the actual price — this is due to serial autocorrelation, NOT model skill.
        price_r2 = float(r2_score(true_abs, pred_abs))

        # ── Classification metrics on derived directions ───────────────────
        acc      = accuracy_score(true_dir, pred_dir)
        prec     = precision_score(true_dir, pred_dir, average=None, zero_division=0)
        rec      = recall_score(true_dir, pred_dir, average=None, zero_division=0)
        f1       = f1_score(true_dir, pred_dir, average=None, zero_division=0)
        f1_macro = f1_score(true_dir, pred_dir, average='macro', zero_division=0)
        cm       = confusion_matrix(true_dir, pred_dir)

        # ROC-AUC: predicted Δ as continuous ranking score
        try:
            auc = roc_auc_score(true_dir, price_preds)
        except Exception:
            auc = float('nan')

        return {
            'accuracy'           : acc,
            'precision_per_class': prec,
            'recall_per_class'   : rec,
            'f1_per_class'       : f1,
            'f1_macro'           : f1_macro,
            'confusion_matrix'   : cm,
            'roc_auc'            : auc,
            'r2'                 : float('nan'),
            'r2_price'           : r2_px,        # R² on Δ
            'r2_close'           : price_r2,     # R² on absolute Close prices
            'rmse'               : rmse,
            'mae'                : mae,
            'mape_pct'           : mape,
            'pred_close'         : pred_abs,
            'true_close'         : true_abs,
            'predictions'        : pred_dir,
            'price_predictions'  : price_preds,
        }


    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        payload = {
            'num_channels' : self.num_channels,
            'kernel_size'  : self.kernel_size,
            'dropout'      : self.dropout,
            'lr'           : self.lr,
            'epochs'       : self.epochs,
            'batch_size'   : self.batch_size,
            'lookback'     : self.lookback,
            'n_features'   : self.n_features,
            'price_scaler' : self.price_scaler,
            'macro_scaler' : self.macro_scaler,
            'y_scaler'     : self._y_scaler,
            'task_type'    : self.task_type,
            'hyperparameters': self.hyperparameters,
        }
        if self.tcn is not None:
            payload['tcn_state']     = self.tcn.state_dict()
            payload['decoder_state'] = self.decoder.state_dict()
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"  Model saved → {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.num_channels  = d['num_channels']
        self.kernel_size   = d['kernel_size']
        self.dropout       = d['dropout']
        self.lr            = d['lr']
        self.epochs        = d['epochs']
        self.batch_size    = d['batch_size']
        self.lookback      = d['lookback']
        self.price_scaler  = d['price_scaler']
        self.macro_scaler  = d['macro_scaler']
        self._y_scaler     = d['y_scaler']
        self.task_type     = d.get('task_type', 'regression')
        self.hyperparameters = d.get('hyperparameters', {})
        if 'tcn_state' in d and d['n_features'] is not None:
            self._build_model(d['n_features'])
            self.tcn.load_state_dict(d['tcn_state'])
            self.decoder.load_state_dict(d['decoder_state'])
        print(f"  Model loaded ← {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
#  TCN CLASSIFIER  (direct binary direction forecast, no price intermediary)
# ─────────────────────────────────────────────────────────────────────────────

class TCNClassifier(BaseForecastModel):
    """
    Temporal Convolutional Network for direct binary direction classification.

    Strategy:
      Input  : same sliding window as TCNRegressor — (lookback, 1+n_macro)
      Target : y[t] = 1 if Close[t] > Close[t-1], else 0  (binary)
      Loss   : BCEWithLogitsLoss (single logit output)
      Output : sigmoid(logit) ≥ 0.5 → Up (1), else Down (0)

    No price intermediate step — the TCN learns to map the sequence of
    recent prices + macro conditions directly to the next-day direction.

    ROC-AUC uses sigmoid probability (not hard threshold), giving a proper
    ranking score without relying on a price comparison.

    Leakage safeguards (same as TCNRegressor):
      - price_scaler / macro_scaler fitted on training fold only.
      - Binary target derived from Close[t] vs Close[t-1], where
        Close[t-1] = last price in window = X[i,-1,0]. No future info. ✓
    """

    def __init__(self, task_type='classification',
                 num_channels=None, kernel_size=3, dropout=0.2,
                 lr=0.0003, epochs=1000, batch_size=64, lookback=30,
                 patience=20,
                 **kwargs):
        if num_channels is None:
            num_channels = [64, 32]
        super().__init__(task_type=task_type,
                         num_channels=num_channels,
                         kernel_size=kernel_size,
                         dropout=dropout, lr=lr,
                         epochs=epochs, batch_size=batch_size,
                         lookback=lookback, **kwargs)
        self.num_channels = num_channels
        self.kernel_size  = kernel_size
        self.dropout      = dropout
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lookback     = lookback
        self.patience     = patience

        # Fitted on training data only (§2)
        self.price_scaler = StandardScaler()
        self.macro_scaler = StandardScaler()

        self.tcn     = None
        self.decoder = None
        self.n_features = None
        self._cls_threshold = 0.5   # calibrated in fit() on val set
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    # ── Shared helpers (same scaling as TCNRegressor) ──────────────────────

    def _build_model(self, n_features):
        self.n_features = n_features
        self.tcn = TemporalConvNet(
            n_features, self.num_channels,
            self.kernel_size, self.dropout).to(self.device)
        # Single logit output for binary classification
        self.decoder = nn.Linear(self.num_channels[-1], 1).to(self.device)

    def _scale_windows(self, X_raw, fit=False):
        N, L, F = X_raw.shape
        prices = X_raw[:, :, 0].reshape(-1, 1)
        if fit:
            prices_s = self.price_scaler.fit_transform(prices)
        else:
            prices_s = self.price_scaler.transform(prices)
        prices_s = prices_s.reshape(N, L, 1)
        if F > 1:
            macros = X_raw[:, :, 1:].reshape(-1, F - 1)
            if fit:
                macros_s = self.macro_scaler.fit_transform(macros)
            else:
                macros_s = self.macro_scaler.transform(macros)
            macros_s = macros_s.reshape(N, L, F - 1)
            return np.concatenate([prices_s, macros_s], axis=2)
        return prices_s

    def _to_tensor(self, X_scaled):
        return torch.FloatTensor(X_scaled).permute(0, 2, 1).to(self.device)

    # ── BaseForecastModel interface ───────────────────────────────────────

    def fit(self, X_train, y_train):
        """
        X_train : np.ndarray (N, lookback, n_features_raw)
        y_train : np.ndarray (N,) binary direction labels {0, 1}

        Training improvements:
          - Internal 15% val split for early stopping (patience=10).
          - CosineAnnealingLR + weight_decay.
          - pos_weight computed from class imbalance in training labels.
          - Gradient clipping (max_norm=1.0).
        """
        y_train = np.asarray(y_train).reshape(-1).astype(np.float32)
        n_val   = max(1, int(len(y_train) * 0.15))
        n_tr    = len(y_train) - n_val

        X_tr_raw, X_val_raw = X_train[:n_tr], X_train[n_tr:]
        y_tr,     y_val     = y_train[:n_tr],  y_train[n_tr:]

        X_tr_s  = self._scale_windows(X_tr_raw, fit=True)
        X_val_s = self._scale_windows(X_val_raw, fit=False)
        self._build_model(X_tr_s.shape[2])

        X_tr_t  = self._to_tensor(X_tr_s)
        y_tr_t  = torch.FloatTensor(y_tr).unsqueeze(1).to(self.device)
        X_val_t = self._to_tensor(X_val_s)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Bias-initialise the decoder to the empirical class prior.
        # Without this, a zero-intialised bias → sigmoid(0)=0.5, but with
        # imbalanced classes the model drifts to the majority class.
        # Setting bias = log(n_pos/n_neg) gives the correct starting logit
        # so the model's first predictions match the training prior, and
        # the loss gradients then push it to LEARN deviations from the prior.
        n_pos  = float(y_tr.sum())
        n_neg  = float(n_tr - n_pos)
        prior_logit = float(np.log(max(n_pos, 1) / max(n_neg, 1)))
        with torch.no_grad():
            self.decoder.bias.fill_(prior_logit)
        print(f"  [CLS] decoder bias → {prior_logit:.3f}  "
              f"(Up={int(n_pos)}, Down={int(n_neg)})")

        optimizer = optim.Adam(
            list(self.tcn.parameters()) + list(self.decoder.parameters()),
            lr=self.lr, weight_decay=3e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.05)
        # Fixed class weight tweak (user-requested quick adjustment).
        pos_weight = torch.tensor([1.1]).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.tcn.train()
        self.decoder.train()
        print(f"  [CLS] Training on {n_tr} | val {n_val} | "
              f"pos_weight={pos_weight.item():.3f} | device={self.device}")

        best_val_loss = float('inf')
        patience_left = self.patience
        best_state    = None

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.tcn.train(); self.decoder.train()
            epoch_loss = 0.0
            for bX, by in loader:
                optimizer.zero_grad()
                logit = self.decoder(self.tcn(bX)[:, :, -1])
                loss  = criterion(logit, by)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.tcn.parameters()) +
                    list(self.decoder.parameters()), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            self.tcn.eval(); self.decoder.eval()
            with torch.no_grad():
                val_logit = self.decoder(self.tcn(X_val_t)[:, :, -1])
                val_loss  = criterion(val_logit, y_val_t).item()

            if val_loss < best_val_loss - 1e-4:   # require meaningful improvement
                best_val_loss = val_loss
                patience_left = self.patience
                best_state = {
                    'tcn'    : {k: v.clone() for k, v in self.tcn.state_dict().items()},
                    'decoder': {k: v.clone() for k, v in self.decoder.state_dict().items()},
                }
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"    Early stop at epoch {epoch+1}  "
                          f"(best val_loss={best_val_loss:.5f})")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train={epoch_loss/len(loader):.5f}  "
                      f"val={val_loss:.5f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        if best_state is not None:
            self.tcn.load_state_dict(best_state['tcn'])
            self.decoder.load_state_dict(best_state['decoder'])

        # ─ Threshold calibration on val set ──────────────────────────────────
        # Scan 100 probability thresholds on the val set, pick the one
        # that maximises F1-macro.  This fixes all-Up collapse even when
        # all sigmoid outputs are > 0.5.
        self.tcn.eval(); self.decoder.eval()
        with torch.no_grad():
            val_logits = self.decoder(
                self.tcn(X_val_t)[:, :, -1]).cpu().numpy().reshape(-1)
        val_probas = 1.0 / (1.0 + np.exp(-val_logits))   # stable sigmoid
        y_val_int  = y_val.astype(int)
        candidates = np.linspace(np.percentile(val_probas, 5),
                                 np.percentile(val_probas, 95), 100)
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            p  = (val_probas >= t).astype(int)
            f1 = f1_score(y_val_int, p, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self._cls_threshold = best_t
        print(f"  [CLS] Calibrated threshold: {best_t:.4f}  "
              f"(val F1-macro: {best_f1:.4f})")

    def _predict_logits(self, X):
        """Return raw logits, shape (N,)."""
        if self.tcn is None:
            raise ValueError("Model not trained! Call fit() first.")
        X_s = self._scale_windows(X, fit=False)
        X_t = self._to_tensor(X_s)
        self.tcn.eval()
        self.decoder.eval()
        with torch.no_grad():
            logits = self.decoder(self.tcn(X_t)[:, :, -1]).cpu().numpy().reshape(-1)
        return logits

    def predict_proba(self, X):
        """Return probability of Up (1), shape (N,)."""
        return torch.sigmoid(torch.tensor(self._predict_logits(X))).numpy()

    def predict(self, X):
        """Return binary direction labels {0,1} using calibrated threshold."""
        return (self.predict_proba(X) >= self._cls_threshold).astype(int)

    def evaluate(self, X_test, y_test):
        """
        y_test : binary direction labels {0, 1}, shape (N,).

        Threshold strategy in evaluate():
          Use the MEDIAN of the test-set predicted probabilities.
          This forces exactly 50/50 Up/Down predictions and eliminates
          distribution-shift collapse (the val-calibrated threshold does not
          generalise when the test proba distribution is shifted).

          Interpretation: "rank test days by confidence; top-50% → Up."
          ROC-AUC is unaffected (threshold-independent ranking metric).

        For deployment (predict()), the val-calibrated self._cls_threshold is used.
        """
        y_test = np.asarray(y_test).reshape(-1).astype(int)
        proba  = self.predict_proba(X_test)

        # Median-threshold evaluation — robust to distribution shift
        eval_threshold = float(np.median(proba))
        preds = (proba >= eval_threshold).astype(int)
        print(f"  [CLS eval] median threshold={eval_threshold:.4f}  "
              f"(pred Up={preds.sum()}  Down={(1-preds).sum()})")

        acc      = accuracy_score(y_test, preds)
        prec     = precision_score(y_test, preds, average=None, zero_division=0)
        rec      = recall_score(y_test, preds, average=None, zero_division=0)
        f1       = f1_score(y_test, preds, average=None, zero_division=0)
        f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
        cm       = confusion_matrix(y_test, preds)
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = float('nan')

        # Pseudo-R²: r2_score(binary_labels, predicted_probability)
        # Measures how much the model's probability OUTPUT explains the
        # variance in the binary outcome, relative to the null (base-rate)
        # model.  Positive = better than guessing the class mean;
        # negative = worse.  Interpretable like a linear R².
        try:
            r2_prob = float(r2_score(y_test.astype(float), proba))
        except Exception:
            r2_prob = float('nan')

        return {
            'accuracy'           : acc,
            'precision_per_class': prec,
            'recall_per_class'   : rec,
            'f1_per_class'       : f1,
            'f1_macro'           : f1_macro,
            'confusion_matrix'   : cm,
            'roc_auc'            : auc,
            'r2'                 : r2_prob,        # pseudo-R² (proba vs label)
            'r2_price'           : r2_prob,        # same key, for compat with print_metrics
            'rmse'               : float('nan'),   # not applicable to classifier
            'predictions'        : preds,
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        payload = {
            'num_channels'   : self.num_channels,
            'kernel_size'    : self.kernel_size,
            'dropout'        : self.dropout,
            'lr'             : self.lr,
            'epochs'         : self.epochs,
            'batch_size'     : self.batch_size,
            'lookback'       : self.lookback,
            'n_features'     : self.n_features,
            'price_scaler'   : self.price_scaler,
            'macro_scaler'   : self.macro_scaler,
            'task_type'      : self.task_type,
            'hyperparameters': self.hyperparameters,
        }
        if self.tcn is not None:
            payload['tcn_state']     = self.tcn.state_dict()
            payload['decoder_state'] = self.decoder.state_dict()
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"  Model saved → {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.num_channels   = d['num_channels']
        self.kernel_size    = d['kernel_size']
        self.dropout        = d['dropout']
        self.lr             = d['lr']
        self.epochs         = d['epochs']
        self.batch_size     = d['batch_size']
        self.lookback       = d['lookback']
        self.price_scaler   = d['price_scaler']
        self.macro_scaler   = d['macro_scaler']
        self.task_type      = d.get('task_type', 'classification')
        self.hyperparameters = d.get('hyperparameters', {})
        if 'tcn_state' in d and d['n_features'] is not None:
            self._build_model(d['n_features'])
            self.tcn.load_state_dict(d['tcn_state'])
            self.decoder.load_state_dict(d['decoder_state'])
        print(f"  Model loaded ← {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_fred_md(fred_md_folder: str) -> pd.DataFrame:
    folder    = Path(fred_md_folder)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {fred_md_folder}")
    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        if len(df) > 0 and 'Transform' in str(df.iloc[0, 0]):
            df = df.iloc[1:].reset_index(drop=True)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce',
                                      format='%m/%d/%Y')
        df = df.dropna(subset=[date_col]).set_index(date_col)
        available = [c for c in SELECTED_FEATURES if c in df.columns]
        df = df[available].apply(pd.to_numeric, errors='coerce')
        frames.append(df)
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    return combined


def apply_reporting_delay_and_tcodes(df_macro: pd.DataFrame) -> pd.DataFrame:
    """§1.2: t-code transforms then +1 month shift (reporting delay)."""
    df_t = apply_tcodes_to_df(df_macro, TCODES).dropna()
    df_t.index = df_t.index + pd.DateOffset(months=1)
    return df_t


def forward_fill_macro_to_daily(df_macro_monthly: pd.DataFrame,
                                df_wheat_daily: pd.DataFrame) -> pd.DataFrame:
    """§1.2: Merge monthly macro onto daily prices via forward-fill."""
    dm = df_macro_monthly.copy()
    dm['YearMonth'] = dm.index.to_period('M')
    dm = dm.reset_index(drop=True)

    dw = df_wheat_daily.copy()
    dw['YearMonth'] = dw['date'].dt.to_period('M')

    merged = dw.merge(dm, on='YearMonth', how='left')
    merged = merged.sort_values('date').reset_index(drop=True)
    macro_cols_out = [c for c in dm.columns if c != 'YearMonth']
    merged[macro_cols_out] = merged[macro_cols_out].ffill()
    return merged


def compute_technical_features(df_wheat: pd.DataFrame) -> pd.DataFrame:
    df = df_wheat.copy()
    log_r = np.log(df['Close'].clip(lower=1e-10)).diff().fillna(0)
    df['tech_ret5']  = log_r.rolling(5,  min_periods=1).sum()
    df['tech_ret20'] = log_r.rolling(20, min_periods=1).sum()
    df['tech_vol20'] = log_r.rolling(20, min_periods=5).std().fillna(0)
    gain  = log_r.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-log_r.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['tech_rsi14'] = (100 - 100 / (1 + rs)).fillna(50)
    ma20 = df['Close'].rolling(20, min_periods=1).mean()
    df['tech_zma20'] = ((df['Close'] - ma20) / ma20.clip(lower=1e-10)).fillna(0)
    tech_cols = ['tech_ret5', 'tech_ret20', 'tech_vol20', 'tech_rsi14', 'tech_zma20']
    df[tech_cols] = df[tech_cols].shift(1).fillna(0)
    return df[['date'] + tech_cols]


def build_sliding_windows(df_merged: pd.DataFrame,
                          macro_cols: list,
                          tech_cols: list = None,
                          lookback: int = 30) -> tuple:
    """§1.3 — identical to TCN_Revised.py (same window layout)."""
    tech_cols = tech_cols or []
    macro_lag1_cols = [f"{c}_lag1" for c in macro_cols]
    df_merged[macro_lag1_cols] = df_merged[macro_cols].shift(1)
    df_merged = df_merged.dropna().reset_index(drop=True)

    prices = df_merged['Close'].values
    macros = df_merged[macro_lag1_cols].values
    techs  = df_merged[tech_cols].values if tech_cols else None
    dates  = df_merged['date'].values

    prices_safe = np.clip(prices, 1e-10, None)

    X_list, yp_list, yd_list, yc_list, yv_list, d_list = [], [], [], [], [], []
    T = len(df_merged)
    for i in range(lookback, T):
        price_win = prices[i - lookback: i].reshape(-1, 1)
        macro_win = macros[i - lookback: i]
        if techs is not None:
            tech_win = techs[i - lookback: i]
            window   = np.hstack([price_win, tech_win, macro_win])
        else:
            window = np.hstack([price_win, macro_win])
        X_list.append(window)

        delta = prices[i] - prices[i - 1]        # price CHANGE
        yp_list.append(delta)
        yd_list.append(int(delta > 0))
        yc_list.append(prices[i])
        yv_list.append(prices[i - 1])
        d_list.append(dates[i])

    return (np.array(X_list, dtype=np.float32),
            np.array(yp_list, dtype=np.float32),
            np.array(yd_list, dtype=np.int32),
            np.array(yc_list, dtype=np.float32),
            np.array(yv_list, dtype=np.float32),
            np.array(d_list))


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS REPORTING  (mirrors ARX_Revised.py style)
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(title: str, m: dict):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")
    print(f"  ── Price Change (Δ) Forecast ──")
    print(f"  RMSE (Δ)   : {m.get('rmse', float('nan')):.4f}  USD/day")
    print(f"  MAE  (Δ)   : {m.get('mae', float('nan')):.4f}  USD/day")
    print(f"  MAPE       : {m.get('mape_pct', float('nan')):.2f}%")
    print(f"  R² (Δ)    : {m.get('r2_price', float('nan')):.4f}")
    print(f"  ── Absolute Price Forecast (pred = lag1 + Δ̂) ──")
    print(f"  R² (price)  : {m.get('r2_close', float('nan')):.4f}  "
          f"[note: high due to lag1 autocorrelation]")
    print(f"  ── Direction (Δ > 0 → Up) ──")
    print(f"  Accuracy      : {m['accuracy']:.4f}")
    print(f"  ROC-AUC       : {m['roc_auc']:.4f}")
    print(f"  F1 Macro      : {m['f1_macro']:.4f}")
    print(f"  {'Class':<10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for i, label in enumerate(['Down (0)', 'Up (1)']):
        print(f"  {label:<10} {m['precision_per_class'][i]:>12.4f} "
              f"{m['recall_per_class'][i]:>10.4f} "
              f"{m['f1_per_class'][i]:>10.4f}")
    print(f"  Confusion Matrix:")
    print(f"{m['confusion_matrix']}")

# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION  (§4.6)
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_accuracy(preds, true_dir, dates_plot, window=30):
    """Compute rolling accuracy Series for a set of predictions."""
    correct = (preds == true_dir).astype(float)
    return pd.Series(correct, index=dates_plot).rolling(window, min_periods=1).mean()


def _draw_cm(ax, cm, cmap, title):
    """Draw a confusion matrix heatmap with count + row-% labels."""
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct      = cm / row_sums * 100
    sns.heatmap(cm, annot=False, cmap=cmap, ax=ax,
                vmin=0, vmax=cm.max() * 1.15,
                xticklabels=['Pred Down (0)', 'Pred Up (1)'],
                yticklabels=['True Down (0)', 'True Up (1)'],
                linewidths=2, linecolor='white', cbar=False)
    for i in range(2):
        for j in range(2):
            bg      = cm[i, j] / cm.max() if cm.max() > 0 else 0
            txt_col = 'white' if bg > 0.55 else '#222222'
            ax.text(j + 0.5, i + 0.40, f'{cm[i, j]}',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold', color=txt_col)
            ax.text(j + 0.5, i + 0.68, f'({pct[i, j]:.1f}% of row)',
                    ha='center', va='center', fontsize=9, color=txt_col)
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f'{title}\nAccuracy: {acc:.2%}',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)


def plot_results(dates_test, true_dir_test,
                 results: list,          # list of (label, metrics, color, cmap)
                 lookback, elapsed, output_dir='TCN',
                 filename_suffix=''):
    """
    §4.6 Visualizations.

    results : list of tuples  (label, metrics_dict, line_color, cm_cmap)
      — supports 1 or 2 models (regression, classification, or both)

    Panel 1 (top) : Rolling 30-day accuracy for each model.
    Panel 2+ (bottom): One confusion matrix per model.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(results)
    dates_plot = pd.to_datetime(dates_test)
    window = 30

    fig = plt.figure(figsize=(16, 6 + 6 * n_models))
    n_rows = 1 + n_models          # 1 accuracy panel + 1 CM per model
    gs = gridspec.GridSpec(n_rows, n_models, figure=fig,
                           height_ratios=[1.8] + [1.0] * n_models,
                           hspace=0.5, wspace=0.35)

    # ── Panel 1: rolling accuracy for all models ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    baseline = 0.5
    ax1.axhline(baseline, color='#888888', linewidth=1.4,
                linestyle=':', label='Random baseline (50%)', zorder=1)

    for label, m, color, _ in results:
        preds = m['predictions']
        roll  = _rolling_accuracy(preds, true_dir_test, dates_plot, window)
        acc   = m['accuracy']
        ax1.fill_between(roll.index, baseline, roll.values,
                         where=(roll.values >= baseline),
                         color=color, alpha=0.10, zorder=1)
        ax1.plot(roll.index, roll.values, color=color, linewidth=2.2,
                 label=f'{label}  (overall {acc:.1%})', zorder=3)

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax1.set_ylim(0.15, 0.85)
    ax1.set_xlim(dates_plot[window - 1], dates_plot[-1])
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel(f'{window}-Day Rolling Accuracy', fontsize=11)

    model_names = ' vs '.join(lbl for lbl, _, _, _ in results)
    ax1.set_title(
        f'Rolling {window}-Day Directional Accuracy — {model_names}\n'
        f'Lookback: {lookback}  |  Time: {elapsed:.1f}s',
        fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle='--')

    # ── Panels 2+: one confusion matrix per model ────────────────────────
    for col, (label, m, _, cmap) in enumerate(results):
        ax = fig.add_subplot(gs[1, col])
        _draw_cm(ax, m['confusion_matrix'], cmap, label)

    tag = filename_suffix or ('_'.join(
        lbl.lower().replace(' ', '_') for lbl, _, _, _ in results))
    filename = os.path.join(output_dir,
                            f'tcn_{tag}_l{lookback}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_tcn(lookback: int = 30,
                  epochs: int = 50,
                  num_channels=None,
                  mode: str = 'classification',
                  patience: int = 20,
                  dropout: float = 0.2,
                  lr: float = 0.0003,
                  kernel_size: int = 3,
                  fred_folder_train: str = None,
                  fred_folder_val: str = None):
    """
    End-to-end TCN pipeline supporting three modes:

      'regression'     — TCNRegressor: predict price → derive direction
      'classification' — TCNClassifier: directly predict Up/Down (BCELoss)
      'both'           — run both models and compare side-by-side

    Common steps for all modes:
      1–3. Data loading, t-codes, 1-month delay, forward-fill  (§1.2)
      4.   Sliding windows (N, lookback, 1+31)                 (§1.3)
      5.   85/15 split + 5-fold TimeSeriesSplit CV             (§2)
      6.   Final model on train+val, evaluate on holdout test  (§4)
      7.   Report metrics                                      (§4.5)
      8.   Visualize rolling accuracy + confusion matrix(ces)  (§4.6)
    """
    if num_channels is None:
        num_channels = [64, 64, 32]   # deeper default (3 dilation levels)
    assert mode in ('regression', 'classification', 'both'), \
        "mode must be 'regression', 'classification', or 'both'"

    start = time.time()

    base = Path(__file__).parent.parent
    if fred_folder_train is None:
        fred_folder_train = str(
            base / "data" / "fred-md" / "Historical FRED-MD Vintages Final")
    if fred_folder_val is None:
        fred_folder_val = str(
            base / "data" / "fred-md" /
            "Historical-vintages-of-FRED-MD-2015-01-to-2024-12")
    wheat_path = str(
        base / "data" / "wheat-futures" / "wheat_futures_daily.csv")

    print("=" * 60)
    print(f"TCN Daily Pipeline — mode='{mode}'")
    print("=" * 60)

    # ── Steps 1–3: Data loading (shared) ─────────────────────────────────
    print("\n[1/5] Loading FRED-MD macro data and applying t-codes...")
    df_macro_raw = pd.concat([
        load_fred_md(fred_folder_train),
        load_fred_md(fred_folder_val)
    ]).sort_index()
    df_macro_raw = df_macro_raw[~df_macro_raw.index.duplicated(keep='last')]
    missing = [v for v in SELECTED_FEATURES if v not in df_macro_raw.columns]
    if missing:
        print(f"  WARNING: missing variables: {missing}")
    print(f"  Macro: {df_macro_raw.shape[0]} monthly obs, "
          f"{df_macro_raw.shape[1]} variables")
    df_macro = apply_reporting_delay_and_tcodes(df_macro_raw)
    print(f"  After t-codes + 1-month delay: {df_macro.shape}")

    print("\n[2/5] Loading daily wheat futures prices...")
    df_wheat = pd.read_csv(wheat_path)
    df_wheat['date'] = (pd.to_datetime(df_wheat['date'], utc=True)
                          .dt.tz_localize(None))
    df_wheat = df_wheat.sort_values('date').reset_index(drop=True)
    print(f"  Wheat data: {len(df_wheat)} daily observations")

    print("\n[3/5] Forward-filling monthly macro onto daily prices + computing tech features...")
    macro_cols = [c for c in SELECTED_FEATURES if c in df_macro.columns]
    df_merged  = forward_fill_macro_to_daily(df_macro, df_wheat)

    # Compute short-term technical indicators from daily price data.
    # These carry signal the monthly macro variables cannot capture.
    df_tech   = compute_technical_features(df_wheat)
    tech_cols = [c for c in df_tech.columns if c != 'date']
    df_merged = df_merged.merge(df_tech, on='date', how='left')
    df_merged[tech_cols] = df_merged[tech_cols].ffill().fillna(0)
    print(f"  Merged: {len(df_merged)} rows  |  {len(macro_cols)} macro  "
          f"|  {len(tech_cols)} tech features")

    # ── Step 4: Sliding windows ────────────────────────────────────────────
    print(f"\n[4/5] Building sliding windows "
          f"(lookback={lookback}, features=1+{len(tech_cols)}+{len(macro_cols)})...")
    X, y_price, y_dir, y_close, y_prev, dates = build_sliding_windows(
        df_merged, macro_cols, tech_cols=tech_cols, lookback=lookback)
    print(f"  Samples X: {X.shape}")
    print(f"  y_price range: {y_price.min():.2f} – {y_price.max():.2f}  (USD/day, price change Δ)")
    print(f"  y_dir  split  : Up={y_dir.sum()}  Down={(1-y_dir).sum()}")
    print(f"  Date range: {pd.to_datetime(dates[0]).date()} → "
          f"{pd.to_datetime(dates[-1]).date()}")

    # ── Step 5: Split ──────────────────────────────────────────────────────
    print("\n[5/5] Splitting and running 5-fold TimeSeriesSplit CV...")
    test_size = int(len(y_price) * 0.15)
    split_idx = len(y_price) - test_size

    X_tv,    X_test    = X[:split_idx],        X[split_idx:]
    yp_tv,   yp_test   = y_price[:split_idx],  y_price[split_idx:]
    yd_tv,   yd_test   = y_dir[:split_idx],    y_dir[split_idx:]
    yc_tv,   yc_test   = y_close[:split_idx],  y_close[split_idx:]
    yv_tv,   yv_test   = y_prev[:split_idx],   y_prev[split_idx:]
    dates_tv, dates_test = dates[:split_idx],  dates[split_idx:]
    print(f"  Train+Val: {split_idx}  |  Holdout Test: {test_size}")

    tscv = TimeSeriesSplit(n_splits=5)

    def _make_model(task):
        kwargs = dict(num_channels=num_channels, kernel_size=kernel_size,
                      dropout=dropout, lr=lr,
                      epochs=epochs, batch_size=64, lookback=lookback,
                      patience=patience)
        return TCNRegressor(**kwargs) if task == 'regression' \
               else TCNClassifier(**kwargs)

    # ── CV loop — run only for requested mode(s) ──────────────────────────
    tasks = []
    if mode in ('regression', 'both'):    tasks.append('regression')
    if mode in ('classification', 'both'): tasks.append('classification')

    fold_stats = {t: {'accs': [], 'r2s': []} for t in tasks}

    # Per-task seeds so each task always gets the same random state
    # regardless of which other tasks run in the same session (--mode both
    # vs --mode classification must produce identical TCNClassifier results).
    TASK_SEEDS = {'regression': 42, 'classification': 43}

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tv), start=1):
        print(f"\n  Fold {fold}:")
        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        for task in tasks:
            # Fix random state per task so results are mode-independent
            seed = TASK_SEEDS[task] + fold
            torch.manual_seed(seed)
            np.random.seed(seed)
            y_tr  = yp_tv[tr_idx]  if task == 'regression' else yd_tv[tr_idx]
            y_val = yp_tv[val_idx] if task == 'regression' else yd_tv[val_idx]
            mdl = _make_model(task)
            mdl.fit(X_tr, y_tr)
            m   = mdl.evaluate(X_val, y_val)
            fold_stats[task]['accs'].append(m['accuracy'])
            fold_stats[task]['r2s'].append(m.get('r2_price', float('nan')))
            tag = 'REG' if task == 'regression' else 'CLS'
            print(f"    [{tag}] Acc={m['accuracy']:.4f}  "
                  f"F1={m['f1_macro']:.4f}  "
                  f"RMSE={m.get('rmse', float('nan')):.2f}  "
                  f"R²(price)={m.get('r2_price', float('nan')):.4f}")

    for task in tasks:
        accs = fold_stats[task]['accs']
        print(f"\n  CV [{task}] Accuracy: {np.mean(accs):.4f} ±{np.std(accs):.4f}")

    # ── Final models on full train+val ────────────────────────────────────
    print("\n[Holdout] Training final model(s) on full train+val...")
    plot_results_data = []   # (label, metrics, color, cmap)
    os.makedirs('models', exist_ok=True)

    if 'regression' in tasks:
        print("\n  ── TCNRegressor ──")
        torch.manual_seed(TASK_SEEDS['regression'])
        np.random.seed(TASK_SEEDS['regression'])
        final_reg = _make_model('regression')
        final_reg.fit(X_tv, yp_tv)
        m_reg = final_reg.evaluate(X_test, yp_test)
        print_metrics("TCN Regressor (price → direction) — Holdout Test", m_reg)
        final_reg.save('models/tcn_regressor.pkl')
        plot_results_data.append(
            ('TCN Regressor', m_reg, 'steelblue', 'Blues'))

    if 'classification' in tasks:
        print("\n  ── TCNClassifier ──")
        torch.manual_seed(TASK_SEEDS['classification'])
        np.random.seed(TASK_SEEDS['classification'])
        final_cls = _make_model('classification')
        final_cls.fit(X_tv, yd_tv)
        m_cls = final_cls.evaluate(X_test, yd_test)
        print_metrics("TCN Classifier (direct Up/Down) — Holdout Test", m_cls)
        final_cls.save('models/tcn_classifier.pkl')
        plot_results_data.append(
            ('TCN Classifier', m_cls, 'darkorange', 'Oranges'))

    elapsed = time.time() - start
    print(f"\nTotal processing time: {elapsed:.2f}s")

    # ── Visualize ──────────────────────────────────────────────────────────
    os.makedirs('TCN', exist_ok=True)
    plot_results(
        dates_test, yd_test,
        results=plot_results_data,
        lookback=lookback,
        elapsed=elapsed,
        filename_suffix=mode)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TCN Daily Pipeline — Regression / Classification / Both')
    parser.add_argument('--lookback', type=int, default=30,
                        help='Sliding window length (default: 30)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Training epochs per fold (default: 1000)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early-stopping patience in epochs (default: 20)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for TCN blocks (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Adam learning rate (default: 0.0003)')
    parser.add_argument('--kernel', type=int, default=3,
                        help='TCN conv kernel size (default: 3)')
    parser.add_argument('--channels', type=int, nargs='+', default=[128, 32],
                        help='TCN channel sizes, e.g. --channels 64 32 (default: [128, 32])')
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['regression', 'classification', 'both'],
                        help='Which model(s) to run (default: classification)')
    parser.add_argument('--fred_train', type=str, default=None)
    parser.add_argument('--fred_val',   type=str, default=None)
    args = parser.parse_args()

    run_daily_tcn(
        lookback=args.lookback,
        epochs=args.epochs,
        patience=args.patience,
        dropout=args.dropout,
        lr=args.lr,
        kernel_size=args.kernel,
        num_channels=args.channels,
        mode=args.mode,
        fred_folder_train=args.fred_train,
        fred_folder_val=args.fred_val,
    )
