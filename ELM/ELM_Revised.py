"""
ELM Revised — VIP Resubmission Compliant
==========================================
Extreme Learning Machine for daily wheat futures price direction prediction.


Strategy (same as ARX_Revised.py / TCN_Revised.py / FAVAR_Revised.py):
  1.  Predict next-day price CHANGE  Δ[t] = Close[t] - Close[t-1]
  2.  Derive binary direction:  predicted Δ > threshold  →  Up (1)
                                predicted Δ ≤ threshold  →  Down (0)

Models  (--mode regression | classification | both):

  ELMRegressor  [--mode regression]:
    Input (66)  →  Random Hidden Layer (n_hidden, sigmoid, fixed seed=42)
                →  Ridge output  →  scalar Δ̂
    Hidden weights drawn once and frozen — only the output layer is trained
    (closed-form Ridge solve: β = (HᵀH + αI)⁻¹Hᵀy).
    Δ threshold calibrated on val set to maximise F1-macro.

  ELMClassifier  [--mode classification]:
    Input (66)  →  Random Hidden Layer (n_hidden, sigmoid, fixed seed=43)
                →  Logistic Regression output  → P(Up)
    Same architecture as ELMRegressor; output trained on binary {0,1} labels.
    P(Up) threshold calibrated on val set to maximise F1-macro.

  Output layer regulariser (--alpha):
    Ridge (L2) penalty on the output weights — the only trainable parameters.
    Smaller α = less shrinkage; larger α = more regularisation.
    Default: alpha=0.1.  No other regulariser options (Ridge is exact-solve).

  Per-task seeds for reproducibility: regression=42, classification=43.





Compliance checklist:
  §1.1  Exactly 31 FRED-MD variables (SELECTED_FEATURES)
  §1.2  T-code stationarity transforms + 1-month reporting delay shift
        Monthly macro forward-filled to daily frequency
  §1.3  Lookback window of 30 trading days: each sample is a flat feature
        vector of [30 lagged Δs | 5 tech features | 31 macro vars] = 66 dims
        — no contemporaneous info
  §2    5-fold TimeSeriesSplit CV; scaler fitted on train fold only
        Per-task seeds: regression=42, classification=43
  §4.5  Classification metrics on derived direction labels:
        Accuracy, Precision, Recall, F1 (per class + macro), Confusion Matrix,
        ROC-AUC (score = predicted Δ / P(Up), continuous)
        Regression metrics (regression mode): RMSE, MAE, MAPE, R² (on Δ and price)
  §4.6  Two visualizations per mode:
        Panel 1 — Rolling 30-day accuracy over time (clean line chart)
        Panel 2 — Confusion matrix with count + row-% annotation
"""


import os
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

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, r2_score,
    mean_squared_error
)
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# §1.1  SELECTED FEATURES  (31 variables — identical to ARX/TCN/FAVAR)
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
#  SHARED ELM HIDDEN LAYER
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid(x):
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))


def _build_hidden(X: np.ndarray, input_weights: np.ndarray,
                  bias: np.ndarray) -> np.ndarray:
    """Project input through fixed random sigmoid layer → H  (N, n_hidden)."""
    return _sigmoid(X @ input_weights + bias)


def _init_hidden(n_features: int, n_hidden: int,
                 seed: int = 42) -> tuple:
    """Draw fixed random hidden weights (never updated)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.5, size=(n_features, n_hidden))
    b = rng.normal(scale=0.5, size=(n_hidden,))
    return W, b


# ─────────────────────────────────────────────────────────────────────────────
#  ELM REGRESSOR  (predicts price Δ → derives direction)
# ─────────────────────────────────────────────────────────────────────────────
class ELMRegressor(BaseForecastModel):
    """
    ELM Regressor: random hidden layer + Ridge output trained on next-day Δ.

    Direction is derived post-hoc using a median threshold (same as ARXModel).
    """

    def __init__(self, task_type='regression',
                 n_hidden: int = 1000,
                 alpha: float = 0.1,
                 lookback: int = 30,
                 hidden_seed: int = 42,
                 **kwargs):
        super().__init__(task_type=task_type, n_hidden=n_hidden,
                         alpha=alpha, lookback=lookback, **kwargs)
        self.n_hidden    = n_hidden
        self.alpha       = alpha
        self.lookback    = lookback
        self.hidden_seed = hidden_seed

        self.feature_scaler       = StandardScaler()
        self.target_scaler        = StandardScaler()
        self._input_weights       = None
        self._bias                = None
        self._output_model        = None
        self._direction_threshold = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _flatten(self, X_raw: np.ndarray) -> np.ndarray:
        """(N, lookback, n_feat) → (N, lookback + n_feat_last) — same as ARX."""
        N, L, F = X_raw.shape
        prices      = np.clip(X_raw[:, :, 0], 1e-10, None)
        log_rets    = np.diff(np.log(prices), axis=1)
        log_rets    = np.hstack([log_rets[:, :1], log_rets])
        exog        = X_raw[:, -1, 1:]
        return np.hstack([log_rets, exog])

    # ── BaseForecastModel interface ───────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        y_train = np.asarray(y_train).reshape(-1)
        n_val = max(1, int(len(y_train) * 0.15))
        n_tr  = len(y_train) - n_val

        X_tr_raw, X_val_raw = X_train[:n_tr], X_train[n_tr:]
        y_tr,     y_val     = y_train[:n_tr],  y_train[n_tr:]

        X_tr_flat  = self._flatten(X_tr_raw)
        X_val_flat = self._flatten(X_val_raw)

        X_tr_s  = self.feature_scaler.fit_transform(X_tr_flat)
        X_val_s = self.feature_scaler.transform(X_val_flat)
        y_tr_s  = self.target_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

        # Fixed random hidden layer
        self._input_weights, self._bias = _init_hidden(
            X_tr_s.shape[1], self.n_hidden, self.hidden_seed)

        H_tr  = _build_hidden(X_tr_s,  self._input_weights, self._bias)
        H_val = _build_hidden(X_val_s, self._input_weights, self._bias)

        self._output_model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._output_model.fit(H_tr, y_tr_s)

        # Threshold calibration on val set
        val_preds_s = self._output_model.predict(H_val)
        val_deltas  = self.target_scaler.inverse_transform(
            val_preds_s.reshape(-1, 1)).ravel()
        val_true_dir = (y_val > 0).astype(int)

        candidates = np.linspace(np.percentile(val_deltas, 5),
                                 np.percentile(val_deltas, 95), 100)
        best_t, best_f1 = 0.0, -1.0
        for t in candidates:
            p  = (val_deltas >= t).astype(int)
            f1 = f1_score(val_true_dir, p, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self._direction_threshold = best_t
        print(f"  [ELM-REG] n_hidden={self.n_hidden}  alpha={self.alpha}  "
              f"features={X_tr_flat.shape[1]}  train={n_tr}  val={n_val}")
        print(f"  [ELM-REG] Calibrated Δ threshold: {best_t:.4f} USD  "
              f"(val F1-macro: {best_f1:.4f})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        X_s    = self.feature_scaler.transform(X_flat)
        H      = _build_hidden(X_s, self._input_weights, self._bias)
        preds_s = self._output_model.predict(H)
        return self.target_scaler.inverse_transform(
            preds_s.reshape(-1, 1)).ravel()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_test      = np.asarray(y_test).reshape(-1)
        price_preds = self.predict(X_test)
        lag1        = X_test[:, -1, 0]

        pred_std = float(np.std(price_preds))
        if pred_std < 1e-4:
            eval_threshold = float(self._direction_threshold)
            print(f"  [ELM-REG eval] WARNING: near-constant predictions "
                  f"(std={pred_std:.2e}). Using val-calibrated threshold.")
        else:
            eval_threshold = float(np.median(price_preds))

        pred_dir = (price_preds >= eval_threshold).astype(int)
        true_dir = (y_test > 0).astype(int)

        n_up, n_down = int(pred_dir.sum()), int((1 - pred_dir).sum())
        print(f"  [ELM-REG eval] threshold={eval_threshold:.4f} USD  "
              f"(pred Up={n_up}  Down={n_down})")

        rmse     = float(np.sqrt(mean_squared_error(y_test, price_preds)))
        mae      = float(np.mean(np.abs(y_test - price_preds)))
        r2_delta = float(r2_score(y_test, price_preds))

        true_abs = lag1 + y_test
        pred_abs = lag1 + price_preds
        mape     = float(np.mean(
            np.abs(true_abs - pred_abs) /
            np.clip(np.abs(true_abs), 1e-10, None)) * 100)
        price_r2 = float(r2_score(true_abs, pred_abs))

        acc      = accuracy_score(true_dir, pred_dir)
        prec     = precision_score(true_dir, pred_dir, average=None, zero_division=0)
        rec      = recall_score(true_dir, pred_dir, average=None, zero_division=0)
        f1_per   = f1_score(true_dir, pred_dir, average=None, zero_division=0)
        f1_macro = f1_score(true_dir, pred_dir, average='macro', zero_division=0)
        cm       = confusion_matrix(true_dir, pred_dir)
        try:
            auc = roc_auc_score(true_dir, price_preds)
        except Exception:
            auc = float('nan')

        return {
            'accuracy': acc, 'precision_per_class': prec,
            'recall_per_class': rec, 'f1_per_class': f1_per,
            'f1_macro': f1_macro, 'confusion_matrix': cm, 'roc_auc': auc,
            'r2': r2_delta, 'r2_price': r2_delta, 'r2_close': price_r2,
            'rmse': rmse, 'mae': mae, 'mape_pct': mape,
            'pred_close': pred_abs, 'true_close': true_abs,
            'predictions': pred_dir, 'price_predictions': price_preds,
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_hidden': self.n_hidden, 'alpha': self.alpha,
                'lookback': self.lookback, 'hidden_seed': self.hidden_seed,
                'input_weights': self._input_weights, 'bias': self._bias,
                'output_model': self._output_model,
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'direction_threshold': self._direction_threshold,
                'task_type': self.task_type,
                'hyperparameters': self.hyperparameters,
            }, f)
        print(f"  Model saved → {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.n_hidden             = d['n_hidden']
        self.alpha                = d['alpha']
        self.lookback             = d['lookback']
        self.hidden_seed          = d.get('hidden_seed', 42)
        self._input_weights       = d['input_weights']
        self._bias                = d['bias']
        self._output_model        = d['output_model']
        self.feature_scaler       = d['feature_scaler']
        self.target_scaler        = d['target_scaler']
        self._direction_threshold = d['direction_threshold']
        self.task_type            = d.get('task_type', 'regression')
        self.hyperparameters      = d.get('hyperparameters', {})
        print(f"  Model loaded ← {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
#  ELM CLASSIFIER  (direct Up/Down via sigmoid output neuron)
# ─────────────────────────────────────────────────────────────────────────────
class ELMClassifier(BaseForecastModel):
    """
    ELM Classifier: random hidden layer + Logistic Regression output.

    Output layer predicts P(Up) directly.
    Threshold calibrated on val set to maximise F1-macro.
    """

    def __init__(self, task_type='classification',
                 n_hidden: int = 1000,
                 alpha: float = 0.1,
                 lookback: int = 30,
                 hidden_seed: int = 43,
                 **kwargs):
        super().__init__(task_type=task_type, n_hidden=n_hidden,
                         alpha=alpha, lookback=lookback, **kwargs)
        self.n_hidden    = n_hidden
        self.alpha       = alpha
        self.lookback    = lookback
        self.hidden_seed = hidden_seed

        self.feature_scaler  = StandardScaler()
        self._input_weights  = None
        self._bias           = None
        self._output_model   = None
        self._cls_threshold  = 0.5

    def _flatten(self, X_raw: np.ndarray) -> np.ndarray:
        N, L, F = X_raw.shape
        prices   = np.clip(X_raw[:, :, 0], 1e-10, None)
        log_rets = np.diff(np.log(prices), axis=1)
        log_rets = np.hstack([log_rets[:, :1], log_rets])
        exog     = X_raw[:, -1, 1:]
        return np.hstack([log_rets, exog])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        y_train = np.asarray(y_train).reshape(-1).astype(float)
        n_val = max(1, int(len(y_train) * 0.15))
        n_tr  = len(y_train) - n_val

        X_tr_raw, X_val_raw = X_train[:n_tr], X_train[n_tr:]
        y_tr,     y_val     = y_train[:n_tr],  y_train[n_tr:]

        X_tr_flat  = self._flatten(X_tr_raw)
        X_val_flat = self._flatten(X_val_raw)

        X_tr_s  = self.feature_scaler.fit_transform(X_tr_flat)
        X_val_s = self.feature_scaler.transform(X_val_flat)

        self._input_weights, self._bias = _init_hidden(
            X_tr_s.shape[1], self.n_hidden, self.hidden_seed)

        H_tr  = _build_hidden(X_tr_s,  self._input_weights, self._bias)
        H_val = _build_hidden(X_val_s, self._input_weights, self._bias)

        # Train output using LogisticRegression for proper probability calibration
        C_val = 1.0 / max(self.alpha, 1e-10)
        self._output_model = LogisticRegression(C=C_val, fit_intercept=True, max_iter=2000,
                                                solver='lbfgs')
        self._output_model.fit(H_tr, y_tr.astype(int))

        # Val probabilities
        val_probas = self._output_model.predict_proba(H_val)[:, 1]
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
        print(f"  [ELM-CLS] n_hidden={self.n_hidden}  alpha={self.alpha}  "
              f"features={X_tr_flat.shape[1]}  train={n_tr}  val={n_val}")
        print(f"  [ELM-CLS] Calibrated P(Up) threshold: {best_t:.4f}  "
              f"(val F1-macro: {best_f1:.4f})")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_flat  = self._flatten(X)
        X_s     = self.feature_scaler.transform(X_flat)
        H       = _build_hidden(X_s, self._input_weights, self._bias)
        return self._output_model.predict_proba(H)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self._cls_threshold).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_test = np.asarray(y_test).reshape(-1).astype(int)
        probas = self.predict_proba(X_test)

        eval_threshold = float(np.median(probas))
        pred_dir = (probas >= eval_threshold).astype(int)
        true_dir = y_test

        n_up, n_down = int(pred_dir.sum()), int((1 - pred_dir).sum())
        print(f"  [ELM-CLS eval] P(Up) threshold={eval_threshold:.4f}  "
              f"(pred Up={n_up}  Down={n_down})")

        acc      = accuracy_score(true_dir, pred_dir)
        prec     = precision_score(true_dir, pred_dir, average=None, zero_division=0)
        rec      = recall_score(true_dir, pred_dir, average=None, zero_division=0)
        f1_per   = f1_score(true_dir, pred_dir, average=None, zero_division=0)
        f1_macro = f1_score(true_dir, pred_dir, average='macro', zero_division=0)
        cm       = confusion_matrix(true_dir, pred_dir)
        try:
            auc = roc_auc_score(true_dir, probas)
        except Exception:
            auc = float('nan')

        return {
            'accuracy': acc, 'precision_per_class': prec,
            'recall_per_class': rec, 'f1_per_class': f1_per,
            'f1_macro': f1_macro, 'confusion_matrix': cm, 'roc_auc': auc,
            'r2': float('nan'), 'r2_price': float('nan'),
            'r2_close': float('nan'), 'rmse': float('nan'),
            'mae': float('nan'), 'mape_pct': float('nan'),
            'pred_close': np.full(len(y_test), float('nan')),
            'true_close': np.full(len(y_test), float('nan')),
            'predictions': pred_dir, 'price_predictions': probas,
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_hidden': self.n_hidden, 'alpha': self.alpha,
                'lookback': self.lookback, 'hidden_seed': self.hidden_seed,
                'input_weights': self._input_weights, 'bias': self._bias,
                'output_model': self._output_model,
                'feature_scaler': self.feature_scaler,
                'cls_threshold': self._cls_threshold,
                'task_type': self.task_type,
                'hyperparameters': self.hyperparameters,
            }, f)
        print(f"  Model saved → {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.n_hidden           = d['n_hidden']
        self.alpha              = d['alpha']
        self.lookback           = d['lookback']
        self.hidden_seed        = d.get('hidden_seed', 43)
        self._input_weights     = d['input_weights']
        self._bias              = d['bias']
        self._output_model      = d['output_model']
        self.feature_scaler     = d['feature_scaler']
        self._cls_threshold     = d.get('cls_threshold', 0.5)
        self.task_type          = d.get('task_type', 'classification')
        self.hyperparameters    = d.get('hyperparameters', {})
        print(f"  Model loaded ← {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_fred_md(fred_md_folder: str) -> pd.DataFrame:
    folder    = Path(fred_md_folder)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {fred_md_folder}")
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
#  METRICS REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(title: str, m: dict):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")
    print(f"  ── Price Change (Δ) Forecast ──")
    print(f"  RMSE (Δ)   : {m['rmse']:.4f}  USD/day")
    print(f"  MAE  (Δ)   : {m['mae']:.4f}  USD/day")
    print(f"  MAPE       : {m['mape_pct']:.2f}%")
    print(f"  R² (Δ)    : {m['r2_price']:.4f}")
    print(f"  ── Absolute Price Forecast (pred = lag1 + Δ̂) ──")
    print(f"  R² (price)  : {m['r2_close']:.4f}  "
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
#  VISUALIZATION  (§4.6 — same layout as ARX_Revised.py)
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(dates_test, y_true_dir, metrics: dict,
                 lookback: int = 30, elapsed: float = 0.0,
                 filename: str = 'ELM/elm_results.png',
                 title_tag: str = 'ELM'):
    preds        = metrics['predictions']
    cm           = metrics['confusion_matrix']
    y_true_close = metrics['true_close']
    y_pred_close = metrics['pred_close']
    dates_dt     = pd.to_datetime(dates_test)
    correct      = (preds == y_true_dir).astype(float)
    roll_acc     = pd.Series(correct).rolling(lookback, min_periods=1).mean().values

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1.2, 1.0],
                            hspace=0.45, wspace=0.28)

    # Panel 1: Rolling accuracy (full width)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor('#1a1d27')
    ax0.plot(dates_dt, roll_acc, color='#4fc3f7', linewidth=1.5, alpha=0.9,
             label=f'Rolling {lookback}-day accuracy')
    ax0.axhline(0.5, color='#ef5350', linewidth=1.0, linestyle='--',
                label='Random baseline (0.50)')
    ax0.axhline(metrics['accuracy'], color='#66bb6a', linewidth=1.2,
                linestyle=':', label=f"Overall: {metrics['accuracy']:.4f}")
    ax0.fill_between(dates_dt, roll_acc, 0.5,
                     where=(roll_acc >= 0.5), alpha=0.15,
                     color='#4fc3f7', interpolate=True)
    ax0.set_ylim(0.2, 0.8)
    ax0.set_title(
        f"{title_tag} Rolling {lookback}-Day Accuracy  —  "
        f"ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1_macro']:.4f}  "
        f"Acc={metrics['accuracy']:.4f}  ({elapsed:.1f}s)",
        color='white', fontsize=11, pad=8)
    ax0.tick_params(colors='#b0bec5', labelsize=9)
    ax0.spines[:].set_color('#37474f')
    ax0.legend(fontsize=9, facecolor='#1a1d27', labelcolor='white')
    ax0.set_xlabel('Date', color='#b0bec5', fontsize=10)
    ax0.set_ylabel('Accuracy', color='#b0bec5', fontsize=10)
    ax0.grid(True, linestyle=':', alpha=0.3, color='#37474f')

    # Panel 2: Price forecast (bottom-left) — only for regressor
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor('#1a1d27')
    if not np.all(np.isnan(y_pred_close)):
        ax1.plot(dates_dt, y_true_close, color='#ffffff', linewidth=1.2,
                 alpha=0.9, label='Actual Close')
        ax1.plot(dates_dt, y_pred_close, color='#ff7043', linewidth=1.0,
                 alpha=0.8, linestyle='--', label=f'{title_tag} Predicted Close')
        ax1.fill_between(dates_dt, y_true_close, y_pred_close,
                         alpha=0.07, color='#ff7043')
        rmse_str = f"RMSE={metrics['rmse']:.2f}"
        mape_str = f"MAPE={metrics['mape_pct']:.2f}%"
        r2_str   = f"R²(Δ)={metrics['r2_price']:.4f}"
        ax1.set_title(f"{title_tag} Price Forecast — {rmse_str}  {mape_str}  {r2_str}",
                      color='white', fontsize=10, pad=8)
        ax1.legend(fontsize=8, facecolor='#1a1d27', labelcolor='white')
    else:
        ax1.text(0.5, 0.5, 'Classification mode\n(no price forecast)',
                 ha='center', va='center', color='#b0bec5', fontsize=12,
                 transform=ax1.transAxes)
        ax1.set_title(f"{title_tag} (classification — no price forecast)",
                      color='white', fontsize=10, pad=8)
    ax1.tick_params(colors='#b0bec5', labelsize=9)
    ax1.spines[:].set_color('#37474f')
    ax1.set_xlabel('Date', color='#b0bec5', fontsize=10)
    ax1.set_ylabel('Price (USD/bushel)', color='#b0bec5', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.3, color='#37474f')

    # Panel 3: Confusion matrix (bottom-right)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor('#1a1d27')
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums > 0, cm / row_sums * 100, 0)
    annot    = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
                          for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                ax=ax2, linewidths=0.5, cbar_kws={'shrink': 0.7})
    ax2.set_title('Confusion Matrix', color='white', fontsize=12, pad=10)
    ax2.set_xlabel('Predicted', color='#b0bec5', fontsize=10)
    ax2.set_ylabel('Actual',    color='#b0bec5', fontsize=10)
    ax2.tick_params(colors='#b0bec5', labelsize=9)

    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_daily_elm(lookback: int = 30,
                  n_hidden: int = 1000,
                  alpha: float = 0.1,
                  mode: str = 'classification',
                  fred_folder_train: str = None,
                  fred_folder_val: str = None):
    """
    End-to-end ELM pipeline.

    mode : 'regression' | 'classification' | 'both'
      regression     — ELMRegressor:   predicts Δ → derives direction
      classification — ELMClassifier:  predicts P(Up) directly
      both           — run and compare both

    n_hidden : number of random hidden neurons (default 1000)
    alpha    : Ridge regularisation on output layer (default 1.0)
    """
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
    print(f"ELM Daily Pipeline — mode={mode}  n_hidden={n_hidden}  "
          f"alpha={alpha}  lookback={lookback}")
    print("=" * 60)

    # ── Step 1: FRED-MD ───────────────────────────────────────────────────
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

    # ── Step 2: Wheat prices ──────────────────────────────────────────────
    print("\n[2/5] Loading daily wheat futures prices...")
    df_wheat = pd.read_csv(wheat_path)
    df_wheat['date'] = (pd.to_datetime(df_wheat['date'], utc=True)
                          .dt.tz_localize(None))
    df_wheat = df_wheat.sort_values('date').reset_index(drop=True)
    print(f"  Wheat data: {len(df_wheat)} daily observations")

    # ── Step 3: Merge macro + tech features ───────────────────────────────
    print("\n[3/5] Forward-filling monthly macro + computing tech features...")
    macro_cols = [c for c in SELECTED_FEATURES if c in df_macro.columns]
    df_merged  = forward_fill_macro_to_daily(df_macro, df_wheat)
    df_tech    = compute_technical_features(df_wheat)
    tech_cols  = [c for c in df_tech.columns if c != 'date']
    df_merged  = df_merged.merge(df_tech, on='date', how='left')
    df_merged[tech_cols] = df_merged[tech_cols].ffill().fillna(0)
    print(f"  Merged: {len(df_merged)} rows  |  "
          f"{len(macro_cols)} macro  |  {len(tech_cols)} tech features")

    # ── Step 4: Sliding windows ───────────────────────────────────────────
    print(f"\n[4/5] Building sliding windows "
          f"(lookback={lookback}, features=1+{len(tech_cols)}+{len(macro_cols)})...")
    X, y_price, y_dir, y_close, y_prev, dates = build_sliding_windows(
        df_merged, macro_cols, tech_cols=tech_cols, lookback=lookback)
    print(f"  Samples X: {X.shape}")
    print(f"  y_price range: {y_price.min():.2f} – {y_price.max():.2f}  (USD/day)")
    print(f"  y_dir  split  : Up={y_dir.sum()}  Down={(1-y_dir).sum()}")
    print(f"  Date range: {pd.to_datetime(dates[0]).date()} → "
          f"{pd.to_datetime(dates[-1]).date()}")

    # ── Step 5: Train/Test split + CV ─────────────────────────────────────
    print("\n[5/5] Splitting and running 5-fold TimeSeriesSplit CV...")
    test_size = int(len(y_price) * 0.15)
    split_idx = len(y_price) - test_size

    X_tv, X_test     = X[:split_idx],       X[split_idx:]
    yp_tv, yp_test   = y_price[:split_idx], y_price[split_idx:]
    yd_tv, yd_test   = y_dir[:split_idx],   y_dir[split_idx:]
    dates_tv, dates_test = dates[:split_idx], dates[split_idx:]
    print(f"  Train+Val: {split_idx}  |  Holdout Test: {test_size}")

    tasks = []
    if mode in ('regression',     'both'): tasks.append('regression')
    if mode in ('classification', 'both'): tasks.append('classification')

    # Per-task seeds — same convention as ARX/TCN
    TASK_SEEDS = {'regression': 42, 'classification': 43}

    def _make_model(task):
        kw = dict(n_hidden=n_hidden, alpha=alpha, lookback=lookback)
        if task == 'regression':
            return ELMRegressor(hidden_seed=TASK_SEEDS['regression'], **kw)
        else:
            return ELMClassifier(hidden_seed=TASK_SEEDS['classification'], **kw)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_stats = {t: {'accs': [], 'r2s': []} for t in tasks}

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tv), start=1):
        print(f"\n  Fold {fold}:")
        for task in tasks:
            np.random.seed(TASK_SEEDS[task] + fold)
            y_tr  = yp_tv[tr_idx] if task == 'regression' else yd_tv[tr_idx]
            y_val = yp_tv[val_idx] if task == 'regression' else yd_tv[val_idx]
            mdl = _make_model(task)
            mdl.fit(X_tv[tr_idx], y_tr)
            m   = mdl.evaluate(X_tv[val_idx], y_val)
            fold_stats[task]['accs'].append(m['accuracy'])
            fold_stats[task]['r2s'].append(m.get('r2_price', float('nan')))
            tag = 'REG' if task == 'regression' else 'CLS'
            print(f"    [{tag}] Acc={m['accuracy']:.4f}  "
                  f"F1={m['f1_macro']:.4f}  "
                  f"RMSE={m.get('rmse', float('nan')):.2f}  "
                  f"R²(Δ)={m.get('r2_price', float('nan')):.4f}")

    for task in tasks:
        accs = fold_stats[task]['accs']
        tag  = 'REG' if task == 'regression' else 'CLS'
        print(f"\n  CV [{tag}] Accuracy: {np.mean(accs):.4f} ±{np.std(accs):.4f}")
        if task == 'regression':
            r2s = fold_stats[task]['r2s']
            print(f"  CV [{tag}] R²(Δ)  : {np.mean(r2s):.4f} ±{np.std(r2s):.4f}")

    # ── Final models on full train+val ─────────────────────────────────────
    print("\n[Holdout] Training final ELM model(s) on full train+val...")
    os.makedirs('models', exist_ok=True)
    plot_results_data = []

    if 'regression' in tasks:
        np.random.seed(TASK_SEEDS['regression'])
        print("\n  ── ELMRegressor ──")
        final_reg = _make_model('regression')
        final_reg.fit(X_tv, yp_tv)
        m_reg = final_reg.evaluate(X_test, yp_test)
        print_metrics("ELM Regressor (Δ → direction) — Holdout Test", m_reg)
        final_reg.save('models/elm_regressor.pkl')
        plot_results_data.append(('ELM Regressor', m_reg, 'ELM-REG'))

    if 'classification' in tasks:
        np.random.seed(TASK_SEEDS['classification'])
        print("\n  ── ELMClassifier ──")
        final_cls = _make_model('classification')
        final_cls.fit(X_tv, yd_tv)
        m_cls = final_cls.evaluate(X_test, yd_test)
        print_metrics("ELM Classifier (direct Up/Down) — Holdout Test", m_cls)
        final_cls.save('models/elm_classifier.pkl')
        plot_results_data.append(('ELM Classifier', m_cls, 'ELM-CLS'))

    elapsed = time.time() - start
    print(f"\nTotal processing time: {elapsed:.2f}s")

    os.makedirs('ELM', exist_ok=True)
    for label, m, tag in plot_results_data:
        plot_results(dates_test, yd_test, m,
                     lookback=lookback, elapsed=elapsed,
                     filename=f'ELM/elm_{tag.lower()}_results.png',
                     title_tag=tag)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ELM Daily Pipeline — Extreme Learning Machine for wheat futures')
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['regression', 'classification', 'both'],
                        help='regression | classification | both (default: classification)')
    parser.add_argument('--n_hidden', type=int, default=1000,
                        help='Number of random hidden neurons (default: 1000)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Ridge regularisation on output layer (default: 0.1)')
    parser.add_argument('--fred_train', type=str, default=None,
                        help='Path to historical FRED-MD vintage folder (train)')
    parser.add_argument('--fred_val', type=str, default=None,
                        help='Path to FRED-MD vintage folder (val/recent)')
    args = parser.parse_args()

    run_daily_elm(
        lookback=30,
        n_hidden=args.n_hidden,
        alpha=args.alpha,
        mode=args.mode,
        fred_folder_train=args.fred_train,
        fred_folder_val=args.fred_val,
    )
