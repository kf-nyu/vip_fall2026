"""
ARX Revised — VIP Resubmission Compliant
==========================================
Autoregressive model with Exogenous Inputs (ARX) for daily wheat futures
price direction prediction.

Strategy (same as TCN_Revised.py / FAVAR_Revised.py / ELM_Revised.py):
  1.  Predict next-day price CHANGE  Δ[t] = Close[t] - Close[t-1]
  2.  Derive binary direction:  predicted Δ > threshold  →  Up (1)
                                predicted Δ ≤ threshold  →  Down (0)

Models  (--mode regression | classification | both):

  ARXModel  [--mode regression]:
    Δ[t] = α
           + β₁·Δ[t-1] + β₂·Δ[t-2] + ... + β₃₀·Δ[t-30]   [AR part]
           + γ₁·macro₁[t-1] + ... + γ₃₁·macro₃₁[t-1]      [X part]
           + ε[t]
    Direction derived post-hoc using a val-calibrated median threshold.

    Regulariser options (--regularizer flag):
      ridge      — L2 penalty; shrinks all coefficients, keeps all features.
                   Principled baseline for high-dimensional ARX.
                   Controlled by --alpha.
      elasticnet — L1 + L2 penalty; zeros irrelevant lags / macro vars while
                   handling collinear interest-rate spreads via the L2 term.
                   Controlled by --alpha and --l1_ratio (default 0.5).
      bayesian   — Bayesian Ridge; estimates regularisation strength via EM
                   — no --alpha tuning required.

  ARXClassifier  [--mode classification]:
    LogisticRegression on the same 66-dim feature vector — directly predicts
    binary Up/Down without an intermediate price forecast.
    Regulariser options: same ridge / elasticnet / bayesian as above.
    P(Up) threshold calibrated on val set to maximise F1-macro.
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

from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, LogisticRegression
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
# §1.1  SELECTED FEATURES (31 variables, fixed — same as TCN/FAVAR)
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
#  BASE CLASS  (mirrors TCN_Revised.py)
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
#  ARX MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ARXModel(BaseForecastModel):
    """
    ARX(p=30, X=31+5) — linear model with pluggable regulariser.

    Input layout (matches TCN sliding window X, shape (N, 30, 37)):
      Flattened to a 1-D feature vector per sample:
        [30 price-change lags]  — log-returns at steps t-30 … t-1
        [5 tech features]       — last-step tech indicators (already t-1 shifted)
        [31 macro features]     — last-step macro values   (already 1-month delayed)
      Total: 30 + 5 + 31 = 66 features

    The model accepts the raw (N, 30, 37) window array exactly as built by
    build_sliding_windows() so it can be dropped into the same pipeline as
    TCN/FAVAR with zero data-prep changes.

    Regulariser (--regularizer):
      'ridge'      — Ridge(alpha)          L2, closed-form
      'elasticnet' — ElasticNet(alpha, l1_ratio)  L1+L2, coordinate descent
      'bayesian'   — BayesianRidge()       L2, alpha auto-estimated via EM

    Threshold calibration:
      Val-set scan over predicted-Δ percentiles to maximise F1-macro on the
      val direction labels — identical to TCNRegressor.
    """

    # Maps CLI name → constructor
    _REGULARIZERS = {
        'ridge'      : lambda alpha, l1_ratio: Ridge(alpha=alpha, fit_intercept=True),
        'elasticnet' : lambda alpha, l1_ratio: ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                                          fit_intercept=True, max_iter=5000),
        'bayesian'   : lambda alpha, l1_ratio: BayesianRidge(),   # alpha auto-estimated
    }

    def __init__(self, task_type: str = 'regression',
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 regularizer: str = 'ridge',
                 lookback: int = 30,
                 **kwargs):
        if regularizer not in self._REGULARIZERS:
            raise ValueError(f"regularizer must be one of "
                             f"{list(self._REGULARIZERS)}; got '{regularizer}'")
        super().__init__(task_type=task_type, alpha=alpha, l1_ratio=l1_ratio,
                         regularizer=regularizer, lookback=lookback, **kwargs)
        self.alpha       = alpha
        self.l1_ratio    = l1_ratio
        self.regularizer = regularizer
        self.lookback    = lookback

        self.regressor      = self._REGULARIZERS[regularizer](alpha, l1_ratio)
        self.feature_scaler = StandardScaler()
        self.target_scaler  = StandardScaler()
        self._direction_threshold = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _flatten(self, X_raw: np.ndarray) -> np.ndarray:
        """
        (N, lookback, n_feat) → (N, lookback + n_feat_last)

        AR part  : log-returns from price channel (col 0) across all lags
        X  part  : tech + macro values at the LAST timestep (t-1)
                   (no stacking all lags — standard ARX uses current exogenous)
        """
        N, L, F = X_raw.shape
        prices = X_raw[:, :, 0]                    # (N, L) raw Close per lag
        # Compute log-returns within the window
        prices_safe = np.clip(prices, 1e-10, None)
        log_rets    = np.diff(np.log(prices_safe), axis=1)   # (N, L-1)
        # Pad with first value repeated so length stays L
        first_ret   = log_rets[:, :1]
        log_rets    = np.hstack([first_ret, log_rets])        # (N, L)
        # Last-step exogenous: tech + macro (channels 1 onward at step t-1)
        exog = X_raw[:, -1, 1:]                               # (N, F-1)
        return np.hstack([log_rets, exog])                    # (N, L + F-1)

    # ── BaseForecastModel interface ───────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        X_train : (N, lookback, n_features_raw)
        y_train : (N,) next-day price changes Δ
        """
        y_train = np.asarray(y_train).reshape(-1)

        # Internal 15% val split for threshold calibration
        n_val = max(1, int(len(y_train) * 0.15))
        n_tr  = len(y_train) - n_val

        X_tr_raw, X_val_raw = X_train[:n_tr], X_train[n_tr:]
        y_tr,     y_val     = y_train[:n_tr],  y_train[n_tr:]

        # Flatten windows → feature vectors
        X_tr_flat  = self._flatten(X_tr_raw)
        X_val_flat = self._flatten(X_val_raw)

        # Scale features (fit on train only — §2)
        X_tr_s  = self.feature_scaler.fit_transform(X_tr_flat)
        X_val_s = self.feature_scaler.transform(X_val_flat)

        # Scale target
        y_tr_s  = self.target_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

        # Fit regressor
        self.regressor.fit(X_tr_s, y_tr_s)

        # Threshold calibration on val set (scan percentiles 5–95)
        val_preds_s = self.regressor.predict(X_val_s)
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
        n_feat = X_tr_flat.shape[1]
        reg_label = (
            f"Ridge(alpha={self.alpha})"
            if self.regularizer == 'ridge' else
            f"ElasticNet(alpha={self.alpha}, l1_ratio={self.l1_ratio})"
            if self.regularizer == 'elasticnet' else
            f"BayesianRidge(alpha_={getattr(self.regressor, 'alpha_', 'auto'):.4f})"
        )
        print(f"  [ARX] Fitted {reg_label}  "
              f"features={n_feat}  "
              f"train={n_tr}  val={n_val}")
        print(f"  [ARX] Calibrated Δ threshold: {best_t:.4f} USD  "
              f"(val F1-macro: {best_f1:.4f})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted next-day price changes Δ, shape (N,)."""
        X_flat = self._flatten(X)
        X_s    = self.feature_scaler.transform(X_flat)
        preds_s = self.regressor.predict(X_s)
        return self.target_scaler.inverse_transform(
            preds_s.reshape(-1, 1)).ravel()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        y_test : next-day price CHANGES Δ, shape (N,)

        Threshold strategy (degenerate-safe):
          1. Prefer median of test predictions — balances Up/Down 50/50.
          2. If prediction std is near zero (heavy regularisation squashes all
             coefficients → near-constant output), every value equals the median
             so >= median gives all-Up.  Fall back to the val-calibrated threshold
             from fit() instead.
          3. Last resort: 0.0 (positive Δ → Up).
        """
        y_test      = np.asarray(y_test).reshape(-1)
        price_preds = self.predict(X_test)          # predicted Δs
        lag1        = X_test[:, -1, 0]              # last raw Close in window

        # Degenerate-safe threshold selection
        pred_std = float(np.std(price_preds))
        if pred_std < 1e-4:
            # Near-constant predictions: median trick collapses to all-Up.
            # Use the threshold calibrated on the validation set during fit().
            eval_threshold = float(self._direction_threshold)
            print(f"  [ARX eval] WARNING: near-constant predictions "
                  f"(std={pred_std:.2e}). Using val-calibrated threshold "
                  f"{eval_threshold:.4f} USD. Regularisation may be too strong "
                  f"— try smaller --alpha or --regularizer ridge.")
        else:
            eval_threshold = float(np.median(price_preds))

        pred_dir = (price_preds >= eval_threshold).astype(int)
        true_dir = (y_test > 0).astype(int)

        n_up, n_down = int(pred_dir.sum()), int((1 - pred_dir).sum())
        if n_up == 0 or n_down == 0:
            print(f"  [ARX eval] WARNING: degenerate — all "
                  f"{'Up' if n_up > 0 else 'Down'}. "
                  f"Metrics will be uninformative.")
        print(f"  [ARX eval] threshold={eval_threshold:.4f} USD  "
              f"(pred Up={n_up}  Down={n_down})")

        # ── Regression metrics on Δ ──
        rmse    = float(np.sqrt(mean_squared_error(y_test, price_preds)))
        mae     = float(np.mean(np.abs(y_test - price_preds)))
        r2_delta = float(r2_score(y_test, price_preds))

        # MAPE on absolute price (avoid division by near-zero Δ)
        true_abs = lag1 + y_test
        pred_abs = lag1 + price_preds
        mape = float(np.mean(
            np.abs(true_abs - pred_abs) /
            np.clip(np.abs(true_abs), 1e-10, None)) * 100)

        # ── Price-level R² (high due to autocorrelation — noted) ──
        price_r2 = float(r2_score(true_abs, pred_abs))

        # ── Classification metrics ──
        acc      = accuracy_score(true_dir, pred_dir)
        prec     = precision_score(true_dir, pred_dir, average=None, zero_division=0)
        rec      = recall_score(true_dir, pred_dir, average=None, zero_division=0)
        f1       = f1_score(true_dir, pred_dir, average=None, zero_division=0)
        f1_macro = f1_score(true_dir, pred_dir, average='macro', zero_division=0)
        cm       = confusion_matrix(true_dir, pred_dir)
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
            'r2'                 : r2_delta,
            'r2_price'           : r2_delta,          # on Δ
            'r2_close'           : price_r2,          # on absolute price
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
            'alpha'              : self.alpha,
            'l1_ratio'           : self.l1_ratio,
            'regularizer'        : self.regularizer,
            'lookback'           : self.lookback,
            'regressor'          : self.regressor,
            'feature_scaler'     : self.feature_scaler,
            'target_scaler'      : self.target_scaler,
            'direction_threshold': self._direction_threshold,
            'task_type'          : self.task_type,
            'hyperparameters'    : self.hyperparameters,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"  Model saved → {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.alpha                  = d['alpha']
        self.l1_ratio               = d.get('l1_ratio', 0.5)
        self.regularizer            = d.get('regularizer', 'ridge')
        self.lookback               = d['lookback']
        self.regressor              = d.get('regressor', d.get('ridge'))  # back-compat
        self.feature_scaler         = d['feature_scaler']
        self.target_scaler          = d['target_scaler']
        self._direction_threshold   = d['direction_threshold']
        self.task_type              = d.get('task_type', 'regression')
        self.hyperparameters        = d.get('hyperparameters', {})
        print(f"  Model loaded ← {filepath}")

    def get_top_features(self, n: int = 10,
                         tech_cols: list = None,
                         macro_cols: list = None) -> pd.DataFrame:
        """
        Return the top-n most influential features by |coefficient|.
        Useful for interpretation (which lags / macro vars matter most).
        """
        coef = self.regressor.coef_
        lookback = self.lookback

        names = [f"Δ_lag{i+1}" for i in range(lookback)]
        if tech_cols:
            names += [f"tech_{c}" for c in tech_cols]
        if macro_cols:
            names += list(macro_cols)
        # Pad names if mismatch
        while len(names) < len(coef):
            names.append(f"feat_{len(names)}")
        names = names[:len(coef)]

        df = pd.DataFrame({'feature': names, 'coefficient': coef})
        df['abs_coef'] = df['coefficient'].abs()
        return df.nlargest(n, 'abs_coef').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ARX CLASSIFIER  (direct Up/Down prediction via Logistic Regression)
# ─────────────────────────────────────────────────────────────────────────────

class ARXClassifier(BaseForecastModel):
    """
    ARX classifier: directly predicts binary Up/Down direction using
    Logistic Regression on the same 66-feature flattened window as ARXModel.

    Regulariser mapping (--regularizer):
      'ridge'      → LR with L2 penalty (solver=lbfgs)   — same spirit as Ridge
      'elasticnet' → LR with ElasticNet penalty (solver=saga, l1_ratio)
      'bayesian'   → LR with L2 penalty (no sklearn Bayesian LR natively)

    Note: LogisticRegression uses C = 1/alpha (C↑ = less regularisation),
    so --alpha semantics are preserved: higher alpha = stronger regularisation.
    """

    _REGULARIZERS = {
        'ridge'      : lambda alpha, l1_ratio: LogisticRegression(
                           penalty='l2',
                           C=1.0 / max(alpha, 1e-10),
                           solver='lbfgs', fit_intercept=True,
                           max_iter=5000, random_state=0),
        'elasticnet' : lambda alpha, l1_ratio: LogisticRegression(
                           penalty='elasticnet',
                           C=1.0 / max(alpha, 1e-10),
                           l1_ratio=l1_ratio,
                           solver='saga', fit_intercept=True,
                           max_iter=5000, random_state=0),
        'bayesian'   : lambda alpha, l1_ratio: LogisticRegression(
                           penalty='l2', solver='lbfgs',
                           fit_intercept=True, max_iter=5000,
                           random_state=0),  # no Bayesian LR in sklearn
    }

    def __init__(self, task_type: str = 'classification',
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 regularizer: str = 'ridge',
                 lookback: int = 30,
                 **kwargs):
        if regularizer not in self._REGULARIZERS:
            raise ValueError(f"regularizer must be one of "
                             f"{list(self._REGULARIZERS)}; got '{regularizer}'")
        super().__init__(task_type=task_type, alpha=alpha, l1_ratio=l1_ratio,
                         regularizer=regularizer, lookback=lookback, **kwargs)
        self.alpha       = alpha
        self.l1_ratio    = l1_ratio
        self.regularizer = regularizer
        self.lookback    = lookback

        self.classifier     = self._REGULARIZERS[regularizer](alpha, l1_ratio)
        self.feature_scaler = StandardScaler()
        self._cls_threshold = 0.5   # calibrated on val set in fit()

    # ── helpers (shared with ARXModel) ───────────────────────────────────────

    def _flatten(self, X_raw: np.ndarray) -> np.ndarray:
        """Same window-flattening as ARXModel."""
        N, L, F = X_raw.shape
        prices      = X_raw[:, :, 0]
        prices_safe = np.clip(prices, 1e-10, None)
        log_rets    = np.diff(np.log(prices_safe), axis=1)
        first_ret   = log_rets[:, :1]
        log_rets    = np.hstack([first_ret, log_rets])
        exog        = X_raw[:, -1, 1:]
        return np.hstack([log_rets, exog])

    # ── BaseForecastModel interface ───────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        X_train : (N, lookback, n_features_raw)
        y_train : (N,) binary direction labels {0, 1}
        """
        y_train = np.asarray(y_train).reshape(-1).astype(int)

        # Internal 15% val split for threshold calibration
        n_val = max(1, int(len(y_train) * 0.15))
        n_tr  = len(y_train) - n_val

        X_tr_raw, X_val_raw = X_train[:n_tr], X_train[n_tr:]
        y_tr,     y_val     = y_train[:n_tr],  y_train[n_tr:]

        X_tr_flat  = self._flatten(X_tr_raw)
        X_val_flat = self._flatten(X_val_raw)

        X_tr_s  = self.feature_scaler.fit_transform(X_tr_flat)
        X_val_s = self.feature_scaler.transform(X_val_flat)

        self.classifier.fit(X_tr_s, y_tr)

        # Threshold calibration: scan P(Up) percentiles to maximise F1-macro
        val_probas = self.classifier.predict_proba(X_val_s)[:, 1]
        candidates = np.linspace(np.percentile(val_probas, 5),
                                 np.percentile(val_probas, 95), 100)
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            p  = (val_probas >= t).astype(int)
            f1 = f1_score(y_val, p, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self._cls_threshold = best_t

        cls_label = (
            f"LogisticRegression-L2(C={1/max(self.alpha,1e-10):.4g})"
            if self.regularizer == 'ridge' else
            f"LogisticRegression-EN(C={1/max(self.alpha,1e-10):.4g}, "
            f"l1_ratio={self.l1_ratio})"
            if self.regularizer == 'elasticnet' else
            "LogisticRegression-L2(auto)"
        )
        print(f"  [ARX-CLS] Fitted {cls_label}  "
              f"features={X_tr_flat.shape[1]}  "
              f"train={n_tr}  val={n_val}")
        print(f"  [ARX-CLS] Calibrated P(Up) threshold: {best_t:.4f}  "
              f"(val F1-macro: {best_f1:.4f})")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(Up) for each sample, shape (N,)."""
        X_flat = self._flatten(X)
        X_s    = self.feature_scaler.transform(X_flat)
        return self.classifier.predict_proba(X_s)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary direction labels {0,1} using calibrated threshold."""
        return (self.predict_proba(X) >= self._cls_threshold).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        y_test : binary direction labels {0, 1}
        Uses median of predicted P(Up) as eval threshold for balance.
        """
        y_test   = np.asarray(y_test).reshape(-1).astype(int)
        probas   = self.predict_proba(X_test)

        eval_threshold = float(np.median(probas))
        pred_dir = (probas >= eval_threshold).astype(int)
        true_dir = y_test

        n_up, n_down = int(pred_dir.sum()), int((1 - pred_dir).sum())
        print(f"  [ARX-CLS eval] P(Up) threshold={eval_threshold:.4f}  "
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
            'accuracy'           : acc,
            'precision_per_class': prec,
            'recall_per_class'   : rec,
            'f1_per_class'       : f1_per,
            'f1_macro'           : f1_macro,
            'confusion_matrix'   : cm,
            'roc_auc'            : auc,
            'r2'                 : float('nan'),
            'r2_price'           : float('nan'),
            'r2_close'           : float('nan'),
            'rmse'               : float('nan'),
            'mae'                : float('nan'),
            'mape_pct'           : float('nan'),
            'pred_close'         : np.full(len(y_test), float('nan')),
            'true_close'         : np.full(len(y_test), float('nan')),
            'predictions'        : pred_dir,
            'price_predictions'  : probas,   # used for ROC-AUC ranking
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'alpha'        : self.alpha,
                'l1_ratio'     : self.l1_ratio,
                'regularizer'  : self.regularizer,
                'lookback'     : self.lookback,
                'classifier'   : self.classifier,
                'feature_scaler': self.feature_scaler,
                'cls_threshold': self._cls_threshold,
                'task_type'    : self.task_type,
                'hyperparameters': self.hyperparameters,
            }, f)
        print(f"  Model saved → {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.alpha           = d['alpha']
        self.l1_ratio        = d.get('l1_ratio', 0.5)
        self.regularizer     = d.get('regularizer', 'ridge')
        self.lookback        = d['lookback']
        self.classifier      = d['classifier']
        self.feature_scaler  = d['feature_scaler']
        self._cls_threshold  = d.get('cls_threshold', 0.5)
        self.task_type       = d.get('task_type', 'classification')
        self.hyperparameters = d.get('hyperparameters', {})
        print(f"  Model loaded ← {filepath}")

    def get_top_features(self, n: int = 10,
                         tech_cols: list = None,
                         macro_cols: list = None) -> pd.DataFrame:
        """Top-n features by |coefficient| (first class's coef if binary)."""
        coef = self.classifier.coef_[0]  # shape (n_features,) for binary LR
        names = [f"Δ_lag{i+1}" for i in range(self.lookback)]
        if tech_cols:
            names += [f"tech_{c}" for c in tech_cols]
        if macro_cols:
            names += list(macro_cols)
        while len(names) < len(coef):
            names.append(f"feat_{len(names)}")
        names = names[:len(coef)]
        df = pd.DataFrame({'feature': names, 'coefficient': coef})
        df['abs_coef'] = df['coefficient'].abs()
        return df.nlargest(n, 'abs_coef').reset_index(drop=True)


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
#  METRICS REPORTING  (mirrors TCN_Revised.py style)
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
#  VISUALIZATION  (§4.6 — same two panels as TCN_Revised.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(dates_test, y_true_dir, metrics: dict,
                 lookback: int = 30, elapsed: float = 0.0,
                 filename: str = 'ARX/arx_results.png'):
    preds        = metrics['predictions']
    cm           = metrics['confusion_matrix']
    y_true_close = metrics['true_close']
    y_pred_close = metrics['pred_close']

    dates_dt = pd.to_datetime(dates_test)

    # Rolling 30-day accuracy
    correct  = (preds == y_true_dir).astype(float)
    roll_acc = pd.Series(correct).rolling(lookback, min_periods=1).mean().values

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[1.4, 1.0, 1.0],
                            hspace=0.45, wspace=0.28)

    # ── Panel 1 (top, full width): Price Forecast ─────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor('#1a1d27')
    ax0.plot(dates_dt, y_true_close, color='#ffffff', linewidth=1.4,
             alpha=0.9, label='Actual Close (USD)')
    ax0.plot(dates_dt, y_pred_close, color='#ff7043', linewidth=1.2,
             alpha=0.85, linestyle='--', label='ARX Predicted Close (USD)')
    ax0.fill_between(dates_dt, y_true_close, y_pred_close,
                     alpha=0.08, color='#ff7043')
    ax0.set_xlabel('Date', color='#b0bec5', fontsize=10)
    ax0.set_ylabel('Price (USD/bushel)', color='#b0bec5', fontsize=10)
    ax0.set_title(
        f"ARX Price Forecast  —  "
        f"RMSE={metrics['rmse']:.2f} USD  |  MAPE={metrics['mape_pct']:.2f}%  |  "
        f"R²(price)={metrics['r2_close']:.4f} [autocorr.]  |  R²(Δ)={metrics['r2_price']:.4f}",
        color='white', fontsize=11, pad=10)
    ax0.tick_params(colors='#b0bec5', labelsize=9)
    ax0.spines[:].set_color('#37474f')
    ax0.legend(fontsize=9, facecolor='#1a1d27', labelcolor='white', loc='upper left')
    ax0.grid(True, linestyle=':', alpha=0.3, color='#37474f')

    # ── Panel 2 (middle, full width): 10-day aggregated stacked bars ────────
    ax1 = fig.add_subplot(gs[1, :])
    ax1.set_facecolor('#1a1d27')

    true_dir = y_true_dir.astype(int)
    pred_dir = preds.astype(int)
    tp_arr = (pred_dir == 1) & (true_dir == 1)   # correct Up
    tn_arr = (pred_dir == 0) & (true_dir == 0)   # correct Down
    fp_arr = (pred_dir == 1) & (true_dir == 0)   # wrong Up
    fn_arr = (pred_dir == 0) & (true_dir == 1)   # wrong Down

    # Aggregate into 10-day windows
    window = 10
    n = len(true_dir)
    bin_dates, tp_bins, tn_bins, fp_bins, fn_bins = [], [], [], [], []
    for i in range(0, n, window):
        sl = slice(i, min(i + window, n))
        bin_dates.append(dates_dt[i + (min(i + window, n) - i) // 2])  # midpoint date
        tp_bins.append(tp_arr[sl].sum())
        tn_bins.append(tn_arr[sl].sum())
        fp_bins.append(fp_arr[sl].sum())
        fn_bins.append(fn_arr[sl].sum())

    bin_dates = pd.DatetimeIndex(bin_dates)
    tp_bins = np.array(tp_bins, dtype=float)
    tn_bins = np.array(tn_bins, dtype=float)
    fp_bins = np.array(fp_bins, dtype=float)
    fn_bins = np.array(fn_bins, dtype=float)
    bar_w   = pd.Timedelta(days=7)  # bar width ~1 week

    # ABOVE zero: Up predictions  (TP=green on bottom, FP=red on top)
    ax1.bar(bin_dates,  tp_bins, width=bar_w, color='#66bb6a', alpha=0.85,
            label=f'Correct ↑ Up (TP={int(tp_bins.sum())})')
    ax1.bar(bin_dates,  fp_bins, width=bar_w, color='#ef5350', alpha=0.85,
            bottom=tp_bins, label=f'Wrong ↑ Up (FP={int(fp_bins.sum())})')

    # BELOW zero: Down predictions (TN=green on top/closer to 0, FN=red below)
    ax1.bar(bin_dates, -tn_bins, width=bar_w, color='#29b6f6', alpha=0.85,
            label=f'Correct ↓ Down (TN={int(tn_bins.sum())})')
    ax1.bar(bin_dates, -fn_bins, width=bar_w, color='#ff7043', alpha=0.85,
            bottom=-tn_bins, label=f'Wrong ↓ Down (FN={int(fn_bins.sum())})')

    # Overlay normalised price
    price_norm = (y_true_close - y_true_close.mean()) / (y_true_close.std() + 1e-10)
    ax1_r = ax1.twinx()
    ax1_r.plot(dates_dt, price_norm * 3, color='#ffffff', linewidth=1.0,
               alpha=0.4, linestyle='-')
    ax1_r.set_yticks([])
    ax1_r.spines[:].set_color('#37474f')

    ax1.axhline(0, color='#90a4ae', linewidth=1.0, linestyle='-')
    ax1.set_xlabel('Date', color='#b0bec5', fontsize=10)
    ax1.set_ylabel('Count per 10-day window', color='#b0bec5', fontsize=10)
    ax1.set_title(
        f"Up/Down Direction — 10-Day Aggregated  |  "
        f"TP={int(tp_arr.sum())} TN={int(tn_arr.sum())} FP={int(fp_arr.sum())} FN={int(fn_arr.sum())}  |  "
        f"Acc={metrics['accuracy']:.4f}  BAR={(tp_arr.sum()/true_dir.sum() + tn_arr.sum()/(1-true_dir).sum())/2:.4f}",
        color='white', fontsize=11, pad=10)
    ax1.tick_params(colors='#b0bec5', labelsize=9)
    ax1.spines[:].set_color('#37474f')

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor='#66bb6a', label=f'Correct ↑ (TP={int(tp_arr.sum())})'),
                  Patch(facecolor='#ef5350', label=f'Wrong ↑ (FP={int(fp_arr.sum())})'),
                  Patch(facecolor='#29b6f6', label=f'Correct ↓ (TN={int(tn_arr.sum())})'),
                  Patch(facecolor='#ff7043', label=f'Wrong ↓ (FN={int(fn_arr.sum())})'),
                  plt.Line2D([0], [0], color='#b0bec5', linewidth=0.9,
                             label='Actual price (norm.)')]
    ax1.legend(handles=legend_els, fontsize=8, facecolor='#1a1d27',
               labelcolor='white', loc='upper left')

    # ── Panel 3 (bottom-left): Rolling accuracy ───────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_facecolor('#1a1d27')
    ax2.plot(dates_dt, roll_acc, color='#4fc3f7', linewidth=1.5, alpha=0.9,
             label=f'Rolling {lookback}-day accuracy')
    ax2.axhline(0.5, color='#ef5350', linewidth=1.0, linestyle='--',
                label='Random baseline (0.50)')
    ax2.axhline(metrics['accuracy'], color='#66bb6a', linewidth=1.2,
                linestyle=':', label=f"Overall: {metrics['accuracy']:.4f}")
    ax2.set_ylim(0.2, 0.8)
    ax2.set_xlabel('Date', color='#b0bec5', fontsize=10)
    ax2.set_ylabel('Accuracy', color='#b0bec5', fontsize=10)
    ax2.set_title(
        f"Rolling {lookback}-Day Accuracy  —  "
        f"ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1_macro']:.4f}  ({elapsed:.1f}s)",
        color='white', fontsize=11, pad=8)
    ax2.tick_params(colors='#b0bec5', labelsize=9)
    ax2.spines[:].set_color('#37474f')
    ax2.legend(fontsize=8, facecolor='#1a1d27', labelcolor='white')
    ax2.fill_between(dates_dt, roll_acc, 0.5,
                     where=(roll_acc >= 0.5), alpha=0.15,
                     color='#4fc3f7', interpolate=True)

    # ── Panel 4 (bottom-right): Confusion matrix ──────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.set_facecolor('#1a1d27')
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums > 0, cm / row_sums * 100, 0)
    annot    = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
                          for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'],
                ax=ax3, linewidths=0.5,
                cbar_kws={'shrink': 0.7})
    ax3.set_title('Confusion Matrix', color='white', fontsize=12, pad=10)
    ax3.set_xlabel('Predicted', color='#b0bec5', fontsize=10)
    ax3.set_ylabel('Actual',    color='#b0bec5', fontsize=10)
    ax3.tick_params(colors='#b0bec5', labelsize=9)

    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_arx(lookback: int = 30,
                  alpha: float = 1.0,
                  l1_ratio: float = 0.5,
                  regularizer: str = 'ridge',
                  mode: str = 'classification',
                  fred_folder_train: str = None,
                  fred_folder_val: str = None):
    """
    End-to-end ARX pipeline.

    mode : 'regression' | 'classification' | 'both'
      regression     — ARXModel (Ridge/ElasticNet/BayesianRidge on Δ)
      classification — ARXClassifier (LogisticRegression on direction {0,1})
      both           — run and compare both models

    regularizer : 'ridge' | 'elasticnet' | 'bayesian'
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
    reg_summary = (
        f"Ridge(alpha={alpha})"
        if regularizer == 'ridge' else
        f"ElasticNet(alpha={alpha}, l1_ratio={l1_ratio})"
        if regularizer == 'elasticnet' else
        "BayesianRidge(auto-alpha)"
    )
    print(f"ARX Daily Pipeline — mode={mode}  {reg_summary}  lookback={lookback}")
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
          f"(lookback={lookback}, "
          f"features=1+{len(tech_cols)}+{len(macro_cols)})...")
    X, y_price, y_dir, y_close, y_prev, dates = build_sliding_windows(
        df_merged, macro_cols, tech_cols=tech_cols, lookback=lookback)
    print(f"  Samples X: {X.shape}")
    print(f"  Feature vector per sample: "
          f"{lookback} price lags + {len(tech_cols)} tech + "
          f"{len(macro_cols)} macro = "
          f"{lookback + len(tech_cols) + len(macro_cols)} dims")
    print(f"  y_price range: {y_price.min():.2f} – {y_price.max():.2f}  "
          f"(USD/day, price change Δ)")
    print(f"  y_dir  split  : Up={y_dir.sum()}  Down={(1-y_dir).sum()}")
    print(f"  Date range: {pd.to_datetime(dates[0]).date()} → "
          f"{pd.to_datetime(dates[-1]).date()}")

    # ── Step 5: Train/Test split + CV ────────────────────────────────────
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

    # Per-task seeds for reproducibility (same as TCN fix)
    TASK_SEEDS = {'regression': 42, 'classification': 43}

    def _make_model(task):
        kw = dict(alpha=alpha, l1_ratio=l1_ratio,
                  regularizer=regularizer, lookback=lookback)
        return ARXModel(**kw) if task == 'regression' else ARXClassifier(**kw)

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
        r2s  = fold_stats[task]['r2s']
        tag  = 'REG' if task == 'regression' else 'CLS'
        print(f"\n  CV [{tag}] Accuracy: {np.mean(accs):.4f} ±{np.std(accs):.4f}")
        if task == 'regression':
            print(f"  CV [{tag}] R²(Δ)  : {np.mean(r2s):.4f} ±{np.std(r2s):.4f}")

    # ── Final models on full train+val ────────────────────────────────────
    print("\n[Holdout] Training final ARX model(s) on full train+val...")
    os.makedirs('models', exist_ok=True)
    plot_results_data = []   # (label, metrics)

    if 'regression' in tasks:
        np.random.seed(TASK_SEEDS['regression'])
        print("\n  ── ARXModel (regression) ──")
        final_reg = _make_model('regression')
        final_reg.fit(X_tv, yp_tv)
        m_reg = final_reg.evaluate(X_test, yp_test)
        print_metrics("ARX Regressor — Holdout Test", m_reg)
        final_reg.save('models/arx_regressor.pkl')
        top = final_reg.get_top_features(n=10, tech_cols=tech_cols,
                                         macro_cols=macro_cols)
        print("\n  Top-10 most influential features (|coefficient|):")
        print(f"  {'Feature':<20} {'Coefficient':>14} {'|Coef|':>10}")
        print(f"  {'-'*46}")
        for _, row in top.iterrows():
            print(f"  {row['feature']:<20} {row['coefficient']:>14.6f} "
                  f"{row['abs_coef']:>10.6f}")
        plot_results_data.append(('ARX Regressor', m_reg))

    if 'classification' in tasks:
        np.random.seed(TASK_SEEDS['classification'])
        print("\n  ── ARXClassifier (classification) ──")
        final_cls = _make_model('classification')
        final_cls.fit(X_tv, yd_tv)
        m_cls = final_cls.evaluate(X_test, yd_test)
        print_metrics("ARX Classifier — Holdout Test", m_cls)
        final_cls.save('models/arx_classifier.pkl')
        top = final_cls.get_top_features(n=10, tech_cols=tech_cols,
                                         macro_cols=macro_cols)
        print("\n  Top-10 most influential features (|coefficient|):")
        print(f"  {'Feature':<20} {'Coefficient':>14} {'|Coef|':>10}")
        print(f"  {'-'*46}")
        for _, row in top.iterrows():
            print(f"  {row['feature']:<20} {row['coefficient']:>14.6f} "
                  f"{row['abs_coef']:>10.6f}")
        plot_results_data.append(('ARX Classifier', m_cls))

    elapsed = time.time() - start
    print(f"\nTotal processing time: {elapsed:.2f}s")

    # ── Visualize — use first available metrics for the plot ───────────────
    os.makedirs('ARX', exist_ok=True)
    primary_label, primary_m = plot_results_data[0]
    plot_results(dates_test, yd_test, primary_m,
                 lookback=lookback, elapsed=elapsed,
                 filename=f'ARX/arx_{mode}_results.png')


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ARX Daily Pipeline — regularised ARX for wheat futures')
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['regression', 'classification', 'both'],
                        help='regression | classification | both (default: classification)')
    parser.add_argument('--regularizer', type=str, default='ridge',
                        choices=['ridge', 'elasticnet', 'bayesian'],
                        help='Regularisation method: ridge | elasticnet | bayesian '
                             '(default: ridge)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Regularisation strength for Ridge / ElasticNet '
                             '(default: 1.0; ignored for bayesian)')
    parser.add_argument('--l1_ratio', type=float, default=0.5,
                        help='ElasticNet / ElasticNet-LR mixing ratio: 0=Ridge, 1=Lasso '
                             '(default: 0.5; used when --regularizer elasticnet)')
    parser.add_argument('--fred_train', type=str, default=None,
                        help='Path to historical FRED-MD vintage folder (train)')
    parser.add_argument('--fred_val', type=str, default=None,
                        help='Path to FRED-MD vintage folder (val/recent)')
    args = parser.parse_args()

    run_daily_arx(
        lookback=30,            # §1.3 — fixed by compliance
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        regularizer=args.regularizer,
        mode=args.mode,
        fred_folder_train=args.fred_train,
        fred_folder_val=args.fred_val,
    )
