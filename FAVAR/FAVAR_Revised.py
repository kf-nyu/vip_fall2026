"""
FAVAR Revised — VIP Resubmission Compliant
===========================================
Factor-Augmented Vector Autoregression (FAVAR) for daily wheat futures
price direction prediction.

Strategy (same as ARX_Revised.py / TCN_Revised.py):
  1.  Predict next-day price CHANGE  Δ[t] = Close[t] - Close[t-1]
  2.  Derive binary direction:  predicted Δ > threshold  →  Up (1)
                                predicted Δ ≤ threshold  →  Down (0)

Models  (--mode regression | classification | both):

  TrueFAVARRegressor  [--mode regression]:
    Z[t] = [Δ[t], F₁[t], ..., Fₖ[t]]ᵀ
    Z[t] = A₀ + A₁·Z[t-1] + ... + Aₗ·Z[t-l] + ε[t]

    k PCA macro factors F₁…Fₖ extracted from 31 FRED-MD variables
    (standardised, fitted on training data only, random_state=0).
    Full VAR(l) system estimated by OLS (statsmodels); the Δ equation
    is read off for 1-step-ahead direction forecasts.
    Pure closed-form OLS — no regulariser, no gradient descent.

  TrueFAVARClassifier  [--mode classification]:
    Logistic Regression on  [30 price lags | k PCA macro factors]
    directly predicts Up/Down — no intermediate price forecast.

    Regulariser options (--regularizer flag):
      ridge      — L2 penalty; C = 1/alpha, solver=lbfgs.
                   Principled baseline for high-dimensional logistic regression.
      elasticnet — L1 + L2 penalty; sparse lags + PCA factors, solver=saga.
                   Controlled by --alpha and --l1_ratio (default 0.5).
      bayesian   — L2 LR (lbfgs); alpha auto-tuned — no --alpha tuning needed.
    P(Up) threshold calibrated on a held-out val slice to maximise F1-macro.



Compliance checklist:
  §1.1  Exactly 31 FRED-MD variables (SELECTED_FEATURES)
  §1.2  T-code stationarity transforms + 1-month reporting delay shift
        Monthly macro forward-filled to daily frequency
  §1.3  Lookback window of 30 trading days; each sample uses
        [30 price lags | k PCA macro factors] — no contemporaneous info

  §2    5-fold TimeSeriesSplit CV; PCA + StandardScaler fitted on train fold only
        Per-task seeds: regression=42, classification=43
  §4.5  Classification metrics on derived direction labels:
        Accuracy, Precision, Recall, F1 (per class + macro), Confusion Matrix,
        ROC-AUC (score = predicted Δ / P(Up), continuous)
        Regression metrics (regression mode only): RMSE, R² (on price level)
  §4.6  Two visualizations per mode:
        Panel 1 — Rolling 30-day accuracy over time (clean line chart)
        Panel 2 — Confusion matrix with count + row-% annotation
"""


import os
import pickle
import time
import argparse
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, r2_score, mean_squared_error
)
from statsmodels.tsa.api import VAR

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# §1.1  EXACTLY 31 FRED-MD VARIABLES (per resubmission guidelines)
# ─────────────────────────────────────────────────────────────────────────────
SELECTED_FEATURES = [
    "RPI", "W875RX1", "CMRMTSPLx", "IPFPNSS", "USWTRADE", "USTRADE",
    "BUSLOANS", "CONSPI", "S&P 500", "S&P PE ratio", "FEDFUNDS",
    "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA",
    "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM",
    "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx",
    "EXCAUSx", "PPICMM", "UMCSENTx",
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
# BASE CLASS
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
# MODEL 1: FACTOR-AUGMENTED ARX REGRESSOR (FAVARRegressor)
# ─────────────────────────────────────────────────────────────────────────────

class TrueFAVARRegressor(BaseForecastModel):
    """
    True FAVAR using a Vector Autoregression on Z_t = [Close_t, F1_t, ..., Fk_t],
    where Close_t is the continuous closing price and F are PCA macro factors.

    Running VAR on continuous prices is far more appropriate than running it on
    binary {0,1} labels. Direction is derived after forecasting:
      predicted_price_t > Close_{t-1}  →  Up (1), else Down (0).

    Rolling 1-step-ahead evaluation:
      At step i, the VAR forecasts using Z[idx-var_lags : idx] as context, where
      idx = len(Z_train) + i. The actual y_test[i] is appended to context only
      AFTER step i is forecasted — no look-ahead bias.
    """

    def __init__(self, task_type='regression', n_components=3,
                 var_lags=2, n_price_lags=30, **kwargs):
        super().__init__(task_type=task_type, n_components=n_components,
                         var_lags=var_lags, n_price_lags=n_price_lags, **kwargs)
        self.n_components = n_components
        self.var_lags     = var_lags
        self.n_price_lags = n_price_lags

        # Fitted on training data only
        self.scaler    = StandardScaler()
        self.pca       = PCA(n_components=self.n_components, random_state=0)
        self.var_model = None
        self.var_res   = None
        self.Z_train   = None

    def fit(self, X_train, y_train):
        """y_train = continuous closing prices."""
        X_macro   = X_train[:, self.n_price_lags:]
        X_macro_s = self.scaler.fit_transform(X_macro)
        F_train   = self.pca.fit_transform(X_macro_s)

        # Z_t = [Close_t, F1_t, ..., Fk_t]  — all continuous
        self.Z_train = np.column_stack([y_train, F_train])
        col_names    = ["Close"] + [f"F{i+1}" for i in range(self.n_components)]
        df_Z         = pd.DataFrame(self.Z_train, columns=col_names)

        self.var_model = VAR(df_Z)
        self.var_res   = self.var_model.fit(maxlags=self.var_lags, ic=None)

    def _predict_prices(self, X, y_test=None):
        """
        Rolling 1-step-ahead forecast returning predicted closing prices.
        y_test (continuous prices) provides the actual context for each step.
        """
        if self.var_res is None:
            raise ValueError("Model not trained! Call fit() first.")

        X_macro   = X[:, self.n_price_lags:]
        X_macro_s = self.scaler.transform(X_macro)
        F_test    = self.pca.transform(X_macro_s)

        if y_test is not None:
            Z_test = np.column_stack([y_test, F_test])
            Z_full = np.vstack([self.Z_train, Z_test])
            price_preds = []
            for i in range(len(y_test)):
                idx       = len(self.Z_train) + i
                Z_context = Z_full[idx - self.var_lags: idx]
                p         = self.var_res.forecast(y=Z_context, steps=1)
                price_preds.append(p[0, 0])   # Close component of Z
            return np.array(price_preds)
        else:
            steps = len(X)
            p     = self.var_res.forecast(y=self.Z_train[-self.var_lags:], steps=steps)
            return p[:, 0]

    def predict(self, X, y_test=None):
        """Return predicted next-day closing prices (continuous)."""
        return self._predict_prices(X, y_test=y_test)

    def evaluate(self, X_test, y_test):
        """
        y_test = actual next-day closing prices.
        Lag1 (Close[t-1]) is recovered from the Z_train/Z_test rolling context:
        the last known price before step i is Z_full[idx-1, 0].
        """
        X_macro   = X_test[:, self.n_price_lags:]
        X_macro_s = self.scaler.transform(X_macro)
        F_test    = self.pca.transform(X_macro_s)

        Z_test = np.column_stack([y_test, F_test])
        Z_full = np.vstack([self.Z_train, Z_test])

        price_preds = []
        lag1_prices = []    # last known price before each forecast step
        for i in range(len(y_test)):
            idx       = len(self.Z_train) + i
            Z_context = Z_full[idx - self.var_lags: idx]
            p         = self.var_res.forecast(y=Z_context, steps=1)
            price_preds.append(p[0, 0])
            lag1_prices.append(Z_full[idx - 1, 0])   # Close component of Z at t-1

        price_preds = np.array(price_preds)
        lag1_prices = np.array(lag1_prices)

        # Derive directions from price comparison
        pred_dir = (price_preds > lag1_prices).astype(int)
        true_dir = (y_test      > lag1_prices).astype(int)

        # ── Regression metrics ─────────────────────────────────────────────
        r2_price = r2_score(y_test, price_preds)
        rmse     = np.sqrt(mean_squared_error(y_test, price_preds))
        r2       = r2_score(true_dir, pred_dir)   # direction-level R²

        # ── Classification metrics on derived directions ────────────────────
        acc      = accuracy_score(true_dir, pred_dir)
        prec     = precision_score(true_dir, pred_dir, average=None, zero_division=0)
        rec      = recall_score(true_dir, pred_dir, average=None, zero_division=0)
        f1       = f1_score(true_dir, pred_dir, average=None, zero_division=0)
        f1_macro = f1_score(true_dir, pred_dir, average='macro', zero_division=0)
        cm       = confusion_matrix(true_dir, pred_dir)

        # ROC-AUC using (predicted_price - lag1) as ranking score
        score = price_preds - lag1_prices
        try:
            auc = roc_auc_score(true_dir, score)
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
            'r2'                 : r2,
            'r2_price'           : r2_price,
            'rmse'               : rmse,
            'predictions'        : pred_dir,
            'price_predictions'  : price_preds,
        }

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_components' : self.n_components,
                'var_lags'     : self.var_lags,
                'n_price_lags' : self.n_price_lags,
                'scaler'       : self.scaler,
                'pca'          : self.pca,
            }, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.n_components = d['n_components']
        self.var_lags     = d['var_lags']
        self.n_price_lags = d.get('n_price_lags', 30)
        self.scaler       = d['scaler']
        self.pca          = d['pca']


# ─────────────────────────────────────────────────────────────────────────────
# TRUE FAVAR CLASSIFIER  (direct Up/Down via Logistic Regression on [lags + PCA])
# ─────────────────────────────────────────────────────────────────────────────

class TrueFAVARClassifier(BaseForecastModel):
    """
    True FAVAR Classifier: Logistic Regression on [price lags + k PCA macro factors].

    Directly predicts binary direction {0, 1} — no intermediate price forecast.
    Same feature engineering as TrueFAVARRegressor (PCA on FRED-MD, fitted on train).
    Threshold calibrated on val set to maximise F1-macro.

    Regularisation options mirror ARXClassifier convention:
      --regularizer ridge      → L2 LR (solver=lbfgs,  C=1/alpha)
      --regularizer elasticnet → EN LR (solver=saga,   C=1/alpha, l1_ratio)
      --regularizer bayesian   → L2 LR (same as ridge; no Bayesian LR in sklearn)
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
                 n_components: int = 3,
                 n_price_lags: int = 30,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 regularizer: str = 'ridge',
                 **kwargs):
        if regularizer not in self._REGULARIZERS:
            raise ValueError(f"regularizer must be one of "
                             f"{list(self._REGULARIZERS)}; got '{regularizer}'")
        super().__init__(task_type=task_type, n_components=n_components,
                         n_price_lags=n_price_lags, **kwargs)
        self.n_components = n_components
        self.n_price_lags = n_price_lags
        self.alpha        = alpha
        self.l1_ratio     = l1_ratio
        self.regularizer  = regularizer

        self.scaler         = StandardScaler()
        self.pca            = PCA(n_components=self.n_components, random_state=0)
        self.classifier     = self._REGULARIZERS[regularizer](alpha, l1_ratio)
        self._cls_threshold = 0.5   # calibrated on val set in fit()


    def fit(self, X_train, y_train):
        """y_train = binary direction labels {0, 1}."""
        y_train = np.asarray(y_train).reshape(-1).astype(int)

        # Internal 15% val split for threshold calibration
        n_val = max(1, int(len(y_train) * 0.15))
        n_tr  = len(y_train) - n_val

        X_tr_raw,  X_val_raw = X_train[:n_tr],  X_train[n_tr:]
        y_tr,      y_val     = y_train[:n_tr],   y_train[n_tr:]

        X_tr_prices  = X_tr_raw[:,  :self.n_price_lags]
        X_tr_macro   = X_tr_raw[:,  self.n_price_lags:]
        X_val_prices = X_val_raw[:, :self.n_price_lags]
        X_val_macro  = X_val_raw[:, self.n_price_lags:]

        # PCA + scaler fitted on training block only
        X_tr_macro_s  = self.scaler.fit_transform(X_tr_macro)
        F_tr          = self.pca.fit_transform(X_tr_macro_s)
        X_val_macro_s = self.scaler.transform(X_val_macro)
        F_val         = self.pca.transform(X_val_macro_s)

        X_tr_s  = np.hstack([X_tr_prices,  F_tr])
        X_val_s = np.hstack([X_val_prices, F_val])

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

        n_feat = X_tr_s.shape[1]
        print(f"  [FAVAR-CLS] LR-{self.regularizer}(C={1/max(self.alpha,1e-10):.4g})  "
              f"features={n_feat}  train={n_tr}  val={n_val}")
        print(f"  [FAVAR-CLS] Calibrated P(Up) threshold: {best_t:.4f}  "
              f"(val F1-macro: {best_f1:.4f})")

    def predict_proba(self, X):
        X_prices  = X[:, :self.n_price_lags]
        X_macro_s = self.scaler.transform(X[:, self.n_price_lags:])
        F         = self.pca.transform(X_macro_s)
        return self.classifier.predict_proba(np.hstack([X_prices, F]))[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= self._cls_threshold).astype(int)

    def evaluate(self, X_test, y_test):
        """y_test = binary direction labels {0, 1}."""
        y_test = np.asarray(y_test).reshape(-1).astype(int)
        probas = self.predict_proba(X_test)

        eval_threshold = float(np.median(probas))
        pred_dir = (probas >= eval_threshold).astype(int)

        n_up, n_down = int(pred_dir.sum()), int((1 - pred_dir).sum())
        print(f"  [FAVAR-CLS eval] P(Up) threshold={eval_threshold:.4f}  "
              f"(pred Up={n_up}  Down={n_down})")

        acc      = accuracy_score(y_test, pred_dir)
        prec     = precision_score(y_test, pred_dir, average=None, zero_division=0)
        rec      = recall_score(y_test, pred_dir, average=None, zero_division=0)
        f1_per   = f1_score(y_test, pred_dir, average=None, zero_division=0)
        f1_macro = f1_score(y_test, pred_dir, average='macro', zero_division=0)
        cm       = confusion_matrix(y_test, pred_dir)
        try:
            auc = roc_auc_score(y_test, probas)
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
            'rmse'               : float('nan'),
            'predictions'        : pred_dir,
            'price_predictions'  : probas,
            'true_dir'           : y_test,
        }

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_components' : self.n_components,
                'n_price_lags' : self.n_price_lags,
                'alpha'        : self.alpha,
                'l1_ratio'     : self.l1_ratio,
                'regularizer'  : self.regularizer,
                'scaler'       : self.scaler,
                'pca'          : self.pca,
                'classifier'   : self.classifier,
                'cls_threshold': self._cls_threshold,
                'task_type'    : self.task_type,
            }, f)
        print(f"  Model saved → {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.n_components   = d['n_components']
        self.n_price_lags   = d.get('n_price_lags', 30)
        self.alpha          = d.get('alpha', 1.0)
        self.l1_ratio       = d.get('l1_ratio', 0.5)
        self.regularizer    = d.get('regularizer', 'ridge')
        self.scaler         = d['scaler']
        self.pca            = d['pca']
        self.classifier     = d['classifier']
        self._cls_threshold = d.get('cls_threshold', 0.5)
        self.task_type      = d.get('task_type', 'classification')
        print(f"  Model loaded ← {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
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


def build_feature_matrix(df_merged: pd.DataFrame,
                         macro_cols: list,
                         n_price_lags: int = 30) -> tuple:
    """§1.3  Build feature matrix with continuous price target."""
    macro_lag1_cols = [f"{c}_lag1" for c in macro_cols]
    df_merged[macro_lag1_cols] = df_merged[macro_cols].shift(1)

    price_lag_cols = []
    for k in range(1, n_price_lags + 1):
        col = f"Lag{k}"
        df_merged[col] = df_merged['Close'].shift(k)
        price_lag_cols.append(col)

    df_merged = df_merged.dropna().reset_index(drop=True)

    X_prices = df_merged[price_lag_cols].values
    X_macro  = df_merged[macro_lag1_cols].values
    X        = np.hstack([X_prices, X_macro])
    y        = df_merged['Close'].values
    dates    = df_merged['date'].values

    return X, y, dates, df_merged


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION REPORTING
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
# VISUALIZATION  (§4.6)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(dates_test, y_true_dir, var_metrics,
                 n_lags, n_factors, elapsed, output_dir='FAVAR',
                 title_tag='True FAVAR'):
    """
    Generate visualizations (§4.6):
    Panel 1 (top, full-width): Rolling 30-day accuracy over time.
    Panel 2 (bottom-centre): Confusion matrix with count + row-% per cell.
    """
    os.makedirs(output_dir, exist_ok=True)

    var_preds = var_metrics['predictions']
    var_acc   = var_metrics['accuracy']
    var_cm    = var_metrics['confusion_matrix']
    dates_plot = pd.to_datetime(dates_test)

    window   = 30
    var_roll = pd.Series((var_preds == y_true_dir).astype(float),
                         index=dates_plot).rolling(window, min_periods=1).mean()

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[1.5, 1], hspace=0.5)

    # ── Panel 1: Rolling accuracy ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    baseline = 0.5
    ax1.axhline(baseline, color='#888888', linewidth=1.4,
                linestyle=':', label='Random baseline (50%)', zorder=1)
    ax1.fill_between(var_roll.index, baseline, var_roll.values,
                     where=(var_roll.values >= baseline),
                     color='darkorange', alpha=0.15, zorder=1)
    ax1.fill_between(var_roll.index, baseline, var_roll.values,
                     where=(var_roll.values <  baseline),
                     color='tomato',     alpha=0.13, zorder=1)
    ax1.plot(var_roll.index, var_roll.values, color='darkorange', linewidth=2.2,
             label=f'{title_tag} — {window}-day rolling acc  (overall {var_acc:.1%})',
             zorder=3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax1.set_ylim(0.15, 0.85)
    ax1.set_xlim(dates_plot[window - 1], dates_plot[-1])
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel(f'{window}-Day Rolling Accuracy', fontsize=11)
    ax1.set_title(
        f'{title_tag} — Rolling {window}-Day Accuracy (Holdout Test)\n'
        f'Lags: {n_lags}  |  Factors: {n_factors}  |  '
        f'Acc: {var_acc:.2%}  |  Time: {elapsed:.1f}s',
        fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle='--')

    # ── Panel 2: Confusion matrix ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    row_sums = var_cm.sum(axis=1, keepdims=True).clip(min=1)
    pct      = var_cm / row_sums * 100
    sns.heatmap(
        var_cm, annot=False, cmap='Oranges', ax=ax2,
        vmin=0, vmax=var_cm.max() * 1.15,
        xticklabels=['Pred Down (0)', 'Pred Up (1)'],
        yticklabels=['True Down (0)', 'True Up (1)'],
        linewidths=2, linecolor='white', cbar=False)
    for i in range(2):
        for j in range(2):
            bg      = var_cm[i, j] / var_cm.max() if var_cm.max() > 0 else 0
            txt_col = 'white' if bg > 0.55 else '#222222'
            ax2.text(j + 0.5, i + 0.40, f'{var_cm[i, j]}',
                     ha='center', va='center',
                     fontsize=16, fontweight='bold', color=txt_col)
            ax2.text(j + 0.5, i + 0.68, f'({pct[i, j]:.1f}% of row)',
                     ha='center', va='center', fontsize=9, color=txt_col)
    ax2.set_title(f'{title_tag} — Confusion Matrix  (Acc: {var_acc:.2%})',
                  fontsize=11, fontweight='bold', pad=8)
    ax2.set_ylabel('True Label', fontsize=10, labelpad=6)
    ax2.set_xlabel('Predicted Label', fontsize=10, labelpad=6)

    filename = os.path.join(output_dir,
                            f'favar_{title_tag.lower().replace(" ","_")}_l{n_lags}_f{n_factors}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_favar(n_lags: int = 30, n_factors: int = 1,
                    var_lags: int = 3,
                    mode: str = 'classification',
                    regularizer: str = 'ridge',
                    alpha: float = 1.0,
                    l1_ratio: float = 0.5,
                    fred_folder_train: str = None,
                    fred_folder_val: str = None):
    """
    End-to-end True FAVAR pipeline.

    mode : 'regression' | 'classification' | 'both'
      regression     — TrueFAVARRegressor:  VAR on [price, factors] → Δ → direction
      classification — TrueFAVARClassifier: LogisticRegression on [lags + PCA factors]
      both           — run and compare both

    regularizer / alpha / l1_ratio only apply to classification mode.
    """
    assert mode in ('regression', 'classification', 'both'), \
        "mode must be 'regression', 'classification', or 'both'"
    start_time = time.time()

    # ── Resolve data paths ─────────────────────────────────────────────────
    base = Path(__file__).parent.parent  # repo root = vip/
    if fred_folder_train is None:
        fred_folder_train = str(base / "data" / "fred-md" / "Historical FRED-MD Vintages Final")
    if fred_folder_val is None:
        fred_folder_val = str(base / "data" / "fred-md" / "Historical-vintages-of-FRED-MD-2015-01-to-2024-12")
    wheat_path = str(base / "data" / "wheat-futures" / "wheat_futures_daily.csv")

    print("="*60)
    print(f"True FAVAR  —  mode={mode}  VAR(lags={var_lags})  "
          f"n_factors={n_factors}  n_lags={n_lags}")
    print("="*60)

    # ── Step 1: Load and transform FRED-MD macro data ──────────────────────
    print("\n[1/5] Loading FRED-MD macro data and applying t-codes...")
    df_train_raw = load_fred_md(fred_folder_train)
    df_val_raw = load_fred_md(fred_folder_val)

    # Combine both vintage folders into one chronological monthly macro series
    df_macro_raw = pd.concat([df_train_raw, df_val_raw])
    df_macro_raw = df_macro_raw.sort_index()
    df_macro_raw = df_macro_raw[~df_macro_raw.index.duplicated(keep='last')]

    # Report which of the 31 variables were found
    missing_vars = [v for v in SELECTED_FEATURES if v not in df_macro_raw.columns]
    if missing_vars:
        print(f"  WARNING: missing variables (will be zero-filled): {missing_vars}")
    print(f"  Macro data loaded: {df_macro_raw.shape[0]} monthly obs, "
          f"{df_macro_raw.shape[1]} variables")

    # Apply t-code stationarity transforms + one-month reporting delay shift
    df_macro = apply_reporting_delay_and_tcodes(df_macro_raw)
    print(f"  After t-code transforms + 1-month delay shift: {df_macro.shape}")

    # ── Step 2: Load daily wheat futures ───────────────────────────────────
    print("\n[2/5] Loading daily wheat futures prices...")
    df_wheat = pd.read_csv(wheat_path)
    df_wheat['date'] = pd.to_datetime(df_wheat['date'], utc=True).dt.tz_localize(None)
    df_wheat = df_wheat.sort_values('date').reset_index(drop=True)
    print(f"  Wheat data: {len(df_wheat)} daily observations")

    # ── Step 3: Forward-fill macro to daily frequency ──────────────────────
    print("\n[3/5] Forward-filling monthly macro onto daily price data...")
    macro_cols = [c for c in SELECTED_FEATURES if c in df_macro.columns]
    df_merged = forward_fill_macro_to_daily(df_macro, df_wheat)
    print(f"  Merged dataset: {len(df_merged)} rows, {len(macro_cols)} macro cols")

    # ── Step 4 & 5: Build features and binary target ───────────────────────
    print(f"\n[4/5] Building feature matrix ({n_lags} price lags + {len(macro_cols)} macro cols)...")
    X, y, dates, df_final = build_feature_matrix(df_merged, macro_cols, n_price_lags=n_lags)
    print(f"  Feature matrix X: {X.shape}  |  Target y (price): {y.shape}")
    close_min, close_max = y.min(), y.max()
    print(f"  Price range: {close_min:.2f} – {close_max:.2f}")
    # Directional split computed for reference (not used as target)
    dir_y = (np.diff(y) > 0).astype(int)
    print(f"  Directional split (ref) — Up: {dir_y.sum()}  |  Down: {(1-dir_y).sum()}")
    print(f"  Date range: {pd.to_datetime(dates[0]).date()} → {pd.to_datetime(dates[-1]).date()}")

    # ── Step 5: Time-based train+val / test split (85/15) ──────────────────
    print("\n[5/5] Splitting data and running 5-fold TimeSeriesSplit CV...")
    test_size = int(len(y) * 0.15)
    split_idx = len(y) - test_size

    # No shuffling — data is already in chronological order
    X_tv, X_test = X[:split_idx], X[split_idx:]
    y_tv, y_test = y[:split_idx], y[split_idx:]
    dates_tv, dates_test = dates[:split_idx], dates[split_idx:]

    print(f"  Train+Val: {split_idx} obs  |  Holdout Test: {test_size} obs")

    # ── 5-fold CV ──────────────────────────────────────────────────────────
    tasks = []
    if mode in ('regression',     'both'): tasks.append('regression')
    if mode in ('classification', 'both'): tasks.append('classification')

    # Per-task seeds for reproducibility (same convention as ARX/TCN/ELM)
    TASK_SEEDS = {'regression': 42, 'classification': 43}

    # Binary direction for train+val block (needed by classifier)
    lag1_tv = X_tv[:, 0]   # Close[t-1] for each sample
    yd_tv   = (y_tv > lag1_tv).astype(int)

    def _make_model(task):
        if task == 'regression':
            return TrueFAVARRegressor(n_components=n_factors, var_lags=var_lags,
                                      n_price_lags=n_lags)
        else:
            return TrueFAVARClassifier(n_components=n_factors, n_price_lags=n_lags,
                                       alpha=alpha, l1_ratio=l1_ratio,
                                       regularizer=regularizer)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_stats = {t: {'accs': []} for t in tasks}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_tv), start=1):
        print(f"\n  Fold {fold}:")
        X_train_f, X_val_f   = X_tv[train_idx],  X_tv[val_idx]
        y_train_f, y_val_f   = y_tv[train_idx],  y_tv[val_idx]
        yd_train_f, yd_val_f = yd_tv[train_idx], yd_tv[val_idx]

        for task in tasks:
            np.random.seed(TASK_SEEDS[task] + fold)
            y_tr  = y_train_f  if task == 'regression' else yd_train_f
            y_val = y_val_f    if task == 'regression' else yd_val_f
            mdl = _make_model(task)
            mdl.fit(X_train_f, y_tr)
            m   = mdl.evaluate(X_val_f, y_val)
            fold_stats[task]['accs'].append(m['accuracy'])
            tag = 'REG' if task == 'regression' else 'CLS'
            print(f"    [{tag}] Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}")

    for task in tasks:
        accs = fold_stats[task]['accs']
        tag  = 'REG' if task == 'regression' else 'CLS'
        print(f"\n  CV [{tag}] Accuracy: {np.mean(accs):.4f} ±{np.std(accs):.4f}")

    # ── Final models on full train+val ─────────────────────────────────────
    print("\n[Holdout Test] Training final FAVAR model(s) on full train+val...")
    os.makedirs('models', exist_ok=True)
    yd_test = (y_test > X_test[:, 0]).astype(int)   # true binary direction for test

    if 'regression' in tasks:
        np.random.seed(TASK_SEEDS['regression'])
        print("\n  ── TrueFAVARRegressor ──")
        final_reg = _make_model('regression')
        final_reg.fit(X_tv, y_tv)
        test_reg = final_reg.evaluate(X_test, y_test)
        print_metrics("True FAVAR Regressor (VAR → Δ → direction) — Holdout Test", test_reg)
        final_reg.save('models/favar_var_regressor.pkl')
        os.makedirs('FAVAR', exist_ok=True)
        plot_results(dates_test, yd_test, test_reg,
                     n_lags=n_lags, n_factors=n_factors, elapsed=0.0,
                     output_dir='FAVAR', title_tag='FAVAR-REG')

    if 'classification' in tasks:
        np.random.seed(TASK_SEEDS['classification'])
        print("\n  ── TrueFAVARClassifier ──")
        final_cls = _make_model('classification')
        final_cls.fit(X_tv, yd_tv)
        test_cls = final_cls.evaluate(X_test, yd_test)
        print_metrics("True FAVAR Classifier (direct Up/Down) — Holdout Test", test_cls)
        final_cls.save('models/favar_cls.pkl')
        os.makedirs('FAVAR', exist_ok=True)
        plot_results(dates_test, yd_test, test_cls,
                     n_lags=n_lags, n_factors=n_factors, elapsed=0.0,
                     output_dir='FAVAR', title_tag='FAVAR-CLS')

    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.2f}s")




# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='True FAVAR Pipeline — VAR on price-change + PCA macro factors'
    )
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['regression', 'classification', 'both'],
                        help='regression | classification | both (default: classification)')
    parser.add_argument('--lags', type=int, default=30,
                        help='Number of daily price lags (default: 30)')
    parser.add_argument('--factors', type=int, default=1,
                        help='Number of PCA macro factors to extract (default: 1)')
    parser.add_argument('--var_lags', type=int, default=3,
                        help='Number of VAR lags (default: 3)')
    parser.add_argument('--regularizer', type=str, default='ridge',
                        choices=['ridge', 'elasticnet', 'bayesian'],
                        help='LR regulariser for classification (default: ridge)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='LR regularisation strength C=1/alpha (default: 1.0)')
    parser.add_argument('--l1_ratio', type=float, default=0.5,
                        help='ElasticNet l1_ratio for classification (default: 0.5)')
    parser.add_argument('--fred_train', type=str, default=None,
                        help='Path to FRED-MD historical vintages folder (train range)')
    parser.add_argument('--fred_val', type=str, default=None,
                        help='Path to FRED-MD historical vintages folder (val range)')
    args = parser.parse_args()

    run_daily_favar(
        n_lags=args.lags,
        n_factors=args.factors,
        var_lags=args.var_lags,
        mode=args.mode,
        regularizer=args.regularizer,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        fred_folder_train=args.fred_train,
        fred_folder_val=args.fred_val,
    )
