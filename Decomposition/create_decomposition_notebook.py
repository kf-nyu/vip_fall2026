import json
import os

NOTEBOOK_NAME = "Decomposition_Comparative_Analysis.ipynb"

# =========================================================================
# Notebook Cells Content
# =========================================================================

CELL_SETUP = """\
# --- Universal Colab Setup ---
import os
import sys

if 'google.colab' in sys.modules:
    print("Detected Colab Environment. Setting up dependencies...")
    !pip install EMD-signal torch yfinance
    
    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Optional: Navigate to your project folder
    # os.chdir('/content/drive/MyDrive/...')
else:
    print("Running Locally.")
"""

CELL_IMPORTS = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn

# Inline plots for Notebook
%matplotlib inline
"""

CELL_DECOMPOSERS_CODE = """\
from abc import ABC, abstractmethod
import numpy as np
from PyEMD import EMD as PyEMD_EMD, EEMD as PyEMD_EEMD, CEEMDAN as PyEMD_CEEMDAN

import warnings
warnings.filterwarnings('ignore')

# Inline plots for Notebook
%matplotlib inline
"""

CELL_VIP_INTERFACE = """\
# ===========================================================================
# VIP Base Forecast Model Interface
# ===========================================================================
class BaseForecastModel(ABC):
    \"\"\"
    Simple base class for forecasting models.
    \"\"\"
    def __init__(self, task_type: str, **hyperparameters):
        self.task_type = task_type
        self.hyperparameters = hyperparameters

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    @abstractmethod
    def save(self, filepath: str):
        pass

    @abstractmethod
    def load(self, filepath: str):
        pass
"""

CELL_DECOMPOSITION_LIBRARY = """\
# ===========================================================================
# Decomposition Preprocessing Library
# ===========================================================================
class BaseDecomposer(ABC):
    @abstractmethod
    def decompose(self, series: np.ndarray) -> np.ndarray:
        \"\"\"Returns IMFs of shape (N, n_imfs)\"\"\"
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        pass

class EMDDecomposer(BaseDecomposer):
    def decompose(self, series: np.ndarray) -> np.ndarray:
        emd = PyEMD_EMD()
        return emd.emd(series).T
        
    def get_name(self) -> str:
        return "EMD"

class EEMDDecomposer(BaseDecomposer):
    def __init__(self, trials=25, noise_width=0.1):
        self.trials = trials
        self.noise_width = noise_width
        
    def decompose(self, series: np.ndarray) -> np.ndarray:
        eemd = PyEMD_EEMD(trials=self.trials, noise_width=self.noise_width)
        return eemd.eemd(series).T
        
    def get_name(self) -> str:
        return "EEMD"

class CEEMDDecomposer(BaseDecomposer):
    def __init__(self, trials=100, noise_width=0.01, seed=42):
        self.trials = trials
        self.noise_width = noise_width 
        self.seed = seed
        
    def decompose(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        noise_std = self.noise_width * np.std(series)
        emd = PyEMD_EMD()
        rng = np.random.RandomState(self.seed)

        all_imfs_list = []
        for t in range(self.trials):
            noise = rng.normal(0, noise_std, n)
            all_imfs_list.append(emd.emd(series + noise))
            all_imfs_list.append(emd.emd(series - noise))

        max_n = max(im.shape[0] for im in all_imfs_list)
        avg_imfs = np.zeros((max_n, n))
        
        for im in all_imfs_list:
            padded = np.zeros((max_n, n))
            padded[:im.shape[0], :] = im
            avg_imfs += padded
            
        avg_imfs /= len(all_imfs_list)
        return avg_imfs.T
        
    def get_name(self) -> str:
        return "CEEMD"

class CEEMDANDecomposer(BaseDecomposer):
    def __init__(self, trials=20, epsilon=0.001):
        self.trials = trials
        self.epsilon = epsilon
        
    def decompose(self, series: np.ndarray) -> np.ndarray:
        ceemdan = PyEMD_CEEMDAN(trials=self.trials, epsilon=self.epsilon)
        return ceemdan.ceemdan(series).T
        
    def get_name(self) -> str:
        return "CEEMDAN"
"""

CELL_ADVANCED_MODELS = """\
# ===========================================================================
# ELM Backend Components
# ===========================================================================
class ELMRegressor:
    \"\"\"Single Extreme Learning Machine using PyTorch.\"\"\"
    def __init__(self, n_hidden=100, alpha=100.0, seed=42, device=None):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.seed = seed
        self.device = device or torch.device('cpu')

    def fit(self, X, y):
        n_features = X.shape[1]
        torch.manual_seed(self.seed)
        self.input_weights = torch.randn(n_features, self.n_hidden, device=self.device) * 0.1
        self.bias = torch.randn(self.n_hidden, device=self.device) * 0.1

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        H = torch.sigmoid(X_t @ self.input_weights + self.bias)
        HtH = H.T @ H
        I = torch.eye(self.n_hidden, device=self.device)
        self.output_weights = torch.linalg.solve(HtH + self.alpha * I, H.T @ y_t)

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        H = torch.sigmoid(X_t @ self.input_weights + self.bias)
        preds = H @ self.output_weights
        return preds.cpu().numpy()

class EnsembleELMBackend:
    \"\"\"Wrapper to act like a scikit-learn regressor\"\"\"
    def __init__(self, n_estimators=10, n_hidden=100, alpha=100.0, device=None):
        self.n_estimators = n_estimators
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.device = device or torch.device('cpu')
        self.models = []

    def fit(self, X, y):
        self.models = []
        for i in range(self.n_estimators):
            model = ELMRegressor(n_hidden=self.n_hidden, alpha=self.alpha, seed=i*123, device=self.device)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return np.mean(predictions, axis=1)

# Note: TCN can be added similarly by mapping a standard PyTorch training loop to fit()
"""

CELL_FORECASTER = """\
# ===========================================================================
# VIP Library Forecaster (With Plug-and-Play Decomposer)
# ===========================================================================
def create_lagged_features(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

class GenericDecompositionModel(BaseForecastModel):
    \"\"\"
    A model that optionally decomposes data before applying a backend regressor to each IMF.
    If decomposer=None, it applies the regressor directly to the raw series.
    \"\"\"
    def __init__(self, model_type='Linear', decomposer=None, lookback=30, **kwargs):
        super().__init__(task_type='regression', **kwargs)
        self.model_type = model_type
        self.decomposer = decomposer
        self.lookback = lookback
        self.n_imfs = 0
        
        # Internal state
        self.models = [] 
        self.is_fitted = False

    def _get_backend_model(self):
        name = self.model_type.lower()
        if name == 'linear':
            return make_pipeline(StandardScaler(), LinearRegression())
        elif name == 'elm':
            # Assumes EnsembleELMBackend is defined
            return make_pipeline(StandardScaler(), EnsembleELMBackend(n_estimators=10))
        elif name == 'tcn':
            # PyTorch TCN Backend
            return DailyTCNForecastModel(lookback=self.lookback)
        else:
            raise ValueError(f"Unknown backend: {self.model_type}")

    def fit(self, X, y=None):
        \"\"\"
        X should be a 1D timeseries (N,).
        If decomp is present, break into (N, n_imfs) and train an AR model per IMF.
        If no decomp, train an AR model on the raw X.
        \"\"\"
        # Convert to 1D
        timeseries = np.asarray(X).flatten()
        
        if self.decomposer is not None:
            self.imfs = self.decomposer.decompose(timeseries)
        else:
            # Fake a single "IMF" which is just the raw signal
            self.imfs = timeseries.reshape(-1, 1)
            
        self.n_imfs = self.imfs.shape[1]
        self.models = []

        for i in range(self.n_imfs):
            series = self.imfs[:, i]
            X_feat, y_target = create_lagged_features(series, self.lookback)
            
            model = self._get_backend_model()
            model.fit(X_feat, y_target)
            self.models.append(model)
            
        self.is_fitted = True
        return self

    def predict(self, X):
        \"\"\"
        X should be the new 1D timeseries block to predict.
        Returns array of shape (N - lookback,).
        \"\"\"
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
            
        timeseries = np.asarray(X).flatten()
        
        if self.decomposer is not None:
            imfs = self.decomposer.decompose(timeseries)
        else:
            imfs = timeseries.reshape(-1, 1)
            
        total_pred = None
        
        # We must align with the inner count of trained modules.
        # Ensure we don't try to predict more IMFs than we trained on.
        evaluate_imfs = min(self.n_imfs, imfs.shape[1])
        
        for i in range(evaluate_imfs):
            series = imfs[:, i]
            X_feat, _ = create_lagged_features(series, self.lookback)
            
            imf_pred = self.models[i].predict(X_feat)
            
            if total_pred is None:
                total_pred = np.zeros_like(imf_pred)
            total_pred += imf_pred
            
        return total_pred

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        
        # True values align post-lookback
        y_true = np.asarray(y_test).flatten()[self.lookback:]
        
        min_len = min(len(y_true), len(preds))
        y_true = y_true[:min_len]
        preds = preds[:min_len]
        
        return {
            'r2': r2_score(y_true, preds),
            'mse': mean_squared_error(y_true, preds)
        }
    
    def save(self, filepath: str): pass
    def load(self, filepath: str): pass
"""

CELL_DATA = """\
# ===========================================================================
# Data Loading
# ===========================================================================
def load_daily_wheat_prices():
    # Trying local paths normally found in the repo
    paths = [
        "../data/wheat-futures/wheat_futures_daily.csv",
        "wheat_futures_daily.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            col = 'date' if 'date' in df.columns else 'Date'
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
            df.sort_values(col, inplace=True)
            return df[col].values, df['Close'].values
    print("Data not found. Downloading via yfinance directly for demonstration...")
    import yfinance as yf
    df = yf.download("ZW=F", start="2000-01-01", end="2024-01-01")
    return df.index.values, df['Close'].values

dates, prices = load_daily_wheat_prices()
print(f"Loaded {len(prices)} samples of wheat futures data.")
"""

CELL_EXPERIMENT = """\
# ===========================================================================
# Evaluation Harness
# ===========================================================================
def evaluate_pipeline(name, model_type, decomposer, prices):
    print(f"\\n--- Running Pipeline: {name} ---")
    
    # We use a TimeSeriesSplit to validate
    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(prices)):
        # Require enough data for decomposition & lookback safely
        if len(train_idx) < 100 or len(test_idx) < 50:
            continue
            
        prices_train = prices[train_idx]
        prices_test = prices[test_idx]
        
        model = GenericDecompositionModel(model_type=model_type, decomposer=decomposer, lookback=30)
        
        # Fit on training portion
        model.fit(prices_train)
        
        # Evaluate on test portion
        # (Pass the test sequence in, which simulates realtime decomposition + predict)
        metrics = model.evaluate(prices_test, prices_test)
        
        print(f"  Fold {fold+1} R²: {metrics['r2']:.4f}")
        r2_scores.append(metrics['r2'])
        
    avg_r2 = np.mean(r2_scores) if r2_scores else 0
    print(f">> Final Average R² for {name}: {avg_r2:.4f}")
    return avg_r2

results = {}

# 1. BASELINE: No Decomposition
results['Baseline_Linear'] = evaluate_pipeline(
    name="Raw Baseline (No Decomp + Linear)", 
    model_type='Linear', 
    decomposer=None, 
    prices=prices
)

# 2. Decomposition + Linear
decomposers_to_test = [
    EMDDecomposer(),
    CEEMDDecomposer(trials=5)  # reduced trials for fast Notebook execution
]

for d in decomposers_to_test:
    d_name = d.get_name()
    score = evaluate_pipeline(
        name=f"Decomposed ({d_name} + Linear)",
        model_type='Linear',
        decomposer=d,
        prices=prices
    )
    results[f'{d_name}_Linear'] = score

# 3. Expansion: Sub in ELM backends
for d in decomposers_to_test:
    d_name = d.get_name()
    score = evaluate_pipeline(
        name=f"Advanced ({d_name} + ELM)",
        model_type='ELM',
        decomposer=d,
        prices=prices
    )
    results[f'{d_name}_ELM'] = score
    
print("\\n==================================")
print("FINAL PIPELINE RESULTS (Average R²)")
print("==================================")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
"""

# =========================================================================
# Notebook Assembler
# =========================================================================

notebook_json = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_cell(source_code, cell_type="code"):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source_code.split("\n")]
    }
    # Clean up empty splits at ends usually introduced by multi-line strings
    if cell["source"] and cell["source"][-1] == "\n":
        cell["source"] = cell["source"][:-1]
    
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
        
    notebook_json["cells"].append(cell)

if __name__ == "__main__":
    add_cell("# Decomposition Labs - Comparative Analysis\nThis notebook analyzes the impact of passing signal decompositions (EMD, CEEMD) into forecasting backends compared to naive base-scaling.", "markdown")
    add_cell(CELL_SETUP)
    add_cell(CELL_IMPORTS)
    add_cell(CELL_VIP_INTERFACE)
    add_cell(CELL_DECOMPOSITION_LIBRARY)
    add_cell(CELL_ADVANCED_MODELS)
    add_cell(CELL_FORECASTER)
    add_cell(CELL_DATA)
    add_cell(CELL_EXPERIMENT)

    out_path = os.path.join(os.path.dirname(__file__), NOTEBOOK_NAME)
    with open(out_path, "w") as f:
        json.dump(notebook_json, f, indent=4)
        
    print(f"Notebook generated successfully at {out_path}.")
