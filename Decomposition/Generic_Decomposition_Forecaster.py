import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Jupyter Notebook context
    _this_dir = '/Users/kfunaki/Projects/vip/Decomposition'
    if not os.path.exists(_this_dir):
        _this_dir = os.getcwd()
        if os.path.exists(os.path.join(_this_dir, 'Decomposition', 'Decomposition_Transformers.py')):
            _this_dir = os.path.join(_this_dir, 'Decomposition')

sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.abspath(os.path.join(_this_dir, '..')))

from Decomposition_Transformers import (
    BaseDecomposer, EMDDecomposer, EEMDDecomposer, CEEMDDecomposer, CEEMDANDecomposer
)

# Old models
try:
    from Decomposition_ELM_VIP_Compliant import EnsembleELM, load_daily_wheat_prices
except ImportError:
    EnsembleELM = None
try:
    from TCN.TCN_Daily_Implementation import DailyTCNForecastModel
except ImportError:
    DailyTCNForecastModel = None

# New Revised Models
from ARX.ARX_Revised import load_fred_md, apply_reporting_delay_and_tcodes, forward_fill_macro_to_daily, SELECTED_FEATURES
from ARX.ARX_Revised import ARXModel
from TCN.TCN_Revised import TCNRegressor
from ELM.ELM_Revised import ELMRegressor
from FAVAR.FAVAR_Revised import TrueFAVARRegressor

def load_data():
    base = os.path.abspath(os.path.join(_this_dir, '..'))
    fred_folder_train = os.path.join(base, "data", "fred-md", "Historical FRED-MD Vintages Final")
    fred_folder_val = os.path.join(base, "data", "fred-md", "Historical-vintages-of-FRED-MD-2015-01-to-2024-12")
    wheat_path = os.path.join(base, "data", "wheat-futures", "wheat_futures_daily.csv")
    
    df_wheat = pd.read_csv(wheat_path)
    if 'date' in df_wheat.columns:
        df_wheat['date'] = pd.to_datetime(df_wheat['date'], utc=True).dt.tz_localize(None)
    else:
        df_wheat['Date'] = pd.to_datetime(df_wheat['Date'], utc=True).dt.tz_localize(None)
        df_wheat.rename(columns={'Date': 'date'}, inplace=True)
    df_wheat = df_wheat.sort_values('date')
    
    df_train_raw = load_fred_md(fred_folder_train) if os.path.exists(fred_folder_train) else pd.DataFrame()
    df_val_raw = load_fred_md(fred_folder_val) if os.path.exists(fred_folder_val) else pd.DataFrame()
    df_macro_raw = pd.concat([df_train_raw, df_val_raw]).sort_index()
    df_macro_raw = df_macro_raw[~df_macro_raw.index.duplicated(keep='last')]
    
    df_macro = apply_reporting_delay_and_tcodes(df_macro_raw)
    df_merged = forward_fill_macro_to_daily(df_macro, df_wheat)
    df_merged = df_merged.dropna().reset_index(drop=True)
    
    dates = df_merged['date'].values
    prices = df_merged['Close'].values
    macros = df_merged[SELECTED_FEATURES].values
    
    return dates, prices, macros


class BaseForecastModel(ABC):
    def __init__(self, task_type='regression', **hyperparameters):
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
# IMPORTANT FOR REVISED MODELS (IMFS need simple differences instead of log returns)
# ─────────────────────────────────────────────────────────────────────────────
class IMF_ARXModel(ARXModel):
    def _flatten(self, X_raw: np.ndarray) -> np.ndarray:
        N, L, F = X_raw.shape
        prices = X_raw[:, :, 0]
        diffs = np.diff(prices, axis=1)
        first_diff = diffs[:, :1]
        diffs = np.hstack([first_diff, diffs])
        exog = X_raw[:, -1, 1:]
        return np.hstack([diffs, exog])

class IMF_ELMRegressor(ELMRegressor):
    def _flatten(self, X_raw: np.ndarray) -> np.ndarray:
        N, L, F = X_raw.shape
        prices = X_raw[:, :, 0]
        diffs = np.diff(prices, axis=1)
        first_diff = diffs[:, :1]
        diffs = np.hstack([first_diff, diffs])
        exog = X_raw[:, -1, 1:]
        return np.hstack([diffs, exog])

class IMF_FAVARModel(TrueFAVARRegressor):
    def _flatten(self, X_raw: np.ndarray) -> np.ndarray:
        N, L, F = X_raw.shape
        prices = X_raw[:, :, 0]
        exog = X_raw[:, -1, 1:]
        return np.hstack([prices, exog])

class GenericDecompositionModel(BaseForecastModel):
    def __init__(
        self,
        task_type='regression',
        decomposer: BaseDecomposer = None,
        backend_name='Linear',
        lookback=30,
        n_denoise=2,
        use_ridge=False,
        **kwargs
    ):
        super().__init__(task_type=task_type, **kwargs)
        self.decomposer = decomposer
        self.backend_name = backend_name
        self.lookback = lookback
        self.n_denoise = n_denoise
        self.use_ridge = use_ridge
        
        self.is_fitted_ = False
        self.models_ = []
        self.meta_model_ = None
        
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _get_backend_model(self, c_idx=0):
        name = self.backend_name.lower()
        if name == 'linear':
            return make_pipeline(StandardScaler(), LinearRegression())
        elif name == 'ridge':
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        elif name == 'lasso':
            return make_pipeline(StandardScaler(), Lasso(alpha=0.01))
        elif name == 'elm':
            if EnsembleELM is None:
                raise ImportError("EnsembleELM not found.")
            return make_pipeline(StandardScaler(), EnsembleELM(
                n_estimators=50, n_hidden=100, alpha=100.0, device=self.device
            ))
        elif name == 'tcn':
            if DailyTCNForecastModel is None:
                raise ImportError("DailyTCNForecastModel not found.")
            class TCNWrapper:
                def __init__(self, lookback, device=None):
                    self.model = DailyTCNForecastModel(lookback=lookback, epochs=10, batch_size=64, device=device)
                def fit(self, X, y):
                    X_3d = X.reshape(X.shape[0], X.shape[1], 1)
                    y_2d = y.reshape(-1, 1)
                    self.model.fit(X_3d, y_2d)
                    return self
                def predict(self, X):
                    X_3d = X.reshape(X.shape[0], X.shape[1], 1)
                    return self.model.predict(X_3d).flatten()
            return make_pipeline(StandardScaler(), TCNWrapper(lookback=self.lookback, device=self.device))
        
        # New models - target shape varies by class
        elif name == 'arx_revised':
            return IMF_ARXModel(regularizer='ridge', alpha=1.0)
        elif name == 'tcn_revised':
            return TCNRegressor(task_type='regression', epochs=100, patience=10, num_channels=[32, 16], batch_size=256)
        elif name == 'elm_revised':
            return IMF_ELMRegressor(alpha=1.0, n_hidden=500, hidden_seed=42+c_idx)
        elif name == 'favar_revised':
            # FAVAR requires its own pipeline wrapper if one uses raw
            return IMF_FAVARModel(task_type='regression')
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")

    def fit_on_imfs(self, imfs, prices, macros, train_end):
        prices = np.asarray(prices).flatten()
        self.n_imfs_ = imfs.shape[1]
        
        T = len(prices)
        component_datasets = []
        is_revised_backend = self.backend_name.lower() in ['arx_revised', 'tcn_revised', 'elm_revised', 'favar_revised']
        
        for c in range(self.n_imfs_):
            if c < self.n_denoise:
                component_datasets.append(None)
                continue
                
            signal = imfs[:, c]
            X_c, y_c = [], []
            for i in range(self.lookback, T):
                imf_win = signal[i - self.lookback : i].reshape(-1, 1)
                if is_revised_backend:
                    macro_win = macros[i - self.lookback : i]
                    X_c.append(np.hstack([imf_win, macro_win]))
                    y_c.append(signal[i] - signal[i-1]) # target is IMF differences
                else:
                    X_c.append(imf_win)
                    y_c.append(signal[i]) # generic models predict raw values
            
            if is_revised_backend:
                component_datasets.append((np.array(X_c, dtype=np.float32), np.array(y_c, dtype=np.float32)))
            else:
                component_datasets.append((np.array(X_c, dtype=np.float32).reshape(-1, self.lookback), np.array(y_c, dtype=np.float32)))
            
        ar_train_end = train_end - self.lookback
        n_total_ar = T - self.lookback
        
        self.models_ = []
        component_preds_train = np.zeros((ar_train_end, self.n_imfs_))
        component_preds_test = np.zeros((n_total_ar - ar_train_end, self.n_imfs_))
        
        for c in range(self.n_imfs_):
            if c < self.n_denoise:
                self.models_.append(None)
                continue
            
            X_c, y_c = component_datasets[c]
            X_train_c = X_c[:ar_train_end]
            y_train_c = y_c[:ar_train_end]
            X_test_c = X_c[ar_train_end:]
            
            model = self._get_backend_model(c_idx=c)
            
            if self.backend_name.lower() in ('elm', 'tcn'):
                scaler = model.steps[0][1]
                X_train_scaled = scaler.fit_transform(X_train_c)
                backend_mdl = model.steps[1][1]
                backend_mdl.fit(X_train_scaled, y_train_c)
                component_preds_train[:, c] = backend_mdl.predict(X_train_scaled)
                if X_test_c.shape[0] > 0:
                    component_preds_test[:, c] = backend_mdl.predict(scaler.transform(X_test_c))
            elif is_revised_backend:
                model.fit(X_train_c, y_train_c)
                component_preds_train[:, c] = model.predict(X_train_c)
                if X_test_c.shape[0] > 0:
                    component_preds_test[:, c] = model.predict(X_test_c)
            else:
                model.fit(X_train_c, y_train_c)
                component_preds_train[:, c] = model.predict(X_train_c)
                if X_test_c.shape[0] > 0:
                    component_preds_test[:, c] = model.predict(X_test_c)
            self.models_.append(model)
        
        valid_prices = prices[self.lookback:]
        
        if is_revised_backend:
            final_pred_train_delta = np.sum(component_preds_train, axis=1)
            final_pred_test_delta  = np.sum(component_preds_test, axis=1)
            self.is_fitted_ = True
            y_target_delta = np.diff(prices[self.lookback-1:])
            return final_pred_train_delta, final_pred_test_delta, y_target_delta, valid_prices
        else:
            train_prices_aligned = valid_prices[:ar_train_end]
            if self.use_ridge:
                self.meta_model_ = Ridge(alpha=self.hyperparameters.get('lasso_alpha', 0.01), fit_intercept=True)
            else:
                self.meta_model_ = Lasso(alpha=self.hyperparameters.get('lasso_alpha', 0.01), positive=True, fit_intercept=True, max_iter=100000)
                
            self.meta_model_.fit(component_preds_train, train_prices_aligned)
            self.is_fitted_ = True
            
            if component_preds_test.shape[0] > 0:
                final_pred_test = self.meta_model_.predict(component_preds_test)
                final_pred_train = self.meta_model_.predict(component_preds_train)
            else:
                final_pred_test = np.array([])
                final_pred_train = np.array([])
            return final_pred_train, final_pred_test, valid_prices, valid_prices
            
    def fit(self, X_train, y_train):
        raise NotImplementedError("Use fit_on_imfs directly for testing")
    def predict(self, X):
        raise NotImplementedError("Use fit_on_imfs directly for testing")
    def evaluate(self, X_test, y_test):
        pass
    def save(self, filepath: str):
        pass
    def load(self, filepath: str):
        pass


def test_all_variants(backend_name='ELM_Revised'):
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 70)
    print(f"Generic Decomposition Forecaster | Backend: {backend_name}")
    print("=" * 70 + "\\n")
    
    dates, prices, macros = load_data()
    if prices is None: return
    
    # We will test EEMD for quick display verification
    decomposers = [EEMDDecomposer(trials=20, noise_width=0.2)]
    
    tscv = TimeSeriesSplit(n_splits=5)
    comparison_results = {}
    final_fold_data = {}
    lookback = 30
    is_revised_backend = backend_name.lower() in ['arx_revised', 'tcn_revised', 'elm_revised', 'favar_revised']
    
    for decomposer in decomposers:
        model_name = decomposer.get_name()
        print(f"\\n{'='*60}")
        print(f"  Model: {model_name}-{backend_name}  (Constrained Stacking)")
        print(f"{'='*60}")
        
        decomp_dir = os.path.join(_this_dir, 'Decomposed_Data')
        decomp_file = os.path.join(decomp_dir, f"{model_name}_Components.csv")
        
        if os.path.exists(decomp_file):
            print(f"\\n  Loading pre-decomposed {model_name} from {decomp_file}...")
            df_decomp = pd.read_csv(decomp_file)
            imf_cols = [c for c in df_decomp.columns if c.startswith('IMF')]
            imfs = df_decomp[imf_cols].values
        else:
            decomp_start = time.time()
            imfs = decomposer.decompose(prices)
            os.makedirs(decomp_dir, exist_ok=True)
            df_save = pd.DataFrame({'Date': dates, 'Close': prices})
            for i in range(imfs.shape[1]):
                df_save[f'IMF{i+1}'] = imfs[:, i]
            df_save.to_csv(decomp_file, index=False)
        
        fold = 0
        fold_r2s = []
        
        for train_idx, val_idx in tscv.split(np.arange(len(prices[lookback:]))):
            fold += 1
            fold_start = time.time()
            train_end = len(train_idx) + lookback
            
            fold_model = GenericDecompositionModel(
                decomposer=decomposer,
                backend_name=backend_name,
                lookback=lookback,
                n_denoise=2
            )
            
            final_train, final_test, targets, actual_prices = fold_model.fit_on_imfs(imfs, prices, macros, train_end)
            ar_train_end = train_end - lookback
            
            y_true = targets[val_idx]
            preds = final_test[:len(val_idx)]
            
            min_len = min(len(y_true), len(preds))
            y_true = y_true[:min_len]
            preds = preds[:min_len]
            
            mse = float(np.mean((y_true - preds) ** 2))
            rmse = float(np.sqrt(mse))
            ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_residual = np.sum((y_true - preds) ** 2)
            r2 = float(1 - ss_residual / ss_total) if ss_total != 0 else 0.0
            
            fold_r2s.append(r2)
            print(f"- Fold {fold} (train={train_end}, val={len(val_idx)}) - RMSE: {rmse:.4f} | R\u00b2: {r2:.4f} | Time: {time.time() - fold_start:.1f}s")
            
            if fold == 5:
                test_start = train_end
                dates_aligned = dates[test_start: test_start + min_len]
                if is_revised_backend:
                    p_prev = prices[test_start - 1 : test_start - 1 + min_len]
                    p_true = prices[test_start : test_start + min_len]
                    p_pred = p_prev + preds
                    final_fold_data[model_name] = (p_true, p_pred, dates_aligned)
                else:
                    p_true = prices[test_start : test_start + min_len]
                    final_fold_data[model_name] = (p_true, preds, dates_aligned)
                
        avg_r2 = np.mean(fold_r2s)
        comparison_results[model_name] = fold_r2s
        print(f">> {model_name}-{backend_name} Average R\u00b2: {avg_r2:.4f}")

if __name__ == "__main__":
    if 'ipykernel' in sys.argv[0] or any(arg.endswith('.json') for arg in sys.argv):
        backend_choice = 'ARX_Revised'
        print(f"Jupyter environment detected. Forcing default backend: {backend_choice}")
        test_all_variants(backend_choice)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--backend', type=str, default='ARX_Revised', help='Backend Model (Linear, Lasso, Ridge, ELM, ARX_Revised, TCN_Revised, FAVAR_Revised, ELM_Revised)')
        args = parser.parse_args()
        test_all_variants(args.backend)
