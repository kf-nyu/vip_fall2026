"""
Decomposition-ELM Implementation - VIP Abstract Class Compliant
================================================================
Signal decomposition (EMD / EEMD / CEEMD / CEEMDAN) combined with Ensemble ELM
and Constrained Lasso Stacking for wheat futures price forecasting.

All models inherit from BaseForecastModel and implement the VIP interface:
    fit(X_train, y_train)
    predict(X)
    evaluate(X_test, y_test)
    save(filepath)
    load(filepath)

Decomposition Methods:
    1. EMD     - Empirical Mode Decomposition (Huang et al., 1998)
    2. EEMD    - Ensemble EMD (Wu & Huang, 2009)
    3. CEEMD   - Complementary Ensemble EMD (Yeh et al., 2010)
    4. CEEMDAN - Complete EEMD with Adaptive Noise (Torres et al., 2011)
"""

import pandas as pd
import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from PyEMD import EMD as PyEMD_EMD, EEMD as PyEMD_EEMD, CEEMDAN as PyEMD_CEEMDAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import time


# ===========================================================================
# Base Abstract Class (VIP Standard)
# ===========================================================================
class BaseForecastModel(ABC):
    """
    Simple base class for forecasting models.
    Your model should inherit from this and implement all the methods below.
    """

    def __init__(self, task_type: str, **hyperparameters):
        """
        Initialize your model.

        Args:
            task_type: Either 'regression' or 'classification'
            **hyperparameters: Your model's parameters
        """
        self.task_type = task_type
        self.hyperparameters = hyperparameters

    @abstractmethod
    def fit(self, X_train, y_train):
        """Train your model on the training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions on new data."""
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save your trained model to a file."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load a previously saved model."""
        pass


# ===========================================================================
# Internal Helper: Single ELM Regressor (PyTorch + GPU)
# ===========================================================================
class ELMRegressor:
    """
    Single Extreme Learning Machine using PyTorch.
    Hidden layer (random, frozen) + Ridge-solved output layer.
    Supports GPU acceleration (MPS / CUDA / CPU).
    """

    def __init__(self, n_hidden=100, alpha=100.0, seed=42, device=None):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.seed = seed
        self.device = device or torch.device('cpu')
        self.input_weights = None  # (n_features, n_hidden) tensor on device
        self.bias = None           # (n_hidden,) tensor on device
        self.output_weights = None # (n_hidden,) tensor on device

    def fit(self, X, y):
        """Train ELM: random hidden projection + Ridge solve on GPU."""
        n_features = X.shape[1]
        torch.manual_seed(self.seed)

        # Random hidden-layer weights (frozen)
        self.input_weights = torch.randn(
            n_features, self.n_hidden, device=self.device
        ) * 0.1
        self.bias = torch.randn(self.n_hidden, device=self.device) * 0.1

        # Move data to device
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Hidden layer activations: H = sigmoid(X @ W + b)
        H = torch.sigmoid(X_t @ self.input_weights + self.bias)

        # Ridge regression closed-form: beta = (H^T H + alpha*I)^{-1} H^T y
        HtH = H.T @ H
        I = torch.eye(self.n_hidden, device=self.device)
        self.output_weights = torch.linalg.solve(
            HtH + self.alpha * I, H.T @ y_t
        )

    def predict(self, X):
        """Predict using trained ELM on GPU, return numpy array."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        H = torch.sigmoid(X_t @ self.input_weights + self.bias)
        preds = H @ self.output_weights
        return preds.cpu().numpy()


# ===========================================================================
# Internal Helper: Ensemble of ELMs (PyTorch + GPU)
# ===========================================================================
class EnsembleELM:
    """Ensemble of PyTorch ELMs with different random seeds, averaged predictions."""

    def __init__(self, n_estimators=10, n_hidden=100, alpha=100.0, device=None):
        self.n_estimators = n_estimators
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.device = device or torch.device('cpu')
        self.models = []

    def fit(self, X, y):
        self.models = []
        for i in range(self.n_estimators):
            model = ELMRegressor(
                n_hidden=self.n_hidden,
                alpha=self.alpha,
                seed=i * 123,
                device=self.device
            )
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return np.mean(predictions, axis=1)


# ===========================================================================
# Base Decomposition-ELM Model (shared logic for all decomposition methods)
# ===========================================================================
class BaseDecompositionELM(BaseForecastModel):
    """
    Base class for Decomposition + Ensemble ELM with Constrained Stacking.

    Pipeline:
        1. Decompose the price signal into IMFs using a subclass-specific method.
        2. Build autoregressive features for each IMF (lag window).
        3. Train an Ensemble of ELMs for each IMF.
        4. Combine IMF forecasts via a Constrained Lasso meta-learner (positive weights).
        5. The meta-learner output is the final price forecast.

    Subclasses only need to implement `_decompose(prices)`.
    """

    def __init__(
        self,
        task_type='regression',
        lookback=30,
        n_estimators=50,
        n_hidden=100,
        elm_alpha=100.0,
        lasso_alpha=0.01,
        n_denoise=2,
        use_ridge=False,
        **kwargs
    ):
        """
        Initialize the Decomposition-ELM model.

        Args:
            task_type: Task type (default 'regression')
            lookback: Number of past lags for autoregressive features (default 30)
            n_estimators: Number of ELMs in the ensemble per IMF (default 10)
            n_hidden: Hidden neurons per ELM (default 100)
            elm_alpha: Ridge regularization for each ELM (default 100.0)
            lasso_alpha: Lasso regularization for the meta-learner (default 0.01)
            n_denoise: Number of leading high-frequency IMFs to ignore (default 2)
            use_ridge: If True, use Ridge instead of Lasso for meta-learner (default False)
            **kwargs: Additional hyperparameters
        """
        super().__init__(
            task_type=task_type,
            lookback=lookback,
            n_estimators=n_estimators,
            n_hidden=n_hidden,
            elm_alpha=elm_alpha,
            lasso_alpha=lasso_alpha,
            n_denoise=n_denoise,
            use_ridge=use_ridge,
            **kwargs
        )
        self.lookback = lookback
        self.n_estimators = n_estimators
        self.n_hidden = n_hidden
        self.elm_alpha = elm_alpha
        self.lasso_alpha = lasso_alpha
        self.n_denoise = n_denoise
        self.use_ridge = use_ridge

        # Device detection (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Trained state (populated by fit())
        self.train_prices_ = None  # Training signal (for predict context)
        self.imfs_ = None          # Decomposed IMFs (n_samples, n_imfs)
        self.n_imfs_ = None        # Number of IMFs
        self.scalers_ = None       # Per-IMF StandardScalers
        self.ensembles_ = None     # Per-IMF EnsembleELMs
        self.meta_model_ = None    # Lasso stacking meta-learner
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Abstract: subclasses implement the specific decomposition method
    # ------------------------------------------------------------------
    @abstractmethod
    def _decompose(self, prices: np.ndarray) -> np.ndarray:
        """
        Decompose a 1-D price signal into IMFs.

        Args:
            prices: 1-D numpy array of price values.

        Returns:
            imfs: 2-D numpy array of shape (n_samples, n_imfs).
                  Each column is one Intrinsic Mode Function.
        """
        pass

    @abstractmethod
    def _decomposition_name(self) -> str:
        """Return a human-readable name for the decomposition method."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_autoregressive_dataset(signal: np.ndarray, lookback: int):
        """Build (X, y) pairs from a 1-D signal using a sliding window."""
        X, y = [], []
        for i in range(lookback, len(signal)):
            X.append(signal[i - lookback: i])
            y.append(signal[i])
        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # VIP Interface: fit
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Train the Decomposition-ELM model.

        The model ignores X_train features—it only uses y_train (the price
        signal) because decomposition methods work directly on the time series.

        Args:
            X_train: Ignored (kept for VIP interface compatibility).
                     Can pass np.arange(len(prices)) as a placeholder.
            y_train: 1-D array of price values to decompose and learn.
        """
        prices = np.asarray(y_train).flatten()
        self.train_prices_ = prices.copy()  # Store for predict() context

        # Step 1: Decompose
        print(f"  Using device: {self.device}")
        print(f"  Running {self._decomposition_name()} decomposition...")
        self.imfs_ = self._decompose(prices)
        self.n_imfs_ = self.imfs_.shape[1]
        print(f"  Decomposed into {self.n_imfs_} components.")

        # Step 2: Build per-IMF autoregressive datasets
        component_datasets = []
        for c in range(self.n_imfs_):
            X_c, y_c = self._build_autoregressive_dataset(
                self.imfs_[:, c], self.lookback
            )
            component_datasets.append((X_c, y_c))

        # Aligned targets (original prices minus the lookback burn-in)
        valid_prices = prices[self.lookback:]
        n_aligned = len(valid_prices)

        # Step 3: Train Ensemble ELM for each IMF + collect stacking features
        self.scalers_ = []
        self.ensembles_ = []
        component_preds = np.zeros((n_aligned, self.n_imfs_))

        for c in range(self.n_imfs_):
            if c < self.n_denoise:
                # Skip high-frequency noise IMFs
                self.scalers_.append(None)
                self.ensembles_.append(None)
                continue

            X_c, y_c = component_datasets[c]
            scaler = StandardScaler()
            X_c_scaled = scaler.fit_transform(X_c)

            ensemble = EnsembleELM(
                n_estimators=self.n_estimators,
                n_hidden=self.n_hidden,
                alpha=self.elm_alpha,
                device=self.device
            )
            ensemble.fit(X_c_scaled, y_c)

            component_preds[:, c] = ensemble.predict(X_c_scaled)
            self.scalers_.append(scaler)
            self.ensembles_.append(ensemble)

        # Step 4: Train meta-learner
        if self.use_ridge:
            self.meta_model_ = Ridge(alpha=self.lasso_alpha, fit_intercept=True)
        else:
            self.meta_model_ = Lasso(
                alpha=self.lasso_alpha,
                positive=True,
                fit_intercept=True
            )
        self.meta_model_.fit(component_preds, valid_prices)
        self.is_fitted_ = True

        print(f"  Meta-model weights: {self.meta_model_.coef_.round(2)}")
        print(f"  {self._decomposition_name()}-ELM training complete.")

    # ------------------------------------------------------------------
    # VIP Interface: predict
    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Make predictions on new data.

        Because decomposition-based models require the raw signal to
        decompose, this method expects X to be a 1-D price signal
        (or 2-D with one column).  It decomposes the signal, builds
        autoregressive features, predicts each IMF, and stacks.

        Args:
            X: 1-D (or single-column 2-D) array of price values to forecast.

        Returns:
            predictions: 1-D numpy array of forecasted prices
                         (length = len(X) - lookback).
        """
        if not self.is_fitted_:
            raise ValueError("Model not trained! Call fit() first.")

        prices = np.asarray(X).flatten()

        # Decompose the new signal
        imfs = self._decompose(prices)
        n_imfs = imfs.shape[1]

        # Ensure the number of IMFs matches (pad or trim)
        if n_imfs < self.n_imfs_:
            pad = np.zeros((imfs.shape[0], self.n_imfs_ - n_imfs))
            imfs = np.hstack([imfs, pad])
        elif n_imfs > self.n_imfs_:
            imfs = imfs[:, :self.n_imfs_]

        # Build predictions per IMF
        n_aligned = len(prices) - self.lookback
        component_preds = np.zeros((n_aligned, self.n_imfs_))

        for c in range(self.n_imfs_):
            if c < self.n_denoise or self.ensembles_[c] is None:
                continue
            X_c, _ = self._build_autoregressive_dataset(imfs[:, c], self.lookback)
            X_c_scaled = self.scalers_[c].transform(X_c)
            component_preds[:, c] = self.ensembles_[c].predict(X_c_scaled)

        # Stack via meta-learner
        predictions = self.meta_model_.predict(component_preds)
        return predictions

    # ------------------------------------------------------------------
    # VIP Interface: evaluate
    # ------------------------------------------------------------------
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.

        Args:
            X_test: 1-D price signal for decomposition.
            y_test: True price values (aligned to the output of predict()).

        Returns:
            metrics: Dictionary with mse, rmse, mae, r2, mape, predictions.
        """
        predictions = self.predict(X_test)

        # Align y_test: predict() returns len(X_test) - lookback values
        y_true = np.asarray(y_test).flatten()
        if len(y_true) > len(predictions):
            y_true = y_true[self.lookback:]

        # Truncate to the shorter of the two (safety)
        min_len = min(len(y_true), len(predictions))
        y_true = y_true[:min_len]
        predictions = predictions[:min_len]

        # Metrics
        mse = float(np.mean((y_true - predictions) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - predictions)))
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - predictions) ** 2)
        r2 = float(1 - ss_residual / ss_total) if ss_total != 0 else 0.0
        mape = float(np.mean(np.abs(
            (y_true - predictions) / (y_true + 1e-10)
        )) * 100)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': predictions
        }

    # ------------------------------------------------------------------
    # Train/predict using pre-decomposed IMFs (avoids re-decomposition)
    # ------------------------------------------------------------------
    def fit_on_imfs(self, imfs, prices, train_end):
        """
        Train on pre-decomposed IMFs and predict the test portion.

        This avoids the decomposition mismatch between fit() and predict()
        by using the same decomposition for both training and testing.
        Decomposition should be done once externally and reused across folds.

        VIP interface methods (fit/predict/evaluate/save/load) still work
        for standalone usage.

        Args:
            imfs:      2-D array (n_samples, n_imfs) — pre-decomposed IMFs.
            prices:    1-D array — the original price signal.
            train_end: int — index separating train from test.

        Returns:
            predictions: 1-D numpy array for the test portion.
            y_test:      1-D numpy array of true test prices (aligned).
        """
        prices = np.asarray(prices).flatten()
        self.train_prices_ = prices[:train_end].copy()
        self.n_imfs_ = imfs.shape[1]
        self.imfs_ = imfs

        # Build per-IMF autoregressive datasets from FULL signal
        all_X = {}
        for c in range(self.n_imfs_):
            if c < self.n_denoise:
                continue
            X_c, y_c = self._build_autoregressive_dataset(imfs[:, c], self.lookback)
            all_X[c] = (X_c, y_c)

        # Row i in AR dataset corresponds to original index (lookback + i)
        ar_train_end = train_end - self.lookback
        valid_prices = prices[self.lookback:]
        n_total_ar = len(valid_prices)

        # Train Ensemble ELM on training portion of each IMF
        self.scalers_ = []
        self.ensembles_ = []
        component_preds_train = np.zeros((ar_train_end, self.n_imfs_))
        component_preds_test = np.zeros((n_total_ar - ar_train_end, self.n_imfs_))

        for c in range(self.n_imfs_):
            if c < self.n_denoise:
                self.scalers_.append(None)
                self.ensembles_.append(None)
                continue

            X_c, y_c = all_X[c]
            X_train_c = X_c[:ar_train_end]
            y_train_c = y_c[:ar_train_end]
            X_test_c = X_c[ar_train_end:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_c)
            X_test_scaled = scaler.transform(X_test_c)

            ensemble = EnsembleELM(
                n_estimators=self.n_estimators,
                n_hidden=self.n_hidden,
                alpha=self.elm_alpha,
                device=self.device
            )
            ensemble.fit(X_train_scaled, y_train_c)

            component_preds_train[:, c] = ensemble.predict(X_train_scaled)
            component_preds_test[:, c] = ensemble.predict(X_test_scaled)

            self.scalers_.append(scaler)
            self.ensembles_.append(ensemble)

        # Train meta-learner on training predictions
        train_prices_aligned = valid_prices[:ar_train_end]
        if self.use_ridge:
            self.meta_model_ = Ridge(alpha=self.lasso_alpha, fit_intercept=True)
        else:
            self.meta_model_ = Lasso(
                alpha=self.lasso_alpha,
                positive=True,
                fit_intercept=True
            )
        self.meta_model_.fit(component_preds_train, train_prices_aligned)
        self.is_fitted_ = True

        print(f"  Meta-model weights: {self.meta_model_.coef_.round(2)}")

        # Predict on test portion
        predictions = self.meta_model_.predict(component_preds_test)
        y_test = valid_prices[ar_train_end:]

        return predictions, y_test

    # ------------------------------------------------------------------
    # VIP Interface: save
    # ------------------------------------------------------------------
    def save(self, filepath: str):
        """
        Save the trained model to a file.

        Args:
            filepath: Path where to save (e.g., 'models/emd_elm.pkl')
        """
        if not self.is_fitted_:
            raise ValueError("No model to save! Train the model first.")

        save_dict = {
            'task_type': self.task_type,
            'hyperparameters': self.hyperparameters,
            'lookback': self.lookback,
            'n_estimators': self.n_estimators,
            'n_hidden': self.n_hidden,
            'elm_alpha': self.elm_alpha,
            'lasso_alpha': self.lasso_alpha,
            'n_denoise': self.n_denoise,
            'n_imfs_': self.n_imfs_,
            'train_prices_': self.train_prices_,
            'scalers_': self.scalers_,
            'ensembles_': self.ensembles_,
            'meta_model_': self.meta_model_,
        }

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else '.',
            exist_ok=True
        )
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"Model saved to {filepath}")

    # ------------------------------------------------------------------
    # VIP Interface: load
    # ------------------------------------------------------------------
    def load(self, filepath: str):
        """
        Load a previously saved model.

        Args:
            filepath: Path to the saved model file.
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        self.task_type = save_dict.get('task_type', 'regression')
        self.hyperparameters = save_dict.get('hyperparameters', {})
        self.lookback = save_dict['lookback']
        self.n_estimators = save_dict['n_estimators']
        self.n_hidden = save_dict['n_hidden']
        self.elm_alpha = save_dict['elm_alpha']
        self.lasso_alpha = save_dict['lasso_alpha']
        self.n_denoise = save_dict['n_denoise']
        self.n_imfs_ = save_dict['n_imfs_']
        self.train_prices_ = save_dict.get('train_prices_', np.array([]))
        self.scalers_ = save_dict['scalers_']
        self.ensembles_ = save_dict['ensembles_']
        self.meta_model_ = save_dict['meta_model_']
        self.is_fitted_ = True

        print(f"Model loaded from {filepath}")


# ===========================================================================
# Concrete Model 1: EMD-ELM
# ===========================================================================
class EMDELMRegressor(BaseDecompositionELM):
    """
    Empirical Mode Decomposition + Ensemble ELM + Constrained Stacking.

    EMD (Huang et al., 1998) is the original adaptive decomposition method.
    It decomposes a signal into Intrinsic Mode Functions via sifting, but
    is susceptible to mode mixing.
    """

    def _decompose(self, prices: np.ndarray) -> np.ndarray:
        emd = PyEMD_EMD()
        imfs = emd.emd(prices)  # shape: (n_imfs, n_samples)
        return imfs.T           # shape: (n_samples, n_imfs)

    def _decomposition_name(self) -> str:
        return "EMD"


# ===========================================================================
# Concrete Model 2: EEMD-ELM
# ===========================================================================
class EEMDELMRegressor(BaseDecompositionELM):
    """
    Ensemble Empirical Mode Decomposition + Ensemble ELM + Constrained Stacking.

    EEMD (Wu & Huang, 2009) is a noise-assisted technique that adds white
    noise to the signal, decomposes multiple times, and averages the results
    to reduce mode mixing.
    """

    def __init__(self, trials=200, noise_width=0.01, **kwargs):
        """
        Args:
            trials: Number of noise-assisted trials (default 1000).
            noise_width: Amplitude of added white noise (default 0.01).
            **kwargs: Passed to BaseDecompositionELM.
        """
        super().__init__(**kwargs)
        self.trials = trials
        self.noise_width = noise_width

    def _decompose(self, prices: np.ndarray) -> np.ndarray:
        eemd = PyEMD_EEMD(trials=self.trials, noise_width=self.noise_width)
        imfs = eemd.eemd(prices)  # shape: (n_imfs, n_samples)
        return imfs.T              # shape: (n_samples, n_imfs)

    def _decomposition_name(self) -> str:
        return "EEMD"


# ===========================================================================
# Concrete Model 3: CEEMD-ELM
# ===========================================================================
class CEEMDELMRegressor(BaseDecompositionELM):

    def __init__(self, trials=200, noise_width=0.01, **kwargs):
        """
        Args:
            trials: Number of complementary noise pairs (default 200).
            noise_width: Standard deviation of added white noise as a
                         fraction of the signal's std (default 0.01).
            **kwargs: Passed to BaseDecompositionELM.
        """
        super().__init__(**kwargs)
        self.trials = trials
        self.noise_width = noise_width

    def _decompose(self, prices: np.ndarray) -> np.ndarray:
        """Complementary EEMD: average IMFs from (+noise, -noise) pairs."""
        n = len(prices)
        noise_std = self.noise_width * np.std(prices)
        emd = PyEMD_EMD()
        rng = np.random.RandomState(42)

        all_imfs = []  # collect IMF arrays from every decomposition

        for t in range(self.trials):
            noise = rng.normal(0, noise_std, n)

            # Positive noise trial
            imfs_pos = emd.emd(prices + noise)   # (n_imfs, n_samples)
            # Negative noise trial
            imfs_neg = emd.emd(prices - noise)   # (n_imfs, n_samples)

            all_imfs.append(imfs_pos)
            all_imfs.append(imfs_neg)

            if (t + 1) % 5 == 0 or t == 0:
                print(f"    CEEMD trial {t+1}/{self.trials} done")

        # Determine the maximum number of IMFs across all trials
        max_n_imfs = max(im.shape[0] for im in all_imfs)

        # Pad each trial's IMFs to max_n_imfs and average
        ensemble = np.zeros((max_n_imfs, n))
        for im in all_imfs:
            padded = np.zeros((max_n_imfs, n))
            padded[:im.shape[0], :] = im
            ensemble += padded

        ensemble /= len(all_imfs)
        return ensemble.T  # shape: (n_samples, n_imfs)

    def _decomposition_name(self) -> str:
        return "CEEMD"


# ===========================================================================
# Concrete Model 4: CEEMDAN-ELM
# ===========================================================================
class CEEMDANELMRegressor(BaseDecompositionELM):

    def __init__(self, trials=50, epsilon=0.005, **kwargs):
        """
        Args:
            trials: Number of ensemble trials (default 50).
            epsilon: Noise amplitude scaling factor (default 0.005).
            **kwargs: Passed to BaseDecompositionELM.
        """
        super().__init__(**kwargs)
        self.trials = trials
        self.epsilon = epsilon

    def _decompose(self, prices: np.ndarray) -> np.ndarray:
        ceemdan = PyEMD_CEEMDAN(trials=self.trials, epsilon=self.epsilon)
        imfs = ceemdan.ceemdan(prices)  # shape: (n_imfs, n_samples)
        return imfs.T                    # shape: (n_samples, n_imfs)

    def _decomposition_name(self) -> str:
        return "CEEMDAN"


# ===========================================================================
# Data Preparation
# ===========================================================================
def load_daily_wheat_prices():
    """Load daily wheat futures close prices and return (dates, prices)."""
    wheat_path = os.path.join(
        os.path.dirname(__file__), '..', 'data',
        'wheat-futures', 'wheat_futures_daily.csv'
    )
    if not os.path.exists(wheat_path):
        print(f"Data not found at {wheat_path}")
        return None

    df = pd.read_csv(wheat_path)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df = df.sort_values('date')

    dates = df['date'].values
    prices = df['Close'].values
    print(f"Loaded {len(prices)} daily wheat price samples.")
    return dates, prices


# ===========================================================================
# Main: Run & Compare All Four Decomposition Models
# ===========================================================================
if __name__ == "__main__":
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    print("Random seed set to 42")

    print("=" * 70)
    print("Decomposition-ELM Comparison - VIP Abstract Class Compliant")
    print("=" * 70 + "\n")

    data = load_daily_wheat_prices()
    if data is None:
        print("Failed to load data.")
        exit(1)

    dates, prices = data

    # ----- Define models -----
    models_config = [
        ("EMD",     EMDELMRegressor(task_type='regression')),
        ("EEMD",    EEMDELMRegressor(task_type='regression')),
        ("CEEMD",   CEEMDELMRegressor(task_type='regression')),
        ("CEEMDAN", CEEMDANELMRegressor(task_type='regression')),
    ]

    # ----- Cross-validation (decompose-once approach) -----
    tscv = TimeSeriesSplit(n_splits=5)
    comparison_results = {}  # model_name -> list of per-fold R² values
    final_fold_data = {}     # model_name -> (y_true, predictions, dates_val)

    for model_name, model in models_config:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}-ELM  (Constrained Stacking)")
        print(f"{'='*60}")

        # ---- Decompose the FULL signal ONCE (reused across all folds) ----
        decomp_start = time.time()
        print(f"\n  Decomposing full signal ({len(prices)} pts) with {model_name}...")
        imfs = model._decompose(prices)
        print(f"  Decomposed into {imfs.shape[1]} components in {time.time() - decomp_start:.1f}s")

        fold = 0
        fold_r2s = []

        for train_idx, val_idx in tscv.split(np.arange(len(prices))):
            fold += 1
            fold_start = time.time()
            train_end = len(train_idx)
            print(f"\n--- Fold {fold} (train={train_end}, val={len(val_idx)}) ---")

            # Create a fresh model instance for each fold
            model_class = type(model)

            if model_name == "EMD":
                fold_model = model_class(task_type='regression')
            elif model_name == "EEMD":
                fold_model = model_class(
                    task_type='regression', trials=25, noise_width=0.1,
                    n_hidden=200, lasso_alpha=0.005
                )
            elif model_name == "CEEMD":
                fold_model = model_class(
                    task_type='regression', trials=10, noise_width=0.2
                )
            else:  # CEEMDAN
                fold_model = model_class(
                    task_type='regression', trials=10, epsilon=0.005
                )

            # Use pre-computed IMFs (no re-decomposition!)
            predictions, y_test = fold_model.fit_on_imfs(imfs, prices, train_end)

            # Skip burn-in period: the first predictions use AR features that
            # cross the train/test boundary and are unreliable for slow IMFs
            burn_in = fold_model.lookback * 2
            predictions = predictions[burn_in:]
            y_test = y_test[burn_in:]

            # Compute metrics
            min_len = min(len(y_test), len(predictions))
            y_true = y_test[:min_len]
            preds = predictions[:min_len]

            mse = float(np.mean((y_true - preds) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(y_true - preds)))
            ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_residual = np.sum((y_true - preds) ** 2)
            r2 = float(1 - ss_residual / ss_total) if ss_total != 0 else 0.0

            fold_r2s.append(r2)
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  Fold time: {time.time() - fold_start:.1f}s")

            # Save last fold for plotting
            if fold == 5:
                lookback = fold_model.lookback
                # test predictions start at train_end + lookback (AR offset)
                # + burn_in (skip unreliable boundary)
                test_start = train_end + lookback + burn_in
                dates_aligned = dates[test_start: test_start + min_len]

                final_fold_data[model_name] = (
                    y_true, preds, dates_aligned
                )

                # Test save/load
                print("\n  Testing save/load...")
                os.makedirs('models', exist_ok=True)
                save_path = f"models/{model_name.lower()}_elm_daily.pkl"
                fold_model.save(save_path)

                loaded_model = model_class(task_type='regression')
                loaded_model.load(save_path)
                print(f"  Model saved and loaded successfully.")

        avg_r2 = np.mean(fold_r2s)
        comparison_results[model_name] = fold_r2s
        print(f"\n>> {model_name}-ELM Average R²: {avg_r2:.4f}")

    # ----- Summary Table -----
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'Fold1':>8} {'Fold2':>8} {'Fold3':>8} {'Fold4':>8} {'Fold5':>8} {'Average':>10}")
    print("-" * 70)

    # Prepare data for CSV
    summary_data = []

    for name, r2s in comparison_results.items():
        avg_r2 = np.mean(r2s)
        row = f"{name + '-ELM':<12}"
        for r2 in r2s:
            row += f" {r2:8.4f}"
        row += f" {avg_r2:10.4f}"
        print(row)
        
        # Collect for CSV
        row_dict = {'Model': name}
        for i, val in enumerate(r2s):
            row_dict[f'Fold{i+1}'] = val
        row_dict['Average'] = avg_r2
        summary_data.append(row_dict)

    print("=" * 70)

    # Save to CSV
    try:
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df.set_index('Model', inplace=True)
            df.to_csv('decomposition_metrics.csv')
            print(f"Metrics saved to decomposition_metrics.csv")
            
            # Print Winner
            winner_idx = df['Average'].idxmax()
            winner_val = df.loc[winner_idx, 'Average']
            print(f"\n>> 🏆 WINNER: {winner_idx}-ELM with Average R²: {winner_val:.4f}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    print("=" * 70)

    # ----- Comparison Plot (Last Fold) -----
    if final_fold_data:
        n_models = len(final_fold_data)
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 5 * n_models), sharex=True)

        if n_models == 1:
            axes = [axes]

        colors = ['green', 'orange', 'red', 'purple']

        for idx, (name, (y_true, preds, d)) in enumerate(final_fold_data.items()):
            ax = axes[idx]
            # Ensure all arrays have the same length
            plot_len = min(len(y_true), len(preds), len(d))
            y_true_p = y_true[:plot_len]
            preds_p = preds[:plot_len]
            d_p = d[:plot_len]
            # Show last 300 points for readability
            start = max(0, plot_len - 300)
            ax.plot(
                pd.to_datetime(d_p[start:]), y_true_p[start:],
                label='Actual', color='blue'
            )
            ax.plot(
                pd.to_datetime(d_p[start:]), preds_p[start:],
                label=f'{name}-ELM Forecast', color=colors[idx % len(colors)],
                linestyle='--', alpha=0.85
            )
            r2_val = comparison_results[name][-1]  # Last fold R²
            ax.set_title(f'{name}-ELM  |  Fold 5 R² = {r2_val:.4f}')
            ax.set_ylabel('Wheat Price')
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.savefig('decomposition_comparison.png', dpi=150)
        print("\nComparison plot saved to decomposition_comparison.png")
        plt.close(fig)
