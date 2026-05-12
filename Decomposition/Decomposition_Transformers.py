"""
Decomposition Transformers Library
==================================
Pluggable decomposition classes that inherit from BaseDecomposer.
Each class uses the PyEMD library to break down a 1D time series signal
into its Intrinsic Mode Functions (IMFs).

These conform to a simple `decompose(series)` interface, returning an 
array of shape (n_samples, n_imfs).
"""

from abc import ABC, abstractmethod
import numpy as np
from PyEMD import EMD as PyEMD_EMD, EEMD as PyEMD_EEMD, CEEMDAN as PyEMD_CEEMDAN


class BaseDecomposer(ABC):
    """
    Base API for Decomposition Preprocessing Plugins.
    """
    @abstractmethod
    def decompose(self, series: np.ndarray) -> np.ndarray:
        """
        Decomposes a 1D timeseries into IMFs.
        
        Args:
            series (np.ndarray): 1D array of shape (N,)
            
        Returns:
            np.ndarray: 2D array of shape (N, n_imfs)
        """
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the decomposition method."""
        pass


class EMDDecomposer(BaseDecomposer):
    """
    Standard Empirical Mode Decomposition (Huang et al., 1998)
    """
    def decompose(self, series: np.ndarray) -> np.ndarray:
        emd = PyEMD_EMD()
        imfs = emd.emd(series) # Returns (n_imfs, N)
        return imfs.T # Transpose to (N, n_imfs)
        
    def get_name(self) -> str:
        return "EMD"


class EEMDDecomposer(BaseDecomposer):
    """
    Ensemble Empirical Mode Decomposition (Wu & Huang, 2009)
    Uses a noise-assisted approach to alleviate mode mixing.
    """
    def __init__(self, trials: int = 25, noise_width: float = 0.1):
        self.trials = trials
        self.noise_width = noise_width
        
    def decompose(self, series: np.ndarray) -> np.ndarray:
        eemd = PyEMD_EEMD(trials=self.trials, noise_width=self.noise_width)
        imfs = eemd.eemd(series)
        return imfs.T
        
    def get_name(self) -> str:
        return "EEMD"


class CEEMDDecomposer(BaseDecomposer):
    """
    Complementary Ensemble EMD (Yeh et al., 2010).
    Adds pairs of positive and negative noise to ensure zero-sum average.
    """
    def __init__(self, trials: int = 1000, noise_width: float = 0.01, seed: int = 42):
        self.trials = trials
        self.noise_width = noise_width 
        self.seed = seed
        
    def decompose(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        noise_std = self.noise_width * np.std(series)
        emd = PyEMD_EMD()
        rng = np.random.RandomState(self.seed)

        all_imfs_list = []

        # Run complementary noise pairs
        for t in range(self.trials):
            noise = rng.normal(0, noise_std, n)
            
            imfs_pos = emd.emd(series + noise)
            all_imfs_list.append(imfs_pos)
            
            imfs_neg = emd.emd(series - noise)
            all_imfs_list.append(imfs_neg)

        # Pad and average across all trials
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
    """
    Complete Ensemble EMD with Adaptive Noise (Torres et al., 2011).
    """
    def __init__(self, trials: int = 20, epsilon: float = 0.001):
        self.trials = trials
        self.epsilon = epsilon
        
    def decompose(self, series: np.ndarray) -> np.ndarray:
        ceemdan = PyEMD_CEEMDAN(trials=self.trials, epsilon=self.epsilon)
        imfs = ceemdan.ceemdan(series)
        return imfs.T
        
    def get_name(self) -> str:
        return "CEEMDAN"
