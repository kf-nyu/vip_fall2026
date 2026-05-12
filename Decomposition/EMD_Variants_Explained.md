# Empirical Mode Decomposition (EMD) Variants

Here is a breakdown of the differences between the four Empirical Mode Decomposition (EMD) variants, both in terms of their theoretical fundamentals and how they are implemented in the code within the `Decomposition_Transformers.py` library.

## Base Interface: `BaseDecomposer`

Before diving into the specific variants, it's helpful to understand the interface they all implement. In the VIP framework, `BaseDecomposer` is an abstract base class that ensures all decomposition methods have a standard way to process a time series and return Intrinsic Mode Functions (IMFs).

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseDecomposer(ABC):
    @abstractmethod
    def decompose(self, series: np.ndarray) -> np.ndarray:
        """
        Decomposes a 1D time series into its constituent IMFs.
        Args: 
            series: 1D numpy array of shape (n_samples,) representing the time series.
        Returns: 
            imfs: 2D numpy array of shape (n_imfs, n_samples) containing the separated modes.
        """
        pass
    @abstractmethod
    def get_name(self) -> str:
        """Returns a string identifier for the decomposer method."""
        pass
```

### Inheritance Chain: `ABC -> BaseDecomposer -> [Specific Decomposer]`

* **`ABC` (Abstract Base Class from Python's `abc` module):** This is the root. By inheriting from `ABC` and using the `@abstractmethod` decorator, Python enforces that any class inheriting from `BaseDecomposer` *must* provide its own working version of `decompose()` and `get_name()`. You cannot instantiate a `BaseDecomposer` directly.
* **`BaseDecomposer`:** This defines the "contract". Any forecasting model in the system (like `GenericDecompositionModel`) knows that if it is handed *any* object that is a `BaseDecomposer`, it can safely call `.decompose(series)` on it and expect a 2D array of IMFs in return.
* **Specific Decomposers (e.g., `CEEMDANDecomposer`):** These are the concrete classes. They inherit from `BaseDecomposer`, fulfilling the contract by bringing in the specific math/algorithm (from the `PyEMD` library) to actually perform the decomposition.

## 1. EMD (Empirical Mode Decomposition)

**Fundamentals:**

* **The Baseline:** EMD is the original algorithm. It is a data-driven method that breaks down a complex, non-stationary time series into simpler oscillatory components called Intrinsic Mode Functions (IMFs).
* **The Problem:** EMD suffers from a major issue known as **"mode mixing"**. This happens when intermittent signals or noise cause a single IMF to contain signals of widely disparate scales, or when signals of the same scale are split across multiple IMFs. This makes the IMFs lose their physical meaning.

**Code Implementation:**

* It is the simplest implementation. It just instantiates the base algorithm and runs it. No noise or ensembles are involved.

```python
from PyEMD import EMD

class EMDDecomposer(BaseDecomposer):
    def decompose(self, series: np.ndarray) -> np.ndarray:
        decomposer = EMD()
        imfs = decomposer.emd(series.flatten())
        return imfs.T # Returns basic separated components
```

## 2. EEMD (Ensemble EMD)

**Fundamentals:**

* **The Fix:** EEMD was created to solve the "mode mixing" problem of standard EMD.
* **How it works:** It creates an "ensemble" (a collection) by adding different realizations of finite white noise to the original signal multiple times. EMD is performed on each noise-added corrupted signal. The final IMFs are obtained by averaging the IMFs across all the ensemble trials. The added white noise populates the entire time-frequency space uniformly, forcing the ensemble to separate scales naturally and preventing mode mixing.

**Code Implementation:**

* Requires two critical hyperparameters: `trials` (how many noisy copies to create) and `noise_width` (the standard deviation of the added noise). It is much slower than EMD because it runs the base EMD algorithm `trials` number of times.

```python
from PyEMD import EEMD

class EEMDDecomposer(BaseDecomposer):
    def __init__(self, trials=1000, noise_width=0.01):
        self.trials = trials
        self.noise_width = noise_width

    def decompose(self, series: np.ndarray) -> np.ndarray:
        decomposer = EEMD(trials=self.trials, noise_width=self.noise_width)
        # Adds independent white noise 1000 times, computes EMD 1000 times, and averages
        imfs = decomposer.eemd(series.flatten())
        return imfs.T
```

## 3. CEEMD (Complementary Ensemble EMD)

**Fundamentals:**

* **The Fix:** EEMD solves mode mixing, but introduces a new problem: the averaged white noise does not perfectly cancel out unless you run an infinite number of trials. The reconstructed signal contains residual noise.
* **How it works:** CEEMD adds noise in carefully constructed pairs. For every trial where it adds a positive white noise sequence, it creates a complementary trial with the exact inverse (negative) of that noise sequence. Because the noise is added in opposing pairs, it statistically cancels out much faster and more cleanly when averaging the final ensemble, leaving almost zero residual noise in the reconstruction.

**Code Implementation:**

* While PyEMD *has* a native `CEEMDAN` class and `EEMD` class, there is actually no standalone native `CEEMD` class exported in the standard library.
* As a result, our `Decomposition_Transformers.py` library implements the complementary noise logic **manually** using the base `EMD` class. This gives us finer control over the noise injection distribution and ensures the exact positive/negative pairs are deterministic (via a seed).

```python
from PyEMD import EMD

class CEEMDDecomposer(BaseDecomposer):
    def __init__(self, trials=1000, noise_width=0.01, seed=42):
        self.trials = trials
        self.noise_width = noise_width 
        self.seed = seed

    def decompose(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        noise_std = self.noise_width * np.std(series)
        emd = EMD()
        rng = np.random.RandomState(self.seed)

        all_imfs_list = []
        # Run complementary noise pairs
        for t in range(self.trials):
            noise = rng.normal(0, noise_std, n)
            imfs_pos = emd.emd(series + noise)
            all_imfs_list.append(imfs_pos)
            
            imfs_neg = emd.emd(series - noise)
            all_imfs_list.append(imfs_neg)

        # Pad differing IMF counts and average across all trials
        max_n = max(im.shape[0] for im in all_imfs_list)
        avg_imfs = np.zeros((max_n, n))
        for im in all_imfs_list:
            padded = np.zeros((max_n, n))
            padded[:im.shape[0], :] = im
            avg_imfs += padded
            
        avg_imfs /= len(all_imfs_list)
        return avg_imfs.T
```

## 4. CEEMDAN (Complete EEMD with Adaptive Noise)

**Fundamentals:**

* **The Fix:** CEEMD is great, but the averaging process can sometimes result in IMFs that don't strictly meet the mathematical definition of an IMF, and computational cost is very high.
* **How it works:** CEEMDAN is the most advanced variant. Instead of adding noise to the original signal and running the whole decomposition blindly (like EEMD/CEEMD), it calculates the first IMF, averages it, subtracts it from the signal to get a residual, and *then* adds a specific, mathematically scaled amount of noise to the residual before calculating the next IMF. The noise is "adaptive"—it is recalculated at every stage of the decomposition process depending on the signal's remaining variance.

**Code Implementation:**

* It requires defining the adaptive noise scale (`epsilon`). Because it computes the ensemble average *per mode* sequentially rather than end-to-end all at once, it achieves a "complete" decomposition with fewer trials required to converge, but the core loops are highly complex.

```python
from PyEMD import CEEMDAN

class CEEMDANDecomposer(BaseDecomposer):
    def __init__(self, trials=50, epsilon=0.001):
        self.trials = trials
        self.epsilon = epsilon # Modulates the adaptive noise added sequentially at each IMF extraction stage

    def decompose(self, series: np.ndarray) -> np.ndarray:
        decomposer = CEEMDAN(trials=self.trials, epsilon=self.epsilon)
        imfs = decomposer.ceemdan(series.flatten())
        return imfs.T
```

### Summary of the Evolution

1. **EMD**: The pure algorithm. Fast, but mixes overlapping signal frequencies.
2. **EEMD**: Brute-force adds random noise to separate frequencies. Prevents mixing but leaves fuzzy noise residue.
3. **CEEMD**: Adds perfectly opposed positive/negative noise pairs. Prevents mixing and cleanly eliminates the noise residue.
4. **CEEMDAN**: Intelligently scales and injects noise layer-by-layer during extraction. The most mathematically rigorous and robust extraction.

---

## References

1. **EMD** - Empirical Mode Decomposition (Huang et al., 1998)
2. **EEMD** - Ensemble EMD (Wu & Huang, 2009)
3. **CEEMD** - Complementary Ensemble EMD (Yeh et al., 2010)
4. **CEEMDAN** - Complete EEMD with Adaptive Noise (Torres et al., 2011)
