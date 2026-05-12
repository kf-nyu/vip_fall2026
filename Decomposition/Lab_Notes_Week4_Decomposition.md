# VIP Lab Notes – Week 4

**Student Name:** Kenji Funaki  
**Date:** February 24, 2026  
**Model:** Decomposition Methods (EMD, EEMD, CEEMD, CEEMDAN) + Ensemble ELM  

## 1. Objectives & Goals

**What are you trying to accomplish in this session?**

* **Overall Goal:** Implement, evaluate, and compare multiple signal decomposition techniques (EMD, EEMD, CEEMD, CEEMDAN) hybridized with an Ensemble ELM and Constrained Stacking meta-learner for forecasting wheat futures prices.
* **Specific Goals:**
  * Transition execution from standalone Python scripts to VIP-Compliant Jupyter Notebooks.
  * Implement standard `BaseForecastModel` methods (`fit`, `predict`, `evaluate`, `save`, `load`) to maintain VIP architectural compliance.
  * Compare base Empirical Mode Decomposition (EMD) against noise-assisted evolutionary variants (EEMD, CEEMD, CEEMDAN).
  * Debug and resolve mathematical performance bottlenecks within the Ensemble ELM backend.
  * Compare final $R^2$ and RMSE scores using a 5-Fold Time Series Cross-Validation framework.

## 2. Approach & Methods

**Describe your implementation strategy, algorithms, or techniques used:**

* **Model Selection Overview:**
  * **Decomposition Front-End:** Time-series decomposition isolates intrinsic mode functions (IMFs) of varying frequencies from the chaotic wheat pricing data. We used the `PyEMD` package.
  * **Autoregressive Backend:** For each IMF, we build a 30-day lag window.
  * **Ensemble ELM Evaluator:** Each IMF's lags are fed into a localized Ensemble ELM (50 separate ELMs using Ridge Regression).
  * **Meta-Learner / Stacking:** A Constrained Lasso or Ridge meta-learner combines the predictions of the IMFs into a final reconstructed price prediction.

* **Variants Implemented:**
    1. **EMD (Empirical Mode Decomposition):** The standard Huang et al. algorithm. Fast, but susceptible to "mode mixing".
    2. **EEMD (Ensemble EMD):** Injects white noise and averages results across multiple trials to smooth out mode mixing. (Configured at 200 trials, 0.01 noise width).
    3. **CEEMD (Complementary Ensemble EMD):** Uses complementary positive and negative noise pairs to ensure the added noise perfectly cancels out in the final average.
    4. **CEEMDAN (Complete EEMD with Adaptive Noise):** Calculates a unique noise profile for each stage of the decomposition process to extract cleaner IMFs.

* **Data Handling Strategy:**
  * Dataset: Chicago SRW Wheat Futures (ZW=F) Close Prices daily indices.
  * Since decomposition runs iteratively over 1D signals, we decompose the *entire* series first, save the intermediate IMFs to disk (`Decomposed_Data/` directory) to save compute time, and then slice the IMFs into sliding feature windows for modeling.

## 3. Implementation Details

**Key functions, classes, or modules implemented:**

* **Key Files and Notebooks:**
  * `Generic_Decomposition_Forecaster.ipynb`: The core interactive VIP notebook comparing all four variants.
  * `Decomposition_Comparative_Analysis.ipynb`: A secondary notebook comparing linear vs ELM backend performance.
  * `VIP_Abstract_Class.ipynb`: The baseline blueprint ensuring uniformity in model functions.
  * `Decomposition_ELM_VIP_Compliant.py`: The robust fallback Python script containing the safe, randomized ELM implementation.

* **Hyperparameters:**
  * `lookback = 30`: 30-day window autoregression for each IMF.
  * `n_denoise = 2`: Drop the first 2 highest-frequency (most chaotic) IMFs to filter out market noise.
  * `n_estimators = 50`: 50 distinct randomized ELMs per IMF ensemble.
  * `n_hidden = 100`: Hidden neurons per ELM.

## 4. Results & Testing

**Test results, output observations, performance metrics:**

* **Cross-Validation Results (5-Fold Time Series Split):**
| Model                             | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average R²|
|-----------------------------------|--------|--------|--------|--------|--------|-----------|
| EMD-ELM                           | -1.8617| 0.6847 | 0.8861 | 0.9660 | 0.9710 | 0.3293    |
| EEMD_t200_nw0.01-ELM              | -1.5878| 0.7357 | 0.8738 | 0.9663 | 0.9426 | 0.3861    |
| CEEMD_t200_nw0.01-ELM             | -1.0105| 0.1333 | 0.8368 | 0.9595 | 0.9757 | 0.3790    |
| CEEMDAN_t50_eps0.005-ELM          | -1.3876|-0.2076 | 0.7804 | 0.9217 | 0.9598 | 0.2133    |

* **Observations:**
  * **Cold Start Penalty:** As with the baseline ELM in Week 3, the early folds (Folds 1 & 2) lack sufficient volume of data to train the Meta-Learner properly, resulting in catastrophic negative $R^2$ values.
  * **Strong Later Performance:** By Folds 4 and 5, the models approach near-perfect interpolation of the testing sequence, consistently hitting $R^2 > 0.95$.
  * **Best Variant:** EEMD offered the best stability and highest average $R^2$ overall (0.3861), though CEEMD had the best Fold 5 closing performance (0.9757). EMD baseline proved surprisingly resilient against the adaptive noise models.

## 5. Issues & Debugging

**Problems encountered, error messages, and solutions attempted:**

**Problem 1: Notebook Environment Path Resolution**

* **Issue:** `NameError: name '__file__' is not defined` when attempting to load the fallback CSVs.
* **Solution:** Replaced `os.path.dirname(__file__)` references with `os.getcwd()` since `__file__` is a Python execution variable unavailable to Jupyter kernels.

**Problem 2: Extreme Hidden Weight Overflow**

* **Issue:** `RuntimeWarning: overflow encountered in exp` continuously printed during the Sigmoid activation of the ELM layer. Caused by massive initialized vector constants pushing the denominator to zero.
* **Solution:** Rewrote the localized `_sigmoid` function in `ELM_Daily_Implementation.py` using `np.where` logic to split paths for negative and positive exponent calculations, stabilizing the operation space for 64-bit numpy floats.

**Problem 3: The Ensemble Cloning Flaw (Performance Degradation)**

* **Issue:** The $R^2$ performance critically degraded when tying the pipeline directly into the base `ELM_Daily_Implementation.py` script instead of using the local `Decomposition_ELM_VIP_Compliant.py` wrapper.
* **Diagnosis:** `np.random.seed(42)` was hardcoded directly inside the core `fit()` method. This meant that across the 50 iteration loops of our `EnsembleELM`, every single ELM generated the exact same random projection matrix. The ensemble functionally collapsed into 50 identical clones of a single ELM, destroying ensemble variance.
* **Solution:** Reverted the backend logic to `Decomposition_ELM_VIP_Compliant.py`, which securely implements `seed=i * 123` inside the loop to ensure mathematical diversity across the ensemble.

**Problem 4: Dynamic Re-Execution for Hyperparameters**

* **Issue:** Changing hyperparameters such as trials or noise width in notebook cells did not rerun the heavy decomposition process, leading to stale model states.
* **Solution:** Rewrote `get_name()` strings for all `BaseDecomposer` overrides to actively map their config states (e.g., `"EEMD_t200_nw0.01"`), guaranteeing the caching mechanism requests a fresh decomposition whenever parameters are tweaked.

## 6. Conclusions & Next Steps

**What did you learn? What needs to be done next?**

* **What We Learned:**
  * Decomposing a non-linear financial time series into IMFs and running individual ELM models per frequency band drastically improves the localized resolution of tracking.
  * Ensemble models *depend* on independent mathematical randomness. If all models are initialized with an identical fixed seed, the entire point of an ensemble (variance reduction) is compromised.
  * Filtering out the high-frequency IMFs (`n_denoise=2`) shields the meta-learner from trying to memorize erratic market noise, yielding a structurally sound forecast.

* **Next (Future) Steps:**
  * **External Meta-Learners:** Expand the stacking mechanism. Currently we use Ridge/Lasso. If we sub in a small neural network or Random Forest for the stacker, we may handle IMF nonlinearities better.
  * **Multivariate Decomposition:** Right now, this framework only works on the 1D Target vector. We need to explore how to apply decomposed modes to the 32 FRED-MD external features for structural coherence.

## 7. References & Resources

* **Documentation:** PyEMD Official Docs / Scikit-Learn Stacking references.
* **Papers:**
  * [Huang et al. 1998 - Empirical Mode Decomposition]
  * [Wu & Huang 2009 - Ensemble EMD]
  * [Yeh et al. 2010 - Complementary EEMD]
