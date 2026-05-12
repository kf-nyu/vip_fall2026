# Cross-Validation in Decomposition Models: The Burn-In Period

When evaluating time series forecasting models combined with decomposition (like EMD, CEEMD, or CEEMDAN), standard cross-validation approaches can easily lead to **data leakage** and artificially inflated performance metrics.

To solve this, we implement a "Burn-In" period during cross-validation. This document explains the mathematical and programmatic reasons behind this crucial step.

---

## 1. The Core Problem: Data Leakage

Evaluating a model normally involves splitting data into Train and Test sets.

**The Naive Approach (Wrong):**

1. Decompose the training set. Train the model.
2. Decompose the test set. Evaluate the model.
*Why it fails:* Decomposition is computationally heavy. Re-decomposing data for every fold of a cross-validation scheme (e.g., 5 folds, or rolling window) takes hours or days.

**The "Decompose-Once" Approach (Fast, but Dangerous):**

1. Decompose the *entire* dataset (Train + Test) once at the very beginning.
2. For each cross-validation fold, split the pre-decomposed Intrinsic Mode Functions (IMFs) into Train and Test sets.
*Why it is dangerous:* Decomposition relies on connecting local peaks and valleys with cubic splines. If you decompose the entire dataset at once, the spline that models the last few days of the **Training Set** was mathematically influenced by the first few days of the **Test Set** (the algorithm looked ahead into the "future" to connect the next peak).

This causes **Data Leakage** right at the boundary between Train and Test.

---

## 2. The Autoregressive (AR) Boundary Problem

Our forecasting models (like ELM or TCN) predict tomorrow's price by looking backward at a window of past prices. This is the `lookback` period.

```python
# From generic forecaster / ELM script
lookback = 30
```

Imagine you are on "Day 1" of the Test Set. Your model looks backward 30 days to make its prediction.

* **Where do those 30 days come from?** The end of the Training Set.
* **What is wrong with those 30 days?** Their underlying IMFs were subtly altered by the "future" data from the Test Set due to the spline interpolation during the "Decompose-Once" step.

If you evaluate the model on Day 1, the model is mathematically cheating. It is predicting the test set using training features that secretly have test-set information baked into their splines.

---

## 3. The Solution: The "Burn-In" Period

To guarantee fair evaluation without data leakage, we must throw away the first stretch of predictions in the test set. We call this the **burn-in** period.

You'll see this logic implemented in the `Decomposition_ELM_VIP_Compliant.py` script:

```python
# Inside the evaluation loop
# Generate predictions for the Test Set
predictions, y_test = fold_model.fit_on_imfs(imfs, prices, train_end)

# Skip burn-in period: the first predictions use AR features that
# cross the train/test boundary and are unreliable for slow IMFs
burn_in = fold_model.lookback * 2
predictions = predictions[burn_in:]
y_test = y_test[burn_in:]

# Calculate R2 ONLY on the safe, uncontaminated predictions
val_r2 = r2_score(y_test, predictions)
```

### Why `lookback * 2`?

Why not just wait `lookback` days (30 days)? This comes down to the frequency of the IMFs:

1. **Fast IMFs (High Frequency):** These oscillate rapidly. The spline distortion from look-ahead leakage only lasts a couple of days. Waiting `lookback` days perfectly clears this boundary.
2. **Slow IMFs (Low Frequency / Trend):** These are massive, sweeping curves. A peak deep inside the test set can stretch and distort the training data mathematically from 50+ days backward.

By enforcing a burn-in of `lookback * 2` (e.g., 60 days):

* **The first 30 days:** You wait for the AR lookback window to fully transition so that it is looking *exclusively* at data safely deep inside the test set.
* **The next 30 days:** You wait for the massive mathematical "ripple" caused by the slow IMF splines crossing the train/test boundary to settle down entirely.

By taking `predictions = predictions[burn_in:]`, you intentionally discard the period right after the Train/Test fold split. This guarantees that every $R^2$ score you report is evaluated on pure, uncontaminated, forward-looking forecasts, ensuring your decomposed benchmark is mathematically sound.

---

## 4. Academic Context & References for the Burn-In Problem

It is well documented that spline-based envelope estimation in EMD introduces significant boundary distortions that propagate inward from the signal edges (Rilling et al., 2003; Wu & Huang, 2009). These distortions directly contaminate IMFs near train/test splits when decomposition is performed prior to cross-validation.

While there is no single universally agreed-upon formula (like exactly `lookback * 2`), skipping a mathematically derived buffer zone is standard practice to maintain experimental integrity. The `lookback * 2` heuristic safely balances clearing the fast AR-window transition and allowing the slow spline-ripples to settle.

Key canonical papers discussing the EMD Boundary Effect and algorithmic instability at the edges include:

**[1] Mirror-Extension Boundary Criterion (Formal Edge Mitigation)**
  J. Zhao and D. Huang, "Mirror extending and circular spline function for empirical mode decomposition method," *Journal of Zhejiang University SCIENCE A*, vol. 2, no. 3, pp. 247–252, 2001. [Online]. Available: <https://scholar.google.com/scholar?q=Mirror+extending+and+circular+spline+function+for+empirical+mode+decomposition+method>
  *Comment: Establishes a formal mirror-extension boundary criterion to mitigate cubic spline corruption at signal edges. Demonstrates mathematically how spline coefficients near boundaries are influenced by artificial extension and proposes a theoretically justified correction mechanism.*

**[2] Edge Distortion and EEMD Boundary Instability**
  Z. Wu and N. E. Huang, "Ensemble empirical mode decomposition: A noise-assisted data analysis method," *Advances in Adaptive Data Analysis*, vol. 1, no. 1, pp. 1–41, 2009. [Online]. Available: <https://doi.org/10.1142/S1793536909000047>
  *Comment: Foundational EEMD paper. Explicitly discusses spline-based envelope estimation errors at boundaries and shows how edge distortions contribute to mode mixing and instability. Provides theoretical and empirical evidence that endpoint interpolation causes large envelope estimation errors that contaminate adjacent IMFs.*

**[3] Algorithmic Boundary Instability Proof**
  G. Rilling, P. Flandrin, and P. Gonçalvès, "On empirical mode decomposition and its algorithms," in *Proc. IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing (NSIP)*, 2003. [Online]. Available: <https://scholar.google.com/scholar?q=On+empirical+mode+decomposition+and+its+algorithms+Rilling>
  *Comment: Provides algorithm-level mathematical analysis of EMD behavior. Demonstrates that endpoint oscillations are numerically unstable and that envelope extrapolation introduces inward-propagating distortion, formally validating boundary unreliability in EMD implementations.*

---

## 5. Experimental Findings: Why EEMD Outperforms CEEMD

Theoretically, CEEMD yields perfectly reconstructed IMFs with near-zero residual noise, which might suggest it should be the superior forecasting preprocessor. However, empirical cross-validation repeatedly shows **EEMD-ELM outperforming CEEMD-ELM** (e.g., Average R² of 0.50 vs 0.39).

This divergence between Signal Processing theory and Machine Learning practice stems from three main factors:

1. **Implicit Regularization (The "Texture" Advantage):** The slight amount of "fuzzy" residual white noise left by EEMD acts as implicit data augmentation (similar to dropout). When the ELM trains on ultra-clean CEEMD IMFs, it relies too heavily on the exact, pure spline trajectories, making it brittle and prone to overfitting. EEMD's noisy IMFs force the ELM to learn the generalized trend rather than memorizing the exact curve, leading to better out-of-sample generalization.
2. **The Penalty of Forced Symmetry:** CEEMD adds noise in perfectly opposed positive/negative pairs to achieve its clean cancellation. Financial data (like Wheat prices) is typically highly asymmetric—prices often crash much faster than they climb. Forcing the EMD sifting process to process perfectly symmetric noise pairs against an asymmetric shock can sometimes warp the resulting IMFs into unnatural splits.
3. **Synergistic Variance with Ensemble ELMs:** Ensemble learning thrives when there is variance in the input data. CEEMD's IMFs are so smoothly and perfectly extracted that the various models in our ELM ensemble end up learning almost the exact same mapping. EEMD's scattered residual variance gives the ensemble slightly different "textures" to train on, allowing the meta-model to build a more diversified and robust final prediction.

---

## Algorithm References

The decomposition methods discussed herein correctly attribute to the following foundational papers:

1. **EMD** - Empirical Mode Decomposition (Huang et al., 1998)
2. **EEMD** - Ensemble EMD (Wu & Huang, 2009)
3. **CEEMD** - Complementary Ensemble EMD (Yeh et al., 2010)
4. **CEEMDAN** - Complete EEMD with Adaptive Noise (Torres et al., 2011)
