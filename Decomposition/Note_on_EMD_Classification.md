# Note on Decomposition Methods (EMD, EEMD, CEEMD, CEEMDAN) and Classification Tasks

## 1. The Core Mathematical Challenge

Applying pure classification models (which predict "1" for Up, or "0" for Down) directly to individual Intrinsic Mode Functions (IMFs) derived from Empirical Mode Decomposition methods (EMD, EEMD, CEEMD, CEEMDAN) fundamentally breaks the mathematical rules of how decomposition works.

To understand why, we have to look at what decomposition actually does.

### What is an IMF?
When we use EMD/EEMD to decompose a time series (like daily wheat prices), we are splitting the single price line into multiple separate waves, called IMFs (Intrinsic Mode Functions). 

The most important rule of decomposition is that **the original price is always exactly equal to the sum of all the separated waves added together:**
$$ Total Price(t) = IMF_1(t) + IMF_2(t) + ... + IMF_n(t) $$

Because of this, if we want to know how much the *price changed* today, we just add up how much *each individual wave changed* today:
$$ Total Price Change = \Delta IMF_1 + \Delta IMF_2 + ... + \Delta IMF_n $$

## 2. Why Direct Classification Fails on IMFs

If we try to use a Classification model on the individual IMFs, we destroy this mathematical relationship. Here is exactly why:

### A. The Loss of Magnitude (Amplitude)
A classifier only predicts a direction (e.g., `1` for Up, `0` for Down). It does **not** predict *how much* it went up or down. 

If we apply a classifier to an IMF, we are converting a continuous number (like "$+2.50") into a simple label (like "Up"). By doing this, we completely destroy the **magnitude** (the size or amplitude) of that wave's movement. We know the wave moved up, but we forgot how strong the wave was.

### B. The "Cancellation" or "Voting" Problem
Because we lose the magnitude, the model cannot correctly combine the different waves together to figure out the final price direction.

Let's look at a concrete example. Imagine our decomposition gives us two waves for a specific day:
* **Wave 1 (High-frequency noise):** It flickers upward by a tiny amount: **+\$0.05**.
* **Wave 5 (Long-term trend):** It crashes downward by a massive amount: **-\$4.00**.

If we just add these actual numbers together ($0.05 - 4.00$), the true total price change is **-\$3.95**. The market clearly went **Down**.

But what happens if we use Classifiers on the IMFs instead of exact numbers?
* **Wave 1 Classifier** sees a $+\$0.05$ change. It predicts **"Up" (1)**.
* **Wave 5 Classifier** sees a $-\$4.00$ change. It predicts **"Down" (0)**.

Now, we have one model voting "Up" and one model voting "Down". To an ensemble or a meta-classifier trying to combine these predictions, these two forces look **exactly equal**. The model does not know that Wave 5 is 80 times stronger than Wave 1, because the classifiers threw away the magnitude. The tiny $+0.05$ noise tick is treated with the exact same voting power as the massive $-4.00$ trend drop. 

Because of this, the classifiers will mathematically "cancel" each other out, leading to highly inaccurate and confused overall directional predictions.

## 3. The Required Solution: Intermediate Regression

Because we absolutely *must* know the exact sizes (magnitudes) of the waves to add them together correctly, **we are forced to use Regression models on the individual IMFs.**

The pipeline must behave like this to be mathematically sound:
1. **Regressors** (like `TCNRegressor`, `ARXModel`, or `ELMRegressor`) are trained on each IMF. Instead of predicting "Up/Down", they predict the exact continuous number value (e.g., predicting the wave will move exactly $-3.80$).
2. We **mathematically sum** all of these continuous predictions together ($Prediction 1 + Prediction 2 + ...$) to reconstruct what we expect the total $\Delta Price$ to be.
3. Finally, at the very end of the pipeline, **we apply a Classification Threshold**. If our reconstructed total prediction is greater than 0 (or a calibrated threshold), we classify the final output as **"Up" (1)**. Otherwise, we classify it as **"Down" (0)**.

### Conclusion for the Implementation
This mathematical constraint strictly dictates our architectural design: the backend models processing the decomposed IMFs *must* be regressors to preserve wave amplitudes, even if the ultimate goal of the forecasting pipeline is binary classification.

### The Required Solution: Intermediate Regression

Because the structural integrity of the prediction relies on the exact arithmetic summation of the IMFs' physical magnitudes, **we must use regression models on the individual IMFs.**

The operation must follow this sequence:
1. **Regressors** (e.g., `TCNRegressor`, `ARXModel`, `ELMRegressor`) predict the continuous $\Delta$ value of each distinct $IMF$.
2. We mathematically **sum** these continuous predictions to reconstruct the expected overall $\Delta P(t)$.
3. Finally, **we apply a classification threshold post-hoc** to the reconstructed continuous signal (e.g., IF $\sum \Delta IMF > Threshold$ THEN Class $1$, ELSE Class $0$).

This constraint dictates our architectural design: the backends processing the decomposed IMFs *must* be regressors, even if the ultimate objective of the forecasting pipeline is binary classification.
