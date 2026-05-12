# The Danger of "Apples-to-Apples" Defaults in EMD Variants

You've made an excellent observation: **setting the noise value to `0.05` across EEMD, CEEMD, and CEEMDAN does not result in an apples-to-apples comparison.** Even worse, a naive default like `0.05` can actually make EEMD perform worse than regular EMD.

Here is why this happens, and why these three parameters—though they sound similar—interact with the algorithms in fundamentally different ways.

---

## 1. Why `0.05` Ruined EEMD (`noise_width`)

In standard **EEMD**, the parameter `noise_width=0.05` defines the standard deviation of the white noise added to the signal.

**The Problem:** EEMD relies purely on the Law of Large Numbers (averaging) to cancel out that noise at the end. The standard rule of thumb is that the residual noise left over in the final IMFs is proportional to:
$$\frac{\text{noise\_width}}{\sqrt{\text{trials}}}$$

If you set `noise_width = 0.05` (meaning you are adding a fairly loud 5% noise background to your signal) but you only run, say, `trials = 100`, the residual noise left behind is $\frac{0.05}{\sqrt{100}} = \frac{0.05}{10} = 0.005$.

This means your "cleaned" IMFs still contain a significant amount of the random noise you injected.
When your ELM or Linear model tries to forecast this, it gets severely confused by that residual artificial noise, leading to worse performance than standard EMD (which has mixing, but at least doesn't have artificial noise).

* **To fix EEMD:** You must drastically increase the number of `trials` (e.g., to 500 or 1000) whenever you use a high `noise_width`, which makes the model painfully slow.

---

## 2. Why CEEMD Rescues the `0.05` (`noise_scale`)

In **CEEMD**, the parameter is `noise_scale=0.05`.

**The Fix:** CEEMD adds noise in perfectly opposed positive and negative pairs. This forces the mathematical cancellation of the noise to happen almost immediately, regardless of the number of trials.

Because the noise cancels out cleanly, the `0.05` scale effectively forces the frequencies to separate (preventing mode mixing) *without* leaving mathematical garbage behind. This is why CEEMD is often the "sweet spot" in empirical testing: you get the frequency-separation benefits of EEMD, but without the high residual noise penalty that damages the ELM forecaster.

---

## 3. Why CEEMDAN with `epsilon=0.05` Behaves Differently

In **CEEMDAN**, the parameter `epsilon=0.05` sounds identical, but it is applied in a completely different mathematical context.

**The Difference:**

* In EEMD and CEEMD, the noise is added to the *original signal* once, at the very beginning.
* In CEEMDAN, the noise is added **at every stage** of the extraction. Before calculating IMF 2, CEEMDAN calculates the standard deviation of the *residual* (what's left of the signal) and then injects noise scaled by `epsilon` multiplied by that standard deviation.

**Why CEEMDAN might perform worse than CEEMD:**
Because CEEMDAN injects fresh noise sequentially at every layer, setting `epsilon=0.05` often turns out to be way too high for the deeper, slower-moving IMFs. As the signal gets smoother (because you are peeling away the high frequencies), adding 5% noise to that already-smooth residual causes the algorithm to create "spurious modes"—fake IMFs that are purely mathematical artifacts of the CEEMDAN noise injection process.

When you feed these fake, noisy artifact IMFs into your Ensemble ELM, the ELM tries to find patterns in pure randomness, leading to poorer cross-validation results compared to the cleaner outputs of CEEMD.

---

### Conclusion for your Model Comparisons

Your feeling that this isn't an apples-to-apples comparison is mathematically correct.

* `noise_width=0.05` in EEMD is often **too loud** for small ensemble sizes, destroying accuracy.
* `noise_scale=0.05` in CEEMD is the **safest default**, offering the best balance of scale separation and clean cancellation.
* `epsilon=0.05` in CEEMDAN is often **too aggressive** for the deeper layers. For an apples-to-apples test, CEEMDAN usually requires a much smaller `epsilon` (e.g., `0.01` or `0.005`) to prevent the creation of spurious modes.
