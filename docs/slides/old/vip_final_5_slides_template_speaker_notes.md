# Speaker Notes: VIP Final Presentation

## Slide 1: Title

This final presentation summarizes the latest progress after the earlier alternative-data presentation. The focus is now the full path from alternative data and PCA-augmented TCN modeling to probabilistic strategy support and a preliminary investment-strategy proxy. The key framing is cautious: this is an empirical forecasting and strategy-evaluation study, not a production trading system.

## Slide 2: Alternative Data Recap

This slide recaps the prior alternative-data work. At that point, the idea was to improve wheat direction prediction by adding cross-asset and macro features such as oil, gold, DXY, soybeans, VIX, and FRED-MD PCA factors. The problem was that the model was still being tuned and the validation results did not clearly beat the baseline TCN. The main lesson from that stage was that adding more data sources does not automatically create a stronger signal; the data must be economically motivated, cleanly aligned, and evaluated without leakage.

## Slide 3: Alternative Data Update

The update this time is a narrower and more disciplined alternative-data design. Instead of expanding broadly, I focused on selected cross-asset returns that have a plausible economic connection to wheat: corn and soybeans as related agricultural markets, UUP as a U.S. dollar proxy, and CAD/USD as a North American export and FX channel. The Pearson correlations supported this story directionally: corn and soy had positive correlations with wheat, while UUP and CAD/USD had modest negative correlations. These features were transformed into one-day log returns, combined with seven wheat-derived features, standardized only inside the training folds, reduced with PCA, and then passed into the TCN. The result was a small improvement for the narrower PCA track, but the broader macro-plus-cross-asset PCA track underperformed. My interpretation is that the real progress is a cleaner ablation, not a strong claim that alternative data solved the forecasting problem.

## Slide 4: Why MCMC

The motivation for MCMC came from the investment-strategy layer. A strategy does not only need a hard up-or-down label; it needs probabilities, thresholds, and some understanding of uncertainty. The TCN can provide predicted probabilities, but a single probability does not show how uncertain the model is. The Bayesian logistic MCMC model supplements the TCN by producing posterior-mean probabilities and uncertainty diagnostics on a flattened wheat-PCA design. I do not present MCMC as the champion forecasting model. Its role is to support calibration, uncertainty analysis, and threshold-aware long-flat strategy rules.

## Slide 5: Investment Strategy

The investment strategy is intentionally simple and preliminary. I retrain the TCN_PCA_WHEAT_ONLY model on the train-plus-validation period only, then score the terminal 15 percent holdout with predicted probability of an up day. The rules convert those probabilities into long-flat exposure: stay long only when the probability clears a threshold, otherwise stay flat. Buy-and-hold stays long every day. On this bearish holdout window, buy-and-hold lost about 46.8 percent, while the long-flat rules reduced exposure and drawdown by sitting out part of the decline. This should be interpreted as a risk-management result on one holdout tail, not proof of alpha. The proxy still excludes commissions, rolls, margin, slippage, storage, and live execution timing.

## Slide 6: Discussion and Conclusion

The main conclusion is that daily wheat direction forecasting remains a weak-signal problem. The TCN and PCA-augmented TCN produced the best descriptive holdout results, but the improvement was modest and should not be overclaimed. The project taught me that chronological discipline is more important than model complexity: train-only scaling, leakage-safe PCA, and honest holdout evaluation are essential. Selective alternative data helped more than broad data expansion, but more data did not automatically improve performance. MCMC was useful as a supplement because it connected probabilities and uncertainty to the strategy layer. Overall, the project became a reproducible and cautious ML forecasting evaluation rather than a claim of easy trading alpha.

## Slide 7: Future Work

If this work continues next semester, I would extend both the strategy side and the probabilistic modeling side. On the strategy side, I would test rules beyond long-flat, including long-short, volatility-scaled exposure, confidence-weighted sizing, and stop-loss or take-profit rules. I would also add transaction-aware testing with commissions, slippage, bid-ask spread, contract rolls, margin, and contract multipliers. On the validation side, I would use expanding-window walk-forward tests so the same rules are evaluated across multiple market regimes. On the Bayesian side, I would let posterior uncertainty affect position size, compare TCN probabilities with Bayesian and ensemble probabilities, and use posterior predictive draws to stress-test downside risk and probability of loss. A later extension could also use regime-aware Bayesian models where parameters or thresholds differ across high-volatility, low-volatility, and macro-stress periods.

## Slide 8: Thank You

Thank you. I am happy to discuss the alternative-data choices, the PCA-augmented TCN results, the MCMC probability layer, or how the preliminary strategy proxy could be made more realistic in a follow-on term.
