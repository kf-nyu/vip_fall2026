# Speaker Notes: VIP Final Presentation

## Slide 1: Title

This final presentation summarizes the most recent stage of the VIP wheat forecasting project. The focus is on the update from the earlier alternative-data work toward a cleaner PCA-augmented TCN pipeline, a Bayesian MCMC probability layer, and a preliminary investment-strategy proxy. The key framing is cautious: this is an empirical machine-learning forecasting study and strategy evaluation, not a production trading system or investment recommendation.

## Slide 2: Alternative Data Recap

This slide recaps the earlier alternative-data presentation. At that point, the goal was to improve wheat direction prediction by adding cross-asset and macro features such as oil, gold, DXY, soybeans, VIX, and FRED-MD PCA factors. The pipeline used feature engineering, macro compression through PCA, sliding windows, and a TCN classifier. The important result was that the approach still could not clearly beat the baseline TCN. That was a useful negative result: it showed that broad alternative data and more features do not automatically create better directional signal.

## Slide 3: Alternative Data Update and Holdout Table

This slide shows the update from the earlier work. I narrowed the alternative-data design to four economically motivated daily return series: corn and soybeans as related agricultural markets, UUP as a U.S. dollar proxy, and CAD/USD as a North American export and FX channel. The Pearson correlations help explain the selection: corn and soy had positive correlations with wheat, while UUP and CAD/USD had modest negative correlations. The final PCA track used seven wheat-derived features plus those four cross-asset returns, with train-fold-only scaling and PCA before the TCN.

The table gives the main result. The baseline TCN reference reached 0.541 holdout accuracy. The narrower `TCN_PCA_WHEAT_ONLY` pipeline reached 0.543, which is only a small descriptive improvement. The broader `TCN_PCA` version, which added the Yahoo overlay and macro PCA block, underperformed at 0.482. My interpretation is that the useful progress is not a claim that alternative data solved the problem, but a cleaner and more defensible ablation showing that selective data worked better than broad expansion.

## Slide 4: Why MCMC

The motivation for MCMC came from the investment-strategy layer. A strategy does not only need a hard up-or-down label; it needs probabilities, thresholds, and some understanding of uncertainty. The TCN can rank days and output probabilities, but a single probability does not show how uncertain the model is. The Bayesian logistic MCMC model supplements the TCN by producing posterior-mean probabilities and uncertainty diagnostics on a flattened wheat-PCA design.

The table compares a standard maximum-likelihood logistic model with the Bayesian posterior-mean probability model. The Bayesian version had 0.542 accuracy and a slightly better Brier score, while the AUC stayed close to coin-flip. I do not present MCMC as the champion forecaster. Its role is to support calibration, uncertainty analysis, and threshold-aware long-flat rules.

## Slide 5: Investment Strategy

This slide explains the preliminary investment-strategy proxy. I retrain `TCN_PCA_WHEAT_ONLY` on the train-plus-validation period only, then score the terminal 15 percent holdout with predicted probability of an up day. The rules convert those probabilities into long-flat exposure: go long only when the probability clears a threshold, otherwise stay flat. Buy-and-hold stays long every day.

The results are best read as a risk-management comparison on one bearish holdout window. Buy-and-hold lost about 46.8 percent, while the long-flat rules reduced exposure by sitting out part of the decline. The validation-threshold rule was long about 42 percent of the time and had a smaller maximum drawdown than full exposure. This is not proof of alpha. It is a frictionless proxy that still excludes commissions, contract rolls, margin, options, slippage, storage, and live order timing.

## Slide 6: Discussion and Conclusion

The main conclusion is that daily wheat direction forecasting remains a weak-signal problem. The TCN and PCA-augmented TCN produced the best descriptive holdout results, but the improvement was modest. Chronological discipline mattered more than model complexity: train-only scaling, leakage-safe PCA, and honest holdout evaluation were essential. Selective alternative data helped more than broad data expansion, but more data did not automatically improve performance.

The investment-strategy work reinforced the same lesson. Model accuracy alone is not enough; probabilities, calibration, and uncertainty matter when forecasts are mapped into positions. MCMC helped supplement the strategy layer with posterior probabilities and uncertainty. The long-flat strategy reduced exposure during a bearish holdout window, but the result should be interpreted as risk management, not proof of alpha. Overall, the project became a reproducible and cautious ML forecasting evaluation.

## Slide 7: Future Work

If this work continues next semester, I would extend both the strategy side and the probabilistic modeling side. On the strategy side, I would test rules beyond long-flat, including long-short, volatility-scaled exposure, confidence-weighted sizing, and stop-loss or take-profit rules. I would also add transaction-aware testing with commissions, slippage, bid-ask spread, futures rolls, margin, and contract multipliers.

On the validation side, I would use expanding-window walk-forward tests so each rule is evaluated across multiple market regimes. On the Bayesian side, I would let posterior uncertainty affect position size, compare TCN probabilities with Bayesian posterior-mean probabilities and ensemble probabilities, and use posterior predictive draws to stress-test downside risk and probability of loss. A later extension could also use regime-aware Bayesian models where parameters or thresholds differ across high-volatility, low-volatility, and macro-stress periods.

## Slide 8: Thank You

Thank you. I am happy to discuss the alternative-data choices, the PCA-augmented TCN results, the Bayesian MCMC probability layer, or how the preliminary long-flat strategy proxy could be made more realistic in a follow-on term.
