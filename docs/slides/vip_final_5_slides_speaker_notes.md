# Speaker Notes: 5-Slide VIP Final Presentation

## Slide 1: Direction Forecasting on Daily U.S. Wheat Futures

This project asks a deliberately modest forecasting question: can public data help predict whether tomorrow's wheat futures close-to-close move is up or down? I focused on direction rather than exact price because direction can feed simple tactical decisions, such as staying long or stepping aside when the predicted probability is weak. The key framing is that this is an empirical evaluation study, not a production trading system or investment recommendation.

## Slide 2: Data Pipeline and Leakage Controls

The main data sources are daily Chicago SRW wheat futures, technical indicators from the price history, and FRED-MD macroeconomic indicators. The macro data required special care because monthly macro releases cannot be treated as if they were known daily in real time, so I applied stationarity transformations and a one-month publication-style lag. I also tested optional cross-asset information such as corn, soybeans, UUP, and CAD. The most important engineering lesson was leakage control: every scaler, PCA step, and model-selection step has to be fit only on data that would have been available at that point in the timeline.

## Slide 3: Models and Main Forecasting Results

I compared transparent baselines and nonlinear models under the same chronological evaluation design. The families were FAVAR-style factor models, ridge ARX, ELM, and TCN classifiers/regressors. The TCN classifier had the best terminal holdout accuracy at about 54.1 percent and the best macro F1, but the cross-validation folds stayed close to 50 percent. That means the right interpretation is a small descriptive lift, not proof of a stable or statistically significant edge.

## Slide 4: Most Recent Work: Probabilities to a Long--Flat Overlay

The most recent extension connected predicted probabilities to a preliminary investment-strategy proxy. Instead of assuming the forecast is directly profitable, I mapped the TCN probability into long--flat rules and compared those rules with buy-and-hold on the same holdout period. On this bearish holdout tail, buy-and-hold lost about 47 percent, while some long--flat rules reduced drawdowns by being out of the market. This is only a frictionless proxy, because it does not include commissions, bid-ask spreads, margin, rolls, options, or live execution timing. The Bayesian MCMC model contributes by giving posterior probability estimates that could support uncertainty-aware thresholds.

## Slide 5: Lessons Learned and What I Would Add

The biggest lesson is that honest time-series evaluation is harder and more important than adding a more complicated model. Alternative data and PCA can make the feature set more structured, but they do not automatically create a strong signal. AI and ML helped with model comparison, probability estimation, and reproducible experimentation, but the result is still a weak-signal forecasting problem. With more time, I would add an expanding-window walk-forward test, nested tuning, FRED-vintage sensitivity checks, transaction costs, and futures roll logic so the strategy result becomes closer to a real trading backtest.
