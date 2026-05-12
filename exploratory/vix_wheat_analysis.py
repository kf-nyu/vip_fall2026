# -------------------------------------------------
# VIX ↔ Wheat (and cross-assets) — exploratory correlation
# -------------------------------------------------
# Not used in headline VIP pipelines (ARX / TCN / FAVAR / ELM).
# "wheat" here is WEAT (ETF), not continuous SRW futures (ZW) in data/wheat-futures/.
# Requires network; pip install yfinance pandas numpy
# Run from repo root: ./venv/bin/python exploratory/vix_wheat_analysis.py
# -------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np


def main() -> None:
    # ------------------------------------------------------------------
    # Download aligned price data
    # ------------------------------------------------------------------
    symbols = {
        "wheat": "WEAT",  # wheat ETF (not ZW=)
        "corn": "ZC=F",
        "soy": "ZS=F",
        "oil": "CL=F",
        "dxy": "DX-Y.NYB",
        "gold": "GC=F",
        "euronext_wheat": "EBM",
        "nikkei": "NK=F",
        "vix": "^VIX",
    }

    raw_data = yf.download(list(symbols.values()), start="2000-01-01")
    if isinstance(raw_data, pd.DataFrame) and isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data["Close"]
    else:
        data = raw_data["Close"] if "Close" in raw_data.columns else raw_data

    valid_cols = [col for col in data.columns if not data[col].isna().all()]
    if not valid_cols:
        raise ValueError("No valid price data downloaded. Check ticker symbols.")

    data = data[valid_cols]
    ticker_to_name = {v: k for k, v in symbols.items()}
    data.columns = [ticker_to_name.get(col, col) for col in data.columns]
    data = data.ffill().dropna()

    # ------------------------------------------------------------------
    # Log returns
    # ------------------------------------------------------------------
    returns = np.log(data / data.shift(1)).dropna()

    # ------------------------------------------------------------------
    # Same-day correlation & covariance
    # ------------------------------------------------------------------
    corr = returns.corr()
    cov = returns.cov()

    print("=== Same-day Correlation ===")
    print(corr)
    print("\n=== Same-day Covariance ===")
    print(cov)

    # ------------------------------------------------------------------
    # Lagged correlation (VIX leading / lagging assets)
    # ------------------------------------------------------------------
    assets = [col for col in returns.columns if col != "vix"]
    lag_corr = {}
    lags = range(-5, 6)

    for lag in lags:
        lag_corr[lag] = {
            asset: returns[asset].corr(returns["vix"].shift(lag)) for asset in assets
        }

    lag_corr_df = pd.DataFrame(lag_corr).T
    print("\n=== Lagged Correlation (Assets vs. VIX) ===")
    print(lag_corr_df)

    def suggest_features_df(df: pd.DataFrame, lag_thr: float = 0.15) -> None:
        for asset in df.columns:
            strong = df[asset][abs(df[asset]) > lag_thr]
            if not strong.empty:
                print(f"\nStrong lagged signals for {asset.upper()} (|corr| > {lag_thr}):")
                print(strong)
            else:
                print(
                    f"\nNo strong lagged correlation for {asset.upper()} exceeds {lag_thr}"
                )

    suggest_features_df(lag_corr_df)

    returns["vix_lag2"] = returns["vix"].shift(2)
    print("\nAdded lagged VIX return column (lag=2). First rows:")
    print(returns.head())


if __name__ == "__main__":
    main()
