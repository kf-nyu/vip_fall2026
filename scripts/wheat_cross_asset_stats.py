#!/usr/bin/env python3
"""
Daily wheat futures vs cross-asset summary stats and correlations.

Reads local wheat CSV (Close), pulls Yahoo Finance proxies for oil, gold,
dollar, yen, soybeans, and a few optional series, aligns to wheat trading
calendar (reindex + forward-fill like TCN_PCA), then reports Pearson and
Spearman correlations and simple return-moment summaries on log returns.

Usage (from repo root or this package root, with venv active):
  python scripts/wheat_cross_asset_stats.py
  python scripts/wheat_cross_asset_stats.py --wheat data/wheat-futures/wheat_futures_daily.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:  # pragma: no cover
    print("Install yfinance: pip install yfinance", file=sys.stderr)
    raise SystemExit(1) from e


# Yahoo symbols: names match TCN_PCA.py where applicable, plus user requests + extras.
YAHOO = [
    ("oil_wti", "CL=F"),
    ("gold", "GC=F"),
    ("dollar_idx", "DX-Y.NYB"),
    ("usd_jpy", "JPY=X"),  # USD/JPY spot; rises = dollar stronger vs yen
    ("soybeans", "ZS=F"),
    ("vix", "^VIX"),
    # Recommended extras (commodity neighbors + macro risk)
    ("corn", "ZC=F"),
    ("copper", "HG=F"),
    ("sp500", "^GSPC"),
    ("ust10y", "^TNX"),
]

# Extra proxies (bonds, other FX/ETFs, commodities, vol) explored with --extended.
EXTRA_YAHOO = [
    ("tlt_long_bond", "TLT"),
    ("ief_med_treasury", "IEF"),
    ("shy_short_treasury", "SHY"),
    ("hy_credit", "HYG"),
    ("usd_bull_etf", "UUP"),
    ("eur_usd", "EURUSD=X"),
    ("gbp_usd", "GBPUSD=X"),
    ("aud_usd", "AUDUSD=X"),
    ("usd_cad", "CAD=X"),
    ("btc", "BTC-USD"),
    ("yen_etf", "FXY"),
    ("franc_etf", "FXF"),
    ("silver", "SI=F"),
    ("platinum", "PL=F"),
    ("gas_ng", "NG=F"),
    ("gasoline_rb", "RB=F"),
    ("kc_wheat", "KE=F"),  # hard red wheat (KC)
    ("oats", "ZO=F"),
    ("coffee", "KC=F"),
    ("sugar", "SB=F"),
    ("cocoa", "CC=F"),
    ("xfe_dba_ag", "DBA"),
    ("msci_eafa", "EFA"),
    ("em_equity", "EEM"),
    ("utilities", "XLU"),
    ("healthcare", "XLV"),
    ("reits", "XLRE"),
    ("energy_sector", "XLE"),
    ("financials", "XLF"),
    ("lng_vol_etn", "VXX"),
    ("inv_sq500", "SH"),  # short S&P ETF (inverse daily)
]


def load_wheat_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["date"] = df["date"].dt.normalize()
    df = df.sort_values("date").set_index("date")
    s = df["Close"].astype(float)
    s.name = "wheat"
    return s


def download_alt_closes(
    tickers: list[tuple[str, str]], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    symbols = [sym for _, sym in tickers]
    raw = yf.download(symbols, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, threads=True)
    out = {}
    for label, sym in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                raise RuntimeError("Unexpected yfinance shape: no Close level")
            ser = raw["Close"][sym].copy()
        else:
            if sym not in raw.columns:
                ser = pd.Series(index=raw.index, dtype=float)
            else:
                ser = raw[sym].copy()
        ser.index = pd.to_datetime(ser.index, utc=True).tz_convert(None).normalize()
        ser = ser[~ser.index.duplicated(keep="last")]
        out[label] = ser
    return pd.DataFrame(out)


def _uniq_tickers_ordered(
    a: list[tuple[str, str]], b: list[tuple[str, str]] | None
) -> list[tuple[str, str]]:
    if not b:
        return list(a)
    seen_sym: set[str] = set()
    out: list[tuple[str, str]] = []
    for tup in list(a) + list(b):
        _lab, sym = tup
        if sym in seen_sym:
            continue
        seen_sym.add(sym)
        out.append(tup)
    return out


def align_to_wheat_calendar(wheat: pd.Series, alt: pd.DataFrame, *, bfill: bool) -> pd.DataFrame:
    """Reindex alt to wheat index; optionally bfill tail gaps (PCA-style) vs ffill-only (scan)."""
    idx = wheat.index
    alt = alt.reindex(idx).sort_index()
    alt = alt.ffill()
    if bfill:
        alt = alt.bfill()
    return pd.concat([wheat.to_frame(name="wheat"), alt], axis=1)


def align_to_wheat_calendar_ffill_only(wheat: pd.Series, alt: pd.DataFrame) -> pd.DataFrame:
    """Like TCN_PCA but omit bfill so pre-instrument dates stay NaN (pairwise correlations)."""
    return align_to_wheat_calendar(wheat, alt, bfill=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    default_wheat = root / "data" / "wheat-futures" / "wheat_futures_daily.csv"
    if not default_wheat.exists():
        default_wheat = root.parent / "data" / "wheat-futures" / "wheat_futures_daily.csv"
    ap.add_argument("--wheat", type=Path, default=default_wheat, help="Wheat daily CSV path")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "docs",
        help="Directory for CSV outputs",
    )
    ap.add_argument(
        "--extended",
        action="store_true",
        help=f"Merge in {len(EXTRA_YAHOO)} extra Yahoo symbols (bonds/FX/other commods/sectors/vol)",
    )
    ap.add_argument(
        "--min-periods",
        type=int,
        default=750,
        help="Minimum overlapping return days required for pairwise corr (pandas min_periods)",
    )
    args = ap.parse_args()

    if not args.wheat.exists():
        print(f"Missing wheat file: {args.wheat}", file=sys.stderr)
        raise SystemExit(1)

    wheat = load_wheat_csv(args.wheat)
    start, end = wheat.index.min(), wheat.index.max()
    print(f"Wheat rows: {len(wheat):,}  range: {start.date()} .. {end.date()}")

    ticker_list = _uniq_tickers_ordered(YAHOO, EXTRA_YAHOO if args.extended else None)
    alt = download_alt_closes(ticker_list, start, end)
    prices = align_to_wheat_calendar(wheat, alt, bfill=not args.extended)
    missing_share = prices.isna().mean().sort_values(ascending=False)
    if (missing_share > 0).any() and not args.extended:
        print("\nWarning: NaN share by column (before drop):")
        print(missing_share[missing_share > 0].to_string())
    elif args.extended and (missing_share > 0).any():
        top_missing = missing_share[missing_share > 0].head(8)
        print("\nInfo: leading NaNs (pre-instrument) on some symbols — using pairwise correlations:")
        print(top_missing.round(3).to_string())

    if not args.extended:
        prices = prices.dropna(how="any")
    else:
        # Keep wheat trajectory; pairwise corr skips NaNs per column pair (min_periods).
        prices = prices.dropna(subset=["wheat"], how="any")
    print(f"Aligned wheat rows used: {len(prices):,}")

    log_px = np.log(prices)
    rets = log_px.diff()
    if args.extended:
        rets = rets.dropna(subset=["wheat"]).copy()
        print(f"Return rows (wheat non-NaN; other columns pairwise): {len(rets):,}\n")
    else:
        rets = rets.dropna(how="any")
        print(f"Return observations: {len(rets):,}\n")

    mp = args.min_periods
    pearson = rets.corr(method="pearson", min_periods=mp)
    spearman = rets.corr(method="spearman", min_periods=mp)

    ann = 252.0
    summ = pd.DataFrame(
        {
            "mean_daily_pct": (rets.mean() * 100.0),
            "annualized_mean_pct": (rets.mean() * ann * 100.0),
            "stdev_daily_pct": (rets.std(ddof=1) * 100.0),
            "annualized_vol_pct": (rets.std(ddof=1) * np.sqrt(ann) * 100.0),
            "skew": rets.skew(),
            "exc_kurtosis": rets.kurtosis(),  # pandas: Fisher (excess)
        }
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = "_extended" if args.extended else ""
    p_pear = out_dir / f"wheat_cross_asset_corr_pearson{suf}.csv"
    p_spr = out_dir / f"wheat_cross_asset_corr_spearman{suf}.csv"
    p_sum = out_dir / f"wheat_cross_asset_returns_summary{suf}.csv"
    pearson.round(4).to_csv(p_pear)
    spearman.round(4).to_csv(p_spr)
    summ.round(4).to_csv(p_sum)
    print(f"Wrote:\n  {p_pear}\n  {p_spr}\n  {p_sum}\n")

    w = pearson["wheat"].drop(labels=["wheat"]).sort_values(key=abs, ascending=False)
    print("Pearson correlation of wheat log-return with other log-returns (sorted by |ρ|):")
    print(w.round(4).to_string())
    print("\nSame (Spearman rank):")
    print(spearman["wheat"].drop(labels=["wheat"]).sort_values(key=abs, ascending=False).round(4).to_string())
    print("\nReturn summary (mean in % per day; vol annualized %):")
    print(summ.round(3).to_string())

    if args.extended:
        base_labs = {lab for lab, _ in YAHOO}
        wcol = pearson["wheat"].dropna().drop(labels=["wheat"], errors="ignore")
        neg = wcol[wcol < 0].sort_values()
        neg_new = neg.loc[[lab for lab in neg.index if lab not in base_labs]]
        print("\n--- Extended scan: NEW tickers vs base basket ---")
        print(f"(Pearson vs wheat returns; min_periods={mp})\nNegative ρ excluding base-panel names:")
        if neg_new.empty:
            print("  (none)")
        else:
            print(neg_new.round(4).to_string())
        p_neg = out_dir / "wheat_pearson_negative_extended.csv"
        neg_all = wcol[wcol < 0].sort_values()
        neg_all.round(4).to_csv(p_neg)
        print(f"\nAll negative correlations (extended set): wrote {p_neg}")


if __name__ == "__main__":
    main()
