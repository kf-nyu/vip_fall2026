#!/usr/bin/env python3
"""
Holdout investment-style validation (long–flat) on Chicago SRW wheat.

Primary path (default): **``TCN_PCA_WHEAT_ONLY``** — train the published PCA+TCN
pipeline on train+val only, score the terminal **15%** holdout with
``predict_proba``, and run long–flat rules on **realized** simple returns aligned
to the same labels as ``create_target`` (default one-day horizon). If every holdout
``p`` sits below ``0.5``, **P≥0.5** / **STRATEGY_HURDLE** stay in cash; the script
prints the holdout ``p`` range and a **train-median** row (``p_up_train`` from the
fitted TCN on train sequences).

**Not modeled (preliminary / classroom scope):** margin, leverage limits, listed
options, contract rolls, commissions beyond the optional ``STRATEGY_COST_BPS``
toy, bid--ask, gaps, or partial fills---only buy-and-hold vs long--flat on
aligned simple returns.

Optional path (**``STRATEGY_PANEL=revised``** or **``both``**): leakage-aware
**``TCN_Revised``** tensor + **sklearn** logistic on flattened windows (requires
populated **FRED-MD** under ``data/fred-md``).

Optional **``STRATEGY_PANEL=bayes``** or **``tcn_bayes``** (alias: expands to TCN + Bayes):
flat Bayesian logit (**PyMC** NUTS) on **wheat-only PCA features** (default ``BAYES_DESIGN=wheat_pca``,
same scaler/PCA + ``create_target`` alignment as ``TCN_PCA_WHEAT_ONLY``); legacy FRED macro design via
``BAYES_DESIGN=fred_macro``. Hold-out **posterior-mean** ``P(y=1|x)`` feeds the same long--flat rules;
with ``STRATEGY_TUNE_HURDLE=1`` (default) an inner chronological slice picks ``h*`` by accuracy, then NUTS
refits on all pre-holdout rows (two passes). Quick draws: ``STRATEGY_BAYES_DRAWS`` / ``STRATEGY_BAYES_TUNE``.

**Calibration caveat:** shrunk logit heads can place **every** holdout posterior-mean ``p`` below ``0.5`` while still weakly ranking up days (AUC ``~0.51``). Then ``P>=0.5`` / default ``STRATEGY_HURDLE`` never go long; the script prints the holdout ``p`` range and adds **Long--flat (P≥train median)** so the panel is not silently all-cash.

Strategies on the holdout slice:

  1. **Buy-and-hold** — always long one notional unit.
  2. **Long–flat (hurdle)** — long iff ``P(Up) >= STRATEGY_HURDLE`` (default 0.52).
  3. **Long–flat (0.5 rule)** — long iff ``P(Up) >= 0.5``.
  4. **Long–flat (TCN val threshold)** — long iff ``P(Up) >=`` internal
     ``TCNClassifier`` validation threshold (printed with results).
  5. **Momentum** — long iff previous day's realized label was up (persistence).

Optional **``STRATEGY_COST_BPS``** subtracts a proportional round-trip when the
position changes.

Env: ``HOLDOUT_FRAC``, ``STRATEGY_HURDLE``, ``STRATEGY_COST_BPS``, ``RANDOM_STATE``,
``STRATEGY_PANEL`` = ``tcn_pca_wheat`` | ``revised`` | ``both`` | ``bayes`` | ``tcn_bayes``,
``WHEAT_CSV``, ``TCN_*`` (TCN path), ``BAYES_*`` / ``STRATEGY_BAYES_*`` / ``STRATEGY_TUNE_HURDLE`` /
``BAYES_DESIGN`` / ``BAYES_INNER_VAL_FRAC`` (Bayes path).

Run from ``vip_fall2026``:

  ./venv/bin/python scripts/wheat_strategy_holdout_validation.py
"""
from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TCN"))

from TCN_PCA_WHEAT_ONLY import tcn_pca_wheat_holdout_proba_and_returns  # noqa: E402

from TCN_Revised import (  # noqa: E402
    SELECTED_FEATURES,
    load_fred_md,
    apply_reporting_delay_and_tcodes,
    forward_fill_macro_to_daily,
    compute_technical_features,
    build_sliding_windows,
)


def _max_drawdown(cum_wealth: np.ndarray) -> float:
    """Max drawdown on a cumulative wealth path (1 + running compound)."""
    if len(cum_wealth) == 0:
        return float("nan")
    peak = np.maximum.accumulate(cum_wealth)
    dd = (cum_wealth / np.maximum(peak, 1e-12)) - 1.0
    return float(dd.min())


def _strategy_stats(
    r: np.ndarray,
    w: np.ndarray,
    cost_bps: float,
    name: str,
) -> dict:
    """Daily P&L proxy: w_t * r_t minus turnover cost when w changes."""
    w = np.asarray(w, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    if len(w) != n:
        raise ValueError("w and r length mismatch")

    dw = np.diff(w, prepend=w[0])
    turn = (np.abs(dw) > 1e-12).astype(np.float64)
    cost = (cost_bps / 10000.0) * turn
    gross = w * r - cost
    wealth = np.cumprod(1.0 + gross)
    total_ret = float(wealth[-1] - 1.0) if n else float("nan")
    vol = float(gross.std() * np.sqrt(252)) if gross.std() > 0 else 0.0
    sharpe = (
        float((gross.mean() * 252) / (gross.std() * np.sqrt(252) + 1e-15))
        if gross.std() > 0
        else float("nan")
    )
    mdd = _max_drawdown(wealth)
    active = float(np.mean(w > 0.5))
    return {
        "name": name,
        "n": n,
        "total_return": total_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "pct_long_days": active,
        "mean_daily": float(gross.mean()),
    }


def _resolve_wheat_csv() -> pathlib.Path:
    env = os.getenv("WHEAT_CSV")
    if env:
        p = pathlib.Path(env).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"WHEAT_CSV does not exist: {p}")
    for p in (
        ROOT / "data" / "wheat-futures" / "wheat_futures_daily.csv",
        ROOT.parent / "data" / "wheat-futures" / "wheat_futures_daily.csv",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Wheat CSV not found. Set WHEAT_CSV or place wheat_futures_daily.csv under "
        f"{ROOT / 'data' / 'wheat-futures'} or {ROOT.parent / 'data' / 'wheat-futures'}."
    )


def _print_block(
    title: str,
    r_te: np.ndarray,
    p_up: np.ndarray,
    y_te: np.ndarray,
    hurdle: float,
    thr: float,
    cost_bps: float,
    *,
    include_tcn_cal_row: bool = True,
    extra_after_half: list[tuple[np.ndarray, str]] | None = None,
    extra_specs: list[tuple[np.ndarray, str]] | None = None,
) -> None:
    w_bh = np.ones_like(r_te, dtype=np.float64)
    w_hurdle = (p_up >= hurdle).astype(np.float64)
    w_half = (p_up >= 0.5).astype(np.float64)
    w_mom = np.zeros_like(r_te, dtype=np.float64)
    if len(r_te) > 1:
        w_mom[1:] = y_te[:-1].astype(np.float64)

    specs: list[tuple[np.ndarray, str]] = [
        (w_bh, "Buy-and-hold"),
        (w_hurdle, f"Long–flat (P≥{hurdle:.2f})"),
        (w_half, "Long–flat (P≥0.50)"),
    ]
    if extra_after_half:
        specs.extend(extra_after_half)
    if include_tcn_cal_row:
        w_cal = (p_up >= thr).astype(np.float64)
        specs.append((w_cal, f"Long–flat (P≥TCN thr={thr:.4f})"))
    specs.append((w_mom, "Momentum (lag-1 up)"))
    if extra_specs:
        specs.extend(extra_specs)

    rows = [_strategy_stats(r_te, w, cost_bps, label) for w, label in specs]

    if title.strip():
        print(title)
    print(
        f"\n  {'Strategy':<30} {'TotRet':>10} {'AnnVol':>10} {'Sharpe':>8} {'MaxDD':>9} {'%Long':>7}"
    )
    print("  " + "-" * 80)
    for row in rows:
        print(
            f"  {row['name']:<30}"
            f"  {row['total_return']:>9.2%}"
            f"  {row['ann_vol']:>9.2%}"
            f"  {row['sharpe']:>8.2f}"
            f"  {row['max_dd']:>8.2%}"
            f"  {row['pct_long_days']:>6.1%}"
        )


def _run_revised_logistic_block(
    hurdle: float, cost_bps: float, seed: int, lookback: int, holdout_frac: float
) -> None:
    fred_train = ROOT / "data" / "fred-md" / "Historical FRED-MD Vintages Final"
    fred_val = ROOT / "data" / "fred-md" / "Historical-vintages-of-FRED-MD-2015-01-to-2024-12"
    wheat_path = _resolve_wheat_csv()

    try:
        df_macro_raw = pd.concat(
            [load_fred_md(str(fred_train)), load_fred_md(str(fred_val))]
        ).sort_index()
    except FileNotFoundError as e:
        print(f"\n  [revised panel skipped] {e}")
        return

    df_macro_raw = df_macro_raw[~df_macro_raw.index.duplicated(keep="last")]
    df_macro = apply_reporting_delay_and_tcodes(df_macro_raw)

    df_wheat = pd.read_csv(wheat_path)
    df_wheat["date"] = pd.to_datetime(df_wheat["date"], utc=True).dt.tz_localize(None)
    df_wheat = df_wheat.sort_values("date").reset_index(drop=True)

    macro_cols = [c for c in SELECTED_FEATURES if c in df_macro.columns]
    df_merged = forward_fill_macro_to_daily(df_macro, df_wheat)
    df_tech = compute_technical_features(df_wheat)
    tech_cols = [c for c in df_tech.columns if c != "date"]
    df_merged = df_merged.merge(df_tech, on="date", how="left")
    df_merged[tech_cols] = df_merged[tech_cols].ffill().fillna(0.0)

    X, y_price, y_dir, y_close, y_prev, dates = build_sliding_windows(
        df_merged, macro_cols, tech_cols=tech_cols, lookback=lookback
    )
    n = len(y_dir)
    test_size = max(1, int(n * holdout_frac))
    split_idx = n - test_size

    r_all = (y_close - y_prev) / np.clip(y_prev, 1e-10, None)

    X_tv, X_te = X[:split_idx], X[split_idx:]
    yd_tv, yd_te = y_dir[:split_idx], y_dir[split_idx:]
    r_te = r_all[split_idx:]

    X_tv_f = X_tv.reshape(X_tv.shape[0], -1)
    X_te_f = X_te.reshape(X_te.shape[0], -1)

    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv_f)
    X_te_s = scaler.transform(X_te_f)

    clf = LogisticRegression(
        max_iter=4000,
        random_state=seed,
        class_weight="balanced",
        C=float(os.getenv("STRATEGY_LR_C", "1.0")),
    )
    clf.fit(X_tv_s, yd_tv)
    p_up = clf.predict_proba(X_te_s)[:, 1]

    print(
        f"\n--- Revised-panel + sklearn logistic (holdout n={len(r_te)}; "
        f"{pd.to_datetime(dates[split_idx]).date()} → {pd.to_datetime(dates[-1]).date()}) ---"
    )
    _print_block(
        "  Logistic panel has no separate validation threshold row.",
        r_te,
        p_up,
        yd_te.astype(np.int32),
        hurdle,
        0.5,
        cost_bps,
        include_tcn_cal_row=False,
    )
    print(
        "\n  Note: flattened logistic on (L×37) is a transparent baseline; "
        "probabilities differ from the TCN head."
    )


def _parse_strategy_panels(raw: str) -> set[str]:
    s = raw.strip().lower()
    if not s:
        return {"tcn_pca_wheat"}
    return {p for p in s.replace(",", " ").split() if p}


def _run_bayes_posterior_mean_block(
    hurdle: float,
    cost_bps: float,
    lookback: int,
    holdout_frac: float,
    wheat_csv: pathlib.Path,
) -> None:
    try:
        from bayes_logit_mcmc_macro_pca import (  # noqa: WPS433 — runtime optional heavy dep
            bayes_holdout_posterior_mean_strategy_bundle,
        )
    except ImportError as e:
        print(
            f"\n  [bayes panel skipped] ImportError ({e}). "
            "Install PyMC stack (see requirements-handin.txt), e.g. pip install 'pymc>=5.16,<6' arviz."
        )
        return

    try:
        bundle = bayes_holdout_posterior_mean_strategy_bundle(
            wheat_csv=wheat_csv,
            lookback=lookback,
            test_frac=holdout_frac,
            verbose=True,
            progressbar=os.getenv("STRATEGY_BAYES_PROGRESS", "").strip().lower()
            in ("1", "true", "yes"),
        )
    except FileNotFoundError as e:
        print(f"\n  [bayes panel skipped] {e}")
        return

    r_te = bundle["r"]
    p_up = bundle["p_up"]
    y_te = bundle["y"]
    print(
        f"\n--- Bayesian flat logit — posterior-mean P(up) on holdout "
        f"(n={bundle['n_holdout']}; {bundle['date_first']} → {bundle['date_last']}) ---"
    )
    des = bundle.get("bayes_design", "?")
    evp = bundle.get("explained_var_pca")
    evm = bundle.get("explained_var_macro_pca")
    if evp is not None and np.isfinite(float(evp)):
        ev_line = f"  design={des}  |  PCA_EV(wheat)={float(evp):.3f}"
    else:
        ev_line = f"  design={des}  |  macro_PCA_EV={float(evm):.3f}"
    print(ev_line + "  |  rules: hurdle / 0.5 / train-median / momentum; no TCN val-threshold row.")
    pmx = float(np.max(p_up))
    pmn = float(np.min(p_up))
    pmed = float(np.median(p_up))
    print(
        f"  holdout posterior-mean p: min={pmn:.3f}  median={pmed:.3f}  max={pmx:.3f}"
    )
    if pmx < 0.5 - 1e-9:
        print(
            "  note: all posterior-mean p<0.5 on this holdout — fixed 0.5 / high hurdles stay in cash; "
            "use train-median row or set STRATEGY_HURDLE below max(p)."
        )
    extra_half: list[tuple[np.ndarray, str]] = []
    p_train = bundle.get("p_up_train")
    if p_train is not None and len(p_train) > 0:
        thr_tr_med = float(np.median(np.asarray(p_train, dtype=np.float64)))
        extra_half.append(
            (
                (p_up >= thr_tr_med).astype(np.float64),
                f"Long–flat (P≥train median {thr_tr_med:.3f})",
            )
        )
    extra_tail: list[tuple[np.ndarray, str]] = []
    h_tune = bundle.get("hurdle_inner_val")
    if h_tune is not None and np.isfinite(float(h_tune)):
        hh = float(h_tune)
        extra_tail.append(
            ((p_up >= hh).astype(np.float64), f"Long–flat (P≥h*={hh:.3f} inner-val tune)")
        )

    _print_block(
        "  Posterior mean averages MC sigmoid draws per day; hurdle is an optional confidence filter.",
        r_te,
        p_up,
        y_te,
        hurdle,
        0.5,
        cost_bps,
        include_tcn_cal_row=False,
        extra_after_half=extra_half or None,
        extra_specs=extra_tail or None,
    )
    if bundle.get("hurdle_inner_msg"):
        print(f"\n  Inner-val hurdle note: {bundle['hurdle_inner_msg']}")
    print(
        "\n  Note: flat linear Bayes vs. sequence TCN use different features/horizon alignment; "
        "hold-out calendar lengths may differ from the TCN_PCA_WHEAT_ONLY tail."
    )


def main() -> None:
    holdout_frac = float(os.getenv("HOLDOUT_FRAC", "0.15"))
    hurdle = float(os.getenv("STRATEGY_HURDLE", "0.52"))
    cost_bps = float(os.getenv("STRATEGY_COST_BPS", "0"))
    seed = int(os.getenv("RANDOM_STATE", "42"))
    lookback = int(os.getenv("TCN_LOOKBACK", "30"))
    raw_panel = os.getenv("STRATEGY_PANEL", "tcn_pca_wheat").strip().lower()
    panels = _parse_strategy_panels(raw_panel)
    if "both" in panels:
        panels |= {"tcn_pca_wheat", "revised"}
        panels.discard("both")
    if "tcn_bayes" in panels:
        panels |= {"tcn_pca_wheat", "bayes"}
        panels.discard("tcn_bayes")

    wheat_csv = _resolve_wheat_csv()

    print("=== Wheat long–flat strategy (holdout validation) ===")
    print(
        f"  panel={sorted(panels)}, holdout_frac={holdout_frac}, hurdle={hurdle}, "
        f"cost_bps={cost_bps}, lookback={lookback}"
    )
    print(f"  wheat_csv={wheat_csv}")

    if "tcn_pca_wheat" in panels:
        bundle = tcn_pca_wheat_holdout_proba_and_returns(
            wheat_csv=wheat_csv,
            lookback=lookback,
            holdout_frac=holdout_frac,
            verbose=True,
        )
        r_te = bundle["r"]
        p_up = bundle["p_up"]
        y_te = bundle["y"]
        thr = bundle["cls_threshold"]
        print(
            f"\n--- TCN_PCA_WHEAT_ONLY (holdout n={bundle['n_holdout']}; "
            f"{bundle['date_first']} → {bundle['date_last']}) ---"
        )
        print(
            f"  train seq: {bundle['n_train_seq']}  |  PCA EV (train+val): "
            f"{bundle['explained_var_pca']:.3f}  |  TCN val thr: {thr:.4f}"
        )
        p_up = np.asarray(bundle["p_up"], dtype=np.float64).reshape(-1)
        pmx = float(np.max(p_up))
        pmn = float(np.min(p_up))
        pmed = float(np.median(p_up))
        print(
            f"  holdout TCN p(Up): min={pmn:.3f}  median={pmed:.3f}  max={pmx:.3f}"
        )
        if pmx < 0.5 - 1e-9:
            print(
                "  note: all holdout p<0.5 — P≥0.5 / STRATEGY_HURDLE>max(p) stay in cash; "
                "use val-calibrated row, train-median row, or lower STRATEGY_HURDLE."
            )
        extra_tcn: list[tuple[np.ndarray, str]] = []
        p_tr = bundle.get("p_up_train")
        if p_tr is not None and len(p_tr) > 0:
            thr_tm = float(np.median(np.asarray(p_tr, dtype=np.float64)))
            extra_tcn.append(
                (
                    (p_up >= thr_tm).astype(np.float64),
                    f"Long–flat (P≥train median {thr_tm:.3f})",
                )
            )
        _print_block(
            "",
            r_te,
            p_up,
            y_te,
            hurdle,
            thr,
            cost_bps,
            extra_after_half=extra_tcn or None,
        )
        print(
            "\n  Interpretation: beating buy-and-hold can mean either higher total simple "
            "return or smaller loss / drawdown on this single tail under a frictionless "
            "long-only toy; neither is guaranteed from headline direction accuracy nor "
            "evidence of alpha after realistic costs."
        )

    if "revised" in panels:
        _run_revised_logistic_block(hurdle, cost_bps, seed, lookback, holdout_frac)

    if "bayes" in panels:
        _run_bayes_posterior_mean_block(hurdle, cost_bps, lookback, holdout_frac, wheat_csv)


if __name__ == "__main__":
    main()
