"""
Exploratory Bayesian logistic regression with NUTS (PyMC) on wheat direction.

**Default design (``BAYES_DESIGN=wheat_pca``):** same engineered wheat + cross-asset
features, ``create_target`` labels, and **train+val-only** ``StandardScaler`` + ``PCA``
as ``TCN_PCA_WHEAT_ONLY``, flattened to ``lookback * n_components`` columns per row for
a flat logit. Optional **inner validation hurdle tuning** for the strategy script
(``STRATEGY_TUNE_HURDLE``) uses two NUTS passes: stage-1 on an early slice to pick ``h*``,
then a refit on all pre-holdout rows before scoring the terminal tail.

**Legacy design (``BAYES_DESIGN=fred_macro``):** FRED-MD macro block + five lagged
wheat log-return features; ``StandardScaler`` + ``PCA`` on macros only (``TCN_Revised``-style).

Usage (from repository root):
  ./venv/bin/python bayes_logit_mcmc_macro_pca.py
  BAYES_DESIGN=fred_macro ./venv/bin/python bayes_logit_mcmc_macro_pca.py

Requires: pip install 'pymc>=5.16,<6'  (matplotlib for figures)

Optional: BAYES_SAVE_FIG=0 to skip PNG export; figures go to docs/figs/.
  BAYES_PRIOR=hier (default): non-centered global shrinkage on coefficients; BAYES_PRIOR=iid
  restores the old iid Normal slab (may diverge with dim≈150). BAYES_TARGET_ACCEPT default 0.95.
"""
from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
import arviz as az

ROOT = pathlib.Path(__file__).resolve().parents[1]
_bundle = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_bundle / "TCN"))

from TCN_Revised import (  # noqa: E402
    SELECTED_FEATURES,
    load_fred_md,
    apply_reporting_delay_and_tcodes,
    forward_fill_macro_to_daily,
    compute_technical_features,
)


def _load_merged_daily(
    wheat_path: str | pathlib.Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    base = ROOT
    fred_train = base / "data" / "fred-md" / "Historical FRED-MD Vintages Final"
    fred_val   = base / "data" / "fred-md" / "Historical-vintages-of-FRED-MD-2015-01-to-2024-12"
    if wheat_path is None:
        wp = base / "data" / "wheat-futures" / "wheat_futures_daily.csv"
    else:
        wp = pathlib.Path(wheat_path).expanduser()
    if not wp.is_file():
        raise FileNotFoundError(f"Wheat CSV not found: {wp}")

    df_macro_raw = pd.concat(
        [load_fred_md(str(fred_train)), load_fred_md(str(fred_val))]
    ).sort_index()
    
    df_macro_raw = df_macro_raw[~df_macro_raw.index.duplicated(keep="last")]
    df_macro     = apply_reporting_delay_and_tcodes(df_macro_raw)

    df_wheat     = pd.read_csv(wp)
    df_wheat["date"] = pd.to_datetime(df_wheat["date"], utc=True).dt.tz_localize(None)
    df_wheat     = df_wheat.sort_values("date").reset_index(drop=True)

    df_merged    = forward_fill_macro_to_daily(df_macro, df_wheat)
    df_tech      = compute_technical_features(df_wheat)
    tech_cols    = [c for c in df_tech.columns if c != "date"]
    df_merged    = df_merged.merge(df_tech, on="date", how="left")
    df_merged[tech_cols] = df_merged[tech_cols].ffill().fillna(0.0)

    macro_cols   = [c for c in SELECTED_FEATURES if c in df_merged.columns]
    return df_merged, macro_cols


def build_flat_xy(
    df_merged: pd.DataFrame,
    macro_cols: list[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One row per TCN-style sample: y = 1{Close[i]-Close[i-1]>0}, x uses info through i-1."""
    macro_lag1 = [f"{c}_lag1" for c in macro_cols]
    df = df_merged.copy()
    df[macro_lag1] = df[macro_cols].shift(1)
    df = df.dropna().reset_index(drop=True)

    prices = df["Close"].values.astype(np.float64)
    n      = len(df)
    rows: list[np.ndarray] = []
    y_list: list[int] = []
    for i in range(lookback, n):
        delta = prices[i] - prices[i - 1]
        y_list.append(int(delta > 0))
        ret_feats: list[float] = []
        p = prices
        for j in range(1, 6):
            ret_feats.append(
                float(np.log(max(p[i - 1], 1e-10) / max(p[i - 1 - j], 1e-10)))
            )
        mvec = df[macro_lag1].iloc[i - 1].values.astype(np.float64)
        rows.append(np.concatenate([np.array(ret_feats, dtype=np.float64), mvec]))
    X = np.stack(rows, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def build_flat_xy_with_returns_dates(
    df_merged: pd.DataFrame,
    macro_cols: list[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same rows as ``build_flat_xy``, plus simple return and label-row date per flat row."""
    macro_lag1 = [f"{c}_lag1" for c in macro_cols]
    df = df_merged.copy()
    df[macro_lag1] = df[macro_cols].shift(1)
    df = df.dropna().reset_index(drop=True)

    prices = df["Close"].values.astype(np.float64)
    n = len(df)
    rows: list[np.ndarray] = []
    y_list: list[int] = []
    r_list: list[float] = []
    d_list: list[pd.Timestamp] = []
    for i in range(lookback, n):
        delta = prices[i] - prices[i - 1]
        y_list.append(int(delta > 0))
        r_list.append(float(delta / max(prices[i - 1], 1e-10)))
        d_list.append(pd.Timestamp(df["date"].iloc[i]))
        ret_feats: list[float] = []
        p = prices
        for j in range(1, 6):
            ret_feats.append(
                float(np.log(max(p[i - 1], 1e-10) / max(p[i - 1 - j], 1e-10)))
            )
        mvec = df[macro_lag1].iloc[i - 1].values.astype(np.float64)
        rows.append(np.concatenate([np.array(ret_feats, dtype=np.float64), mvec]))
    X = np.stack(rows, axis=0)
    y = np.array(y_list, dtype=np.int32)
    r = np.array(r_list, dtype=np.float64)
    dates = np.array(d_list, dtype="datetime64[ns]")
    return X, y, r, dates


def design_matrix_fold_safe(
    X_raw: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_pca: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns X_tr, X_va, y_tr, y_va, macro_var_explained (train PCA)."""
    n_ret  = 5  # log-returns from price channel (col 0) across all lags
    Xr, Xm = X_raw[:, :n_ret], X_raw[:, n_ret:]  # Xr: log-returns, Xm: macro features
    tr, va = train_idx, val_idx  # train and validation indices

    scaler_m = StandardScaler()
    Xm_tr = scaler_m.fit_transform(Xm[tr])
    Xm_va = scaler_m.transform(Xm[va])

    k = min(n_pca, Xm_tr.shape[1], Xm_tr.shape[0])
    pca = PCA(n_components=k, random_state=42)
    Zm_tr = pca.fit_transform(Xm_tr)
    Zm_va = pca.transform(Xm_va)
    ev = float(pca.explained_variance_ratio_.sum())

    scaler_r = StandardScaler()
    R_tr = scaler_r.fit_transform(Xr[tr])
    R_va = scaler_r.transform(Xr[va])

    X_tr = np.hstack([R_tr, Zm_tr]).astype(np.float64)
    X_va = np.hstack([R_va, Zm_va]).astype(np.float64)
    return X_tr, X_va, y[tr], y[va], ev


def _test_posterior_prob_matrix(
    alpha_s: np.ndarray,
    beta_s: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Per-draw probabilities (n, S) for uncertainty bands / histograms."""
    if beta_s.shape[1] == X.shape[1]:
        b = beta_s
    elif beta_s.shape[0] == X.shape[1]:
        b = beta_s.T
    else:
        raise ValueError(f"Unexpected beta shape {beta_s.shape} for X {X.shape}")
    logits = X @ b.T + alpha_s  # (n, S)
    return 1.0 / (1.0 + np.exp(-logits))


def plot_bayes_graphical_summary(
    idata,
    y_te: np.ndarray,
    p_bay_te: np.ndarray,
    p_mle_te: np.ndarray,
    alpha_s: np.ndarray,
    beta_s: np.ndarray,
    X_te: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """Write ROC, calibration, forest, and predictive-uncertainty PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.calibration import calibration_curve

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (1) ROC: Bayes posterior-mean vs MLE (same holdout) ---
    fig1, ax = plt.subplots(figsize=(5, 5))
    fpr_b, tpr_b, _ = roc_curve(y_te, p_bay_te)
    fpr_m, tpr_m, _ = roc_curve(y_te, p_mle_te)
    ax.plot(fpr_b, tpr_b, label=f"Bayes mean (AUC={auc(fpr_b, tpr_b):.3f})", lw=2)
    ax.plot(fpr_m, tpr_m, "--", label=f"MLE logit (AUC={auc(fpr_m, tpr_m):.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Holdout ROC — exploratory Bayesian logit")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig1.tight_layout()
    fig1.savefig(out_dir / "bayes_mcmc_roc_holdout.png", dpi=150)
    plt.close(fig1)

    # --- (2) Calibration (reliability): 10 bins ---
    fig2, ax = plt.subplots(figsize=(5, 5))
    n_bins = 10
    pt_b, pp_b = calibration_curve(y_te, p_bay_te, n_bins=n_bins, strategy="uniform")
    pt_m, pp_m = calibration_curve(y_te, p_mle_te, n_bins=n_bins, strategy="uniform")
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Perfect calibration")
    ax.plot(pp_b, pt_b, "o-", label="Bayes mean predictive p", lw=2)
    ax.plot(pp_m, pt_m, "s--", label="MLE logit", lw=2)
    ax.set_xlabel("Mean predicted P(up)")
    ax.set_ylabel("Fraction of positives (empirical)")
    ax.set_title("Holdout calibration (10 uniform bins)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    fig2.tight_layout()
    fig2.savefig(out_dir / "bayes_mcmc_calibration_holdout.png", dpi=150)
    plt.close(fig2)

    # --- (3) Posterior uncertainty per test day (std across draws) ---
    p_draws = _test_posterior_prob_matrix(alpha_s, beta_s, X_te)
    unc = p_draws.std(axis=1)
    fig3, ax = plt.subplots(figsize=(6, 4))
    ax.hist(unc, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("SD of P(up) across posterior draws")
    ax.set_ylabel("Count of holdout days")
    ax.set_title("Day-level predictive uncertainty (holdout)")
    fig3.tight_layout()
    fig3.savefig(out_dir / "bayes_mcmc_predictive_uncertainty_holdout.png", dpi=150)
    plt.close(fig3)

    # --- (4) Forest plot: alpha + beta ---
    az.plot_forest(
        idata,
        var_names=["alpha", "beta"],
        combined=True,
        figsize=(6, 5),
        colors="steelblue",
    )
    fig4 = plt.gcf()
    fig4.tight_layout()
    fig4.savefig(out_dir / "bayes_mcmc_forest.png", dpi=150)
    plt.close(fig4)


def posterior_mean_probs(
    alpha: np.ndarray,
    beta: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """alpha (S,), beta (S, p) or (p, S), X (n, p) -> mean prob (n,)."""
    if beta.shape[1] == X.shape[1]:
        b = beta  # (S, p)
    elif beta.shape[0] == X.shape[1]:
        b = beta.T  # (S, p)
    else:
        raise ValueError(f"Unexpected beta shape {beta.shape} for X {X.shape}")
    logits = X @ b.T + alpha  # (n, S)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.mean(axis=1)


def _mcmc_bayes_logistic_idata(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    progressbar: bool,
):
    import pymc as pm
    import pytensor.tensor as pt

    prior = os.getenv("BAYES_PRIOR", "hier").strip().lower()
    max_treedepth = int(os.getenv("BAYES_MAX_TREE_DEPTH", "12"))
    # Many collinear flattened PCA-lag features + logistic link -> stiff geometry for NUTS.
    # Default ``hier``: non-centered global shrinkage (scalar σ_β × standard normal z); optional
    # ``BAYES_PRIOR=iid`` restores the older iid Normal(0, BAYES_BETA_SIGMA) slab.

    with pm.Model() as model:
        X_data = pm.Data("X_obs", X_tr)
        alpha_sd = float(os.getenv("BAYES_ALPHA_SIGMA", "1.0"))
        alpha = pm.Normal("alpha", mu=0.0, sigma=alpha_sd)
        p = int(X_tr.shape[1])
        if prior in ("iid", "normal", "flat"):
            beta_sd = float(os.getenv("BAYES_BETA_SIGMA", "0.8"))
            beta = pm.Normal("beta", mu=0.0, sigma=beta_sd, shape=p)
        else:
            shrink_scale = float(os.getenv("BAYES_GLOBAL_SHRINKAGE_SIGMA", "1.0"))
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=shrink_scale)
            z = pm.Normal("z", mu=0.0, sigma=1.0, shape=p)
            beta = pm.Deterministic("beta", z * sigma_beta)
        logit_p = alpha + pt.dot(X_data, beta)
        logit_p = pt.clip(logit_p, -18.0, 18.0)
        pm.Bernoulli("y", logit_p=logit_p, observed=y_tr)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=42,
            target_accept=target_accept,
            progressbar=progressbar,
            compute_convergence_checks=False,
            nuts={"max_treedepth": max_treedepth},
        )
    return idata


def _posterior_coeff_samples(idata) -> tuple[np.ndarray, np.ndarray]:
    post = idata.posterior
    alpha_s = post["alpha"].stack(sample=("chain", "draw")).values.astype(np.float64)
    beta_s = post["beta"].stack(sample=("chain", "draw")).values.astype(np.float64)
    return alpha_s, beta_s


def _tune_hurdle_accuracy(p_val: np.ndarray, y_val: np.ndarray) -> tuple[float, str]:
    """Pick hurdle h in [lo, hi] maximizing direction accuracy on a chronological val slice."""
    if len(y_val) < 40:
        return 0.52, "inner-val n<40; fallback h=0.52"
    step = float(os.getenv("STRATEGY_HGRID_STEP", "0.01"))
    lo = float(os.getenv("STRATEGY_HGRID_LO", "0.46"))
    hi = float(os.getenv("STRATEGY_HGRID_HI", "0.60"))
    grid = np.arange(lo, hi + 1e-9, step, dtype=np.float64)
    best_h = 0.5
    best_acc = -1.0
    for h in grid:
        acc = accuracy_score(y_val, (p_val >= h).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_h = float(h)
    return best_h, f"max val acc={best_acc:.4f} on h grid [{lo:.2f},{hi:.2f}] step {step:.2f}"


def bayes_holdout_posterior_mean_strategy_bundle(
    *,
    wheat_csv: str | pathlib.Path,
    lookback: int | None = None,
    test_frac: float | None = None,
    n_pca: int | None = None,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    target_accept: float | None = None,
    max_train: int | None = None,
    verbose: bool = False,
    progressbar: bool = False,
) -> dict[str, object]:
    """
    Flat Bayes logit: NUTS on a chronological train slice, posterior-mean probabilities
    on the terminal ``test_frac`` hold-out rows, with **simple returns** aligned for
    ``wheat_strategy_holdout_validation.py``.

    **Design** (``BAYES_DESIGN``): default ``wheat_pca`` uses the same engineered wheat +
    cross-asset columns, ``create_target`` labels, and **train+val-only scaler + PCA** as
    ``TCN_PCA_WHEAT_ONLY``, flattened to ``lookback * n_components`` features per row.
    Set ``BAYES_DESIGN=fred_macro`` for the legacy FRED macro + 5 lag-return design.

    **Hurdle tuning (B)** on ``wheat_pca`` when ``STRATEGY_TUNE_HURDLE`` is not disabled:
    fit NUTS on an early train slice, pick ``h*`` maximizing direction accuracy on the
    last ``BAYES_INNER_VAL_FRAC`` fraction of pre-holdout rows, then **refit** NUTS on
    **all** pre-holdout rows before scoring the terminal hold-out (two MCMC passes).

    Environment: ``STRATEGY_BAYES_*``, ``BAYES_*``, ``STRATEGY_HGRID_*``, ``TCN_HORIZON_DAYS``.
    """
    lookback = int(lookback if lookback is not None else os.getenv("BAYES_LOOKBACK", os.getenv("TCN_LOOKBACK", "30")))
    n_pca = int(n_pca if n_pca is not None else os.getenv("BAYES_PCA_COMPONENTS", os.getenv("TCN_PCA_COMPONENTS", "5")))
    draws = int(
        draws
        if draws is not None
        else os.getenv("STRATEGY_BAYES_DRAWS", os.getenv("BAYES_DRAWS", "120"))
    )
    tune = int(
        tune if tune is not None else os.getenv("STRATEGY_BAYES_TUNE", os.getenv("BAYES_TUNE", "120"))
    )
    chains = int(
        chains
        if chains is not None
        else os.getenv("STRATEGY_BAYES_CHAINS", os.getenv("BAYES_CHAINS", "2"))
    )
    target_accept = float(
        target_accept
        if target_accept is not None
        else float(os.getenv("BAYES_TARGET_ACCEPT", "0.95"))
    )
    max_train = int(
        max_train if max_train is not None else os.getenv("BAYES_MAX_TRAIN", "0")
    )
    test_frac = float(
        test_frac if test_frac is not None else os.getenv("BAYES_TEST_FRAC", "0.15")
    )
    design = os.getenv("BAYES_DESIGN", "wheat_pca").strip().lower()
    tune_h = os.getenv("STRATEGY_TUNE_HURDLE", "1").strip().lower() not in ("0", "false", "no")

    if design == "wheat_pca":
        sys.path.insert(0, str(ROOT))
        from TCN_PCA_WHEAT_ONLY import wheat_pca_flat_holdout_design  # noqa: WPS433

        horizon_days = int(os.getenv("TCN_HORIZON_DAYS", "1"))
        D = wheat_pca_flat_holdout_design(
            wheat_csv=wheat_csv,
            lookback=lookback,
            horizon_days=horizon_days,
            pca_components=n_pca,
            holdout_frac=test_frac,
            verbose=verbose,
        )
        X = D["X_flat"].astype(np.float64)
        y = D["y"].astype(np.int32)
        r_all = D["r"].astype(np.float64)
        dates_all = D["dates"]
        split_flat = int(D["split_flat"])
        ev_pca = float(D["explained_var_pca"])

        train_ix = np.arange(0, split_flat)
        test_ix = np.arange(split_flat, len(X))
        if max_train > 0 and len(train_ix) > max_train:
            train_ix = train_ix[-max_train:]
            if verbose:
                print(f"  [bayes strategy] wheat_pca: truncated train to last {max_train} rows")

        inner_frac = float(os.getenv("BAYES_INNER_VAL_FRAC", "0.12"))
        min_train_a = int(os.getenv("BAYES_TUNE_MIN_TRAIN_A", "500"))
        val_len = max(80, int(len(train_ix) * inner_frac))
        val_len = min(val_len, max(0, len(train_ix) - min_train_a - 1))

        h_star: float | None = None
        h_msg = ""
        idata_final = None

        if tune_h and val_len >= 60 and len(train_ix) > val_len + min_train_a:
            tr_a = train_ix[:-val_len]
            va = train_ix[-val_len:]
            if verbose:
                print(
                    f"  [bayes strategy] wheat_pca: inner-val tune — stage-1 train n={len(tr_a)}, "
                    f"val n={len(va)}"
                )
            idata1 = _mcmc_bayes_logistic_idata(
                X[tr_a], y[tr_a],
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, progressbar=progressbar,
            )
            a1, b1 = _posterior_coeff_samples(idata1)
            p_va = posterior_mean_probs(a1, b1, X[va])
            h_star, h_msg = _tune_hurdle_accuracy(p_va, y[va])
            if verbose:
                print(f"  [bayes strategy] tuned hurdle h*={h_star:.4f} ({h_msg})")
            idata_final = _mcmc_bayes_logistic_idata(
                X[train_ix], y[train_ix],
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, progressbar=progressbar,
            )
        else:
            h_msg = "inner-val hurdle tuning skipped (STRATEGY_TUNE_HURDLE=0 or insufficient train)"
            idata_final = _mcmc_bayes_logistic_idata(
                X[train_ix], y[train_ix],
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, progressbar=progressbar,
            )

        af, bf = _posterior_coeff_samples(idata_final)
        p_up_tr = posterior_mean_probs(af, bf, X[train_ix])
        p_up = posterior_mean_probs(af, bf, X[test_ix])
        r_te = r_all[test_ix]
        y_te = y[test_ix]
        d_te = dates_all[test_ix]
        date_first = pd.Timestamp(d_te[0]).date()
        date_last = pd.Timestamp(d_te[-1]).date()

        if verbose:
            print(
                f"  [bayes strategy] wheat_pca: n_train={len(train_ix)}, n_holdout={len(y_te)}, "
                f"design_dim={X.shape[1]}, PCA_EV={ev_pca:.3f}"
            )

        return {
            "r": r_te.astype(np.float64),
            "p_up": p_up.astype(np.float64),
            "p_up_train": p_up_tr.astype(np.float64),
            "y": y_te.astype(np.int32),
            "date_first": date_first,
            "date_last": date_last,
            "n_holdout": int(len(y_te)),
            "explained_var_pca": ev_pca,
            "explained_var_macro_pca": float("nan"),
            "idata": idata_final,
            "bayes_design": design,
            "hurdle_inner_val": h_star,
            "hurdle_inner_msg": h_msg,
        }

    # --- Legacy FRED-macro flat design (single NUTS on full pre-holdout train) ---
    df_merged, macro_cols = _load_merged_daily(wheat_csv)
    X_raw, y, r_all, dates_all = build_flat_xy_with_returns_dates(
        df_merged, macro_cols, lookback=lookback
    )
    n = len(y)
    split = int(n * (1.0 - test_frac))
    idx_all = np.arange(n)
    train_full = idx_all[:split]
    test_idx = idx_all[split:]

    if max_train > 0 and len(train_full) > max_train:
        train_idx = train_full[-max_train:]
        if verbose:
            print(f"  [bayes strategy] fred_macro: subsampled train to last {max_train} rows")
    else:
        train_idx = train_full

    X_tr, X_te, y_tr, y_te, ev_macro = design_matrix_fold_safe(
        X_raw, y, train_idx, test_idx, n_pca=n_pca
    )
    r_te = r_all[test_idx].astype(np.float64)
    d_te = dates_all[test_idx]
    date_first = pd.Timestamp(d_te[0]).date()
    date_last = pd.Timestamp(d_te[-1]).date()

    if verbose:
        print(
            f"  [bayes strategy] fred_macro: n_train={len(y_tr)}, n_holdout={len(y_te)}, "
            f"design_dim={X_tr.shape[1]}, macro_PCA_EV={ev_macro:.3f}"
        )

    idata = _mcmc_bayes_logistic_idata(
        X_tr, y_tr,
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, progressbar=progressbar,
    )
    alpha_s, beta_s = _posterior_coeff_samples(idata)
    p_up_tr = posterior_mean_probs(alpha_s, beta_s, X_tr)
    p_up = posterior_mean_probs(alpha_s, beta_s, X_te)

    return {
        "r": r_te,
        "p_up": p_up.astype(np.float64),
        "p_up_train": p_up_tr.astype(np.float64),
        "y": y_te.astype(np.int32),
        "date_first": date_first,
        "date_last": date_last,
        "n_holdout": int(len(y_te)),
        "explained_var_macro_pca": float(ev_macro),
        "explained_var_pca": float("nan"),
        "idata": idata,
        "bayes_design": design,
        "hurdle_inner_val": None,
        "hurdle_inner_msg": "fred_macro: use STRATEGY_HURDLE only (no inner-val tune in bundle)",
    }


def main() -> None:
    lookback = int(os.getenv("BAYES_LOOKBACK", os.getenv("TCN_LOOKBACK", "30")))
    n_pca = int(os.getenv("BAYES_PCA_COMPONENTS", os.getenv("TCN_PCA_COMPONENTS", "5")))
    draws = int(os.getenv("BAYES_DRAWS", "400"))
    tune = int(os.getenv("BAYES_TUNE", "400"))
    chains = int(os.getenv("BAYES_CHAINS", "2"))
    target_accept = float(os.getenv("BAYES_TARGET_ACCEPT", "0.95"))
    max_train = int(os.getenv("BAYES_MAX_TRAIN", "0"))
    test_frac = float(os.getenv("BAYES_TEST_FRAC", "0.15"))
    design = os.getenv("BAYES_DESIGN", "wheat_pca").strip().lower()
    wheat_env = os.getenv("WHEAT_CSV")
    wheat_path = (
        pathlib.Path(wheat_env).expanduser()
        if wheat_env
        else ROOT / "data" / "wheat-futures" / "wheat_futures_daily.csv"
    )

    if design == "wheat_pca":
        sys.path.insert(0, str(ROOT))
        from TCN_PCA_WHEAT_ONLY import wheat_pca_flat_holdout_design  # noqa: WPS433

        horizon_days = int(os.getenv("TCN_HORIZON_DAYS", "1"))
        print(
            "Bayesian logit (PyMC NUTS) — wheat-only PCA + flattened windows "
            "(``BAYES_DESIGN=wheat_pca``; matches ``TCN_PCA_WHEAT_ONLY`` scaler/PCA)"
        )
        print(
            f"  lookback={lookback}, n_pca={n_pca}, horizon_days={horizon_days}, "
            f"draws={draws}, tune={tune}, chains={chains}"
        )
        D = wheat_pca_flat_holdout_design(
            wheat_csv=wheat_path,
            lookback=lookback,
            horizon_days=horizon_days,
            pca_components=n_pca,
            holdout_frac=test_frac,
            verbose=True,
        )
        X = D["X_flat"].astype(np.float64)
        y = D["y"].astype(np.int32)
        split_flat = int(D["split_flat"])
        ev_report = float(D["explained_var_pca"])
        X_tr, X_te = X[:split_flat], X[split_flat:]
        y_tr, y_te = y[:split_flat], y[split_flat:]

        if max_train > 0 and len(X_tr) > max_train:
            X_tr = X_tr[-max_train:]
            y_tr = y_tr[-max_train:]
            print(f"  subsampled train to last {max_train} rows for MCMC speed")

        print(f"  n_train={len(X_tr)}, n_test={len(X_te)}, design dim={X_tr.shape[1]}")
        print(f"  PCA explained variance (train+val fit, wheat-only block): {ev_report:.3f}")
    else:
        print("Bayesian logit (PyMC NUTS) — FRED macro + 5 lag returns (``BAYES_DESIGN=fred_macro``)")
        print(f"  lookback={lookback}, n_pca={n_pca}, draws={draws}, tune={tune}, chains={chains}")

        df_merged, macro_cols = _load_merged_daily(
            str(wheat_path) if wheat_path.is_file() else None
        )
        X_raw, y = build_flat_xy(df_merged, macro_cols, lookback=lookback)
        n = len(y)
        split = int(n * (1.0 - test_frac))
        idx_all = np.arange(n)
        train_full = idx_all[:split]
        test_idx = idx_all[split:]

        if max_train > 0 and len(train_full) > max_train:
            train_idx = train_full[-max_train:]
            print(f"  subsampled train to last {max_train} rows for MCMC speed")
        else:
            train_idx = train_full

        X_tr, X_te, y_tr, y_te, ev_macro = design_matrix_fold_safe(
            X_raw, y, train_idx, test_idx, n_pca=n_pca
        )
        ev_report = float(ev_macro)
        print(f"  n_train={len(y_tr)}, n_test={len(y_te)}, design dim={X_tr.shape[1]}")
        print(f"  PCA macro explained variance (train): {ev_macro:.3f}")

    # --- Frequentist baseline (same features) ---
    logit = LogisticRegression(max_iter=4000, C=1.0, random_state=42)
    logit.fit(X_tr, y_tr)
    p_mle_te = logit.predict_proba(X_te)[:, 1]
    print(
        "  sklearn LogisticRegression (test): "
        f"acc={accuracy_score(y_te, (p_mle_te >= 0.5).astype(int)):.4f}, "
        f"auc={roc_auc_score(y_te, p_mle_te):.4f}, "
        f"brier={brier_score_loss(y_te, p_mle_te):.4f}"
    )

    idata = _mcmc_bayes_logistic_idata(
        X_tr,
        y_tr,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        progressbar=True,
    )

    print("\n  --- ArviZ summary: alpha, beta (mean, SD, 94% HDI, r_hat, ess) ---")
    _summ = az.summary(
        idata,
        var_names=["alpha", "beta"],
        round_to=4,
    )
    with pd.option_context("display.max_rows", 30, "display.width", 100):
        print(_summ.to_string())

    post     = idata.posterior
    alpha_s  = post["alpha"].stack(sample=("chain", "draw")).values.astype(np.float64)  # (S,)
    beta_s   = post["beta"].stack(sample=("chain", "draw")).values.astype(np.float64)
    p_bay_te = posterior_mean_probs(alpha_s, beta_s, X_te)
    p_bay_tr = posterior_mean_probs(alpha_s, beta_s, X_tr)

    print(
        "  Bayes logit posterior mean p(y=1|x) (test): "
        f"acc={accuracy_score(y_te, (p_bay_te >= 0.5).astype(int)):.4f}, "
        f"auc={roc_auc_score(y_te, p_bay_te):.4f}, "
        f"brier={brier_score_loss(y_te, p_bay_te):.4f}"
    )
    print(
        "  (train, in-sample posterior mean): "
        f"acc={accuracy_score(y_tr, (p_bay_tr >= 0.5).astype(int)):.4f}, "
        f"brier={brier_score_loss(y_tr, p_bay_tr):.4f}"
    )
    print("\n  Appendix: compare Brier on test — lower is better calibrated probability.")
    print("  Next semester: use posterior draws for sizing / CVaR, not only mean p.")

    if os.getenv("BAYES_SAVE_FIG", "1").strip().lower() in ("0", "false", "no"):
        print("  BAYES_SAVE_FIG=0 — skipping figure export.")
    else:
        fig_dir = ROOT / "docs" / "figs"
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Writing graphical diagnostics to {fig_dir}/")
        plot_bayes_graphical_summary(
            idata,
            y_te,
            p_bay_te,
            p_mle_te,
            alpha_s,
            beta_s,
            X_te,
            fig_dir,
        )
        print("  Files: bayes_mcmc_roc_holdout.png, bayes_mcmc_calibration_holdout.png,")
        print("         bayes_mcmc_predictive_uncertainty_holdout.png, bayes_mcmc_forest.png")


if __name__ == "__main__":
    main()
