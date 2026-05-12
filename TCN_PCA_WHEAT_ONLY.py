import os
import sys
import pathlib
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ------------------------------------------------------------------
# Wheat-only + PCA + TCN_Revised classifier
# ------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

_bundle = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_bundle / "TCN"))
from TCN_Revised import TCNClassifier as RevisedTCNClassifier  # noqa: E402

# Daily Yahoo closes aligned to wheat calendar (same convention as `TCN_PCA.py`).
YAHOO_MACRO_DAILY = {"corn": "ZC=F", "soy": "ZS=F", "uup": "UUP", "cad": "CAD=X"}


def _yahoo_closes_aligned(ticker_map: dict, wheat_index: pd.DatetimeIndex) -> pd.DataFrame:
    r"""Align daily Yahoo closes to the wheat calendar.

    Batch ``yf.download`` can fail when one ticker errors (e.g. ``CAD=X`` "possibly
    delisted"); we fetch each series with ``Ticker.history`` and CAD fallbacks so
    Bayes / wheat-only pipelines do not collapse to zero rows.
    """
    import yfinance as yf

    start_str = wheat_index.min().strftime("%Y-%m-%d")
    end_str = (wheat_index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    norm_idx = pd.DatetimeIndex(wheat_index).normalize()

    def _history_close(symbol: str) -> pd.Series:
        sym = symbol.strip()
        if not sym:
            return pd.Series(dtype=np.float64)
        hist = yf.Ticker(sym).history(start=start_str, end=end_str, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return pd.Series(dtype=np.float64)
        s = hist["Close"].astype(np.float64)
        s.index = pd.to_datetime(s.index, utc=True).tz_localize(None).normalize()
        return s[~s.index.duplicated(keep="last")]

    cols: dict[str, np.ndarray] = {}
    cad_alts = [x.strip() for x in os.getenv("YAHOO_CAD_FALLBACKS", "CAD=X,USDCAD=X").split(",") if x.strip()]

    for k, sym in ticker_map.items():
        s = _history_close(sym)
        if s.empty and k == "cad":
            for alt in cad_alts:
                if alt == sym:
                    continue
                s = _history_close(alt)
                if not s.empty:
                    break
        if s.empty:
            v = pd.Series(np.nan, index=norm_idx)
        else:
            v = s.reindex(norm_idx).ffill().bfill()
        cols[k] = v.to_numpy(dtype=np.float64)

    return pd.DataFrame(cols, index=wheat_index)


def engineer_wheat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price-derived wheat features plus daily cross-asset log returns (corn, soy, UUP, CAD=X)."""
    out = df.copy()
    out["ret_1d"] = np.log(out["price"]).diff()
    out["ret_3d"] = np.log(out["price"]).diff(3)
    out["ret_5d"] = np.log(out["price"]).diff(5)
    out["vol_5d"] = out["ret_1d"].rolling(5).std()
    out["vol_10d"] = out["ret_1d"].rolling(10).std()
    out["mom_5"] = out["price"] / out["price"].shift(5) - 1
    out["mom_10"] = out["price"] / out["price"].shift(10) - 1

    for col in YAHOO_MACRO_DAILY.keys():
        if col not in out.columns or out[col].isna().all():
            out[f"{col}_ret1"] = 0.0
            continue
        safe_series = out[col].replace(0, np.nan).ffill().bfill()
        if safe_series.isna().all():
            out[f"{col}_ret1"] = 0.0
            continue
        out[f"{col}_ret1"] = np.log(safe_series).diff()

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def create_target(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """target[t]=sign(log(price[t+h]/price[t]))."""
    out = df.copy()
    fwd_ret = np.log(out["price"].shift(-horizon_days) / out["price"])
    out["target"] = (fwd_ret > 0).astype(int)
    out = out.iloc[:-horizon_days]
    return out


def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


def create_sequences_index_range(
    X: np.ndarray, y: np.ndarray, lookback: int, k_min: int, k_max: int
):
    """Windows X[k:k+lookback] predicting y[k+lookback], for k in [k_min, k_max]."""
    Xs, ys = [], []
    for k in range(k_min, k_max + 1):
        if k < 0 or k + lookback > len(X):
            continue
        Xs.append(X[k : k + lookback])
        ys.append(y[k + lookback])
    return np.array(Xs), np.array(ys)


def wheat_pca_flat_holdout_design(
    *,
    wheat_csv: str | pathlib.Path | None = None,
    lookback: int | None = None,
    horizon_days: int | None = None,
    pca_components: int | None = None,
    holdout_frac: float | None = None,
    n_splits: int | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """
    Same wheat + Yahoo cross-asset engineering, ``create_target`` labels, and
    train+val-only scaler + PCA as ``TCN_PCA_WHEAT_ONLY``, but flattened to
    ``(n_seq, lookback * n_components)`` rows for logistic / Bayes MCMC.

    Row ``k`` uses window ``X_pca[k:k+lookback]`` and predicts ``y[k+lookback]`` with
    return aligned to ``create_target`` / hold-out simple return convention in
    ``tcn_pca_wheat_holdout_proba_and_returns``.
    """
    lookback = int(lookback if lookback is not None else os.getenv("TCN_LOOKBACK", "30"))
    n_splits = int(n_splits if n_splits is not None else os.getenv("TCN_SPLITS", "5"))
    horizon_days = int(
        horizon_days if horizon_days is not None else os.getenv("TCN_HORIZON_DAYS", "1")
    )
    pca_components = int(
        pca_components if pca_components is not None else os.getenv("TCN_PCA_COMPONENTS", "5")
    )
    holdout_frac = float(
        holdout_frac if holdout_frac is not None else os.getenv("TCN_HOLDOUT_FRAC", "0.15")
    )

    path = pathlib.Path(wheat_csv) if wheat_csv is not None else DATA_ROOT / "wheat-futures" / "wheat_futures_daily.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Wheat CSV not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.rename(columns={"Close": "price"}, inplace=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    if verbose:
        print("  [wheat_pca_flat] downloading cross-asset closes (Yahoo) ...")
    cross = _yahoo_closes_aligned(YAHOO_MACRO_DAILY, df.index)
    df = df.join(cross, how="left")

    df_eng = engineer_wheat_features(df)
    prices_for_returns = df_eng["price"].to_numpy(dtype=np.float64)
    engineer_index = df_eng.index
    df = create_target(df_eng, horizon_days=horizon_days)

    feat_cols = [
        "ret_1d", "ret_3d", "ret_5d", "vol_5d", "vol_10d", "mom_5", "mom_10",
        "corn_ret1", "soy_ret1", "uup_ret1", "cad_ret1",
    ]
    X_raw = df[feat_cols].values
    y = df["target"].values.astype(np.int32)

    if len(X_raw) == 0:
        raise ValueError(
            "Not enough rows after preprocessing (0). Cross-asset Yahoo data may have failed "
            "(e.g. CAD=X). See _yahoo_closes_aligned; try YAHOO_CAD_FALLBACKS=USDCAD=X or rerun."
        )
    if len(X_raw) <= n_splits:
        raise ValueError(f"Not enough rows after preprocessing: {len(X_raw)}")

    split_idx = len(X_raw)
    if holdout_frac > 0:
        test_size = max(1, int(len(X_raw) * holdout_frac))
        split_idx = len(X_raw) - test_size
        if split_idx <= n_splits + lookback + 1:
            raise ValueError(
                f"Train+val too short after holdout (split_idx={split_idx}). "
                "Lower TCN_HOLDOUT_FRAC or n_splits."
            )

    scaler_h = StandardScaler()
    scaler_h.fit(X_raw[:split_idx])
    X_all_s = scaler_h.transform(X_raw)
    n_comp_h = min(pca_components, X_all_s.shape[1])
    pca_h = PCA(n_components=n_comp_h, random_state=42)
    pca_h.fit(X_all_s[:split_idx])
    X_all_pca = pca_h.transform(X_all_s)
    ev = float(pca_h.explained_variance_ratio_.sum())

    n_seq = len(X_all_pca) - lookback
    n_feat = lookback * X_all_pca.shape[1]
    X_flat = np.zeros((n_seq, n_feat), dtype=np.float64)
    y_seq = np.zeros(n_seq, dtype=np.int32)
    r_seq = np.zeros(n_seq, dtype=np.float64)
    date_list: list[pd.Timestamp] = []
    for k in range(n_seq):
        X_flat[k] = X_all_pca[k : k + lookback].reshape(-1)
        j = k + lookback
        y_seq[k] = int(y[j])
        pj = prices_for_returns[j]
        pjh = prices_for_returns[j + horizon_days]
        r_seq[k] = float((pjh - pj) / max(pj, 1e-12))
        ts_end = engineer_index[j + horizon_days]
        date_list.append(pd.Timestamp(ts_end))

    split_flat = int(split_idx - lookback)

    return {
        "X_flat": X_flat,
        "y": y_seq,
        "r": r_seq,
        "dates": np.array(date_list, dtype="datetime64[ns]"),
        "split_flat": split_flat,
        "explained_var_pca": ev,
        "n_comp": int(n_comp_h),
        "lookback": int(lookback),
        "horizon_days": int(horizon_days),
    }


def tcn_pca_wheat_holdout_proba_and_returns(
    *,
    wheat_csv: str | pathlib.Path | None = None,
    lookback: int | None = None,
    n_splits: int | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    horizon_days: int | None = None,
    pca_components: int | None = None,
    channels: list[int] | None = None,
    holdout_frac: float | None = None,
    patience: int | None = None,
    fit_seed: int = 43,
    verbose: bool = False,
) -> dict:
    """
    Train ``TCN_PCA_WHEAT_ONLY`` on train+val rows only, then return holdout
    sequence-level TCN probabilities and **realized** next-bar simple returns
    aligned to each prediction (same labeling as ``create_target``).

    Returns
    -------
    dict with keys:
        ``p_up`` : predicted probability of up move (horizon ``horizon_days``).
        ``p_up_train`` : ``predict_proba`` on the **fitted** train sequences (for median thresholds).
        ``y`` : realized 0/1 labels on holdout sequences.
        ``r`` : simple returns $(P_{t+h}-P_t)/P_t$ matching that horizon (``h=1`` → one day).
        ``cls_threshold`` : validation-tuned threshold inside ``TCNClassifier``.
        ``n_holdout``, ``n_train_seq``, ``split_idx``, ``explained_var_pca``,
        ``date_first``, ``date_last`` (end of return window as ``datetime.date``).
    """
    lookback = int(lookback if lookback is not None else os.getenv("TCN_LOOKBACK", "30"))
    n_splits = int(n_splits if n_splits is not None else os.getenv("TCN_SPLITS", "5"))
    lr = float(lr if lr is not None else os.getenv("TCN_LR", "3e-6"))
    epochs = int(epochs if epochs is not None else os.getenv("TCN_EPOCHS", "30"))
    horizon_days = int(
        horizon_days if horizon_days is not None else os.getenv("TCN_HORIZON_DAYS", "1")
    )
    pca_components = int(
        pca_components if pca_components is not None else os.getenv("TCN_PCA_COMPONENTS", "5")
    )
    if channels is None:
        channels_str = os.getenv("TCN_CHANNELS", "128,32")
        channels = [int(x.strip()) for x in channels_str.split(",") if x.strip()]
    holdout_frac = float(
        holdout_frac if holdout_frac is not None else os.getenv("TCN_HOLDOUT_FRAC", "0.15")
    )
    patience = int(patience if patience is not None else os.getenv("TCN_PATIENCE", "20"))

    path = pathlib.Path(wheat_csv) if wheat_csv is not None else DATA_ROOT / "wheat-futures" / "wheat_futures_daily.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Wheat CSV not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.rename(columns={"Close": "price"}, inplace=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    if verbose:
        print("Downloading daily cross-asset closes (Yahoo): corn, soy, UUP, CAD=X ...")
    cross = _yahoo_closes_aligned(YAHOO_MACRO_DAILY, df.index)
    df = df.join(cross, how="left")

    df_eng = engineer_wheat_features(df)
    # Forward returns for the last label use closes one step beyond the trimmed panel.
    prices_for_returns = df_eng["price"].to_numpy(dtype=np.float64)
    engineer_index = df_eng.index
    df = create_target(df_eng, horizon_days=horizon_days)

    feat_cols = [
        "ret_1d", "ret_3d", "ret_5d", "vol_5d", "vol_10d", "mom_5", "mom_10",
        "corn_ret1", "soy_ret1", "uup_ret1", "cad_ret1",
    ]
    X_raw = df[feat_cols].values
    y = df["target"].values.astype(np.int32)

    if len(X_raw) <= n_splits:
        raise ValueError(f"Not enough rows after preprocessing: {len(X_raw)}")

    split_idx = len(X_raw)
    if holdout_frac > 0:
        test_size = max(1, int(len(X_raw) * holdout_frac))
        split_idx = len(X_raw) - test_size
        if split_idx <= n_splits + lookback + 1:
            raise ValueError(
                f"Train+val too short after holdout (split_idx={split_idx}). "
                "Lower TCN_HOLDOUT_FRAC or n_splits."
            )

    scaler_h = StandardScaler()
    scaler_h.fit(X_raw[:split_idx])
    X_all_s = scaler_h.transform(X_raw)
    n_comp_h = min(pca_components, X_all_s.shape[1])
    pca_h = PCA(n_components=n_comp_h, random_state=42)
    pca_h.fit(X_all_s[:split_idx])
    X_all_pca = pca_h.transform(X_all_s)
    ev = float(pca_h.explained_variance_ratio_.sum())

    k_train_end = split_idx - lookback - 1
    k_test_start = split_idx - lookback
    k_test_end = len(X_all_pca) - lookback - 1
    X_tr_seq_h, y_tr_seq_h = create_sequences_index_range(
        X_all_pca, y, lookback, 0, k_train_end
    )
    X_te_seq_h, y_te_seq_h = create_sequences_index_range(
        X_all_pca, y, lookback, k_test_start, k_test_end
    )

    torch.manual_seed(fit_seed)
    np.random.seed(fit_seed)
    clf_h = RevisedTCNClassifier(
        task_type="classification",
        num_channels=channels,
        kernel_size=3,
        dropout=0.2,
        lr=lr,
        epochs=epochs,
        batch_size=64,
        lookback=lookback,
        patience=patience,
    )
    clf_h.fit(X_tr_seq_h, y_tr_seq_h)
    proba_h = np.asarray(clf_h.predict_proba(X_te_seq_h), dtype=np.float64).reshape(-1)
    p_up_train = np.asarray(clf_h.predict_proba(X_tr_seq_h), dtype=np.float64).reshape(-1)

    # Per-sequence index j = k + lookback: realized simple return over horizon_days
    r_list = []
    date_list = []
    for k in range(k_test_start, k_test_end + 1):
        j = k + lookback
        pj = prices_for_returns[j]
        pjh = prices_for_returns[j + horizon_days]
        r_list.append((pjh - pj) / max(pj, 1e-12))
        ts_end = engineer_index[j + horizon_days]
        date_list.append(ts_end.date() if hasattr(ts_end, "date") else ts_end)

    r_te = np.asarray(r_list, dtype=np.float64)

    return {
        "p_up": proba_h,
        "p_up_train": p_up_train,
        "y": y_te_seq_h.astype(np.int32),
        "r": r_te,
        "cls_threshold": float(clf_h._cls_threshold),
        "n_holdout": int(len(proba_h)),
        "n_train_seq": int(len(y_tr_seq_h)),
        "split_idx": int(split_idx),
        "explained_var_pca": ev,
        "date_first": date_list[0] if date_list else None,
        "date_last": date_list[-1] if date_list else None,
    }


def main():
    # Config
    lookback = int(os.getenv("TCN_LOOKBACK", "30"))
    n_splits = int(os.getenv("TCN_SPLITS", "5"))
    lr = float(os.getenv("TCN_LR", "3e-6"))
    epochs = int(os.getenv("TCN_EPOCHS", "30"))
    horizon_days = int(os.getenv("TCN_HORIZON_DAYS", "1"))
    pca_components = int(os.getenv("TCN_PCA_COMPONENTS", "5"))
    channels_str = os.getenv("TCN_CHANNELS", "128,32")
    channels = [int(x.strip()) for x in channels_str.split(",") if x.strip()]
    holdout_frac = float(os.getenv("TCN_HOLDOUT_FRAC", "0.15"))
    patience = int(os.getenv("TCN_PATIENCE", "20"))

    print(f"Wheat-only PCA TCN")
    print(f"  lookback={lookback}, splits={n_splits}, lr={lr}, epochs={epochs}")
    print(f"  horizon=t+{horizon_days}, pca_components={pca_components}, channels={channels}")
    print(f"  early_stop_patience={patience} (set TCN_PATIENCE)")
    if holdout_frac > 0:
        print(f"  holdout_frac={holdout_frac} (CV on train+val only; final eval on tail)")
    else:
        print("  holdout_frac=0 (CV on full series; no terminal holdout block)")

    # Data load
    wheat_path = DATA_ROOT / "wheat-futures" / "wheat_futures_daily.csv"
    df = pd.read_csv(wheat_path, parse_dates=["date"], index_col="date")
    df.rename(columns={"Close": "price"}, inplace=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    print("Downloading daily cross-asset closes (Yahoo): corn, soy, UUP, CAD=X ...")
    cross = _yahoo_closes_aligned(YAHOO_MACRO_DAILY, df.index)
    df = df.join(cross, how="left")

    # Feature engineering + target
    df = engineer_wheat_features(df)
    df = create_target(df, horizon_days=horizon_days)
    print(f"  rows after feature/target prep: {len(df)}")

    feat_cols = [
        "ret_1d", "ret_3d", "ret_5d", "vol_5d", "vol_10d", "mom_5", "mom_10",
        "corn_ret1", "soy_ret1", "uup_ret1", "cad_ret1",
    ]
    X_raw = df[feat_cols].values
    y = df["target"].values.astype(np.int32)

    if len(X_raw) <= n_splits:
        raise ValueError(f"Not enough rows after preprocessing: {len(X_raw)}")

    split_idx = len(X_raw)
    if holdout_frac > 0:
        test_size = max(1, int(len(X_raw) * holdout_frac))
        split_idx = len(X_raw) - test_size
        if split_idx <= n_splits + lookback + 1:
            raise ValueError(
                f"Train+val too short after holdout (split_idx={split_idx}). "
                "Lower TCN_HOLDOUT_FRAC or n_splits."
            )

    X_cv = X_raw[:split_idx] if holdout_frac > 0 else X_raw
    y_cv = y[:split_idx] if holdout_frac > 0 else y

    if len(X_cv) <= n_splits:
        raise ValueError(f"Not enough rows for CV: {len(X_cv)}")

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    model_acc_05, model_f1_05, model_auc = [], [], []
    model_acc_cal, model_f1_cal = [], []
    maj_acc, maj_f1, maj_auc = [], [], []
    mom_acc, mom_f1, mom_auc = [], [], []

    for fold, (train_idx, val_idx) in enumerate(ts_cv.split(X_cv), start=1):
        print(f"\n=== Fold {fold} ===")
        X_tr_raw, X_va_raw = X_cv[train_idx], X_cv[val_idx]
        y_tr_raw, y_va_raw = y_cv[train_idx], y_cv[val_idx]

        # Scale + PCA (fit on train only)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        X_va_s = scaler.transform(X_va_raw)

        n_comp = min(pca_components, X_tr_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_s)
        X_va_pca = pca.transform(X_va_s)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        # Sequences
        X_tr_seq, y_tr_seq = create_sequences(X_tr_pca, y_tr_raw, lookback)
        X_va_seq, y_va_seq = create_sequences(X_va_pca, y_va_raw, lookback)
        print(f"  train seq shape: {X_tr_seq.shape}")

        clf = RevisedTCNClassifier(
            task_type="classification",
            num_channels=channels,
            kernel_size=3,
            dropout=0.2,
            lr=lr,
            epochs=epochs,
            batch_size=64,
            lookback=lookback,
            patience=patience,
        )
        clf.fit(X_tr_seq, y_tr_seq)

        # Threshold strategies
        proba = clf.predict_proba(X_va_seq)
        pred_05 = (proba >= 0.5).astype(np.int32)
        pred_cal = (proba >= clf._cls_threshold).astype(np.int32)

        try:
            auc = roc_auc_score(y_va_seq, proba)
        except ValueError:
            auc = np.nan

        a05 = accuracy_score(y_va_seq, pred_05)
        f05 = f1_score(y_va_seq, pred_05, average="macro", zero_division=0)
        acal = accuracy_score(y_va_seq, pred_cal)
        fcal = f1_score(y_va_seq, pred_cal, average="macro", zero_division=0)

        model_acc_05.append(float(a05))
        model_f1_05.append(float(f05))
        model_auc.append(float(auc))
        model_acc_cal.append(float(acal))
        model_f1_cal.append(float(fcal))

        # Baseline: majority
        majority = int(np.round(np.mean(y_tr_seq)) >= 0.5)
        pred_maj = np.full_like(y_va_seq, majority, dtype=np.int32)
        maj_acc.append(float(accuracy_score(y_va_seq, pred_maj)))
        maj_f1.append(float(f1_score(y_va_seq, pred_maj, average="macro", zero_division=0)))
        try:
            maj_auc.append(float(roc_auc_score(y_va_seq, pred_maj)))
        except ValueError:
            maj_auc.append(np.nan)

        # Baseline: momentum label persistence
        pred_mom = y_va_raw[lookback - 1:-1].astype(np.int32)
        mom_acc.append(float(accuracy_score(y_va_seq, pred_mom)))
        mom_f1.append(float(f1_score(y_va_seq, pred_mom, average="macro", zero_division=0)))
        try:
            mom_auc.append(float(roc_auc_score(y_va_seq, pred_mom)))
        except ValueError:
            mom_auc.append(np.nan)

        print(
            f"  Model thr=0.5: acc={a05:.4f}, f1={f05:.4f}, auc={auc:.4f} | "
            f"thr=cal({clf._cls_threshold:.4f}): acc={acal:.4f}, f1={fcal:.4f}"
        )

    print("\n=== 5-Fold Summary (mean ± std) ===")
    print(
        f"TCN (thr=0.5): acc={np.nanmean(model_acc_05):.4f}±{np.nanstd(model_acc_05):.4f}, "
        f"f1={np.nanmean(model_f1_05):.4f}±{np.nanstd(model_f1_05):.4f}, "
        f"auc={np.nanmean(model_auc):.4f}±{np.nanstd(model_auc):.4f}"
    )
    print(
        f"TCN (thr=cal): acc={np.nanmean(model_acc_cal):.4f}±{np.nanstd(model_acc_cal):.4f}, "
        f"f1={np.nanmean(model_f1_cal):.4f}±{np.nanstd(model_f1_cal):.4f}, "
        f"auc={np.nanmean(model_auc):.4f}±{np.nanstd(model_auc):.4f}"
    )
    print(
        f"Majority:      acc={np.nanmean(maj_acc):.4f}±{np.nanstd(maj_acc):.4f}, "
        f"f1={np.nanmean(maj_f1):.4f}±{np.nanstd(maj_f1):.4f}, "
        f"auc={np.nanmean(maj_auc):.4f}±{np.nanstd(maj_auc):.4f}"
    )
    print(
        f"Momentum:      acc={np.nanmean(mom_acc):.4f}±{np.nanstd(mom_acc):.4f}, "
        f"f1={np.nanmean(mom_f1):.4f}±{np.nanstd(mom_f1):.4f}, "
        f"auc={np.nanmean(mom_auc):.4f}±{np.nanstd(mom_auc):.4f}"
    )

    if holdout_frac <= 0:
        return

    print(
        f"\n=== Holdout test (terminal {holdout_frac:.0%} rows; "
        f"Scaler+PCA fit on train+val only, same chronology as TCN_Revised) ==="
    )
    print(f"  Train+val rows: {split_idx}  |  Holdout rows: {len(X_raw) - split_idx}")

    scaler_h = StandardScaler()
    scaler_h.fit(X_raw[:split_idx])
    X_all_s = scaler_h.transform(X_raw)
    n_comp_h = min(pca_components, X_all_s.shape[1])
    pca_h = PCA(n_components=n_comp_h, random_state=42)
    pca_h.fit(X_all_s[:split_idx])
    X_all_pca = pca_h.transform(X_all_s)
    print(f"  PCA (train+val) explained variance: {pca_h.explained_variance_ratio_.sum():.3f}")

    k_train_end = split_idx - lookback - 1
    k_test_start = split_idx - lookback
    k_test_end = len(X_all_pca) - lookback - 1
    X_tr_seq_h, y_tr_seq_h = create_sequences_index_range(
        X_all_pca, y, lookback, 0, k_train_end
    )
    X_te_seq_h, y_te_seq_h = create_sequences_index_range(
        X_all_pca, y, lookback, k_test_start, k_test_end
    )
    print(f"  train seq: {X_tr_seq_h.shape}  |  holdout seq: {X_te_seq_h.shape}")

    torch.manual_seed(43)
    np.random.seed(43)
    clf_h = RevisedTCNClassifier(
        task_type="classification",
        num_channels=channels,
        kernel_size=3,
        dropout=0.2,
        lr=lr,
        epochs=epochs,
        batch_size=64,
        lookback=lookback,
        patience=patience,
    )
    clf_h.fit(X_tr_seq_h, y_tr_seq_h)
    proba_h = clf_h.predict_proba(X_te_seq_h)
    pred_05_h = (proba_h >= 0.5).astype(np.int32)
    pred_cal_h = (proba_h >= clf_h._cls_threshold).astype(np.int32)
    try:
        auc_h = roc_auc_score(y_te_seq_h, proba_h)
    except ValueError:
        auc_h = np.nan

    maj_h = int(np.round(np.mean(y_tr_seq_h)) >= 0.5)
    pred_maj_h = np.full_like(y_te_seq_h, maj_h, dtype=np.int32)
    g0 = split_idx
    mom_pred_h = y[g0 - 1 : g0 - 1 + len(y_te_seq_h)].astype(np.int32)

    a05 = accuracy_score(y_te_seq_h, pred_05_h)
    f05 = f1_score(y_te_seq_h, pred_05_h, average="macro", zero_division=0)
    acal = accuracy_score(y_te_seq_h, pred_cal_h)
    fcal = f1_score(y_te_seq_h, pred_cal_h, average="macro", zero_division=0)
    amaj = accuracy_score(y_te_seq_h, pred_maj_h)
    fmaj = f1_score(y_te_seq_h, pred_maj_h, average="macro", zero_division=0)
    amom = accuracy_score(y_te_seq_h, mom_pred_h)
    fmom = f1_score(y_te_seq_h, mom_pred_h, average="macro", zero_division=0)

    print(
        f"TCN holdout (thr=0.5): acc={a05:.4f}, f1={f05:.4f}, auc={auc_h:.4f}"
    )
    print(
        f"TCN holdout (thr=cal={clf_h._cls_threshold:.4f}): "
        f"acc={acal:.4f}, f1={fcal:.4f}, auc={auc_h:.4f}"
    )
    print(f"Majority baseline:       acc={amaj:.4f}, f1={fmaj:.4f}")
    print(f"Momentum baseline:       acc={amom:.4f}, f1={fmom:.4f}")


if __name__ == "__main__":
    main()
