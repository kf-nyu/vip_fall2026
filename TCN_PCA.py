# -------------------------------------------------
# TCN_PCA – VIP Resubmission Compliant Pipeline
# -------------------------------------------------
# This script implements the preprocessing pipeline described in the
# user request, adding PCA (5 factors) on macro variables (31 FRED‑MD series
# plus four daily Yahoo macro returns: corn, soy, UUP, CAD=X), return‑based
# features, and alt‑data integration. It prepares data for a Temporal
# Convolutional Network (TCN) model.
# -------------------------------------------------

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# 1️⃣  LOAD DATA (placeholder – replace with actual paths)
# ------------------------------------------------------------------
# Expected inputs:
#   df        – DataFrame with daily price series for wheat ("price") and
#                alt‑data columns: oil, gold, dxy, vix; plus Yahoo levels
#                corn, soy, uup, cad for macro PCA block.
#   macro_df  – DataFrame with 31 macro variables (FRED‑MD) already
#                transformed (t‑code) and forward‑filled to daily (Yahoo
#                macro daily returns are appended inside the macro PCA block).
# Adjust the paths below to your environment.

# ------------------------------------------------------------------
# 1️⃣  DATA LOADING – use pre‑processed arrays if available, otherwise fall back to raw CSVs
# ------------------------------------------------------------------
import pathlib
import sys

_hub = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_hub / "TCN"))

horizon_days = int(os.getenv("TCN_HORIZON_DAYS", "1"))
DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"
PROCESSED = DATA_ROOT / "processed"
if (PROCESSED / "X_train.npy").exists():
    # Load pre‑processed training / validation arrays (no further preprocessing needed)
    X_train = np.load(PROCESSED / "X_train.npy")
    X_val   = np.load(PROCESSED / "X_val.npy")
    y_train = np.load(PROCESSED / "y_train.npy")
    y_val   = np.load(PROCESSED / "y_val.npy")
    # These arrays already contain the concatenated price, alt‑data, and PCA‑reduced macro features.
    # We'll skip the df‑based pipeline and jump straight to the TCN sequence creation.
    df = None  # mark that we are using pre‑processed data
else:
    # Fallback: load raw wheat futures and macro vintages natively
    import yfinance as yf
    from TCN_Revised import load_fred_md, apply_reporting_delay_and_tcodes

    wheat_path = DATA_ROOT / "wheat-futures" / "wheat_futures_daily.csv"
    df_raw = pd.read_csv(wheat_path, parse_dates=["date"], index_col="date")
    df_raw.rename(columns={"Close": "price"}, inplace=True)
    df_raw.index = pd.to_datetime(df_raw.index, utc=True).tz_localize(None)

    print("Downloading alt-data from Yahoo Finance...")
    tickers = {
        "oil": "CL=F",
        "gold": "GC=F",
        "dxy": "DX-Y.NYB",
        "soy": "ZS=F",
        "vix": "^VIX",
        "corn": "ZC=F",
        "uup": "UUP",
        "cad": "CAD=X",
    }
    start_str = df_raw.index.min().strftime("%Y-%m-%d")
    # Yahoo end date is exclusive in practice; +1 day matches `TCN_PCA_WHEAT_ONLY._yahoo_closes_aligned`.
    end_str = (df_raw.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    target_idx = pd.DatetimeIndex(df_raw.index).normalize()

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

    # Batch yf.download fails when any ticker errors; CAD=X often returns empty via yfinance
    # even though the pair is listed—try USDCAD=X and optional YAHOO_CAD_FALLBACKS.
    cad_alts = [
        x.strip()
        for x in os.getenv("YAHOO_CAD_FALLBACKS", "CAD=X,USDCAD=X").split(",")
        if x.strip()
    ]
    alt_cols: dict[str, np.ndarray] = {}
    for k, sym in tickers.items():
        s = _history_close(sym)
        if s.empty and k == "cad":
            for alt in cad_alts:
                if alt == sym:
                    continue
                s = _history_close(alt)
                if not s.empty:
                    break
        if s.empty:
            v = pd.Series(np.nan, index=target_idx)
        else:
            v = s.reindex(target_idx).ffill().bfill()
        alt_cols[k] = v.to_numpy(dtype=np.float64)

    alt_df = pd.DataFrame(alt_cols, index=df_raw.index)
    df_raw = df_raw.join(alt_df, how="left")

    print("Loading FRED-MD macro vintages and applying T-codes...")
    fred_train_dir = str(DATA_ROOT / "fred-md" / "Historical FRED-MD Vintages Final")
    fred_val_dir = str(DATA_ROOT / "fred-md" / "Historical-vintages-of-FRED-MD-2015-01-to-2024-12")

    from TCN_Revised import load_fred_md, apply_reporting_delay_and_tcodes, forward_fill_macro_to_daily

    df_macro_raw = pd.concat([load_fred_md(fred_train_dir), load_fred_md(fred_val_dir)]).sort_index()
    df_macro_raw = df_macro_raw[~df_macro_raw.index.duplicated(keep='last')]
    macro_df = apply_reporting_delay_and_tcodes(df_macro_raw)
    macro_df.index.name = "date"

    # Merge correctly using TCN_Revised logic (outer join -> ffill -> filter daily)
    df_raw = df_raw.reset_index()
    df = forward_fill_macro_to_daily(macro_df, df_raw)
    df = df.set_index("date")
    df = df.dropna()


# ------------------------------------------------------------------
# 2️⃣  MERGE MACRO (forward fill monthly → daily) & ALIGN
# ------------------------------------------------------------------
# Ensure df and macro_df share the same index (Date). Missing dates are
# forward‑filled, then rows with any remaining NA are dropped.

def merge_data(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merge price/alt‑data with macro variables and clean NaNs."""
    # Align on index, forward‑fill macro, then join
    macro_filled = macro_df.ffill()
    merged = df.join(macro_filled, how="left")
    merged = merged.ffill().dropna()
    return merged

# ------------------------------------------------------------------
# 3️⃣  FEATURE ENGINEERING
# ------------------------------------------------------------------
# Yahoo series whose 1d log returns enter the macro StandardScaler + PCA block (with FRED‑MD).
MACRO_YAHOO_LEVELS = ["corn", "soy", "uup", "cad"]
MACRO_YAHOO_FEATS = [f"{c}_macro_ret1" for c in MACRO_YAHOO_LEVELS]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create return, volatility, momentum, and alt‑data features.

    Assumes columns:
        price, oil, gold, dxy, vix, corn, soy, uup, cad (Yahoo closes for the last four).
    """
    # ----- PRICE FEATURES (log returns) -----
    df["ret_1d"] = np.log(df["price"]).diff()
    df["ret_3d"] = np.log(df["price"]).diff(3)
    df["ret_5d"] = np.log(df["price"]).diff(5)

    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()

    df["mom_5"] = df["price"] / df["price"].shift(5) - 1
    df["mom_10"] = df["price"] / df["price"].shift(10) - 1

    # ----- ALT‑DATA FEATURES -----
    alt_cols = ["oil", "gold", "dxy", "vix"]
    for col in alt_cols:
        # If a downloaded alt series is missing, fall back to neutral features.
        if col not in df.columns or df[col].isna().all():
            df[f"{col}_ret1"] = 0.0
            df[f"{col}_ret3"] = 0.0
            df[f"{col}_vol5"] = 0.0
            continue

        safe_series = df[col].replace(0, np.nan).ffill().bfill()
        if safe_series.isna().all():
            df[f"{col}_ret1"] = 0.0
            df[f"{col}_ret3"] = 0.0
            df[f"{col}_vol5"] = 0.0
            continue

        df[f"{col}_ret1"] = np.log(safe_series).diff()
        df[f"{col}_ret3"] = np.log(safe_series).diff(3)
        df[f"{col}_vol5"] = df[f"{col}_ret1"].rolling(5).std()

    for col in MACRO_YAHOO_LEVELS:
        feat = f"{col}_macro_ret1"
        if col not in df.columns or df[col].isna().all():
            df[feat] = 0.0
            continue
        safe_series = df[col].replace(0, np.nan).ffill().bfill()
        if safe_series.isna().all():
            df[feat] = 0.0
            continue
        df[feat] = np.log(safe_series).diff()

    # ----- CLEAN -----
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ------------------------------------------------------------------
# 4️⃣  TARGET (binary direction based on 3‑day return)
# ------------------------------------------------------------------
def create_sequences_index_range(X, y, lookback, k_min, k_max):
    """Windows X[k:k+lookback] predicting y[k+lookback] for k in [k_min, k_max]."""
    Xs, ys = [], []
    for k in range(k_min, k_max + 1):
        if k < 0 or k + lookback > len(X):
            continue
        Xs.append(X[k : k + lookback])
        ys.append(y[k + lookback])
    return np.array(Xs), np.array(ys)


def create_target(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Binary target for horizon direction.

    target[t] = 1 if log(price[t + horizon] / price[t]) > 0 else 0.
    This is leakage-safe because the future return is only used as label.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    out = df.copy()
    fwd_ret = np.log(out["price"].shift(-horizon_days) / out["price"])
    out["target"] = (fwd_ret > 0).astype(int)
    # Drop tail rows where future label is unavailable.
    out = out.iloc[:-horizon_days] if horizon_days > 0 else out
    return out

# ------------------------------------------------------------------
# 5️⃣  FEATURE GROUPS
# ------------------------------------------------------------------
price_feats = [
    "ret_1d", "ret_3d", "ret_5d",
    "vol_5d", "vol_10d",
    "mom_5", "mom_10",
]

alt_cols = ["oil", "gold", "dxy", "vix"]
alt_feats = []
for col in alt_cols:
    alt_feats += [f"{col}_ret1", f"{col}_ret3", f"{col}_vol5"]

# ------------------------------------------------------------------
# 5️⃣  PREPARE FEATURE MATRICES – handle raw DataFrame or pre‑processed arrays
# ------------------------------------------------------------------
if df is not None:
    df = engineer_features(df)
    df = create_target(df, horizon_days=horizon_days)
    print(f"Target definition: t+{horizon_days} direction (log-return > 0)")
    # Macro columns should be the numeric FRED-MD variables only.
    # Avoid non-numeric merge artifacts such as YearMonth (Period dtype).
    from TCN_Revised import SELECTED_FEATURES
    macro_cols = [c for c in SELECTED_FEATURES if c in df.columns] + [
        f for f in MACRO_YAHOO_FEATS if f in df.columns
    ]
    X_price = df[price_feats].values
    X_alt   = df[alt_feats].values
    X_macro = df[macro_cols].values
    y       = df["target"].values
else:
    # Pre‑processed arrays: stack train+val for a full timeline when val file exists.
    total_price = len(price_feats)
    total_alt   = len(alt_feats)
    if len(X_val) > 0:
        X_price = np.vstack([X_train[:, :total_price], X_val[:, :total_price]])
        X_alt = np.vstack(
            [X_train[:, total_price : total_price + total_alt],
             X_val[:, total_price : total_price + total_alt]]
        )
        X_macro = np.vstack(
            [X_train[:, total_price + total_alt :],
             X_val[:, total_price + total_alt :]]
        )
        y = np.concatenate([y_train, y_val])
    else:
        X_price = X_train[:, :total_price]
        X_alt   = X_train[:, total_price:total_price + total_alt]
        X_macro = X_train[:, total_price + total_alt:]
        y       = y_train

# ------------------------------------------------------------------
# 6️⃣  TRAIN / TEST SPLIT (TimeSeriesSplit)
# ------------------------------------------------------------------
lookback = 30  # days per TCN sample
n_splits = 5
channels_str = os.getenv("TCN_CHANNELS", "128,32")
channels = [int(x.strip()) for x in channels_str.split(",") if x.strip()]
lr = float(os.getenv("TCN_LR", "3e-4"))
epochs = int(os.getenv("TCN_EPOCHS", "30"))
horizon_days = int(os.getenv("TCN_HORIZON_DAYS", "1"))
holdout_frac = float(os.getenv("TCN_HOLDOUT_FRAC", "0.15"))
patience = int(os.getenv("TCN_PATIENCE", "20"))
print(f"TCN channels: {channels}")
print(f"TCN early_stop_patience: {patience} (set TCN_PATIENCE)")
print(f"TCN learning rate: {lr}")
print(f"TCN epochs: {epochs}")
print(f"Target horizon days: t+{horizon_days}")
if holdout_frac > 0:
    print(f"TCN holdout_frac={holdout_frac} (CV on train+val only; final eval on tail)")
else:
    print("TCN holdout_frac=0 (CV on full series; no terminal holdout block)")

ts_cv = TimeSeriesSplit(n_splits=n_splits)

if len(X_price) == 0:
    debug_counts = {}
    if df is not None:
        key_cols = ["price"] + [
            c for c in ["oil", "gold", "dxy", "vix", "corn", "soy", "uup", "cad"] if c in df.columns
        ]
        debug_counts = {c: int(df[c].isna().sum()) for c in key_cols}
    raise ValueError(
        "No samples available after preprocessing (X_price is empty). "
        f"Check data alignment and NaN handling. Debug NaN counts: {debug_counts}"
    )
split_idx = len(X_price)
if holdout_frac > 0:
    test_size_h = max(1, int(len(X_price) * holdout_frac))
    split_idx = len(X_price) - test_size_h
    if split_idx <= n_splits + lookback + 1:
        raise ValueError(
            f"Train+val too short after holdout (split_idx={split_idx}). "
            "Lower TCN_HOLDOUT_FRAC or n_splits."
        )

Xp_cv = X_price[:split_idx] if holdout_frac > 0 else X_price
Xa_cv = X_alt[:split_idx] if holdout_frac > 0 else X_alt
Xm_cv = X_macro[:split_idx] if holdout_frac > 0 else X_macro
y_cv = y[:split_idx] if holdout_frac > 0 else y

if len(Xp_cv) <= n_splits:
    raise ValueError(
        f"Not enough samples for TimeSeriesSplit: n_samples={len(Xp_cv)}, "
        f"n_splits={n_splits}. Reduce n_splits or verify preprocessing output."
    )

# ------------------------------------------------------------------
# 7️⃣  CROSS‑VALIDATION LOOP (illustrative – adapt as needed)
# ------------------------------------------------------------------
model_accs_05, model_f1s_05, model_aucs = [], [], []
model_accs_cal, model_f1s_cal = [], []
maj_accs, maj_f1s, maj_aucs = [], [], []
mom_accs, mom_f1s, mom_aucs = [], [], []

for fold, (train_idx, val_idx) in enumerate(ts_cv.split(Xp_cv)):
    print(f"\n=== Fold {fold+1} ===")

    # ----- Split raw arrays -----
    Xp_tr, Xp_va = Xp_cv[train_idx], Xp_cv[val_idx]
    Xa_tr, Xa_va = Xa_cv[train_idx], Xa_cv[val_idx]
    Xm_tr, Xm_va = Xm_cv[train_idx], Xm_cv[val_idx]
    y_tr,   y_va = y_cv[train_idx], y_cv[val_idx]

    # ----- Scaling (no leakage) -----
    scaler_price = StandardScaler()
    scaler_alt   = StandardScaler()
    scaler_macro = StandardScaler()

    Xp_tr_s = scaler_price.fit_transform(Xp_tr)
    Xp_va_s = scaler_price.transform(Xp_va)

    Xa_tr_s = scaler_alt.fit_transform(Xa_tr)
    Xa_va_s = scaler_alt.transform(Xa_va)

    Xm_tr_s = scaler_macro.fit_transform(Xm_tr)
    Xm_va_s = scaler_macro.transform(Xm_va)

    # ----- PCA on macro (train‑only) -----
    pca = PCA(n_components=5, random_state=42)
    Xm_tr_pca = pca.fit_transform(Xm_tr_s)
    Xm_va_pca = pca.transform(Xm_va_s)
    print("PCA variance explained:", np.round(pca.explained_variance_ratio_.sum(), 3))

    # ----- Final feature matrix -----
    X_tr = np.concatenate([Xp_tr_s, Xa_tr_s, Xm_tr_pca], axis=1)
    X_va = np.concatenate([Xp_va_s, Xa_va_s, Xm_va_pca], axis=1)
    print("Feature shape (train):", X_tr.shape)

    # ----- Create TCN sequences -----
    def create_sequences(X, y, lookback=lookback):
        Xs, ys = [], []
        for i in range(len(X) - lookback):
            Xs.append(X[i:i+lookback])
            ys.append(y[i+lookback])
        return np.array(Xs), np.array(ys)

    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr)
    X_va_seq, y_va_seq = create_sequences(X_va, y_va)
    print("TCN input shape (train):", X_tr_seq.shape)

    # ------------------------------------------------------------------
    # 8️⃣  Train/evaluate using TCN_Revised model class directly
    # ------------------------------------------------------------------
    from TCN_Revised import TCNClassifier as RevisedTCNClassifier

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

    clf.fit(X_tr_seq, y_tr_seq.astype(np.int32))
    # Compare threshold strategies:
    # 1) fixed 0.5
    # 2) validation-calibrated threshold learned in clf.fit (clf._cls_threshold)
    y_true = y_va_seq.astype(np.int32)
    proba = clf.predict_proba(X_va_seq)
    pred_05 = (proba >= 0.5).astype(np.int32)
    pred_cal = (proba >= clf._cls_threshold).astype(np.int32)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan

    metrics_05 = {
        "accuracy": accuracy_score(y_true, pred_05),
        "f1_macro": f1_score(y_true, pred_05, average="macro", zero_division=0),
        "roc_auc": auc,
        "predictions": pred_05,
    }
    metrics_cal = {
        "accuracy": accuracy_score(y_true, pred_cal),
        "f1_macro": f1_score(y_true, pred_cal, average="macro", zero_division=0),
        "roc_auc": auc,
        "predictions": pred_cal,
    }

    print(
        "Validation metrics (threshold=0.5):",
        f"acc={metrics_05['accuracy']:.4f}",
        f"f1_macro={metrics_05['f1_macro']:.4f}",
        f"roc_auc={metrics_05['roc_auc']:.4f}",
    )
    print(
        "Validation metrics (threshold=calibrated):",
        f"acc={metrics_cal['accuracy']:.4f}",
        f"f1_macro={metrics_cal['f1_macro']:.4f}",
        f"roc_auc={metrics_cal['roc_auc']:.4f}",
        f"cal_t={clf._cls_threshold:.4f}",
    )
    print("Validation predictions shape:", metrics_05["predictions"].shape)

    # Track model metrics
    model_accs_05.append(float(metrics_05["accuracy"]))
    model_f1s_05.append(float(metrics_05["f1_macro"]))
    model_aucs.append(float(metrics_05["roc_auc"]))
    model_accs_cal.append(float(metrics_cal["accuracy"]))
    model_f1s_cal.append(float(metrics_cal["f1_macro"]))

    # ----- Baseline 1: majority class from training labels -----
    majority_class = int(np.round(np.mean(y_tr_seq)) >= 0.5)
    maj_pred = np.full_like(y_va_seq, fill_value=majority_class, dtype=np.int32)
    maj_acc = accuracy_score(y_va_seq, maj_pred)
    maj_f1 = f1_score(y_va_seq, maj_pred, average="macro", zero_division=0)
    try:
        maj_auc = roc_auc_score(y_va_seq, maj_pred)
    except ValueError:
        maj_auc = np.nan
    maj_accs.append(float(maj_acc))
    maj_f1s.append(float(maj_f1))
    maj_aucs.append(float(maj_auc))

    # ----- Baseline 2: naive momentum (predict previous day's direction) -----
    # y_va_seq[k] corresponds to y_va[lookback + k], so previous label is y_va[lookback + k - 1].
    mom_pred = y_va[lookback - 1:-1].astype(np.int32)
    mom_acc = accuracy_score(y_va_seq, mom_pred)
    mom_f1 = f1_score(y_va_seq, mom_pred, average="macro", zero_division=0)
    try:
        mom_auc = roc_auc_score(y_va_seq, mom_pred)
    except ValueError:
        mom_auc = np.nan
    mom_accs.append(float(mom_acc))
    mom_f1s.append(float(mom_f1))
    mom_aucs.append(float(mom_auc))

    print(
        "Baselines:",
        f"majority acc={maj_acc:.4f}, f1={maj_f1:.4f}, auc={maj_auc:.4f}",
        f"| momentum acc={mom_acc:.4f}, f1={mom_f1:.4f}, auc={mom_auc:.4f}",
    )

print("\n=== 5-Fold Summary (mean ± std) ===")
print(
    f"TCN classifier (thr=0.5): "
    f"acc={np.nanmean(model_accs_05):.4f}±{np.nanstd(model_accs_05):.4f}, "
    f"f1={np.nanmean(model_f1s_05):.4f}±{np.nanstd(model_f1s_05):.4f}, "
    f"auc={np.nanmean(model_aucs):.4f}±{np.nanstd(model_aucs):.4f}"
)
print(
    f"TCN classifier (thr=cal): "
    f"acc={np.nanmean(model_accs_cal):.4f}±{np.nanstd(model_accs_cal):.4f}, "
    f"f1={np.nanmean(model_f1s_cal):.4f}±{np.nanstd(model_f1s_cal):.4f}, "
    f"auc={np.nanmean(model_aucs):.4f}±{np.nanstd(model_aucs):.4f}"
)
print(
    f"Majority baseline:      "
    f"acc={np.nanmean(maj_accs):.4f}±{np.nanstd(maj_accs):.4f}, "
    f"f1={np.nanmean(maj_f1s):.4f}±{np.nanstd(maj_f1s):.4f}, "
    f"auc={np.nanmean(maj_aucs):.4f}±{np.nanstd(maj_aucs):.4f}"
)
print(
    f"Momentum baseline:      "
    f"acc={np.nanmean(mom_accs):.4f}±{np.nanstd(mom_accs):.4f}, "
    f"f1={np.nanmean(mom_f1s):.4f}±{np.nanstd(mom_f1s):.4f}, "
    f"auc={np.nanmean(mom_aucs):.4f}±{np.nanstd(mom_aucs):.4f}"
)

if holdout_frac > 0:
    print(
        f"\n=== Holdout test (terminal {holdout_frac:.0%} rows; "
        f"Scalers+PCA fit on train+val only, same chronology as TCN_Revised) ==="
    )
    print(f"  Train+val rows: {split_idx}  |  Holdout rows: {len(X_price) - split_idx}")

    scaler_price_h = StandardScaler()
    scaler_alt_h = StandardScaler()
    scaler_macro_h = StandardScaler()
    scaler_price_h.fit(X_price[:split_idx])
    scaler_alt_h.fit(X_alt[:split_idx])
    scaler_macro_h.fit(X_macro[:split_idx])
    Xp_all_s = scaler_price_h.transform(X_price)
    Xa_all_s = scaler_alt_h.transform(X_alt)
    Xm_all_s = scaler_macro_h.transform(X_macro)
    pca_h = PCA(n_components=5, random_state=42)
    pca_h.fit(Xm_all_s[:split_idx])
    Xm_all_pca = pca_h.transform(Xm_all_s)
    X_all_feat = np.concatenate([Xp_all_s, Xa_all_s, Xm_all_pca], axis=1)
    print("  PCA (train+val) variance explained:", np.round(pca_h.explained_variance_ratio_.sum(), 3))

    k_train_end = split_idx - lookback - 1
    k_test_start = split_idx - lookback
    k_test_end = len(X_all_feat) - lookback - 1
    X_tr_seq_h, y_tr_seq_h = create_sequences_index_range(
        X_all_feat, y, lookback, 0, k_train_end
    )
    X_te_seq_h, y_te_seq_h = create_sequences_index_range(
        X_all_feat, y, lookback, k_test_start, k_test_end
    )
    print(f"  train seq: {X_tr_seq_h.shape}  |  holdout seq: {X_te_seq_h.shape}")

    from TCN_Revised import TCNClassifier as RevisedTCNClassifierHoldout

    torch.manual_seed(43)
    np.random.seed(43)
    clf_h = RevisedTCNClassifierHoldout(
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
    clf_h.fit(X_tr_seq_h, y_tr_seq_h.astype(np.int32))
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

    print(
        f"TCN holdout (thr=0.5): acc={accuracy_score(y_te_seq_h, pred_05_h):.4f}, "
        f"f1={f1_score(y_te_seq_h, pred_05_h, average='macro', zero_division=0):.4f}, "
        f"auc={auc_h:.4f}"
    )
    print(
        f"TCN holdout (thr=cal={clf_h._cls_threshold:.4f}): "
        f"acc={accuracy_score(y_te_seq_h, pred_cal_h):.4f}, "
        f"f1={f1_score(y_te_seq_h, pred_cal_h, average='macro', zero_division=0):.4f}, "
        f"auc={auc_h:.4f}"
    )
    print(
        f"Majority baseline:       acc={accuracy_score(y_te_seq_h, pred_maj_h):.4f}, "
        f"f1={f1_score(y_te_seq_h, pred_maj_h, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Momentum baseline:       acc={accuracy_score(y_te_seq_h, mom_pred_h):.4f}, "
        f"f1={f1_score(y_te_seq_h, mom_pred_h, average='macro', zero_division=0):.4f}"
    )

# ------------------------------------------------------------------
# NOTE:
# • Replace the placeholder data loading section with your actual CSV / DB reads.
# • Adjust hyper‑parameters (lookback, channels, epochs, batch size) as needed.
# • The script follows the VIP R‑submission checklist: no leakage, PCA on macro,
#   returns‑based features, and a ready‑to‑train TCN model.
# ------------------------------------------------------------------
