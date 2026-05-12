import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Ensure the root project directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load decomposers and wrapper
from Decomposition.Decomposition_Transformers import CEEMDANDecomposer
from Decomposition.Generic_Decomposition_Forecaster import GenericDecompositionModel
from Decomposition.Decomposition_ELM_VIP_Compliant import load_daily_wheat_prices

def run_favar_benchmark():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(root_dir, 'data', 'processed')
    
    X_monthly_windows = np.load(os.path.join(data_dir, 'X_train.npy'))
    dates_monthly = np.load(os.path.join(data_dir, 'dates_train.npy'), allow_pickle=True)
    
    n_macro = 32
    X_monthly_reshaped = X_monthly_windows.reshape(X_monthly_windows.shape[0], 30, n_macro)
    X_monthly_current = X_monthly_reshaped[:, -1, :]
    
    pca = PCA(n_components=3)
    factors_monthly = pca.fit_transform(X_monthly_current)
    
    df_factors = pd.DataFrame(factors_monthly, columns=['F1', 'F2', 'F3'])
    macro_dates = pd.to_datetime(dates_monthly).tz_localize(None)
    df_factors['MonthStart'] = macro_dates
    df_factors['YearMonth'] = df_factors['MonthStart'].dt.to_period('M')
    df_factors = df_factors.set_index('YearMonth').drop(columns=['MonthStart'])
    df_factors = df_factors[~df_factors.index.duplicated(keep='last')]
    
    wheat_path = os.path.join(root_dir, 'data', 'wheat-futures', 'wheat_futures_daily.csv')
    df_wheat = pd.read_csv(wheat_path)
    df_wheat['date'] = pd.to_datetime(df_wheat['date'], utc=True).dt.tz_localize(None)
    df_wheat = df_wheat.sort_values('date')
    df_wheat['YearMonth'] = df_wheat['date'].dt.to_period('M')
    
    # Merge and Lag
    df_merged = df_wheat.merge(df_factors, on='YearMonth', how='inner')
    df_merged = df_merged.sort_values('date')
    
    df_merged['Lag1'] = df_merged['Close'].shift(1)
    df_merged['Lag2'] = df_merged['Close'].shift(2)
    df_merged = df_merged.dropna()
    
    feature_cols = ['Lag1', 'Lag2', 'F1', 'F2', 'F3']
    X = df_merged[feature_cols].values
    y = df_merged['Close'].values
    dates = df_merged['date'].values
    
    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []
    
    last_preds = None
    last_dates = None
    last_y_true = None
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        r2 = r2_score(y_val, preds)
        r2_scores.append(r2)
        
        if fold == 4:
            last_preds = preds
            last_dates = dates[val_idx]
            last_y_true = y_val
            
    return np.mean(r2_scores), last_preds, last_dates, last_y_true

def run_grand_comparison():
    print("="*70)
    print("  GRAND COMPARISON: CEEMDAN (Linear, ELM, TCN) vs FAVAR")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    dates, prices = load_daily_wheat_prices()
    
    decomposer = CEEMDANDecomposer(trials=50, epsilon=0.005)
    decomp_dir = os.path.join(os.path.dirname(__file__), 'Decomposed_Data')
    decomp_file = os.path.join(decomp_dir, "CEEMDAN_Components.csv")
    
    if os.path.exists(decomp_file):
        print(f"\nLoading pre-decomposed CEEMDAN from {decomp_file}...")
        df_decomp = pd.read_csv(decomp_file)
        imf_cols = [c for c in df_decomp.columns if c.startswith('IMF')]
        imfs = df_decomp[imf_cols].values
    else:
        print("\nDecomposing full signal with CEEMDAN...")
        t0 = time.time()
        imfs = decomposer.decompose(prices)
        print(f"Decomposition complete in {time.time()-t0:.1f}s")
        os.makedirs(decomp_dir, exist_ok=True)
        df_save = pd.DataFrame({'Date': dates, 'Close': prices})
        for i in range(imfs.shape[1]):
            df_save[f'IMF{i+1}'] = imfs[:, i]
        df_save.to_csv(decomp_file, index=False)
    
    backends = ['Linear', 'ELM', 'TCN']
    results = {}
    plot_data = {}
    
    for backend in backends:
        print(f"\n--- Running CEEMDAN-{backend} ---")
        tscv = TimeSeriesSplit(n_splits=5)
        fold_r2s = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(prices)))):
            t_fold = time.time()
            model = GenericDecompositionModel(decomposer=decomposer, backend_name=backend, lookback=30)
            preds, y_test = model.fit_on_imfs(imfs, prices, len(train_idx))
            
            burn_in = model.lookback * 2
            preds = preds[burn_in:]
            y_test = y_test[burn_in:]
            min_len = min(len(y_test), len(preds))
            y_true = y_test[:min_len]
            preds = preds[:min_len]
            
            r2 = r2_score(y_true, preds)
            fold_r2s.append(r2)
            print(f"  Fold {fold+1} R2: {r2:.4f} | Time: {time.time()-t_fold:.1f}s")
            
            if fold == 4:
                test_start = len(train_idx) + model.lookback + burn_in
                dates_aligned = dates[test_start: test_start + min_len]
                plot_data[f"CEEMDAN-{backend}"] = (y_true, preds, dates_aligned)
                
        results[f"CEEMDAN-{backend}"] = np.mean(fold_r2s)
        print(f">> CEEMDAN-{backend} Avg R2: {results[f'CEEMDAN-{backend}']:.4f}")
        
    print(f"\n--- Running FAVAR Benchmark ---")
    favar_r2, f_preds, f_dates, f_true = run_favar_benchmark()
    results['FAVAR'] = favar_r2
    plot_data['FAVAR'] = (f_true, f_preds, f_dates)
    print(f">> FAVAR Avg R2: {favar_r2:.4f}")
    
    print("\n" + "="*50)
    print("  FINAL LEADERBOARD")
    print("="*50)
    for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:<15} | Average R2: {v:.4f}")
    print("="*50)
        
    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    colors = ['orange', 'red', 'purple', 'green']
    models_to_plot = [f"CEEMDAN-{b}" for b in backends] + ["FAVAR"]
    
    for idx, name in enumerate(models_to_plot):
        ax = axes[idx]
        y_t, p_t, d_t = plot_data[name]
        
        plot_len = min(len(y_t), len(p_t), len(d_t))
        y_t, p_t, d_t = y_t[:plot_len], p_t[:plot_len], d_t[:plot_len]
        
        start = max(0, plot_len - 300)
        ax.plot(pd.to_datetime(d_t[start:]), y_t[start:], label='Actual (Wheat Price)', color='blue')
        ax.plot(pd.to_datetime(d_t[start:]), p_t[start:], label=f'{name} Forecast', color=colors[idx], linestyle='--')
        
        fold_5_r2 = r2_score(y_t, p_t)
        ax.set_title(f"{name}  |  Fold 5 R² = {fold_5_r2:.4f}")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'Grand_Model_Comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved comparison plot to {plot_path}")

if __name__ == '__main__':
    run_grand_comparison()
