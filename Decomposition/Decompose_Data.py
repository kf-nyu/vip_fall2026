
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PyEMD import EMD as PyEMD_EMD, EEMD as PyEMD_EEMD, CEEMDAN as PyEMD_CEEMDAN

# --- Configuration ---
OUTPUT_DIR = 'Decomposed_Data'

def load_daily_wheat_prices():
    """Load daily wheat futures close prices and return (dates, prices)."""
    # Use path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wheat_path = os.path.join(script_dir, '..', 'data', 'wheat-futures', 'wheat_futures_daily.csv')

    if not os.path.exists(wheat_path):
        print(f"Data not found at {wheat_path}")
        return None, None

    print(f"Loading data from {wheat_path}...")
    df = pd.read_csv(wheat_path)
    # Ensure date column handled correctly (check column names)
    # Based on main script: 'date' and 'Close'
    if 'date' in df.columns:
        date_col = 'date'
    elif 'Date' in df.columns:
        date_col = 'Date'
    else:
        print("Date column not found.")
        return None, None
        
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df.sort_values(date_col, inplace=True)
    
    return df[date_col].values, df['Close'].values

def save_components(name, imfs_T, dates):
    """Save decomposed components to CSV."""
    # imfs_T shape: (n_samples, n_imfs)
    n_imfs = imfs_T.shape[1]
    columns = [f'IMF_{i+1}' for i in range(n_imfs)]
    
    df = pd.DataFrame(imfs_T, columns=columns)
    df.insert(0, 'Date', dates)
    
    filepath = os.path.join(OUTPUT_DIR, f'{name}_Components.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved {name} components to {filepath}")
    
    # Optional: Plot
    plot_path = os.path.join(OUTPUT_DIR, f'{name}_Vis.png')
    plot_decomposition(imfs_T, dates, name, plot_path)

def plot_decomposition(imfs_T, dates, name, save_path):
    n_imfs = imfs_T.shape[1]
    fig, axes = plt.subplots(n_imfs, 1, figsize=(12, 2 * n_imfs), sharex=True)
    if n_imfs == 1: axes = [axes]
    
    for i in range(n_imfs):
        axes[i].plot(dates, imfs_T[:, i])
        axes[i].set_ylabel(f'IMF {i+1}')
        axes[i].grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Date')
    plt.suptitle(f'{name} Decomposition')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_ceemd(prices, trials=10, noise_width=0.2, seed=42):
    """
    Replicate the custom CEEMD logic from CEEMDELMRegressor.
    (Avg of +noise and -noise pairs).
    """
    print(f"  Running CEEMD custom logic (trials={trials}, noise={noise_width})...")
    n = len(prices)
    noise_std = noise_width * np.std(prices)
    emd = PyEMD_EMD()
    rng = np.random.RandomState(seed)

    all_imfs_list = []

    for t in range(trials):
        noise = rng.normal(0, noise_std, n)
        
        # Positive noise
        imfs_pos = emd.emd(prices + noise) # (n_imfs, n)
        all_imfs_list.append(imfs_pos)
        
        # Negative noise
        imfs_neg = emd.emd(prices - noise)
        all_imfs_list.append(imfs_neg)
        
        if (t+1) % 5 == 0:
            print(f"    Trial {t+1}/{trials} pairs done")

    # Max number of IMFs
    max_n = max(im.shape[0] for im in all_imfs_list)
    
    # Accumulate
    avg_imfs = np.zeros((max_n, n))
    for im in all_imfs_list:
        # Pad with zeros if fewer IMFs
        padded = np.zeros((max_n, n))
        padded[:im.shape[0], :] = im
        avg_imfs += padded
        
    avg_imfs /= len(all_imfs_list)
    return avg_imfs.T # (n_samples, n_imfs)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    dates, prices = load_daily_wheat_prices()
    if prices is None:
        return

    # 1. EMD
    print("\n[1/4] Processing EMD...")
    emd = PyEMD_EMD()
    imfs_emd = emd.emd(prices).T
    save_components('EMD', imfs_emd, dates)

    # 2. EEMD (Using Tuned Parameters: trials=25, noise=0.1)
    print("\n[2/4] Processing EEMD (Tuned)...")
    np.random.seed(42) # Fix seed for EEMD
    eemd = PyEMD_EEMD(trials=25, noise_width=0.1)
    imfs_eemd = eemd.eemd(prices).T
    save_components('EEMD', imfs_eemd, dates)

    # 3. CEEMD (Custom Logic, trials=10)
    print("\n[3/4] Processing CEEMD (Custom Pairwise)...")
    imfs_ceemd = run_ceemd(prices, trials=10, noise_width=0.2, seed=42)
    save_components('CEEMD', imfs_ceemd, dates)

    # 4. CEEMDAN (trials=10, epsilon=0.005)
    print("\n[4/4] Processing CEEMDAN...")
    np.random.seed(42) 
    ceemdan = PyEMD_CEEMDAN(trials=10, epsilon=0.005)
    imfs_ceemdan = ceemdan.ceemdan(prices).T
    save_components('CEEMDAN', imfs_ceemdan, dates)

    print("\nDone! All decomposed components saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
