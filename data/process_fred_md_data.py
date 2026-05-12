"""
Process FRED-MD data for FAVAR model training

This script processes the downloaded FRED-MD historical vintages and current data
according to the preprocessing steps from the VIP requirements.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
import re
warnings.filterwarnings('ignore')

# Feature selection from VIP requirements
SELECTED_FEATURES = [
    "RPI", "W875RX1", "CMRMTSPLx", "IPFPNSS", "USWTRADE", "USTRADE", 
    "BUSLOANS", "CONSPI", "S&P 500", "S&P PE ratio", "FEDFUNDS", 
    "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA", 
    "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", 
    "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx", 
    "EXCAUSx", "OILPRICEx", "PPICMM", "UMCSENTx"
]

def load_vintage_data(folder_path):
    """
    Load data from a folder containing historical vintage CSV files
    
    Args:
        folder_path: Path to the folder containing CSV files
    
    Returns:
        Dictionary mapping vintage dates to DataFrames
    """
    vintages = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder not found: {folder_path}")
        return vintages
    
    # Get all CSV files
    csv_files = sorted(folder.glob("*.csv"))
    
    print(f"  Found {len(csv_files)} CSV files in {folder_path}")
    
    for csv_file in csv_files:
        try:
            # Extract vintage date from filename
            # Formats: "YYYY-MM.csv" or "FRED-MD_YYYYmMM.csv"
            filename = csv_file.stem
            
            # Try to parse date from filename
            date_match = re.search(r'(\d{4})-(\d{2})', filename)
            if not date_match:
                # Try FRED-MD format: FRED-MD_2024m03
                date_match = re.search(r'(\d{4})m(\d{2})', filename)
            
            if date_match:
                year, month = date_match.groups()
                vintage_key = f"{year}-{month}"
            else:
                vintage_key = filename
            
            # Read CSV file - handle the "Transform:" row that appears as first data row
            # The structure is: header row, then "Transform:" row with codes, then actual data
            df = pd.read_csv(csv_file)
            
            # Check if first data row contains "Transform:" and remove it
            if len(df) > 0:
                first_val = str(df.iloc[0, 0]).strip()
                if 'Transform' in first_val or first_val == 'Transform:':
                    df = df.iloc[1:].reset_index(drop=True)
            
            vintages[vintage_key] = df
            
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
            continue
    
    return vintages

def preprocess_fred_md_data(df, selected_features=None, shift_dates=True, lookback_window=30):
    """
    Preprocess FRED-MD data according to VIP requirements
    
    Args:
        df: DataFrame with FRED-MD data
        selected_features: List of feature names to select
        shift_dates: Whether to shift dates by one month
        lookback_window: Number of days for lookback window (not used here, kept for compatibility)
    
    Returns:
        Processed DataFrame
    """
    # Make a copy
    data = df.copy()
    
    # The "Transform:" row should already be removed in load_vintage_data, but check again
    if len(data) > 0:
        first_val = str(data.iloc[0, 0]).strip()
        if 'Transform' in first_val:
            data = data.iloc[1:].reset_index(drop=True)
    
    # Set date column (assuming first column is date or has 'date' in name)
    date_col = None
    for col in data.columns:
        if 'date' in col.lower() or col.lower() == 'sasdate':
            date_col = col
            break
    
    if date_col:
        # Convert to datetime, handling various formats
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        except:
            # Try parsing manually if standard parsing fails
            data[date_col] = pd.to_datetime(data[date_col], format='%m/%d/%Y', errors='coerce')
        
        # Remove rows where date parsing failed
        data = data.dropna(subset=[date_col])
        
        if shift_dates:
            # Shift dates by one month (Forward shift = Lagging)
            # Jan Data (t) becomes Feb Index (t+1). Target(Feb) sees Data(Jan).
            data[date_col] = data[date_col] + pd.DateOffset(months=1)
        
        data = data.set_index(date_col)
        data = data.sort_index()
    
    # Select features if specified
    if selected_features:
        available_features = [f for f in selected_features if f in data.columns]
        missing_features = [f for f in selected_features if f not in data.columns]
        if len(available_features) == 0:
            print(f"    ERROR: None of the selected features found in data!")
            print(f"    Available columns in data: {list(data.columns)[:20]}")
            print(f"    Selected features: {selected_features[:10]}...")
            print(f"    Missing features: {missing_features[:10]}...")
        elif len(missing_features) > 0:
            print(f"    Warning: {len(missing_features)} missing features: {missing_features[:5]}...")
        data = data[available_features]
    
    # Convert to numeric, coercing errors to NaN
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Check data shape before handling NaNs
    if data.shape[0] == 0:
        print(f"    Warning: No rows remaining after date parsing!")
        return pd.DataFrame()
    
    if data.shape[1] == 0:
        print(f"    Warning: No columns remaining!")
        return pd.DataFrame()
    
    # Handle missing values (forward fill, then backward fill)
    # Use new pandas API if available, otherwise use old method
    try:
        data = data.ffill().bfill()
    except:
        try:
            data = data.fillna(method='ffill').fillna(method='bfill')
        except:
            data = data.fillna(method='ffill', limit=None).fillna(method='bfill', limit=None)
    
    # Drop rows where ALL values are NaN (but keep rows with some valid values)
    data = data.dropna(how='all')
    
    # Drop columns that are entirely NaN
    data = data.dropna(axis=1, how='all')
    
    # For remaining NaNs, drop rows only if critical columns are missing
    # But first try to fill remaining NaNs with column mean
    for col in data.columns:
        if data[col].isna().any():
            mean_val = data[col].mean()
            if pd.notna(mean_val):
                data[col] = data[col].fillna(mean_val)
    
    # Final cleanup - drop rows that still have any NaN
    data = data.dropna()
    
    return data

def create_lookback_windows(data, lookback_window=30):
    """
    Create lookback windows for time series data
    
    Args:
        data: DataFrame with time series data (index is date)
        lookback_window: Number of time steps to look back
    
    Returns:
        X: Features with lookback windows (n_samples - lookback_window, n_features * lookback_window)
        y: Targets (n_samples - lookback_window, n_features)
        dates: Dates corresponding to targets (n_samples - lookback_window,)
    """
    n_samples, n_features = data.shape
    
    if n_samples <= lookback_window:
        raise ValueError(f"Not enough samples ({n_samples}) for lookback window ({lookback_window})")
    
    X = []
    y = []
    dates = []
    
    # data.index contains dates
    all_dates = data.index
    
    for i in range(lookback_window, n_samples):
        # Get lookback window
        window = data.iloc[i-lookback_window:i].values.flatten()
        X.append(window)
        # Target is current values
        y.append(data.iloc[i].values)
        # Date for this target
        dates.append(all_dates[i])
    
    return np.array(X), np.array(y), np.array(dates)

def prepare_fred_md_for_favar(train_folder_path, val_folder_path, test_csv_path, 
                                selected_features=None, lookback_window=30):
    """
    Prepare FRED-MD data for FAVAR model training
    
    Args:
        train_folder_path: Path to training historical vintages folder
        val_folder_path: Path to validation historical vintages folder
        test_csv_path: Path to test CSV file
        selected_features: List of features to select
        lookback_window: Lookback window size
    
    Returns:
        Dictionary with train, validation, and test data
    """
    results = {}
    
    print("Loading training data...")
    train_vintages = load_vintage_data(train_folder_path)
    print(f"  Loaded {len(train_vintages)} training vintages")
    
    print("Loading validation data...")
    val_vintages = load_vintage_data(val_folder_path)
    print(f"  Loaded {len(val_vintages)} validation vintages")
    
    print("Loading test data...")
    test_data = pd.read_csv(test_csv_path)
    print(f"  Test data shape: {test_data.shape}")
    
    # Process training data
    print("\nProcessing training data...")
    train_processed = []
    for vintage_name, df in train_vintages.items():
        try:
            processed = preprocess_fred_md_data(df, selected_features, shift_dates=True)
            train_processed.append(processed)
        except Exception as e:
            print(f"  Error processing {vintage_name}: {e}")
    
    if train_processed:
        train_combined = pd.concat(train_processed, axis=0)
        train_combined = train_combined.sort_index()
        train_combined = train_combined.drop_duplicates()
        print(f"  Combined training data shape: {train_combined.shape}")
        
        # Create lookback windows
        X_train, y_train, dates_train = create_lookback_windows(train_combined, lookback_window)
        results['train'] = {'X': X_train, 'y': y_train, 'dates': dates_train, 'data': train_combined}
    
    # Process validation data
    print("\nProcessing validation data...")
    val_processed = []
    for vintage_name, df in val_vintages.items():
        try:
            processed = preprocess_fred_md_data(df, selected_features, shift_dates=True)
            val_processed.append(processed)
        except Exception as e:
            print(f"  Error processing {vintage_name}: {e}")
    
    if val_processed:
        val_combined = pd.concat(val_processed, axis=0)
        val_combined = val_combined.sort_index()
        val_combined = val_combined.drop_duplicates()
        print(f"  Combined validation data shape: {val_combined.shape}")
        
        # Create lookback windows
        X_val, y_val, dates_val = create_lookback_windows(val_combined, lookback_window)
        results['val'] = {'X': X_val, 'y': y_val, 'dates': dates_val, 'data': val_combined}
    
    # Process test data if available
    if test_csv_path:
        print("\nProcessing test data...")
        test_processed = preprocess_fred_md_data(test_data, selected_features, shift_dates=False)
        print(f"  Test data shape: {test_processed.shape}")
        
        # Create lookback windows
        X_test, y_test, dates_test = create_lookback_windows(test_processed, lookback_window)
        results['test'] = {'X': X_test, 'y': y_test, 'dates': dates_test, 'data': test_processed}
    
    # Apply StandardScaler
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    results['train']['X'] = scaler.fit_transform(results['train']['X'])
    results['val']['X'] = scaler.transform(results['val']['X'])
    if 'test' in results:
        results['test']['X'] = scaler.transform(results['test']['X'])
    results['scaler'] = scaler
    
    print("\nData preparation complete!")
    print(f"  Training samples: {results['train']['X'].shape[0]}")
    print(f"  Validation samples: {results['val']['X'].shape[0]}")
    if 'test' in results:
        print(f"  Test samples: {results['test']['X'].shape[0]}")
    
    return results

if __name__ == "__main__":
    # Paths - using unzipped folders
    data_dir = Path("data/fred-md")
    train_folder = data_dir / "Historical FRED-MD Vintages Final"
    val_folder = data_dir / "Historical-vintages-of-FRED-MD-2015-01-to-2024-12"
    test_csv = data_dir / "current_fred-md.csv"
    
    # Check if folders exist
    if not train_folder.exists():
        print(f"Error: Training folder not found: {train_folder}")
        print("Please unzip the training data first. See data/README_FRED_MD.md for instructions.")
        exit(1)
    
    if not val_folder.exists():
        print(f"Error: Validation folder not found: {val_folder}")
        print("Please unzip the validation data first. See data/README_FRED_MD.md for instructions.")
        exit(1)
    
    if not test_csv.exists():
        print(f"Warning: Test file not found: {test_csv}")
        print("You can download it from: https://research.stlouisfed.org/econ/mccracken/fred-databases")
        print("Continuing without test data...")
        test_csv = None
    
    # Prepare data
    if test_csv:
        data = prepare_fred_md_for_favar(
            train_folder, 
            val_folder, 
            test_csv,
            selected_features=SELECTED_FEATURES,
            lookback_window=30
        )
    else:
        # Process without test data
        print("Loading training data...")
        train_vintages = load_vintage_data(train_folder)
        print(f"  Loaded {len(train_vintages)} training vintages")
        
        print("Loading validation data...")
        val_vintages = load_vintage_data(val_folder)
        print(f"  Loaded {len(val_vintages)} validation vintages")
        
        # Process training data
        print("\nProcessing training data...")
        train_processed = []
        for i, (vintage_name, df) in enumerate(train_vintages.items()):
            try:
                if i < 3:  # Debug first 3 vintages
                    print(f"  Processing {vintage_name}...")
                processed = preprocess_fred_md_data(df, SELECTED_FEATURES, shift_dates=True)
                if processed.shape[0] > 0 and processed.shape[1] > 0:
                    train_processed.append(processed)
                    if i < 3:
                        print(f"    Result shape: {processed.shape}")
                else:
                    print(f"    Warning: {vintage_name} resulted in empty dataframe")
            except Exception as e:
                print(f"  Error processing {vintage_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if train_processed:
            train_combined = pd.concat(train_processed, axis=0)
            train_combined = train_combined.sort_index()
            train_combined = train_combined.drop_duplicates()
            print(f"  Combined training data shape: {train_combined.shape}")
            print(f"  Available columns: {list(train_combined.columns)[:10]}...")  # Show first 10 columns
            
            # Check if we have any features
            if train_combined.shape[1] == 0:
                print("ERROR: No features remaining after preprocessing!")
                print("Checking first vintage columns...")
                if train_vintages:
                    first_vintage = list(train_vintages.values())[0]
                    print(f"  First vintage columns: {list(first_vintage.columns)[:20]}")
                    print(f"  Selected features: {SELECTED_FEATURES[:10]}...")
                exit(1)
            
            # Create lookback windows
            X_train, y_train, dates_train = create_lookback_windows(train_combined, 30)
            print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            
            # Process validation data
            print("\nProcessing validation data...")
            val_processed = []
            for vintage_name, df in val_vintages.items():
                try:
                    processed = preprocess_fred_md_data(df, SELECTED_FEATURES, shift_dates=True)
                    val_processed.append(processed)
                except Exception as e:
                    print(f"  Error processing {vintage_name}: {e}")
            
            if val_processed:
                val_combined = pd.concat(val_processed, axis=0)
                val_combined = val_combined.sort_index()
                val_combined = val_combined.drop_duplicates()
                print(f"  Combined validation data shape: {val_combined.shape}")
                
                # Check if we have any features
                if val_combined.shape[1] == 0:
                    print("ERROR: No features remaining after preprocessing!")
                    exit(1)
                
                # Create lookback windows
                X_val, y_val, dates_val = create_lookback_windows(val_combined, 30)
                print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
                
                # Apply StandardScaler
                print("\nApplying StandardScaler...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                data = {
                    'train': {'X': X_train_scaled, 'y': y_train, 'dates': dates_train, 'data': train_combined},
                    'val': {'X': X_val_scaled, 'y': y_val, 'dates': dates_val, 'data': val_combined},
                    'scaler': scaler
                }
                
                print("\nData preparation complete!")
                print(f"  Training samples: {data['train']['X'].shape[0]}")
                print(f"  Validation samples: {data['val']['X'].shape[0]}")
    
    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    print("\nSaving processed data...")
    np.save(output_dir / "X_train.npy", data['train']['X'])
    np.save(output_dir / "y_train.npy", data['train']['y'])
    np.save(output_dir / "dates_train.npy", data['train']['dates']) # Save dates
    
    np.save(output_dir / "X_val.npy", data['val']['X'])
    np.save(output_dir / "y_val.npy", data['val']['y'])
    np.save(output_dir / "dates_val.npy", data['val']['dates']) # Save dates
    
    if 'test' in data:
        np.save(output_dir / "X_test.npy", data['test']['X'])
        np.save(output_dir / "y_test.npy", data['test']['y'])
        np.save(output_dir / "dates_test.npy", data['test']['dates']) # Save dates
    
    import pickle
    with open(output_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(data['scaler'], f)
    
    print(f"Processed data saved to {output_dir}/")
if __name__ == "__main__":
    # Paths - using unzipped folders
    data_dir = Path("data/fred-md")
    train_folder = data_dir / "Historical FRED-MD Vintages Final"
    val_folder = data_dir / "Historical-vintages-of-FRED-MD-2015-01-to-2024-12"
    test_csv = data_dir / "current_fred-md.csv"
    
    # Check if folders exist
    if not train_folder.exists():
        print(f"Error: Training folder not found: {train_folder}")
        print("Please unzip the training data first. See data/README_FRED_MD.md for instructions.")
        exit(1)
    
    if not val_folder.exists():
        print(f"Error: Validation folder not found: {val_folder}")
        print("Please unzip the validation data first. See data/README_FRED_MD.md for instructions.")
        exit(1)
    
    if not test_csv.exists():
        print(f"Warning: Test file not found: {test_csv}")
        print("You can download it from: https://research.stlouisfed.org/econ/mccracken/fred-databases")
        print("Continuing without test data...")
        test_csv = None
    
    # Prepare data
    if test_csv:
        data = prepare_fred_md_for_favar(
            train_folder, 
            val_folder, 
            test_csv,
            selected_features=SELECTED_FEATURES,
            lookback_window=30
        )
    else:
        # Process without test data
        print("Loading training data...")
        train_vintages = load_vintage_data(train_folder)
        print(f"  Loaded {len(train_vintages)} training vintages")
        
        print("Loading validation data...")
        val_vintages = load_vintage_data(val_folder)
        print(f"  Loaded {len(val_vintages)} validation vintages")
        
        # Process training data
        print("\nProcessing training data...")
        train_processed = []
        for i, (vintage_name, df) in enumerate(train_vintages.items()):
            try:
                if i < 3:  # Debug first 3 vintages
                    print(f"  Processing {vintage_name}...")
                processed = preprocess_fred_md_data(df, SELECTED_FEATURES, shift_dates=True)
                if processed.shape[0] > 0 and processed.shape[1] > 0:
                    train_processed.append(processed)
                    if i < 3:
                        print(f"    Result shape: {processed.shape}")
                else:
                    print(f"    Warning: {vintage_name} resulted in empty dataframe")
            except Exception as e:
                print(f"  Error processing {vintage_name}: {e}")
                import traceback
                traceback.print_exc()
        
        if train_processed:
            train_combined = pd.concat(train_processed, axis=0)
            train_combined = train_combined.sort_index()
            train_combined = train_combined.drop_duplicates()
            print(f"  Combined training data shape: {train_combined.shape}")
            print(f"  Available columns: {list(train_combined.columns)[:10]}...")  # Show first 10 columns
            
            # Check if we have any features
            if train_combined.shape[1] == 0:
                print("ERROR: No features remaining after preprocessing!")
                print("Checking first vintage columns...")
                if train_vintages:
                    first_vintage = list(train_vintages.values())[0]
                    print(f"  First vintage columns: {list(first_vintage.columns)[:20]}")
                    print(f"  Selected features: {SELECTED_FEATURES[:10]}...")
                exit(1)
            
            # Create lookback windows
            X_train, y_train, dates_train = create_lookback_windows(train_combined, 30)
            print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            
            # Process validation data
            print("\nProcessing validation data...")
            val_processed = []
            for vintage_name, df in val_vintages.items():
                try:
                    processed = preprocess_fred_md_data(df, SELECTED_FEATURES, shift_dates=True)
                    val_processed.append(processed)
                except Exception as e:
                    print(f"  Error processing {vintage_name}: {e}")
            
            if val_processed:
                val_combined = pd.concat(val_processed, axis=0)
                val_combined = val_combined.sort_index()
                val_combined = val_combined.drop_duplicates()
                print(f"  Combined validation data shape: {val_combined.shape}")
                
                # Check if we have any features
                if val_combined.shape[1] == 0:
                    print("ERROR: No features remaining after preprocessing!")
                    exit(1)
                
                # Create lookback windows
                X_val, y_val, dates_val = create_lookback_windows(val_combined, 30)
                print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
                
                # Apply StandardScaler
                print("\nApplying StandardScaler...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                data = {
                    'train': {'X': X_train_scaled, 'y': y_train, 'dates': dates_train, 'data': train_combined},
                    'val': {'X': X_val_scaled, 'y': y_val, 'dates': dates_val, 'data': val_combined},
                    'scaler': scaler
                }
                
                print("\nData preparation complete!")
                print(f"  Training samples: {data['train']['X'].shape[0]}")
                print(f"  Validation samples: {data['val']['X'].shape[0]}")
    
    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    print("\nSaving processed data...")
    np.save(output_dir / "X_train.npy", data['train']['X'])
    np.save(output_dir / "y_train.npy", data['train']['y'])
    np.save(output_dir / "X_val.npy", data['val']['X'])
    np.save(output_dir / "y_val.npy", data['val']['y'])
    
    if 'test' in data:
        np.save(output_dir / "X_test.npy", data['test']['X'])
        np.save(output_dir / "y_test.npy", data['test']['y'])
    
    import pickle
    with open(output_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(data['scaler'], f)
    
    print(f"Processed data saved to {output_dir}/")
