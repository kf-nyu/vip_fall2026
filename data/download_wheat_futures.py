"""
Download US Wheat Futures daily price data
"""

import pandas as pd
from pathlib import Path
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Please install it with:")
    print("  pip install yfinance")
    print("Or download wheat futures data manually from Yahoo Finance")

def download_wheat_futures(symbol="ZW=F", start_date="1999-01-01", end_date=None):
    """
    Download wheat futures data from Yahoo Finance
    
    Args:
        symbol: Futures symbol (ZW=F for Chicago SRW Wheat, KE=F for KC HRW Wheat)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), None for today
    
    Returns:
        DataFrame with OHLCV data
    """
    if not YFINANCE_AVAILABLE:
        print("ERROR: yfinance is not installed.")
        print("\nTo install yfinance, run:")
        print("  pip install yfinance")
        print("\nOr download wheat futures data manually:")
        print("  1. Visit: https://finance.yahoo.com/quote/ZW=F/history")
        print("  2. Download historical data as CSV")
        print("  3. Save to: data/wheat-futures/wheat_futures_daily.csv")
        return None
    
    print(f"Downloading {symbol} from {start_date} to {end_date or 'today'}...")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        print(f"Warning: No data retrieved for {symbol}")
        return None
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Rename Date column if it exists
    if 'Date' in data.columns:
        data = data.rename(columns={'Date': 'date'})
    
    print(f"Downloaded {len(data)} days of data")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    return data

def aggregate_to_monthly(daily_data):
    """
    Aggregate daily futures data to monthly
    
    Args:
        daily_data: DataFrame with daily OHLCV data
    
    Returns:
        DataFrame with monthly aggregated data
    """
    # Set date as index
    data = daily_data.set_index('date')
    
    # Resample to monthly, taking last value of month (end-of-month)
    # Use 'ME' for newer pandas versions, fallback to 'M' for older versions
    try:
        monthly = data.resample('ME').last()
        monthly_avg = data['Close'].resample('ME').mean()
    except ValueError:
        monthly = data.resample('M').last()
        monthly_avg = data['Close'].resample('M').mean()
    monthly['Close_Avg'] = monthly_avg
    
    # Reset index
    monthly = monthly.reset_index()
    
    return monthly

if __name__ == "__main__":
    # Create data directory
    data_dir = Path("data/wheat-futures")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Chicago SRW Wheat Futures (most liquid)
    wheat_data = download_wheat_futures("ZW=F", start_date="1999-01-01")
    
    if wheat_data is not None:
        # Save daily data
        daily_path = data_dir / "wheat_futures_daily.csv"
        wheat_data.to_csv(daily_path, index=False)
        print(f"\nDaily data saved to {daily_path}")
        
        # Aggregate to monthly
        monthly_data = aggregate_to_monthly(wheat_data)
        monthly_path = data_dir / "wheat_futures_monthly.csv"
        monthly_data.to_csv(monthly_path, index=False)
        print(f"Monthly data saved to {monthly_path}")
        
        print(f"\nData summary:")
        print(f"  Daily records: {len(wheat_data)}")
        print(f"  Monthly records: {len(monthly_data)}")
        print(f"  Price range: ${wheat_data['Close'].min():.2f} - ${wheat_data['Close'].max():.2f}")
