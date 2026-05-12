# FRED-MD Data Download Instructions

## Required Datasets

You need to download the following FRED-MD datasets:

### 1. Training Data (Historical Vintages 1999-08 to 2014-12)
- **File**: Historical Vintages of FRED-MD 1999-08 to 2014-12 (.zip 45 MB)
- **URL**: https://research.stlouisfed.org/econ/mccracken/fred-databases
- **Direct Link**: https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical_fred-md.zip
- **Save as**: `data/fred-md/historical_fred-md_1999-2014.zip`

### 2. Validation Data (Historical Vintages 2015-01 to 2024-12)
- **File**: Historical Vintages of FRED-MD 2015-01 to 2024-12 (.zip 32 MB)
- **URL**: https://research.stlouisfed.org/econ/mccracken/fred-databases
- **Direct Link**: https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2024-12.zip
- **Save as**: `data/fred-md/historical_fred-md_2015-2024.zip`

### 3. Test Data (Current FRED-MD Monthly Data)
- **File**: FRED-MD: Monthly Data (current.csv)
- **URL**: https://research.stlouisfed.org/econ/mccracken/fred-databases
- **Direct Link**: https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2025-11-md.csv
- **Save as**: `data/fred-md/current_fred-md.csv`

## Download Methods

### Method 1: Manual Download (Recommended)
1. Visit: https://research.stlouisfed.org/econ/mccracken/fred-databases
2. Scroll to the "FRED-MD" section
3. Click on each download link:
   - "Historical Vintages of FRED-MD 1999-08 to 2014-12 (.zip 45 MB)"
   - "Historical Vintages of FRED-MD 2015-01 to 2024-12 (.zip 32 MB)"
   - "current.csv" (under FRED-MD: Monthly Data)
4. Save files to `data/fred-md/` directory

### Method 2: Using wget/curl (if direct links work)
```bash
cd data/fred-md

# Download training data
wget -O historical_fred-md_1999-2014.zip \
  "https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical_fred-md.zip"

# Download validation data
wget -O historical_fred-md_2015-2024.zip \
  "https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2024-12.zip"

# Download current data
wget -O current_fred-md.csv \
  "https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2025-11-md.csv"
```

### Method 3: Using Python requests
From the **repository root**:
```bash
python3 data/download_fred_md.py
```

## Expected File Structure

After downloading, your `data/fred-md/` directory should contain:
```
data/fred-md/
├── historical_fred-md_1999-2014.zip  (45 MB)
├── historical_fred-md_2015-2024.zip  (32 MB)
└── current_fred-md.csv               (varies)
```

## Next Steps

After downloading the files, you can:
1. Extract the ZIP files to see the historical vintages structure
2. Use the provided data processing scripts to prepare the data for FAVAR modeling
3. Follow the preprocessing steps outlined in your VIP requirements

## Notes

- Historical vintages are important for avoiding look-ahead bias in time series forecasting
- The ZIP files contain multiple CSV files, one for each vintage date
- Make sure to handle the date shifting (by one month) as mentioned in your preprocessing requirements
