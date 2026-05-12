#!/usr/bin/env python3
"""
Script to download FRED-MD datasets
"""

import requests
import os
from pathlib import Path

# Create data directory
data_dir = Path("data/fred-md")
data_dir.mkdir(parents=True, exist_ok=True)

# URLs for FRED-MD datasets
urls = {
    "historical_fred-md_1999-2014.zip": "https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical_fred-md.zip",
    "historical_fred-md_2015-2024.zip": "https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2024-12.zip",
    "current_fred-md.csv": "https://research.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2025-11-md.csv"
}

# Headers to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://research.stlouisfed.org/econ/mccracken/fred-databases'
}

def download_file(url, filename):
    """Download a file with proper headers"""
    filepath = data_dir / filename
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        # Check if we got HTML instead of the actual file
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"Warning: Received HTML instead of file. Response preview:")
            print(response.text[:500])
            return False
        
        # Download the file
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n  Saved to: {filepath}")
        print(f"  File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return False

if __name__ == "__main__":
    print("Downloading FRED-MD datasets...")
    print("=" * 60)
    
    for filename, url in urls.items():
        success = download_file(url, filename)
        if success:
            print(f"✓ Successfully downloaded {filename}\n")
        else:
            print(f"✗ Failed to download {filename}\n")
    
    print("=" * 60)
    print("Download complete!")
