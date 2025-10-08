import os
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from tqdm import tqdm

LOCATIONS = [
    ("Corpus Christi, TX, USA", 27.8006, -97.3964),
    ("New Delhi, India", 28.6139, 77.2090),
    ("San Francisco, CA, USA", 37.7749, -122.4194),
    ("Chicago, IL, USA", 41.8781, -87.6298),
]

START = "2000-01-01"
END = "2024-12-31"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

#Data URLs
PREC_TPL = "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_precip/precip.{year}.nc"
TMIN_TPL = "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_temp/tmin.{year}.nc"
TMAX_TPL = "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_temp/tmax.{year}.nc"

def slug(name):
    return name.replace(",", "").replace(" ", "_").lower()

def extract_point_data(url_template, var_name, lat, lon, start_year, end_year):
    data = []
    dates = []
    
    for year in tqdm(range(start_year, end_year + 1), desc=f"Extracting {var_name}"):
        try:
            # Open dataset for the year
            url = url_template.format(year=year)
            ds = xr.open_dataset(url, decode_times=True)
            
            # Handle longitude conversion if needed 
            adj_lon = lon if lon >= 0 else lon + 360
            
            # Select nearest point
            point_data = ds[var_name].sel(
                lat=lat, 
                lon=adj_lon, 
                method='nearest',
                time=slice(f"{year}-01-01", f"{year}-12-31")
            )
            
            # Convert to pandas 
            series = point_data.to_pandas()
            data.extend(series.values)
            dates.extend(series.index)
            
            ds.close()
            
        except Exception as e:
            print(f"  Warning {year}: {e}")
    
    if not data:
        return None
    
    # Create DF
    df = pd.DataFrame({var_name: data}, index=pd.DatetimeIndex(dates))
    return df

def process_temperature_point(lat, lon, start_year, end_year):
    """Extract and calculate average temperature for a point"""
    print("  Extracting tmin...")
    tmin = extract_point_data(TMIN_TPL, "tmin", lat, lon, start_year, end_year)
    
    print("  Extracting tmax...")
    tmax = extract_point_data(TMAX_TPL, "tmax", lat, lon, start_year, end_year)
    
    if tmin is None or tmax is None:
        print(" Could not extract temperature data")
        return None
    
    # Calculate average temperature
    tavg = pd.DataFrame(index=tmin.index)
    tavg['temperature_c'] = (tmin['tmin'] + tmax['tmax']) / 2.0
    
    # Resample to monthly
    tavg_monthly = tavg.resample('MS').mean()
    
    return tavg_monthly

def process_precipitation_point(lat, lon, start_year, end_year):
    """Extract precipitation for a point"""
    print("  Extracting precipitation...")
    precip = extract_point_data(PREC_TPL, "precip", lat, lon, start_year, end_year)
    
    if precip is None:
        print("Could not extract precipitation data")
        return None
    
    # Rename column
    precip.columns = ['precipitation_mm']
    
    # Resample to monthly (sum for precipitation)
    precip_monthly = precip.resample('MS').sum()
    
    return precip_monthly

def main():
    """Main processing function"""
    print(f"CPC Point Data Extraction")
    print(f"Period: {START} to {END}")
    print(f"Output directory: {OUTDIR}\n")
    
    start_year = int(START[:4])
    end_year = int(END[:4])
    
    for name, lat, lon in LOCATIONS:
        print(f"\nProcessing: {name}")
        print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")
        
        city_slug = slug(name)
        
        # Temperature
        temp_file = os.path.join(OUTDIR, f"{city_slug}_temp_monthly.csv")
        if not os.path.exists(temp_file):
            temp_data = process_temperature_point(lat, lon, start_year, end_year)
            if temp_data is not None:
                temp_data.to_csv(temp_file)
                print(f"  Saved: {temp_file}")
        else:
            print(f"  Temperature file exists: {temp_file}")
        
        # Precipitation
        precip_file = os.path.join(OUTDIR, f"{city_slug}_precip_monthly.csv")
        if not os.path.exists(precip_file):
            precip_data = process_precipitation_point(lat, lon, start_year, end_year)
            if precip_data is not None:
                precip_data.to_csv(precip_file)
                print(f"  Saved: {precip_file}")
        else:
            print(f"  Precipitation file exists: {precip_file}")
    
    print("\n" + "="*50)
    print("CPC POINT DATA EXTRACTION COMPLETE")
    print(f"Files saved to: {OUTDIR}")
    print("\nNote: This extracts point data from CPC gridded datasets.")
    print("For true station observations, use NOAA Climate Data Online or GHCN-Daily.")

if __name__ == "__main__":
    main()