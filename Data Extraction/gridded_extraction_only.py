import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

LOCATIONS = [
    ("Corpus Christi, TX, USA", 27.8006, -97.3964),
    ("New Delhi, India",        28.6139,  77.2090),
    ("San Francisco, CA, USA",  37.7749, -122.4194),
    ("Chicago, IL, USA",        41.8781,  -87.6298),
]

START = "2000-01-01"
END = "2024-12-31"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

SLP_TPL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface/slp.{year}.nc"
RHUM_TPL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface/rhum.sig995.{year}.nc"

def slug(s):
    return s.replace(",", "").replace(" ", "_").lower()

def nearest_idx(url, lat, lon):
    ds = xr.open_dataset(url)
    lats = ds["lat"].values
    lons = ds["lon"].values
    if lons.max() > 180 and lon < 0:
        lon = (lon + 360) % 360
    if lons.min() < 0 and lon > 180:
        lon = ((lon + 180) % 360) - 180
    ilat = int(np.abs(lats - lat).argmin())
    ilon = int(np.abs(lons - lon).argmin())
    return ilat, ilon

def fetch_point_series_reanalysis(template, var_name, ilat, ilon, start, end, desc=""):
    years = pd.period_range(start=start, end=end, freq="Y").year
    parts = []
    
    for y in tqdm(years, desc=desc, leave=False):
        try:
            url = template.format(year=y)
            ds = xr.open_dataset(url, decode_times=True)
            da = ds[var_name].isel(lat=ilat, lon=ilon).sel(time=slice(start, end))
            parts.append(da)
        except Exception as e:
            print(f"Warning: {y} failed for {desc}: {e}")
            continue
    
    if not parts:
        return None
    return xr.concat(parts, dim="time").sortby("time")

def create_spatial_maps_additional():
    years = [2020, 2021, 2022, 2023]
    
    slp_datasets = []
    for year in tqdm(years, desc="SLP years"):
        try:
            url = SLP_TPL.format(year=year)
            ds = xr.open_dataset(url)
            annual_mean = ds.mean(dim='time')['slp']
            units = (annual_mean.attrs.get("units", "") or "").lower()
            if units in ("pa", "pascals", "pascal"):
                annual_mean = annual_mean / 100.0
            slp_datasets.append(annual_mean)
        except Exception as e:
            print(f"Warning: SLP {year} failed: {e}")
    
    rhum_datasets = []
    for year in tqdm(years, desc="RHUM years"):
        try:
            url = RHUM_TPL.format(year=year)
            ds = xr.open_dataset(url)
            annual_mean = ds.mean(dim='time')['rhum']
            rhum_datasets.append(annual_mean)
        except Exception as e:
            print(f"Warning: RHUM {year} failed: {e}")
    
    if slp_datasets and rhum_datasets:
        slp_mean = xr.concat(slp_datasets, dim='year').mean(dim='year')
        rhum_mean = xr.concat(rhum_datasets, dim='year').mean(dim='year')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        slp_mean.plot(ax=ax1, cmap='RdBu_r', add_colorbar=True,
                     cbar_kwargs={'label': 'Sea Level Pressure (hPa)', 'shrink': 0.8})
        ax1.set_title('Mean Sea Level Pressure (2020-2023)')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        rhum_mean.plot(ax=ax2, cmap='Blues', add_colorbar=True,
                      cbar_kwargs={'label': 'Relative Humidity (%)', 'shrink': 0.8})
        ax2.set_title('Mean Relative Humidity (2020-2023)')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        
        for name, lat, lon in LOCATIONS:
            plot_lon = lon if lon > 0 else lon + 360
            
            for ax in [ax1, ax2]:
                ax.plot(plot_lon, lat, 'ro', markersize=12,
                       markeredgecolor='white', markeredgewidth=3)
                
                city_name = name.split(',')[0]
                ax.annotate(city_name, (plot_lon, lat),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", 
                                   alpha=0.9, edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "additional_variables_spatial.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: additional_variables_spatial.png")

def main():
    print("EXTRACTING ADDITIONAL VARIABLES")
    print(f"Time range: {START} to {END}")
    
    sample_slp_url = SLP_TPL.format(year=2020)
    sample_rhum_url = RHUM_TPL.format(year=2020)
    
    slp_idx = {n: nearest_idx(sample_slp_url, la, lo) for n, la, lo in LOCATIONS}
    rhum_idx = {n: nearest_idx(sample_rhum_url, la, lo) for n, la, lo in LOCATIONS}
    
    for name, lat, lon in LOCATIONS:
        print(f"\nProcessing {name}...")
        s = slug(name)
        
        try:
            slp_ilat, slp_ilon = slp_idx[name]
            slp_data = fetch_point_series_reanalysis(
                SLP_TPL, "slp", slp_ilat, slp_ilon, START, END, f"{name} SLP"
            )
            
            if slp_data is not None:
                units = (slp_data.attrs.get("units", "") or "").lower()
                if units in ("pa", "pascals", "pascal"):
                    slp_data = slp_data / 100.0
                slp_monthly = slp_data.resample(time="MS").mean().to_pandas()
                slp_monthly.to_csv(os.path.join(OUTDIR, f"{s}_slp_monthly.csv"), 
                                  header=["slp_hpa"])
                print(f"  Saved: {s}_slp_monthly.csv")
            
            rhum_ilat, rhum_ilon = rhum_idx[name]
            rhum_data = fetch_point_series_reanalysis(
                RHUM_TPL, "rhum", rhum_ilat, rhum_ilon, START, END, f"{name} RHUM"
            )
            
            if rhum_data is not None:
                rhum_monthly = rhum_data.resample(time="MS").mean().to_pandas()
                rhum_monthly.to_csv(os.path.join(OUTDIR, f"{s}_rhum_monthly.csv"), 
                                   header=["rhum_percent"])
                print(f"  Saved: {s}_rhum_monthly.csv")
            
        except Exception as e:
            print(f"  Failed for {name}: {e}")
    
    create_spatial_maps_additional()
    
    plot_data = {}
    for name, _, _ in LOCATIONS:
        s = slug(name)
        try:
            slp_file = os.path.join(OUTDIR, f"{s}_slp_monthly.csv")
            rhum_file = os.path.join(OUTDIR, f"{s}_rhum_monthly.csv")
            
            if os.path.exists(slp_file) and os.path.exists(rhum_file):
                slp_df = pd.read_csv(slp_file, index_col=0, parse_dates=True)
                rhum_df = pd.read_csv(rhum_file, index_col=0, parse_dates=True)
                plot_data[name] = (slp_df, rhum_df)
        except:
            continue
    
    if len(plot_data) >= 3:
        fig, axes = plt.subplots(3, 2, figsize=(8, 9), dpi=60, constrained_layout=True)
        
        location_names = list(plot_data.keys())[:3]
        for i, location in enumerate(location_names):
            slp_df, rhum_df = plot_data[location]
            
            slp_df.plot(ax=axes[i, 0])
            axes[i, 0].set_title(f"{location.split(',')[0]} Sea Level Pressure", fontsize=8)
            axes[i, 0].set_ylabel("hPa", fontsize=8)
            axes[i, 0].tick_params(labelsize=6)
            axes[i, 0].set_xlabel("")
            
            rhum_df.plot(ax=axes[i, 1])
            axes[i, 1].set_title(f"{location.split(',')[0]} Relative Humidity", fontsize=8)
            axes[i, 1].set_ylabel("%", fontsize=8)
            axes[i, 1].tick_params(labelsize=6)
            axes[i, 1].set_xlabel("")
        
        plt.suptitle("Additional Variables: Sea Level Pressure & Humidity", fontsize=10)
        plt.savefig(os.path.join(OUTDIR, "additional_variables_timeseries.png"), dpi=60)
        plt.close()
        print("Created: additional_variables_timeseries.png")
    
    print(f"\nFiles created:")
    print("- CSV files: *_slp_monthly.csv, *_rhum_monthly.csv")
    print("- Spatial map: additional_variables_spatial.png")  
    print("- Time series: additional_variables_timeseries.png")

if __name__ == "__main__":
    main()