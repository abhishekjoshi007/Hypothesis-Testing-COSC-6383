#Extracted Sea Level Pressure and Relative Humidity
import os, numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
from tqdm import tqdm

LOCATIONS = [
    ("Corpus Christi, TX, USA", 27.8006, -97.3964),
    ("New Delhi, India",        28.6139, 77.2090),
    ("San Francisco, CA, USA",  37.7749,-122.4194),
    ("Chicago, IL, USA",        41.8781, -87.6298),
]
START, END = "2000-01-01", "2024-12-31"
OUTDIR = os.path.abspath("outputs"); os.makedirs(OUTDIR, exist_ok=True)
SLP_TPL  = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface/slp.{year}.nc"
RHUM_TPL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface/rhum.sig995.{year}.nc"

def slug(s): return s.replace(",", "").replace(" ", "_").lower()

def check_existing_data():
    existing = {}
    for name, _, _ in LOCATIONS:
        s = slug(name)
        slp_file = os.path.join(OUTDIR, f"{s}_slp_monthly.csv")
        rhum_file = os.path.join(OUTDIR, f"{s}_rhum_monthly.csv")
        existing[name] = {
            'slp': os.path.exists(slp_file),
            'rhum': os.path.exists(rhum_file)
        }
    return existing

def nearest_idx(url, lat, lon):
    with xr.open_dataset(url) as ds:
        lats, lons = ds["lat"].values, ds["lon"].values
    if lons.max() > 180 and lon < 0: lon = (lon + 360) % 360
    if lons.min() < 0 and lon > 180: lon = ((lon + 180) % 360) - 180
    ilat = int(np.abs(lats - lat).argmin()); ilon = int(np.abs(lons - lon).argmin())
    return ilat, ilon

def fetch_point_series(tpl, var, ilat, ilon, start, end, desc=""):
    if ilat is None or ilon is None: return None
    years = pd.period_range(start=start, end=end, freq="Y").year
    parts = []
    for y in tqdm(years, desc=desc, leave=False):
        try:
            with xr.open_dataset(tpl.format(year=y), decode_times=True) as ds:
                da = ds[var].isel(lat=ilat, lon=ilon).sel(time=slice(start, end))
                parts.append(da)
        except Exception as e:
            print(f"  skip {y}: {e}")
    if not parts: return None
    return xr.concat(parts, dim="time").sortby("time").to_pandas()

def create_spatial_maps(years=(2022, 2023)):
    spatial_png = os.path.join(OUTDIR, f"additional_variables_spatial_{years[0]}_{years[-1]}.png")
    if os.path.exists(spatial_png):
        print(f"  spatial map exists: {spatial_png}")
        return
    
    slp_list, rh_list = [], []
    for y in years:
        try:
            with xr.open_dataset(SLP_TPL.format(year=y), decode_times=True) as ds:
                slp_list.append(ds["slp"].mean("time")/100.0)
        except Exception as e:
            print("  SLP map skip", y, e)
        try:
            with xr.open_dataset(RHUM_TPL.format(year=y), decode_times=True) as ds:
                rh_list.append(ds["rhum"].mean("time"))
        except Exception as e:
            print("  RHUM map skip", y, e)
    if not slp_list or not rh_list:
        print("  no spatial map saved (no data)")
        return
    slp = xr.concat(slp_list, dim="year").mean("year"); rh = xr.concat(rh_list, dim="year").mean("year")
    lats, lons = slp.lat.values, slp.lon.values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    im1 = ax1.contourf(lons, lats, slp.values, levels=20, cmap="RdBu_r"); plt.colorbar(im1, ax=ax1, label="SLP (hPa)")
    im2 = ax2.contourf(lons, lats, rh.values,  levels=20, cmap="Blues");  plt.colorbar(im2, ax=ax2, label="RH (%)")
    for name, lat, lon in LOCATIONS:
        plon = lon if lon >= 0 else lon + 360
        for ax in (ax1, ax2):
            ax.plot(plon, lat, "ro", ms=7, mec="white", mew=1.6)
            ax.text(plon+5, lat+2, name.split(",")[0], fontsize=8, bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
    plt.tight_layout()
    plt.savefig(spatial_png, dpi=300, bbox_inches="tight"); plt.close()
    print("  saved:", spatial_png)

def main():
    print("OUTDIR:", OUTDIR)
    
    existing = check_existing_data()
    sample_slp, sample_rh = SLP_TPL.format(year=2020), RHUM_TPL.format(year=2020)
    slp_idx = {n: nearest_idx(sample_slp, la, lo) for n, la, lo in LOCATIONS}
    rh_idx  = {n: nearest_idx(sample_rh,  la, lo) for n, la, lo in LOCATIONS}
    
    for name, lat, lon in LOCATIONS:
        print(name)
        s = slug(name)
        
        slp_path = os.path.join(OUTDIR, f"{s}_slp_monthly.csv")
        if existing[name]['slp']:
            print(f"  SLP data exists: {slp_path}")
        else:
            ilat, ilon = slp_idx[name]
            slp_series = fetch_point_series(SLP_TPL, "slp", ilat, ilon, START, END, f"{name} SLP")
            if slp_series is not None and len(slp_series):
                slp_monthly = (slp_series/100.0).resample("MS").mean()
                slp_monthly.to_csv(slp_path, header=["slp_hpa"])
                print("  saved:", slp_path, "| rows:", len(slp_monthly))
            else:
                print("  no SLP series")
        
        rhum_path = os.path.join(OUTDIR, f"{s}_rhum_monthly.csv")
        if existing[name]['rhum']:
            print(f"  RHUM data exists: {rhum_path}")
        else:
            ilat, ilon = rh_idx[name]
            rh_series = fetch_point_series(RHUM_TPL, "rhum", ilat, ilon, START, END, f"{name} RHUM")
            if rh_series is not None and len(rh_series):
                rh_monthly = rh_series.resample("MS").mean()
                rh_monthly.to_csv(rhum_path, header=["rhum_percent"])
                print("  saved:", rhum_path, "| rows:", len(rh_monthly))
            else:
                print("  no RH series")
    
    print("spatial maps…")
    create_spatial_maps()
    
    print("timeseries figure…")
    ts_png = os.path.join(OUTDIR, "additional_variables_timeseries.png")
    if os.path.exists(ts_png):
        print(f"  timeseries plot exists: {ts_png}")
    else:
        plot_data = {}
        for name, _, _ in LOCATIONS:
            s = slug(name)
            p1, p2 = os.path.join(OUTDIR, f"{s}_slp_monthly.csv"), os.path.join(OUTDIR, f"{s}_rhum_monthly.csv")
            if os.path.exists(p1) and os.path.exists(p2):
                slp_df  = pd.read_csv(p1, index_col=0, parse_dates=True).dropna()
                rh_df   = pd.read_csv(p2, index_col=0, parse_dates=True).dropna()
                if not slp_df.empty and not rh_df.empty: plot_data[name] = (slp_df, rh_df)
        if len(plot_data) >= 3:
            fig, axes = plt.subplots(3, 2, figsize=(8, 9), dpi=300, constrained_layout=True)
            for i, name in enumerate(list(plot_data.keys())[:3]):
                slp_df, rh_df = plot_data[name]
                slp_df.plot(ax=axes[i, 0], legend=False); axes[i, 0].set_title(f"{name.split(',')[0]} Sea Level Pressure", fontsize=8)
                axes[i, 0].set_ylabel("hPa", fontsize=8); axes[i, 0].set_xlabel(""); axes[i, 0].tick_params(labelsize=6)
                rh_df.plot(ax=axes[i, 1], legend=False);  axes[i, 1].set_title(f"{name.split(',')[0]} Relative Humidity", fontsize=8)
                axes[i, 1].set_ylabel("%",   fontsize=8); axes[i, 1].set_xlabel(""); axes[i, 1].tick_params(labelsize=6)
            plt.suptitle("Additional Variables: Sea Level Pressure & Relative Humidity", fontsize=10)
            plt.savefig(ts_png, dpi=300); plt.close()
            print("  saved:", ts_png)
        else:
            print("  no figure (insufficient CSVs)")
    
    print("done. ls:", OUTDIR); print("\n".join(sorted(os.listdir(OUTDIR))))

if __name__ == "__main__":
    main()