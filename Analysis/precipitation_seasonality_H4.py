import os, pandas as pd, numpy as np, xarray as xr, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

DATA_DIRS = ["../data extraction/outputs", "../Data Extraction/outputs", "./data"]
OUT_DIR = "../results/H8"; os.makedirs(OUT_DIR, exist_ok=True)
TEMP_NC, PRECIP_NC = "temp_grid_2000_2024.nc", "precip_grid_2000_2024.nc"

# city -> (file, lat, lon)
CITIES = {
    "Corpus Christi": ("corpus_christi_tx_usa_all_variables.csv", 27.8006, -97.3964),
    "New Delhi":      ("new_delhi_india_all_variables.csv",       28.6139,  77.2090),
    "San Francisco":  ("san_francisco_ca_usa_all_variables.csv",  37.7749, -122.4194),
    "Chicago":        ("chicago_il_usa_all_variables.csv",        41.8781,  -87.6298),

    # "New York":     ("new_york_ny_usa__all_variables.csv",      40.7128,  -74.0060),
    # "Columbus":     ("columbus_oh_usa__all_variables.csv",      39.9612,  -82.9988),
    # "Mumbai":       ("mumbai_mh_india__all_variables.csv",      19.0760,   72.8777),
}

def find_path(fname):
    for d in DATA_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p): return p
    return None

def r2_rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 6: return np.nan, np.nan
    a, b = a[mask], b[mask]
    r = np.corrcoef(a, b)[0,1]
    r2 = 0 if np.isnan(r) else r**2
    rmse = float(np.sqrt(np.mean((a-b)**2)))
    return float(r2), rmse

# load grids 
tpath, ppath = find_path(TEMP_NC), find_path(PRECIP_NC)
if not (tpath and ppath): raise SystemExit("NetCDF grids not found. Check TEMP_NC/PRECIP_NC paths.")
t_ds, p_ds = xr.open_dataset(tpath), xr.open_dataset(ppath)
t_var = list(t_ds.data_vars)[0]   
p_var = list(p_ds.data_vars)[0]   
grid_lon_0360 = float(t_ds.lon.min()) >= 0

# compute per-city metrics 
rows, scatters = [], []  
for city, (csvfile, lat, lon) in CITIES.items():
    cpath = find_path(csvfile)
    if not cpath: continue
    df = pd.read_csv(cpath, index_col=0, parse_dates=True)
    if not {"temperature_c","precipitation_mm"} <= set(df.columns): continue
    sT = df["temperature_c"].asfreq("MS")        # monthly start
    sP = df["precipitation_mm"].asfreq("MS")

    glon = (lon + 360) if (grid_lon_0360 and lon < 0) else lon
    gT = t_ds[t_var].sel(lat=lat, lon=glon, method="nearest").to_series().asfreq("MS")  
    gP = p_ds[p_var].sel(lat=lat, lon=glon, method="nearest").to_series().asfreq("MS")  

    idx = sT.index.intersection(gT.index).intersection(sP.index).intersection(gP.index)
    r2_T, rmse_T = r2_rmse(sT.loc[idx], gT.loc[idx])
    r2_P, rmse_P = r2_rmse(sP.loc[idx], gP.loc[idx])

    rows.append({"City":city, "R2_Temp":r2_T, "R2_Precip":r2_P, "RMSE_Temp":rmse_T, "RMSE_Precip":rmse_P})

    scatters.append((city, sT.loc[idx], gT.loc[idx], sP.loc[idx], gP.loc[idx]))

stats_df = pd.DataFrame(rows).dropna()
stats_df.to_csv(os.path.join(OUT_DIR, "point_vs_grid_stats.csv"), index=False)
if stats_df.empty: raise SystemExit("No stats computed. Check inputs/columns.")

# paired tests (Temp vs Precip across cities) 
def paired_test(x, y):
    d = (x - y).dropna()
    if len(d) < 3: return ("NA", np.nan, np.nan)
    # normal check is noisy with n~4, prefer Wilcoxon; also report t-test
    try:
        w = wilcoxon(x, y, alternative="greater")   # R2_T > R2_P
        t = ttest_rel(x, y, alternative="greater")
        return ("Wilcoxon> & t>", float(w.pvalue), float(t.pvalue))
    except Exception:
        return ("Wilcoxon_failed", np.nan, np.nan)

res_R2 = paired_test(stats_df["R2_Temp"], stats_df["R2_Precip"])
res_RM = paired_test(-stats_df["RMSE_Temp"], -stats_df["RMSE_Precip"])  

pd.DataFrame([{
    "n_cities": len(stats_df),
    "R2_test": res_R2[0], "R2_p_wilcoxon": res_R2[1], "R2_p_ttest": res_R2[2],
    "RMSE_test": res_RM[0], "RMSE_p_wilcoxon": res_RM[1], "RMSE_p_ttest": res_RM[2],
    "R2_Temp_mean": stats_df["R2_Temp"].mean(), "R2_Precip_mean": stats_df["R2_Precip"].mean(),
    "RMSE_Temp_mean": stats_df["RMSE_Temp"].mean(), "RMSE_Precip_mean": stats_df["RMSE_Precip"].mean(),
}]).to_csv(os.path.join(OUT_DIR, "H8_test_summary.csv"), index=False)

# dumbbell plot (R²) 
df_m = stats_df.melt(id_vars="City", value_vars=["R2_Precip","R2_Temp"],
                     var_name="Metric", value_name="R2")
order = stats_df.sort_values("R2_Temp")["City"]
plt.figure(figsize=(8,5), dpi=180)
for _, r in stats_df.iterrows():
    plt.plot([r["R2_Precip"], r["R2_Temp"]], [r["City"], r["City"]], "-",
             color="0.75", zorder=1)
sns.scatterplot(data=df_m, x="R2", y="City", hue="Metric", s=80, palette=["tab:blue","tab:red"], hue_order=["R2_Precip","R2_Temp"])
plt.yticks(order)
plt.xlim(0,1); plt.xlabel("R² (Point vs Grid)"); plt.ylabel("")
plt.title("H8: Point–Grid Consistency (R²) — Temp vs Precip")
plt.legend(title="", loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "H8_dumbbell_R2.png"), dpi=300)

# scatter (first 4 cities) 
n = min(4, len(scatters))
fig, axes = plt.subplots(n, 2, figsize=(8, 3*n), dpi=180, constrained_layout=True)
for i in range(n):
    city, sT, gT, sP, gP = scatters[i]
    ax = axes[i,0]; ax.scatter(gT, sT, s=8, alpha=.6); lo, hi = np.nanmin([gT.min(),sT.min()]), np.nanmax([gT.max(),sT.max()])
    ax.plot([lo,hi],[lo,hi],'r--',lw=1); ax.set_title(f"{city} — Temp"); ax.set_xlabel("Grid"); ax.set_ylabel("Point")
    ax = axes[i,1]; ax.scatter(gP, sP, s=8, alpha=.6); lo, hi = np.nanmin([gP.min(),sP.min()]), np.nanmax([gP.max(),sP.max()])
    ax.plot([lo,hi],[lo,hi],'r--',lw=1); ax.set_title(f"{city} — Precip"); ax.set_xlabel("Grid"); ax.set_ylabel("Point")
plt.suptitle("H8: Point vs Grid Scatter (Monthly Alignments)", y=1.02, fontsize=12)
plt.savefig(os.path.join(OUT_DIR, "H8_scatter_examples.png"), dpi=300, bbox_inches="tight")

print("Done:",
      f"\n  cities: {list(stats_df.City)}",
      f"\n  mean R² temp={stats_df.R2_Temp.mean():.3f}, precip={stats_df.R2_Precip.mean():.3f}",
      f"\n  Wilcoxon p (R² temp>precip)={res_R2[1]:.4g}",
      f"\n  Wilcoxon p (RMSE temp<precip)={res_RM[1]:.4g}",
      f"\n  outputs→ {OUT_DIR}")
