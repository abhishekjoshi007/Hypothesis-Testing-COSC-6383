"""
Script Name: point_grid_consistency_H8.py
Purpose:
    Tests Hypothesis H8 – Temperature point data track gridded cutout averages
    more closely than precipitation.

Inputs:
    • Point-source data (city CSVs)
    • Gridded NOAA dataset (regional cutouts)

Outputs:
    • ../results/H8/point_vs_grid_comparison.png
    • ../results/H8/grid_correlation_summary.csv

Methods:
    • Pearson correlation between point and grid values.
    • RMSE computation per variable (temperature vs precipitation).
    • Visualization of point–grid deviations using scatter and heatmaps.

Expected Result:
    • Temperature correlations > precipitation (higher spatial consistency).
"""

import os, pandas as pd, numpy as np, xarray as xr, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

DATA_DIRS = ["../data extraction/outputs", "../Data Extraction/outputs", "./data", "./"]
NC_DIRS   = [d+"/nc" if not d.endswith("/nc") else d for d in DATA_DIRS]
OUT_DIR   = "../results/H8_all7"; os.makedirs(OUT_DIR, exist_ok=True)

# point CSVs (monthly)
CITIES = {
    "Corpus Christi": ("corpus_christi_tx_usa_all_variables.csv", 27.8006, -97.3964, "grid"),
    "San Francisco":  ("san_francisco_ca_usa_all_variables.csv", 37.7749, -122.4194, "grid"),
    "New Delhi":      ("new_delhi_india_all_variables.csv",       28.6139,  77.2090, "grid"),
    "Chicago":        ("chicago_il_usa_all_variables.csv",        41.8781,  -87.6298, "grid"),
    "New York":       ("new_york_ny_usa__all_variables.csv",      40.7128,  -74.0060, "cutout"),
    "Mumbai":         ("mumbai_mh_india__all_variables.csv",      19.0760,  72.8777, "cutout"),
    "Columbus":       ("columbus_oh_usa__all_variables.csv",      39.9612,  -82.9988, "cutout"),
}

# shared grids for 4 core cities
BIG_TEMP_NC   = "temp_grid_2000_2024.nc"
BIG_PRECIP_NC = "precip_grid_2000_2024.nc"

# city cutouts (temp+precip in one file)
CUTOUT_NC = {
    "New York": "new_york_ny_usa_cutout_monthly_pr_temp.nc",
    "Mumbai":   "mumbai_mh_india_cutout_monthly_pr_temp.nc",
    "Columbus": "columbus_oh_usa_cutout_monthly_pr_temp.nc",
}

def find(path_or_name, roots):
    for r in roots:
        p = os.path.join(r, path_or_name)
        if os.path.exists(p): return p
    return None

def pick_var(ds, candidates):  
    for c in candidates:
        if c in ds.data_vars: return c
    return list(ds.data_vars)[0]

def r2_rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 6: return np.nan, np.nan
    a, b = a[m], b[m]
    r = np.corrcoef(a, b)[0,1]; r2 = 0 if np.isnan(r) else r**2
    rmse = float(np.sqrt(np.mean((a-b)**2)))
    return float(r2), rmse

# open big grids 
T_BIG = find(BIG_TEMP_NC, NC_DIRS)
P_BIG = find(BIG_PRECIP_NC, NC_DIRS)
t_big = xr.open_dataset(T_BIG) if T_BIG else None
p_big = xr.open_dataset(P_BIG) if P_BIG else None
t_big_var = pick_var(t_big, ["tavg","tas","air","temperature"]) if t_big else None
p_big_var = pick_var(p_big, ["pr","precip","precipitation","tp"]) if p_big else None
big_lon0360 = (float(t_big.lon.min()) >= 0) if t_big else False

rows, scatters = [], []

for city, (csv, lat, lon, src) in CITIES.items():
    csv_path = find(csv, DATA_DIRS)
    if not csv_path: continue
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if not {"temperature_c","precipitation_mm"} <= set(df.columns): continue
    sT = df["temperature_c"].asfreq("MS")
    sP = df["precipitation_mm"].asfreq("MS")

    if src == "grid" and t_big is not None and p_big is not None:
        glon = (lon + 360) if (big_lon0360 and lon < 0) else lon
        gT = t_big[t_big_var].sel(lat=lat, lon=glon, method="nearest").to_series().asfreq("MS")
        gP = p_big[p_big_var].sel(lat=lat, lon=glon, method="nearest").to_series().asfreq("MS")
    else:
        cut = find(CUTOUT_NC[city], NC_DIRS)
        ds  = xr.open_dataset(cut)
        tv  = pick_var(ds, ["tavg","tas","air","temperature"])
        pv  = pick_var(ds, ["pr","precip","precipitation","tp"])
        # cutouts already track the city; take spatial mean if small tile
        gT = ds[tv].mean(dim=[d for d in ds[tv].dims if d!="time"]).to_series().asfreq("MS")
        gP = ds[pv].mean(dim=[d for d in ds[pv].dims if d!="time"]).to_series().asfreq("MS")

    idx = sT.index.intersection(gT.index).intersection(sP.index).intersection(gP.index)
    r2_T, rmse_T = r2_rmse(sT.loc[idx], gT.loc[idx])
    r2_P, rmse_P = r2_rmse(sP.loc[idx], gP.loc[idx])
    rows.append({"City":city, "R2_Temp":r2_T, "R2_Precip":r2_P, "RMSE_Temp":rmse_T, "RMSE_Precip":rmse_P})
    scatters.append((city, sT.loc[idx], gT.loc[idx], sP.loc[idx], gP.loc[idx]))

stats_df = pd.DataFrame(rows).dropna()
stats_df.to_csv(os.path.join(OUT_DIR, "H8_point_vs_grid_stats.csv"), index=False)
if stats_df.empty: raise SystemExit("No stats computed.")

def paired(x, y):
    d = (x - y).dropna()
    if len(d) < 3: return ("NA", np.nan, np.nan)
    w = wilcoxon(x, y, alternative="greater")
    t = ttest_rel(x, y, alternative="greater")
    return ("Wilcoxon> & t>", float(w.pvalue), float(t.pvalue))

res_R2 = paired(stats_df["R2_Temp"],  stats_df["R2_Precip"])        
res_RM = paired(-stats_df["RMSE_Temp"], -stats_df["RMSE_Precip"])   

pd.DataFrame([{
    "n_cities": len(stats_df),
    "R2_test": res_R2[0], "R2_p_wilcoxon": res_R2[1], "R2_p_ttest": res_R2[2],
    "RMSE_test": res_RM[0], "RMSE_p_wilcoxon": res_RM[1], "RMSE_p_ttest": res_RM[2],
    "R2_Temp_mean": stats_df["R2_Temp"].mean(), "R2_Precip_mean": stats_df["R2_Precip"].mean(),
    "RMSE_Temp_mean": stats_df["RMSE_Temp"].mean(), "RMSE_Precip_mean": stats_df["RMSE_Precip"].mean(),
}]).to_csv(os.path.join(OUT_DIR, "H8_test_summary.csv"), index=False)

# dumbbell R² (all 7)
dfm = stats_df.melt(id_vars="City", value_vars=["R2_Precip","R2_Temp"], var_name="Metric", value_name="R2")
order = stats_df.sort_values("R2_Temp")["City"]
plt.figure(figsize=(9,5), dpi=200)
for _, r in stats_df.iterrows():
    plt.plot([r["R2_Precip"], r["R2_Temp"]], [r["City"], r["City"]], "-", color="0.8", zorder=1)
sns.scatterplot(data=dfm, x="R2", y="City", hue="Metric", s=80,
                palette=["tab:blue","tab:orange"], hue_order=["R2_Precip","R2_Temp"])
plt.yticks(order); plt.xlim(0,1); plt.xlabel("R² (Point vs Grid)"); plt.ylabel("")
plt.title("H8: Point–Grid Consistency — Temp vs Precip (All 7 Cities)")
plt.legend(title="", loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "H8_dumbbell_R2_all7.png"), dpi=300)

# scatter grid (all cities)
n = len(scatters)
fig, axes = plt.subplots(n, 2, figsize=(8, 3*n), dpi=180, constrained_layout=True)
for i, (city, sT, gT, sP, gP) in enumerate(scatters):
    ax = axes[i,0]; ax.scatter(gT, sT, s=8, alpha=.6)
    lo, hi = np.nanmin([gT.min(),sT.min()]), np.nanmax([gT.max(),sT.max()])
    ax.plot([lo,hi],[lo,hi],'r--',lw=1); ax.set_title(f"{city} — Temp"); ax.set_xlabel("Grid"); ax.set_ylabel("Point")
    ax = axes[i,1]; ax.scatter(gP, sP, s=8, alpha=.6)
    lo, hi = np.nanmin([gP.min(),sP.min()]), np.nanmax([gP.max(),sP.max()])
    ax.plot([lo,hi],[lo,hi],'r--',lw=1); ax.set_title(f"{city} — Precip"); ax.set_xlabel("Grid"); ax.set_ylabel("Point")
plt.suptitle("H8: Point vs Grid Scatter (Monthly Alignments)", y=1.01, fontsize=12)
plt.savefig(os.path.join(OUT_DIR, "H8_scatter_all7.png"), dpi=300, bbox_inches="tight")

print("Done:",
      f"\n  cities: {list(stats_df.City)}",
      f"\n  mean R² temp={stats_df.R2_Temp.mean():.3f}, precip={stats_df.R2_Precip.mean():.3f}",
      f"\n  Wilcoxon p (R² temp>precip)={res_R2[1]:.4g}",
      f"\n  Wilcoxon p (RMSE temp<precip)={res_RM[1]:.4g}",
      f"\n  outputs→ {OUT_DIR}")
