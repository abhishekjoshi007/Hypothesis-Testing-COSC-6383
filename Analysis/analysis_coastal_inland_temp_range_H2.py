"""
Script Name: analysis_coastal_inland_temp_range_H2.py
Purpose:
    Tests Hypothesis H2 – Coastal cities show smaller annual temperature
    ranges than inland cities.

Inputs:
    • City-level monthly temperature data for San Francisco, Corpus Christi,
      Mumbai (Coastal) and Chicago, New Delhi, Columbus (Inland).

Outputs:
    • ../results/H2/annual_temperature_ranges.csv
    • ../results/H2/temp_range_comparison.png
    • ../results/H2/temp_range_summary.csv

Methods:
    • Shapiro–Wilk test for normality.
    • Levene’s test for homogeneity of variances.
    • Mann–Whitney U or t-test for mean comparison.
    • Cliff’s delta (effect size) and bootstrap CI.

Expected Result:
    • Coastal cities exhibit significantly smaller annual ranges (p < 0.001).
"""


import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene

DATA_DIRS = ["../data extraction/outputs", "../Data Extraction/outputs", "./data"]
OUT_DIR = "../results/H2"; os.makedirs(OUT_DIR, exist_ok=True)

CITIES = {
    "san_francisco_ca_usa_all_variables.csv": ("San Francisco","Coastal"),
    "corpus_christi_tx_usa_all_variables.csv": ("Corpus Christi","Coastal"),
    "mumbai_mh_india__all_variables.csv": ("Mumbai","Coastal"),
    "chicago_il_usa_all_variables.csv": ("Chicago","Inland"),
    "columbus_oh_usa__all_variables.csv": ("Columbus","Inland"),
    "new_delhi_india_all_variables.csv": ("New Delhi","Inland"),
}
NY_FILE = "new_york_ny_usa__all_variables.csv"   # reference-only
USE_COMMON_YEARS = True

def resolve(fname):
    for d in DATA_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p): return p
    return None

def pick_temp_column(df):
    for c in ["temperature_c","temp_c","temp","tavg","temperature"]:
        if c in df.columns: return c
    return None

def annual_ranges(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    tcol = pick_temp_column(df); 
    if not tcol: return pd.Series(dtype=float)
    s = df[tcol].dropna()
    if s.empty: return pd.Series(dtype=float)
    return s.groupby(s.index.year).apply(lambda x: x.max()-x.min())

#  load tested cities 
records = []
for fname, (label, ctype) in CITIES.items():
    p = resolve(fname)
    if not p: continue
    ar = annual_ranges(p)
    for y, v in ar.items():
        records.append({"City": label, "Type": ctype, "Year": int(y), "Annual_Temp_Range": float(v)})
df_all = pd.DataFrame(records)
if df_all.empty:
    raise SystemExit("No data loaded for hypothesis.")

if USE_COMMON_YEARS:
    years_sets = df_all.groupby("City")["Year"].apply(set).tolist()
    common_years = set.intersection(*years_sets) if years_sets else set()
    df_all = df_all[df_all["Year"].isin(common_years)]

df_all.to_csv(os.path.join(OUT_DIR, "annual_temperature_ranges.csv"), index=False)

coastal = df_all.loc[df_all["Type"]=="Coastal","Annual_Temp_Range"].values
inland  = df_all.loc[df_all["Type"]=="Inland","Annual_Temp_Range"].values

sc = shapiro(coastal) if len(coastal)>=3 else None
si = shapiro(inland)  if len(inland)>=3 else None
lv = levene(coastal, inland) if len(coastal)>=2 and len(inland)>=2 else None
use_t = (sc and si and lv) and (sc.pvalue>0.05 and si.pvalue>0.05 and lv.pvalue>0.05)

if use_t:
    r = ttest_ind(coastal, inland, equal_var=True)
    test, stat, p = "t-test", float(r.statistic), float(r.pvalue)
else:
    r = mannwhitneyu(coastal, inland, alternative="two-sided")
    test, stat, p = "Mann–Whitney U", float(r.statistic), float(r.pvalue)

def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a)*len(b))
delta = cliffs_delta(coastal, inland)

rng = np.random.default_rng(42)
def boot_mean_ci(x, B=5000, alpha=0.05):
    x = np.asarray(x); n=len(x)
    boots = rng.choice(x, size=(B, n), replace=True).mean(axis=1)
    return np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
c_ci = boot_mean_ci(coastal) if len(coastal)>1 else (np.nan,np.nan)
i_ci = boot_mean_ci(inland)  if len(inland)>1 else (np.nan,np.nan)

ny_series = pd.Series(dtype=float)
ny_path = resolve(NY_FILE)
if ny_path:
    ny_series = annual_ranges(ny_path)
    if USE_COMMON_YEARS and len(ny_series)>0 and 'common_years' in locals():
        ny_series = ny_series[ny_series.index.isin(common_years)]
    pd.DataFrame({"Year": ny_series.index, "Annual_Temp_Range": ny_series.values}) \
        .to_csv(os.path.join(OUT_DIR, "new_york_annual_temperature_ranges.csv"), index=False)
 
sns.set_context("talk")
fig, axes = plt.subplots(1, 2, figsize=(13,6), dpi=200, gridspec_kw={"width_ratios":[3,2]})

ax = axes[0]
sns.boxplot(x="Type", y="Annual_Temp_Range", data=df_all, hue="Type",
            palette="Set2", width=0.5, dodge=False, legend=False, ax=ax)
sns.swarmplot(x="Type", y="Annual_Temp_Range", data=df_all,
              color="k", alpha=0.6, size=3, ax=ax)

coastal_cities = sorted(df_all.loc[df_all["Type"]=="Coastal","City"].unique())
inland_cities  = sorted(df_all.loc[df_all["Type"]=="Inland","City"].unique())

ymin = df_all["Annual_Temp_Range"].min(); ymax = df_all["Annual_Temp_Range"].max()
offset = (ymax - ymin) * 0.12
ax.text(0, ymin - offset, f"Coastal: {', '.join(coastal_cities)}", ha='center', fontsize=10, color='teal')
ax.text(1, ymin - offset, f"Inland: {', '.join(inland_cities)}",  ha='center', fontsize=10, color='brown')
ax.set_ylim(ymin - offset*1.5, ymax + offset/2)
ax.set_title("Coastal vs Inland Annual Temperature Range (2000–2024)")
ax.set_ylabel("Annual Range (°C)")
ax.set_xlabel("Type")

ax2 = axes[1]
if len(ny_series)>0:
    sns.boxplot(y=ny_series.values, color="#a6cee3", width=0.4, ax=ax2)
    sns.stripplot(y=ny_series.values, color="k", size=3, alpha=0.7, ax=ax2)
    ax2.set_xticks([]); ax2.set_ylabel("Annual Range (°C)")
    ax2.set_title("New York (reference)")
    ax2.axhline(np.mean(coastal), ls="--", lw=1, color="teal", label="Coastal mean")
    ax2.axhline(np.mean(inland),  ls="--", lw=1, color="#e07a5f", label="Inland mean")
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
else:
    ax2.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "temp_range_comparison.png"), dpi=300)

summary = {
    "Test": test, "Statistic": stat, "p_value": p,
    "Effect_Cliffs_delta": float(delta),
    "Coastal_n": int(len(coastal)), "Inland_n": int(len(inland)),
    "Coastal_mean": float(np.mean(coastal)), "Inland_mean": float(np.mean(inland)),
    "Coastal_mean_CI95_low": float(c_ci[0]), "Coastal_mean_CI95_high": float(c_ci[1]),
    "Inland_mean_CI95_low": float(i_ci[0]), "Inland_mean_CI95_high": float(i_ci[1]),
    "Used_common_years": bool(USE_COMMON_YEARS),
    "New_York_included_in_stats": False,
    "New_York_n": int(len(ny_series))
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "temp_range_summary.csv"), index=False)

print(f"{test} → p={p:.4g}, δ={delta:.2f} | Coastal mean={np.mean(coastal):.2f} "
      f"[{c_ci[0]:.2f},{c_ci[1]:.2f}] vs Inland {np.mean(inland):.2f} "
      f"[{i_ci[0]:.2f},{i_ci[1]:.2f}] | NY_n={len(ny_series)} | common_years={USE_COMMON_YEARS}")
