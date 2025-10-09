import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import mannwhitneyu

DATA_DIRS = ["../data extraction/outputs", "../Data Extraction/outputs", "./data"]
OUT_DIR = "../results/Hypothesis (H4): Precipitation Seasonality"; os.makedirs(OUT_DIR, exist_ok=True)
CITIES = {
    "mumbai_mh_india__all_variables.csv":     ("Mumbai", "Monsoon"),
    "new_delhi_india_all_variables.csv":      ("New Delhi", "Monsoon"),
    "corpus_christi_tx_usa_all_variables.csv":("Corpus Christi", "Monsoon"),  # treat as Gulf–monsoon influenced
    "san_francisco_ca_usa_all_variables.csv": ("San Francisco", "Temperate"),
    "chicago_il_usa_all_variables.csv":       ("Chicago", "Temperate"),
    "columbus_oh_usa__all_variables.csv":     ("Columbus", "Temperate"),
    "new_york_ny_usa__all_variables.csv":     ("New York", "Temperate"),
}
USE_COMMON_YEARS = True
REPRESENTATIVE = ["mumbai_mh_india__all_variables.csv",
                  "new_delhi_india_all_variables.csv",
                  "chicago_il_usa_all_variables.csv",
                  "san_francisco_ca_usa_all_variables.csv"]

def resolve(fname):
    for d in DATA_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p): return p
    return None

def pick_precip_col(df):
    for c in ["precipitation_mm","precip_mm","precip","pr","ppt"]:
        if c in df.columns: return c
    return None

# ------------ load ------------
raw = {}
years_per_city = {}
missing = []
for fname,(label,regime) in CITIES.items():
    path = resolve(fname)
    if not path: missing.append(fname); continue
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    pcol = pick_precip_col(df)
    if not pcol: continue
    s = df[pcol].dropna()
    if s.empty: continue
    raw[fname] = {"label":label, "regime":regime, "series":s.sort_index()}
    years_per_city[label] = set(s.index.year.unique())

if not raw: raise SystemExit("No data found. Check DATA_DIRS and file names.")


if USE_COMMON_YEARS:
    common = set.intersection(*[set(v["series"].index.year.unique()) for v in raw.values()])
else:
    common = None

# climatology + indices 
clims = []
index_rows = []
for fname,info in raw.items():
    s = info["series"]
    if common is not None:
        s = s[s.index.year.isin(common)]
    monthly = s.groupby(s.index.month).mean()
    annual_mean = monthly.mean()
    si = (monthly.max()-monthly.min()) / annual_mean if annual_mean>0 else np.nan
    cv = monthly.std(ddof=1) / annual_mean if annual_mean>0 else np.nan
    clims.append(pd.DataFrame({
        "City":info["label"], "Regime":info["regime"],
        "Month":np.arange(1,13), "MonthlyMean_mm":monthly.values
    }))
    index_rows.append({
        "City":info["label"], "Regime":info["regime"],
        "SI":float(si), "CV":float(cv),
        "AnnualMean_mm":float(annual_mean)
    })

clim_df = pd.concat(clims, ignore_index=True)
idx_df  = pd.DataFrame(index_rows).sort_values(["Regime","City"])
clim_df.to_csv(os.path.join(OUT_DIR,"precip_monthly_climatology.csv"), index=False)
idx_df.to_csv(os.path.join(OUT_DIR,"precip_seasonality_index.csv"), index=False)

# test (Monsoon > Temperate) 
mon = idx_df.loc[idx_df.Regime=="Monsoon","SI"].dropna().values
tem = idx_df.loc[idx_df.Regime=="Temperate","SI"].dropna().values
u = mannwhitneyu(mon, tem, alternative="greater")
def cliffs_delta(a,b):
    a=np.asarray(a); b=np.asarray(b)
    gt=sum((x>y) for x in a for y in b)
    lt=sum((x<y) for x in a for y in b)
    return (gt-lt)/(len(a)*len(b))
delta = cliffs_delta(mon, tem)
summary = pd.DataFrame([{
    "Test":"Mann–Whitney U (Monsoon>Temperate)",
    "U":float(u.statistic),"p_value":float(u.pvalue),
    "Monsoon_n":int(len(mon)),"Temperate_n":int(len(tem)),
    "Monsoon_SI_mean":float(np.mean(mon)),"Temperate_SI_mean":float(np.mean(tem)),
    "Cliffs_delta":float(delta),
    "Used_common_years":bool(USE_COMMON_YEARS)
}])
summary.to_csv(os.path.join(OUT_DIR,"H4_test_summary.csv"), index=False)

sns.set_style("whitegrid")

# 1) Representative monthly climatologies (2×2)
rep = [r for r in REPRESENTATIVE if r in raw]
n = len(rep); rows, cols = 2, 2
fig, axes = plt.subplots(rows, cols, figsize=(10,7), dpi=200, constrained_layout=True)
axes = axes.flat
for i,fname in enumerate(rep):
    info = raw[fname]
    s = info["series"]
    if common is not None: s = s[s.index.year.isin(common)]
    monthly = s.groupby(s.index.month).mean()
    axes[i].bar(range(1,13), monthly.values)
    axes[i].set_xticks(range(1,13)); axes[i].set_xticklabels(list("JFMAMJJASOND"))
    axes[i].set_title(f"{info['label']} ({info['regime']})")
    axes[i].set_ylabel("mm/month")
for j in range(len(rep), rows*cols):
    fig.delaxes(axes[j])
fig.suptitle("Monthly Precipitation Climatology (2000–2024)", fontsize=12)
fig.savefig(os.path.join(OUT_DIR,"H4_monthly_climatology_2x2.png"), dpi=300)

# 2) Seasonality Index box+swarm
plt.figure(figsize=(7,5), dpi=200)
sns.boxplot(x="Regime", y="SI", data=idx_df, width=0.45, palette="Set2")
sns.swarmplot(x="Regime", y="SI", data=idx_df, color="k", alpha=0.7, size=4)
plt.title("Seasonality Index (SI) by Regime")
plt.ylabel("SI = (max−min)/annual mean")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H4_seasonality_index_boxplot.png"), dpi=300)

# 3) Annotated SI bar by city
plt.figure(figsize=(10,5), dpi=200)
order = idx_df.sort_values("SI", ascending=False)
sns.barplot(x="City", y="SI", hue="Regime", data=order, dodge=False, palette="Set2")
plt.xticks(rotation=30, ha="right"); plt.ylabel("Seasonality Index")
plt.title("City-wise Precip Seasonality")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H4_seasonality_index_by_city.png"), dpi=300)

print(f"U={u.statistic:.1f}, p={u.pvalue:.4g}, δ={delta:.2f} | Monsoon SI mean={np.mean(mon):.2f}, Temperate SI mean={np.mean(tem):.2f}")
print(f"Saved outputs in: {OUT_DIR}")
