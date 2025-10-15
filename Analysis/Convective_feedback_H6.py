"""
Script Name: convective_feedback_H6.py
Purpose:
    Tests Hypothesis H6 – Periods of higher temperature anomalies coincide with
    reduced precipitation in some cities, reflecting hydrological trade-offs.

Inputs:
    • Monthly temperature and precipitation data (2000–2024).

Outputs:
    • ../results/H6/convective_feedback_scatter.png
    • ../results/H6/convective_feedback_summary.csv

Methods:
    • Correlation analysis between temperature anomalies and precipitation.
    • Regression to estimate slope and sign of feedback relationship.

Expected Result:
    • Negative correlation between temperature anomaly and precipitation,
      especially in arid and inland locations.
"""


import os, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon

DATA_DIRS = ["../Data Extraction/outputs","../data extraction/outputs","./data","./"]
OUT_DIR = "../results/H6"; os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "Corpus Christi":"corpus_christi_tx_usa_all_variables.csv",
    "San Francisco":"san_francisco_ca_usa_all_variables.csv",
    "New Delhi":"new_delhi_india_all_variables.csv",
    "Chicago":"chicago_il_usa_all_variables.csv",
    "New York":"new_york_ny_us_all_variables.csv",
    "Mumbai":"mumbai_mh_india_all_variables.csv",
    "Columbus":"columbus_oh_usa_all_variables.csv",
}

TEMP_CANDS   = ["temperature_c","temp_c","tavg","tas","temperature"]
PRECIP_CANDS = ["precipitation_mm","precip_mm","pr","precip","tp"]

def find(fname):
    for d in DATA_DIRS:
        p=os.path.join(d,fname)
        if os.path.exists(p): return p
    return None

def pick(df,cands):
    for c in cands:
        if c in df.columns: return c
    return None

def monthly_anom(s, idx):
    clim = s.groupby(idx.month).transform("mean")
    return s - clim

rows, skipped = [], []

for city,fname in FILES.items():
    p=find(fname)
    if not p:
        skipped.append((city,"missing file")); continue

    df=pd.read_csv(p,parse_dates=[0],index_col=0).sort_index()
    tcol = pick(df, TEMP_CANDS)
    pcol = pick(df, PRECIP_CANDS)
    if not all([tcol,pcol]):
        skipped.append((city,"missing required columns")); continue

    df = df[[tcol,pcol]].resample("MS").mean()
    df["t_anom"] = monthly_anom(df[tcol], df.index)
    df["p_anom"] = monthly_anom(df[pcol], df.index)

    df = df.dropna(subset=["t_anom","p_anom"])
    if len(df) < 12:
        skipped.append((city,"not enough data")); continue

    r, pval = pearsonr(df["t_anom"], df["p_anom"])
    rows.append({"City":city,"r_temp_precip":r,"p_value":pval})
    
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR,"H6_convective_feedback_summary.csv"),index=False)

neg = (res["r_temp_precip"]<0).sum()
n = len(res)
p_sign = wilcoxon(res["r_temp_precip"], alternative="less")[1] if n>1 else np.nan

ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5*nrows), dpi=160)
axes = axes.flatten()

for i,(city,fname) in enumerate(FILES.items()):
    if city not in res["City"].values: continue
    p=find(fname)
    df=pd.read_csv(p,parse_dates=[0],index_col=0).sort_index()
    tcol = pick(df,TEMP_CANDS); pcol = pick(df,PRECIP_CANDS)
    df=df.resample("MS").mean()
    df["t_anom"]=monthly_anom(df[tcol],df.index)
    df["p_anom"]=monthly_anom(df[pcol],df.index)
    ax=axes[i]
    sns.regplot(x="t_anom",y="p_anom",data=df,ax=ax,scatter_kws={"s":8,"alpha":.6},line_kws={"color":"red"})
    r = res.loc[res.City==city,"r_temp_precip"].values[0]
    pval = res.loc[res.City==city,"p_value"].values[0]
    ax.set_title(f"{city} (r={r:.2f}, p={pval:.3f})")
    ax.set_xlabel("Temp anomaly (°C)")
    ax.set_ylabel("Precip anomaly (mm)")

for j in range(i+1, len(axes)): axes[j].axis("off")
fig.suptitle("H6: Convective Feedback — Temp vs Precip Anomalies (2000–2024)", y=1.02, fontsize=13)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H6_city_scatter.png"),dpi=300,bbox_inches="tight")

plt.figure(figsize=(6,4),dpi=200)
sns.violinplot(y=res["r_temp_precip"], inner="point", color="skyblue")
plt.axhline(0, ls="--", c="k")
plt.ylabel("Pearson r (Temp vs Precip Anomaly)")
plt.title(f"H6: Distribution of Temp–Precip Correlations (Sign test p={p_sign:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H6_violin_correlations.png"),dpi=300)

print("\n[RESULTS]")
print(res)
print(f"\nNegative correlations: {neg}/{n} cities, Sign-test p={p_sign:.3f}")
if skipped:
    print("\n[SKIPPED]")
    for c,why in skipped: print(f" - {c}: {why}")
print("\nSaved to:", OUT_DIR)

# Convective feedback (hotter months ↔ reduced precip)
import os, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon

DATA_DIRS = ["../Data Extraction/outputs","../data extraction/outputs","./data","./"]
OUT_DIR = "../results/H6"; os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "Corpus Christi":"corpus_christi_tx_usa_all_variables.csv",
    "San Francisco":"san_francisco_ca_usa_all_variables.csv",
    "New Delhi":"new_delhi_india_all_variables.csv",
    "Chicago":"chicago_il_usa_all_variables.csv",
    "New York":"new_york_ny_us_all_variables.csv",
    "Mumbai":"mumbai_mh_india_all_variables.csv",
    "Columbus":"columbus_oh_usa_all_variables.csv",
}

TEMP_CANDS   = ["temperature_c","temp_c","tavg","tas","temperature"]
PRECIP_CANDS = ["precipitation_mm","precip_mm","pr","precip","tp"]

def find(fname):
    for d in DATA_DIRS:
        p=os.path.join(d,fname)
        if os.path.exists(p): return p
    return None

def pick(df,cands):
    for c in cands:
        if c in df.columns: return c
    return None

def monthly_anom(s, idx):
    clim = s.groupby(idx.month).transform("mean")
    return s - clim

rows, skipped = [], []

for city,fname in FILES.items():
    p=find(fname)
    if not p:
        skipped.append((city,"missing file")); continue

    df=pd.read_csv(p,parse_dates=[0],index_col=0).sort_index()
    tcol = pick(df, TEMP_CANDS)
    pcol = pick(df, PRECIP_CANDS)
    if not all([tcol,pcol]):
        skipped.append((city,"missing required columns")); continue

    df = df[[tcol,pcol]].resample("MS").mean()
    df["t_anom"] = monthly_anom(df[tcol], df.index)
    df["p_anom"] = monthly_anom(df[pcol], df.index)

    df = df.dropna(subset=["t_anom","p_anom"])
    if len(df) < 12:
        skipped.append((city,"not enough data")); continue

    r, pval = pearsonr(df["t_anom"], df["p_anom"])
    rows.append({"City":city,"r_temp_precip":r,"p_value":pval})
    
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR,"H6_convective_feedback_summary.csv"),index=False)

neg = (res["r_temp_precip"]<0).sum()
n = len(res)
p_sign = wilcoxon(res["r_temp_precip"], alternative="less")[1] if n>1 else np.nan

ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5*nrows), dpi=160)
axes = axes.flatten()

for i,(city,fname) in enumerate(FILES.items()):
    if city not in res["City"].values: continue
    p=find(fname)
    df=pd.read_csv(p,parse_dates=[0],index_col=0).sort_index()
    tcol = pick(df,TEMP_CANDS); pcol = pick(df,PRECIP_CANDS)
    df=df.resample("MS").mean()
    df["t_anom"]=monthly_anom(df[tcol],df.index)
    df["p_anom"]=monthly_anom(df[pcol],df.index)
    ax=axes[i]
    sns.regplot(x="t_anom",y="p_anom",data=df,ax=ax,scatter_kws={"s":8,"alpha":.6},line_kws={"color":"red"})
    r = res.loc[res.City==city,"r_temp_precip"].values[0]
    pval = res.loc[res.City==city,"p_value"].values[0]
    ax.set_title(f"{city} (r={r:.2f}, p={pval:.3f})")
    ax.set_xlabel("Temp anomaly (°C)")
    ax.set_ylabel("Precip anomaly (mm)")

for j in range(i+1, len(axes)): axes[j].axis("off")
fig.suptitle("H6: Convective Feedback — Temp vs Precip Anomalies (2000–2024)", y=1.02, fontsize=13)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H6_city_scatter.png"),dpi=300,bbox_inches="tight")

plt.figure(figsize=(6,4),dpi=200)
sns.violinplot(y=res["r_temp_precip"], inner="point", color="skyblue")
plt.axhline(0, ls="--", c="k")
plt.ylabel("Pearson r (Temp vs Precip Anomaly)")
plt.title(f"H6: Distribution of Temp–Precip Correlations (Sign test p={p_sign:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H6_violin_correlations.png"),dpi=300)

print("\n[RESULTS]")
print(res)
print(f"\nNegative correlations: {neg}/{n} cities, Sign-test p={p_sign:.3f}")
if skipped:
    print("\n[SKIPPED]")
    for c,why in skipped: print(f" - {c}: {why}")
print("\nSaved to:", OUT_DIR)
