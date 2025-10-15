"""
Script Name: monthly_mean_temperature_distribution_H5.py
Purpose:
    Tests Hypothesis H5 – Monthly mean temperatures converge toward a normal
    distribution due to averaging of daily fluctuations.

Inputs:
    • Monthly temperature data (2000–2024) for all seven cities.

Outputs:
    • ../results/H5/temp_distribution_plots.png – Q-Q plots per city
    • ../results/H5/temp_normality_tests.csv – Shapiro–Wilk results

Methods:
    • Histogram and Q–Q plots for visual inspection.
    • Shapiro–Wilk test for normality.
    • Comparison of skewness and kurtosis.

Expected Result:
    • Not showing normality.
"""


import os, math, pandas as pd, numpy as np, matplotlib.pyplot as plt, scipy.stats as st

DATA_DIRS = ["../Data Extraction/outputs","../data extraction/outputs","./data","./"]
OUT_DIR = "../results/H5_clean"; os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "Corpus Christi":"corpus_christi_tx_usa_all_variables.csv",
    "San Francisco":"san_francisco_ca_usa_all_variables.csv",
    "New Delhi":"new_delhi_india_all_variables.csv",
    "Chicago":"chicago_il_usa_all_variables.csv",
    "New York":"new_york_ny_us_all_variables.csv",
    "Mumbai":"mumbai_mh_india_all_variables.csv",
    "Columbus":"columbus_oh_usa_all_variables.csv",
}

def find(f):
    for d in DATA_DIRS:
        p=os.path.join(d,f)
        if os.path.exists(p): return p
    return None

def load_z(city,fname):
    p=find(fname); 
    if not p: return None
    df=pd.read_csv(p,parse_dates=[0],index_col=0)
    if "temperature_c" not in df.columns: return None
    m=df["temperature_c"].resample("MS").mean()
    if m.notna().sum()<24: return None
    clim=m.groupby(m.index.month).mean()
    anom=m-m.index.month.map(clim)
    z=(anom-anom.mean())/anom.std(ddof=1)
    return z.dropna()

# collect series + stats
series, stats_rows = {}, []
for city,fname in FILES.items():
    z=load_z(city,fname)
    if z is None: continue
    w=st.shapiro(z.values)
    skew=st.skew(z.values, bias=False)
    series[city]=z
    stats_rows.append({"City":city,"n":len(z),"W":w.statistic,"p":w.pvalue,"skew":skew})
stats=pd.DataFrame(stats_rows).sort_values("City")
stats.to_csv(os.path.join(OUT_DIR,"H5_normality_summary.csv"),index=False)

# unified axes
allvals=np.concatenate([v.values for v in series.values()])
osm_all=np.sort(st.norm.ppf((np.arange(len(allvals))+0.5)/len(allvals)))
xlim=(np.percentile(osm_all,1),np.percentile(osm_all,99))
ylim=(np.percentile(allvals,1),np.percentile(allvals,99))

cities=list(series.keys())
r,c=2,4
plt.style.use("default")
fig,axes=plt.subplots(r,c,figsize=(16,7),dpi=200)
axes=axes.flatten()

for ax in axes: 
    ax.axis("off")

for ax,city in zip(axes,cities):
    z=series[city].values
    q_theo=st.norm.ppf((np.arange(len(z))+0.5)/len(z))
    q_data=np.sort(z)
    ax.scatter(q_theo,q_data,s=10,alpha=.8)
    lo=min(xlim[0],q_theo.min()); hi=max(xlim[1],q_theo.max())
    ax.plot([lo,hi],[lo,hi],'r-',lw=1)
    p=stats.loc[stats.City.eq(city),"p"].iloc[0]
    sk=stats.loc[stats.City.eq(city),"skew"].iloc[0]
    ax.set_title(f"{city}  (p={p:.3f}, skew={sk:.2f})",fontsize=11, pad=6)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(alpha=.2); ax.axis("on")
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Theoretical quantiles",fontsize=9)
    ax.set_ylabel("Ordered values",fontsize=9)

fig.suptitle("H5: Normality of Monthly Temperature Anomalies (2000–2024)",fontsize=14,y=0.98)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT_DIR,"H5_QQplots_clean.png"),dpi=300)
print("Saved:", OUT_DIR)
