import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene

DATA_DIR = "data"
OUT_DIR = "../results/H2"
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = {
    "san_francisco_ca_usa_all_variables.csv": {"name": "San Francisco", "type": "Coastal"},
    "corpus_christi_tx_usa_all_variables.csv": {"name": "Corpus Christi", "type": "Coastal"},
    "mumbai_mh_india__all_variables.csv": {"name": "Mumbai", "type": "Coastal"},
    # "new_york_ny_usa__all_variables.csv": {"name": "New York", "type": "Coastal"}, // Since it lies in boderline
    "chicago_il_usa_all_variables.csv": {"name": "Chicago", "type": "Inland"},
    "columbus_oh_usa__all_variables.csv": {"name": "Columbus", "type": "Inland"},
    "new_delhi_india_all_variables.csv": {"name": "New Delhi", "type": "Inland"}
}

records = []
for file, meta in CITIES.items():
    path = os.path.join(DATA_DIR, file)
    if not os.path.exists(path): 
        continue
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if 'temperature_c' not in df.columns: 
        continue
    df = df[df['temperature_c'].notna()]
    annual = df.groupby(df.index.year)['temperature_c'].agg(lambda x: x.max() - x.min())
    for y, v in annual.items():
        records.append({"City": meta["name"], "Type": meta["type"], "Year": int(y), "Annual_Temp_Range": v})

df_all = pd.DataFrame(records)
df_all.to_csv(os.path.join(OUT_DIR, "annual_temperature_ranges.csv"), index=False)

coastal = df_all.loc[df_all["Type"] == "Coastal", "Annual_Temp_Range"]
inland = df_all.loc[df_all["Type"] == "Inland", "Annual_Temp_Range"]

sc, si = shapiro(coastal), shapiro(inland)
lv = levene(coastal, inland)
if sc.pvalue > 0.05 and si.pvalue > 0.05 and lv.pvalue > 0.05:
    t = ttest_ind(coastal, inland, equal_var=True)
    test, stat, p = "t-test", t.statistic, t.pvalue
else:
    u = mannwhitneyu(coastal, inland, alternative='two-sided')
    test, stat, p = "Mann–Whitney U", u.statistic, u.pvalue

plt.figure(figsize=(7,5), dpi=150)
sns.boxplot(x="Type", y="Annual_Temp_Range", data=df_all, palette="Set2", width=0.5)
sns.swarmplot(x="Type", y="Annual_Temp_Range", data=df_all, color="k", alpha=0.6)
plt.title("Coastal vs Inland Annual Temperature Range (2000–2024)")
plt.ylabel("Annual Range (°C)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "temp_range_comparison.png"), dpi=200)
plt.show()

summary = {
    "Test": test, "Statistic": stat, "p_value": p,
    "Coastal_mean": coastal.mean(), "Inland_mean": inland.mean(),
    "Coastal_std": coastal.std(), "Inland_std": inland.std()
}
pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "temp_range_summary.csv"), index=False)
print(f"{test} → p={p:.4f}")
