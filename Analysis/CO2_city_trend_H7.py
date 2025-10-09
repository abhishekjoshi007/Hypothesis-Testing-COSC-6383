import pandas as pd, numpy as np, statsmodels.api as sm, os, matplotlib.pyplot as plt

OUT_DIR = "../results/H7"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = "../Data Extraction/outputs"   # or wherever your CSVs are stored

files = {
    "New York": f"{DATA_DIR}/new_york_ny_us_all_variables.csv",
    "Mumbai": f"{DATA_DIR}/mumbai_mh_india_all_variables.csv",
    "Corpus Christi": f"{DATA_DIR}/corpus_christi_tx_usa_all_variables.csv",
    "Columbus": f"{DATA_DIR}/columbus_oh_usa_all_variables.csv",
    "San Francisco": f"{DATA_DIR}/san_francisco_ca_usa_all_variables.csv",
    "New Delhi": f"{DATA_DIR}/new_delhi_india_all_variables.csv",
    "Chicago": f"{DATA_DIR}/chicago_il_usa_all_variables.csv",
}


def load_city_data(path):
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    if 'co2_ppm' not in df: df['co2_ppm'] = np.nan
    if 'mei' not in df: df['mei'] = np.nan
    return df.sort_values('time')

def detrend_and_deseason(s):
    x = np.arange(len(s))
    detr = s - np.poly1d(np.polyfit(x, s, 1))(x)
    return detr - detr.groupby(s.index.month).transform('mean')

def newey_west_ols(y, X, lags=6):
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

city_dfs = {c: load_city_data(p) for c, p in files.items()}
co2_mei = pd.concat([d[['time', 'co2_ppm', 'mei']] for d in city_dfs.values() if 'co2_ppm' in d], axis=0).dropna()
co2_mei = co2_mei.groupby('time').mean().reset_index()

for c, d in city_dfs.items():
    d = d.merge(co2_mei, on='time', how='left', suffixes=('', '_g'))
    d['co2_ppm'] = d['co2_ppm'].fillna(d['co2_ppm_g'])
    d['mei'] = d['mei'].fillna(d['mei_g'])
    d.drop(columns=['co2_ppm_g', 'mei_g'], inplace=True)
    city_dfs[c] = d

coefs = []
for city, df in city_dfs.items():
    df = df.dropna(subset=['temperature_c', 'co2_ppm', 'mei']).copy()
    df.set_index('time', inplace=True)
    df['temp_anom'] = detrend_and_deseason(df['temperature_c'])
    m = newey_west_ols(df['temp_anom'], df[['co2_ppm', 'mei']], lags=6)
    coefs.append({
        "City": city,
        "β_CO2": m.params['co2_ppm'],
        "p_CO2": m.pvalues['co2_ppm'],
        "β_MEI": m.params['mei'],
        "p_MEI": m.pvalues['mei'],
        "R²": m.rsquared
    })
    df[['temperature_c', 'co2_ppm']].dropna().to_csv(f"{OUT_DIR}/{city.replace(' ','_')}_aligned.csv", index=True)

summary = pd.DataFrame(coefs)
summary.to_csv(f"{OUT_DIR}/H7_regression_summary.csv", index=False)
print(summary.round(4))

plt.figure(figsize=(9,6))
plt.bar(summary['City'], summary['β_CO2'], color='steelblue', edgecolor='black')
plt.axhline(0, color='gray', lw=0.8)
plt.ylabel("β_CO₂"); plt.title("CO₂ Effect on Temperature (After ENSO Control)")
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/H7_CO2_coefficients.png", dpi=300)
plt.close()

for c, df in city_dfs.items():
    df = df.dropna(subset=['temperature_c','co2_ppm'])
    fig, ax1 = plt.subplots(figsize=(9,5))
    ax2 = ax1.twinx()
    ax1.plot(df['time'], df['temperature_c'], color='tab:red', label='Temp (°C)')
    ax2.plot(df['time'], df['co2_ppm'], color='tab:blue', alpha=0.7, label='CO₂ (ppm)')
    ax1.set_xlabel("Year"); ax1.set_ylabel("Temp (°C)", color='tab:red')
    ax2.set_ylabel("CO₂ (ppm)", color='tab:blue')
    plt.title(f"{c}: CO₂ vs Temp Trend")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/H7_dual_axis_{c.replace(' ','_')}.png", dpi=300)
    plt.close()

print(f"\nResults saved in {OUT_DIR}")
