import pandas as pd, numpy as np, os, matplotlib.pyplot as plt
from scipy.signal import periodogram

OUT_DIR = "../results/H3_SST"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = "../Data Extraction/outputs"

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
    df = df.sort_values('time')
    df = df[['time', 'temperature_c']].dropna()
    df = df.set_index('time').asfreq('MS')
    return df

def detrend(y):
    x = np.arange(len(y))
    return y - np.poly1d(np.polyfit(x, y, 1))(x)

def fft_analysis(y, fs=12):
    f, Pxx = periodogram(y, fs=fs)
    return f, Pxx

results = []
climatologies = {}
spectra = {}

for city, path in files.items():
    df = load_city_data(path)
    df['temp_detr'] = detrend(df['temperature_c'])
    df['month'] = df.index.month
    clim = df.groupby('month')['temperature_c'].mean()
    climatologies[city] = clim

    amp = clim.max() - clim.min()
    f, Pxx = fft_analysis(df['temp_detr'])
    spectra[city] = (f, Pxx)

    dom_idx = np.argmax(Pxx)
    dom_freq, dom_power = f[dom_idx], Pxx[dom_idx]
    annual_power = Pxx[np.argmin(np.abs(f - 1))]

    r = np.abs(np.mean(np.exp(1j*2*np.pi*df['month']/12)))
    n = len(df)
    p_rayleigh = np.exp(-n * r**2)

    results.append({
        "City": city,
        "DominantFreq(cycles/yr)": round(dom_freq, 3),
        "Power@1cyc/yr": round(annual_power, 4),
        "Rayleigh_p": round(p_rayleigh, 5),
        "Amplitude(°C)": round(amp, 2)
    })

# === Combined Climatology Plot ===
plt.figure(figsize=(10, 6))
for city, clim in climatologies.items():
    plt.plot(clim.index, clim.values, marker='o', label=city)
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.title("Monthly SST (Proxy) Climatology Across Cities")
plt.legend(loc="best", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/H3_combined_climatology.png", dpi=300)
plt.close()

# === Combined Power Spectrum Plot ===
# === Improved Combined Power Spectrum Plot ===
plt.figure(figsize=(10, 6))

for city, (f, Pxx) in spectra.items():
    plt.plot(f, Pxx, lw=1.8, label=city)

plt.axvline(1, color='k', ls='--', lw=1.2, label='1 cycle/year')
plt.xlabel("Frequency (cycles/year)", fontsize=12)
plt.ylabel("Power (log scale)", fontsize=12)
plt.title("SST (Proxy) Power Spectrum Across Cities (Zoomed)", fontsize=13)
plt.xlim(0, 2)
plt.yscale('log')
plt.grid(alpha=0.4, which='both', linestyle=':')
plt.legend(loc="upper right", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/H3_combined_power_spectrum_enhanced.png", dpi=400)
plt.close()


# === Results Summary ===
res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUT_DIR}/H3_SST_results.csv", index=False)
print("\n=== H3: SST Seasonal Cycle Results ===")
print(res_df)
print(f"\nCombined plots and CSV saved in {OUT_DIR}")
