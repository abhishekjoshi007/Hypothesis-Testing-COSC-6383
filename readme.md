# Hypothesis Testing - Climate Analysis (COSC-6383)

This repository contains code and figures for testing **eight climate hypotheses** using monthly data (January 2000 – December 2024) for several cities around the world. Each hypothesis is implemented as a single Python script that reads the same city CSVs, performs the required statistics, and writes both figures and tabular summaries to a results folder.

---

## 📁 Repository Structure

```
.
├── Analysis/
│   ├── monsoon_trigger_H1.py
│   ├── analysis_coastal_inland_temp_range_H2.py
│   ├── SST_seasonal_cycle_H3.py
│   ├── precipitation_seasonality_H4.py
│   ├── monthly_mean_temperature_normality_H5.py
│   ├── Convective_feedback_H6.py
│   ├── CO2_city_trend_H7.py
│   └── hpoint_vs_grid_consistency_H8.py
│
├── Data Extraction/
│   ├── point_extract.py
│   ├── gridded_extraction_only.py
│   ├── additional_pt.py                    # SLP + RH extraction and merge
│   └── outputs/
│       ├── chicago_il_usa_all_variables.csv
│       ├── columbus_oh_usa_all_variables.csv
│       ├── corpus_christi_tx_usa_all_variables.csv
│       ├── mumbai_mh_india_all_variables.csv
│       ├── new_delhi_india_all_variables.csv
│       ├── new_york_ny_usa_all_variables.csv
│       └── san_francisco_ca_usa_all_variables.csv
│
└── results/
    ├── monsoon_trigger_H1/
    ├── analysis_coastal_inland_temp_range_H2/
    ├── SST_seasonal_cycle_H3/
    ├── precipitation_seasonality_H4/
    ├── monthly_mean_temperature_normality_H5/
    ├── Convective_feedback_H6/
    ├── CO2_city_trend_H7/
    └── hpoint_vs_grid_consistency_H8/
```

### City CSV Schema

All scripts expect the following column names:

- `temperature_c`
- `precipitation_mm`
- `sea_level_pressure_hpa`
- `relative_humidity_pct`

> **Note:** Scripts are tolerant to common alternates (e.g., `temp`, `tavg`), but the above names are preferred.

---

## 🚀 Quick Start

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install Dependencies

Create a `requirements.txt` file with:

```
pandas
numpy
matplotlib
seaborn
scipy
xarray
```

Then install:

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place city CSV files in `Data Extraction/outputs/` (already provided for 6–7 cities).

### 4. Run Analysis Scripts

Execute any hypothesis script from the `Analysis/` folder:

```bash
python Analysis/analysis_coastal_inland_temp_range_H2.py
```

### 5. View Results

Find generated figures (`.png`) and summaries (`.csv`) in:

```
results/<Hypothesis_Folder>/
```

---

## 📊 Hypotheses Overview

### H1: Monsoon Trigger

**Statement:** High temperature + low sea-level pressure increases probability of convective rainfall events.

**Script:** `Analysis/monsoon_trigger_H1.py`

**Method:**
- Build monthly anomalies for `temperature_c` and `sea_level_pressure_hpa` per city
- Binary rain event = precipitation above percentile threshold (e.g., 75th)
- Logistic regression: `RainEvent ~ TempAnom + (−SLP_Anom)`
- Optional lag checks (e.g., SLP leads P by 1 month)

**Outputs:** `results/monsoon_trigger_H1/`
- `monsoon_trigger_coefficients.csv` — coefficients, odds ratios, p-values by city
- `monsoon_trigger_calibration.png` — reliability/ROC or calibration curve per city
- `monsoon_trigger_examples.png` — example time windows with low SLP → high P

**Key Config:** Rain threshold percentile, lag window, list of cities

---

### H2: Coastal vs Inland Annual Temperature Range ✅

**Statement:** Coastal cities have smaller annual temperature ranges than inland cities.

**Script:** `Analysis/analysis_coastal_inland_temp_range_H2.py`

**Method:**
- Compute annual range = `max(monthly temp) − min(monthly temp)` per year
- Group into Coastal vs Inland lists
- Normality + equal variance checks → t-test or Mann–Whitney U
- Report Cliff's δ and bootstrapped 95% CI

**Outputs:** `results/analysis_coastal_inland_temp_range_H2/`
- `annual_temperature_ranges.csv` — one row per city-year
- `temp_range_comparison.png` — box + swarm plot with city labels
- `temp_range_summary.csv` — test used, p-value, Cliff's δ, mean ± CI

**Key Config:** City lists & coastal/inland tags, `USE_COMMON_YEARS`, data paths

---

### H3: Sea-Surface Temperature Seasonal Cycle

**Statement:** SST/near-surface temperature follows a 1 cycle/year seasonal cycle.

**Script:** `Analysis/SST_seasonal_cycle_H3.py`

**Method:**
- Detrend monthly temperature per city
- FFT to identify dominant frequency & amplitude
- Compute power at 1.0 cycles/yr and phase (month of peak)

**Outputs:** `results/SST_seasonal_cycle_H3/`
- `fft_peaks_by_city.csv` — dominant frequency, power, phase per city
- `seasonal_fft_panels.png` — log-power vs frequency with red line at 1.0 cycles/yr
- `seasonal_monthly_climatology.png` — average annual cycle (12-month climatology)

**Key Config:** Detrend flag, frequency grid, variable selection

---

### H4: Precipitation Seasonality (Monsoon vs Temperate)

**Statement:** Monsoon regions exhibit stronger precipitation seasonality than temperate ones.

**Script:** `Analysis/precipitation_seasonality_H4.py`

**Method:**
- Compute climatological monthly means (12-month cycle)
- Calculate Seasonality Index: `(max − min) / mean` or coefficient of variation
- Compare Monsoon vs Temperate groups (t-test or Mann–Whitney U)

**Outputs:** `results/precipitation_seasonality_H4/`
- `seasonality_metrics.csv` — city-level SI, CV, max/min months
- `seasonality_group_comparison.png` — box + swarm plot
- `seasonality_radar_or_bars.png` — 12-month climatology per city

**Key Config:** Group assignment for cities, SI definition

---

### H5: Monthly Mean Temperature Normality

**Statement:** Monthly means tend toward a Normal distribution (CLT effect).

**Script:** `Analysis/monthly_mean_temperature_normality_H5.py`

**Method:**
- For each city, take all monthly means (2000–2024)
- Shapiro–Wilk test and Q–Q plots
- Fit N(μ, σ) and overlay, report p-values

**Outputs:** `results/monthly_mean_temperature_normality_H5/`
- `normality_tests.csv` — Shapiro W, p-value, μ, σ per city
- `temp_hist_qq_panels.png` — histogram + Normal fit + Q–Q per city

**Key Config:** Outlier handling, raw vs anomaly data

---

### H6: Convective Feedback (Hotter → Drier?)

**Statement:** In some climates, higher temperature anomalies coincide with reduced precipitation.

**Script:** `Analysis/Convective_feedback_H6.py`

**Method:**
- Build anomalies (remove monthly climatology) for temperature & precipitation
- Pearson/Spearman correlation per city
- Optional lagged correlations
- Global sign test/meta-analysis across cities

**Outputs:** `results/Convective_feedback_H6/`
- `temp_precip_anomaly_correlations.csv` — r, p per city (+ best lag)
- `scatter_with_regression.png` — anomaly scatter + fit per city
- `lag_correlogram.png` — correlation vs lag

**Key Config:** Anomaly method, lag window, correlation type

---

### H7: CO₂ — City Temperature Trend

**Statement:** Rising global CO₂ is correlated with long-term increases in city-level temperature.

**Script:** `Analysis/CO2_city_trend_H7.py`

**Method:**
- Import monthly global CO₂ series (e.g., Mauna Loa) and city `temperature_c`
- Regress: `Temperature ~ Time + CO₂` (separate pure time trend vs CO₂)
- Report slope, p-values, and partial R² for CO₂

**Outputs:** `results/CO2_city_trend_H7/`
- `co2_regression_by_city.csv` — coefficients, p, R², diagnostics
- `co2_temp_trend_panels.png` — time series with fitted trend per city
- `co2_vs_temp_scatter.png` — mean-removed CO₂ vs temp with regression line

**Key Config:** CO₂ file path, detrend options, per-city sensitivity plot

---

### H8: Point vs Grid Consistency

**Statement:** Temperature at station (point) agrees with grid cutout better than precipitation.

**Script:** `Analysis/hpoint_vs_grid_consistency_H8.py`  
*(Alternative filename: `point_vs_grid_consistency_H8.py`)*

**Method:**
- For each city, align monthly point series vs grid cell (nearest or averaged 3×3)
- Compute R², RMSE for temperature & precipitation
- Summarize by city and globally

**Outputs:** `results/hpoint_vs_grid_consistency_H8/`
- `grid_point_metrics.csv` — R² & RMSE by city/variable
- `grid_point_scatter_panels.png` — point vs grid with 1:1 line
- `metrics_summary.csv` — group means and deltas (Temp vs Precip)

**Key Config:** netCDF paths, lon convention (−180–180 vs 0–360), neighborhood size

---

## 📝 Output Naming Conventions

Each script generates:

1. **Figures:** `*.png` (publication-ready, 300 dpi)
2. **Tables:** `*.csv` (reproducible key numbers)

All outputs are saved to: `results/<script_stem>/`

**Example:**
```
results/analysis_coastal_inland_temp_range_H2/
├── annual_temperature_ranges.csv
├── temp_range_comparison.png
└── temp_range_summary.csv
```

---

## 🔄 Reproducibility & Adding Cities

### Adding a New City

1. Drop a new city CSV in `Data Extraction/outputs/` with the four standard columns
2. Add the filename and metadata (label, group) at the top of any script:
   ```python
   CITIES = {
       'new_city_name': {
           'file': 'new_city_data.csv',
           'label': 'New City',
           'group': 'Coastal'  # or 'Inland', 'Monsoon', etc.
       }
   }
   ```
3. Re-run the script — new city appears automatically in figures and summaries

> **Tip:** To keep groups balanced (e.g., Coastal vs Inland), update group tags when adding cities.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"No data loaded"** | Check `DATA_DIR(S)` in the script and verify CSV file names |
| **Longitude mismatch (grid files)** | Use `lon % 360` or add 360 to negative longitudes (handled in H8) |
| **Normality/variance warnings** | Scripts auto-fallback to non-parametric tests (e.g., Mann–Whitney U) |
| **Overlapping labels on plots** | Increase figure height or adjust margin/annotation offsets |

---

## 🔧 Batch Execution (Optional)

Run all hypotheses in sequence:

```bash
python Analysis/monsoon_trigger_H1.py && \
python Analysis/analysis_coastal_inland_temp_range_H2.py && \
python Analysis/SST_seasonal_cycle_H3.py && \
python Analysis/precipitation_seasonality_H4.py && \
python Analysis/monthly_mean_temperature_normality_H5.py && \
python Analysis/Convective_feedback_H6.py && \
python Analysis/CO2_city_trend_H7.py && \
python Analysis/hpoint_vs_grid_consistency_H8.py
```

This will refresh all results and figures under `results/`.

---

## 👥 Credits

**Team:** Abhishek & Nikhilesh

**Data Extraction:**
- `point_extract.py`
- `gridded_extraction_only.py`
- `additional_pt.py` (SLP & RH merge)

**Libraries:** pandas, numpy, scipy, matplotlib, seaborn, xarray


