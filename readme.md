## Hypothesis-Testing-COSC-6383

This repository contains code and figures for testing eight climate hypotheses using monthly data (Jan-2000 … Dec-2024) for several cities around the world. Each hypothesis is implemented as a single Python script that reads the same city CSVs, performs the required statistics, and writes both figures and tabular summaries to a results folder.

# 1) What’s in here
.
├─ Analysis/
│  ├─ analysis_coastal_inland_temp_range_H2.py
│  ├─ hpoint_vs_grid_consistency_H8.py         # (a.k.a. point_vs_grid_consistency_H8.py)
│  ├─ monsoon_trigger_H1.py
│  ├─ monthly_mean_temperature_normality_H5.py
│  ├─ precipitation_seasonality_H4.py
│  ├─ SST_seasonal_cycle_H3.py
│  ├─ Convective_feedback_H6.py
│  └─ CO2_city_trend_H7.py
│
├─ Data Extraction/
│  ├─ point_extract.py
│  ├─ gridded_extraction_only.py
│  ├─ additional_pt.py                  # SLP + RH extraction and merge
│  └─ outputs/
│      ├─ chicago_il_usa_all_variables.csv
│      ├─ columbus_oh_usa_all_variables.csv
│      ├─ corpus_christi_tx_usa_all_variables.csv
│      ├─ mumbai_mh_india_all_variables.csv
│      ├─ new_delhi_india_all_variables.csv
│      ├─ new_york_ny_usa_all_variables.csv
│      └─ san_francisco_ca_usa_all_variables.csv
│
└─ results/
   ├─ analysis_coastal_inland_temp_range_H2/
   ├─ hpoint_vs_grid_consistency_H8/
   ├─ monsoon_trigger_H1/
   ├─ monthly_mean_temperature_normality_H5/
   ├─ precipitation_seasonality_H4/
   ├─ SST_seasonal_cycle_H3/
   ├─ Convective_feedback_H6/
   └─ CO2_city_trend_H7/

City CSV schema (columns that scripts look for)

temperature_c

precipitation_mm

sea_level_pressure_hpa

relative_humidity_pct

Scripts are tolerant to a few common alternates (e.g., temp, tavg) but the above names are preferred.

# 2) Quick start

Create & activate env

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # see minimal list below


Minimal requirements.txt:

pandas
numpy
matplotlib
seaborn
scipy
xarray


Put city CSVs in Data Extraction/outputs/ (already provided for 6–7 cities).

Run any hypothesis script from the Analysis/ folder (or repo root) e.g.

python Analysis/analysis_coastal_inland_temp_range_H2.py


Find results in results/<Hypothesis_Folder>/ as .png figures and .csv summaries.

# 3) Hypotheses: which script triggers what, and what gets saved

Below, each hypothesis lists: script → method → outputs (filenames and where they go) and the key config knobs you can adjust at the top of the script (paths, city lists, flags).

H1) Monsoon Trigger

Statement: High temperature + low sea-level pressure increases probability of convective rainfall events.

Script: Analysis/monsoon_trigger_H1.py

Method:

Build monthly anomalies for temperature_c and sea_level_pressure_hpa per city.

Binary rain event = precipitation above a percentile threshold (e.g., 75th).

Logistic regression: RainEvent ~ TempAnom + (−SLP_Anom), with odds ratios reported per city.

Optional lag checks (e.g., SLP leads P by 1 month).

Outputs (saved to results/monsoon_trigger_H1/)

monsoon_trigger_coefficients.csv — coefficients, odds ratios, p-values by city

monsoon_trigger_calibration.png — reliability/ROC or calibration curve per city

monsoon_trigger_examples.png — example time windows with low SLP → high P

Key config: rain threshold percentile; lag window; list of cities.

H2) Coastal vs Inland Annual Temperature Range ✅ (already executed)

Statement: Coastal cities have smaller annual temperature ranges than inland cities.

Script: Analysis/analysis_coastal_inland_temp_range_H2.py

Method:

For each city, compute annual range = max(monthly temp) − min(monthly temp) per year.

Group into Coastal vs Inland lists; optionally restrict to common years across cities.

Normality + equal variance checks → t-test else Mann–Whitney U; report Cliff’s δ and bootstrapped 95% CI of means.

Outputs (saved to results/analysis_coastal_inland_temp_range_H2/)

annual_temperature_ranges.csv — one row per city-year (City, Type, Year, Range)

temp_range_comparison.png — box + swarm plot with city labels under each group

temp_range_summary.csv — test used, p-value, Cliff’s δ, mean ± CI, sample sizes

Key config: list of cities & coastal/inland tags; USE_COMMON_YEARS; data paths list.

H3) Sea-Surface / Surface Temperature Seasonal Cycle

Statement: SST/near-surface temperature follows a 1 cycle/year seasonal cycle.

Script: Analysis/SST_seasonal_cycle_H3.py

Method:

Detrend monthly temperature per city; FFT; identify dominant frequency & amplitude.

Optionally compute power at 1.0 cycles/yr and phase (month of peak).

Outputs (saved to results/SST_seasonal_cycle_H3/)

fft_peaks_by_city.csv — dominant frequency, power, phase per city

seasonal_fft_panels.png — log-power vs frequency with red line at 1.0 cycles/yr

seasonal_monthly_climatology.png — average annual cycle (12-month climatology)

Key config: detrend flag; frequency grid; which variable (SST or near-surface temp).

H4) Precipitation Seasonality (Monsoon vs Temperate)

Statement: Monsoon regions exhibit stronger precipitation seasonality than temperate ones.

Script: Analysis/precipitation_seasonality_H4.py

Method:

Compute climatological monthly means (12-month cycle) then a Seasonality Index
(e.g., (max − min) / mean or coefficient of variation).

Compare Monsoon group vs Temperate group (t-test or Mann–Whitney U).

Outputs (saved to results/precipitation_seasonality_H4/)

seasonality_metrics.csv — city-level SI, CV, max month, min month

seasonality_group_comparison.png — box + swarm

seasonality_radar_or_bars.png — 12-month climatology per city

Key config: group assignment for cities; SI definition.

H5) Monthly Mean Temperature ~ Normal

Statement: Monthly means tend toward a Normal distribution (CLT effect).

Script: Analysis/monthly_mean_temperature_normality_H5.py

Method:

For each city, take all monthly means (2000–2024).

Shapiro–Wilk and Q–Q plots; fit N(μ, σ) and overlay; report p-values.

Outputs (saved to results/monthly_mean_temperature_normality_H5/)

normality_tests.csv — Shapiro W, p-value, μ, σ per city

temp_hist_qq_panels.png — hist + Normal fit + Q–Q per city

Key config: outlier handling; whether to use anomalies or raw monthly means.

H6) Convective Feedback (Hotter → Drier?)

Statement: In some climates, higher temperature anomalies coincide with reduced precipitation.

Script: Analysis/Convective_feedback_H6.py

Method:

Build anomalies (remove monthly climatology) for temperature & precipitation.

Pearson/Spearman correlation per city; optional lagged correlations.

Global sign test/meta-analysis across cities.

Outputs (saved to results/Convective_feedback_H6/)

temp_precip_anomaly_correlations.csv — r, p per city (+ best lag)

scatter_with_regression.png — anomaly scatter + fit per city

lag_correlogram.png — correlation vs lag, temperature leading precipitation

Key config: anomaly method; lag window; correlation type.

H7) CO₂ — City Temperature Trend

Statement: Rising global CO₂ is correlated with long-term increases in city-level temperature.

Script: Analysis/CO2_city_trend_H7.py

Method:

Import monthly global CO₂ series (e.g., Mauna Loa) and city temperature_c.

Regress Temperature ~ Time + CO₂ (to separate pure time trend vs CO₂).

Report slope, p-values, and partial R² for CO₂.

Outputs (saved to results/CO2_city_trend_H7/)

co2_regression_by_city.csv — coefficients, p, R², diagnostics

co2_temp_trend_panels.png — time series with fitted trend per city

co2_vs_temp_scatter.png — mean-removed CO₂ vs temp with regression line

Key config: CO₂ file path; detrend options; per-city sensitivity plot.

H8) Point vs Grid Consistency

Statement: Temperature at station (point) agrees with grid cutout better than precipitation.

Script: Analysis/hpoint_vs_grid_consistency_H8.py
(If filename differs locally as point_vs_grid_consistency_H8.py, it’s the same script.)

Method:

For each city, align monthly point series vs grid cell (nearest or averaged 3×3).

Compute R², RMSE for temperature & precipitation; summarize by city and globally.

Outputs (saved to results/hpoint_vs_grid_consistency_H8/)

grid_point_metrics.csv — R² & RMSE by city/variable

grid_point_scatter_panels.png — point vs grid with 1:1 line

metrics_summary.csv — group means and deltas (Temp vs Precip)

Key config: netCDF paths; lon convention (−180–180 vs 0–360); neighborhood size.

4) How results are saved (naming conventions)

Each script writes both:

One or more figures: *.png — publication-ready (300 dpi)

One or more tables: *.csv — everything needed to reproduce the key numbers

All outputs go under results/<script_stem>/.
Examples:

results/analysis_coastal_inland_temp_range_H2/annual_temperature_ranges.csv

results/analysis_coastal_inland_temp_range_H2/temp_range_comparison.png

results/analysis_coastal_inland_temp_range_H2/temp_range_summary.csv

This is consistent for all H1–H8 scripts.

5) Reproducibility & adding cities

Drop a new city CSV in Data Extraction/outputs/ with the four standard columns.

Add the filename and metadata (label, group) at the top of any script (CITIES = {...}).

Re-run the script — new city appears automatically in figures and summaries.

Tip: To keep groups balanced (e.g., Coastal vs Inland), update group tags when adding cities.

6) Troubleshooting

“No data loaded” → Check DATA_DIR(S) in the script and actual CSV file names.

Longitude mismatch (grid files) → Use lon % 360 or add 360 to negative longitudes as already handled in H8.

Weird normality or variance warnings → The scripts automatically fall back to non-parametric tests where appropriate (e.g., Mann–Whitney U).

Overlapping labels on plots → Increase figure height or use the provided margin/annotation offsets (already done in H2).

7) Credits

Team: Abhishek & Nikhilesh

Data: Extracted with point_extract.py, gridded_extraction_only.py, additional_pt.py (SLP & RH merge).

Libraries: pandas, numpy, scipy, matplotlib, seaborn, xarray.

8) One-command batch run (optional)

Create a tiny runner to execute all hypotheses in sequence:

python Analysis/monsoon_trigger_H1.py && \
python Analysis/analysis_coastal_inland_temp_range_H2.py && \
python Analysis/SST_seasonal_cycle_H3.py && \
python Analysis/precipitation_seasonality_H4.py && \
python Analysis/monthly_mean_temperature_normality_H5.py && \
python Analysis/Convective_feedback_H6.py && \
python Analysis/CO2_city_trend_H7.py && \
python Analysis/hpoint_vs_grid_consistency_H8.py


This will refresh all results and figures under results/.

If you want, I can drop this into a Markdown file for you and tailor the city lists/group labels in each script section exactly to what you currently have checked into the repo.