# H1: Monsoon trigger (high temp + low SLP → convective rain)
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_DIRS = ["../Data Extraction/outputs","../data extraction/outputs","./data","./"]
OUT_DIR = "../results/H1"; os.makedirs(OUT_DIR, exist_ok=True)

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
SLP_CANDS    = ["sea_level_pressure_hpa","slp_hpa","sea_level_pressure","msl","slp"]

def find(fname):
    for d in DATA_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p): return p
    return None

def pick(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def monthly_anom(s, idx):
    clim = s.groupby(idx.month).transform("mean")
    return s - clim

rows, skipped = [], []

for city, fname in FILES.items():
    p = find(fname)
    if not p: 
        skipped.append((city,"file not found")); 
        continue

    df = pd.read_csv(p, parse_dates=[0], index_col=0).sort_index()
    tcol = pick(df, TEMP_CANDS)
    pcol = pick(df, PRECIP_CANDS)
    scol = pick(df, SLP_CANDS)
    if not all([tcol,pcol,scol]):
        skipped.append((city,"missing required columns")); 
        continue

    df = df[[tcol,pcol,scol]].resample("MS").mean()

    df["t_anom"]   = monthly_anom(df[tcol], df.index)
    df["slp_anom"] = monthly_anom(df[scol], df.index)
    thr = np.nanpercentile(df[pcol], 75)
    df["event"] = (df[pcol] > thr).astype(int)

    X = df[["t_anom","slp_anom"]].dropna()
    y = df.loc[X.index,"event"]
    if len(np.unique(y)) < 2:
        skipped.append((city,"no variance in event")); 
        continue

    X["neg_slp_anom"] = -X["slp_anom"]
    X = X[["t_anom","neg_slp_anom"]]

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    try:
        clf = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=200)
        clf.fit(Xz, y)
        proba = clf.predict_proba(Xz)[:,1]
        auc   = roc_auc_score(y, proba)

        # OR per 1 std increase (because features are z-scored)
        or_temp = float(np.exp(clf.coef_[0,0]))
        or_slp  = float(np.exp(clf.coef_[0,1]))

        rows.append({
            "City": city,
            "OR_temp": or_temp,
            "OR_slp":  or_slp,
            "Coef_temp": float(clf.coef_[0,0]),
            "Coef_negSLP": float(clf.coef_[0,1]),
            "AUC": float(auc),
            "n": int(len(y))
        })
    except Exception as e:
        skipped.append((city, f"fit error: {e}"))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "H1_monsoon_trigger_summary.csv"), index=False)

if not res.empty:
    plt.figure(figsize=(7.5,4.2), dpi=200)
    m = res.melt(id_vars="City", value_vars=["OR_temp","OR_slp"], 
                 var_name="Predictor", value_name="OR")
    sns.pointplot(data=m, x="OR", y="City", hue="Predictor", join=False,
                  palette={"OR_temp":"tab:red","OR_slp":"tab:blue"})
    plt.axvline(1, ls="--", c="k", lw=1)
    plt.xscale("log"); plt.xlabel("Odds ratio (log scale)")
    plt.title("H1: Monsoon trigger — odds ratios (Temp↑, SLP↓)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "H1_odds_ratio_forest.png"), dpi=300)

print("\n[RESULTS]")
print(res if not res.empty else "No valid fits; see skipped list.")
if skipped:
    print("\n[SKIPPED]")
    for c, why in skipped: 
        print(f" - {c}: {why}")
print("\nSaved to:", OUT_DIR)
