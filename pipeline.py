
# Import the necessary packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from pathlib import Path

# Choose random seed
np.random.seed(42)

# Output path (portable) - create ./outputs next to this script
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate the dataset
dates = pd.date_range(start="1960-01", end="2022-12", freq="MS")
N = len(dates)
t = np.arange(N)
year_frac = 1960 + t / 12

# NOAA GML
seasonal_co2 = 3.0 * np.sin(2 * np.pi * t/12 + 0.5)
co2 = 315 + 1.73*(t/12) + 0.002*(t/12)**2 + seasonal_co2 + np.random.normal(0, 0.3, N)

ch4_base = np.where(t < 288, 1580 + 2.5*(t/12 - 24),
           np.where(t < 408, 1700 + 0.3*(t/12 - 48),
                              1735 + 4.8*(t/12 - 54)))
ch4 = ch4_base + 20*np.sin(2*np.pi*t/12 + 1.2) + np.random.normal(0, 5, N)

n2o = 298 + 0.75*(t/12) + np.random.normal(0, 0.2, N)


# NASA GISS
solar = 1361 + 0.7*np.sin(2*np.pi*t/132) + 0.3*np.sin(2*np.pi*t/66) \
        + np.random.normal(0, 0.15, N)

volcanic_forcing = np.zeros(N)
volcanic_events  = {"Agung_1963": (36, -0.3, 24),
                    "El_Chichon_1982": (264, -0.4, 24),
                    "Pinatubo_1991": (372, -0.5, 36)}
for _, (idx, amp, dur) in volcanic_events.items():
    for i in range(min(dur, N - idx)):
        volcanic_forcing[idx + i] += amp * np.exp(-i / 12)

enso = 0.15*np.sin(2*np.pi*t/57 + 0.8) + 0.10*np.sin(2*np.pi*t/30 + 1.2)

temp_anomaly = (
    -0.35
    + 0.0012*(co2 - 315)**1.2
    + 0.00025*(ch4 - 1580)
    + 0.00012*(n2o - 298)
    + 0.0018*(solar - 1361)
    + volcanic_forcing
    + enso
    + np.random.normal(0, 0.05, N))

# OWID
co2_emissions = 9 + 0.45*(year_frac - 1960) + np.random.normal(0, 0.4, N)
land_use      = 3.5 + 0.012*(year_frac - 1960) + np.random.normal(0, 0.3, N)
aod           = np.clip(0.08 + 0.0004*(year_frac-1960) + np.abs(volcanic_forcing)*0.5
                        + np.random.normal(0, 0.005, N), 0, None)

# Assemble raw dataframe
df = pd.DataFrame({
    "date":             dates,
    "year":             dates.year,
    "month":            dates.month,
    "co2_ppm":          co2,
    "ch4_ppb":          ch4,
    "n2o_ppb":          n2o,
    "solar_W_m2":       solar,
    "temp_anomaly_C":   temp_anomaly,
    "co2_emissions_Gt": co2_emissions,
    "land_use_Gt":      land_use,
    "aerosol_AOD":      aod,
    "volcanic_flag":    0,
})
for _, (idx, amp, dur) in volcanic_events.items():
    df.loc[df.index[idx:idx+dur], "volcanic_flag"] = 1


# Feature Engineering
df = df.sort_values("date").reset_index(drop=True)
t_eng = np.arange(len(df))

df["months_since_1960"] = t_eng
df["co2_growth"]        = df["co2_ppm"].diff().fillna(0)
df["ch4_growth"]        = df["ch4_ppb"].diff().fillna(0)
df["n2o_growth"]        = df["n2o_ppb"].diff().fillna(0)

for col in ["co2_ppm", "ch4_ppb", "solar_W_m2", "aerosol_AOD"]:
    df[f"{col}_ma12"] = df[col].rolling(12, min_periods=1).mean()

df["cum_aerosol"]  = df["aerosol_AOD"].cumsum() / 100
df["enso_proxy"]   = 0.15*np.sin(2*np.pi*t_eng/57 + 0.8)
df["co2_x_solar"]  = (df["co2_ppm"] - 315) * (df["solar_W_m2"] - 1361)
df["month_sin"]    = np.sin(2*np.pi*df["month"]/12)
df["month_cos"]    = np.cos(2*np.pi*df["month"]/12)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Dataset shape after feature engineering: {df.shape}")

# Modeling
FEATURES = [
    "co2_ppm", "ch4_ppb", "n2o_ppb",
    "solar_W_m2", "aerosol_AOD",
    "co2_emissions_Gt", "land_use_Gt", "volcanic_flag",
    "months_since_1960",
    "co2_growth", "ch4_growth", "n2o_growth",
    "co2_ppm_ma12", "ch4_ppb_ma12", "solar_W_m2_ma12", "aerosol_AOD_ma12",
    "cum_aerosol", "enso_proxy", "co2_x_solar",
    "month_sin", "month_cos",
]
TARGET = "temp_anomaly_C"

X = df[FEATURES].values
y = df[TARGET].values

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.30, random_state=42, shuffle=False
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
}

fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    r2  = r2_score(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"  {name:<25s}  R²={r2:.4f}  RMSE={rmse:.4f}")

# Feature importance
lr_coefs = np.abs(fitted["Linear Regression"].coef_)
rf_imp   = fitted["Random Forest"].feature_importances_
gb_imp   = fitted["Gradient Boosting"].feature_importances_
perm     = permutation_importance(fitted["Gradient Boosting"], X_test, y_test,
                                   n_repeats=10, random_state=42)

importance_df = pd.DataFrame({
    "feature":              FEATURES,
    "linear_std_coef":      lr_coefs,
    "random_forest":        rf_imp,
    "gradient_boosting":    gb_imp,
    "permutation":          perm.importances_mean,
    "permutation_std":      perm.importances_std,
    # Factor category labels for coloring in R/Julia/MATLAB
    "category": [
        "GHG","GHG","GHG",             # co2, ch4, n2o ppm
        "Solar","Natural",              # solar, aerosol
        "GHG","GHG","Natural",          # emissions, land_use, volcanic
        "Temporal",                     # months_since_1960
        "GHG","GHG","GHG",             # growth rates
        "GHG","GHG","Solar","Natural",  # MAs
        "Natural","Natural","GHG",      # cum_aerosol, enso, co2xsolar
        "Seasonal","Seasonal",          # month_sin, month_cos
    ]
})
importance_df.sort_values("permutation", ascending=False, inplace=True)

# Model Predictions
pred_df = df[["date","year","month","temp_anomaly_C",
              "co2_ppm","ch4_ppb","solar_W_m2","aerosol_AOD",
              "enso_proxy","volcanic_flag"]].copy()

pred_df["split"] = "train"
pred_df.loc[pred_df.index[-len(y_test):], "split"] = "test"

for name, model in fitted.items():
    col = name.lower().replace(" ", "_")
    pred_df[f"pred_{col}"] = model.predict(X_sc)

# Export CSVs
df.to_csv(OUT_DIR / "climate_features.csv", index=False)
importance_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
pred_df.to_csv(OUT_DIR / "model_predictions.csv", index=False)

print("\n✓ Exported:")
print(f"  climate_features.csv     → {df.shape[0]} rows × {df.shape[1]} cols")
print(f"  feature_importance.csv   → {importance_df.shape[0]} features × 6 importance columns")
print(f"  model_predictions.csv    → {pred_df.shape[0]} rows, 4 model prediction columns")
print(f"\nAll files saved to: {OUT_DIR}")