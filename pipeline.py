"""
pipeline.py – End-to-end climate-drivers regression pipeline.

This script:
  1. Generates a realistic multi-source climate dataset (monthly, 1960-2022)
     modelled after NOAA GML, NASA GISS, and Our World in Data sources.
  2. Engineers derived features (growth rates, moving averages, interaction
     terms, seasonal encodings, volcanic flags, etc.).
  3. Trains four regression models to predict global temperature anomaly.
  4. Computes feature-importance scores using four complementary methods.
  5. Exports three CSV artefacts to ./outputs/ for downstream analysis and
     visualisation.

Usage
-----
    python pipeline.py
"""

# ── imports ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Major volcanic eruptions used to inject aerosol / cooling signals
VOLCANIC_EVENTS = {
    "Agung_1963":      (36,  -0.3, 24),
    "El_Chichon_1982": (264, -0.4, 24),
    "Pinatubo_1991":   (372, -0.5, 36),
}

# Model feature list (all engineered columns fed to the regressors)
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

# Category labels for each feature (used in visualisation)
FEATURE_CATEGORIES = [
    "GHG", "GHG", "GHG",               # co2, ch4, n2o concentrations
    "Solar", "Natural",                 # solar irradiance, aerosol AOD
    "GHG", "GHG", "Natural",            # emissions, land-use, volcanic flag
    "Temporal",                         # months_since_1960
    "GHG", "GHG", "GHG",               # growth rates
    "GHG", "GHG", "Solar", "Natural",   # 12-month moving averages
    "Natural", "Natural", "GHG",        # cum_aerosol, enso_proxy, co2×solar
    "Seasonal", "Seasonal",             # month_sin, month_cos
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data generation (simulating multi-source collection)
# ═══════════════════════════════════════════════════════════════════════════
def generate_raw_data() -> pd.DataFrame:
    """Synthesise a realistic monthly climate dataset (1960-01 to 2022-12).

    Returns a DataFrame with columns modelled after three real-world sources:
      - NOAA GML  → CO₂, CH₄, N₂O concentrations
      - NASA GISS → solar irradiance, volcanic forcing, ENSO proxy,
                     temperature anomaly
      - OWID      → CO₂ emissions, land-use emissions, aerosol optical depth
    """
    dates = pd.date_range(start="1960-01", end="2022-12", freq="MS")
    N = len(dates)
    t = np.arange(N)
    year_frac = 1960 + t / 12

    # -- NOAA GML: greenhouse-gas concentrations -------------------------
    seasonal_co2 = 3.0 * np.sin(2 * np.pi * t / 12 + 0.5)
    co2 = (315 + 1.73 * (t / 12) + 0.002 * (t / 12) ** 2
           + seasonal_co2 + np.random.normal(0, 0.3, N))

    ch4_base = np.where(
        t < 288, 1580 + 2.5 * (t / 12 - 24),
        np.where(t < 408, 1700 + 0.3 * (t / 12 - 48),
                 1735 + 4.8 * (t / 12 - 54)),
    )
    ch4 = ch4_base + 20 * np.sin(2 * np.pi * t / 12 + 1.2) + np.random.normal(0, 5, N)

    n2o = 298 + 0.75 * (t / 12) + np.random.normal(0, 0.2, N)

    # -- NASA GISS: solar, volcanic, ENSO, temperature anomaly -----------
    solar = (1361
             + 0.7 * np.sin(2 * np.pi * t / 132)
             + 0.3 * np.sin(2 * np.pi * t / 66)
             + np.random.normal(0, 0.15, N))

    volcanic_forcing = np.zeros(N)
    for _, (idx, amp, dur) in VOLCANIC_EVENTS.items():
        for i in range(min(dur, N - idx)):
            volcanic_forcing[idx + i] += amp * np.exp(-i / 12)

    enso = (0.15 * np.sin(2 * np.pi * t / 57 + 0.8)
            + 0.10 * np.sin(2 * np.pi * t / 30 + 1.2))

    # Target variable – temperature anomaly driven by multiple factors
    co2_delta = np.maximum(co2 - 315, 0)  # clip to avoid NaN from neg ** 1.2
    temp_anomaly = (
        -0.35
        + 0.0012 * co2_delta ** 1.2
        + 0.00025 * (ch4 - 1580)
        + 0.00012 * (n2o - 298)
        + 0.0018 * (solar - 1361)
        + volcanic_forcing
        + enso
        + np.random.normal(0, 0.05, N)
    )

    # -- OWID: anthropogenic indicators ----------------------------------
    co2_emissions = 9 + 0.45 * (year_frac - 1960) + np.random.normal(0, 0.4, N)
    land_use = 3.5 + 0.012 * (year_frac - 1960) + np.random.normal(0, 0.3, N)
    aod = np.clip(
        0.08 + 0.0004 * (year_frac - 1960)
        + np.abs(volcanic_forcing) * 0.5
        + np.random.normal(0, 0.005, N),
        0, None,
    )

    # -- Assemble raw DataFrame ------------------------------------------
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
    for _, (idx, _, dur) in VOLCANIC_EVENTS.items():
        df.loc[df.index[idx:idx + dur], "volcanic_flag"] = 1

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature engineering
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive climate-driver features from the raw merged dataset.

    Adds growth rates, 12-month moving averages, cumulative aerosol load,
    ENSO proxy, a CO₂×solar interaction term, and cyclical month encodings.
    """
    df = df.sort_values("date").reset_index(drop=True)
    t = np.arange(len(df))

    # Time index
    df["months_since_1960"] = t

    # Month-over-month growth rates
    df["co2_growth"] = df["co2_ppm"].diff().fillna(0)
    df["ch4_growth"] = df["ch4_ppb"].diff().fillna(0)
    df["n2o_growth"] = df["n2o_ppb"].diff().fillna(0)

    # 12-month rolling averages (smooth seasonal noise)
    for col in ["co2_ppm", "ch4_ppb", "solar_W_m2", "aerosol_AOD"]:
        df[f"{col}_ma12"] = df[col].rolling(12, min_periods=1).mean()

    # Cumulative aerosol burden (proxy for long-term volcanic dimming)
    df["cum_aerosol"] = df["aerosol_AOD"].cumsum() / 100

    # ENSO proxy (quasi-periodic oscillation)
    df["enso_proxy"] = 0.15 * np.sin(2 * np.pi * t / 57 + 0.8)

    # Interaction: CO₂ deviation × solar deviation
    df["co2_x_solar"] = (df["co2_ppm"] - 315) * (df["solar_W_m2"] - 1361)

    # Cyclical month encoding (preserves continuity Dec → Jan)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Model training & evaluation
# ═══════════════════════════════════════════════════════════════════════════
def train_models(X_train, X_test, y_train, y_test):
    """Train four regression models and print evaluation metrics.

    Returns a dict mapping model name → fitted estimator.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_SEED,
        ),
    }

    fitted = {}
    print("\nModel evaluation (test set):")
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  {name:<25s}  R²={r2:.4f}  RMSE={rmse:.4f}")

    return fitted


# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature importance
# ═══════════════════════════════════════════════════════════════════════════
def compute_feature_importance(fitted, X_test, y_test) -> pd.DataFrame:
    """Compute importance scores from four complementary methods:
       standardised |coefficients|, RF impurity, GB impurity, and
       permutation importance.
    """
    lr_coefs = np.abs(fitted["Linear Regression"].coef_)
    rf_imp   = fitted["Random Forest"].feature_importances_
    gb_imp   = fitted["Gradient Boosting"].feature_importances_
    perm     = permutation_importance(
        fitted["Gradient Boosting"], X_test, y_test,
        n_repeats=10, random_state=RANDOM_SEED,
    )

    importance_df = pd.DataFrame({
        "feature":           FEATURES,
        "linear_std_coef":   lr_coefs,
        "random_forest":     rf_imp,
        "gradient_boosting": gb_imp,
        "permutation":       perm.importances_mean,
        "permutation_std":   perm.importances_std,
        "category":          FEATURE_CATEGORIES,
    })
    importance_df.sort_values("permutation", ascending=False, inplace=True)
    return importance_df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Prediction table
# ═══════════════════════════════════════════════════════════════════════════
def build_prediction_table(df, fitted, X_sc, y_test) -> pd.DataFrame:
    """Create a DataFrame with actual values, train/test labels, and
    predictions from every fitted model."""
    pred_df = df[[
        "date", "year", "month", "temp_anomaly_C",
        "co2_ppm", "ch4_ppb", "solar_W_m2", "aerosol_AOD",
        "enso_proxy", "volcanic_flag",
    ]].copy()

    pred_df["split"] = "train"
    pred_df.loc[pred_df.index[-len(y_test):], "split"] = "test"

    for name, model in fitted.items():
        col = name.lower().replace(" ", "_")
        pred_df[f"pred_{col}"] = model.predict(X_sc)

    return pred_df


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # Step 1 – generate (or load) raw data
    print("Generating multi-source climate dataset …")
    df = generate_raw_data()

    # Step 2 – feature engineering
    df = engineer_features(df)
    print(f"Dataset shape after feature engineering: {df.shape}")

    # Step 3 – prepare modelling matrices
    X = df[FEATURES].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.30, random_state=RANDOM_SEED,
    )

    # Step 4 – train models
    fitted = train_models(X_train, X_test, y_train, y_test)

    # Step 5 – feature importance
    importance_df = compute_feature_importance(fitted, X_test, y_test)

    # Step 6 – prediction table
    pred_df = build_prediction_table(df, fitted, X_sc, y_test)

    # Step 7 – export
    df.to_csv(OUT_DIR / "climate_features.csv", index=False)
    importance_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    pred_df.to_csv(OUT_DIR / "model_predictions.csv", index=False)

    print("\n✓ Exported:")
    print(f"  climate_features.csv     → {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  feature_importance.csv   → {importance_df.shape[0]} features × "
          f"{importance_df.shape[1]} columns")
    print(f"  model_predictions.csv    → {pred_df.shape[0]} rows, "
          f"{sum(c.startswith('pred_') for c in pred_df.columns)} model columns")
    print(f"\nAll files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
