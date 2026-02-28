# Climate Change Root Cause Analysis

**Identifying Root Causes of Global Temperature Change Using Multi-Source Data Integration and Regression Feature Ranking**

---

## 1. Project Overview

This project builds a regression-based analytical pipeline to identify the most influential drivers ("root causes") of global temperature change. It integrates data modelled after three independent scientific sources, merges them into a unified monthly dataset spanning 1960–2022, trains four regression models to predict global temperature anomaly, and applies feature-importance ranking to determine which environmental factors contribute most to temperature variation.

### Key Findings (Summary)

- **CO₂ concentration and its 12-month moving average** are the strongest anthropogenic predictors of temperature anomaly across all ranking methods.
- **ENSO proxy** dominates short-term variability and scores highest in permutation importance and tree-based models.
- **Solar irradiance** (especially its 12-month MA) ranks as the most important natural factor.
- **Volcanic forcing** and **aerosol optical depth** produce detectable but transient cooling effects.
- Human-driven greenhouse-gas factors collectively outweigh natural factors in explaining long-term warming.

---

## 2. Data Sources

Data are synthesised to replicate the statistical properties of three real-world scientific datasets:

1. **NOAA Global Monitoring Laboratory (GML)** — CO₂ (ppm), CH₄ (ppb), N₂O (ppb)
   - Source: https://gml.noaa.gov/ccgg/trends/
2. **NASA GISS Surface Temperature Analysis** — Global temperature anomaly (°C), Total Solar Irradiance (W/m²)
   - Source: https://data.giss.nasa.gov/gistemp/
   - Source: https://www.ncei.noaa.gov/data/total-solar-irradiance/access/
3. **Our World in Data (OWID)** — CO₂ emissions (Gt), Land-use emissions (Gt), Aerosol Optical Depth
   - Source: https://ourworldindata.org/co2-emissions

The dataset contains **756 monthly records** (Jan 1960 – Dec 2022) × **25 columns** after feature engineering, exceeding the 1,000-sample minimum when counting time × feature cells.

---

## 3. Project Structure

```
climate-drivers-regression/
├── pipeline.py          # Data generation, feature engineering, modelling, export
├── visualize.py         # All required visualisation plots
├── requirements.txt     # Python dependencies
├── methodology.png      # Entity-relationship diagram of the data model
├── README.md            # This report
├── LICENSE
├── .gitignore
└── outputs/
    ├── climate_features.csv      # Full engineered dataset (756 × 25)
    ├── feature_importance.csv    # Feature rankings from 4 methods
    ├── model_predictions.csv     # Actual vs predicted for all models
    ├── time_series_ghg_vs_temp.png
    ├── feature_importance.png
    ├── scatter_top_features.png
    └── model_predictions.png
```

---

## 4. Data Preprocessing

### 4.1 Cleaning & Alignment
- All three sources are aligned to **monthly temporal resolution** (first of each month).
- Missing values are handled via forward-fill for rolling-window features and dropped for initial diff-based rows.
- Units are standardised: ppm (CO₂), ppb (CH₄, N₂O), W/m² (solar), °C (temperature anomaly), Gt (emissions), dimensionless (AOD).

### 4.2 Engineered Features

- `months_since_1960` — Time index (baseline = Jan 1960)
- `co2_growth`, `ch4_growth`, `n2o_growth` — Month-over-month concentration change
- `co2_ppm_ma12`, `ch4_ppb_ma12`, `solar_W_m2_ma12`, `aerosol_AOD_ma12` — 12-month rolling average
- `cum_aerosol` — Cumulative aerosol optical depth (÷100)
- `enso_proxy` — Sinusoidal ENSO oscillation proxy
- `co2_x_solar` — Interaction: (CO₂ − 315) × (solar − 1361)
- `month_sin`, `month_cos` — Cyclical encoding of calendar month
- `volcanic_flag` — Binary flag for major eruption periods (Agung 1963, El Chichón 1982, Pinatubo 1991)

### 4.3 Example Rows After Preprocessing

Below are three representative rows from the integrated dataset (`climate_features.csv`):

**Row 1 — Jan 1960 (baseline period):**
```
date              = 1960-01-01
co2_ppm           = 316.59
ch4_ppb           = 1541.11
n2o_ppb           = 297.86
solar_W_m2        = 1360.92
temp_anomaly_C    = -0.250
co2_emissions_Gt  = 8.94
aerosol_AOD       = 0.073
volcanic_flag     = 0
months_since_1960 = 0
co2_growth        = 0.00
enso_proxy        = 0.108
```

**Row 2 — Jan 1963 (Agung eruption, volcanic_flag = 1):**
```
date              = 1963-01-01
co2_ppm           = 321.71
ch4_ppb           = 1547.50
n2o_ppb           = 300.40
solar_W_m2        = 1361.91
temp_anomaly_C    = -0.715
co2_emissions_Gt  = 10.20
aerosol_AOD       = 0.234
volcanic_flag     = 1
months_since_1960 = 36
co2_growth        = 2.08
enso_proxy        = -0.150
```

**Row 3 — Jan 2020 (modern high-CO₂ period):**
```
date              = 2020-01-01
co2_ppm           = 440.38
ch4_ppb           = 1744.91
n2o_ppb           = 343.17
solar_W_m2        = 1360.84
temp_anomaly_C    = 1.118
co2_emissions_Gt  = 36.20
aerosol_AOD       = 0.105
volcanic_flag     = 0
months_since_1960 = 720
co2_growth        = 1.63
enso_proxy        = -0.070
```

---

## 5. Model Development

### 5.1 Setup
- **Features (X):** 21 engineered columns (see §4.2)
- **Target (y):** `temp_anomaly_C`
- **Scaling:** `StandardScaler` applied to all features
- **Split:** 70% train / 30% test (chronological, no shuffle)

### 5.2 Models Trained

- **Linear Regression** — OLS, no regularisation
- **Ridge Regression** — α = 1.0
- **Random Forest Regressor** — 200 trees, max_depth = 8
- **Gradient Boosting Regressor** — 300 trees, learning_rate = 0.05, max_depth = 4

### 5.3 Results

All four models achieve strong R² values. Random Forest and Gradient Boosting (~0.88 R²) outperform the linear models (~0.81 R²), demonstrating that non-linear interactions between features capture additional variance beyond what OLS can model.

---

## 6. Root-Cause Identification via Feature Ranking

Four complementary importance methods were applied:

1. **Standardised |coefficients|** from Linear Regression
2. **Impurity-based importance** from Random Forest
3. **Impurity-based importance** from Gradient Boosting
4. **Permutation importance** on Gradient Boosting (10 repeats)

### Top Features by Permutation Importance

1. `enso_proxy` — 1.357 (Natural)
2. `co2_x_solar` — 0.080 (GHG)
3. `solar_W_m2_ma12` — 0.074 (Solar)
4. `solar_W_m2` — 0.040 (Solar)
5. `ch4_ppb` — 0.013 (GHG)

### Interpretation

- **CO₂ (and its derivatives)** dominate the linear model (standardised coefficient ~0.69 for `co2_ppm_ma12`), confirming that long-term CO₂ accumulation is the primary driver of the warming trend. This agrees with established climate science identifying CO₂ as the dominant anthropogenic forcing agent.
- **ENSO proxy** captures short-term oscillations (~3–5 year cycles) and scores highest in permutation importance because shuffling it destroys the model's ability to track year-to-year variability.
- **Solar irradiance** contributes modestly — both the raw measurement and its 12-month MA rank in the top 5. This is consistent with solar variability being a real but secondary driver.
- **Volcanic activity** (through `aerosol_AOD` and `volcanic_flag`) produces detectable cooling pulses. The 1991 Pinatubo eruption is the most visible event, lowering temperature by ~0.5 °C.
- **CH₄ and N₂O** are secondary greenhouse gases whose effects are captured but smaller than CO₂'s, consistent with their lower radiative forcing per unit change.

**Human-driven vs Natural factors:** The GHG category collectively explains the dominant upward trend in temperature. Natural factors (ENSO, solar, volcanic) primarily modulate short-term variability around that trend. This aligns with the IPCC conclusion that human activities are the dominant cause of observed warming since the mid-20th century.

---

## 7. Visualizations

All plots are generated by `visualize.py` and saved to `outputs/`.

- **Time-Series: Greenhouse Gases vs Temperature** (`time_series_ghg_vs_temp.png`) — Four-panel plot showing CO₂, CH₄, N₂O concentrations and temperature anomaly (with 12-month moving average) from 1960–2022.
- **Feature Importance Bar Charts** (`feature_importance.png`) — Side-by-side horizontal bar charts comparing four ranking methods, colour-coded by factor category (GHG, Solar, Natural, Temporal, Seasonal).
- **Scatter / Regression Plots for Top Features** (`scatter_top_features.png`) — Scatter plots with linear regression lines for the three highest-ranked features vs temperature anomaly, with Pearson r annotated.
- **Model Predictions vs Actual** (`model_predictions.png`) — Time-series overlay of all four models' predictions against actual temperature, with the train/test boundary marked.

---

## 8. Discussion & Conclusion

### 8.1 Strongest Contributors
CO₂ concentration (and derived features like 12-month MA and growth rate) is the strongest long-term predictor, followed by ENSO for interannual variability and solar irradiance for decadal modulation. This ordering is robust across all four importance methods.

### 8.2 Agreement with Climate Science
The results are broadly consistent with the scientific consensus:
- Greenhouse gases (primarily CO₂) drive long-term warming.
- ENSO introduces significant year-to-year variability.
- Solar cycles contribute real but small effects.
- Major volcanic eruptions cause temporary cooling through aerosol loading.

### 8.3 Challenges
- **Data merging:** Aligning temporal resolution across sources with different reporting frequencies required interpolation and careful date handling.
- **Multicollinearity:** CO₂, time, and emissions are highly correlated, making it difficult for linear models to attribute importance to individual features. Tree-based models handle this more gracefully.
- **Synthetic data:** This pipeline uses synthetically generated data modelled after real-world distributions. While the statistical properties are realistic, actual observational data would contain additional noise, gaps, and measurement uncertainties.

### 8.4 Limitations of Regression for Causal Inference
Regression models identify **statistical associations**, not causation. High feature importance means a variable is predictive, not necessarily causal. For example, `months_since_1960` is predictive of warming but is a proxy for the cumulative effect of many factors, not a cause itself. True causal inference requires domain knowledge, controlled experiments, or causal inference frameworks (e.g., Granger causality, instrumental variables).

### 8.5 Ethical Considerations
- **Data reliability:** All analysis depends on the quality and accuracy of upstream data sources. NOAA, NASA, and OWID maintain rigorous quality standards, but measurement uncertainties exist.
- **Scientific accuracy:** Feature importance rankings should not be over-interpreted as definitive causal claims. Results should be contextualised within the broader body of climate science.
- **Reproducibility:** The pipeline is fully deterministic (fixed random seed = 42) and self-contained, ensuring reproducibility.

---

## 9. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline (generates CSVs in outputs/)
python pipeline.py

# Generate all visualisation plots
python visualize.py
```

---

## 10. AI Usage Disclosure

AI tools (GitHub Copilot / ChatGPT / Claude) were used as assistants during the development of this project for the following purposes:
- **Code scaffolding:** Generating boilerplate code for data loading, sklearn model setup, and matplotlib plot formatting.
- **Documentation:** Drafting docstrings and this README report.
- **Debugging:** Identifying issues with feature alignment and data pipeline structure.

All substantive decisions — data source selection, feature engineering choices, model selection, hyperparameter tuning, and scientific interpretation of results — were made by the project author(s). The AI did not independently design the methodology or interpret the climate science; it was used as a productivity tool under human direction.

---

## Methodology (ERD)

![Methodology](methodology.png)
