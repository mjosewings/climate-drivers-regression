# Methodology

This document outlines the pipeline and shows the main result figures produced by the project.

## Pipeline overview

1. **Data collection** (`src/data_collection.py`) — Fetches monthly data from NOAA GML (CO₂, CH₄, N₂O), NASA GISS (temperature anomaly for Global, NH, SH), an NCEI-consistent solar series, and OWID (emissions). Merges on year/month and writes `output/raw/raw_merged.csv` (≥1000 rows).
2. **Pipeline** (`src/pipeline.py`) — Loads raw data, adds volcanic/aerosol proxies, builds features (growth rates, 12‑month moving averages, etc.), fits linear, ridge, random forest, and gradient boosting models, and exports feature matrix, importance table, and predictions to `output/processed/`.
3. **Visualization** (`src/visualize.py`) — Reads the processed CSVs and generates the figures below, saved in `output/figures/`.

![End-to-end pipeline and data model](docs/methodology_diagram.png)

**Figure 1.** High-level pipeline and data model used in the project, showing how raw sources, engineered features, models, and outputs relate to each other.

---

## Result figures

### 1. Greenhouse gases and temperature anomaly

![Greenhouse gases and temperature over time](output/figures/time_series_ghg_vs_temp.png)

**Figure 2.** Time series of atmospheric CO₂ (ppm), CH₄ (ppb), and N₂O (ppb) with global temperature anomaly (°C). Temperature is shown as monthly values and as a 12‑month moving average. Used to compare long‑term GHG trends with warming.

---

### 2. Feature importance by method

![Feature importance across four ranking methods](output/figures/feature_importance.png)

**Figure 3.** Feature importance from four methods: standardized absolute linear coefficients, random forest impurity, gradient boosting impurity, and permutation importance. Bars are grouped by factor type (GHG, solar, natural, temporal, seasonal). Supports comparison of which drivers matter for temperature in each model.

---

### 3. Top three features vs temperature

![Scatter plots of top three features vs temperature anomaly](output/figures/scatter_top_features.png)

**Figure 4.** Scatter plots of the three highest-ranked features (by permutation importance) against temperature anomaly, with linear fit and Pearson *r*. Shows strength and sign of the relationship for the main drivers.

---

### 4. Model predictions vs actual temperature

![Actual vs predicted temperature for all models](output/figures/model_predictions.png)

**Figure 5.** Actual global temperature anomaly (black) and predictions from the four fitted models over time. The vertical dashed line marks the start of the test set (70/30 split). Used to compare model fit and generalization.

---

### 5. Additional diagnostics (R)

![Correlation of engineered features with temperature anomaly](output/figures/r_feature_corr_top15.png)

**Figure 6.** Pearson correlation between engineered features and temperature anomaly (top 15 by absolute correlation). This helps summarize which variables move together with the target in the integrated dataset.

---

![Residuals by model on the test set](output/figures/r_residuals_by_model.png)

**Figure 7.** Residual (actual − predicted) distributions for each regression model on the test set. This highlights whether models systematically over- or under-predict.

---

![Predicted vs actual temperature anomaly (R)](output/figures/r_predicted_vs_actual.png)

**Figure 8.** Predicted vs actual temperature anomaly on the test set for each model, with a 1:1 reference line. Points close to the line indicate stronger predictive accuracy.

---

![Binned response curves for top drivers](output/figures/extra_r_response_curves_top_drivers.png)

**Figure 9.** Binned response curves for the top-ranked driver features (selected by permutation importance) versus temperature anomaly. The curves summarize how the average temperature anomaly changes across the driver’s value range, and the shaded band indicates uncertainty (standard error) around each bin.

---

To regenerate the figures, run:

1. `python -m src.visualize` (after running the pipeline) for the main Python plots.
2. `Rscript R/visualize_r.R` for the R diagnostics figures.
3. `Rscript scripts/extra_visualizations/r/plot_response_curves_top_drivers.R` for the additional binned response curves (Figure 9).

Optional multi-language scripts:
- Julia: `julia scripts/extra_visualizations/julia/plot_residuals_test_models.jl` (writes `output/figures/extra_julia_test_residuals.png`)
- MATLAB: run `scripts/extra_visualizations/matlab/plot_lagged_correlation_top_drivers.m` in MATLAB (writes `output/figures/extra_matlab_lagged_corr_top_drivers.png`)
