# Figures

Plots produced by `visualize.py`:

- **time_series_ghg_vs_temp.png** — GHG concentrations and temperature anomaly.
- **feature_importance.png** — Feature importance by method.
- **scatter_top_features.png** — Top 3 features vs temperature.
- **model_predictions.png** — Model predictions vs actual.
- **r_feature_corr_top15.png** — R: feature correlation vs temperature anomaly (top 15 by absolute correlation).
- **r_residuals_by_model.png** — R: residual (actual − predicted) distributions by model on the test set.
- **r_predicted_vs_actual.png** — R: predicted vs actual temperature anomaly on the test set (faceted by model).
- **extra_r_response_curves_top_drivers.png** — R: binned response curves for the top-ranked driver features.

Run `python -m src.visualize`, `Rscript scripts/multi_language_visualizations/r/visualize_r.R`, and
`Rscript scripts/multi_language_visualizations/r/plot_response_curves_top_drivers.R` to regenerate the full set.

Optional multi-language extras:
- Julia: `julia scripts/multi_language_visualizations/julia/plot_residuals_test_models.jl` (writes `output/figures/extra_julia_test_residuals.png`)
- MATLAB: run `scripts/multi_language_visualizations/matlab/plot_lagged_correlation_top_drivers.m` in MATLAB (writes `output/figures/extra_matlab_lagged_corr_top_drivers.png`)
