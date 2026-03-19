# Figures

Plots produced by `visualize.py`:

- **time_series_ghg_vs_temp.png** — GHG concentrations and temperature anomaly.
- **feature_importance.png** — Feature importance by method.
- **scatter_top_features.png** — Top 3 features vs temperature.
- **model_predictions.png** — Model predictions vs actual.
- **r_feature_corr_top15.png** — R: feature correlation vs temperature anomaly (top 15 by absolute correlation).
- **r_residuals_by_model.png** — R: residual (actual − predicted) distributions by model on the test set.
- **r_predicted_vs_actual.png** — R: predicted vs actual temperature anomaly on the test set (faceted by model).

Run `python -m src.visualize` and `Rscript R/visualize_r.R` to regenerate all figures.
