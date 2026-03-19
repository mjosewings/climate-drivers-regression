#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# Read processed outputs from the project
features_path <- 'output/processed/climate_features.csv'
preds_path    <- 'output/processed/model_predictions.csv'
fig_dir <- 'output/figures'

if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

features <- read.csv(features_path, stringsAsFactors = FALSE)
preds <- read.csv(preds_path, stringsAsFactors = FALSE)

# ------------------------------
# Figure: Feature correlation vs target
# ------------------------------
# Feature columns: numeric columns excluding identifiers + target
id_cols <- c('date', 'year', 'month', 'region', 'temp_anomaly_C')
num_cols <- names(features)[sapply(features, is.numeric)]
feature_cols <- setdiff(num_cols, c('temp_anomaly_C'))
feature_cols <- setdiff(feature_cols, c('year', 'month'))

corr_list <- lapply(feature_cols, function(f) {
  x <- features[[f]]
  y <- features[['temp_anomaly_C']]
  c <- suppressWarnings(cor(x, y, use = 'complete.obs'))
  data.frame(feature = f, corr = c)
})

corr_df <- do.call(rbind, corr_list)

corr_df <- corr_df |>
  mutate(abs_corr = abs(corr)) |>
  arrange(desc(abs_corr)) |>
  slice_head(n = 15)

p1 <- ggplot(corr_df, aes(x = reorder(feature, corr), y = corr, fill = abs_corr)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_y_continuous(limits = c(-1, 1)) +
  labs(
    title = 'Correlation between engineered features and temperature anomaly',
    x = NULL,
    y = 'Pearson correlation (r)'
  ) +
  theme_minimal(base_size = 12)

out1 <- file.path(fig_dir, 'r_feature_corr_top15.png')
ggsave(out1, p1, width = 11, height = 7, dpi = 220)

# ------------------------------
# Figure: Residual distribution by model (test set)
# ------------------------------
# Identify prediction columns
pred_cols <- names(preds)[grepl('^pred_', names(preds))]

preds_test <- preds |> filter(split == 'test')

res_long <- do.call(rbind, lapply(pred_cols, function(pc) {
  model <- gsub('^pred_', '', pc)
  model <- gsub('_', ' ', model)
  model <- tools::toTitleCase(model)
  data.frame(
    model = model,
    residual = preds_test[['temp_anomaly_C']] - preds_test[[pc]]
  )
}))

p2 <- ggplot(res_long, aes(x = model, y = residual, fill = model)) +
  geom_boxplot(outlier.size = 0.3, alpha = 0.9, show.legend = FALSE) +
  labs(
    title = 'Residuals by model on the test set',
    x = NULL,
    y = 'Residual (actual - predicted) (°C)'
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

out2 <- file.path(fig_dir, 'r_residuals_by_model.png')
ggsave(out2, p2, width = 11, height = 7, dpi = 220)

# ------------------------------
# Figure: Predicted vs actual (faceted, test set)
# ------------------------------
pred_long <- do.call(rbind, lapply(pred_cols, function(pc) {
  model <- gsub('^pred_', '', pc)
  model <- gsub('_', ' ', model)
  model <- tools::toTitleCase(model)
  data.frame(
    model = model,
    actual = preds_test[['temp_anomaly_C']],
    predicted = preds_test[[pc]]
  )
}))

# Keep axis limits consistent across facets
lims <- range(c(pred_long$actual, pred_long$predicted), na.rm = TRUE)

p3 <- ggplot(pred_long, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.25, size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', color = 'grey40') +
  facet_wrap(~ model, nrow = 2) +
  coord_cartesian(xlim = lims, ylim = lims) +
  labs(
    title = 'Predicted vs actual temperature anomaly (test set)',
    x = 'Actual temperature anomaly (°C)',
    y = 'Predicted temperature anomaly (°C)'
  ) +
  theme_minimal(base_size = 12)

out3 <- file.path(fig_dir, 'r_predicted_vs_actual.png')
ggsave(out3, p3, width = 12, height = 8, dpi = 220)

cat('Saved figures to output/figures:\n')
cat(' -', out1, '\n')
cat(' -', out2, '\n')
cat(' -', out3, '\n')
