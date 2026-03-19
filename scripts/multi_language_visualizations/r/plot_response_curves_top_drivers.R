# Response curves for top drivers (R)
#
# This script:
#   1) Reads `output/processed/feature_importance.csv`
#   2) Selects the top drivers by permutation importance
#   3) Filters `output/processed/climate_features.csv` to region == "Global"
#   4) Creates binned response curves (mean temperature vs driver value)
#   5) Writes a publication-style PNG to `output/figures/`

library(readr)
library(dplyr)
library(ggplot2)

input_importance <- "output/processed/feature_importance.csv"
input_features  <- "output/processed/climate_features.csv"
output_png       <- "output/figures/extra_r_response_curves_top_drivers.png"
output_csv       <- "output/figures/extra_r_top_drivers.csv"

top_n <- 3
bins  <- 25

# Optional: skip purely temporal/seasonal encoding features
exclude_features <- c("months_since_1960", "month_sin", "month_cos")

feature_importance <- read_csv(input_importance, show_col_types = FALSE)
features_df <- read_csv(input_features, show_col_types = FALSE)

feature_importance <- feature_importance %>%
  mutate(permutation = as.numeric(permutation)) %>%
  filter(!is.na(permutation))

pick_top_drivers <- function(df) {
  ordered <- df %>% arrange(desc(permutation))
  chosen <- ordered %>%
    filter(!(feature %in% exclude_features)) %>%
    slice_head(n = top_n) %>%
    pull(feature)

  if (length(chosen) < top_n) {
    remainder <- ordered %>%
      filter(feature %in% exclude_features) %>%
      slice_head(n = top_n - length(chosen)) %>%
      pull(feature)
    chosen <- c(chosen, remainder)
  }
  chosen
}

top_drivers <- pick_top_drivers(feature_importance)
write_csv(tibble(feature = top_drivers), output_csv)

global_df <- features_df %>%
  # Keep only the global region (the response curves are meant for the main target)
  filter(region == "Global") %>%
  select(any_of(c("temp_anomaly_C", "region", top_drivers)))

global_df <- global_df %>%
  filter(!is.na(temp_anomaly_C))

make_response_stats <- function(df, driver, bins) {
  # Compute quantile breakpoints so each bin has roughly equal counts.
  # This improves stability compared to uniform bin widths when drivers
  # are skewed.
  x <- df[[driver]]
  qs <- quantile(x, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE, type = 7)
  qs <- unique(qs)
  if (length(qs) < 4) {
    # Not enough variability; fall back to fewer bins
    qs <- unique(quantile(x, probs = seq(0, 1, length.out = 10), na.rm = TRUE, type = 7))
  }

  df_tmp <- df %>%
    mutate(bin = cut(.data[[driver]], breaks = qs, include.lowest = TRUE, labels = FALSE)) %>%
    filter(!is.na(bin), !is.na(.data[[driver]]))

  stats <- df_tmp %>%
    group_by(bin) %>%
    summarise(
      x_mean = mean(.data[[driver]], na.rm = TRUE),
      y_mean = mean(temp_anomaly_C, na.rm = TRUE),
      y_se   = sd(temp_anomaly_C, na.rm = TRUE) / sqrt(sum(!is.na(temp_anomaly_C))),
      n       = sum(!is.na(temp_anomaly_C)),
      .groups = "drop"
    )

  stats
}

all_stats <- lapply(top_drivers, function(d) make_response_stats(global_df, d, bins)) %>%
  Map(function(stats, d) {
    stats$driver <- d
    stats
  }, ., top_drivers) %>%
  bind_rows()

all_stats <- all_stats %>%
  mutate(driver = factor(driver, levels = top_drivers))

p <- ggplot(all_stats, aes(x = x_mean, y = y_mean)) +
  geom_ribbon(aes(ymin = y_mean - y_se, ymax = y_mean + y_se), alpha = 0.25, fill = "#1f77b4") +
  geom_line(color = "#1f77b4", linewidth = 1.2) +
  geom_point(color = "#1f77b4", size = 1.8, alpha = 0.85) +
  facet_wrap(~ driver, ncol = 1, scales = "free_x") +
  labs(
    title = "Binned response curves for top-ranked drivers",
    x = "Driver value (binned by quantiles)",
    y = "Mean temperature anomaly (°C)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(output_png, p, width = 9, height = 11, dpi = 220)
cat("Wrote:", output_png, "\n")

