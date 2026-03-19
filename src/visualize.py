# Plotting step.
# Reads the engineered CSV outputs from output/processed/ and writes figures
# into output/figures/ used in the methodology write-up.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Patch

from .paths import DATA_PROCESSED, RESULTS

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("colorblind")


def plot_timeseries(df):
    # If multiple regions exist, draw the Global series once per date
    if "region" in df.columns and df["region"].nunique() > 1:
        plot_df = df[df["region"] == "Global"].sort_values("date").copy()
    else:
        plot_df = df.sort_values("date").copy()
    # Fallback: if Global is missing, show the first available per-date series.
    if plot_df.empty:
        plot_df = df.drop_duplicates(subset="date").sort_values("date")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(plot_df["date"], plot_df["co2_ppm"], color=COLORS[0], linewidth=0.8)
    axes[0].set_ylabel("CO2 (ppm)")
    axes[0].set_title("Atmospheric CO2 Concentration")

    axes[1].plot(plot_df["date"], plot_df["ch4_ppb"], color=COLORS[1], linewidth=0.8)
    axes[1].set_ylabel("CH4 (ppb)")
    axes[1].set_title("Atmospheric CH4 Concentration")

    axes[2].plot(plot_df["date"], plot_df["n2o_ppb"], color=COLORS[2], linewidth=0.8)
    axes[2].set_ylabel("N2O (ppb)")
    axes[2].set_title("Atmospheric N2O Concentration")

    axes[3].plot(plot_df["date"], plot_df["temp_anomaly_C"], color=COLORS[3], linewidth=0.8, alpha=0.5, label="Monthly")
    axes[3].plot(plot_df["date"], plot_df["temp_anomaly_C"].rolling(12, min_periods=1).mean(), color="black", linewidth=1.5, label="12-month MA")
    axes[3].set_ylabel("Temp Anomaly (°C)")
    axes[3].set_title("Global Temperature Anomaly")
    axes[3].legend(loc="upper left")

    axes[3].xaxis.set_major_locator(mdates.YearLocator(10))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Greenhouse-Gas Trends vs Global Temperature Anomaly", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS / "time_series_ghg_vs_temp.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ time_series_ghg_vs_temp.png")


def plot_feature_importance(imp_df):
    # Use only top features so the chart stays readable.
    top = imp_df.copy()
    top["abs_perm"] = top["permutation"].abs()
    top = top.nlargest(15, "abs_perm")
    methods = ["linear_std_coef", "random_forest", "gradient_boosting", "permutation"]
    labels = ["Linear (|coef|)", "Random Forest", "Gradient Boosting", "Permutation"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 7), sharey=True)
    for ax, method, label in zip(axes, methods, labels):
        order = top.sort_values(method)
        palette = order["category"].map({
            "GHG": COLORS[0], "Solar": COLORS[1], "Natural": COLORS[2],
            "Temporal": COLORS[4], "Seasonal": COLORS[5],
        })
        ax.barh(order["feature"], order[method], color=palette.values)
        ax.set_xlabel("Importance")
        ax.set_title(label)
    handles = [
        Patch(facecolor=COLORS[0], label="GHG / Anthropogenic"),
        Patch(facecolor=COLORS[1], label="Solar"),
        Patch(facecolor=COLORS[2], label="Natural"),
        Patch(facecolor=COLORS[4], label="Temporal"),
        Patch(facecolor=COLORS[5], label="Seasonal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, frameon=True)
    fig.suptitle("Feature Importance Across Ranking Methods", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(RESULTS / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ feature_importance.png")


def plot_scatter_top_features(df, imp_df):
    # Scatter + linear fit for the top drivers so readers can see direction
    # and strength of association with temperature.
    top3 = imp_df.nlargest(3, "permutation")["feature"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, feat in zip(axes, top3):
        ax.scatter(df[feat], df["temp_anomaly_C"], s=6, alpha=0.35, color=COLORS[0])
        m, b = np.polyfit(df[feat], df["temp_anomaly_C"], 1)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 200)
        ax.plot(x_line, m * x_line + b, color="red", linewidth=2)
        r = np.corrcoef(df[feat], df["temp_anomaly_C"])[0, 1]
        ax.set_title(f"{feat}  (r = {r:.3f})")
        ax.set_xlabel(feat)
        ax.set_ylabel("Temperature Anomaly (°C)")
    fig.suptitle("Top-3 Features vs Temperature Anomaly", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "scatter_top_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ scatter_top_features.png")


def plot_predictions(pred_df):
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    # Keep a stable, reader-friendly order for legend entries
    col_order = [
        "pred_linear_regression",
        "pred_ridge_regression",
        "pred_random_forest",
        "pred_gradient_boosting",
    ]
    col_order = [c for c in col_order if c in pred_cols]

    # Compute test-set R² for each model (used in legend only)
    test_df = pred_df[pred_df["split"] == "test"].copy()
    y_true = test_df["temp_anomaly_C"].to_numpy()
    y_mean = y_true.mean() if len(y_true) else 0.0
    ss_tot = ((y_true - y_mean) ** 2).sum()

    def r2(y_pred) -> float:
        if len(y_true) == 0 or ss_tot == 0:
            return float("nan")
        ss_res = ((y_true - y_pred) ** 2).sum()
        return 1.0 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        pred_df["date"],
        pred_df["temp_anomaly_C"],
        color="black",
        linewidth=1.3,
        alpha=0.75,
        label="Actual",
        zorder=3,
    )

    # Light styling: remove heavy grids and reduce visual clutter
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, col in enumerate(col_order):
        # Convert column name to a readable model label
        label = col.replace("pred_", "").replace("_", " ").title()
        r2_score = r2(test_df[col].to_numpy())
        ax.plot(
            pred_df["date"],
            pred_df[col],
            linewidth=1.1,
            alpha=0.85,
            color=COLORS[i],
            label=f"{label} (R²={r2_score:.2f})",
            zorder=2,
        )

    test_start = pred_df.loc[pred_df["split"] == "test", "date"].iloc[0]
    ax.axvline(test_start, linestyle="--", color="grey", linewidth=1, label="_nolegend_")

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature Anomaly (°C)")
    ax.set_title(
        "Model Predictions vs Actual Temperature Anomaly",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / "model_predictions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ model_predictions.png")


def run():
    # Read processed outputs and generate the full set of course figures.
    print("Loading data …")
    features_df = pd.read_csv(DATA_PROCESSED / "climate_features.csv", parse_dates=["date"])
    importance_df = pd.read_csv(DATA_PROCESSED / "feature_importance.csv")
    predictions_df = pd.read_csv(DATA_PROCESSED / "model_predictions.csv", parse_dates=["date"])
    print(f"  climate_features:   {features_df.shape}")
    print(f"  feature_importance: {importance_df.shape}")
    print(f"  model_predictions:  {predictions_df.shape}\n")

    print("Generating plots …")
    plot_timeseries(features_df)
    plot_feature_importance(importance_df)
    plot_scatter_top_features(features_df, importance_df)
    plot_predictions(predictions_df)
    print(f"\nAll plots saved to: {RESULTS}")


if __name__ == "__main__":
    run()
