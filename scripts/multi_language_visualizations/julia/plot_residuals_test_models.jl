# Residuals over time (Julia)
#
# Generates a 2x2 panel figure showing test-set residuals for each fitted model.
#
# Output:
#   output/figures/extra_julia_test_residuals.png
#
# Notes:
# - This script expects `output/processed/model_predictions.csv` to exist.
# - It automatically selects residual series from columns that start with `pred_`.

using CSV
using DataFrames
using Statistics
using Dates
using CairoMakie

importance_path = "output/processed/feature_importance.csv" # used only to pick driver set if you extend later
pred_path        = "output/processed/model_predictions.csv"
out_path         = "output/figures/extra_julia_test_residuals.png"

window = 6  # rolling mean window (months)

df = CSV.read(pred_path, DataFrame)

# Ensure date type
if !(df.date isa Vector{Date})
    df.date = Date.(df.date)
end

df.split = string.(df.split)

pred_cols = filter(c -> startswith(c, "pred_"), names(df))
if length(pred_cols) == 0
    error("No prediction columns found. Expected columns starting with 'pred_'.")
end

function rolling_mean(v::Vector{Float64}, w::Int)
    w <= 1 && return v
    n = length(v)
    out = similar(v)
    for i in 1:n
        a = max(1, i - w + 1)
        out[i] = mean(v[a:i])
    end
    return out
end

models = sort(pred_cols)
fig = Figure(resolution = (1100, 800), fontsize = 14)

axs = [
    Axis(fig[1, 1], title = replace(replace(models[1], "pred_" => ""), "_" => " "), xlabel = "Date", ylabel = "Residual (deg C)"),
    Axis(fig[1, 2], title = replace(replace(models[2], "pred_" => ""), "_" => " "), xlabel = "Date", ylabel = "Residual (deg C)"),
    Axis(fig[2, 1], title = replace(replace(models[3], "pred_" => ""), "_" => " "), xlabel = "Date", ylabel = "Residual (deg C)"),
    Axis(fig[2, 2], title = replace(replace(models[4], "pred_" => ""), "_" => " "), xlabel = "Date", ylabel = "Residual (deg C)"),
]

test_df = df[df.split .== "test", :]

for (i, col) in enumerate(models)
    ax = axs[i]
    sub = test_df
    # residual = actual - predicted
    resid = sub.temp_anomaly_C .- sub[!, col]
    tmp = DataFrame(date = sub.date, residual = resid)
    g = combine(groupby(tmp, :date), :residual => mean => :residual_mean)
    sort!(g, :date)

    dates = g.date
    vals = Float64.(g.residual_mean)
    sm = rolling_mean(vals, window)

    lines!(ax, dates, vals, color = "#1f77b4", linewidth = 1, label = "Residual mean")
    lines!(ax, dates, sm, color = "#d62728", linewidth = 2, label = "Rolling mean")
    ax.legendvisible = false
end

save(out_path, fig)
println("Wrote: ", out_path)

