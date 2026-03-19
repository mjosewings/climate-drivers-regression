% Lagged correlations (MATLAB)
%
% This script:
%   1) Reads output/processed/feature_importance.csv
%   2) Picks the top drivers by permutation importance
%   3) Filters output/processed/climate_features.csv to region == "Global"
%   4) Computes time-lagged Pearson correlations between temperature anomaly
%      and each driver for lags from -12..+12 months
%   5) Saves a heatmap to output/figures/
%
% Output:
%   output/figures/extra_matlab_lagged_corr_top_drivers.png

importancePath = "output/processed/feature_importance.csv";
featuresPath   = "output/processed/climate_features.csv";
outPng         = "output/figures/extra_matlab_lagged_corr_top_drivers.png";

topN = 3;
excludeFeatures = ["months_since_1960", "month_sin", "month_cos"];
lags = -12:12; % months

fi = readtable(importancePath);
fi.permutation = double(fi.permutation);

% Sort by permutation importance (descending)
[~, idx] = sort(fi.permutation, "descend");
chosen = strings(0);

for k = 1:length(idx)
    feat = string(fi.feature(idx(k)));
    if any(excludeFeatures == feat)
        continue;
    end
    chosen(end+1) = feat; %#ok<AGROW>
    if length(chosen) >= topN
        break;
    end
end

features = readtable(featuresPath);

% Keep Global series
if any(strcmp(features.Properties.VariableNames, "region"))
    globalMask = string(features.region) == "Global";
else
    error("Expected a 'region' column in climate_features.csv.");
end

features = features(globalMask, :);
temp = double(features.temp_anomaly_C);

numDrivers = length(chosen);
numLags = length(lags);
R = NaN(numDrivers, numLags);

for i = 1:numDrivers
    driver = chosen(i);
    x = double(features{:, driver});

    for j = 1:numLags
        lag = lags(j);

        if lag >= 0
            % Correlate temp(t+lag) with x(t)
            a = temp(1+lag:end);
            b = x(1:end-lag);
        else
            % Correlate temp(t) with x(t-lag)
            L = -lag;
            a = temp(1:end-L);
            b = x(1+L:end);
        end

        idxValid = ~isnan(a) & ~isnan(b);
        if sum(idxValid) >= 10
            cc = corr(a(idxValid), b(idxValid));
            R(i, j) = cc;
        end
    end
end

figure('Color', 'w');
imagesc(lags, 1:numDrivers, R);
set(gca, 'YTick', 1:numDrivers, 'YTickLabel', chosen);
colormap('coolwarm');
colorbar;
xlabel('Lag (months)');
ylabel('Driver feature');
title('Lagged correlation between temperature anomaly and top drivers');

% Improve readability
set(gca, 'FontSize', 12);
grid on;

exportgraphics(gca, outPng, 'Resolution', 300);
disp("Wrote: " + outPng);

