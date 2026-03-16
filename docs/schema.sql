-- Schema for climate-drivers regression pipeline (for ER diagram in MySQL Workbench)
-- Run this in MySQL Workbench to create the tables, then use Reverse Engineer or EER diagram to redraw.

-- Create a database so Workbench can reverse-engineer an EER diagram cleanly
CREATE DATABASE IF NOT EXISTS climate_drivers_regression
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;
USE climate_drivers_regression;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ---------------------------------------------------------------------
-- Data sources (NOAA GML, NASA GISS, OWID)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS data_source;
CREATE TABLE data_source (
  source_id    INT          NOT NULL AUTO_INCREMENT,
  source_name  VARCHAR(100) NOT NULL,
  organization VARCHAR(150) NULL,
  url          TEXT         NULL,
  temporal_res VARCHAR(20)  NULL COMMENT 'monthly or yearly',
  last_updated DATE         NULL,
  PRIMARY KEY (source_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- Merged raw data (one row per date + region); from data/raw/raw_merged.csv
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS raw_merged;
CREATE TABLE raw_merged (
  record_id       BIGINT        NOT NULL AUTO_INCREMENT,
  period_date     DATE          NOT NULL,
  region          VARCHAR(50)   NOT NULL,
  co2_ppm         DECIMAL(10,4) NULL,
  ch4_ppb         DECIMAL(10,2) NULL,
  n2o_ppb         DECIMAL(10,2) NULL,
  solar_w_m2      DECIMAL(8,4)  NULL,
  temp_anomaly_c  DECIMAL(6,4)  NULL,
  co2_emissions_gt DECIMAL(12,4) NULL,
  land_use_gt     DECIMAL(12,4)  NULL,
  volcanic_flag   TINYINT(1)    DEFAULT 0,
  aerosol_aod     DECIMAL(8,6)  NULL,
  PRIMARY KEY (record_id),
  KEY idx_raw_date (period_date),
  KEY idx_raw_region (region)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- Processed records with engineered features; from data/processed/climate_features.csv
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS processed_record;
CREATE TABLE processed_record (
  record_id          BIGINT        NOT NULL AUTO_INCREMENT,
  raw_record_id      BIGINT        NOT NULL,
  period_date        DATE          NOT NULL,
  region             VARCHAR(50)   NULL,
  months_since_base  INT           NULL,
  co2_ppm            DECIMAL(10,4) NULL,
  ch4_ppb            DECIMAL(10,2) NULL,
  n2o_ppb            DECIMAL(10,2) NULL,
  solar_w_m2         DECIMAL(8,4)  NULL,
  aerosol_aod        DECIMAL(8,6)  NULL,
  co2_growth         DECIMAL(10,6) NULL,
  ch4_growth         DECIMAL(10,6) NULL,
  n2o_growth         DECIMAL(10,6) NULL,
  co2_ppm_ma12       DECIMAL(10,4) NULL,
  volcanic_flag      TINYINT(1)    DEFAULT 0,
  enso_proxy         DECIMAL(8,6)  NULL,
  temp_anomaly_c     DECIMAL(6,4)  NULL COMMENT 'target',
  PRIMARY KEY (record_id),
  KEY idx_proc_date (period_date),
  CONSTRAINT fk_proc_raw FOREIGN KEY (raw_record_id) REFERENCES raw_merged (record_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- Regression models (Linear, Ridge, Random Forest, Gradient Boosting)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS regression_model;
CREATE TABLE regression_model (
  model_id       INT          NOT NULL AUTO_INCREMENT,
  model_name     VARCHAR(100) NOT NULL,
  model_type     VARCHAR(50)  NULL COMMENT 'linear or tree',
  target_var     VARCHAR(80)  DEFAULT 'temp_anomaly_c',
  PRIMARY KEY (model_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- Training run (one per model per pipeline run)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS training_run;
CREATE TABLE training_run (
  run_id     INT          NOT NULL AUTO_INCREMENT,
  model_id   INT          NOT NULL,
  run_date   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
  train_size INT          NULL,
  test_size  INT          NULL,
  r2_score   DECIMAL(8,6) NULL,
  rmse       DECIMAL(8,6) NULL,
  status     VARCHAR(20)  DEFAULT 'completed',
  PRIMARY KEY (run_id),
  CONSTRAINT fk_run_model FOREIGN KEY (model_id) REFERENCES regression_model (model_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- Feature importance per run; from data/processed/feature_importance.csv
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS feature_importance;
CREATE TABLE feature_importance (
  importance_id     INT          NOT NULL AUTO_INCREMENT,
  run_id            INT          NOT NULL,
  feature_name      VARCHAR(80)  NOT NULL,
  method            VARCHAR(50)  NULL COMMENT 'linear_std_coef, random_forest, gradient_boosting, permutation',
  importance_score  DECIMAL(12,8) NULL,
  importance_rank   INT          NULL,
  driver_type       VARCHAR(30)  NULL COMMENT 'GHG, Solar, Natural, Temporal, Seasonal',
  PRIMARY KEY (importance_id),
  KEY idx_fi_run (run_id),
  CONSTRAINT fk_fi_run FOREIGN KEY (run_id) REFERENCES training_run (run_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- Predictions (actual vs predicted per record per run); from model_predictions.csv
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS prediction;
CREATE TABLE prediction (
  pred_id           BIGINT       NOT NULL AUTO_INCREMENT,
  run_id            INT          NOT NULL,
  record_id         BIGINT       NOT NULL,
  split_set         VARCHAR(10)  NULL COMMENT 'train or test',
  actual_anomaly    DECIMAL(6,4) NULL,
  predicted_anomaly DECIMAL(6,4) NULL,
  residual          DECIMAL(6,4) NULL,
  PRIMARY KEY (pred_id),
  KEY idx_pred_run (run_id),
  KEY idx_pred_record (record_id),
  CONSTRAINT fk_pred_run    FOREIGN KEY (run_id)    REFERENCES training_run (run_id) ON DELETE CASCADE,
  CONSTRAINT fk_pred_record FOREIGN KEY (record_id) REFERENCES processed_record (record_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- Visualizations (plots saved in results/)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS visualization;
CREATE TABLE visualization (
  viz_id        INT          NOT NULL AUTO_INCREMENT,
  run_id        INT          NULL COMMENT 'optional: link to a run',
  plot_type     VARCHAR(80)  NOT NULL COMMENT 'e.g. time_series_ghg, feature_importance, scatter_top, predictions',
  title         VARCHAR(200) NULL,
  file_path     VARCHAR(500) NULL,
  created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (viz_id),
  CONSTRAINT fk_viz_run FOREIGN KEY (run_id) REFERENCES training_run (run_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SET FOREIGN_KEY_CHECKS = 1;

-- Optional: seed data sources
INSERT INTO data_source (source_name, organization, temporal_res) VALUES
  ('NOAA GML', 'NOAA Global Monitoring Laboratory', 'monthly'),
  ('NASA GISS', 'NASA GISS Surface Temperature', 'monthly'),
  ('OWID', 'Our World in Data', 'yearly');

-- Optional: seed models
INSERT INTO regression_model (model_name, model_type) VALUES
  ('Linear Regression', 'linear'),
  ('Ridge Regression', 'linear'),
  ('Random Forest', 'tree'),
  ('Gradient Boosting', 'tree');
