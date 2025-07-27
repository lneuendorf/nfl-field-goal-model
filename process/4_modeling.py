from pathlib import Path
import os
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
import arviz as az
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

n_cores = os.cpu_count()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),  # logs to console
        logging.FileHandler("../models/model_training.log")  # logs to file
    ]
)
logger = logging.getLogger(__name__)

n_cores = os.cpu_count()

train_data_path = Path('../data/processed/field_goal_data.parquet')
models_dir = Path('../models')
models_dir.mkdir(parents=True, exist_ok=True)

logger.info("Loading data...")
fg_attempts = (
    pd.read_parquet(train_data_path)
    .assign(
        iced_kicker=lambda x: x['iced_kicker'].astype(int)
    )
    .merge(
        pd.read_csv('../data/processed/stadium_elevations.csv')
            [['stadium_id','elevation_feet']],
        on='stadium_id',
        how='left'
    )
    .drop_duplicates()
)
logger.info(f"Data loaded with {len(fg_attempts)} rows")

def evaluate_model(model, trace, y, coefficients):
    logger.info("Starting model evaluation...")
    with model:
        pp_samples = pm.sample_posterior_predictive(trace, var_names=['p'])
    mean_p = pp_samples['posterior_predictive']['p'].mean(axis=1).mean(axis=0)
    brier = brier_score_loss(y, mean_p)
    auc = roc_auc_score(y, mean_p)
    logger.info(f"Brier Score: {brier:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")

    loo = az.loo(trace, pointwise=True)
    logger.info(f"LOOIC Summary:\n{loo}")

    waic = az.waic(trace, pointwise=True)
    logger.info(f"WAIC Summary:\n{waic}")

    logger.info("Model Diagnostics:")
    summary = az.summary(trace, var_names=coefficients, kind='stats', round_to=2)
    logger.info(f"\n{summary}")

    calibration_df = pd.DataFrame({'pred': mean_p, 'actual': y})
    calibration_df['decile'] = pd.qcut(calibration_df['pred'], q=10, duplicates='drop')
    calib_summary = calibration_df.groupby('decile').agg(
        actual_mean=('actual', 'mean'),
        pred_mean=('pred', 'mean'),
        count=('pred', 'count')
    )
    logger.info(f"Calibration by Decile:\n{calib_summary}")
                                            
scaler = StandardScaler()

dataset = (
    fg_attempts
    .query('field_goal_result.isin(["made", "missed"])')
    .copy()
    .query('season >= 2015') # Train on 10 seasons
)

ytg_poly2 = scaler.fit_transform(
    PolynomialFeatures(
        degree=2, 
        include_bias=False
    ).fit_transform(dataset[['yardline_100']])
)[:, 1]

# Weather features
temperature = scaler.fit_transform(dataset[['temperature']]).flatten()
precipitation_chance = scaler.fit_transform(dataset[['chance_of_rain']]).flatten()
snow_severity = scaler.fit_transform(dataset[['snow_severity']]).flatten()
wind_gust = scaler.fit_transform(dataset[['wind_gust']]).flatten()
elevation = scaler.fit_transform(dataset[['elevation_feet']]).flatten()

# Kicker-season ID for hierarchical intercept and slope
dataset["kicker_season"] = dataset["kicker_player_id"].astype(str) + "_" + dataset["season"].astype(str)
kicker_season_ids, kicker_season_idx = np.unique(dataset["kicker_season"], return_inverse=True)
n_kicker_seasons = len(kicker_season_ids)

outdoor_indicator = dataset['is_indoor'].eq(0).astype(int).values  # 1 if outdoor, else 0

# Other fixed variables
is_home = scaler.fit_transform(dataset['is_home'].astype(int).values.reshape(-1, 1)).flatten()
lighting_condition = scaler.fit_transform(dataset['lighting_condition'].astype(int).values.reshape(-1, 1)).flatten()
pressure_rating = scaler.fit_transform(dataset['pressure_rating'].values.reshape(-1, 1)).flatten()
iced_kicker = scaler.fit_transform(dataset['iced_kicker'].astype(int).values.reshape(-1, 1)).flatten()
season_scaled = scaler.fit_transform(dataset[['season']]).flatten()

# Response
y = (dataset['field_goal_result'] == 'made').astype(int).values

logger.info("Starting model setup and sampling...")

with pm.Model() as model_poly2:
    # Global intercept
    alpha = pm.Normal("alpha", 0, 5)

    # Fixed effects (same as before)
    beta_home = pm.Normal("beta_home", 0.05, .5)
    beta_timeofday = pm.Normal("beta_timeofday", 0.05, .5)
    beta_pressure = pm.Normal("beta_pressure", -0.05, .5)
    beta_iced = pm.Normal("beta_iced", -0.05, .5)
    beta_pressure_iced = pm.Normal("beta_pressure_iced", -0.05, .5)
    beta_season = pm.Normal("beta_season", 0.05, .5)

    # Weather effects (same as before)
    beta_wind = pm.Normal("beta_wind", -0.2, .5)
    beta_wind_ytg = pm.Normal("beta_wind_ytg", 0, 1)
    beta_temp = pm.Normal("beta_temp", 0.1, .5)
    beta_temp_ytg = pm.Normal("beta_temp_ytg", 0, 1)
    beta_precip = pm.Normal("beta_precip", 0, 1)
    beta_precip_ytg = pm.Normal("beta_precip_ytg", 0, 1)
    beta_snow = pm.Normal("beta_snow", 0, 1)
    beta_snow_ytg = pm.Normal("beta_snow_ytg", 0, 1)
    beta_elev = pm.Normal("beta_elev", 0.05, .5)
    beta_elev_ytg = pm.Normal("beta_elev_ytg", 0, 1)

    # Kicker intercepts
    from collections import defaultdict
    kicker_season_map = defaultdict(list)
    for i, (kicker_id, season) in enumerate(zip(dataset["kicker_player_id"], dataset["season"])):
        kicker_season_map[kicker_id].append((season, i))
    dataset["kicker_season"] = [
        f"{kicker_id}_{season}" for kicker_id, season in zip(dataset["kicker_player_id"], dataset["season"])
    ]  

    unique_kicker_seasons = dataset["kicker_season"].unique()
    n_kicker_seasons = len(unique_kicker_seasons)
    logger.info(f"Number of unique kicker-seasons: {n_kicker_seasons}")

    # Create a matrix to map from kicker-season to their position in the GRW
    kicker_season_grw_idx = np.zeros(len(dataset), dtype=int)
    for kicker_season_idx, kicker_season in enumerate(unique_kicker_seasons):
        kicker_season_grw_idx[dataset["kicker_season"] == kicker_season] = kicker_season_idx

    # Hyperpriors for GRW
    sigma_rw_intercept = pm.Exponential("sigma_rw_intercept", 1.0)
    sigma_rw_slope = pm.Exponential("sigma_rw_slope", 1.0)

    # Gaussian Random Walk for kicker-season intercepts
    kicker_season_intercepts = pm.GaussianRandomWalk(
        "kicker_season_intercept",
        sigma=sigma_rw_intercept,
        shape=n_kicker_seasons,
        init_dist=pm.Normal.dist(0, 1)
    )
    kicker_season_ytg_slopes = pm.GaussianRandomWalk(
        "kicker_season_ytg_slope", 
        sigma=sigma_rw_slope, 
        shape=n_kicker_seasons,
        init_dist=pm.Normal.dist(mu=-1, sigma=1)
    )
    
    logit_p = (
        alpha
        + kicker_season_intercepts[kicker_season_grw_idx]
        + kicker_season_ytg_slopes[kicker_season_grw_idx] * ytg_poly2
        + beta_home * is_home
        + beta_timeofday * lighting_condition
        + beta_pressure * pressure_rating
        + beta_iced * iced_kicker
        + beta_pressure_iced * pressure_rating * iced_kicker
        + beta_season * season_scaled
        + beta_wind * wind_gust * outdoor_indicator
        + beta_wind_ytg * wind_gust * ytg_poly2 * outdoor_indicator
        + beta_temp * temperature * outdoor_indicator
        + beta_temp_ytg * temperature * ytg_poly2 * outdoor_indicator
        + beta_precip * precipitation_chance * outdoor_indicator
        + beta_precip_ytg * precipitation_chance * ytg_poly2 * outdoor_indicator
        + beta_snow * snow_severity * outdoor_indicator
        + beta_snow_ytg * snow_severity * ytg_poly2 * outdoor_indicator
        + beta_elev * elevation * outdoor_indicator
        + beta_elev_ytg * elevation * ytg_poly2 * outdoor_indicator
    )

    # Likelihood
    p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)

    # Sampling with increased tuning
    trace_poly2 = pm.sample(
        draws=1000,
        tune=2000,
        target_accept=0.95,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True},
        cores=n_cores,
        progressbar=True,
        init="jitter+adapt_diag"
    )

logger.info("Sampling complete, saving trace...")
trace_path = os.path.join(models_dir, 'fg_model_trace.nc')
az.to_netcdf(trace_poly2, trace_path)
logger.info(f"Trace saved to {trace_path}")

evaluate_model(model_poly2, trace_poly2, y, [
    'alpha', 'kicker_season_intercept', 'kicker_season_ytg_slope',
    'beta_home', 'beta_timeofday', 'beta_pressure', 'beta_iced',
    'beta_pressure_iced', 'beta_season',
    'beta_wind', 'beta_wind_ytg', 'beta_temp', 'beta_temp_ytg',
    'beta_precip', 'beta_precip_ytg', 'beta_snow', 'beta_snow_ytg',
    'beta_elev', 'beta_elev_ytg'
])

logger.info("Model evaluation finished.")