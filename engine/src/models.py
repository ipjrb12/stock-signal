"""
src/models.py -- Workstream B (Modelling) deliverable
Capstone: Demand Forecasting and Inventory Optimisation for Retail

Interface (per the frozen module contract, Slide 12):
    fit(X, y, alpha=None) -> Model
        alpha=None      -> point model, squared-error loss on log1p(sales_units)
        alpha=0..1      -> quantile model at that quantile

Input: features_2017-06-01_onward.csv, as built by Workstream A.
Output consumed by Workstream C: calibrated quantile forecasts
    (P50, P80, P90, P95, P98) in original sales_units scale, per
    date / store_nbr / family, feeding src/policy_engine.py's
    newsvendor calculation.

Primary model: sklearn HistGradientBoostingRegressor.
NOTE: LightGBM/XGBoost are not installable in this environment (no
network access). HistGradientBoostingRegressor is sklearn's direct
equivalent -- histogram-based gradient boosting, native categorical
support, native missing-value handling, and a built-in quantile loss
-- and is a drop-in swap for LightGBM if/when it becomes available;
no feature engineering or interface changes would be needed.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

# P10 added in v2 so the exception detector can flag demand collapses as well
# as spikes. The newsvendor engine still reads P50..P98 only.
QUANTILES = (0.1, 0.5, 0.8, 0.9, 0.95, 0.98)
QCOLS = ['P10', 'P50', 'P80', 'P90', 'P95', 'P98']

# ---------------------------------------------------------------------------
# 1. Leakage-safe feature list
# ---------------------------------------------------------------------------
# Columns excluded because they describe the *target date itself* rather
# than what is known at the forecast origin (16 days earlier). Some of
# these are silent leaks: they are fully populated in the labelled
# train/valid data (so a naive backtest would not catch the problem) but
# are genuinely unknowable -- and in this file, actually NULL -- for the
# blind future window (`transactions_LEAKY`, `sales_units`).
LEAKY_COLS = [
    'transactions_LEAKY',                 # same-day transactions: a near-proxy for same-day sales
    'oil_price', 'oil_price_imputed',     # contemporaneous oil price, not known 16 days ahead
    'temp_mean', 'temp_max', 'temp_min',  # actual observed weather for the target date
    'precip_mm', 'wind_max_kmh', 'humidity_pct', 'sunshine_hrs',
]
NON_FEATURE_COLS = [
    'id', 'date', 'sales_units', 'split', 'series_start',
    'holiday_name', 'event_name', 'y',
]
CAT_COLS = ['store_nbr', 'family', 'city', 'state', 'store_type', 'cluster']


def load_panel(path):
    """Load the feature file and return (df, feature_cols, cat_cols)."""
    df = pd.read_csv(path, parse_dates=['date'])
    feature_cols = [c for c in df.columns if c not in LEAKY_COLS + NON_FEATURE_COLS]
    for c in CAT_COLS:
        df[c] = df[c].astype('category')
    return df, feature_cols, CAT_COLS


def fit(X, y, alpha=None, cat_cols=CAT_COLS):
    """Train the primary model. alpha=None -> point model, else quantile alpha."""
    cat_mask = [c in cat_cols for c in X.columns]
    if alpha is None:
        model = HistGradientBoostingRegressor(
            loss='squared_error', max_iter=350, learning_rate=0.06,
            max_leaf_nodes=63, min_samples_leaf=30,
            categorical_features=cat_mask, random_state=0, early_stopping=False)
    else:
        model = HistGradientBoostingRegressor(
            loss='quantile', quantile=alpha, max_iter=350, learning_rate=0.06,
            max_leaf_nodes=63, min_samples_leaf=30,
            categorical_features=cat_mask, random_state=0, early_stopping=False)
    model.fit(X, y)
    return model


def predict_units(model, X):
    """Predict in original sales_units scale (inverts the log1p target)."""
    return np.clip(np.expm1(model.predict(X)), 0, None)


def fit_quantile_suite(X, y, quantiles=QUANTILES, cat_cols=CAT_COLS):
    """Fit one model per quantile. Returns {alpha: model}."""
    return {q: fit(X, y, alpha=q, cat_cols=cat_cols) for q in quantiles}


def enforce_monotonic_quantiles(pred_df, qcols=('P50', 'P80', 'P90', 'P95', 'P98')):
    """Independently-trained quantile models can 'cross' (e.g. P50 > P80) on a
    small share of rows. Sort each row's quantile values ascending -- the
    standard rearrangement fix -- so the output is a valid, monotonic
    distribution for the newsvendor policy to consume."""
    vals = np.sort(pred_df[list(qcols)].values, axis=1)
    for i, c in enumerate(qcols):
        pred_df[c] = vals[:, i]
    return pred_df


def fit_family_conformal_correction(cal_actual_units, cal_pred_units, cal_family, target_q=0.9):
    """Split-conformal correction for a quantile model that under- or
    over-covers on a held-out calibration window. Returns a per-family
    additive correction: correction[f] = the target_q quantile of
    (actual - predicted) residuals observed for family f during calibration.
    Add this to future predictions for that family."""
    resid = cal_actual_units - cal_pred_units
    return pd.Series(resid, index=cal_family).groupby(level=0).quantile(target_q)


# ---------------------------------------------------------------------------
# 2. Persistence (v2): the exact models behind the delivered forecast files
# ---------------------------------------------------------------------------
# One bundle holds everything needed to reproduce a forecast row, so the agent's
# explain_forecast tool explains the same model whose numbers it quotes.
def calibrate_and_sort(out, p90_correction):
    """The single post-processing step behind every delivered forecast.

    1. Per-family additive split-conformal correction on P90 (raw kept as P90_raw).
    2. Clip at 0 (sales cannot be negative).
    3. Sort P50..P98 row-wise, so a corrected P90 above P95 pushes P95 up rather
       than being capped back down (v1's test file capped; v2 sorts both files).
    4. P10 is kept out of the sort and capped at P50, so adding it can never
       move the newsvendor inputs. P10 is NOT conformal-corrected: per-family
       additive and log-scale corrections both made its coverage worse on the
       valid window (see MODELLING_REPORT.md section 3).
    """
    fam = out['family'].astype(str)
    out['P90_raw'] = out['P90']
    out['P90'] = out['P90'] + fam.map(p90_correction).fillna(0).values
    cols = [c for c in QCOLS if c in out]
    out[cols] = out[cols].clip(lower=0)
    out = enforce_monotonic_quantiles(out)
    if 'P10' in out:
        out['P10'] = np.minimum(out['P10'], out['P50'])
    return out


def save_bundle(path, suite, p90_correction, feature_cols, df, cutoff, model_version):
    """suite: {alpha: model}; p90_correction: per-family Series; df: the loaded
    panel, used to record the category levels each model was fit on."""
    bundle = dict(
        models=suite,
        p90_correction=p90_correction.to_dict(),
        feature_cols=list(feature_cols),
        cat_cols=list(CAT_COLS),
        categories={c: list(df[c].cat.categories) for c in CAT_COLS},
        quantiles=list(suite),
        cutoff=str(pd.Timestamp(cutoff).date()),
        model_version=model_version,
        sklearn_version=sklearn.__version__,
        target='log1p(sales_units)',
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return bundle


def load_bundle(path):
    b = joblib.load(path)
    if b['sklearn_version'] != sklearn.__version__:
        import warnings
        warnings.warn(f"bundle built with scikit-learn {b['sklearn_version']}, "
                      f"running {sklearn.__version__}")
    return b


def prepare_X(df, bundle):
    """Feature matrix with categoricals encoded exactly as at training time."""
    X = df[bundle['feature_cols']].copy()
    for c in bundle['cat_cols']:
        X[c] = pd.Categorical(X[c], categories=bundle['categories'][c])
    return X


def predict_quantiles(df, bundle):
    """Calibrated, monotonic quantiles in sales units, identical to the
    delivered forecast files (see calibrate_and_sort)."""
    X = prepare_X(df, bundle)
    out = df[['date', 'store_nbr', 'family']].copy()
    out['store_nbr'] = out['store_nbr'].astype(int)
    out['family'] = out['family'].astype(str)
    for q, col in zip(QUANTILES, QCOLS):
        out[col] = predict_units(bundle['models'][q], X)
    out = calibrate_and_sort(out, bundle['p90_correction'])
    out['model_version'] = bundle['model_version']
    return out
