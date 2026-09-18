"""
src/policies.py -- the five policies compared in simulation (slide 10).

    P1  7 day moving average x (L+R)        status quo
    P2  model P50 over (L+R)                isolates forecast from safety stock
    P3  model quantile at q* per family     our proposal
    P4  model P95 for all items             cost of excess service
    P5  perfect foresight                   upper bound

Each policy returns an order-up-to level S per series; src/inventory.py turns
it into order = max(0, S - on_hand - on_order). P2-P4 are the same newsvendor
engine (policy_engine.protection_quantile) at different service levels, so the
comparison isolates the choice of quantile, not the code path.

Information discipline on day t (05:00, before trading):
    P1 reads recorded demand for days < t only.
    P2-P4 read quantile forecasts from the latest model whose training cutoff
          is <= t-1. Those forecasts' features use data <= date-16.
    P5 reads actual demand for t .. t+L. It is the only policy allowed to.
"""
import warnings

import numpy as np
import pandas as pd

from src.policy_engine import QCOLS, protection_quantile


class MovingAveragePolicy:
    def __init__(self, demand, window=7, review_period=1, name="P1_moving_avg_7d"):
        self.demand, self.window, self.R, self.name = demand, window, review_period, name

    def order_up_to(self, s):
        hist = self.demand.values[:, max(0, s.t - self.window):s.t]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # series with no history yet
            mean = np.nan_to_num(np.nanmean(hist, axis=1))
        return mean * (s.lead_time + self.R)


class PerfectForesightPolicy:
    def __init__(self, demand, review_period=1, name="P5_perfect_foresight"):
        self.demand, self.R, self.name = demand, review_period, name

    def order_up_to(self, s):
        out = np.zeros(len(s.lead_time))
        for L in np.unique(s.lead_time):
            m = s.lead_time == L
            window = self.demand.values[m, s.t:s.t + L + self.R]
            out[m] = np.nansum(window, axis=1)
        return out


class ForecastCube:
    """Quantile forecasts aligned to a DemandPanel, one layer per model cutoff.

    forecasts: long DataFrame with cutoff, date, store_nbr, family, P50..P98.
    On day t only the latest cutoff <= t-1 is used.
    """

    def __init__(self, forecasts, demand):
        f = forecasts.copy()
        f["date"] = pd.to_datetime(f["date"])
        f["cutoff"] = pd.to_datetime(f["cutoff"])
        key_index = pd.MultiIndex.from_frame(demand.keys)
        self.cutoffs = np.array(sorted(f.cutoff.unique()), dtype="datetime64[ns]")
        n, T = demand.values.shape
        self.cube = np.full((len(self.cutoffs), n, T, len(QCOLS)), np.nan)
        for ci, c in enumerate(self.cutoffs):
            g = f[f.cutoff == c]
            ri = key_index.get_indexer(pd.MultiIndex.from_arrays([g.store_nbr.astype(int), g.family.astype(str)]))
            di = demand.dates.get_indexer(g.date)
            ok = (ri >= 0) & (di >= 0)
            self.cube[ci, ri[ok], di[ok]] = g.loc[ok, QCOLS].to_numpy(float)
        self.dates = demand.dates

    def layer_for(self, date):
        ci = np.searchsorted(self.cutoffs, np.datetime64(pd.Timestamp(date) - pd.Timedelta(days=1)), side="right") - 1
        if ci < 0:
            raise LookupError(f"no model cutoff at or before {date - pd.Timedelta(days=1)}")
        return self.cube[ci]


class QuantilePolicy:
    """Newsvendor order-up-to at service level q (scalar or per-series array)."""

    def __init__(self, cube, q, name, review_period=1, aggregation="sum"):
        self.cube, self.q, self.name = cube, q, name
        self.R, self.aggregation = review_period, aggregation
        self.missing_forecast_cells = 0

    def order_up_to(self, s):
        layer = self.cube.layer_for(s.date)
        q = np.broadcast_to(np.asarray(self.q, float), s.lead_time.shape)
        out = np.zeros(len(s.lead_time))
        for L in np.unique(s.lead_time):
            m = s.lead_time == L
            window = layer[m, s.t:s.t + L + self.R]                 # (m, L+R, 5)
            nan = np.isnan(window).any(-1)
            self.missing_forecast_cells += int(nan.sum())
            out[m] = protection_quantile(np.nan_to_num(window), q[m], self.aggregation)
        return out
