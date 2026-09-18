"""
src/policy_engine.py -- Workstream C deliverable: newsvendor recommendation engine.

Interface (frozen, slide 12):
    plan(store, product, H, on_hand, on_order) -> dict
        Pure function, no model calls inside.

Policy (slide 10):
    q*    = Cu / (Cu + Co)
    S     = F^-1(q*) of demand over the protection interval L + R
    order = max(0, S - on_hand - on_order)

The forecast distribution is the five calibrated quantiles Workstream B
delivers per date x store x family (P50, P80, P90, P95, P98). The engine never
fits or calls a model: it reads a quantile table and does arithmetic, so the
same inputs always give the same order, and every figure it returns can be
checked by the provenance guardrail.

Two design choices, both checked against data (see INVENTORY_REPORT.md):

1. Quantile of a multi-day sum. Quantiles do not add in general. The default
   "sum" method adds the daily quantiles (exact if daily demands move
   together, conservative otherwise). On the valid window with actuals, summed
   P90 covers 91.5% of 2-day totals and 96.3% of 8-day totals, whereas a
   normal approximation with independent days under-covers the tail (P98 ->
   93%). "normal" is kept as an option with a day-to-day correlation rho.

2. q* outside the grid. B delivers 0.50..0.98 only. q* is linearly
   interpolated between grid points and clipped to that range; the clip is
   reported back (q_star_raw vs q_star) rather than extrapolated silently.

The same vectorised core (`protection_quantile`, `order_quantity`) is used by
src/inventory.py's simulator, so the simulated policy and the recommendation
the agent shows a store manager are one piece of code.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.inventory_params import (CostParams, FORECAST_HORIZON_DAYS,
                                  family_table)

QUANTILE_LEVELS = np.array([0.50, 0.80, 0.90, 0.95, 0.98])
QCOLS = ["P50", "P80", "P90", "P95", "P98"]
_Z = {0.50: 0.0, 0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.98: 2.0537}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECAST_PATH = ROOT / "runs" / "forecasts" / "latest_quantiles.parquet"
TUNED_Q_STAR_PATH = ROOT / "runs" / "inventory" / "q_star_tuned.csv"


class MissingStockError(ValueError):
    """Stock position not supplied. The agent must ask the user, not guess."""


class HorizonError(ValueError):
    """Planning horizon exceeds what the forecast can cover (16 - lead time)."""


class NoForecastError(LookupError):
    """No forecast for this series/date. Reported as a gap, never estimated."""


# ---------------------------------------------------------------------------
# Vectorised core (shared with the simulator)
# ---------------------------------------------------------------------------
def critical_ratio(cu, co):
    cu, co = np.asarray(cu, float), np.asarray(co, float)
    return cu / (cu + co)


def interpolate_quantile(qvals, q):
    """Value at level q from a (..., 5) array on QUANTILE_LEVELS.

    q may be a scalar or broadcast against qvals[..., 0]. Linear in the level,
    clipped to [0.50, 0.98].
    """
    qvals = np.asarray(qvals, float)
    q = np.clip(np.asarray(q, float), QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1])
    j = np.clip(np.searchsorted(QUANTILE_LEVELS, q, side="right") - 1, 0, len(QUANTILE_LEVELS) - 2)
    lo, hi = QUANTILE_LEVELS[j], QUANTILE_LEVELS[j + 1]
    w = (q - lo) / (hi - lo)
    v_lo = np.take_along_axis(qvals, np.broadcast_to(j, qvals.shape[:-1])[..., None], -1)[..., 0]
    v_hi = np.take_along_axis(qvals, np.broadcast_to(j + 1, qvals.shape[:-1])[..., None], -1)[..., 0]
    return (1 - w) * v_lo + w * v_hi


def protection_quantile(daily_q, q, method="sum", rho=0.3):
    """Quantile q of total demand over k days.

    daily_q : (..., k, 5) daily quantile forecasts over the protection interval
    q       : scalar or (...) service level
    method  : "sum"    -- sum daily quantiles, then interpolate (default)
              "normal" -- per-day mean ~ P50, sd ~ (P90-P50)/z90, days with
                          pairwise correlation rho, normal quantile of the total
    """
    daily_q = np.asarray(daily_q, float)
    if method == "sum":
        return interpolate_quantile(daily_q.sum(axis=-2), q)
    if method == "normal":
        from scipy.stats import norm
        mu = daily_q[..., 0]
        sd = np.maximum(daily_q[..., 2] - daily_q[..., 0], 0) / _Z[0.90]
        var = (sd ** 2).sum(-1) + rho * (sd.sum(-1) ** 2 - (sd ** 2).sum(-1))
        qq = np.clip(np.asarray(q, float), 0.5, 0.999)
        return np.maximum(mu.sum(-1) + norm.ppf(qq) * np.sqrt(var), 0)
    raise ValueError(f"unknown aggregation method {method!r}")


def order_quantity(order_up_to, on_hand, on_order):
    return np.maximum(0.0, np.asarray(order_up_to, float) - on_hand - on_order)


# ---------------------------------------------------------------------------
# Forecast table access (data, not a model)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_forecast(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise NoForecastError(f"forecast file not found: {p}")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df["store_nbr"] = df["store_nbr"].astype(int)
    df["family"] = df["family"].astype(str)
    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def load_tuned_q_star(path=TUNED_Q_STAR_PATH):
    """{family: q*} chosen by the simulation feedback loop, or {} if not built yet."""
    p = Path(path)
    if not p.exists():
        return {}
    t = pd.read_csv(p)
    return dict(zip(t.family, t.q_star_tuned))


def _series_forecast(forecast, store, product):
    f = forecast[(forecast.store_nbr == int(store)) & (forecast.family == str(product))]
    if f.empty:
        raise NoForecastError(f"no forecast for store {store}, family {product!r}")
    return f.set_index("date").sort_index()


def _check_stock(name, value):
    if value is None or (np.isscalar(value) and pd.isna(value)):
        raise MissingStockError(f"{name} not supplied; ask for the current stock position")
    if np.isscalar(value) and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def plan(store, product, H, on_hand, on_order, *, as_of=None, forecast=None,
         cost=CostParams(), q_star=None, aggregation="sum"):
    """Newsvendor order schedule for one store x family over the next H days.

    Parameters
    ----------
    store, product : store_nbr and family
    H              : days to plan; capped at 16 - lead time (raises HorizonError)
    on_hand        : units on the shelf now (required; MissingStockError if None)
    on_order       : units already ordered and not yet received. Either a
                     number (all assumed to land within the first protection
                     interval) or {date: qty} with expected arrival dates.
    as_of          : first planning date; defaults to the forecast's first date
    forecast       : DataFrame with date, store_nbr, family, P50..P98. Defaults
                     to runs/forecasts/latest_quantiles.parquet
    cost           : CostParams (holding rate, penalty, review period)
    q_star         : service level. None -> the family's q* tuned in simulation
                     (runs/inventory/q_star_tuned.csv), falling back to the
                     analytic Cu/(Cu+Co) if that file is absent. A float or a
                     {family: q} mapping overrides it (what-if analysis).
                     The source used is echoed as q_star_source.
    aggregation    : "sum" or "normal" (see protection_quantile)

    Returns a JSON-serialisable dict that echoes every input it used, the
    newsvendor parameters, and one row per day:
        date, arrival_date, position_before, demand_P50, order_up_to,
        order_qty, position_after
    Only day 0 is actionable; later rows assume P50 demand realises and are a
    projection, recomputed each morning in practice.
    """
    _check_stock("on_hand", on_hand)
    _check_stock("on_order", on_order)
    fam = family_table(cost)
    if str(product) not in fam.index:
        raise NoForecastError(f"unknown family {product!r}")
    row = fam.loc[str(product)]
    L, R = int(row.lead_time), cost.review_period
    k = L + R
    max_h = FORECAST_HORIZON_DAYS - L
    if not 1 <= int(H) <= max_h:
        raise HorizonError(f"H={H} outside 1..{max_h} (16 minus lead time {L}) for {product}")
    H = int(H)

    if forecast is None:
        forecast = _load_forecast(str(DEFAULT_FORECAST_PATH))
    f = _series_forecast(forecast, store, product)
    as_of = pd.Timestamp(as_of) if as_of is not None else f.index.min()
    dates = pd.date_range(as_of, periods=H + L, freq="D")
    missing = dates.difference(f.index)
    if len(missing):
        raise NoForecastError(
            f"forecast for store {store} {product} missing {len(missing)} of the "
            f"{len(dates)} dates needed ({missing.min().date()}..{missing.max().date()})")
    fq = f.loc[dates, QCOLS].to_numpy(float)                       # (H+L, 5)
    if np.isnan(fq).any():
        raise NoForecastError(f"forecast for store {store} {product} contains NaN")

    if q_star is None:
        tuned = load_tuned_q_star()
        q_used, q_source = ((tuned[str(product)], "tuned_in_simulation") if str(product) in tuned
                            else (row.q_star, "analytic_cu_co"))
    elif isinstance(q_star, dict):
        if str(product) not in q_star:
            raise NoForecastError(f"q_star mapping has no entry for {product!r}")
        q_used, q_source = q_star[str(product)], "caller_mapping"
    else:
        q_used, q_source = q_star, "caller_override"
    q_used = float(np.clip(q_used, *cost.q_star_bounds))

    if isinstance(on_order, dict):
        arrivals = {pd.Timestamp(d): float(v) for d, v in on_order.items()}
        for v in arrivals.values():
            _check_stock("on_order", v)
    else:
        arrivals = None

    def pipeline_by(day):
        # scalar on_order: assumed to land inside the first protection interval.
        # dated on_order: only what arrives by `day` protects the interval.
        if arrivals is None:
            return float(on_order)
        return sum(v for d, v in arrivals.items() if d <= day)

    stock = float(on_hand)          # projected own stock, excluding the dated pipeline
    schedule = []
    for i in range(H):
        window = fq[i:i + k]                                        # days i .. i+L
        S = float(protection_quantile(window, q_used, aggregation))
        position = stock + pipeline_by(dates[i] + pd.Timedelta(days=L))
        qty = float(order_quantity(S, position, 0.0))
        d50 = float(fq[i, 0])
        schedule.append(dict(
            date=str(dates[i].date()),
            arrival_date=str((dates[i] + pd.Timedelta(days=L)).date()),
            position_before=position,
            demand_P50=d50,
            order_up_to=S,
            order_qty=qty,
            position_after=position + qty - d50,
        ))
        stock = max(0.0, stock + qty - d50)

    return dict(
        store=int(store), product=str(product), as_of=str(as_of.date()), H=H,
        on_hand=float(on_hand),
        on_order=({str(d.date()): v for d, v in arrivals.items()} if arrivals is not None
                  else float(on_order)),
        lead_time=L, review_period=R, protection_days=k,
        unit_cost=float(row.unit_cost), unit_margin=float(row.unit_margin),
        cu=float(row.cu), co=float(row.co),
        q_star_raw=float(row.q_star_raw), q_star_analytic=float(row.q_star),
        q_star=q_used, q_star_source=q_source,
        aggregation=aggregation,
        model_version=(str(f["model_version"].iloc[0]) if "model_version" in f else None),
        recommended_order=schedule[0]["order_qty"],
        schedule=schedule,
    )
