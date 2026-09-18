"""
src/inventory.py -- Workstream C deliverable: inventory simulator.

Interface (frozen, slide 12):
    simulate(demand, policy, params) -> Ledger
        Daily loop: receive, sell, expire, reorder.

All 1,729 store x family series are simulated at once as numpy arrays.

Daily cycle for day t (the 05:00 run of slide 10 places the order before the
store opens, so "reorder" sits before "sell" in code; it is the same decision
as an end-of-day order on t-1):

    1. receive  pipeline units due on t go on the shelf (fresh batch)
    2. reorder  policy sees on_hand + on_order and the history it is allowed
                to see; the order arrives at the start of day t + L
    3. sell     sales = min(demand, on_hand), oldest batch first (FIFO);
                unmet demand is LOST, not backordered
    4. expire   perishable units on their last sellable day are written off

With review period R = 1, an order placed on t must cover days t .. t+L, i.e.
L + R days, which is exactly the protection interval policy_engine uses.

Shelf life s means a batch received on day a can be sold on days a .. a+s-1.
s = 0 marks a non-perishable item.

Costs per series-day:
    holding  = closing on_hand * unit_cost * holding_rate / 365
    stockout = lost units * stockout_penalty_mult * unit_margin
    waste    = expired units * unit_cost

Known bias, stated plainly: `demand` is recorded sales, which is itself
censored on historical stockout days. True demand was higher on those days, so
every policy's simulated fill rate is optimistic by the same mechanism.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.inventory_params import CostParams, family_table


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class DemandPanel:
    """Demand as a dense (n_series, n_days) matrix. NaN = series not active."""
    keys: pd.DataFrame          # store_nbr, family (row order of the matrix)
    dates: pd.DatetimeIndex
    values: np.ndarray

    @classmethod
    def from_long(cls, df, value_col="sales_units"):
        wide = (df.assign(date=pd.to_datetime(df["date"]), store_nbr=df["store_nbr"].astype(int),
                          family=df["family"].astype(str))
                  .pivot_table(index=["store_nbr", "family"], columns="date",
                               values=value_col, aggfunc="sum", dropna=False))
        dates = pd.date_range(wide.columns.min(), wide.columns.max(), freq="D")
        wide = wide.reindex(columns=dates)
        wide = wide[wide.notna().any(axis=1)]     # pivot re-creates never-selling pairs
        return cls(wide.index.to_frame(index=False), dates, wide.to_numpy(float))

    def day(self, date):
        return int(self.dates.get_loc(pd.Timestamp(date)))


@dataclass
class SimParams:
    start: str                       # first simulated day
    end: str                         # last simulated day
    score_start: str                 # metrics exclude the warm-up before this
    cost: CostParams = CostParams()
    initial_cover_days: int = 28     # history used to size the opening stock

    def series_table(self, keys):
        fam = family_table(self.cost)
        return fam.loc[keys["family"].values].reset_index()


class SimState:
    """What a policy may look at on day t (read-only by convention)."""

    def __init__(self, t, date, on_hand, on_order, lead_time, series):
        self.t, self.date = t, date
        self.on_hand, self.on_order = on_hand, on_order
        self.lead_time, self.series = lead_time, series


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
LEDGER_COLS = ["on_hand_open", "received", "order_up_to", "order_qty", "demand",
               "sold", "lost", "expired", "on_hand_close", "on_order_close",
               "holding_cost", "stockout_cost", "waste_cost"]


@dataclass
class Ledger:
    policy: str
    keys: pd.DataFrame
    dates: pd.DatetimeIndex
    arrays: dict                 # column -> (n_series, n_days)
    active: np.ndarray           # (n_series, n_days) bool, series has demand data
    scored: np.ndarray           # (n_days,) bool, inside the scoring window
    series: pd.DataFrame         # per-series parameters used

    def to_frame(self, scored_only=False):
        n, T = self.active.shape
        out = pd.DataFrame({
            "policy": self.policy,
            "date": np.tile(self.dates.values, n),
            "store_nbr": np.repeat(self.keys.store_nbr.values, T),
            "family": np.repeat(self.keys.family.values, T),
            "group": np.repeat(self.series.group.values, T),
        })
        for c in LEDGER_COLS:
            out[c] = self.arrays[c].reshape(-1)
        out["total_cost"] = out.holding_cost + out.stockout_cost + out.waste_cost
        out["scored"] = np.tile(self.scored, n)
        mask = self.active.reshape(-1).copy()   # copy: `&=` below must not mutate self.active
        if scored_only:
            mask &= out.scored.values
        return out[mask].reset_index(drop=True)

    def summary(self, by=None, window=None):
        """Metrics over the scoring window (or window=(first, last) dates):
        fill rate, stockout days, units held, waste, cost."""
        if window is None:
            df = self.to_frame(scored_only=True)
            n_days = int(self.scored.sum())
        else:
            lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
            df = self.to_frame()
            df = df[(df.date >= lo) & (df.date <= hi)]
            n_days = int(((self.dates >= lo) & (self.dates <= hi)).sum())
        g = df.groupby(by) if by else [((), df)]
        rows = []
        for k, x in g:
            demand_days = (x.demand > 0).sum()
            so_days = ((x.lost > 1e-9)).sum()
            rows.append({
                **({by: k} if by else {}),
                "policy": self.policy,
                "fill_rate": x.sold.sum() / x.demand.sum(),
                "stockout_days": int(so_days),
                "stockout_day_rate": so_days / max(demand_days, 1),
                "mean_units_held": x.on_hand_close.sum() / n_days,
                "units_sold": x.sold.sum(),
                "units_lost": x.lost.sum(),
                "waste_units": x.expired.sum(),
                "waste_rate": x.expired.sum() / max(x.sold.sum(), 1e-9),
                "holding_cost": x.holding_cost.sum(),
                "stockout_cost": x.stockout_cost.sum(),
                "waste_cost": x.waste_cost.sum(),
                "total_cost": x.total_cost.sum(),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def simulate(demand: DemandPanel, policy, params: SimParams) -> Ledger:
    """Run `policy` against `demand` over params.start..params.end.

    policy: object with `name` and `order_up_to(state) -> (n,) array`, the
            target inventory position. The simulator turns it into an order
            with policy_engine.order_quantity, so every policy is compared
            through the same order-up-to mechanics.
    """
    from src.policy_engine import order_quantity

    series = params.series_table(demand.keys)
    n = len(series)
    lead = series.lead_time.to_numpy(int)
    shelf = series.shelf_life.to_numpy(int)
    unit_cost = series.unit_cost.to_numpy(float)
    unit_margin = series.unit_margin.to_numpy(float)
    c = params.cost
    R = c.review_period

    t0, t1 = demand.day(params.start), demand.day(params.end)
    T = t1 - t0 + 1
    dates = demand.dates[t0:t1 + 1]
    D = demand.values[:, t0:t1 + 1]
    active = ~np.isnan(D)
    D = np.nan_to_num(D)
    rows = np.arange(n)

    # inventory by remaining shelf life: column r (1..smax) = r sellable days
    # left including today; last column = non-perishable stock (never ages)
    smax = max(int(shelf.max()), 1)
    NP = smax + 1
    inv = np.zeros((n, smax + 2))
    recv_col = np.where(shelf > 0, shelf, NP)

    # pipeline: column j = arrives j days from today
    pipe = np.zeros((n, int(lead.max()) + 1))

    # opening position: recent mean daily demand * (L+R) on the shelf, plus one
    # day's worth already in transit for each of days 1..L-1. Identical for all
    # policies; the warm-up period absorbs it before scoring starts.
    hist = demand.values[:, max(0, t0 - params.initial_cover_days):t0]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # series with no history yet
        mean_d = np.nan_to_num(np.nanmean(hist, axis=1)) if hist.shape[1] else np.zeros(n)
    inv[rows, recv_col] = mean_d * (lead + R)
    for j in range(1, pipe.shape[1]):
        pipe[:, j] = np.where(j < lead, mean_d, 0.0)

    A = {k: np.zeros((n, T)) for k in LEDGER_COLS}
    for i in range(T):
        t = t0 + i
        # 1. receive
        received = pipe[:, 0].copy()
        pipe[:, :-1] = pipe[:, 1:]
        pipe[:, -1] = 0.0
        inv[rows, recv_col] += received
        on_hand_open = inv.sum(1)

        # 2. reorder (05:00, before trading)
        state = SimState(t, demand.dates[t], on_hand_open, pipe.sum(1), lead, series)
        S = np.maximum(np.asarray(policy.order_up_to(state), float), 0.0)
        q = order_quantity(S, on_hand_open, pipe.sum(1))
        q = np.where(active[:, i], q, 0.0)
        # lead time L: arrives L days from today (after today's shift it is col L-1)
        pipe[rows, lead - 1] += q

        # 3. sell, oldest batch first
        need = D[:, i].copy()
        sold = np.zeros(n)
        for col in list(range(1, smax + 1)) + [NP]:
            take = np.minimum(inv[:, col], need)
            inv[:, col] -= take
            need -= take
            sold += take
        lost = need

        # 4. expire: last-day perishable units, then everything ages one day
        expired = inv[:, 1].copy()
        inv[:, 1:smax] = inv[:, 2:smax + 1]
        inv[:, smax] = 0.0
        on_hand_close = inv.sum(1)

        A["on_hand_open"][:, i] = on_hand_open
        A["received"][:, i] = received
        A["order_up_to"][:, i] = S
        A["order_qty"][:, i] = q
        A["demand"][:, i] = D[:, i]
        A["sold"][:, i] = sold
        A["lost"][:, i] = lost
        A["expired"][:, i] = expired
        A["on_hand_close"][:, i] = on_hand_close
        A["on_order_close"][:, i] = pipe.sum(1)
        A["holding_cost"][:, i] = on_hand_close * unit_cost * c.holding_rate_per_year / 365
        A["stockout_cost"][:, i] = lost * c.stockout_penalty_mult * unit_margin
        A["waste_cost"][:, i] = expired * unit_cost

    scored = np.asarray(dates >= pd.Timestamp(params.score_start))
    return Ledger(policy.name, demand.keys, dates, A, active, scored, series)
