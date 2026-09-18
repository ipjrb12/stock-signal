"""
src/inventory_params.py -- Workstream C assumption table.

Every number the simulator and the newsvendor engine use that is NOT in the
data lives here, in one place, so it can be printed in the report, varied in
the sensitivity analysis, and echoed by the agent tools.

Favorita publishes no prices, costs or supplier terms. Unit prices and gross
margins below are PLACEHOLDER assumptions (rough Ecuadorian grocery retail
levels, USD per sales unit) chosen only to weight families sensibly against
each other. Unit-based metrics (fill rate, stockout days, units held, waste
units) do not depend on them; money-based metrics do.

Lead times, shelf lives, holding rate and stockout penalty follow the plan
(slide 10):
    Lead time, perishables     1 day
    Lead time, grocery         2 to 3 days
    Lead time, hardware        7 days
    Shelf life, perishables    2 to 5 days
    Holding cost               25% per year
    Stockout penalty           1.5 x unit margin
"""
from dataclasses import dataclass, field, replace

import pandas as pd

HOLDING_RATE_PER_YEAR = 0.25
STOCKOUT_PENALTY_MULT = 1.5
REVIEW_PERIOD_DAYS = 1
FORECAST_HORIZON_DAYS = 16
Q_STAR_BOUNDS = (0.50, 0.98)   # the quantile grid Workstream B delivers

# family: (group, lead_time_days, shelf_life_days or 0 = non-perishable, unit_price, gross_margin)
_FAMILY_TABLE = {
    # perishables -- lead time 1 day, shelf life 2-5 days
    "BREAD/BAKERY":               ("perishable", 1, 2, 1.50, 0.35),
    "PREPARED FOODS":             ("perishable", 1, 2, 5.00, 0.40),
    "SEAFOOD":                    ("perishable", 1, 2, 7.00, 0.25),
    "DELI":                       ("perishable", 1, 3, 4.00, 0.30),
    "MEATS":                      ("perishable", 1, 3, 5.00, 0.20),
    "POULTRY":                    ("perishable", 1, 3, 4.00, 0.18),
    "PRODUCE":                    ("perishable", 1, 4, 1.20, 0.30),
    "DAIRY":                      ("perishable", 1, 5, 1.80, 0.20),
    "EGGS":                       ("perishable", 1, 5, 3.00, 0.15),
    # grocery, fast-moving -- lead time 2 days
    "GROCERY I":                  ("grocery", 2, 0, 2.00, 0.20),
    "BEVERAGES":                  ("grocery", 2, 0, 1.50, 0.20),
    "CLEANING":                   ("grocery", 2, 0, 3.00, 0.25),
    "HOME CARE":                  ("grocery", 2, 0, 4.00, 0.25),
    "PERSONAL CARE":              ("grocery", 2, 0, 4.00, 0.30),
    "FROZEN FOODS":               ("grocery", 2, 0, 4.00, 0.25),
    "LIQUOR,WINE,BEER":           ("grocery", 2, 0, 8.00, 0.25),
    # grocery, slower-moving -- lead time 3 days
    "GROCERY II":                 ("grocery", 3, 0, 3.00, 0.22),
    "BABY CARE":                  ("grocery", 3, 0, 6.00, 0.25),
    "BEAUTY":                     ("grocery", 3, 0, 5.00, 0.35),
    "PET SUPPLIES":               ("grocery", 3, 0, 6.00, 0.30),
    "CELEBRATION":                ("grocery", 3, 0, 4.00, 0.35),
    "MAGAZINES":                  ("grocery", 3, 0, 3.00, 0.30),
    "BOOKS":                      ("grocery", 3, 0, 10.00, 0.30),
    "SCHOOL AND OFFICE SUPPLIES": ("grocery", 3, 0, 3.00, 0.40),
    # hardware and general merchandise -- lead time 7 days
    "HARDWARE":                   ("hardware", 7, 0, 10.00, 0.35),
    "AUTOMOTIVE":                 ("hardware", 7, 0, 8.00, 0.30),
    "HOME APPLIANCES":            ("hardware", 7, 0, 40.00, 0.25),
    "HOME AND KITCHEN I":         ("hardware", 7, 0, 12.00, 0.35),
    "HOME AND KITCHEN II":        ("hardware", 7, 0, 12.00, 0.35),
    "LAWN AND GARDEN":            ("hardware", 7, 0, 8.00, 0.35),
    "PLAYERS AND ELECTRONICS":    ("hardware", 7, 0, 50.00, 0.20),
    "LADIESWEAR":                 ("hardware", 7, 0, 15.00, 0.45),
    "LINGERIE":                   ("hardware", 7, 0, 10.00, 0.45),
}


@dataclass(frozen=True)
class CostParams:
    """Global economic parameters (the family table carries the per-item ones)."""
    holding_rate_per_year: float = HOLDING_RATE_PER_YEAR
    stockout_penalty_mult: float = STOCKOUT_PENALTY_MULT
    review_period: int = REVIEW_PERIOD_DAYS
    q_star_bounds: tuple = Q_STAR_BOUNDS
    # multiplicative tweaks used only by the sensitivity analysis
    lead_time_add: dict = field(default_factory=dict)     # group -> +days
    shelf_life_add: int = 0                               # +days for perishables

    def with_(self, **kw):
        return replace(self, **kw)


def family_table(cost: CostParams = CostParams()) -> pd.DataFrame:
    """One row per family with lead time, shelf life, unit economics, Cu, Co and q*.

    Newsvendor over the protection interval L+R:
        Cu = stockout_penalty_mult * unit_margin
        Co = unit_cost * holding_rate/365 * (L+R)                  (all items)
           + unit_cost * min(1, (L+R) / shelf_life)                (perishables)
        q* = Cu / (Cu + Co), clipped to the delivered quantile grid.

    The perishable term is the expected write-off of a leftover unit: a unit
    left at the end of the interval has used (L+R)/shelf_life of its life, so a
    2-day bread order with a 2-day shelf life is a pure newsvendor (Co = cost)
    while dairy (5 days) can roll most leftovers into the next cycle.
    """
    rows = []
    for fam, (group, lt, shelf, price, gm) in _FAMILY_TABLE.items():
        lt = lt + cost.lead_time_add.get(group, 0)
        if shelf:
            shelf = max(1, shelf + cost.shelf_life_add)
        cycle = lt + cost.review_period
        unit_cost = price * (1 - gm)
        unit_margin = price * gm
        cu = cost.stockout_penalty_mult * unit_margin
        co = unit_cost * cost.holding_rate_per_year / 365 * cycle
        if shelf:
            co += unit_cost * min(1.0, cycle / shelf)
        q_raw = cu / (cu + co)
        lo, hi = cost.q_star_bounds
        rows.append(dict(
            family=fam, group=group, lead_time=lt, shelf_life=shelf,
            unit_price=price, gross_margin=gm, unit_cost=unit_cost,
            unit_margin=unit_margin, cu=cu, co=co, q_star_raw=q_raw,
            q_star=min(max(q_raw, lo), hi),
        ))
    return pd.DataFrame(rows).set_index("family")
