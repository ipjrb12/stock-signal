"""Recommendation engine: the deterministic layer the ordering agent calls."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import figures as fg
import lib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from src.policy_engine import (HorizonError, MissingStockError,  # noqa: E402
                               NoForecastError, plan)

GUARDS = """<ul class='guards'>
<li><b>Plan past the forecast.</b> The horizon is capped at 16 minus the lead time; beyond it the
engine raises <code>HorizonError</code> instead of extrapolating.</li>
<li><b>Guess the stock position.</b> No on_hand, no plan: <code>MissingStockError</code>. A stock
figure is the one input no model can supply.</li>
<li><b>Fill a forecast gap.</b> A missing or NaN quantile raises <code>NoForecastError</code> and
is reported as a gap, never interpolated.</li>
<li><b>Write anything.</b> The function is pure: same inputs, same order, every time. Placing the
order stays with a person.</li></ul>"""


@st.cache_data(show_spinner=False)
def _forecast():
    f = lib.parquet("forecast_test.parquet").copy()
    f["date"] = pd.to_datetime(f["date"])
    return f


@st.cache_data(show_spinner=False)
def _plan(store, family, horizon, on_hand, on_order):
    return plan(store, family, horizon, on_hand=on_hand, on_order=on_order, forecast=_forecast())


@st.cache_data(show_spinner=True)
def _store_brief(store, horizon):
    state = lib.parquet("closing_state.parquet")
    st_state = state[(state.policy == fg.PROPOSAL) & (state.store_nbr == store)]
    rows = []
    for _, r in st_state.iterrows():
        try:
            p = _plan(int(store), str(r.family), int(horizon), float(r.on_hand),
                      float(r.on_order))
        except (HorizonError, NoForecastError, MissingStockError) as exc:
            rows.append(dict(family=r.family, group=r["group"], status=type(exc).__name__,
                             on_hand=r.on_hand, on_order=r.on_order, order_qty=np.nan,
                             order_up_to=np.nan, demand_P50=np.nan))
            continue
        d0 = p["schedule"][0]
        rows.append(dict(family=r.family, group=r["group"], status="ok", on_hand=r.on_hand,
                         on_order=r.on_order, order_qty=d0["order_qty"],
                         order_up_to=d0["order_up_to"], demand_P50=d0["demand_P50"]))
    return pd.DataFrame(rows)


def render():
    meta = lib.csv("series_meta.csv")
    state = lib.parquet("closing_state.parquet")
    state = state[state.policy == fg.PROPOSAL]
    lead = lib.csv("assumptions_family_table.csv").set_index("family").lead_time

    a, b, c, d = st.columns([1, 1.2, 1, 1])
    store = a.selectbox("Store", sorted(meta.store_nbr.unique()),
                        format_func=lambda s: f"Store {s}")
    brief = _store_brief(store, 7)
    fams = sorted(brief.family.unique())
    default = fams.index("GROCERY I") if "GROCERY I" in fams else 0
    family = b.selectbox("Product family", fams, index=default, format_func=str.title)
    horizon = c.number_input("Planning horizon (days)", 1, 14, 7, step=1)
    row = state[(state.store_nbr == store) & (state.family == family)]
    cur_hand = float(row.on_hand.iloc[0]) if len(row) else 0.0
    cur_order = float(row.on_order.iloc[0]) if len(row) else 0.0
    on_hand = d.number_input("On hand", min_value=0.0, value=round(cur_hand, 1), step=1.0)

    brief = _store_brief(store, int(horizon))
    ok = brief[brief.status == "ok"].copy()
    refused = brief[brief.status == "HorizonError"]
    if ok.empty:
        st.warning("The engine returned no plan for this store at this horizon.")
        return

    try:
        p = _plan(int(store), str(family), int(horizon), float(on_hand), float(cur_order))
    except HorizonError as exc:
        st.markdown(f"<div class='flag'><b>The engine refused this horizon.</b> {exc} "
                    "This is the guardrail working: a plan longer than the forecast can support "
                    "is refused rather than extrapolated.</div>", unsafe_allow_html=True)
        p = None

    if p:
        sched = pd.DataFrame(p["schedule"])
        lib.tiles([
            {"label": "Order today", "value": f"{p['recommended_order']:,.0f}",
             "note": f"units of {family.title()}, arriving {sched.arrival_date.iloc[0]}",
             "tone": "good"},
            {"label": "Order-up-to", "value": f"{sched.order_up_to.iloc[0]:,.0f}",
             "note": f"{p['protection_days']} days of cover at q*={p['q_star']:.2f}"},
            {"label": "Position before", "value": f"{sched.position_before.iloc[0]:,.0f}",
             "note": f"{on_hand:,.0f} on hand + {cur_order:,.0f} on order"},
            {"label": "Median demand", "value": f"{sched.demand_P50.iloc[0]:,.0f}/day",
             "note": f"lead time {p['lead_time']}d · review {p['review_period']}d"},
            {"label": "Lines to order", "value": f"{int((ok.order_qty > 0).sum())}",
             "note": f"of {len(ok)} the engine would plan"},
        ])
    if len(refused):
        longest = int(refused.merge(lib.csv("assumptions_family_table.csv")[["family",
                                                                            "lead_time"]],
                                    on="family", how="left").lead_time.max())
        st.markdown(
            f"<div class='flag'><b>{len(refused)} lines refused at {horizon} days.</b> Their lead "
            f"times run up to {longest} days and the forecast only reaches 16, so the engine will "
            f"not plan that far for them. It raises HorizonError rather than extrapolating, and "
            f"they are left off the list instead of being shown with a made-up number. Drop the "
            f"horizon to {16 - longest} to bring them back.</div>", unsafe_allow_html=True)

    top = ok.sort_values("order_qty", ascending=False).head(12)
    left, right = st.columns([1.9, 1])
    with left:
        lib.chart(fg.engine_bars(top), "Today's order list",
                  f"Store {store}, {horizon}-day plan: twelve largest lines, live call",
                  height=330, legend=True)
        if p:
            lib.chart(fg.engine_schedule(sched), "One line, day by day",
                      f"Store {store} {family.title()}: orders, position and cover",
                      height=300, legend=True)
    with right:
        if p:
            box = lib.panel("What the answer rests on",
                            "Every input the engine returned with the order")
            rows = [("Store", f"{store}"), ("Product", family.title()),
                    ("Unit cost", f"${p['unit_cost']:.2f}"),
                    ("Unit margin", f"${p['unit_margin']:.2f}"),
                    ("Cost of being short, Cu", f"${p['cu']:.3f}"),
                    ("Cost of holding, Co", f"${p['co']:.4f}"),
                    ("Critical ratio", f"{p['q_star_raw']:.3f}"),
                    ("Analytic q*", f"{p['q_star_analytic']:.2f}"),
                    ("q* used", f"{p['q_star']:.2f}"),
                    ("Source of q*", p["q_star_source"].replace("_", " ")),
                    ("Aggregation", p["aggregation"]),
                    ("Protection interval", f"{p['protection_days']} days"),
                    ("Model version", (p["model_version"] or "n/a")
                     .replace("hgb_quantile_v2_cutoff_", "v2 · cutoff "))]
            box.markdown("<table class='tbl'><tbody>" + "".join(
                f"<tr><td class='l'>{k}</td><td class='s'>{v}</td></tr>" for k, v in rows)
                + "</tbody></table>", unsafe_allow_html=True)
        guard = lib.panel("What the engine refuses to do",
                          "Guardrails that fire before any number reaches a screen")
        guard.markdown(GUARDS, unsafe_allow_html=True)
