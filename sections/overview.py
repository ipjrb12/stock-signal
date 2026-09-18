"""Overview: the headline figures and the shape of the demand behind them."""
import streamlit as st

import figures as fg
import lib


def render():
    K = lib.kpi()
    lib.tiles([
        {"label": "Fill rate", "value": f"{K['best_fill']:.1%}",
         "note": f"vs {K['base_fill']:.1%} today", "tone": "good"},
        {"label": "Inventory cost", "value": lib.money(K["best_cost"]),
         "note": f"{K['cost_saving']:.0%} below {lib.money(K['base_cost'])}", "tone": "good"},
        {"label": "Stockout days", "value": f"{K['best_stockout_days']:,}",
         "note": f"down from {K['base_stockout_days']:,}", "tone": "good"},
        {"label": "Forecast RMSLE", "value": f"{K['hgb_rmsle_mean']:.3f}",
         "note": f"{(K['naive_rmsle_mean']-K['hgb_rmsle_mean'])/K['naive_rmsle_mean']:.0%} under "
                 f"seasonal naive", "tone": "good"},
        {"label": "P90 coverage", "value": f"{K['p90_after']:.1%}",
         "note": "gate is 88 to 92%", "tone": "warn"},
    ])

    a, b = st.columns([1.1, 1])
    with a:
        lib.chart(fg.cost_by_policy(), "Cost by policy",
                  "Holding, spoilage and lost margin, held-out window · P3b is the proposal",
                  height=330, legend=True)
    with b:
        lib.chart(fg.service_vs_cost(), "Service against cost",
                  "Up and to the left is better · bubble area is average units held", height=330)

    lib.chart(fg.demand_timeline(), "Demand the model tracks",
              "Panel-wide units a day, 2016 to the last day with recorded sales", height=250)
