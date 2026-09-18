"""Forecast accuracy: model choice and where the error sits."""
import streamlit as st

import figures as fg
import lib


def render():
    K, bt = lib.kpi(), lib.csv("backtest_results.csv")
    lib.tiles([
        {"label": "HGB mean RMSLE", "value": f"{bt.hgb_rmsle.mean():.3f}",
         "note": f"spread ±{bt.hgb_rmsle.std():.3f}", "tone": "good"},
        {"label": "Seasonal naive", "value": f"{bt.naive_rmsle.mean():.3f}",
         "note": "baseline to beat"},
        {"label": "Gain, final fold", "value": f"{K['holdout_gain_vs_naive']:.0%}",
         "note": "target was 15%", "tone": "good"},
        {"label": "Features", "value": "77", "note": "71 numeric, 6 categorical"},
        {"label": "Leaky columns cut", "value": "9",
         "note": "same-day weather, oil, transactions", "tone": "warn"},
    ])

    a, b, c = st.columns([1.3, 1, 1])
    with a:
        lib.chart(fg.leaderboard(), "Model leaderboard",
                  "RMSLE per rolling-origin fold, five models", height=300, legend=True)
    with b:
        lib.chart(fg.importance(), "What the model leans on",
                  "Permutation importance, top 12", height=300)
    with c:
        lib.chart(fg.rmsle_by_horizon(), "Error by distance",
                  "RMSLE against days after the origin", height=300)

    d, e, f = st.columns([1.3, 1, 1])
    with d:
        lib.chart(fg.pred_vs_actual(), "Predicted against actual",
                  "3,500 sampled store-family-days, held out", height=300)
    with e:
        lib.chart(fg.wape_by_family(), "Where error concentrates",
                  "WAPE per family, dot size is volume", height=300)
    with f:
        lib.chart(fg.wape_by_horizon(), "Error by distance, in volume terms",
                  "WAPE against days after the origin", height=300)
