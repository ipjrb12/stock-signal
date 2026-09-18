"""
Stock Signal: demand forecasting and inventory dashboard.
Group 5 capstone, workstream E.

    streamlit run app.py
"""
import streamlit as st

import figures as fg  # noqa: F401  (registers the dark plotly template on import)
import lib
from sections import calibration, engine, explorer, forecast, overview, policy

st.set_page_config(page_title="Stock Signal", page_icon="📦", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(lib.CSS, unsafe_allow_html=True)

K = lib.kpi()

PAGES = {
    "Overview": (overview.render, ["valid"],
                 "Headline service and cost figures, where the money goes under each ordering "
                 "policy, and the demand the forecast has to track."),
    "Forecast accuracy": (forecast.render, ["train", "valid"],
                          "Model bake-off across rolling origins, feature importance, error by "
                          "distance and by family, and forecasts against outcomes."),
    "Calibration": (calibration.render, ["valid"],
                    "Whether a P90 forecast really covers 90% of days, which is what the policy "
                    "assumes."),
    "Series explorer": (explorer.render, ["sim", "valid", "live"],
                        "One store and product at a time: forecast, outcome and the stock "
                        "underneath. All 1,729 series."),
    "Inventory policy": (policy.render, ["sim", "valid"],
                         "Five ordering rules on one simulated ledger. P3b, with the service "
                         "level tuned per family, is the proposal; P3 is the diagnostic that "
                         "exposed the perishable trap."),
    "Recommendation engine": (engine.render, ["live"],
                              "The deterministic engine behind the ordering agent: what it "
                              "recommends today, what each figure rests on, and what it refuses "
                              "to answer."),
}

meta = (f"{K['series']:,} series · {K['stores']} stores · {K['families']} families · "
        f"{K['horizon_days']}-day quantile forecast · simulation {K['sim_window']} · "
        f"built {K['generated']}")
st.markdown(f"<div class='brandrow'><div class='brand'>Stock <span>Signal</span></div>"
            f"<div class='hmeta'>{meta}</div></div>", unsafe_allow_html=True)

nav, chips = st.columns([2.1, 1])
with nav:
    page = st.radio("Section", list(PAGES), horizontal=True, label_visibility="collapsed")
render, active, lede = PAGES[page]
chips.markdown(lib.timeframes(active), unsafe_allow_html=True)
st.markdown(f"<div class='lede'>{lede}</div>", unsafe_allow_html=True)

render()
