"""Calibration: does a P90 forecast cover 90% of days."""
import streamlit as st

import figures as fg
import lib


def render():
    K = lib.kpi()
    lib.tiles([
        {"label": "P90 before", "value": f"{K['p90_before']:.1%}", "note": "raw quantile",
         "tone": "warn"},
        {"label": "P90 after", "value": f"{K['p90_after']:.1%}",
         "note": "conformal correction applied", "tone": "warn"},
        {"label": "Gate", "value": "88 to 92%", "note": "not met", "tone": "warn"},
        {"label": "Crossings fixed", "value": "13.2%", "note": "none left in shipped files"},
        {"label": "Rows checked", "value": "27,664", "note": "held-out window with actuals"},
    ])
    st.markdown(
        f"<div class='flag'><b>Gate not met, and the number moved.</b> The modelling write-up "
        f"quotes 86.1% rising to 89.8%. Recomputed from the quantile files that actually shipped, "
        f"the correction takes P90 from {K['p90_before']:.1%} to {K['p90_after']:.1%}. Every "
        f"figure here is measured from those files rather than copied from a report.</div>",
        unsafe_allow_html=True)

    a, b, c = st.columns([1, 1, 1.05])
    with a:
        lib.chart(fg.coverage_dumbbell(), "Promised against delivered",
                  "Empirical coverage per quantile, held-out window", height=290, legend=True)
        lib.chart(fg.interval_width(), "Interval width",
                  "Median quantile as a multiple of the P50, by group", height=290, legend=True)
    with b:
        lib.chart(fg.rolling_coverage(), "Coverage across cutoffs",
                  "Three model cutoffs, dotted lines are the targets", height=290, legend=True)
        st.container(border=True).markdown(
            "<div class='ch'><span class='ct'>What the correction does</span>"
            "<span class='cs'>Per-family offset, estimated on one fortnight</span></div>"
            "<ul class='guards'><li>A second model trains on data through 07-14 only.</li>"
            "<li>Its residuals are measured per family on 07-15 to 07-30.</li>"
            "<li>The 90th percentile residual is added back as that family's offset.</li>"
            "<li>Families already inside the band barely move; the ones far short move most.</li>"
            "</ul>", unsafe_allow_html=True)
    with c:
        lib.chart(fg.p90_before_after(), "Conformal correction, per family",
                  "Raw against corrected P90, shaded band is the gate", height=620, legend=True)
