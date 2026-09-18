"""Inventory policy: the five-way comparison, with P3b as the proposal."""
import streamlit as st

import figures as fg
import lib

ROLE = {fg.PROPOSAL: "<span class='tag hot'>proposed</span>",
        "P3_newsvendor_q*": "<span class='tag'>diagnostic</span>",
        "P1_moving_avg_7d": "<span class='tag'>today</span>",
        "P5_perfect_foresight": "<span class='tag'>bound</span>"}
HEAD = ["Policy", "Fill", "Stockout days", "Units held", "Spoiled", "Total cost", "vs P1"]


def render():
    left, right = st.columns([1, 2.4])
    window = left.selectbox("Scoring window",
                            ["Held out · 07-31 to 08-15", "Full simulation · 07-15 to 08-15"])
    key = "holdout" if window.startswith("Held") else "full"
    pc = fg.comparison(key)
    base = pc[pc.policy == "P1_moving_avg_7d"].iloc[0]
    prop = pc[pc.policy == fg.PROPOSAL]
    if len(prop):
        lead, label, note = prop.iloc[0], "Proposal", "q* tuned per family"
    else:
        real = pc[pc.policy != "P5_perfect_foresight"]
        lead = real.loc[real.total_cost.idxmin()]
        label, note = "Cheapest here", "P3b was not run on this window"

    lib.tiles([
        {"label": label, "value": lib.POLICY_LABEL[lead.policy].split(" · ")[0], "note": note},
        {"label": "Fill rate", "value": f"{lead.fill_rate:.1%}",
         "note": f"{(lead.fill_rate-base.fill_rate)*100:+.1f} points vs P1", "tone": "good"},
        {"label": "Total cost", "value": lib.money(lead.total_cost),
         "note": f"{(lead.total_cost/base.total_cost-1):.0%} vs P1", "tone": "good"},
        {"label": "Units held", "value": lib.units(lead.mean_units_held),
         "note": f"{lead.mean_units_held/base.mean_units_held:.1f}x today", "tone": "warn"},
        {"label": "Units spoiled", "value": f"{lead.waste_units:,.0f}",
         "note": f"{lead.waste_rate:.2%} of units handled", "tone": "warn"},
    ])

    rows = ""
    for _, r in pc.iterrows():
        hot = " class='hot'" if r.policy == fg.PROPOSAL else ""
        rows += (f"<tr{hot}><td class='l'>{lib.POLICY_LABEL[r.policy]}"
                 f"{ROLE.get(r.policy, '')}</td><td>{r.fill_rate:.1%}</td>"
                 f"<td>{int(r.stockout_days):,}</td><td>{r.mean_units_held:,.0f}</td>"
                 f"<td>{r.waste_units:,.0f}</td><td class='s'>${r.total_cost:,.0f}</td>"
                 f"<td>{(r.total_cost/base.total_cost-1):.0%}</td></tr>")
    sub = ("P3b is the proposal, P3 the diagnostic that exposed the perishable trap"
           if len(prop) else "P3b was only run on the held-out window, so it is absent here")

    a, b = st.columns([1.35, 1])
    with a:
        box = lib.panel("Five policies, one ledger", sub)
        box.markdown("<table class='tbl'><thead><tr>"
                     + "".join(f"<th>{c}</th>" for c in HEAD)
                     + "</tr></thead><tbody>" + rows + "</tbody></table>",
                     unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            lib.chart(fg.q_sweep(), "Service level against cost",
                      "One q for every family, full simulation window", height=290)
        with c2:
            lib.chart(fg.sensitivity(), "Robustness",
                      "P3 against P1 under nine cost and logistics variants", height=290)
    with b:
        lib.chart(fg.q_star_slope(), "Why P3 became P3b",
                  "The textbook ratio pins perishables at the floor; simulation moves 18 of 33",
                  height=620, legend=True)
