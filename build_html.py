"""
Build the single-file dashboard: dark theme, one screen per section, no scrolling.

Charts come from figures.py, the same module the Streamlit app uses, so the two
front ends cannot drift apart. What lives here is the page: the tile grid, the
filter bar, the dropdowns that stand in for Streamlit widgets, and the copy.

    python build_html.py   ->  stock-signal-preview.html   (open in any browser)
                               stock-signal-artifact.html  (same page, for hosting)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as po

import figures as fg
import lib

HERE = Path(__file__).resolve().parent
OUT = HERE / "stock-signal-preview.html"
ART = HERE / "stock-signal-artifact.html"
K = lib.kpi()

TEAL, AMBER, BLUE, RED, PURPLE = fg.TEAL, fg.AMBER, fg.BLUE, fg.RED, fg.PURPLE
PAGE, PANEL, LINE = fg.PAGE, fg.PANEL, fg.LINE
INK, MUTED, DIM, GRID = fg.INK, fg.MUTED, fg.DIM, fg.GRID
GREY, PALE = fg.GREY, fg.PALE
POLICY_ORDER, PL, PC, GC = fg.POLICY_ORDER, fg.PL, fg.PC, fg.GC

_n = [0]
_ids: list[str] = []
LINKS: list[list] = []


def plot(fig, legend=False, m=None):
    """Return a chart div that fills whatever tile it is dropped into."""
    fig.update_layout(autosize=True)
    if m:
        fig.update_layout(margin=m)
    else:
        fig.update_layout(margin=dict(t=26 if legend else 10))
    if legend:
        fig.update_layout(legend=dict(y=1.02))
    _n[0] += 1
    _ids.append(f"p{_n[0]}")
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"p{_n[0]}",
                       default_width="100%", default_height="100%",
                       config={"displayModeBar": False, "responsive": True})


def card(title, sub, body, cls="", cid=None):
    sid = f" id='{cid}-sub'" if cid else ""
    return (f"<div class='card {cls}'><div class='ch'><span class='ct'>{title}</span>"
            f"<span class='cs'{sid}>{sub}</span></div><div class='plot'>{body}</div></div>")


def kpi(items):
    out = "<div class='kpis'>"
    for it in items:
        tone = it.get("tone", "flat")
        out += (f"<div class='kpi'><span class='kl'>{it['label']}</span>"
                f"<span class='kv'>{it['value']}</span>"
                f"<span class='kn {tone}'>{it['note']}</span></div>")
    return out + "</div>"


def hide_menu(fig, spans, labels=None):
    """A hidden updatemenu whose buttons the page's own <select> drives."""
    btns = []
    for i, (a, b) in enumerate(spans):
        vis = [False] * len(fig.data)
        for j in range(a, b):
            vis[j] = True
        btns.append(dict(label=(labels[i] if labels else str(i)), method="update",
                         args=[{"visible": vis}]))
    for j in range(*spans[0]):
        fig.data[j].visible = True
    fig.update_layout(updatemenus=[dict(active=0, buttons=btns, visible=False)])


def stack(builder, variants):
    """Build one figure per variant and merge them into a single switchable figure."""
    out, spans = go.Figure(), []
    for i, args in enumerate(variants):
        f = builder(*args)
        a = len(out.data)
        for tr in f.data:
            tr.visible = False       # hidden traces drop out of the legend too
            out.add_trace(tr)
        spans.append([a, len(out.data)])
        if i == 0:
            out.update_layout(f.layout)
    return out, spans


# ================================================================== 1 OVERVIEW
def overview():
    return "", kpi([
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
    ]) + (
        card("Cost by policy",
             "Holding, spoilage and lost margin, held-out window · P3b is the proposal",
             plot(fg.cost_by_policy(), legend=True))
        + card("Service against cost",
               "Up and to the left is better · bubble area is average units held",
               plot(fg.service_vs_cost()))
        + card("Demand the model tracks",
               "Panel-wide units a day, 2016 to the last day with recorded sales",
               plot(fg.demand_timeline()), cls="wide")
    )


# ========================================================== 2 FORECAST ACCURACY
def forecast():
    bt = lib.csv("backtest_results.csv")
    return "", kpi([
        {"label": "HGB mean RMSLE", "value": f"{bt.hgb_rmsle.mean():.3f}",
         "note": f"spread ±{bt.hgb_rmsle.std():.3f}", "tone": "good"},
        {"label": "Seasonal naive", "value": f"{bt.naive_rmsle.mean():.3f}",
         "note": "baseline to beat", "tone": "flat"},
        {"label": "Gain, final fold", "value": f"{K['holdout_gain_vs_naive']:.0%}",
         "note": "target was 15%", "tone": "good"},
        {"label": "Features", "value": "77", "note": "71 numeric, 6 categorical", "tone": "flat"},
        {"label": "Leaky columns cut", "value": "9",
         "note": "same-day weather, oil, transactions", "tone": "warn"},
    ]) + (
        card("Model leaderboard", "RMSLE per rolling-origin fold, five models",
             plot(fg.leaderboard(), legend=True), cls="wide")
        + card("What the model leans on", "Permutation importance, top 12",
               plot(fg.importance()))
        + card("Error by distance", "RMSLE against days after the origin",
               plot(fg.rmsle_by_horizon()))
        + card("Error by distance, in volume terms", "WAPE against days after the origin",
               plot(fg.wape_by_horizon()))
        + card("Where error concentrates", "WAPE per family, dot size is volume",
               plot(fg.wape_by_family()))
        + card("Predicted against actual", "3,500 sampled store-family-days, held out",
               plot(fg.pred_vs_actual()))
    )


# ================================================================ 3 CALIBRATION
def calibration():
    return "", kpi([
        {"label": "P90 before", "value": f"{K['p90_before']:.1%}", "note": "raw quantile",
         "tone": "warn"},
        {"label": "P90 after", "value": f"{K['p90_after']:.1%}",
         "note": "conformal correction applied", "tone": "warn"},
        {"label": "Gate", "value": "88 to 92%", "note": "not met", "tone": "warn"},
        {"label": "Crossings fixed", "value": "13.2%", "note": "none left in shipped files",
         "tone": "flat"},
        {"label": "Rows checked", "value": "27,664", "note": "held-out window with actuals",
         "tone": "flat"},
    ]) + (
        card("Promised against delivered", "Empirical coverage per quantile, held-out window",
             plot(fg.coverage_dumbbell(), legend=True))
        + card("Coverage across cutoffs", "Three model cutoffs, dotted lines are the targets",
               plot(fg.rolling_coverage(), legend=True))
        + card("Interval width", "Median quantile as a multiple of the P50, by group",
               plot(fg.interval_width(), legend=True), cls="wide")
        + card("Conformal correction, per family",
               "Raw against corrected P90, shaded band is the gate",
               plot(fg.p90_before_after(), legend=True), cls="tall")
    )


# ============================================================ 4 SERIES EXPLORER
PICKS = [(44, "GROCERY I"), (47, "PRODUCE"), (3, "BEVERAGES"), (45, "DAIRY"),
         (8, "BREAD/BAKERY"), (2, "CLEANING"), (24, "MEATS"), (11, "POULTRY"),
         (39, "HOME CARE"), (6, "EGGS"), (20, "AUTOMOTIVE"), (36, "HARDWARE")]


def explorer():
    meta = lib.csv("series_meta.csv")
    stats = lib.parquet("ledger_by_series.parquet")
    stats = stats[stats.policy == fg.PROPOSAL]

    titles = [f"Store {st} · {fa.title()}" for st, fa in PICKS]
    fig, spans = stack(fg.series_forecast, PICKS)
    fig2, spans2 = stack(fg.series_ledger, PICKS)
    hide_menu(fig, spans, titles)
    hide_menu(fig2, spans2, titles)
    fg.series_origin_marker(fig)

    facts, groups = [], []
    for store, family in PICKS:
        m = meta[(meta.store_nbr == store) & (meta.family == family)].iloc[0]
        s_ = stats[(stats.store_nbr == store) & (stats.family == family)]
        fill = float(s_.fill_rate.iloc[0]) if len(s_) else float("nan")
        sod = int(s_.stockout_days.iloc[0]) if len(s_) else 0
        shelf = f"shelf life {int(m.shelf_life)}d" if m.shelf_life > 0 else "no expiry"
        groups.append(m["group"])
        facts.append([
            {"label": "Location", "value": str(m.city), "note": f"{m['group']} · store {store}"},
            {"label": "Mean demand", "value": f"{m.mean_units:,.0f}/day",
             "note": f"{m.zero_share:.0%} zero days"},
            {"label": "Service level", "value": f"{m.q_star:.2f}",
             "note": f"lead time {int(m.lead_time)}d · {shelf}"},
            {"label": "Fill under P3b", "value": f"{fill:.1%}" if fill == fill else "n/a",
             "note": f"{sod} stockout day{'' if sod == 1 else 's'} in the scored month"},
        ])

    top = plot(fig, legend=True, m=dict(l=56, r=16, t=30, b=30))
    top_id = _ids[-1]
    bottom = plot(fig2, legend=True, m=dict(l=56, r=16, t=30, b=30))
    LINKS.append([top_id, _ids[-1]])

    opts = "".join(f"<option value='{i}' data-group='{g}'>{t}</option>"
                   for i, (t, g) in enumerate(zip(titles, groups)))
    ctl = (f"<label>Product group<select id='f-group'>"
           f"<option value='all'>All three</option>"
           f"<option value='grocery'>Grocery</option>"
           f"<option value='perishable'>Perishable</option>"
           f"<option value='hardware'>Hardware</option></select></label>"
           f"<label>Series<select id='f-series'>{opts}</select></label>"
           f"<span class='ctl-note'>{len(PICKS)} of 1,729 series carried in this preview</span>")

    body = ("<div class='kpis' id='series-facts'></div>"
            + card("Forecast against outcome",
                   "Median and intervals over the held-out window, then the live 16 days",
                   top, cls="wide")
            + card("Stock position under P3b",
                   "Simulated on hand, orders, demand and misses, 06-29 to 08-15",
                   bottom, cls="wide")
            + f"<script>window.__facts={json.dumps(facts)};"
            f"window.__explorer={json.dumps([top_id, _ids[-1]])};</script>")
    return ctl, body


# ============================================================ 5 INVENTORY POLICY
def policy():
    head = ["Policy", "Fill", "Stockout days", "Units held", "Spoiled", "Total cost", "vs P1"]
    role = {fg.PROPOSAL: "<span class='tag hot'>proposed</span>",
            "P3_newsvendor_q*": "<span class='tag'>diagnostic</span>",
            "P1_moving_avg_7d": "<span class='tag'>today</span>",
            "P5_perfect_foresight": "<span class='tag'>bound</span>"}
    windows = [
        ("holdout", "Held out 2017-07-31 to 08-15 · P3b is the proposal, P3 the diagnostic "
                    "that exposed the perishable trap"),
        ("full", "Full simulation 2017-07-15 to 08-15 · P3b was only run on the held-out "
                 "window, so the proposal is absent here"),
    ]
    tables, kpis, subs = [], [], []
    for window, sub in windows:
        pc = fg.comparison(window)
        base = pc[pc.policy == "P1_moving_avg_7d"].iloc[0]
        rows = ""
        for _, r in pc.iterrows():
            hot = " class='hot'" if r.policy == fg.PROPOSAL else ""
            rows += (f"<tr{hot}><td class='l'>{PL[r.policy]}{role.get(r.policy, '')}</td>"
                     f"<td>{r.fill_rate:.1%}</td><td>{int(r.stockout_days):,}</td>"
                     f"<td>{r.mean_units_held:,.0f}</td><td>{r.waste_units:,.0f}</td>"
                     f"<td class='s'>${r.total_cost:,.0f}</td>"
                     f"<td>{(r.total_cost/base.total_cost-1):.0%}</td></tr>")
        tables.append("<table class='tbl'><thead><tr>"
                      + "".join(f"<th>{c}</th>" for c in head)
                      + "</tr></thead><tbody>" + rows + "</tbody></table>")
        subs.append(sub)
        prop = pc[pc.policy == fg.PROPOSAL]
        if len(prop):
            lead, label, note = prop.iloc[0], "Proposal", "q* tuned per family"
        else:
            real = pc[pc.policy != "P5_perfect_foresight"]
            lead = real.loc[real.total_cost.idxmin()]
            label, note = "Cheapest here", "P3b was not run on this window"
        kpis.append(kpi([
            {"label": label, "value": PL[lead.policy].split(" · ")[0], "note": note,
             "tone": "flat"},
            {"label": "Fill rate", "value": f"{lead.fill_rate:.1%}",
             "note": f"{(lead.fill_rate-base.fill_rate)*100:+.1f} points vs P1", "tone": "good"},
            {"label": "Total cost", "value": lib.money(lead.total_cost),
             "note": f"{(lead.total_cost/base.total_cost-1):.0%} vs P1", "tone": "good"},
            {"label": "Units held", "value": lib.units(lead.mean_units_held),
             "note": f"{lead.mean_units_held/base.mean_units_held:.1f}x today", "tone": "warn"},
            {"label": "Units spoiled", "value": f"{lead.waste_units:,.0f}",
             "note": f"{lead.waste_rate:.2%} of units handled", "tone": "warn"},
        ]))

    ctl = ("<label>Scoring window<select id='f-window'>"
           "<option value='0'>Held out · 07-31 to 08-15</option>"
           "<option value='1'>Full simulation · 07-15 to 08-15</option>"
           "</select></label>"
           "<span class='ctl-note'>switches the table and the tiles above it</span>")
    body = ("<div id='policy-kpis'>" + kpis[0] + "</div>"
            + card("Five policies, one ledger", subs[0],
                   f"<div class='tblwrap' id='policy-table'>{tables[0]}</div>",
                   cls="wide flat", cid="policy-card")
            + card("Service level against cost", "One q for every family, full simulation window",
                   plot(fg.q_sweep()))
            + card("Robustness", "P3 against P1 under nine cost and logistics variants; P3b uses "
                   "the same engine at tuned service levels", plot(fg.sensitivity()))
            + card("Why P3 became P3b",
                   "The textbook ratio pins perishables at the floor; simulation moves 18 of 33",
                   plot(fg.q_star_slope(), legend=True), cls="tall")
            + "<script>window.__policy=" + json.dumps(
                {"kpis": kpis, "tables": tables, "subs": subs}) + ";</script>")
    return ctl, body


# ========================================================= 6 RECOMMENDATION ENGINE
STORES, HORIZONS = [1, 3, 44, 47], [3, 7, 10, 14]


def engine():
    sys.path.insert(0, str(HERE / "engine"))
    from src.policy_engine import HorizonError, plan

    fc = lib.parquet("forecast_test.parquet").copy()
    fc["date"] = pd.to_datetime(fc["date"])
    state = lib.parquet("closing_state.parquet")
    state = state[state.policy == fg.PROPOSAL]
    lead = lib.csv("assumptions_family_table.csv").set_index("family").lead_time

    bar_figs, sch_figs, kpis, provs, subs_b, subs_s = [], [], [], [], [], []
    for store in STORES:
        st_state = state[state.store_nbr == store]
        for H in HORIZONS:
            rows, refused = [], 0
            for _, r in st_state.iterrows():
                try:
                    p_ = plan(int(store), str(r.family), H, on_hand=float(r.on_hand),
                              on_order=float(r.on_order), forecast=fc)
                except HorizonError:
                    refused += 1
                    continue
                d0 = p_["schedule"][0]
                rows.append(dict(family=r.family, group=r["group"], on_hand=r.on_hand,
                                 on_order=r.on_order, order_qty=d0["order_qty"],
                                 order_up_to=d0["order_up_to"]))
            ok = pd.DataFrame(rows).sort_values("order_qty", ascending=False).head(12)
            bar_figs.append(fg.engine_bars(ok))
            refuse_txt = (f", {refused} line{'' if refused == 1 else 's'} refused: "
                          f"lead time leaves less than {H} days" if refused else "")
            subs_b.append(f"Store {store}, {H}-day plan: twelve largest lines{refuse_txt}")

            fam = "GROCERY I"
            row = st_state[st_state.family == fam].iloc[0]
            Hs = min(H, 16 - int(lead[fam]))
            p = plan(int(store), fam, Hs, on_hand=float(row.on_hand),
                     on_order=float(row.on_order), forecast=fc)
            sc = pd.DataFrame(p["schedule"])
            sch_figs.append(fg.engine_schedule(sc))
            subs_s.append(f"Store {store} Grocery I over {Hs} days: orders, position and cover")

            kpis.append(kpi([
                {"label": "Order today", "value": f"{p['recommended_order']:,.0f}",
                 "note": f"units of Grocery I, arriving {sc.arrival_date.iloc[0]}",
                 "tone": "good"},
                {"label": "Order-up-to", "value": f"{sc.order_up_to.iloc[0]:,.0f}",
                 "note": f"{p['protection_days']} days of cover at q*={p['q_star']:.2f}",
                 "tone": "flat"},
                {"label": "Position before", "value": f"{sc.position_before.iloc[0]:,.0f}",
                 "note": f"{row.on_hand:,.0f} on hand + {row.on_order:,.0f} on order",
                 "tone": "flat"},
                {"label": "Median demand", "value": f"{sc.demand_P50.iloc[0]:,.0f}/day",
                 "note": f"lead time {p['lead_time']}d · review {p['review_period']}d",
                 "tone": "flat"},
                {"label": "Lines to order", "value": f"{int((ok.order_qty > 0).sum())}",
                 "note": f"of {len(ok)} shown, {H}-day horizon", "tone": "flat"},
            ]))
            provs.append("<table class='tbl'><tbody>" + "".join(
                f"<tr><td class='l'>{k}</td><td class='s'>{v}</td></tr>" for k, v in [
                    ("Store", f"{store}"), ("Product", "Grocery I"),
                    ("Unit cost", f"${p['unit_cost']:.2f}"),
                    ("Unit margin", f"${p['unit_margin']:.2f}"),
                    ("Cost of being short, Cu", f"${p['cu']:.3f}"),
                    ("Cost of holding, Co", f"${p['co']:.4f}"),
                    ("Critical ratio", f"{p['q_star_raw']:.3f}"),
                    ("q* used", f"{p['q_star']:.2f}"),
                    ("Source of q*", p["q_star_source"].replace("_", " ")),
                    ("Protection interval", f"{p['protection_days']} days"),
                    ("Model version", (p["model_version"] or "n/a")
                     .replace("hgb_quantile_v2_cutoff_", "v2 · cutoff "))])
                + "</tbody></table>")

    def merge(figs):
        out, spans = go.Figure(), []
        for i, f in enumerate(figs):
            a = len(out.data)
            for tr in f.data:
                tr.visible = False
                out.add_trace(tr)
            spans.append([a, len(out.data)])
            if i == 0:
                out.update_layout(f.layout)
        return out, spans

    bars, spans_b = merge(bar_figs)
    sched, spans_s = merge(sch_figs)
    hide_menu(bars, spans_b)
    hide_menu(sched, spans_s)

    bars_html = plot(bars, legend=True, m=dict(l=158, r=16, t=30, b=34))
    bars_id = _ids[-1]
    sched_html = plot(sched, legend=True, m=dict(l=56, r=16, t=30, b=34))
    sched_id = _ids[-1]

    guards = ("<ul class='guards'>"
              "<li><b>Plan past the forecast.</b> Horizon is capped at 16 minus the lead time; "
              "beyond it the engine raises <code>HorizonError</code>, which is why some lines "
              "drop out at 10 and 14 days.</li>"
              "<li><b>Guess the stock position.</b> No on_hand, no plan: "
              "<code>MissingStockError</code>.</li>"
              "<li><b>Fill a forecast gap.</b> A missing quantile is reported as a gap, never "
              "interpolated.</li>"
              "<li><b>Write anything.</b> Pure function; placing the order stays with a person."
              "</li></ul>")

    ctl = ("<label>Store<select id='f-store'>"
           + "".join(f"<option value='{i}'>Store {st}</option>" for i, st in enumerate(STORES))
           + "</select></label><label>Planning horizon<select id='f-horizon'>"
           + "".join(f"<option value='{i}'{' selected' if h == 7 else ''}>{h} days</option>"
                     for i, h in enumerate(HORIZONS))
           + "</select></label>"
           "<span class='ctl-note'>every combination is a real call to plan(), made when "
           "this page was built</span>")

    body = ("<div id='agent-kpis'>" + kpis[1] + "</div>"
            + card("Today's order list", subs_b[1], bars_html, cls="wide", cid="ag-bars")
            + card("What the answer rests on", "Every input the engine returned with the order",
                   f"<div class='tblwrap' id='agent-prov'>{provs[1]}</div>", cls="flat")
            + card("One line, day by day", subs_s[1], sched_html, cls="wide", cid="ag-sched")
            + card("What the engine refuses to do",
                   "Guardrails that fire before any number reaches a screen", guards, cls="flat")
            + "<script>window.__agent=" + json.dumps({
                "bars": bars_id, "sched": sched_id, "kpis": kpis, "provs": provs,
                "subsB": subs_b, "subsS": subs_s, "nH": len(HORIZONS)}) + ";</script>")
    return ctl, body


# ===================================================================== assemble
SECTIONS = [
    ("Overview", "overview",
     "Headline service and cost figures, where the money goes under each ordering policy, "
     "and the demand the forecast has to track.", ["valid"], overview),
    ("Forecast accuracy", "forecast",
     "Model bake-off across rolling origins, feature importance, error by distance and by "
     "family, and forecasts against outcomes.", ["train", "valid"], forecast),
    ("Calibration", "calibration",
     "Whether a P90 forecast really covers 90% of days, which is what the policy assumes.",
     ["valid"], calibration),
    ("Series explorer", "explorer",
     "One store and product at a time: forecast, outcome and the stock underneath.",
     ["sim", "valid", "live"], explorer),
    ("Inventory policy", "policy",
     "Five ordering rules on one simulated ledger. P3b, with the service level tuned per "
     "family, is the proposal; P3 is the diagnostic that exposed the perishable trap.",
     ["sim", "valid"], policy),
    ("Recommendation engine", "engine",
     "The deterministic engine behind the ordering agent: what it recommends today, what "
     "each figure rests on, and what it refuses to answer.", ["live"], engine),
]

TIMEFRAMES = [
    ("train", "Training", "2013-01-01 to 2017-07-30"),
    ("sim", "Simulated", "2017-06-29 to 08-15"),
    ("valid", "Held out", "2017-07-31 to 08-15"),
    ("live", "Live forecast", "2017-08-16 to 08-31"),
]

CSS = f"""
*{{box-sizing:border-box}}
html,body{{margin:0;height:100%;overflow:hidden;background:{PAGE};color:{INK};
 font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
 font-size:14px;-webkit-font-smoothing:antialiased}}
#app{{display:flex;flex-direction:column;height:100vh}}
header{{padding:12px 20px 0 20px;border-bottom:1px solid {LINE};background:{PAGE};flex:none}}
.htop{{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}}
.brand{{font-size:1.12rem;font-weight:700;letter-spacing:-.01em}}
.brand span{{color:{TEAL}}}
.hmeta{{font-size:.72rem;color:{DIM};margin-left:auto}}
.lede{{font-size:.78rem;color:{MUTED};margin:3px 0 8px 0}}
nav{{display:flex;gap:3px;flex-wrap:wrap}}
nav button{{background:transparent;border:0;border-bottom:2px solid transparent;color:{MUTED};
 font:inherit;font-size:.82rem;font-weight:500;padding:7px 13px;cursor:pointer;
 border-radius:7px 7px 0 0}}
nav button:hover{{color:{INK};background:rgba(255,255,255,.04)}}
nav button.on{{color:{INK};border-bottom-color:{TEAL};background:rgba(0,163,146,.13)}}
.fbar{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:8px 20px;
 background:{PAGE};border-bottom:1px solid {LINE};flex:none}}
.ctlwrap{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;min-height:26px}}
.ctl{{display:none;align-items:center;gap:14px;flex-wrap:wrap}}
.ctl.on{{display:flex}}
.ctl label{{display:flex;align-items:center;gap:7px;font-size:.7rem;text-transform:uppercase;
 letter-spacing:.07em;color:{DIM};font-weight:600}}
.ctl select{{background:{PANEL};color:{INK};border:1px solid {LINE};border-radius:7px;
 padding:4px 26px 4px 9px;font:inherit;font-size:.78rem;text-transform:none;letter-spacing:0;
 font-weight:500;cursor:pointer;appearance:none;
 background-image:linear-gradient(45deg,transparent 50%,{MUTED} 50%),
  linear-gradient(135deg,{MUTED} 50%,transparent 50%);
 background-position:calc(100% - 14px) 52%,calc(100% - 9px) 52%;
 background-size:5px 5px,5px 5px;background-repeat:no-repeat}}
.ctl select:hover{{border-color:{TEAL}}}
.ctl select:focus{{outline:1px solid {TEAL};outline-offset:1px}}
.ctl-note{{font-size:.7rem;color:{DIM}}}
.tl{{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}}
.tf{{display:flex;flex-direction:column;line-height:1.25;font-size:.66rem;color:{DIM};
 border:1px solid {LINE};border-radius:7px;padding:3px 9px;background:transparent}}
.tf b{{font-size:.63rem;text-transform:uppercase;letter-spacing:.07em;color:{DIM};
 font-weight:600}}
.tf.on{{border-color:{TEAL};background:rgba(0,163,146,.12);color:{INK}}}
.tf.on b{{color:{TEAL}}}
main{{flex:1;min-height:0;position:relative}}
section{{position:absolute;inset:0;display:none;grid-gap:10px;padding:10px 20px 14px 20px}}
section.on{{display:grid}}
.kpis{{grid-column:1/-1;display:flex;gap:8px}}
.kpi{{flex:1;background:{PANEL};border:1px solid {LINE};border-radius:10px;
 padding:7px 11px 8px 11px;display:flex;flex-direction:column;gap:1px;min-width:0}}
.kl{{font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;color:{DIM};font-weight:600}}
.kv{{font-size:1.28rem;font-weight:700;line-height:1.15;letter-spacing:-.02em}}
.kn{{font-size:.68rem;line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;
 -webkit-box-orient:vertical;overflow:hidden}}
#policy-kpis,#agent-kpis{{grid-column:1/-1}}
#agent-prov .tbl{{table-layout:fixed}}
#agent-prov .tbl td.l{{width:54%;white-space:normal}}
#agent-prov .tbl td.s{{white-space:nowrap}}
.kn.good{{color:{TEAL}}}.kn.warn{{color:{AMBER}}}.kn.flat{{color:{DIM}}}
.card{{background:{PANEL};border:1px solid {LINE};border-radius:12px;padding:9px 12px 8px 12px;
 display:flex;flex-direction:column;min-height:0;min-width:0}}
.ch{{display:flex;flex-direction:column;gap:1px;margin-bottom:2px;flex:none}}
.ct{{font-size:.82rem;font-weight:650;letter-spacing:-.01em}}
.cs{{font-size:.68rem;color:{DIM};white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.plot{{flex:1;min-height:0;position:relative}}
.card.flat .plot{{overflow:auto}}
.tblwrap{{height:100%;overflow:auto}}
.tbl{{width:100%;border-collapse:collapse;font-size:.76rem}}
.tbl th{{position:sticky;top:0;background:{PANEL};color:{DIM};font-weight:600;text-align:right;
 padding:5px 7px;font-size:.64rem;text-transform:uppercase;letter-spacing:.05em;
 border-bottom:1px solid {LINE}}}
.tbl th:first-child{{text-align:left}}
.tbl td{{padding:5px 7px;text-align:right;border-bottom:1px solid rgba(255,255,255,.045);
 color:{MUTED}}}
.tbl td.l{{text-align:left}}
.tbl td.s{{color:{INK};font-weight:600}}
.tbl tr.hot td{{color:{INK}}}
.tbl tr.hot td.l{{color:{TEAL};font-weight:600}}
.tag{{display:inline-block;margin-left:8px;padding:1px 6px;border-radius:5px;
 font-size:.6rem;text-transform:uppercase;letter-spacing:.06em;font-weight:600;
 color:{DIM};border:1px solid {LINE};vertical-align:1px}}
.tag.hot{{color:{TEAL};border-color:rgba(0,163,146,.5);background:rgba(0,163,146,.12)}}
.guards{{margin:2px 0 0 0;padding-left:16px;font-size:.75rem;color:{MUTED};line-height:1.5}}
.guards li{{margin-bottom:6px}}
.guards b{{color:{INK};font-weight:600}}
.guards code{{background:rgba(255,255,255,.07);padding:1px 4px;border-radius:3px;
 font-size:.9em;color:{AMBER}}}
.modebar{{display:none!important}}
#s-overview{{grid-template-columns:1.1fr 1fr;grid-template-rows:auto 1.15fr .85fr}}
#s-overview .wide{{grid-column:1/-1}}
#s-forecast{{grid-template-columns:repeat(3,1fr);grid-template-rows:auto 1fr 1fr}}
#s-calibration{{grid-template-columns:1fr 1fr 1.05fr;grid-template-rows:auto 1fr .9fr}}
#s-calibration .wide{{grid-column:1/3}}
#s-calibration .tall{{grid-column:3;grid-row:2/4}}
#s-explorer{{grid-template-columns:1fr;grid-template-rows:auto 1.15fr 1fr}}
#s-policy{{grid-template-columns:1.25fr 1fr 1fr;grid-template-rows:auto .68fr 1.32fr}}
#s-policy .wide{{grid-column:1/3}}
#s-policy .tall{{grid-column:3;grid-row:2/4}}
#s-engine{{grid-template-columns:1fr 1fr .85fr;grid-template-rows:auto 1.1fr 1fr}}
#s-engine .wide{{grid-column:1/3}}
@media (max-width:1100px),(max-height:620px){{
  html,body{{overflow:auto;height:auto}}
  #app{{height:auto}}
  main{{position:static}}
  section{{position:static;grid-template-columns:1fr!important;grid-auto-rows:minmax(300px,auto)}}
  section .wide,section .tall{{grid-column:1!important;grid-row:auto!important}}
  .kpis{{flex-wrap:wrap}}.kpi{{min-width:150px}}
}}
"""

JS = """
const tabs=[...document.querySelectorAll('nav button')],
      secs=[...document.querySelectorAll('section')],
      ledes=[...document.querySelectorAll('nav button')].map(b=>b.dataset.lede);
function fit(sec){sec.querySelectorAll('.js-plotly-plot')
  .forEach(p=>{try{Plotly.Plots.resize(p)}catch(e){}});}
const ctls=[...document.querySelectorAll('.ctl')];
function show(i){
  tabs.forEach((t,j)=>t.classList.toggle('on',i===j));
  secs.forEach((s,j)=>s.classList.toggle('on',i===j));
  ctls.forEach((c,j)=>c.classList.toggle('on',i===j));
  const keys=(window.__tf||[])[i]||[];
  document.querySelectorAll('.tf').forEach(t=>
    t.classList.toggle('on',keys.indexOf(t.dataset.k)>=0));
  document.getElementById('lede').textContent=ledes[i];
  requestAnimationFrame(()=>fit(secs[i]));
  history.replaceState(null,'','#'+i);
}
function applyBtn(id,i){
  const d=document.getElementById(id); if(!d||!d.layout.updatemenus) return Promise.resolve();
  const b=d.layout.updatemenus[0].buttons[i];
  return Plotly.update(d,b.args[0],b.args[1]||{}).then(()=>
    Plotly.relayout(d,{'updatemenus[0].active':i}));
}
function setSub(cid,txt){const e=document.getElementById(cid+'-sub'); if(e) e.textContent=txt;}
tabs.forEach((t,i)=>t.addEventListener('click',()=>show(i)));
let rt; addEventListener('resize',()=>{clearTimeout(rt);
  rt=setTimeout(()=>fit(document.querySelector('section.on')),120);});

function facts(i){
  const box=document.getElementById('series-facts'); if(!box||!window.__facts) return;
  box.innerHTML=window.__facts[i].map(f=>
    `<div class="kpi"><span class="kl">${f.label}</span><span class="kv">${f.value}</span>`+
    `<span class="kn flat">${f.note}</span></div>`).join('');
}
// series explorer: group narrows the list, series drives both charts and the tiles
const gSel=document.getElementById('f-group'), sSel=document.getElementById('f-series');
if(sSel){
  const all=[...sSel.options].map(o=>({v:o.value,t:o.text,g:o.dataset.group}));
  const pick=i=>{ const ids=window.__explorer||[];
    ids.forEach(id=>applyBtn(id,i)); facts(i); };
  sSel.addEventListener('change',()=>pick(+sSel.value));
  if(gSel) gSel.addEventListener('change',()=>{
    const g=gSel.value;
    sSel.innerHTML=all.filter(o=>g==='all'||o.g===g)
      .map(o=>`<option value="${o.v}">${o.t}</option>`).join('');
    pick(+sSel.value);
  });
  facts(+sSel.value);
}
// inventory policy: scoring window swaps the tiles, the table and the caption
const wSel=document.getElementById('f-window');
if(wSel) wSel.addEventListener('change',()=>{
  const i=+wSel.value, P=window.__policy;
  document.getElementById('policy-kpis').innerHTML=P.kpis[i];
  document.getElementById('policy-table').innerHTML=P.tables[i];
  setSub('policy-card',P.subs[i]);
});
// agent brief: store x horizon, both precomputed from the engine
const stSel=document.getElementById('f-store'), hSel=document.getElementById('f-horizon');
function agentPick(){
  const A=window.__agent; if(!A) return;
  const i=(+stSel.value)*A.nH+(+hSel.value);
  applyBtn(A.bars,i); applyBtn(A.sched,i);
  document.getElementById('agent-kpis').innerHTML=A.kpis[i];
  document.getElementById('agent-prov').innerHTML=A.provs[i];
  setSub('ag-bars',A.subsB[i]); setSub('ag-sched',A.subsS[i]);
}
if(stSel&&hSel){ stSel.addEventListener('change',agentPick);
  hSel.addEventListener('change',agentPick); }
show(parseInt((location.hash||'#0').slice(1))||0);
"""


def build():
    built = [(sid, fn()) for _, sid, _, _, fn in SECTIONS]
    nav = "".join(f"<button data-lede=\"{lede}\">{name}</button>"
                  for name, _, lede, _, _ in SECTIONS)
    secs = "".join(f"<section id='s-{sid}'>{body}</section>" for sid, (_, body) in built)
    ctls = "".join(f"<div class='ctl' id='c-{sid}'>{c}</div>" for sid, (c, _) in built)
    chips = "".join(f"<span class='tf' data-k='{k}'><b>{lab}</b>{rng}</span>"
                    for k, lab, rng in TIMEFRAMES)
    tfmap = json.dumps([tf for _, _, _, tf, _ in SECTIONS])
    meta = (f"{K['series']:,} series · {K['stores']} stores · {K['families']} families · "
            f"{K['horizon_days']}-day quantile forecast · simulation {K['sim_window']} · "
            f"built {K['generated']}")
    page = f"""<div id="app">
<header>
  <div class="htop"><div class="brand">Stock <span>Signal</span></div>
    <div class="hmeta">{meta}</div></div>
  <div class="lede" id="lede"></div>
  <nav>{nav}</nav>
</header>
<div class="fbar">
  <div class="ctlwrap">{ctls}</div>
  <div class="tl">{chips}</div>
</div>
<main>{secs}</main>
</div>
<script>window.__links={json.dumps(LINKS)};window.__tf={tfmap};{JS}</script>"""

    full = (f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stock Signal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
 rel="stylesheet">
<style>{CSS}</style>
<script>{po.get_plotlyjs()}</script>
</head><body>{page}</body></html>""")
    OUT.write_text(full, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {_n[0]} charts)")

    art = (f"<style>{CSS}</style>\n<script>{po.get_plotlyjs()}</script>\n{page}")
    # plotly.js carries one literal U+FFFD inside a regex; escape it for the publisher
    ART.write_text(art.replace("�", "\\uFFFD"), encoding="utf-8")
    print(f"wrote {ART}  ({ART.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    build()
