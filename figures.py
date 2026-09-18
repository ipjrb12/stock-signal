"""
Every chart in the dashboard, built once.

Both front ends import from here: `app.py` (Streamlit) and `build_html.py` (the
single-file page). That is deliberate — when the two had their own copies of the
chart code they drifted, and the static page ended up two redesigns ahead of the
app. One module, one look, one set of numbers.

Each builder returns a plotly Figure with its own left/right/bottom margins
already set, because those depend on the labels rather than on the host. The host
sets the height and patches the top margin depending on whether it shows a legend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import lib

# ---------------------------------------------------------------- dark palette
# The five project hues re-stepped for the dark surface and checked with the
# dataviz validator: lightness band, chroma floor, CVD separation, normal-vision
# separation and contrast against #16262C all pass.
TEAL, AMBER, BLUE, RED, PURPLE = "#00A392", "#C28416", "#5482FF", "#E8563A", "#B35CEA"
PAGE, PANEL, LINE = "#0E191E", "#16262C", "#24393F"
INK, MUTED, DIM = "#E6EEF0", "#93A7AF", "#6A7F88"
GRID = "#22363D"
GREY, PALE = "#7E8F98", "#43565E"
DEEP = "#1C6B63"          # recessive teal for secondary bars
ONHAND, ONORDER = "#5B7079", "#9DB0B8"

POLICY_ORDER = lib.POLICY_ORDER
PL = lib.POLICY_LABEL
PC = {"P1_moving_avg_7d": GREY, "P2_model_P50": BLUE, "P3_newsvendor_q*": TEAL,
      "P3b_newsvendor_q*_tuned": "#00D9C0", "P4_model_P95": AMBER,
      "P5_perfect_foresight": PALE}
GC = {"grocery": BLUE, "perishable": TEAL, "hardware": AMBER}
QS = ["P50", "P80", "P90", "P95", "P98"]
TARGETS = {"P50": .50, "P80": .80, "P90": .90, "P95": .95, "P98": .98}
PROPOSAL = "P3b_newsvendor_q*_tuned"


def install() -> None:
    """Register the dark template and make it the default."""
    pio.templates["stocksignal"] = go.layout.Template(layout=dict(
        font=dict(family="Inter,-apple-system,Segoe UI,system-ui,sans-serif", size=11.5,
                  color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[TEAL, AMBER, BLUE, RED, PURPLE],
        margin=dict(l=52, r=16, t=28, b=34),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=MUTED, size=10.5),
                   title=dict(font=dict(color=DIM, size=10.5), standoff=6)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=MUTED, size=10.5),
                   title=dict(font=dict(color=DIM, size=10.5), standoff=6)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                    font=dict(size=10.5, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=11.5, bgcolor=PANEL, bordercolor=LINE,
                        font=dict(color=INK)),
    ))
    pio.templates.default = "stocksignal"


install()

M = lambda l=52, r=16, t=26, b=34: dict(l=l, r=r, t=t, b=b)  # noqa: E731


def comparison(window: str = "holdout") -> pd.DataFrame:
    """The policy table for a window, ordered P1 to P5."""
    f = "policy_comparison_holdout.csv" if window == "holdout" else "policy_comparison.csv"
    pc = lib.csv(f).copy()
    pc["o"] = pc.policy.map({p: i for i, p in enumerate(POLICY_ORDER)})
    return pc.sort_values("o").drop(columns="o")


# ================================================================== 1 OVERVIEW
def cost_by_policy(window: str = "holdout") -> go.Figure:
    pc = comparison(window)
    pc = pc[pc.policy != "P5_perfect_foresight"]
    labels = [PL[p].replace(" · ", "<br>") for p in pc.policy]
    f = go.Figure()
    for col, name, colour in [("stockout_cost", "Lost margin", RED),
                              ("holding_cost", "Holding", BLUE),
                              ("waste_cost", "Spoilage", AMBER)]:
        f.add_bar(x=labels, y=pc[col], name=name, marker_color=colour,
                  marker_line=dict(color=PANEL, width=2),
                  hovertemplate="%{x}<br>" + name + ": $%{y:,.0f}<extra></extra>")
    f.update_layout(barmode="stack", yaxis_title="USD", margin=M())
    f.update_yaxes(tickformat="$,.0s")
    for x, y in zip(labels, pc.total_cost):
        f.add_annotation(x=x, y=y, text=lib.money(y), showarrow=False, yshift=11,
                         font=dict(size=10.5, color=INK))
    return f


def service_vs_cost(window: str = "holdout") -> go.Figure:
    pc = comparison(window)
    f = go.Figure()
    for _, r in pc.iterrows():
        f.add_trace(go.Scatter(
            x=[r.total_cost], y=[r.fill_rate], mode="markers+text",
            marker=dict(size=max(10, min(30, (r.mean_units_held / 42000) + 10)),
                        color=PC[r.policy], opacity=.9, line=dict(color=PANEL, width=2)),
            text=[PL[r.policy].split(" · ")[0]], textposition="middle right",
            textfont=dict(size=10.5, color=MUTED), showlegend=False,
            hovertemplate=(f"<b>{PL[r.policy]}</b><br>fill {r.fill_rate:.1%}"
                           f"<br>cost {lib.money(r.total_cost)}"
                           f"<br>{r.mean_units_held:,.0f} units held<extra></extra>")))
    f.update_layout(xaxis_title="Total cost, log scale", yaxis_title="Fill rate", margin=M())
    f.update_xaxes(type="log", tickformat="$,.0s")
    f.update_yaxes(tickformat=".0%", range=[.91, 1.008])
    return f


def demand_timeline() -> go.Figure:
    d = lib.csv("daily_totals.csv").copy()
    d["date"] = pd.to_datetime(d["date"])
    f = go.Figure(go.Scatter(x=d.date, y=d.units, mode="lines",
                             line=dict(color=TEAL, width=1.1), fill="tozeroy",
                             fillcolor="rgba(0,163,146,.13)",
                             hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} units<extra></extra>"))
    f.add_vrect(x0="2017-07-31", x1="2017-08-15", fillcolor=AMBER, opacity=.18, line_width=0,
                annotation_text="scored window", annotation_position="top left",
                annotation_font=dict(size=10, color=AMBER))
    f.update_layout(margin=M())
    f.update_yaxes(tickformat=",.0s", title="Units per day")
    return f


# ========================================================== 2 FORECAST ACCURACY
def leaderboard() -> go.Figure:
    bt = lib.csv("backtest_results.csv")
    f = go.Figure()
    for col, name, colour in [("naive_rmsle", "Seasonal naive", GREY),
                              ("ma_rmsle", "Moving average", PALE),
                              ("ridge_rmsle", "Ridge", PURPLE),
                              ("rf_rmsle", "Random forest", BLUE),
                              ("hgb_rmsle", "HGB (chosen)", TEAL)]:
        f.add_bar(x=bt.fold.str.replace(r"\s*\(.*\)", "", regex=True), y=bt[col], name=name,
                  marker_color=colour, marker_line=dict(color=PANEL, width=1.5),
                  hovertemplate="%{x}<br>" + name + ": %{y:.3f}<extra></extra>")
    f.update_layout(barmode="group", bargap=.3, yaxis_title="RMSLE", margin=M())
    return f


def importance() -> go.Figure:
    fi = lib.csv("feature_importance_top15.csv").copy()
    fi.columns = ["feature", "importance"]
    fi = fi.sort_values("importance").tail(12)
    f = go.Figure(go.Bar(x=fi.importance, y=fi.feature, orientation="h",
                         marker_color=[TEAL if v == fi.importance.max() else DEEP
                                       for v in fi.importance],
                         hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    f.update_layout(xaxis_title="Permutation importance", margin=M(l=112, t=10))
    f.update_yaxes(tickfont=dict(size=10))
    return f


def rmsle_by_horizon() -> go.Figure:
    eh = lib.csv("error_by_horizon.csv")
    f = go.Figure(go.Scatter(x=eh.horizon, y=eh.rmsle, mode="lines+markers",
                             line=dict(color=TEAL, width=2.2), marker=dict(size=6),
                             hovertemplate="day +%{x}<br>RMSLE %{y:.3f}<extra></extra>"))
    f.update_layout(xaxis_title="Days after the forecast origin", yaxis_title="RMSLE",
                    margin=M(t=10))
    f.update_yaxes(range=[0, .55])
    f.update_xaxes(dtick=2)
    return f


def wape_by_horizon() -> go.Figure:
    eh = lib.csv("error_by_horizon.csv")
    f = go.Figure(go.Scatter(x=eh.horizon, y=eh.wape, mode="lines+markers",
                             line=dict(color=AMBER, width=2.2), marker=dict(size=6),
                             hovertemplate="day +%{x}<br>WAPE %{y:.1%}<extra></extra>"))
    f.update_layout(xaxis_title="Days after the forecast origin", yaxis_title="WAPE",
                    margin=M(t=10))
    f.update_yaxes(tickformat=".0%", range=[0, float(eh.wape.max()) * 1.18])
    f.update_xaxes(dtick=2)
    return f


def wape_by_family(n: int = 14) -> go.Figure:
    ef = lib.csv("error_by_family.csv").head(n)
    meta = lib.csv("series_meta.csv")[["family", "group"]].drop_duplicates()
    ef = ef.merge(meta, on="family", how="left").sort_values("wape")
    f = go.Figure()
    for _, r in ef.iterrows():
        f.add_trace(go.Scatter(x=[0, r.wape], y=[r.family.title()] * 2, mode="lines",
                               line=dict(color=GC.get(r["group"], MUTED), width=2),
                               showlegend=False, hoverinfo="skip"))
    f.add_trace(go.Scatter(
        x=ef.wape, y=ef.family.str.title(), mode="markers", showlegend=False,
        marker=dict(size=[7 + 10 * (u / ef.units.max()) ** .5 for u in ef.units],
                    color=[GC.get(g, MUTED) for g in ef["group"]],
                    line=dict(color=PANEL, width=1.4)),
        customdata=ef[["units", "rmsle"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>WAPE %{x:.1%}<br>RMSLE %{customdata[1]:.3f}"
                      "<br>%{customdata[0]:,.0f} units<extra></extra>"))
    f.update_layout(xaxis_title="WAPE", margin=M(l=124, t=10))
    f.update_xaxes(tickformat=".0%")
    f.update_yaxes(tickfont=dict(size=10))
    return f


def pred_vs_actual(n: int = 3500) -> go.Figure:
    v = lib.parquet("forecast_valid.parquet")
    samp = v.sample(min(n, len(v)), random_state=7)
    f = go.Figure()
    f.add_trace(go.Scattergl(x=np.log1p(samp.sales_units), y=np.log1p(samp.P50), mode="markers",
                             marker=dict(size=3.5, color=TEAL, opacity=.3),
                             customdata=np.c_[samp.sales_units, samp.P50], showlegend=False,
                             hovertemplate="actual %{customdata[0]:,.0f}<br>"
                                           "P50 %{customdata[1]:,.0f}<extra></extra>"))
    hi = float(np.log1p(samp.sales_units).max())
    f.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", showlegend=False,
                           line=dict(color=MUTED, width=1.1, dash="dash"), hoverinfo="skip"))
    f.update_layout(xaxis_title="Actual units, log1p", yaxis_title="Forecast P50, log1p",
                    margin=M(t=10))
    return f


# ================================================================ 3 CALIBRATION
def coverage_dumbbell() -> go.Figure:
    cov = lib.csv("coverage_valid.csv")
    c = cov[cov["quantile"].isin(QS)].copy()
    f = go.Figure()
    for _, r in c.iterrows():
        colour = TEAL if abs(r.empirical - r.target) < .03 else AMBER
        f.add_trace(go.Scatter(x=[r.target, r.empirical], y=[r["quantile"]] * 2, mode="lines",
                               line=dict(color=colour, width=3), showlegend=False,
                               hoverinfo="skip"))
    f.add_trace(go.Scatter(x=c.target, y=c["quantile"], mode="markers", name="Target",
                           marker=dict(size=11, color=PAGE, line=dict(color=MUTED, width=2)),
                           hovertemplate="target %{x:.0%}<extra></extra>"))
    f.add_trace(go.Scatter(x=c.empirical, y=c["quantile"], mode="markers", name="Delivered",
                           marker=dict(size=11, color=TEAL),
                           hovertemplate="delivered %{x:.1%}<extra></extra>"))
    f.update_layout(margin=M())
    f.update_xaxes(tickformat=".0%", range=[.42, 1.0])
    return f


def rolling_coverage() -> go.Figure:
    rc = lib.csv("rolling_coverage.csv")
    f = go.Figure()
    for q, colour in zip(QS, [PURPLE, BLUE, TEAL, AMBER, RED]):
        f.add_trace(go.Scatter(x=rc.cutoff, y=rc[q], mode="lines+markers", name=q,
                               line=dict(color=colour, width=2), marker=dict(size=7),
                               hovertemplate=q + " at %{x}: %{y:.1%}<extra></extra>"))
        f.add_hline(y=TARGETS[q], line_dash="dot", line_color=colour, line_width=.9, opacity=.4)
    f.update_layout(xaxis_title="Model training cutoff", margin=M())
    f.update_yaxes(tickformat=".0%", range=[.45, 1.0])
    return f


def interval_width() -> go.Figure:
    v = lib.parquet("forecast_valid.parquet")
    meta = lib.csv("series_meta.csv")[["store_nbr", "family", "group"]]
    v = v.merge(meta, on=["store_nbr", "family"], how="left")
    w = (v.assign(**{q: v[q] / v.P50.clip(lower=.5) for q in ["P80", "P90", "P95", "P98"]})
         .groupby("group")[["P80", "P90", "P95", "P98"]].median().reset_index())
    f = go.Figure()
    for _, r in w.iterrows():
        f.add_trace(go.Scatter(x=["P80", "P90", "P95", "P98"], y=[r.P80, r.P90, r.P95, r.P98],
                               mode="lines+markers", name=r.group.title(),
                               line=dict(color=GC.get(r.group, MUTED), width=2.2),
                               marker=dict(size=8),
                               hovertemplate=r.group.title() +
                               " %{x}: %{y:.2f}x the P50<extra></extra>"))
    f.update_layout(yaxis_title="Multiple of the P50", margin=M())
    return f


def p90_before_after() -> go.Figure:
    ba = lib.csv("p90_before_after.csv").sort_values("raw")
    f = go.Figure()
    for _, r in ba.iterrows():
        f.add_trace(go.Scatter(x=[r.raw, r.corrected], y=[r.family.title()] * 2, mode="lines",
                               line=dict(color=PALE, width=1.6), showlegend=False,
                               hoverinfo="skip"))
    f.add_trace(go.Scatter(x=ba.raw, y=ba.family.str.title(), mode="markers", name="Raw P90",
                           marker=dict(size=8, color=AMBER),
                           hovertemplate="%{y}<br>raw %{x:.1%}<extra></extra>"))
    f.add_trace(go.Scatter(x=ba.corrected, y=ba.family.str.title(), mode="markers",
                           name="After correction", marker=dict(size=8, color=TEAL),
                           hovertemplate="%{y}<br>corrected %{x:.1%}<extra></extra>"))
    f.add_vrect(x0=.88, x1=.92, fillcolor=TEAL, opacity=.13, line_width=0)
    f.add_vline(x=.90, line_dash="dot", line_color=MUTED, line_width=1)
    f.update_layout(margin=M(l=142))
    f.update_xaxes(tickformat=".0%")
    f.update_yaxes(dtick=1, tickfont=dict(size=9.5))
    return f


# ============================================================ 4 SERIES EXPLORER
def series_forecast(store: int, family: str) -> go.Figure:
    hist = lib.parquet("history.parquet")
    val = lib.parquet("forecast_valid.parquet")
    tst = lib.parquet("forecast_test.parquet")
    hh = hist[(hist.store_nbr == store) & (hist.family == family)].sort_values("date")
    hh = hh[hh.date <= "2017-08-15"]
    vv = val[(val.store_nbr == store) & (val.family == family)].sort_values("date")
    tt = tst[(tst.store_nbr == store) & (tst.family == family)].sort_values("date")

    f = go.Figure()
    for frame, show in [(vv, True), (tt, False)]:
        for lo, hi, alpha in [("P10", "P98", .13), ("P50", "P90", .22)]:
            f.add_trace(go.Scatter(
                x=list(frame.date) + list(frame.date[::-1]),
                y=list(frame[hi]) + list(frame[lo][::-1]),
                fill="toself", fillcolor=f"rgba(0,163,146,{alpha})", line=dict(width=0),
                name=f"{lo} to {hi}", hoverinfo="skip", showlegend=show))
    f.add_trace(go.Scatter(x=hh.date, y=hh.sales_units, mode="lines", name="Actual",
                           line=dict(color=INK, width=1.3),
                           hovertemplate="%{x|%a %d %b}<br>%{y:,.0f} units<extra></extra>"))
    f.add_trace(go.Scatter(x=vv.date, y=vv.P50, mode="lines", name="P50 held out",
                           line=dict(color=TEAL, width=2.2),
                           hovertemplate="%{x|%a %d %b}<br>P50 %{y:,.1f}<extra></extra>"))
    f.add_trace(go.Scatter(x=tt.date, y=tt.P50, mode="lines", name="P50 live",
                           line=dict(color=TEAL, width=2.2, dash="dot"),
                           hovertemplate="%{x|%a %d %b}<br>P50 %{y:,.1f}<extra></extra>"))
    f.update_layout(hovermode="x unified", yaxis_title="Units per day", margin=M(l=56, b=30),
                    legend=dict(x=.02))
    return f


def series_origin_marker(f: go.Figure) -> go.Figure:
    f.add_vline(x="2017-07-30", line_dash="dash", line_color=MUTED, line_width=1)
    f.add_annotation(x="2017-07-30", y=1, yref="paper", text=" origin", showarrow=False,
                     xanchor="left", font=dict(size=10, color=MUTED))
    return f


def series_ledger(store: int, family: str) -> go.Figure:
    tr = lib.parquet("ledger_trace_p3b.parquet")
    g = tr[(tr.store_nbr == store) & (tr.family == family)].sort_values("date")
    f = go.Figure()
    f.add_trace(go.Scatter(x=g.date, y=g.on_hand_open, mode="lines", name="On hand",
                           line=dict(color=BLUE, width=2), fill="tozeroy",
                           fillcolor="rgba(84,130,255,.16)",
                           hovertemplate="%{x|%d %b}<br>on hand %{y:,.0f}<extra></extra>"))
    f.add_bar(x=g.date, y=g.order_qty, name="Ordered", marker_color=TEAL, opacity=.6,
              hovertemplate="%{x|%d %b}<br>ordered %{y:,.0f}<extra></extra>")
    f.add_trace(go.Scatter(x=g.date, y=g.demand, mode="lines", name="Demand",
                           line=dict(color=INK, width=1.1, dash="dot"),
                           hovertemplate="%{x|%d %b}<br>demand %{y:,.0f}<extra></extra>"))
    lost = g[g.lost > 0]
    f.add_trace(go.Scatter(x=lost.date, y=[0] * len(lost), mode="markers", name="Unmet",
                           marker=dict(symbol="x", size=8, color=RED), customdata=lost.lost,
                           hovertemplate="%{x|%d %b}: %{customdata:,.0f} lost<extra></extra>"))
    exp = g[g.expired > 0]
    f.add_trace(go.Scatter(x=exp.date, y=exp.expired, mode="markers", name="Expired",
                           marker=dict(symbol="diamond", size=7, color=AMBER),
                           hovertemplate="%{x|%d %b}: %{y:,.0f} expired<extra></extra>"))
    f.update_layout(hovermode="x unified", yaxis_title="Units", margin=M(l=56, b=30),
                    legend=dict(x=.02))
    return f


# ============================================================ 5 INVENTORY POLICY
def q_sweep() -> go.Figure:
    qs = lib.csv("q_sweep.csv").sort_values("q")
    f = go.Figure(go.Scatter(
        x=qs.fill_rate, y=qs.total_cost, mode="lines+markers+text",
        text=[f"{q:.2f}" for q in qs.q], textposition="top right",
        textfont=dict(size=9.5, color=MUTED), line=dict(color=TEAL, width=2.2),
        marker=dict(size=8, color=TEAL), customdata=qs.q,
        hovertemplate="q=%{customdata:.2f}<br>fill %{x:.1%}<br>cost $%{y:,.0f}<extra></extra>"))
    f.update_layout(xaxis_title="Fill rate", yaxis_title="Total cost", margin=M(l=58, r=34, t=10))
    f.update_xaxes(tickformat=".1%")
    f.update_yaxes(tickformat="$.2s")
    return f


def q_star_slope() -> go.Figure:
    qt = lib.csv("q_star_tuned.csv")
    f = go.Figure()
    for _, r in qt.sort_values(["group", "q_star_tuned"]).iterrows():
        moved = r.q_star_analytic != r.q_star_tuned
        f.add_trace(go.Scatter(x=[r.q_star_analytic, r.q_star_tuned], y=[r.family.title()] * 2,
                               mode="lines", showlegend=False, hoverinfo="skip",
                               line=dict(color=GC.get(r["group"], MUTED) if moved else LINE,
                                         width=2 if moved else 1.2)))
    f.add_trace(go.Scatter(x=qt.q_star_analytic, y=qt.family.str.title(), mode="markers",
                           name="Analytic",
                           marker=dict(size=7, color=PAGE, line=dict(color=MUTED, width=1.6)),
                           hovertemplate="%{y}<br>analytic %{x:.2f}<extra></extra>"))
    f.add_trace(go.Scatter(x=qt.q_star_tuned, y=qt.family.str.title(), mode="markers",
                           name="Tuned",
                           marker=dict(size=8, color=[GC.get(g, MUTED) for g in qt["group"]]),
                           hovertemplate="%{y}<br>tuned %{x:.2f}<extra></extra>"))
    f.update_layout(xaxis_title="Service level q*", margin=M(l=138))
    f.update_xaxes(range=[.45, 1.0])
    f.update_yaxes(dtick=1, tickfont=dict(size=9.5))
    return f


def sensitivity() -> go.Figure:
    sen = lib.csv("sensitivity.csv").copy()
    sen["saving"] = -sen.P3_cost_vs_P1
    sen["variant"] = sen.variant.replace({
        "lead times +1 day (perish., grocery), +3 (hardware)": "lead times +1d, +3d hardware",
        "aggregation: normal, rho=0.3": "normal aggregation, rho 0.3"})
    sen = sen.sort_values("saving")
    f = go.Figure(go.Bar(x=sen.saving, y=sen.variant, orientation="h",
                         marker_color=[TEAL if v == "base" else DEEP for v in sen.variant],
                         text=[f"{v:.0%}" for v in sen.saving], textposition="outside",
                         textfont=dict(size=10, color=MUTED),
                         hovertemplate="%{y}<br>saves %{x:.1%} against P1<extra></extra>"))
    f.update_layout(xaxis_title="Cost saving of P3 against P1", margin=M(l=168, r=40, t=10))
    f.update_xaxes(tickformat=".0%", range=[0, .95])
    f.update_yaxes(tickfont=dict(size=9.5))
    return f


# ========================================================= 6 RECOMMENDATION ENGINE
def engine_bars(rows: pd.DataFrame) -> go.Figure:
    """rows: family, group, on_hand, on_order, order_qty, order_up_to."""
    f = go.Figure()
    f.add_bar(y=rows.family.str.title(), x=rows.on_hand, orientation="h", name="On hand",
              marker_color=ONHAND, hovertemplate="%{y}<br>on hand %{x:,.0f}<extra></extra>")
    f.add_bar(y=rows.family.str.title(), x=rows.on_order, orientation="h", name="On order",
              marker_color=ONORDER, hovertemplate="%{y}<br>on order %{x:,.0f}<extra></extra>")
    f.add_bar(y=rows.family.str.title(), x=rows.order_qty, orientation="h", name="Order today",
              marker_color=[GC.get(g, TEAL) for g in rows["group"]],
              hovertemplate="%{y}<br>order %{x:,.0f}<extra></extra>")
    f.add_trace(go.Scatter(y=rows.family.str.title(), x=rows.order_up_to, mode="markers",
                           name="Order-up-to",
                           marker=dict(symbol="line-ns", size=14, line=dict(color=INK, width=2)),
                           hovertemplate="%{y}<br>up to %{x:,.0f}<extra></extra>"))
    f.update_layout(barmode="stack", xaxis_title="Units", margin=M(l=158), legend=dict(x=.02))
    f.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return f


def engine_schedule(sched: pd.DataFrame) -> go.Figure:
    """sched: the schedule frame returned by policy_engine.plan()."""
    d = pd.to_datetime(sched.date)
    f = go.Figure()
    f.add_bar(x=d, y=sched.order_qty, name="Order raised", marker_color=TEAL, opacity=.8,
              hovertemplate="%{x|%d %b}<br>order %{y:,.0f}<extra></extra>")
    f.add_trace(go.Scatter(x=d, y=sched.order_up_to, mode="lines+markers", name="Order-up-to",
                           line=dict(color=INK, width=1.8), marker=dict(size=5),
                           hovertemplate="%{x|%d %b}<br>S = %{y:,.0f}<extra></extra>"))
    f.add_trace(go.Scatter(x=d, y=sched.position_before, mode="lines+markers", name="Position",
                           line=dict(color=BLUE, width=1.8, dash="dot"), marker=dict(size=5),
                           hovertemplate="%{x|%d %b}<br>position %{y:,.0f}<extra></extra>"))
    f.add_trace(go.Scatter(x=d, y=sched.demand_P50, mode="lines", name="Median demand",
                           line=dict(color=AMBER, width=1.5),
                           hovertemplate="%{x|%d %b}<br>P50 %{y:,.0f}<extra></extra>"))
    f.update_layout(hovermode="x unified", yaxis_title="Units", margin=M(l=56),
                    legend=dict(x=.02))
    return f
