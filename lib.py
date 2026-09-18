"""Loading, formatting and page furniture for the Streamlit app.

Charts live in figures.py, which both this app and the single-file HTML build
import. Nothing here knows how to draw; it knows how to load and how to dress.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

POLICY_ORDER = [
    "P1_moving_avg_7d",
    "P2_model_P50",
    "P3_newsvendor_q*",
    "P3b_newsvendor_q*_tuned",
    "P4_model_P95",
    "P5_perfect_foresight",
]
POLICY_LABEL = {
    "P1_moving_avg_7d": "P1 · 7-day average",
    "P2_model_P50": "P2 · model P50",
    "P3_newsvendor_q*": "P3 · newsvendor q*",
    "P3b_newsvendor_q*_tuned": "P3b · q* tuned",
    "P4_model_P95": "P4 · P95 everywhere",
    "P5_perfect_foresight": "P5 · perfect foresight",
}

TIMEFRAMES = [
    ("train", "Training", "2013-01-01 to 2017-07-30"),
    ("sim", "Simulated", "2017-06-29 to 08-15"),
    ("valid", "Held out", "2017-07-31 to 08-15"),
    ("live", "Live forecast", "2017-08-16 to 08-31"),
]


# --------------------------------------------------------------------- loading
@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data(show_spinner=False)
def parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / name)


@st.cache_data(show_spinner=False)
def kpi() -> dict:
    return json.loads((DATA / "kpi.json").read_text())


@st.cache_data(show_spinner=False)
def q_star_map() -> dict:
    t = csv("q_star_tuned.csv")
    return dict(zip(t.family, t.q_star_tuned))


# ------------------------------------------------------------------ formatting
def money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x/1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:,.0f}k"
    return f"${x:,.0f}"


def units(x: float) -> str:
    if abs(x) >= 1e6:
        return f"{x/1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:,.0f}k"
    return f"{x:,.0f}"


# ------------------------------------------------------------------ furniture
def tiles(items: list[dict]) -> None:
    html = "<div class='kpis'>"
    for it in items:
        tone = it.get("tone", "flat")
        note = f"<span class='kn {tone}'>{it['note']}</span>" if it.get("note") else ""
        html += (f"<div class='kpi'><span class='kl'>{it['label']}</span>"
                 f"<span class='kv'>{it['value']}</span>{note}</div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def timeframes(active: list[str]) -> str:
    return "<div class='tl'>" + "".join(
        f"<span class='tf{' on' if k in active else ''}'><b>{lab}</b>{rng}</span>"
        for k, lab, rng in TIMEFRAMES) + "</div>"


def chart(fig, title: str, sub: str, height: int = 300, legend: bool = False) -> None:
    """One tile: a header line, a caption, and the chart filling the rest."""
    with st.container(border=True):
        st.markdown(f"<div class='ch'><span class='ct'>{title}</span>"
                    f"<span class='cs'>{sub}</span></div>", unsafe_allow_html=True)
        fig.update_layout(height=height, autosize=True,
                          margin=dict(t=26 if legend else 8))
        if legend:
            fig.update_layout(legend=dict(y=1.03))
        st.plotly_chart(fig, width="stretch", theme=None,
                        config={"displayModeBar": False})


def panel(title: str, sub: str):
    """A tile that holds something other than a chart. Use as a context manager."""
    box = st.container(border=True)
    box.markdown(f"<div class='ch'><span class='ct'>{title}</span>"
                 f"<span class='cs'>{sub}</span></div>", unsafe_allow_html=True)
    return box


CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  :root{--page:#0E191E;--panel:#16262C;--line:#24393F;--ink:#E6EEF0;--muted:#93A7AF;
        --dim:#6A7F88;--teal:#00A392;--amber:#C28416}
  html,body,.stApp{background:var(--page);color:var(--ink);
    font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif}
  .block-container{padding:1.1rem 1.6rem 2rem 1.6rem;max-width:1800px}
  header[data-testid="stHeader"]{background:transparent;height:0}
  #MainMenu,footer{visibility:hidden}

  .brandrow{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:.1rem}
  .brand{font-size:1.15rem;font-weight:700;letter-spacing:-.01em;color:var(--ink)}
  .brand span{color:var(--teal)}
  .hmeta{font-size:.72rem;color:var(--dim);margin-left:auto;text-align:right}
  .lede{font-size:.8rem;color:var(--muted);margin:.1rem 0 .5rem 0;max-width:120ch}

  /* top navigation, styled from the horizontal radio */
  div[role="radiogroup"]{gap:.15rem!important}
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[role="radiogroup"]
    label{border-bottom:2px solid transparent;border-radius:7px 7px 0 0;padding:6px 13px;
    margin:0;background:transparent}
  div[role="radiogroup"] label p{font-size:.84rem!important;color:var(--muted);
    font-weight:500;margin:0}
  div[role="radiogroup"] label:hover{background:rgba(255,255,255,.04)}
  div[role="radiogroup"] label:has(input:checked){background:rgba(0,163,146,.13);
    border-bottom-color:var(--teal)}
  div[role="radiogroup"] label:has(input:checked) p{color:var(--ink);font-weight:600}
  label[data-testid="stRadioOption"] > div > div:first-child{display:none!important}
  label[data-testid="stRadioOption"]{border-bottom:2px solid transparent;
    border-radius:7px 7px 0 0;padding:6px 13px;margin:0;background:transparent}
  label[data-testid="stRadioOption"]:hover{background:rgba(255,255,255,.04)}
  label[data-testid="stRadioOption"][data-selected="true"]{
    background:rgba(0,163,146,.13)!important;border-bottom:2px solid var(--teal)!important}
  label[data-testid="stRadioOption"][data-selected="true"] p{color:var(--ink);
    font-weight:600}
  [data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]
    {display:none!important}

  .tl{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
  .tf{display:flex;flex-direction:column;line-height:1.25;font-size:.66rem;color:var(--dim);
      border:1px solid var(--line);border-radius:7px;padding:3px 9px}
  .tf b{font-size:.63rem;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);
        font-weight:600}
  .tf.on{border-color:var(--teal);background:rgba(0,163,146,.12);color:var(--ink)}
  .tf.on b{color:var(--teal)}

  .kpis{display:flex;gap:8px;margin:.2rem 0 .7rem 0;flex-wrap:wrap}
  .kpi{flex:1;min-width:150px;background:var(--panel);border:1px solid var(--line);
       border-radius:10px;padding:7px 11px 8px 11px;display:flex;flex-direction:column;gap:1px}
  .kl{font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);
      font-weight:600}
  .kv{font-size:1.28rem;font-weight:700;line-height:1.15;letter-spacing:-.02em;color:var(--ink)}
  .kn{font-size:.68rem;line-height:1.3}
  .kn.good{color:var(--teal)}.kn.warn{color:var(--amber)}.kn.flat{color:var(--dim)}

  /* plotly must sit flat on the tile, not on Streamlit's own surface */
  div[data-testid="stPlotlyChart"],div[data-testid="stPlotlyChart"]>div,
  .js-plotly-plot,.js-plotly-plot .plot-container,.js-plotly-plot .main-svg
    {background:transparent!important}
  .js-plotly-plot .bg{fill:transparent!important}

  /* every bordered container is a tile */
  div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);
    border:1px solid var(--line)!important;border-radius:12px;padding:2px 4px}
  .ch{display:flex;flex-direction:column;gap:1px;margin:2px 0 0 4px}
  .ct{font-size:.82rem;font-weight:650;color:var(--ink);letter-spacing:-.01em}
  .cs{font-size:.68rem;color:var(--dim)}

  /* widgets */
  label[data-testid="stWidgetLabel"] p{font-size:.64rem!important;text-transform:uppercase;
    letter-spacing:.07em;color:var(--dim);font-weight:600}
  div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:var(--panel)!important;
    border-color:var(--line)!important;color:var(--ink)!important;font-size:.8rem}
  div[data-testid="stSlider"] label p{color:var(--dim)}
  .stSlider [data-baseweb="slider"] div[role="slider"]{background:var(--teal)}

  /* tables */
  div[data-testid="stDataFrame"]{border:0}
  .tbl{width:100%;border-collapse:collapse;font-size:.76rem}
  .tbl th{color:var(--dim);font-weight:600;text-align:right;padding:5px 7px;font-size:.64rem;
    text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--line)}
  .tbl th:first-child{text-align:left}
  .tbl td{padding:5px 7px;text-align:right;color:var(--muted);
    border-bottom:1px solid rgba(255,255,255,.045)}
  .tbl td.l{text-align:left}
  .tbl td.s{color:var(--ink);font-weight:600}
  .tbl tr.hot td{color:var(--ink)}
  .tbl tr.hot td.l{color:var(--teal);font-weight:600}
  .tag{display:inline-block;margin-left:8px;padding:1px 6px;border-radius:5px;font-size:.6rem;
    text-transform:uppercase;letter-spacing:.06em;font-weight:600;color:var(--dim);
    border:1px solid var(--line)}
  .tag.hot{color:var(--teal);border-color:rgba(0,163,146,.5);background:rgba(0,163,146,.12)}

  .guards{margin:4px 0 6px 0;padding-left:18px;font-size:.75rem;color:var(--muted);
    line-height:1.5}
  .guards li{margin-bottom:6px}
  .guards b{color:var(--ink);font-weight:600}
  .guards code{background:rgba(255,255,255,.07);padding:1px 4px;border-radius:3px;
    color:var(--amber);font-size:.9em}
  .flag{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--amber);
    border-radius:10px;padding:.7rem .9rem;margin:.1rem 0 .7rem 0;color:var(--muted);
    font-size:.8rem;line-height:1.5}
  .flag b{color:var(--ink)}
  hr{border-color:var(--line);margin:.6rem 0}
</style>
"""
