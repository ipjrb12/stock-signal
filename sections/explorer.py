"""Series explorer: one store and product at a time, across all 1,729 series."""
import streamlit as st

import figures as fg
import lib


def render():
    meta = lib.csv("series_meta.csv")
    stats = lib.parquet("ledger_by_series.parquet")
    stats = stats[stats.policy == fg.PROPOSAL]

    a, b, c = st.columns([1, 1, 1.4])
    groups = ["All three"] + sorted(meta["group"].unique().tolist())
    group = a.selectbox("Product group", groups, format_func=str.title)
    pool = meta if group == "All three" else meta[meta["group"] == group]
    stores = sorted(pool.store_nbr.unique())
    store = b.selectbox("Store", stores, format_func=lambda s: f"Store {s}")
    fams = sorted(pool[pool.store_nbr == store].family.unique())
    default = fams.index("GROCERY I") if "GROCERY I" in fams else 0
    family = c.selectbox("Product family", fams, index=default, format_func=str.title)

    m = meta[(meta.store_nbr == store) & (meta.family == family)].iloc[0]
    s_ = stats[(stats.store_nbr == store) & (stats.family == family)]
    fill = float(s_.fill_rate.iloc[0]) if len(s_) else float("nan")
    sod = int(s_.stockout_days.iloc[0]) if len(s_) else 0
    shelf = f"shelf life {int(m.shelf_life)}d" if m.shelf_life > 0 else "no expiry"
    lib.tiles([
        {"label": "Location", "value": str(m.city), "note": f"{m['group']} · store {store}"},
        {"label": "Mean demand", "value": f"{m.mean_units:,.0f}/day",
         "note": f"{m.zero_share:.0%} zero days"},
        {"label": "Service level", "value": f"{m.q_star:.2f}",
         "note": f"lead time {int(m.lead_time)}d · {shelf}"},
        {"label": "Fill under P3b", "value": f"{fill:.1%}" if fill == fill else "n/a",
         "note": f"{sod} stockout day{'' if sod == 1 else 's'} in the scored month",
         "tone": "good" if fill == fill and fill > .97 else "warn"},
    ])

    fig = fg.series_origin_marker(fg.series_forecast(int(store), str(family)))
    lib.chart(fig, "Forecast against outcome",
              "Median and intervals over the held-out window, then the live 16 days",
              height=330, legend=True)
    lib.chart(fg.series_ledger(int(store), str(family)), "Stock position under P3b",
              "Simulated on hand, orders, demand and misses, 06-29 to 08-15",
              height=300, legend=True)
