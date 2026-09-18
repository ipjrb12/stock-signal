# Stock Signal: the dashboard

Workstream E. One Streamlit app over everything workstreams A, B and C produced:
the demand panel, the quantile forecasts, the calibration evidence and the
inventory simulation ledger.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. Nothing else is needed: the app reads only
from `data/`, which is already built and committed here (6.4 MB), so there is no
model to load, no parquet panel to find and no first-run wait.

## The no-install preview

`stock-signal-preview.html` is the whole dashboard flattened into one file:
same data, same charts, Plotly and all. Double-click it and it opens in any
browser, offline, with nothing to install, and every section fits one screen.
Charts still hover and zoom; the pages that need a control carry a dropdown in
the filter bar. It carries twelve series and four stores rather than all of them,
because the picks are baked in at build time. Rebuild it after any data change
with:

```bash
python build_html.py
```

Use the preview to show the work to someone; use the Streamlit app when you want
every store and family, the scoring-window switch and the horizon slider.

## The six sections

| Section | What it answers |
|---|---|
| **Overview** | Is the idea working? Fill rate, cost, stockout days and forecast error against the current rule, plus where the money actually goes |
| **Forecast accuracy** | Which model, and how good? Five models across three rolling origins, error by horizon and by family, feature importance, predicted against actual |
| **Calibration** | Can the policy trust the quantiles? Promised against delivered coverage, the per-family conformal correction, and how wide the interval has to be |
| **Series explorer** | What does one store and product look like? Forecast fan against actuals, and the simulated stock position underneath it |
| **Inventory policy** | Which ordering rule, and why? The five-policy comparison, the service-level sweep, the per-family tuning and nine sensitivity variants |
| **Recommendation engine** | What do I order this morning? The deterministic engine the ordering agent calls: its schedule, the numbers behind it and what it refuses to do |

## The Recommendation engine page calls the real engine

That page is not a mock-up and it is not a re-implementation. It imports
`plan()` from workstream C's `policy_engine.py` (vendored unchanged under
`engine/src/`) and calls it with whatever store, family, horizon and stock
position you set. Workstream D's language layer calls the same function, so
the number on this page and the number in the agent's sentence come from one
piece of code:

```python
from src.policy_engine import plan
plan(store=1, product="GROCERY I", H=7, on_hand=3844, on_order=4766)
```

Everything the panel shows, from the critical ratio to the model version, comes
back inside that one return value. The guardrails are live too: push the horizon
past `16 - lead_time` and the engine raises `HorizonError` rather than
extrapolating, and the page shows the refusal instead of hiding it.

## Layout

```
app.py                  page config, top navigation, timeframe chips
figures.py              every chart, shared by the app and the HTML build
lib.py                  cached loaders, formatters, tiles and CSS
sections/
  overview.py           headline numbers, cost by policy, demand
  forecast.py           model selection and error decomposition
  calibration.py        coverage, conformal correction, interval width
  explorer.py           one store and family at a time, all 1,729
  policy.py             the five-policy comparison and the q* tuning
  engine.py             live calls into the recommendation engine
engine/src/             workstream C's modules, vendored unchanged
data/                   29 pre-aggregated files, 6.4 MB
prepare_data.py         rebuilds data/ from the raw workstream outputs
build_html.py           builds the single-file preview and the hosted page
.streamlit/config.toml  dark theme
```

**One chart module, two front ends.** `figures.py` builds every figure; `app.py`
renders them through Streamlit and `build_html.py` writes them into the static
page. They had separate copies once and drifted a whole redesign apart, so if you
change a chart, change it there and both follow.

## Rebuilding the data pack

Only needed if B or C ships new numbers. `prepare_data.py` reads the raw
deliverables and the cleaned panel, then writes `data/`:

```bash
python prepare_data.py
```

It expects `../incoming/modelling/Modelling_files_v2/`,
`../incoming/inventory/Inventory_Policy_files_v2/runs/` and the workstream A
panel at `/home/claude/out/favorita_demand_weather.parquet`. Edit the three
paths at the top of the file if yours sit elsewhere.

The heavy inputs never reach the app. The 28 MB simulation ledger becomes four
small summaries: by day, by family, by series, and the closing stock position on
the last simulated day. The 2.5 million row panel becomes a four-month history
slice plus one row per series of metadata.

## Notes worth keeping in mind when presenting it

- Every accuracy and coverage figure is measured on **2017-07-31 to 08-15**, a
  window the model never trained on. The live forecast on the Recommendation engine page covers
  08-16 to 08-31 and has no actuals by construction.
- **P3b is the proposal.** P3, the same newsvendor rule on the textbook
  Cu/(Cu+Co) ratio, is kept in the comparison as the diagnostic that exposed the
  perishable trap: that ratio lands below 0.5 for short shelf-life families, gets
  clipped to the grid floor, and leaves them under-stocked, which is why P3 is
  beaten by the crude P4. Tuning the service level per family turns that around
  on service, cost and waste at once.
- The policy table has a window switch. The held-out window is the honest one to
  quote; the full simulation window includes replica forecasts for the first
  fortnight, and the q sweep and sensitivity charts are measured on it, which is
  why their titles say so.
- **P90 coverage misses the gate.** The modelling write-up quotes 86.1% rising to
  89.8% after the conformal correction. Recomputed from the quantile files that
  shipped, it is 86.1% to 87.9%, and the rolling cutoffs agree at 87.4 to 87.9%.
  The plan's gate is 88 to 92%, so it is narrowly missed. Every calibration
  figure in the app is measured from the delivered files rather than copied from
  a report, which is why the app and the write-up differ. Quote the app.
- `sales_units` is censored: on a day something sold out, recorded sales
  understate real demand. Both the forecast and the simulation inherit that, so
  every fill rate here is, if anything, flattering.
