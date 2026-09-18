"""
Build the dashboard data pack.

Reads the raw deliverables from workstreams A, B and C and writes small,
pre-aggregated files into data/. The Streamlit app reads only from data/, so it
starts fast and the whole dashboard can be shipped as one zip.

    python prepare_data.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INC = ROOT.parent / "incoming"
MOD = INC / "modelling" / "Modelling_files_v2"
INV = INC / "inventory" / "Inventory_Policy_files_v2" / "runs"
PANEL = Path("/home/claude/out/favorita_demand_weather.parquet")
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

HISTORY_FROM = "2017-04-01"   # context window shown behind the forecast
VALID_CUTOFF = pd.Timestamp("2017-07-30")
TEST_CUTOFF = pd.Timestamp("2017-08-15")


def rmsle(a, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(a, 0, None)) - np.log1p(np.clip(p, 0, None))) ** 2)))


def wape(a, p):
    return float(np.sum(np.abs(a - p)) / np.sum(np.abs(a)))


# ---------------------------------------------------------------- copy-through
for src in [
    MOD / "backtest_results.csv",
    MOD / "coverage_valid.csv",
    MOD / "feature_importance_top15.csv",
    INV / "inventory" / "policy_comparison.csv",
    INV / "inventory" / "policy_comparison_holdout.csv",
    INV / "inventory" / "policy_comparison_by_group.csv",
    INV / "inventory" / "policy_comparison_holdout_by_group.csv",
    INV / "inventory" / "policy_comparison_by_window.csv",
    INV / "inventory" / "q_sweep.csv",
    INV / "inventory" / "q_star_tuned.csv",
    INV / "inventory" / "sensitivity.csv",
    INV / "inventory" / "assumptions_family_table.csv",
    INV / "inventory" / "forecast_coverage.csv",
    INV / "forecasts" / "rolling_coverage.csv",
]:
    shutil.copy(src, OUT / src.name)
print("copied 14 tables")

# ------------------------------------------------------------------- forecasts
valid = pd.read_csv(MOD / "quantile_forecast_valid_2017-07-31_to_08-15_WITH_ACTUALS.csv", parse_dates=["date"])
test = pd.read_parquet(INV / "forecasts" / "latest_quantiles.parquet")
test["date"] = pd.to_datetime(test["date"])

valid["horizon"] = (valid["date"] - VALID_CUTOFF).dt.days
test["horizon"] = (test["date"] - TEST_CUTOFF).dt.days

qcols = ["P10", "P50", "P80", "P90", "P95", "P98"]
valid.to_parquet(OUT / "forecast_valid.parquet", index=False)
test[["date", "store_nbr", "family", "horizon"] + qcols + ["model_version"]].to_parquet(
    OUT / "forecast_test.parquet", index=False)
print(f"forecast_valid {len(valid):,}   forecast_test {len(test):,}")

# --------------------------------------------------------- error decompositions
err_h = (
    valid.groupby("horizon")
    .apply(lambda g: pd.Series({"rmsle": rmsle(g.sales_units, g.P50), "wape": wape(g.sales_units, g.P50), "n": len(g)}),
           include_groups=False)
    .reset_index()
)
err_h.to_csv(OUT / "error_by_horizon.csv", index=False)

err_f = (
    valid.groupby("family")
    .apply(lambda g: pd.Series({
        "rmsle": rmsle(g.sales_units, g.P50),
        "wape": wape(g.sales_units, g.P50),
        "units": float(g.sales_units.sum()),
        "bias": float((g.P50 - g.sales_units).mean()),
    }), include_groups=False)
    .reset_index()
    .sort_values("units", ascending=False)
)
err_f.to_csv(OUT / "error_by_family.csv", index=False)

# coverage per family, on the valid window (where actuals exist)
cov_f = (
    valid.groupby("family")
    .apply(lambda g: pd.Series({q: float((g.sales_units <= g[q]).mean()) for q in ["P50", "P80", "P90", "P95", "P98"]}),
           include_groups=False)
    .reset_index()
)
cov_f.to_csv(OUT / "coverage_by_family.csv", index=False)

# raw vs conformal-corrected P90, overall and by family
cov_fix = pd.DataFrame({
    "family": sorted(valid.family.unique()),
}).set_index("family")
cov_fix["raw"] = valid.groupby("family").apply(lambda g: float((g.sales_units <= g.P90_raw).mean()), include_groups=False)
cov_fix["corrected"] = valid.groupby("family").apply(lambda g: float((g.sales_units <= g.P90).mean()), include_groups=False)
cov_fix.reset_index().to_csv(OUT / "p90_before_after.csv", index=False)
print("error + coverage decompositions written")

# ----------------------------------------------------------------- series meta
meta_cols = ["date", "store_nbr", "family", "sales_units", "items_on_promo",
             "city", "state", "store_type", "cluster", "is_holiday"]
panel = pd.read_parquet(PANEL, columns=meta_cols)
panel["date"] = pd.to_datetime(panel["date"])

meta = (
    panel.groupby(["store_nbr", "family"], observed=True)
    .agg(city=("city", "first"), state=("state", "first"), store_type=("store_type", "first"),
         cluster=("cluster", "first"), mean_units=("sales_units", "mean"),
         total_units=("sales_units", "sum"), zero_share=("sales_units", lambda s: float((s == 0).mean())))
    .reset_index()
)
groups = pd.read_csv(INV / "inventory" / "assumptions_family_table.csv")[["family", "group", "lead_time", "shelf_life", "q_star"]]
meta = meta.merge(groups, on="family", how="left")
meta.to_csv(OUT / "series_meta.csv", index=False)
print(f"series_meta {len(meta):,} series")

hist = panel[panel.date >= HISTORY_FROM][
    ["date", "store_nbr", "family", "sales_units", "items_on_promo", "is_holiday"]
].copy()
hist["sales_units"] = hist["sales_units"].astype("float32")
hist.to_parquet(OUT / "history.parquet", index=False)
print(f"history {len(hist):,} rows from {HISTORY_FROM}")

# daily panel totals for the overview sparkline
# stop at the last day with recorded sales; the test split carries NaN targets
daily = (panel[panel.date <= "2017-08-15"]
         .groupby("date", as_index=False).agg(units=("sales_units", "sum")))
daily[daily.date >= "2016-01-01"].to_csv(OUT / "daily_totals.csv", index=False)
del panel

# ---------------------------------------------------------------- the ledger
led = pd.read_parquet(INV / "inventory" / "ledger.parquet")
led["date"] = pd.to_datetime(led["date"])
cost_cols = ["holding_cost", "stockout_cost", "waste_cost", "total_cost"]
flow = ["demand", "sold", "lost", "expired", "order_qty", "received", "on_hand_close"]

# policy x date totals, so the app can draw cost and service over time
by_day = led.groupby(["policy", "date"], observed=True).agg(
    **{c: (c, "sum") for c in cost_cols + flow},
    stockout_days=("lost", lambda s: int((s > 0).sum())),
    n=("lost", "size"),
    scored=("scored", "max"),
).reset_index()
by_day["fill_rate"] = by_day["sold"] / (by_day["sold"] + by_day["lost"])
by_day.to_csv(OUT / "ledger_by_day.csv", index=False)

# policy x family, to show where each policy wins or loses
by_fam = led[led.scored].groupby(["policy", "family", "group"], observed=True).agg(
    **{c: (c, "sum") for c in cost_cols + flow},
    stockout_days=("lost", lambda s: int((s > 0).sum())),
    n=("lost", "size"),
).reset_index()
by_fam["fill_rate"] = by_fam["sold"] / (by_fam["sold"] + by_fam["lost"])
by_fam.to_csv(OUT / "ledger_by_family.csv", index=False)

# per-series summary on the scored window, for the exception list
by_series = led[led.scored].groupby(["policy", "store_nbr", "family"], observed=True).agg(
    demand=("demand", "sum"), sold=("sold", "sum"), lost=("lost", "sum"),
    expired=("expired", "sum"), total_cost=("total_cost", "sum"),
    mean_on_hand=("on_hand_close", "mean"),
    stockout_days=("lost", lambda s: int((s > 0).sum())),
).reset_index()
by_series["fill_rate"] = by_series["sold"] / (by_series["sold"] + by_series["lost"]).replace(0, np.nan)
by_series.to_parquet(OUT / "ledger_by_series.parquet", index=False)

# the closing position on the last simulated day: what the agent starts from
last_day = led.date.max()
state = led[(led.date == last_day)][
    ["policy", "store_nbr", "family", "group", "on_hand_close", "on_order_close"]
].rename(columns={"on_hand_close": "on_hand", "on_order_close": "on_order"})
state.to_parquet(OUT / "closing_state.parquet", index=False)

# one policy's full daily trace per series, for the explorer's inventory panel
trace = led[led.policy == "P3b_newsvendor_q*_tuned"][
    ["date", "store_nbr", "family", "on_hand_open", "order_qty", "demand", "sold", "lost", "expired"]
].copy()
for c in ["on_hand_open", "order_qty", "demand", "sold", "lost", "expired"]:
    trace[c] = trace[c].astype("float32")
trace.to_parquet(OUT / "ledger_trace_p3b.parquet", index=False)
print(f"ledger summaries written (last simulated day {last_day.date()})")
del led

# ------------------------------------------------------------------- headline
bt = pd.read_csv(OUT / "backtest_results.csv")
pch = pd.read_csv(OUT / "policy_comparison_holdout.csv").set_index("policy")
best = pch["total_cost"].drop("P5_perfect_foresight").idxmin()

kpi = {
    "generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
    "series": int(len(meta)),
    "stores": int(meta.store_nbr.nunique()),
    "families": int(meta.family.nunique()),
    "horizon_days": 16,
    "hgb_rmsle_mean": float(bt.hgb_rmsle.mean()),
    "hgb_rmsle_sd": float(bt.hgb_rmsle.std()),
    "naive_rmsle_mean": float(bt.naive_rmsle.mean()),
    "ma_rmsle_mean": float(bt.ma_rmsle.mean()),
    "rf_rmsle_mean": float(bt.rf_rmsle.mean()),
    "ridge_rmsle_mean": float(bt.ridge_rmsle.mean()),
    "holdout_gain_vs_naive": float((bt.naive_rmsle.iloc[-1] - bt.hgb_rmsle.iloc[-1]) / bt.naive_rmsle.iloc[-1]),
    # measured from the delivered v2 quantiles, not copied from either report:
    # B's write-up quotes 0.861 -> 0.898, C's recomputation found 0.879, and the
    # shipped files agree with C. The gate is 0.88 to 0.92, so it is missed.
    "p90_before": float((valid.sales_units <= valid.P90_raw).mean()),
    "p90_after": float((valid.sales_units <= valid.P90).mean()),
    "p90_gate_met": bool(0.88 <= float((valid.sales_units <= valid.P90).mean()) <= 0.92),
    "p10_below": float((valid.sales_units < valid.P10).mean()),
    "valid_p50_rmsle": rmsle(valid.sales_units, valid.P50),
    "best_policy": best,
    "best_fill": float(pch.loc[best, "fill_rate"]),
    "base_fill": float(pch.loc["P1_moving_avg_7d", "fill_rate"]),
    "best_cost": float(pch.loc[best, "total_cost"]),
    "base_cost": float(pch.loc["P1_moving_avg_7d", "total_cost"]),
    "cost_saving": float(-pch.loc[best, "cost_vs_P1"]),
    "best_stockout_days": int(pch.loc[best, "stockout_days"]),
    "base_stockout_days": int(pch.loc["P1_moving_avg_7d", "stockout_days"]),
    "sim_window": "2017-07-31 to 2017-08-15",
    "forecast_window": f"{test.date.min().date()} to {test.date.max().date()}",
    "ledger_rows": 497886,
}
(OUT / "kpi.json").write_text(json.dumps(kpi, indent=2))
print(json.dumps(kpi, indent=2))

total = sum(p.stat().st_size for p in OUT.iterdir())
print(f"\ndata pack: {len(list(OUT.iterdir()))} files, {total/1e6:.1f} MB")
