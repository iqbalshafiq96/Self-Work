"""Refinery Production Planning Optimizer – interactive workflow canvas (Streamlit host).

    streamlit run app.py

The canvas (flow_component/index.html) is a two-way Streamlit component:
  Python -> canvas : tables, validation, model coefficients, optimizer results (JSON "payload")
  canvas -> Python : user actions {nonce, action, ...} e.g. edit a table, change settings, run, reset
Every action re-validates, re-solves the LP (SciPy HiGHS) and re-renders the canvas.
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import data_loader
from model import coeffs, evaluate, margin_per_unit
from optimizer import CONSTRAINT_LABEL, binding_summary, explain, optimize

DEFAULT_SOURCE = "https://github.com/iqbalshafiq96/Self-Work/tree/main/Refinery_Optimizer"
INPUT_TABLES = ["crude_data", "unit_data", "product_data", "tank_master",
                "crude_receipts", "product_liftings", "daily_decisions"]
TABLE_META = {
    "crude_data": dict(readonly=["crude_id", "tank_id"],
                       desc="Delivered price, CDU cut yields (must sum to 1.000), residue sulfur and charge limits per crude."),
    "unit_data": dict(readonly=["param_key", "symbol", "unit", "parameter", "uom", "used_in", "comment"],
                      desc="Unit capacities, ramp limit, variable opex, conversion-unit yields and blend specs. Edit the 'value' column."),
    "product_data": dict(readonly=["product_id", "tank_id", "make_formula", "produced_from"],
                         desc="Product netback prices. 'make_formula' shows how each product is made from x, r, f."),
    "tank_master": dict(readonly=["tank_id", "service", "material_id", "type"],
                        desc="Tank working capacity, minimum heel, opening inventory (day 1) and end-of-horizon target."),
    "crude_receipts": dict(readonly=["receipt_id", "date", "tank_id"], dynamic=True,
                           desc="Crude cargo arrivals. Change 'day' to simulate a vessel delay; add or delete cargoes."),
    "product_liftings": dict(readonly=["lifting_id", "date", "tank_id"], dynamic=True,
                             desc="Product liftings / cargoes. Change 'day' or 'qty_kbbl' for demand scenarios."),
    "daily_decisions": dict(readonly=["day", "date", "status"],
                            desc="Recorded rates. Rows before the first plan day are locked actuals; Plan rows are the current (manual) plan used as the comparison baseline."),
}

st.set_page_config(page_title="Refinery Planning Flow", page_icon="🛢️", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:2.6rem;padding-bottom:1rem;max-width:100%}</style>",
            unsafe_allow_html=True)
flow_canvas = components.declare_component("flow_canvas", path=str(Path(__file__).parent / "flow_component"))


# ---------------------------------------------------------------- helpers
def clean(o):
    """Make any nested structure JSON-safe (numpy -> python, NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.ndarray, pd.Series, pd.Index)):
        return clean(list(o.tolist()))
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict("records"))
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    return o


def table_payload(name, df, crudes, products):
    meta = TABLE_META[name]
    selects = {}
    if name == "crude_receipts":
        selects = {"crude_id": crudes}
    elif name == "product_liftings":
        selects = {"product_id": products}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if name == "unit_data":
        numeric = ["value"]
    return {"columns": list(df.columns), "rows": clean(df.astype(object).values.tolist()),
            "readonly": meta["readonly"], "dynamic": meta.get("dynamic", False), "selects": selects,
            "numeric": numeric, "desc": meta["desc"], "file": data_loader.FILES[name]}


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        return a.shape == b.shape and a.astype(str).reset_index(drop=True).equals(b.astype(str).reset_index(drop=True))
    except Exception:
        return False


def rows_to_frame(name, columns, rows, frames):
    """Convert edited rows from the canvas back into a typed DataFrame."""
    df = pd.DataFrame(rows, columns=columns)
    orig = st.session_state.base_frames[name]
    for c in df.columns:
        if c in orig.columns and pd.api.types.is_numeric_dtype(orig[c]) or (name == "unit_data" and c == "value"):
            df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    if name in ("crude_receipts", "product_liftings"):
        idcol, prefix = ("crude_id", "R") if name == "crude_receipts" else ("product_id", "L")
        key = "receipt_id" if name == "crude_receipts" else "lifting_id"
        tank_of = dict(zip(frames["tank_master"]["material_id"], frames["tank_master"]["tank_id"]))
        dates = dict(zip(pd.to_numeric(frames["daily_decisions"]["day"]), frames["daily_decisions"]["date"]))
        df = df.sort_values("day", kind="stable").reset_index(drop=True)
        df[key] = [f"{prefix}{i + 1:03d}" for i in range(len(df))]
        if "tank_id" in df.columns:
            df["tank_id"] = df[idcol].map(tank_of)
        if "date" in df.columns:
            df["date"] = df["day"].map(dates)
    return df


def default_first_plan_day(frames):
    dec = frames["daily_decisions"]
    plan = dec.loc[dec["status"] == "Plan", "day"]
    return int(plan.min()) if len(plan) else len(dec)


def load_source(url):
    frames, sources = data_loader.read_csvs_from_url(url)
    st.session_state.base_frames = {k: v.copy() for k, v in frames.items()}
    st.session_state.frames = {k: v.copy() for k, v in frames.items()}
    st.session_state.sources = sources
    st.session_state.settings = {"first_plan_day": default_first_plan_day(frames), "soft": False, "penalty": 1000.0}
    st.session_state.solve_cache = {}


# ---------------------------------------------------------------- state
if "frames" not in st.session_state:
    try:
        st.session_state.source_url = st.secrets.get("DATA_URL", DEFAULT_SOURCE)
    except Exception:  # no secrets.toml
        st.session_state.source_url = DEFAULT_SOURCE
    try:
        load_source(st.session_state.source_url)
    except Exception:
        st.session_state.source_url = DEFAULT_SOURCE
        load_source(DEFAULT_SOURCE)
    st.session_state.last_nonce = None

with st.sidebar:
    st.header("Data source")
    url = st.text_input("GitHub folder or raw URL", st.session_state.source_url,
                        help="Folder containing the CSVs (directly or in a data/ subfolder).")
    if st.button("⟳ Load from GitHub", use_container_width=True):
        st.session_state.source_url = url
        load_source(url)
        st.rerun()
    src = st.session_state.sources
    n_gh = sum(v != "local" for v in src.values())
    st.caption(f"{n_gh}/{len(src)} files from GitHub, {len(src) - n_gh} from the bundled data/ folder.")
    for k, v in src.items():
        st.caption(f"• {data_loader.FILES[k]} — {'local' if v == 'local' else 'GitHub'}")

frames = st.session_state.frames
settings = st.session_state.settings
T = len(frames["daily_decisions"])
settings["first_plan_day"] = int(min(max(settings["first_plan_day"], 1), T))

# ---------------------------------------------------------------- compute
issues = data_loader.validate(frames)
errors = [m for lvl, m in issues if lvl == "error"]
crudes = frames["crude_data"]["crude_id"].astype(str).tolist() if "crude_data" in frames else []
products = frames["product_data"]["product_id"].astype(str).tolist() if "product_data" in frames else []

payload = {
    "ack": st.session_state.last_nonce,
    "source": {"url": st.session_state.source_url,
               "github_files": sum(v != "local" for v in st.session_state.sources.values()),
               "total_files": len(st.session_state.sources)},
    "tables": {n: table_payload(n, frames[n], crudes, products) for n in INPUT_TABLES},
    "edited": [n for n in INPUT_TABLES if not frames_equal(frames[n], st.session_state.base_frames[n])],
    "issues": [{"level": l, "msg": m} for l, m in issues],
    "settings": settings | {"T": T},
    "summary": {}, "model": None, "result": None, "status": "invalid" if errors else "pending",
}

opt_bundle = None
if not errors:
    data = data_loader.build(frames, settings["first_plan_day"])
    base = evaluate(data)
    fingerprint = (tuple(int(pd.util.hash_pandas_object(f.astype(str), index=False).sum()) for f in frames.values()),
                   settings["first_plan_day"], settings["soft"], float(settings["penalty"]))
    cache = st.session_state.solve_cache
    if fingerprint not in cache:
        res = optimize(data, soft_tanks=settings["soft"], penalty_usd_bbl=float(settings["penalty"]))
        if len(cache) > 8:
            cache.clear()
        cache[fingerprint] = res
    res = cache[fingerprint]
    opt = evaluate(data, res.V) if res.success else None
    view = opt or base
    M = coeffs(data)
    mpu = margin_per_unit(data)
    P = data.P
    days = data.decisions["day"].tolist()

    # ---- model (process + LP formulation, for the calculation blocks) ----
    payload["model"] = {
        "var_names": data.var_names, "crudes": crudes,
        "crude_names": data.crude["crude_name"].tolist(),
        "products": products, "product_names": data.product["product_name"].tolist(),
        "P": {k: float(v) for k, v in P.items()},
        "crude": clean(data.crude.reset_index().to_dict("records")),
        "margin_per_unit": clean(mpu.to_dict("records")),
        "make": {p: clean(M[f"MAKE_{p}"]) for p in products},
        "dht": clean(M["DHT"]), "sulfur_row": clean(M["S_EXCESS"]), "ron_row": clean(M["RON_EXCESS"]),
        "n_rows": res.n_rows, "n_vars": res.n_vars,
        "tanks": clean(data.tank.reset_index()[["tank_id", "service", "type", "capacity_kbbl", "min_heel_kbbl",
                                                  "opening_kbbl", "end_target_kbbl"]].to_dict("records")),
    }

    # ---- results ----
    def tank_series(ev):
        tb = ev["tank_balance"]
        return {tid: g.sort_values("day") for tid, g in tb.groupby("tank_id")}

    tb_base, tb_view = tank_series(base), tank_series(view)
    tanks = []
    for tid, row in data.tank.iterrows():
        g, gb = tb_view[tid], tb_base[tid]
        tanks.append({"id": tid, "service": row["service"], "type": row["type"],
                      "cap": row["capacity_kbbl"], "heel": row["min_heel_kbbl"], "target": row["end_target_kbbl"],
                      "opening": g["opening"].tolist(), "inflow": g["inflow"].tolist(), "outflow": g["outflow"].tolist(),
                      "closing": g["closing"].tolist(), "closing_base": gb["closing"].tolist(),
                      "status": g["status"].tolist()})
    dpv, dpb = view["daily_plan"], base["daily_plan"]
    plan_mask = dpb["status"] == "Plan"
    econ_cols = ["revenue_kusd", "crude_cost_kusd", "opex_kusd", "margin_kusd"]
    bsum = binding_summary(data, res) if res.success else pd.DataFrame()
    check_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C13"]
    payload["result"] = {
        "success": res.success, "message": res.message,
        "days": days, "dates": data.decisions["date"].astype(str).tolist(),
        "status": dpb["status"].tolist(), "first_plan_day": data.first_plan_day,
        "V_base": clean(np.round(base["daily_plan"][data.var_names].to_numpy(float), 3)),
        "V_opt": clean(np.round(view["daily_plan"][data.var_names].to_numpy(float), 3)),
        "kpi_base": base["kpi"], "kpi_opt": opt["kpi"] if opt else None,
        "benchmark": float(data.cfg["benchmark_lp_optimum_kusd"]) if "benchmark_lp_optimum_kusd" in data.cfg else None,
        "penalty_kusd": res.penalty_kusd,
        "margin_base": clean(dpb["margin_kusd"].round(2)), "margin_opt": clean(dpv["margin_kusd"].round(2)),
        "cdu_opt": clean(dpv["cdu_total"].round(2)),
        "ron_opt": clean(dpv["gasoline_ron"].round(2)), "sulfur_opt": clean(dpv["residue_s_wt"].round(3)),
        "econ": {"labels": ["Product revenue", "Crude cost", "Opex", "Gross margin"],
                 "base": clean(dpb.loc[plan_mask, econ_cols].sum().tolist()),
                 "opt": clean(dpv.loc[plan_mask, econ_cols].sum().tolist())},
        "make": [{"id": p, "name": data.product.loc[p, "product_name"],
                  "price": data.product.loc[p, "price_usd_bbl"],
                  "base": dpb.loc[plan_mask, f"make_{p}"].sum(), "opt": dpv.loc[plan_mask, f"make_{p}"].sum()}
                 for p in products],
        "tanks": tanks,
        "c14": clean(view["c14"].to_dict("records")),
        "checks": {c: int((~view["checks"][c]).sum()) for c in check_cols} | {
            "C14": int((~view["c14"]["C14_ok"]).sum())},
        "binding": clean(bsum.to_dict("records")),
        "binding_count": clean(res.binding.groupby("constraint").size().to_dict()) if res.success and not res.binding.empty else {},
        "explain": explain(data, base["kpi"], opt["kpi"], res) if opt else [res.message],
        "shortfalls": clean(res.shortfalls.to_dict("records")),
    }
    payload["status"] = "optimal" if res.success else "infeasible"
    opt_bundle = (data, opt, res) if opt else None

    # ---- one-line stats shown on each block ----
    cr, pr, tk = frames["crude_data"], frames["product_data"], frames["tank_master"]
    top = mpu.sort_values("margin_usd_bbl", ascending=False).iloc[0]
    kv = (opt or base)["kpi"]
    payload["summary"] = {
        "crude_data": f"{len(cr)} crudes · {cr.price_usd_bbl.min():g}–{cr.price_usd_bbl.max():g} $/bbl",
        "unit_data": f"CDU {P['CDU_MAX']:g} · REF {P['REF_MAX']:g} · RFCC {P['RFCC_MAX']:g} · DHT {P['DHT_MAX']:g} kbd",
        "product_data": f"{len(pr)} products · {pr.price_usd_bbl.min():g}–{pr.price_usd_bbl.max():g} $/bbl",
        "tank_master": f"{len(tk)} tanks · {tk.opening_kbbl.sum():,.0f} kbbl opening",
        "crude_receipts": f"{len(frames['crude_receipts'])} cargoes · {frames['crude_receipts'].qty_kbbl.sum():,.0f} kbbl",
        "product_liftings": f"{len(frames['product_liftings'])} liftings · {frames['product_liftings'].qty_kbbl.sum():,.0f} kbbl",
        "daily_decisions": f"Actual 1–{data.first_plan_day - 1} · Plan {data.first_plan_day}–{data.T}",
        "validate": "All checks passed" if not issues else f"{len(issues)} warning(s)",
        "model": f"Best: {top['var']} {top['margin_usd_bbl']:.2f} $/bbl",
        "optimizer": f"Z = {kv['plan_margin_kusd']:,.0f} $k" if opt else "No feasible plan",
        "decisions": f"{(data.T - data.first_plan_day + 1) * len(data.var_names)} variables solved" if opt else "Showing current plan",
        "economics": (f"{opt['kpi']['plan_margin_kusd'] - base['kpi']['plan_margin_kusd']:+,.0f} $k vs current plan"
                      if opt else f"{base['kpi']['plan_margin_kusd']:,.0f} $k current"),
        "tank_balance": f"{sum((~view['checks']['C13'])) } breach day(s) · C14 {int(view['c14']['C14_ok'].sum())}/{len(tk)} OK",
        "tank_curves": "Closing inventory vs limits",
        "why": (bsum.iloc[0]["label"] + " is the top constraint") if len(bsum) else res.message[:60],
    }
else:
    payload["summary"] = {"validate": f"{len(errors)} error(s) – click to see"}

# ---------------------------------------------------------------- canvas
st.markdown("#### Refinery Production Planning – interactive optimizer flow")
action = flow_canvas(payload=clean(payload), key="flow", default=None)

if action and action.get("nonce") and action["nonce"] != st.session_state.last_nonce:
    st.session_state.last_nonce = action["nonce"]
    kind = action.get("action")
    if kind == "edit" and action.get("table") in INPUT_TABLES:
        frames[action["table"]] = rows_to_frame(action["table"], action["columns"], action["rows"], frames)
    elif kind == "reset_table" and action.get("table") in INPUT_TABLES:
        frames[action["table"]] = st.session_state.base_frames[action["table"]].copy()
    elif kind == "reset_all":
        st.session_state.frames = {k: v.copy() for k, v in st.session_state.base_frames.items()}
        settings["first_plan_day"] = default_first_plan_day(st.session_state.base_frames)
        settings["soft"] = False
    elif kind == "settings":
        settings["first_plan_day"] = int(action.get("first_plan_day", settings["first_plan_day"]))
        settings["soft"] = bool(action.get("soft", settings["soft"]))
        settings["penalty"] = float(min(max(float(action.get("penalty", settings["penalty"])), 1.0), 1e6))
    elif kind == "run":
        st.session_state.solve_cache = {}
    st.rerun()

# ---------------------------------------------------------------- downloads
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, df in st.session_state.frames.items():
            z.writestr(data_loader.FILES[n], df.to_csv(index=False))
    st.download_button("⬇ Edited input CSVs", buf.getvalue(), "inputs_edited.zip", "application/zip",
                       use_container_width=True, help="Commit these to GitHub to make the scenario the new base case.")
with c2:
    if opt_bundle:
        data, opt, res = opt_bundle
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            dec = opt["daily_plan"][["day", "date", "status"] + data.var_names]
            z.writestr("daily_decisions_optimized.csv", dec.to_csv(index=False))
            z.writestr("daily_plan.csv", opt["daily_plan"].to_csv(index=False))
            z.writestr("tank_balance.csv", opt["tank_balance"].to_csv(index=False))
            z.writestr("constraint_check.csv", opt["checks"].to_csv(index=False))
            z.writestr("binding_constraints.csv", res.binding.to_csv(index=False))
        st.download_button("⬇ Optimizer results", buf.getvalue(), "optimizer_results.zip", "application/zip",
                           use_container_width=True)
with c3:
    st.caption("Click any block on the canvas to open it. Input blocks are editable; every change re-validates "
               "and re-solves the LP. Sample data are illustrative only.")
