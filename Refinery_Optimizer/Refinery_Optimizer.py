"""Refinery Production Planning Optimizer - Streamlit web app.

Run locally:   streamlit run app.py
Pipeline:      CSV inputs -> validation -> LP optimizer -> daily plan + economics -> explanation
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_loader
from model import evaluate, margin_per_unit
from optimizer import explain, optimize

st.set_page_config(page_title="Refinery Planning Optimizer", page_icon="🛢️", layout="wide")

LABELS = {
    "crude_data": "Crude data", "unit_data": "Unit data", "product_data": "Product data",
    "tank_master": "Tank master & opening", "crude_receipts": "Crude receipts",
    "product_liftings": "Product liftings", "daily_decisions": "Daily decisions (actuals)", "config": "Config",
}

# ------------------------------------------------------------------ state
if "base_frames" not in st.session_state:
    st.session_state.base_frames = data_loader.read_csvs()
    st.session_state.editor_ver = 0
if "result" not in st.session_state:
    st.session_state.result = None


def reset_to(frames):
    st.session_state.base_frames = frames
    st.session_state.editor_ver += 1          # new widget keys -> editors reload clean data
    st.session_state.result = None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Run settings")
    up = st.file_uploader("Upload your own CSVs (same file names)", type="csv", accept_multiple_files=True)
    if up and st.button("Load uploaded CSVs", use_container_width=True):
        frames = {k: v.copy() for k, v in st.session_state.base_frames.items()}
        names = {v: k for k, v in data_loader.FILES.items()}
        loaded = []
        for f in up:
            if f.name in names:
                frames[names[f.name]] = pd.read_csv(f)
                loaded.append(f.name)
        reset_to(frames)
        st.success(f"Loaded: {', '.join(loaded) or 'no matching file names'}")
    if st.button("↺ Reset to sample data", use_container_width=True):
        reset_to(data_loader.read_csvs())

    T = len(st.session_state.base_frames["daily_decisions"])
    default_fp = int(st.session_state.base_frames["daily_decisions"]
                     .pipe(lambda d: d.loc[d.status == "Plan", "day"].min() if (d.status == "Plan").any() else T))
    first_plan_day = st.number_input("First plan day (days before = Actual, locked)", 1, T, default_fp, key="fpd",
                                     help="Rolling horizon: move this forward each day after entering actuals.")
    soft = st.toggle("Soft tank limits", value=False, key="soft",
                     help="If a scenario makes tank min-heel / end targets impossible, allow a penalised shortfall "
                          "instead of returning 'infeasible'.")
    penalty = st.number_input("Shortfall penalty ($/bbl)", 10.0, 100000.0, 1000.0, 100.0, disabled=not soft)

    st.divider()
    st.subheader("🚢 Quick scenario: cargo delay")
    rec = st.session_state.base_frames["crude_receipts"]
    cargo = st.selectbox("Cargo", key="cargo", options= ["(none)"] + [f"{r.receipt_id} | day {r.day} | {r.crude_id} {r.qty_kbbl} kbbl"
                                                 for r in rec.itertuples()])
    delay = st.number_input("Delay (days)", 0, T, 0, key="delay", disabled=cargo == "(none)")

# ------------------------------------------------------------------ header
st.title("🛢️ Refinery Production Planning Optimizer")
st.caption("CSV inputs → validation → LP optimizer (SciPy HiGHS) → daily plan & economics → explanation. "
           "Sample data are illustrative only.")

tabs = st.tabs(["1 · Inputs", "2 · Validation", "3 · Optimize", "4 · Daily plan",
                "5 · Tank balance", "6 · Economics", "7 · Why? (explain)"])

# ------------------------------------------------------------------ 1. inputs
with tabs[0]:
    st.markdown("Edit any table below - changes are used on the next optimizer run. "
                "Each tab is one CSV file in `data/`.")
    frames = {}
    sub = st.tabs([LABELS[k] for k in data_loader.FILES])
    for tab, name in zip(sub, data_loader.FILES):
        with tab:
            dyn = name in ("crude_receipts", "product_liftings")
            frames[name] = st.data_editor(
                st.session_state.base_frames[name], key=f"ed_{name}_{st.session_state.editor_ver}",
                num_rows="dynamic" if dyn else "fixed", use_container_width=True, hide_index=True)
    with st.expander("Column dictionary"):
        st.dataframe(pd.read_csv(data_loader.DATA_DIR / "data_dictionary.csv"), hide_index=True,
                     use_container_width=True)

# apply the quick cargo-delay scenario on top of the edited tables
if cargo != "(none)" and delay > 0:
    rid = cargo.split(" | ")[0]
    r = frames["crude_receipts"].copy()
    mask = r["receipt_id"] == rid
    new_day = int(min(r.loc[mask, "day"].iloc[0] + delay, T))
    r.loc[mask, "day"] = new_day
    if "date" in r.columns:
        r.loc[mask, "date"] = frames["daily_decisions"].loc[new_day - 1, "date"]
    frames["crude_receipts"] = r
    st.info(f"Scenario active: cargo **{rid}** delayed by {delay} day(s) → arrives day {new_day}.")

# ------------------------------------------------------------------ 2. validation
issues = data_loader.validate(frames)
errors = [m for lvl, m in issues if lvl == "error"]
with tabs[1]:
    if not issues:
        st.success("All input checks passed ✔")
    for lvl, m in issues:
        (st.error if lvl == "error" else st.warning)(m)
    st.markdown("""
**Checks performed:** required columns · crude yields sum to 1 · min ≤ max for every unit and crude ·
reformer/RFCC yields ≤ 1 · each crude/product mapped to exactly one tank · opening stock inside tank window ·
receipts/liftings reference known IDs and valid days · decisions complete for every day.
""")

if errors:
    with tabs[2]:
        st.error("Fix the input errors in tab 2 before optimizing.")
    st.stop()

data = data_loader.build(frames, int(first_plan_day))
base = evaluate(data)
# fingerprint of everything that affects the solution -> detect stale results
fingerprint = (tuple(int(pd.util.hash_pandas_object(f.astype(str), index=False).sum()) for f in frames.values()),
               data.first_plan_day, soft, penalty)

# ------------------------------------------------------------------ 3. optimize
with tabs[2]:
    c1, c2 = st.columns([1, 3])
    with c1:
        run = st.button("▶ Run optimizer", type="primary", use_container_width=True)
    with c2:
        st.caption(f"Horizon {data.T} days · days 1–{data.first_plan_day - 1} locked as Actual · "
                   f"{(data.T - data.first_plan_day + 1) * len(data.var_names)} decision variables")
    if run:
        with st.spinner("Solving LP..."):
            res = optimize(data, soft_tanks=soft, penalty_usd_bbl=penalty)
        st.session_state.result = (res, fingerprint) if res.success else None
        if not res.success:
            st.error(res.message)

    stored = st.session_state.result
    if stored and stored[1] == fingerprint:
        res = stored[0]
        opt = evaluate(data, res.V)
    else:
        res, opt = None, None
        if stored:
            st.warning("Inputs or settings changed since the last run - press **Run optimizer** again.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current plan margin (plan days)", f"{base['kpi']['plan_margin_kusd']:,.0f} $k",
              help="Decisions as stored in daily_decisions.csv")
    if opt:
        up_ = opt["kpi"]["plan_margin_kusd"] - base["kpi"]["plan_margin_kusd"]
        k2.metric("Optimised margin (plan days)", f"{opt['kpi']['plan_margin_kusd']:,.0f} $k", f"{up_:+,.0f} $k")
        k3.metric("Constraint violations", opt["kpi"]["violations"])
        k4.metric("Crude run (month)", f"{opt['kpi']['crude_run_kbbl']:,.0f} kbbl")
        bm = data.cfg.get("benchmark_lp_optimum_kusd")
        if bm is not None and data.first_plan_day == 12 and not soft:
            st.caption(f"Workbook reference optimum for plan days 12–31: {float(bm):,.0f} $k "
                       f"(gap {opt['kpi']['plan_margin_kusd'] - float(bm):+.2f} $k).")
        if res.penalty_kusd > 0:
            st.warning(f"Soft mode used: tank shortfall penalty {res.penalty_kusd:,.0f} $k - see tab 7.")

        dec = opt["daily_plan"][["day", "date", "status"] + data.var_names].round(3)
        st.subheader("Optimised decisions")
        st.dataframe(dec.style.apply(lambda r: ["background-color: rgba(0,160,80,.15)" if r.status == "Plan"
                                                else "" for _ in r], axis=1),
                     use_container_width=True, hide_index=True, height=360)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("daily_decisions_optimized.csv", dec.to_csv(index=False))
            z.writestr("daily_plan.csv", opt["daily_plan"].to_csv(index=False))
            z.writestr("tank_balance.csv", opt["tank_balance"].to_csv(index=False))
            z.writestr("constraint_check.csv", opt["checks"].to_csv(index=False))
            z.writestr("binding_constraints.csv", res.binding.to_csv(index=False))
        st.download_button("⬇ Download results (zip)", buf.getvalue(), "optimizer_results.zip", "application/zip")
    else:
        k2.metric("Optimised margin (plan days)", "—")
        st.info("Press **Run optimizer** to solve the plan days.")

view = opt if opt else base
view_name = "Optimised" if opt else "Current"

# ------------------------------------------------------------------ 4. daily plan
with tabs[3]:
    st.subheader(f"{view_name} daily plan (kbbl/d, $k/d)")
    dp = view["daily_plan"]
    groups = {
        "Decisions": ["day", "date", "status"] + data.var_names,
        "Units & cuts": ["day", "cdu_total", "sr_lpg", "sr_naphtha", "sr_kero", "sr_gasoil", "atm_residue",
                         "reformate", "ref_lpg", "fcc_gasoline", "fcc_lpg", "lco", "slurry", "dht_feed"],
        "Products": ["day"] + [f"make_{p}" for p in data.products],
        "Quality": ["day", "gasoline_ron", "residue_s_wt"],
        "Economics": ["day", "revenue_kusd", "crude_cost_kusd", "opex_kusd", "margin_kusd", "margin_usd_bbl"],
    }
    g = st.radio("Columns", list(groups), horizontal=True)
    st.dataframe(dp[groups[g]].round(3), use_container_width=True, hide_index=True, height=420)

    fig = go.Figure()
    for i, c in enumerate(data.crudes):
        fig.add_bar(x=dp["day"], y=dp[f"x_{c}"], name=data.crude.loc[c, "crude_name"])
    fig.add_vline(x=data.first_plan_day - 0.5, line_dash="dash", annotation_text="Plan →")
    fig.update_layout(barmode="stack", title="Crude slate to CDU", xaxis_title="Day",
                      yaxis_title="kbbl/d", height=380, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ 5. tanks
with tabs[4]:
    tb_base, tb_view = base["tank_balance"], view["tank_balance"]
    tank_opts = [f"{t} – {r.service} ({r.type})" for t, r in data.tank.iterrows()]
    tid = st.selectbox("Tank", tank_opts, key="tank_pick").split(" – ")[0]
    a, b_ = tb_base[tb_base.tank_id == tid], tb_view[tb_view.tank_id == tid]
    row = data.tank.loc[tid]
    fig = go.Figure()
    fig.add_scatter(x=a["day"], y=a["closing"], name="Current plan", line=dict(dash="dot"))
    if opt:
        fig.add_scatter(x=b_["day"], y=b_["closing"], name="Optimised", line=dict(width=3))
    fig.add_hline(y=row["capacity_kbbl"], line_color="red", annotation_text="Capacity")
    fig.add_hline(y=row["min_heel_kbbl"], line_color="orange", annotation_text="Min heel")
    fig.add_scatter(x=[data.T], y=[row["end_target_kbbl"]], mode="markers", name="End-of-month target",
                    marker=dict(symbol="triangle-up", size=14))
    fig.add_vline(x=data.first_plan_day - 0.5, line_dash="dash")
    fig.update_layout(title=f"{tid} closing inventory (kbbl)", xaxis_title="Day", height=420,
                      margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    wide = tb_view.pivot(index="day", columns="tank_id", values="fill_pct")
    st.markdown(f"**{view_name} fill % – all tanks**")
    st.dataframe(wide.style.format("{:.0%}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=1),
                 use_container_width=True, height=360)
    st.markdown("**C14 – end-of-month targets**")
    st.dataframe(view["c14"].round(2), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 6. economics
with tabs[5]:
    dp_b, dp_v = base["daily_plan"], view["daily_plan"]
    plan = dp_b["status"] == "Plan"
    fig = go.Figure()
    fig.add_bar(x=dp_b["day"], y=dp_b["margin_kusd"], name="Current plan")
    if opt:
        fig.add_bar(x=dp_v["day"], y=dp_v["margin_kusd"], name="Optimised")
    fig.add_vline(x=data.first_plan_day - 0.5, line_dash="dash")
    fig.update_layout(barmode="group", title="Daily gross margin ($k/d)", height=380, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    cols = ["revenue_kusd", "crude_cost_kusd", "opex_kusd", "margin_kusd"]
    summ = pd.DataFrame({"Current plan": dp_b.loc[plan, cols].sum()})
    if opt:
        summ["Optimised"] = dp_v.loc[plan, cols].sum()
        summ["Change"] = summ["Optimised"] - summ["Current plan"]
    summ.index = ["Product revenue", "Crude cost", "Opex", "Gross margin"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Plan-days economics ($k)**")
        st.dataframe(summ.style.format("{:,.0f}"), use_container_width=True)
        prod_cols = [f"make_{p}" for p in data.products]
        rev = pd.DataFrame({"product": data.products,
                            "volume_kbbl": dp_v.loc[plan, prod_cols].sum().values,
                            "price_usd_bbl": data.product["price_usd_bbl"].values})
        rev["revenue_kusd"] = rev["volume_kbbl"] * rev["price_usd_bbl"]
        st.markdown(f"**{view_name} product make – plan days**")
        st.dataframe(rev.round(1), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Margin per extra barrel of each decision ($/bbl)**")
        st.caption("Because the model is linear, each decision has a fixed margin per barrel. "
                   "The optimizer pushes the highest ones until a constraint stops it.")
        mpu = margin_per_unit(data)
        st.dataframe(mpu.style.format({c: "{:,.2f}" for c in mpu.columns[1:]})
                     .background_gradient(subset=["margin_usd_bbl"], cmap="RdYlGn"),
                     hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 7. explain
with tabs[6]:
    if not opt:
        st.info("Run the optimizer first.")
    else:
        st.subheader("Why the optimizer chose this plan")
        for line in explain(data, base["kpi"], opt["kpi"], res):
            st.markdown(f"- {line}")
        st.caption("Shadow price = margin gained ($k) if that limit is relaxed by one unit. "
                   "The LP can be degenerate, so shadow prices are indicative, not unique.")
        if not res.shortfalls.empty:
            st.markdown("**Tank shortfalls (soft mode)**")
            st.dataframe(res.shortfalls.round(2), hide_index=True)
        with st.expander("All binding constraints"):
            st.dataframe(res.binding[["constraint", "label", "day", "item", "side", "shadow_price"]]
                         .round(3).sort_values(["constraint", "day"]), hide_index=True, use_container_width=True)
        st.markdown("**Constraint check – optimised plan (True = OK)**")
        st.dataframe(opt["checks"], hide_index=True, use_container_width=True, height=360)
