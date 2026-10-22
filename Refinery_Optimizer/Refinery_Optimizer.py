"""Process model: decisions (x, r, f) -> flows, products, economics, tank balances and checks.

Python twin of the Daily_Plan, Tank_Balance and Constraint_Check sheets.
Every quantity is LINEAR in the decision vector v = [x_c1..x_cn, r, f], so each one is stored
as a coefficient vector: quantity_t = V[t] @ M['NAME']. The optimizer re-uses the same vectors,
so plan evaluation and optimization can never disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import PlanData

TOL = 1e-6


def coeffs(d: PlanData) -> dict[str, np.ndarray]:
    P, y = d.P, d.crude
    nc = len(d.crudes)
    z = np.zeros(nc)

    def v(xc, r=0.0, f=0.0):
        return np.r_[xc, r, f].astype(float)

    M = {
        "SR_LPG": v(y["y_lpg"].to_numpy(float)),
        "SR_NAPH": v(y["y_naph"].to_numpy(float)),
        "SR_KERO": v(y["y_kero"].to_numpy(float)),
        "SR_GO": v(y["y_go"].to_numpy(float)),
        "AR": v(y["y_ar"].to_numpy(float)),
        "CDU": v(np.ones(nc)),
        "REF_FEED": v(z, 1.0),
        "RFCC_FEED": v(z, 0.0, 1.0),
        "REFORMATE": v(z, P["REF_Y_REFORMATE"]),
        "REF_LPG": v(z, P["REF_Y_LPG"]),
        "FCC_GAS": v(z, 0.0, P["RFCC_Y_GASOLINE"]),
        "FCC_LPG": v(z, 0.0, P["RFCC_Y_LPG"]),
        "LCO": v(z, 0.0, P["RFCC_Y_LCO"]),
        "SLURRY": v(z, 0.0, P["RFCC_Y_SLURRY"]),
        "AR_S_MASS": v(y["y_ar"].to_numpy(float) * y["ar_sulfur_wt"].to_numpy(float)),
        "CRUDE_COST": v(y["price_usd_bbl"].to_numpy(float)),
    }
    M["DHT"] = M["SR_GO"] + M["LCO"]
    make = {
        "LPG": M["SR_LPG"] + M["REF_LPG"] + M["FCC_LPG"],
        "NAP": M["SR_NAPH"] - M["REF_FEED"],
        "GAS95": M["REFORMATE"] + M["FCC_GAS"],
        "JET": M["SR_KERO"],
        "DSL": P["DHT_Y_DIESEL"] * M["DHT"],
        "FO": M["AR"] - M["RFCC_FEED"] + M["SLURRY"],
    }
    for p in d.products:
        M[f"MAKE_{p}"] = make[p]
    M["NAPH_LEFT"] = M["SR_NAPH"] - M["REF_FEED"]                                    # C5 >= 0
    M["AR_LEFT"] = M["AR"] - M["RFCC_FEED"]                                          # C7 >= 0
    M["RON_EXCESS"] = ((P["RON_REFORMATE"] - P["RON_SPEC_MIN"]) * M["REFORMATE"]
                       + (P["RON_FCC_GASOLINE"] - P["RON_SPEC_MIN"]) * M["FCC_GAS"])  # C9 >= 0
    M["S_EXCESS"] = M["AR_S_MASS"] - P["AR_SULFUR_MAX"] * M["AR"]                    # C10 <= 0
    M["REVENUE"] = sum(d.product.loc[p, "price_usd_bbl"] * M[f"MAKE_{p}"] for p in d.products)
    M["OPEX"] = (P["CDU_OPEX"] * M["CDU"] + P["REF_OPEX"] * M["REF_FEED"]
                 + P["RFCC_OPEX"] * M["RFCC_FEED"] + P["DHT_OPEX"] * M["DHT"])
    M["MARGIN"] = M["REVENUE"] - M["CRUDE_COST"] - M["OPEX"]                          # $k per kbbl = $/bbl
    return M


def tank_vector(d: PlanData, M: dict, tank_id: str):
    """(coef vector of decision-driven net inflow, fixed net inflow per day) for one tank."""
    row = d.tank.loc[tank_id]
    mid = row["material_id"]
    if row["type"] == "Crude":
        return -np.eye(len(d.var_names))[d.crudes.index(mid)], d.R[mid].to_numpy(float)
    return M[f"MAKE_{mid}"], -d.L[mid].to_numpy(float)


def margin_per_unit(d: PlanData) -> pd.DataFrame:
    M = coeffs(d)
    names = [f"Crude {c} ({d.crude.loc[c, 'crude_name']})" for c in d.crudes] + ["Naphtha → Reformer", "Residue → RFCC"]
    return pd.DataFrame({"decision": names, "var": d.var_names, "revenue_usd_bbl": M["REVENUE"],
                         "crude_cost_usd_bbl": M["CRUDE_COST"], "opex_usd_bbl": M["OPEX"],
                         "margin_usd_bbl": M["MARGIN"]}).round(3)


def evaluate(d: PlanData, V: np.ndarray | None = None) -> dict:
    """Recalculate the full plan for decision matrix V (T x n). Default = decisions in the CSV."""
    V = d.V if V is None else np.asarray(V, float)
    M = coeffs(d)
    P = d.P
    q = {k: V @ vec for k, vec in M.items()}

    dp = d.decisions[["day", "date"]].copy()
    dp["status"] = np.where(dp["day"] < d.first_plan_day, "Actual", "Plan")
    for i, name in enumerate(d.var_names):
        dp[name] = V[:, i]
    for k, col in [("CDU", "cdu_total"), ("SR_LPG", "sr_lpg"), ("SR_NAPH", "sr_naphtha"), ("SR_KERO", "sr_kero"),
                   ("SR_GO", "sr_gasoil"), ("AR", "atm_residue"), ("REFORMATE", "reformate"), ("REF_LPG", "ref_lpg"),
                   ("FCC_GAS", "fcc_gasoline"), ("FCC_LPG", "fcc_lpg"), ("LCO", "lco"), ("SLURRY", "slurry"),
                   ("DHT", "dht_feed")]:
        dp[col] = q[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        dp["residue_s_wt"] = np.where(q["AR"] > 0, q["AR_S_MASS"] / q["AR"], 0.0)
    for p in d.products:
        dp[f"make_{p}"] = q[f"MAKE_{p}"]
    gas = q["REFORMATE"] + q["FCC_GAS"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dp["gasoline_ron"] = np.where(gas > 0, (q["REFORMATE"] * P["RON_REFORMATE"]
                                                + q["FCC_GAS"] * P["RON_FCC_GASOLINE"]) / gas, 0.0)
        dp["margin_usd_bbl"] = np.where(q["CDU"] > 0, q["MARGIN"] / q["CDU"], 0.0)
    dp["revenue_kusd"] = q["REVENUE"]
    dp["crude_cost_kusd"] = q["CRUDE_COST"]
    dp["opex_kusd"] = q["OPEX"]
    dp["margin_kusd"] = q["MARGIN"]

    tb_rows = []
    for tid, row in d.tank.iterrows():
        vec, fixed = tank_vector(d, M, tid)
        closing = row["opening_kbbl"] + np.cumsum(V @ vec + fixed)
        opening = np.r_[row["opening_kbbl"], closing[:-1]]
        inflow, outflow = (fixed, -(V @ vec)) if row["type"] == "Crude" else (V @ vec, -fixed)
        for t in range(d.T):
            c = closing[t]
            status = "OK" if row["min_heel_kbbl"] - TOL <= c <= row["capacity_kbbl"] + TOL else (
                "LOW" if c < row["min_heel_kbbl"] else "HIGH")
            tb_rows.append({"tank_id": tid, "service": row["service"], "type": row["type"], "day": t + 1,
                            "date": d.decisions["date"].iat[t], "opening": opening[t], "inflow": inflow[t],
                            "outflow": outflow[t], "closing": c, "min_heel": row["min_heel_kbbl"],
                            "capacity": row["capacity_kbbl"], "fill_pct": c / row["capacity_kbbl"], "status": status})
    tb = pd.DataFrame(tb_rows)

    xmax, xmin = d.crude["x_max_kbd"].to_numpy(float), d.crude["x_min_kbd"].to_numpy(float)
    X = V[:, :len(d.crudes)]
    cdu = q["CDU"]
    ramp = np.r_[0.0, np.diff(cdu)]
    tank_breach = tb.groupby("day")["status"].apply(lambda s: (s != "OK").sum()).to_numpy()
    ck = dp[["day", "date", "status"]].copy()
    ck["C1"] = (cdu >= P["CDU_MIN"] - TOL) & (cdu <= P["CDU_MAX"] + TOL)
    ck["C2"] = ((X >= xmin - TOL) & (X <= xmax + TOL)).all(axis=1)
    ck["C3"] = np.abs(ramp) <= P["CDU_RAMP"] + TOL
    ck["C4"] = (q["REF_FEED"] >= P["REF_MIN"] - TOL) & (q["REF_FEED"] <= P["REF_MAX"] + TOL)
    ck["C5"] = q["NAPH_LEFT"] >= -TOL
    ck["C6"] = (q["RFCC_FEED"] >= P["RFCC_MIN"] - TOL) & (q["RFCC_FEED"] <= P["RFCC_MAX"] + TOL)
    ck["C7"] = q["AR_LEFT"] >= -TOL
    ck["C8"] = (q["DHT"] >= P["DHT_MIN"] - TOL) & (q["DHT"] <= P["DHT_MAX"] + TOL)
    ck["C9"] = q["RON_EXCESS"] >= -TOL
    ck["C10"] = q["S_EXCESS"] <= TOL
    ck["C13"] = tank_breach == 0
    flags = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C13"]
    ck["violations"] = (~ck[flags]).sum(axis=1)
    ck["day_result"] = np.where(ck["violations"] == 0, "FEASIBLE", "VIOLATION")

    last = tb[tb["day"] == d.T].set_index("tank_id")
    c14 = pd.DataFrame({"tank_id": d.tank.index, "service": d.tank["service"].values,
                        "closing_last_day": last.loc[d.tank.index, "closing"].values,
                        "end_target": d.tank["end_target_kbbl"].values})
    c14["C14_ok"] = c14["closing_last_day"] >= c14["end_target"] - TOL

    plan = dp["status"] == "Plan"
    kpi = {"month_margin_kusd": float(dp["margin_kusd"].sum()),
           "actual_margin_kusd": float(dp.loc[~plan, "margin_kusd"].sum()),
           "plan_margin_kusd": float(dp.loc[plan, "margin_kusd"].sum()),
           "crude_run_kbbl": float(cdu.sum()),
           "violations": int(ck["violations"].sum() + (~c14["C14_ok"]).sum())}
    return {"daily_plan": dp, "tank_balance": tb, "checks": ck, "c14": c14, "kpi": kpi}
