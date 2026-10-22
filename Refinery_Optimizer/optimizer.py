"""LP optimizer: maximise plan-days gross margin subject to C1-C16 (SciPy HiGHS, no licence needed).

Decision vector per plan day t: [x_c1..x_cn, r, f] (kbbl/d).
Soft mode adds penalised slack on tank min-heel / end-of-month targets so infeasible
scenarios still return a plan plus the size of each shortfall.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from data_loader import PlanData
from model import coeffs, tank_vector

CONSTRAINT_LABEL = {
    "C1": "CDU capacity", "C2": "Crude rate limit", "C3": "CDU ramp rate", "C4": "Reformer capacity",
    "C5": "Naphtha availability", "C6": "RFCC capacity", "C7": "Residue availability", "C8": "DHT capacity",
    "C9": "Gasoline RON spec", "C10": "Residue sulfur spec", "C13": "Tank operating window",
    "C14": "End-of-month inventory",
}


@dataclass
class OptResult:
    success: bool
    message: str
    V: np.ndarray
    plan_margin_kusd: float = float("nan")
    penalty_kusd: float = 0.0
    binding: pd.DataFrame = field(default_factory=pd.DataFrame)
    shortfalls: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_rows: int = 0
    n_vars: int = 0


class _LP:
    """Collects A_ub x <= b_ub rows, each with a readable label."""

    def __init__(self, n: int, days: list[int]):
        self.n, self.days = n, days
        self.n_dec = n * len(days)
        self.n_slack = 0
        self.rows, self.rhs, self.meta = [], [], []

    def col(self, day: int) -> int:
        return (day - self.days[0]) * self.n

    def new_slack(self) -> int:
        self.n_slack += 1
        return self.n_dec + self.n_slack - 1

    def le(self, terms, rhs, cid, day, item, side, slack=None):
        self.rows.append((terms, slack))
        self.rhs.append(rhs)
        self.meta.append({"constraint": cid, "day": day, "item": item, "side": side})

    def matrix(self):
        A = np.zeros((len(self.rows), self.n_dec + self.n_slack))
        for i, (terms, slack) in enumerate(self.rows):
            for day, vec in terms:
                c = self.col(day)
                A[i, c:c + self.n] += vec
            if slack is not None:
                A[i, slack] -= 1.0
        return A, np.array(self.rhs, float)


def optimize(d: PlanData, soft_tanks: bool = False, penalty_usd_bbl: float = 1000.0) -> OptResult:
    M, P = coeffs(d), d.P
    V_hist = d.V.copy()
    fp, T = d.first_plan_day, d.T
    days = list(range(fp, T + 1))
    if not days:
        return OptResult(False, "No Plan days left in the horizon - nothing to optimise.", V_hist)
    n = len(d.var_names)
    lp = _LP(n, days)

    for t in days:
        lp.le([(t, M["CDU"])], P["CDU_MAX"], "C1", t, "CDU", "max")
        lp.le([(t, -M["CDU"])], -P["CDU_MIN"], "C1", t, "CDU", "min")
        lp.le([(t, -M["NAPH_LEFT"])], 0.0, "C5", t, "SR naphtha", "max")
        lp.le([(t, -M["AR_LEFT"])], 0.0, "C7", t, "Atm residue", "max")
        lp.le([(t, M["DHT"])], P["DHT_MAX"], "C8", t, "DHT", "max")
        lp.le([(t, -M["DHT"])], -P["DHT_MIN"], "C8", t, "DHT", "min")
        lp.le([(t, -M["RON_EXCESS"])], 0.0, "C9", t, "Gasoline 95", "min")
        lp.le([(t, M["S_EXCESS"])], 0.0, "C10", t, "Residue pool", "max")
        if t == fp and fp > 1:                                   # C3 linked to last actual day
            prev = V_hist[fp - 2] @ M["CDU"]
            lp.le([(t, M["CDU"])], P["CDU_RAMP"] + prev, "C3", t, "CDU", "up")
            lp.le([(t, -M["CDU"])], P["CDU_RAMP"] - prev, "C3", t, "CDU", "down")
        elif t > fp:
            lp.le([(t, M["CDU"]), (t - 1, -M["CDU"])], P["CDU_RAMP"], "C3", t, "CDU", "up")
            lp.le([(t, -M["CDU"]), (t - 1, M["CDU"])], P["CDU_RAMP"], "C3", t, "CDU", "down")

    # C11/C12 balances substituted into C13/C14:  I[k,t] = base[t] + sum_{s=fp..t} vec . x_s
    for tid, row in d.tank.iterrows():
        vec, fixed = tank_vector(d, M, tid)
        hist = np.r_[V_hist[:fp - 1] @ vec, np.zeros(T - fp + 1)]
        base = row["opening_kbbl"] + np.cumsum(fixed + hist)
        for t in days:
            lp.le([(s, vec) for s in range(fp, t + 1)], row["capacity_kbbl"] - base[t - 1], "C13", t, tid, "max")
            lp.le([(s, -vec) for s in range(fp, t + 1)], base[t - 1] - row["min_heel_kbbl"], "C13", t, tid, "min",
                  slack=lp.new_slack() if soft_tanks else None)
        lp.le([(s, -vec) for s in days], base[T - 1] - row["end_target_kbbl"], "C14", T, tid, "min",
              slack=lp.new_slack() if soft_tanks else None)

    day_bounds = ([(d.crude.loc[c, "x_min_kbd"], d.crude.loc[c, "x_max_kbd"]) for c in d.crudes]
                  + [(P["REF_MIN"], P["REF_MAX"]), (P["RFCC_MIN"], P["RFCC_MAX"])])
    bounds = day_bounds * len(days) + [(0, None)] * lp.n_slack
    c = np.r_[np.tile(-M["MARGIN"], len(days)), np.full(lp.n_slack, penalty_usd_bbl)]
    A, b = lp.matrix()
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.status != 0:
        hint = "" if soft_tanks else " Turn on 'soft tank limits' to see which tank target cannot be met."
        return OptResult(False, f"{res.message}{hint}", V_hist, n_rows=A.shape[0], n_vars=A.shape[1])

    x = res.x
    V = V_hist.copy()
    V[fp - 1:] = x[:lp.n_dec].reshape(len(days), n)
    plan_margin = float((V[fp - 1:] @ M["MARGIN"]).sum())
    penalty = float(penalty_usd_bbl * x[lp.n_dec:].sum()) if lp.n_slack else 0.0

    meta = pd.DataFrame(lp.meta)
    meta["slack"] = res.ineqlin.residual
    meta["shadow_price"] = -res.ineqlin.marginals
    binding = meta[meta["slack"] < 1e-6].copy()
    names = [f"Crude {c}" for c in d.crudes] + ["Reformer", "RFCC"]
    cids = ["C2"] * len(d.crudes) + ["C4", "C6"]
    extra = []
    for k, t in enumerate(days):
        for j in range(n):
            i = k * n + j
            lo, hi = day_bounds[j]
            for side, hit, mu in [("max", x[i] >= hi - 1e-6, -res.upper.marginals[i]),
                                  ("min", x[i] <= lo + 1e-6, res.lower.marginals[i])]:
                if hit and not (side == "min" and lo == 0 and j < len(d.crudes)):
                    extra.append({"constraint": cids[j], "day": t, "item": names[j], "side": side,
                                  "slack": 0.0, "shadow_price": mu})
    binding = pd.concat([binding, pd.DataFrame(extra)], ignore_index=True)
    binding["label"] = binding["constraint"].map(CONSTRAINT_LABEL)

    sf = []
    if lp.n_slack:
        for i, (_, slack) in enumerate(lp.rows):
            if slack is not None and x[slack] > 1e-6:
                m = lp.meta[i]
                sf.append({"constraint": m["constraint"], "day": m["day"], "tank_id": m["item"],
                           "shortfall_kbbl": float(x[slack])})
    return OptResult(True, "Optimal", V, plan_margin, penalty, binding, pd.DataFrame(sf),
                     n_rows=A.shape[0], n_vars=A.shape[1])


def binding_summary(d: PlanData, res: OptResult) -> pd.DataFrame:
    if res.binding.empty:
        return pd.DataFrame(columns=["constraint", "label", "item", "side", "days", "value"])
    return (res.binding.groupby(["constraint", "label", "item", "side"])
            .agg(days=("day", "nunique"), value=("shadow_price", "sum")).reset_index()
            .sort_values("value", key=np.abs, ascending=False))


def explain(d: PlanData, base_kpi: dict, opt_kpi: dict, res: OptResult) -> list[str]:
    """Plain-language reasons built from binding constraints and shadow prices."""
    out = [f"Plan-days margin moves from {base_kpi['plan_margin_kusd']:,.0f} $k (current plan) to "
           f"{opt_kpi['plan_margin_kusd']:,.0f} $k (optimised): "
           f"{opt_kpi['plan_margin_kusd'] - base_kpi['plan_margin_kusd']:+,.0f} $k."]
    n_days = d.T - d.first_plan_day + 1
    for _, r in binding_summary(d, res).head(8).iterrows():
        txt = (f"**{r['label']} ({r['constraint']}) – {r['item']} {r['side']}** is limiting on "
               f"{r['days']} of {n_days} plan days")
        if abs(r["value"]) >= 0.05:
            txt += f"; relaxing it by one unit on those days would add about {abs(r['value']):,.1f} $k."
        else:
            txt += " (at the limit, but relaxing it adds no margin)."
        out.append(txt)
    for _, r in res.shortfalls.iterrows():
        out.append(f"Tank **{r['tank_id']}** misses its {CONSTRAINT_LABEL[r['constraint']].lower()} on day "
                   f"{r['day']} by {r['shortfall_kbbl']:,.1f} kbbl – this scenario is not fully feasible.")
    return out
