"""Load and validate the refinery planning CSVs (from GitHub or the local data/ folder).

Pipeline layer 1-2: CSV files -> validated PlanData object.
Nothing in here knows about the optimizer or the UI.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

FILES = {
    "crude_data": "crude_data.csv",
    "unit_data": "unit_data.csv",
    "product_data": "product_data.csv",
    "tank_master": "tank_master.csv",
    "crude_receipts": "crude_receipts.csv",
    "product_liftings": "product_liftings.csv",
    "daily_decisions": "daily_decisions.csv",
    "config": "config.csv",
}

REQUIRED_COLS = {
    "crude_data": ["crude_id", "crude_name", "tank_id", "price_usd_bbl", "y_lpg", "y_naph", "y_kero",
                   "y_go", "y_ar", "y_loss", "ar_sulfur_wt", "x_min_kbd", "x_max_kbd"],
    "unit_data": ["param_key", "value"],
    "product_data": ["product_id", "product_name", "tank_id", "price_usd_bbl"],
    "tank_master": ["tank_id", "service", "material_id", "type", "capacity_kbbl", "min_heel_kbbl",
                    "opening_kbbl", "end_target_kbbl"],
    "crude_receipts": ["day", "crude_id", "qty_kbbl"],
    "product_liftings": ["day", "product_id", "qty_kbbl"],
    "daily_decisions": ["day", "date", "status"],
    "config": ["key", "value"],
}

REQUIRED_PARAMS = [
    "CDU_MAX", "CDU_MIN", "CDU_RAMP", "CDU_OPEX",
    "REF_MAX", "REF_MIN", "REF_OPEX", "REF_Y_REFORMATE", "REF_Y_LPG",
    "RFCC_MAX", "RFCC_MIN", "RFCC_OPEX", "RFCC_Y_GASOLINE", "RFCC_Y_LPG", "RFCC_Y_LCO", "RFCC_Y_SLURRY",
    "DHT_MAX", "DHT_MIN", "DHT_OPEX", "DHT_Y_DIESEL",
    "RON_REFORMATE", "RON_FCC_GASOLINE", "RON_SPEC_MIN", "AR_SULFUR_MAX",
]

KNOWN_PRODUCTS = ["LPG", "NAP", "GAS95", "JET", "DSL", "FO"]   # products the process model can make
YIELD_COLS = ["y_lpg", "y_naph", "y_kero", "y_go", "y_ar", "y_loss"]


# ---------------------------------------------------------------- reading
def read_csvs(folder: str | Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    folder = Path(folder)
    return {name: pd.read_csv(folder / fname) for name, fname in FILES.items()}


def to_raw_url(url: str) -> str:
    """Accept a github.com/.../tree/<branch>/<path> link or a raw URL; return the raw base URL."""
    url = url.strip().rstrip("/")
    for marker in ("/tree/", "/blob/"):
        if "github.com/" in url and marker in url and "raw.githubusercontent.com" not in url:
            owner_repo, rest = url.split("github.com/", 1)[1].split(marker, 1)
            return f"https://raw.githubusercontent.com/{owner_repo}/{rest}"
    return url


def read_csvs_from_url(base_url: str, fallback: str | Path = DATA_DIR, timeout: float = 10.0):
    """Read every CSV from a GitHub folder (tries <base>/<file>, then <base>/data/<file>).

    Any file that cannot be fetched falls back to the local copy.
    Returns (frames, sources) where sources[name] = URL used, or 'local'.
    """
    base = to_raw_url(base_url)
    frames, sources = {}, {}
    for name, fname in FILES.items():
        for url in (f"{base}/{fname}", f"{base}/data/{fname}"):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    frames[name] = pd.read_csv(io.StringIO(r.read().decode("utf-8-sig")))
                sources[name] = url
                break
            except Exception:
                continue
        if name not in frames:
            frames[name] = pd.read_csv(Path(fallback) / fname)
            sources[name] = "local"
    return frames, sources


# ---------------------------------------------------------------- data object
@dataclass
class PlanData:
    crude: pd.DataFrame            # index = crude_id
    P: dict                        # unit parameters, P['CDU_MAX'] ...
    product: pd.DataFrame          # index = product_id
    tank: pd.DataFrame             # index = tank_id
    R: pd.DataFrame                # receipts  (day 1..T x crude_id)
    L: pd.DataFrame                # liftings  (day 1..T x product_id)
    decisions: pd.DataFrame        # daily_decisions
    cfg: dict
    first_plan_day: int
    crudes: list = field(default_factory=list)
    products: list = field(default_factory=list)

    @property
    def T(self) -> int:
        return len(self.decisions)

    @property
    def var_names(self) -> list[str]:
        return [f"x_{c}" for c in self.crudes] + ["r_reformer", "f_rfcc"]

    @property
    def V(self) -> np.ndarray:
        return self.decisions[self.var_names].to_numpy(float)


def _num(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------------------------------------------------- validation
def validate(frames: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    """Return [(level, message)]; level 'error' blocks the run, 'warning' does not."""
    issues: list[tuple[str, str]] = []
    err = lambda m: issues.append(("error", m))
    warn = lambda m: issues.append(("warning", m))

    for name, cols in REQUIRED_COLS.items():
        if name not in frames:
            err(f"{FILES[name]} is missing.")
            continue
        missing = [c for c in cols if c not in frames[name].columns]
        if missing:
            err(f"{FILES[name]}: missing columns {missing}.")
    if any(lvl == "error" for lvl, _ in issues):
        return issues

    crude = _num(frames["crude_data"], REQUIRED_COLS["crude_data"][3:])
    tank = _num(frames["tank_master"], REQUIRED_COLS["tank_master"][4:])
    unit, prod, dec = frames["unit_data"], frames["product_data"], frames["daily_decisions"]
    crudes = crude["crude_id"].astype(str).tolist()
    products = prod["product_id"].astype(str).tolist()

    if crude[REQUIRED_COLS["crude_data"][3:]].isna().any().any():
        err("Crude data: blank or non-numeric values.")
    for _, r in crude.iterrows():
        s = r[YIELD_COLS].sum()
        if abs(s - 1) > 1e-6:
            err(f"Crude data: yields for {r.crude_id} sum to {s:.4f}, must be 1.000.")
        if r.x_min_kbd > r.x_max_kbd:
            err(f"Crude data: {r.crude_id} x_min_kbd > x_max_kbd.")
        if r.price_usd_bbl < 0:
            err(f"Crude data: {r.crude_id} negative price.")
    if crude["crude_id"].duplicated().any():
        err("Crude data: duplicate crude_id.")

    keys = set(unit["param_key"].astype(str))
    miss = [k for k in REQUIRED_PARAMS if k not in keys]
    if miss:
        err(f"Unit data: missing param_key {miss}.")
    vals = pd.to_numeric(unit["value"], errors="coerce")
    if vals.isna().any():
        err("Unit data: blank or non-numeric value.")
    P = dict(zip(unit["param_key"], vals))
    for u in ["CDU", "REF", "RFCC", "DHT"]:
        if P.get(f"{u}_MIN", 0) > P.get(f"{u}_MAX", np.inf):
            err(f"Unit data: {u}_MIN > {u}_MAX.")
    if P.get("REF_Y_REFORMATE", 0) + P.get("REF_Y_LPG", 0) > 1 + 1e-9:
        err("Unit data: Reformer yields exceed 1.0.")
    if sum(P.get(k, 0) for k in ["RFCC_Y_GASOLINE", "RFCC_Y_LPG", "RFCC_Y_LCO", "RFCC_Y_SLURRY"]) > 1 + 1e-9:
        err("Unit data: RFCC yields exceed 1.0.")

    unknown = [p for p in products if p not in KNOWN_PRODUCTS]
    if unknown:
        err(f"Product data: product_id {unknown} not in the process model {KNOWN_PRODUCTS}.")
    if pd.to_numeric(prod["price_usd_bbl"], errors="coerce").isna().any():
        err("Product data: blank or non-numeric price.")

    if tank[REQUIRED_COLS["tank_master"][4:]].isna().any().any():
        err("Tank master: blank or non-numeric values.")
    else:
        for _, r in tank.iterrows():
            if r.min_heel_kbbl > r.capacity_kbbl:
                err(f"Tank master: {r.tank_id} min heel > capacity.")
            if not (r.min_heel_kbbl <= r.opening_kbbl <= r.capacity_kbbl):
                warn(f"Tank master: {r.tank_id} opening {r.opening_kbbl:g} is outside [min heel, capacity].")
            if r.end_target_kbbl > r.capacity_kbbl:
                err(f"Tank master: {r.tank_id} end target > capacity.")
    for m in crudes + products:
        if (tank["material_id"] == m).sum() != 1:
            err(f"Tank master: material {m} must map to exactly one tank.")

    T = len(dec)
    if list(pd.to_numeric(dec["day"], errors="coerce")) != list(range(1, T + 1)):
        err("Actuals: 'day' must run 1..T with no gaps.")
    need = [f"x_{c}" for c in crudes] + ["r_reformer", "f_rfcc"]
    miss = [c for c in need if c not in dec.columns]
    if miss:
        err(f"Actuals: missing decision columns {miss}.")
    elif _num(dec, need)[need].isna().any().any():
        err("Actuals: blank decision values (use 0 for Plan rows if unknown).")

    for name, label, idcol, valid in [("crude_receipts", "Crude receipts", "crude_id", crudes),
                                      ("product_liftings", "Product liftings", "product_id", products)]:
        df = frames[name]
        bad = sorted(set(df[idcol].astype(str)) - set(valid))
        if bad:
            err(f"{label}: unknown {idcol} {bad}.")
        d = pd.to_numeric(df["day"], errors="coerce")
        if d.isna().any() or ((d < 1) | (d > T)).any() or (d % 1 != 0).any():
            err(f"{label}: every day must be a whole number between 1 and {T}.")
        q = pd.to_numeric(df["qty_kbbl"], errors="coerce")
        if q.isna().any() or (q < 0).any():
            err(f"{label}: qty_kbbl must be a number ≥ 0.")
    return issues


# ---------------------------------------------------------------- build
def build(frames: dict[str, pd.DataFrame], first_plan_day: int | None = None) -> PlanData:
    """Turn validated frames into a PlanData object. Call validate() first."""
    crude = _num(frames["crude_data"], REQUIRED_COLS["crude_data"][3:]).set_index("crude_id")
    unit = frames["unit_data"]
    P = dict(zip(unit["param_key"], pd.to_numeric(unit["value"])))
    product = frames["product_data"].copy()
    product["price_usd_bbl"] = pd.to_numeric(product["price_usd_bbl"])
    product = product.set_index("product_id")
    tank = _num(frames["tank_master"], REQUIRED_COLS["tank_master"][4:]).set_index("tank_id")
    dec = frames["daily_decisions"].copy().reset_index(drop=True)
    crudes, products = list(crude.index), list(product.index)
    var = [f"x_{c}" for c in crudes] + ["r_reformer", "f_rfcc"]
    dec[var] = dec[var].apply(pd.to_numeric)
    dec["day"] = pd.to_numeric(dec["day"]).astype(int)
    T = len(dec)
    days = range(1, T + 1)

    def pivot(df, idcol, cols):
        df = df.assign(day=pd.to_numeric(df["day"]).astype(int), qty_kbbl=pd.to_numeric(df["qty_kbbl"]))
        if df.empty:
            return pd.DataFrame(0.0, index=days, columns=cols)
        return (df.pivot_table(index="day", columns=idcol, values="qty_kbbl", aggfunc="sum")
                  .reindex(index=days, columns=cols).fillna(0.0))

    R = pivot(frames["crude_receipts"], "crude_id", crudes)
    L = pivot(frames["product_liftings"], "product_id", products)
    cfg = dict(zip(frames["config"]["key"], frames["config"]["value"]))

    if first_plan_day is None:
        plan_rows = dec.index[dec["status"] == "Plan"]
        first_plan_day = int(dec.loc[plan_rows[0], "day"]) if len(plan_rows) else T + 1
    first_plan_day = int(min(max(first_plan_day, 1), T + 1))
    dec["status"] = np.where(dec["day"] < first_plan_day, "Actual", "Plan")
    return PlanData(crude=crude, P=P, product=product, tank=tank, R=R, L=L, decisions=dec,
                    cfg=cfg, first_plan_day=first_plan_day, crudes=crudes, products=products)


def load(folder: str | Path = DATA_DIR, first_plan_day: int | None = None) -> PlanData:
    """Read + validate + build from a local folder. Raises ValueError on input errors."""
    frames = read_csvs(folder)
    errors = [m for lvl, m in validate(frames) if lvl == "error"]
    if errors:
        raise ValueError("Input data errors:\n- " + "\n- ".join(errors))
    return build(frames, first_plan_day)
