"""Command-line runner (no browser): CSV in -> optimise -> CSV out. Also used by the Excel VBA button.

    python run_optimizer.py                                   # local ./data
    python run_optimizer.py --url https://github.com/<user>/<repo>/tree/main/<folder>
    python run_optimizer.py --first-plan-day 13 --soft
"""
import argparse
from pathlib import Path

import data_loader
from model import evaluate
from optimizer import explain, optimize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(data_loader.DATA_DIR), help="local folder with the CSVs")
    ap.add_argument("--url", default=None, help="GitHub folder URL (overrides --data)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--first-plan-day", type=int, default=None)
    ap.add_argument("--soft", action="store_true", help="soft tank min-heel / end-target")
    ap.add_argument("--penalty", type=float, default=1000.0)
    a = ap.parse_args()

    frames = data_loader.read_csvs_from_url(a.url)[0] if a.url else data_loader.read_csvs(a.data)
    errors = [m for lvl, m in data_loader.validate(frames) if lvl == "error"]
    if errors:
        raise SystemExit("Input errors:\n- " + "\n- ".join(errors))
    d = data_loader.build(frames, a.first_plan_day)
    base = evaluate(d)
    res = optimize(d, soft_tanks=a.soft, penalty_usd_bbl=a.penalty)
    print(f"Horizon {d.T} days | locked 1-{d.first_plan_day - 1} | solver: {res.message}")
    if not res.success:
        raise SystemExit(1)
    opt = evaluate(d, res.V)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    opt["daily_plan"][["day", "date", "status"] + d.var_names].to_csv(out / "daily_decisions_optimized.csv", index=False)
    opt["daily_plan"].to_csv(out / "daily_plan.csv", index=False)
    opt["tank_balance"].to_csv(out / "tank_balance.csv", index=False)
    opt["checks"].to_csv(out / "constraint_check.csv", index=False)
    res.binding.to_csv(out / "binding_constraints.csv", index=False)
    print(f"Current plan margin : {base['kpi']['plan_margin_kusd']:,.2f} $k")
    print(f"Optimised margin    : {opt['kpi']['plan_margin_kusd']:,.2f} $k")
    print(f"Violations          : {opt['kpi']['violations']}")
    for line in explain(d, base["kpi"], opt["kpi"], res):
        print(" -", line.replace("**", ""))
    print(f"Results -> {out.resolve()}")


if __name__ == "__main__":
    main()
