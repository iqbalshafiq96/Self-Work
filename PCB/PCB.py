#!/usr/bin/env python3
"""
PCB / MTD (Potongan Cukai Bulanan) Estimator - Year of Assessment 2026
=======================================================================
Implements LHDN "Specification for Monthly Tax Deduction (MTD) Calculations
Using Computerized Calculation for 2026" (updated 01 Jan 2026).

Covers:
  a. Non-resident                  -> 30% flat of remuneration
  b. Resident
     1) Normal remuneration        -> [(P - M)R + B - (Z + X)] / (n + 1)
     2) Additional remuneration    -> 5-step method (bonus, arrears, etc.)
     3) Returning Expert (REP)     -> [(PR - T) - (Z + X)] / (n + 1), R = 15%
     4) Knowledge worker           -> same as REP
     5) Non-citizen C-Suite        -> [PR - (Z + X)] / (n + 1), R = 15%

Rules applied (Section E, Terms & Conditions):
  - Truncate to 2 decimals, then round UP to nearest 5 sen
  - MTD for current month < RM10        -> 0
  - MTD for additional remuneration < 10 -> 0
  - Net MTD after zakat < RM10           -> still deducted
  - EPF (K + K1 + K2*n + Kt) capped at RM4,000/yr
  - K2 = min( [4000 - (K + K1 + Kt)] / n , K1 )

Run:  python pcb_calculator_2026.py
"""

from dataclasses import dataclass, field
from decimal import Decimal as D, ROUND_DOWN, ROUND_UP, ROUND_CEILING

# --------------------------------------------------------------------------
# 1. CONSTANTS (from the 2026 specification)
# --------------------------------------------------------------------------
EPF_CAP = D("4000")

RELIEF_INDIVIDUAL = D("9000")      # D
RELIEF_SPOUSE = D("4000")          # S  (Category 2 only)
RELIEF_DISABLED_SELF = D("7000")   # Du
RELIEF_DISABLED_SPOUSE = D("6000") # Su
RELIEF_PER_CHILD = D("2000")       # Q

# Table 1: (lower bound P, M, R%, B cat 1&3, B cat 2)
TABLE_1 = [
    (D("2000000"), D("2000000"), D("30"), D("528400"), D("528400")),
    (D("600000"),  D("600000"),  D("28"), D("136400"), D("136400")),
    (D("400000"),  D("400000"),  D("26"), D("84400"),  D("84400")),
    (D("100000"),  D("100000"),  D("25"), D("9400"),   D("9400")),
    (D("70000"),   D("70000"),   D("19"), D("3700"),   D("3700")),
    (D("50000"),   D("50000"),   D("11"), D("1500"),   D("1500")),
    (D("35000"),   D("35000"),   D("6"),  D("600"),    D("600")),
    (D("20000"),   D("20000"),   D("3"),  D("-250"),   D("-650")),
    (D("5000"),    D("5000"),    D("1"),  D("-400"),   D("-800")),
]

# Child "units" (Q x C): how many children the relief counts as
CHILD_TYPES = {
    "1": ("Child below 18 (or 18+ studying below diploma level)", 1),
    "2": ("Child 18+ studying diploma+ (Malaysia) / degree+ (overseas)", 4),
    "3": ("Disabled child", 4),
    "4": ("Disabled child 18+ studying diploma+ / degree+", 8),
}

# Optional TP1 deductions (LP) with 2026 annual caps
TP1_ITEMS = [
    ("Medical/special needs/carer for parents & grandparents", D("8000")),
    ("Basic supporting equipment (disabled)",                  D("6000")),
    ("Higher education fees (self)",                           D("7000")),
    ("Medical: serious disease/fertility/vaccine/dental/checkup/learning disability", D("10000")),
    ("Lifestyle (books, PC/phone/tablet, internet, courses)",  D("2500")),
    ("Additional lifestyle - sports",                          D("1000")),
    ("Breastfeeding equipment",                                D("1000")),
    ("Childcare / kindergarten / daycare fees",                D("3000")),
    ("SSPN net deposit",                                       D("8000")),
    ("Alimony to former wife",                                 D("4000")),
    ("Life insurance / takaful / voluntary EPF",               D("3000")),
    ("PRS / deferred annuity",                                 D("3000")),
    ("Education & medical insurance",                          D("4000")),
    ("SOCSO contribution",                                     D("350")),
    ("EV charger / food composter / food waste grinder / home CCTV", D("2500")),
    ("Housing loan interest (first home)",                     D("7000")),
    ("Admission fees - tourist attractions / cultural programmes", D("1000")),
]

SCHEMES = {
    "1": "NORMAL",        # standard resident
    "2": "REP",           # Returning Expert Programme
    "3": "KNOWLEDGE",     # Knowledge worker, specified region
    "4": "CSUITE",        # Non-citizen C-suite, approved company
}


# --------------------------------------------------------------------------
# 2. ROUNDING HELPERS (Section E items 1 & 2)
# --------------------------------------------------------------------------
def trunc2(x: D) -> D:
    """Keep 2 decimals, drop the rest (123.4534 -> 123.45)."""
    return D(x).quantize(D("0.01"), rounding=ROUND_DOWN)


def round_up_5sen(x: D) -> D:
    """Truncate to 2dp, then round UP to nearest 0.05 (287.02 -> 287.05)."""
    x = trunc2(x)
    return (x / D("0.05")).to_integral_value(rounding=ROUND_CEILING) * D("0.05")


# --------------------------------------------------------------------------
# 3. DATA STRUCTURES
# --------------------------------------------------------------------------
@dataclass
class Profile:
    resident: bool = True
    scheme: str = "NORMAL"          # NORMAL / REP / KNOWLEDGE / CSUITE
    category: int = 1               # 1 single, 2 married spouse not working, 3 others
    child_units: int = 0            # C (already weighted 1/4/8)
    disabled_self: bool = False
    disabled_spouse: bool = False


@dataclass
class MonthData:
    month: int                      # 1..12
    Y1: D = D("0")                  # gross normal remuneration this month (incl. BIK/VOLA)
    K1: D = D("0")                  # EPF on Y1
    Yt: D = D("0")                  # gross additional remuneration this month (bonus etc.)
    Kt: D = D("0")                  # EPF on Yt
    LP1: D = D("0")                 # TP1 deductions claimed this month
    zakat_now: D = D("0")           # zakat this month
    # Year-to-date before this month (incl. previous employer via TP3)
    Y: D = D("0")                   # accumulated gross remuneration
    K: D = D("0")                   # accumulated EPF
    X: D = D("0")                   # accumulated MTD paid
    Z: D = D("0")                   # accumulated zakat (excl. this month)
    LP: D = D("0")                  # accumulated TP1 deductions


# --------------------------------------------------------------------------
# 4. CORE FORMULA BLOCKS
# --------------------------------------------------------------------------
def fixed_reliefs(p: Profile) -> D:
    """D + S + Du + Su + QC"""
    total = RELIEF_INDIVIDUAL
    if p.category == 2:
        total += RELIEF_SPOUSE
    if p.disabled_self:
        total += RELIEF_DISABLED_SELF
    if p.disabled_spouse:
        total += RELIEF_DISABLED_SPOUSE
    if p.category in (2, 3):
        total += RELIEF_PER_CHILD * p.child_units
    return total


def cap_epf(K, K1, Kt):
    """Apply RM4,000 cap in order K -> K1 -> Kt."""
    K = min(K, EPF_CAP)
    K1 = min(K1, EPF_CAP - K)
    Kt = min(Kt, EPF_CAP - K - K1)
    return K, K1, Kt


def calc_K2(K, K1, Kt, n):
    """K2 = min([4000 - (K + K1 + Kt)] / n, K1)."""
    if n <= 0:
        return D("0")
    balance = max(D("0"), EPF_CAP - (K + K1 + Kt))
    return min(trunc2(balance / n), K1)


def chargeable_income(p: Profile, m: MonthData, include_additional: bool):
    """
    P = [Σ(Y-K) + (Y1-K1) + (Y2-K2)n + (Yt-Kt)] - [D+S+Du+Su+QC+(ΣLP+LP1)]
    Returns (P, K2, breakdown dict).
    """
    n = 12 - m.month
    Kt_raw = m.Kt if include_additional else D("0")
    K, K1, Kt = cap_epf(m.K, m.K1, Kt_raw)
    K2 = calc_K2(K, K1, Kt, n)
    Y2 = m.Y1

    income = (m.Y - K) + (m.Y1 - K1) + (Y2 - K2) * n
    if include_additional:
        income += (m.Yt - Kt)
    reliefs = fixed_reliefs(p) + m.LP + m.LP1
    P = trunc2(income - reliefs)
    return P, K2, {"n": n, "K_capped": K, "K1_capped": K1, "Kt_capped": Kt,
                   "K2": K2, "gross_net_income": trunc2(income), "total_reliefs": reliefs}


def annual_tax(p: Profile, P: D) -> D:
    """Annual tax on chargeable income P, before zakat and before MTD paid."""
    if P <= 0:
        return D("0")
    if p.scheme == "NORMAL":
        if P <= D("5000"):
            return D("0")
        for lower, M, R, B13, B2 in TABLE_1:
            if P > lower:
                B = B2 if p.category == 2 else B13
                return (P - M) * R / 100 + B
    if p.scheme in ("REP", "KNOWLEDGE"):
        T = D("0")
        if P <= D("35000"):
            T = D("800") if p.category == 2 else D("400")
        return P * D("0.15") - T
    if p.scheme == "CSUITE":
        return P * D("0.15")
    raise ValueError("Unknown scheme")


# --------------------------------------------------------------------------
# 5. MAIN MTD CALCULATION
# --------------------------------------------------------------------------
def calc_mtd(p: Profile, m: MonthData) -> dict:
    # ---- a. Non-resident: 30% flat ----
    if not p.resident:
        mtd = round_up_5sen((m.Y1 + m.Yt) * D("0.30"))
        return {"payable": mtd, "method": "Non-resident 30%", "steps": {}}

    n = 12 - m.month

    # ---- Step 1: MTD on normal remuneration (excl. current additional) ----
    P1, K2_1, b1 = chargeable_income(p, m, include_additional=False)
    tax1 = annual_tax(p, P1)
    mtd_c = round_up_5sen((tax1 - (m.Z + m.X)) / (n + 1))       # Step 1[C]
    mtd_c_deduct = mtd_c if mtd_c >= 10 else D("0")             # <RM10 rule
    net_mtd = mtd_c_deduct - m.zakat_now                         # Step 1[D]
    total_mtd_year = m.X + mtd_c * (n + 1)                       # Step 1[E]

    steps = {"Step1_P": P1, "Step1_K2": K2_1, "Step1_annual_tax": trunc2(tax1),
             "Step1C_MTD_current_month": mtd_c,
             "Step1D_net_MTD_after_zakat": net_mtd,
             "Step1E_total_MTD_for_year": total_mtd_year, **b1}

    if m.Yt <= 0:
        payable = round_up_5sen(max(D("0"), net_mtd))
        return {"payable": payable, "method": f"Resident {p.scheme} - normal remuneration",
                "steps": steps}

    # ---- Steps 2-5: additional remuneration (bonus etc.) ----
    P2, K2_2, b2 = chargeable_income(p, m, include_additional=True)   # Step 2
    tax2 = annual_tax(p, P2)                                          # Step 3
    add_mtd = round_up_5sen(tax2 - (total_mtd_year + m.Z))            # Step 4
    if add_mtd < 10:
        add_mtd = D("0")
    payable = round_up_5sen(max(D("0"), net_mtd + add_mtd))           # Step 5

    steps.update({"Step2_P_incl_additional": P2, "Step2_K2": K2_2,
                  "Step3_annual_tax_incl_additional": trunc2(tax2),
                  "Step4_MTD_on_additional": add_mtd,
                  "Step5_total_payable": payable})
    return {"payable": payable, "method": f"Resident {p.scheme} - with additional remuneration",
            "steps": steps}


# --------------------------------------------------------------------------
# 6. FULL-YEAR SIMULATION (auto-accumulates Y, K, X, Z, LP month by month)
# --------------------------------------------------------------------------
def simulate_year(p: Profile, salary, epf, start_month=1, end_month=12,
                  bonuses=None, bonus_epf=None, lp_by_month=None, zakat_monthly=D("0"),
                  ytd_seed=None):
    """bonuses / bonus_epf / lp_by_month: dict {month: amount}"""
    bonuses = bonuses or {}
    bonus_epf = bonus_epf or {}
    lp_by_month = lp_by_month or {}
    acc = ytd_seed or {"Y": D("0"), "K": D("0"), "X": D("0"), "Z": D("0"), "LP": D("0")}
    rows = []
    for mo in range(start_month, end_month + 1):
        m = MonthData(month=mo, Y1=salary, K1=epf,
                      Yt=bonuses.get(mo, D("0")), Kt=bonus_epf.get(mo, D("0")),
                      LP1=lp_by_month.get(mo, D("0")), zakat_now=zakat_monthly, **acc)
        r = calc_mtd(p, m)
        rows.append((mo, m, r))
        K, K1, Kt = cap_epf(m.K, m.K1, m.Kt)
        acc["Y"] += m.Y1 + m.Yt
        acc["K"] = K + K1 + Kt
        acc["X"] += r["payable"]
        acc["Z"] += m.zakat_now
        acc["LP"] += m.LP1
    return rows


# --------------------------------------------------------------------------
# 7. INTERACTIVE INPUT HELPERS
# --------------------------------------------------------------------------
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def ask_num(prompt, default="0"):
    while True:
        s = input(f"{prompt} [{default}]: ").strip() or default
        try:
            v = D(s.replace(",", ""))
            if v < 0:
                raise ValueError
            return v
        except Exception:
            print("  -> Please enter a valid non-negative number.")


def ask_int(prompt, lo, hi, default):
    while True:
        s = input(f"{prompt} [{default}]: ").strip() or str(default)
        if s.isdigit() and lo <= int(s) <= hi:
            return int(s)
        print(f"  -> Enter a number between {lo} and {hi}.")


def ask_yn(prompt, default="n"):
    s = (input(f"{prompt} (y/n) [{default}]: ").strip() or default).lower()
    return s.startswith("y")


def default_epf(salary: D) -> D:
    """Approximation of 11% employee EPF, rounded up to the next ringgit."""
    return (salary * D("0.11")).to_integral_value(rounding=ROUND_UP)


def ask_profile() -> Profile:
    print("\n=== EMPLOYEE PROFILE ===")
    p = Profile()
    p.resident = ask_yn("Are you a tax resident in Malaysia?", "y")
    if not p.resident:
        return p
    print("Tax scheme: 1) Normal  2) Returning Expert (REP)  3) Knowledge worker  4) Non-citizen C-Suite")
    p.scheme = SCHEMES[str(ask_int("Choose scheme", 1, 4, 1))]
    print("Category:\n  1) Single\n  2) Married, spouse NOT working\n"
          "  3) Married & spouse working / divorced / widowed / single with adopted child")
    p.category = ask_int("Choose category", 1, 3, 1)
    if p.category in (2, 3):
        units = 0
        for k, (label, w) in CHILD_TYPES.items():
            cnt = ask_int(f"  No. of children - {label}", 0, 30, 0)
            units += cnt * w
        p.child_units = units
    p.disabled_self = ask_yn("Are you a disabled person (OKU)?")
    if p.category == 2:
        p.disabled_spouse = ask_yn("Is your spouse disabled (OKU)?")
    return p


def ask_tp1(p: Profile) -> D:
    """Itemised TP1 claims, each capped at its 2026 annual limit."""
    if not ask_yn("Any TP1 optional reliefs to claim?"):
        return D("0")
    total = D("0")
    for name, cap in TP1_ITEMS:
        if name.startswith("Alimony") and p.category == 2:
            continue  # not allowed if wife relief claimed
        amt = ask_num(f"  {name} (max RM{cap:,.0f})", "0")
        if amt > cap:
            print(f"    capped to RM{cap:,.2f}")
            amt = cap
        total += amt
    print(f"  Total TP1 deductions: RM{total:,.2f}")
    return total


def print_breakdown(r: dict):
    print(f"\nMethod: {r['method']}")
    for k, v in r["steps"].items():
        if isinstance(v, D):
            print(f"  {k:<36}: {v:>14,.2f}")
        else:
            print(f"  {k:<36}: {v}")
    print(f"\n  >>> PCB PAYABLE THIS MONTH : RM {r['payable']:,.2f}")


# --------------------------------------------------------------------------
# 8. CLI MODES
# --------------------------------------------------------------------------
def mode_quick():
    """Same salary every month - simulates Jan..Dec and builds YTD automatically."""
    p = ask_profile()
    print("\n=== MONTHLY INCOME ===")
    salary = ask_num("Gross taxable monthly salary incl. taxable allowances/BIK (RM)", "5000")
    epf = ask_num("Monthly employee EPF (RM) - Enter for 11% estimate",
                  str(default_epf(salary)) if p.resident else "0")
    zakat = ask_num("Monthly zakat deducted via payroll (RM)", "0")

    bonuses, bonus_epf = {}, {}
    if ask_yn("Expecting a bonus / additional remuneration this year?"):
        while True:
            mo = ask_int("  Bonus month (1-12)", 1, 12, 12)
            amt = ask_num("  Bonus amount (RM)", "0")
            bonuses[mo] = bonuses.get(mo, D("0")) + amt
            bonus_epf[mo] = bonus_epf.get(mo, D("0")) + ask_num(
                "  EPF on bonus (RM)", str(default_epf(amt)))
            if not ask_yn("  Add another?"):
                break

    lp_by_month = {}
    if p.resident:
        tp1 = ask_tp1(p)
        if tp1 > 0:
            mo = ask_int("  Month you submit TP1 to employer (1-12)", 1, 12, 1)
            lp_by_month[mo] = tp1

    rows = simulate_year(p, salary, epf, bonuses=bonuses, bonus_epf=bonus_epf,
                         lp_by_month=lp_by_month, zakat_monthly=zakat)

    print("\n=== ESTIMATED PCB 2026 ===")
    print(f"{'Month':<6}{'Salary':>12}{'Bonus':>12}{'TP1':>10}{'PCB':>12}")
    total = D("0")
    for mo, m, r in rows:
        total += r["payable"]
        print(f"{MONTHS[mo-1]:<6}{m.Y1:>12,.2f}{m.Yt:>12,.2f}{m.LP1:>10,.2f}{r['payable']:>12,.2f}")
    print(f"{'TOTAL':<40}{total:>12,.2f}")

    if ask_yn("\nShow step-by-step breakdown for a specific month?", "y"):
        mo = ask_int("Month (1-12)", 1, 12, 1)
        print_breakdown(rows[mo - 1][2])


def mode_exact():
    """Payslip mode - enter actual YTD figures (e.g. from payslip / TP3)."""
    p = ask_profile()
    print("\n=== CURRENT MONTH ===")
    mo = ask_int("Month of calculation (1-12)", 1, 12, 1)
    Y1 = ask_num("Gross normal remuneration this month incl. taxable allowances/BIK (Y1)", "5000")
    K1 = ask_num("EPF on normal remuneration (K1)", str(default_epf(Y1)))
    Yt = ask_num("Additional remuneration this month - bonus/arrears/etc. (Yt)", "0")
    Kt = ask_num("EPF on additional remuneration (Kt)", str(default_epf(Yt)) if Yt else "0")
    zakat_now = ask_num("Zakat this month", "0")
    LP1 = ask_tp1(p) if p.resident else D("0")

    Y = K = X = Z = LP = D("0")
    if mo > 1:
        print(f"\n=== YEAR-TO-DATE (Jan - {MONTHS[mo-2]}, incl. previous employer/TP3) ===")
        Y = ask_num("Accumulated gross remuneration (ΣY)", "0")
        K = ask_num("Accumulated EPF (ΣK)", "0")
        X = ask_num("Accumulated PCB paid (X)", "0")
        Z = ask_num("Accumulated zakat paid (Z)", "0")
        LP = ask_num("Accumulated TP1 deductions claimed (ΣLP)", "0")

    m = MonthData(month=mo, Y1=Y1, K1=K1, Yt=Yt, Kt=Kt, LP1=LP1,
                  zakat_now=zakat_now, Y=Y, K=K, X=X, Z=Z, LP=LP)
    print_breakdown(calc_mtd(p, m))


# --------------------------------------------------------------------------
# 9. SELF-TEST against LHDN Exhibit 5 worked example
# --------------------------------------------------------------------------
def self_test():
    p = Profile(category=3, child_units=3)
    sal, epf = D("5500"), D("605")
    rows = simulate_year(p, sal, epf, end_month=4,
                         bonuses={4: D("8250")}, bonus_epf={4: D("908")},
                         lp_by_month={3: D("300"), 4: D("300")})
    expected = [D("110.00"), D("110.00"), D("108.20"), D("833.70")]
    got = [r["payable"] for _, _, r in rows]
    ok = got == expected
    print("Self-test vs LHDN Exhibit 5 (Jan-Apr):", [f"{g:.2f}" for g in got],
          "PASS" if ok else f"FAIL (expected {expected})")
    return ok


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" PCB / MTD ESTIMATOR 2026  (LHDN computerised method)")
    print("=" * 60)
    self_test()
    print("\nMode:\n  1) Quick estimate - fixed monthly salary, full-year projection"
          "\n  2) Exact month    - enter your actual year-to-date figures")
    if ask_int("Choose mode", 1, 2, 1) == 1:
        mode_quick()
    else:
        mode_exact()
    print("\nNote: Estimate only. Final PCB follows your employer's payroll system.")
