"""
app.py

Two-page LP optimizer Streamlit app — single file.
Page 1: worked refinery crude-oil purchasing example (readable variable names).
Page 2: user builds their own LP using plain-word variable names (e.g. Utility,
RawMaterial) — declared variables update live as the user types.

Both pages share the exact same solving pipeline (solve_lp) and the exact
same result-display logic, so behavior is consistent between them.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from scipy.optimize import linprog

st.set_page_config(page_title="LP Optimizer", page_icon="📈", layout="centered")

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ========================================================================
# LP ENGINE — parsing, linearity check, solving
# ========================================================================
def extract_identifiers(text):
    """Find every word-like token in a string (candidate variable names)."""
    return set(IDENTIFIER_RE.findall(text))


def build_symbol_locals(*texts):
    """
    Build a dict of plain sympy Symbols for every word found across the given
    strings. Passing this as `locals=` to sympify forces words like 'Utility'
    or 'beta' to always be treated as ordinary variables, never as a sympy
    built-in function or constant (e.g. E, I, S, O, pi, beta, gamma).
    """
    names = set()
    for t in texts:
        names |= extract_identifiers(t)
    return {name: sp.Symbol(name) for name in names}


def parse_expression(expr_str, local_dict=None):
    """Parse a string like '4*Utility + 7*RawMaterial' into a sympy expression."""
    expr_str = expr_str.strip().replace("^", "**")
    if not expr_str:
        raise ValueError("Empty expression.")
    return sp.sympify(expr_str, locals=local_dict or {})


def parse_constraint(constraint_str, local_dict=None):
    """
    Parse a constraint string like 'Utility + RawMaterial <= 50' into
    (lhs_minus_rhs_expr, relation_string). Everything is moved to one side
    so it can be compared to zero.
    """
    constraint_str = constraint_str.strip().replace("^", "**")
    for rel in ["<=", ">=", "=="]:
        if rel in constraint_str:
            lhs_str, rhs_str = constraint_str.split(rel)
            lhs = sp.sympify(lhs_str.strip(), locals=local_dict or {})
            rhs = sp.sympify(rhs_str.strip(), locals=local_dict or {})
            return lhs - rhs, rel
    raise ValueError(f"Constraint '{constraint_str}' must contain <=, >=, or ==")


def extract_variables(expressions):
    """Union of all symbols found across a list of sympy expressions, sorted by name."""
    variables = set()
    for e in expressions:
        variables |= e.free_symbols
    return sorted(variables, key=lambda v: v.name)


def check_linearity(expr, variables):
    """
    Deterministic linearity check: an expression is usable in an LP only if
    every variable appears with total degree <= 1, with no products between
    variables (e.g. Utility*RawMaterial or Utility**2 are rejected).
    """
    poly = sp.Poly(sp.expand(expr), *variables)
    if poly.total_degree() > 1:
        return False, f"'{sp.expand(expr)}' contains a non-linear term (degree > 1)."
    return True, ""


def build_lp_matrices(objective_expr, constraint_tuples, variables, sense="max"):
    """Convert sympy objective + constraints into matrices scipy.linprog needs."""
    objective_expr = sp.expand(objective_expr)
    c = np.array([float(objective_expr.coeff(v)) for v in variables], dtype=float)
    if sense == "max":
        c = -c  # linprog always minimizes

    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    for lhs, rel in constraint_tuples:
        lhs = sp.expand(lhs)
        row = [float(lhs.coeff(v)) for v in variables]
        const_term = float(lhs.subs({v: 0 for v in variables}))

        if rel == "<=":
            A_ub.append(row)
            b_ub.append(-const_term)
        elif rel == ">=":
            A_ub.append([-x for x in row])
            b_ub.append(const_term)
        elif rel == "==":
            A_eq.append(row)
            b_eq.append(-const_term)

    bounds = [(0, None) for _ in variables]  # default non-negativity, x_i >= 0

    return (
        c,
        np.array(A_ub) if A_ub else None,
        np.array(b_ub) if b_ub else None,
        np.array(A_eq) if A_eq else None,
        np.array(b_eq) if b_eq else None,
        bounds,
    )


def solve_lp(objective_str, constraint_strs, sense="max"):
    """
    End-to-end pipeline: parse -> check linearity -> build matrices -> solve.

    Returns a dict:
        linear (bool)            - False if objective/constraints aren't linear
        feasible (bool)          - False if no solution exists (or unbounded)
        message (str)            - human-readable status
        variables (list[str])
        values (dict[str, float])
        objective_value (float)
    """
    result = {
        "linear": True,
        "feasible": False,
        "message": "",
        "variables": [],
        "values": {},
        "objective_value": None,
    }

    # Build one shared symbol table so word-like names (Utility, FuelGas, ...)
    # are always treated as plain variables, in both objective and constraints.
    local_dict = build_symbol_locals(objective_str, *constraint_strs)

    # --- Step 1: parse objective ---
    try:
        objective_expr = parse_expression(objective_str, local_dict)
    except Exception as e:
        result["linear"] = False
        result["message"] = f"Could not parse objective function: {e}"
        return result

    # --- Step 2: parse constraints ---
    constraint_tuples = []
    for c_str in constraint_strs:
        if not c_str.strip():
            continue
        try:
            constraint_tuples.append(parse_constraint(c_str, local_dict))
        except Exception as e:
            result["linear"] = False
            result["message"] = f"Could not parse constraint '{c_str}': {e}"
            return result

    variables = extract_variables([objective_expr] + [lhs for lhs, _ in constraint_tuples])
    if not variables:
        result["linear"] = False
        result["message"] = "No variables found in the objective or constraints."
        return result

    # --- Step 3: deterministic linearity check ---
    is_linear, reason = check_linearity(objective_expr, variables)
    if not is_linear:
        result["linear"] = False
        result["message"] = f"Objective function is not linear: {reason}"
        return result

    for lhs, _ in constraint_tuples:
        is_linear, reason = check_linearity(lhs, variables)
        if not is_linear:
            result["linear"] = False
            result["message"] = f"A constraint is not linear: {reason}"
            return result

    # --- Step 4: solve ---
    c, A_ub, b_ub, A_eq, b_eq, bounds = build_lp_matrices(
        objective_expr, constraint_tuples, variables, sense
    )
    lp_result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    result["variables"] = [str(v) for v in variables]

    if lp_result.status == 0:
        result["feasible"] = True
        result["values"] = {str(v): round(float(val), 4) for v, val in zip(variables, lp_result.x)}
        obj_val = lp_result.fun if sense == "min" else -lp_result.fun
        result["objective_value"] = round(float(obj_val), 4)
        result["message"] = "Optimal solution found."
    elif lp_result.status == 2:
        result["message"] = "Infeasible: no combination of variables satisfies every constraint at once."
    elif lp_result.status == 3:
        result["message"] = "Unbounded: the objective can improve forever — a constraint is likely missing."
    else:
        result["message"] = f"Solver status {lp_result.status}: {lp_result.message}"

    return result


# ========================================================================
# SHARED UI HELPERS — used by both pages
# ========================================================================
def display_results(objective_str, constraint_strs, sense, result):
    st.subheader("Linearity & feasibility check")

    if not result["linear"]:
        st.error(f"❌ Not solvable as an LP — {result['message']}")
        st.caption("Every term must be degree ≤ 1 in each variable (no A·B, no A², no 1/A).")
        return
    st.success("✅ Objective and constraints are linear — valid LP formulation.")

    if not result["feasible"]:
        st.error(f"❌ {result['message']}")
        return
    st.success(f"✅ {result['message']}")

    st.subheader("Optimal solution")
    st.metric("Objective value (Z)", f"{result['objective_value']:,}")

    df = pd.DataFrame(
        {"Variable": list(result["values"].keys()), "Optimal value": list(result["values"].values())}
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(df["Variable"], df["Optimal value"], color="#2a78d6")
    ax.set_ylabel("Value")
    ax.set_title("Optimal variable values")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

    # Bonus: for exactly 2 variables, draw the classic feasible-region plot
    if len(result["variables"]) == 2:
        st.subheader("Graphical view (2-variable reduction)")
        plot_feasible_region(objective_str, constraint_strs, result)


def plot_feasible_region(objective_str, constraint_strs, result):
    """Draws the feasible region, constraint lines, and optimal point for a 2-variable LP."""
    var_names = result["variables"]
    local_dict = build_symbol_locals(objective_str, *constraint_strs)
    vx, vy = local_dict[var_names[0]], local_dict[var_names[1]]

    constraints = []
    for c_str in constraint_strs:
        if c_str.strip():
            lhs, rel = parse_constraint(c_str, local_dict)
            constraints.append((sp.expand(lhs), rel))

    opt_x = result["values"][var_names[0]]
    opt_y = result["values"][var_names[1]]
    bound = max(opt_x, opt_y, 10) * 1.6

    xs = np.linspace(0, bound, 250)
    ys = np.linspace(0, bound, 250)
    X, Y = np.meshgrid(xs, ys)
    feasible = np.ones_like(X, dtype=bool)

    fig, ax = plt.subplots(figsize=(5, 5))
    for lhs, rel in constraints:
        f = sp.lambdify((vx, vy), lhs, "numpy")
        Z = f(X, Y)
        if rel == "<=":
            feasible &= Z <= 0
        elif rel == ">=":
            feasible &= Z >= 0
        elif rel == "==":
            feasible &= np.abs(Z) < 1e-6
        ax.contour(X, Y, Z, levels=[0], colors="#D85A30", linewidths=1.2)

    feasible &= (X >= 0) & (Y >= 0)
    ax.contourf(X, Y, feasible.astype(int), levels=[0.5, 1], colors=["#9FE1CB"], alpha=0.5)
    ax.scatter([opt_x], [opt_y], color="#0C447C", s=70, zorder=5, label="Optimal point")
    ax.set_xlabel(var_names[0])
    ax.set_ylabel(var_names[1])
    ax.set_title("Feasible region & optimal vertex")
    ax.legend()
    st.pyplot(fig)


# ========================================================================
# PAGE 1 — worked example (readable variable names: Oman, Tapis, Labuan, Murban)
# ========================================================================
def page_example():
    st.header("Refinery crude oil purchasing")

    st.markdown(
        """
**Problem statement**

A refinery buys crude oil from several suppliers and refines it into
gasoline, diesel, and fuel oil. Each crude type costs a different amount
per m³ and yields a different mix of finished products — some crudes are
richer in gasoline, others in diesel or fuel oil.

The refinery wants to decide **how much of each crude to buy and process
per day** in order to **maximize its Gross Refinery Margin (GRM)** —
total product revenue minus crude purchase cost — while staying within:
- how much of each crude its suppliers can actually deliver,
- the refinery's total daily processing capacity, and
- how much gasoline, diesel, and fuel oil the market will buy (demand bounds).

Four crude types are available: **Oman, Tapis, Labuan,** and **Murban.**
"""
    )

    sense = "max"
    objective_str = (
        "650*(0.35*Oman+0.45*Tapis+0.30*Labuan+0.40*Murban) + "
        "580*(0.40*Oman+0.30*Tapis+0.25*Labuan+0.35*Murban) + "
        "350*(0.15*Oman+0.10*Tapis+0.30*Labuan+0.15*Murban) - "
        "420*Oman - 460*Tapis - 440*Labuan - 450*Murban"
    )
    constraint_strs = [
        "Oman <= 260000",
        "Tapis <= 45000",
        "Labuan <= 40000",
        "Murban <= 95000",
        "Oman + Tapis + Labuan + Murban <= 300000",
        "0.35*Oman + 0.45*Tapis + 0.30*Labuan + 0.40*Murban >= 90000",
        "0.35*Oman + 0.45*Tapis + 0.30*Labuan + 0.40*Murban <= 130000",
        "0.40*Oman + 0.30*Tapis + 0.25*Labuan + 0.35*Murban >= 60000",
        "0.40*Oman + 0.30*Tapis + 0.25*Labuan + 0.35*Murban <= 90000",
        "0.15*Oman + 0.10*Tapis + 0.30*Labuan + 0.15*Murban >= 20000",
    ]

    with st.expander("Show objective function & constraints"):
        st.markdown("**Decision variables:** daily m³ of each crude processed — "
                     "`Oman`, `Tapis`, `Labuan`, `Murban`")
        st.markdown("**Objective (maximize GRM):**")
        st.code(objective_str, language="text")
        st.markdown("**Constraints:**")
        for c in constraint_strs:
            st.code(c, language="text")

    if st.button("Solve example", type="primary"):
        result = solve_lp(objective_str, constraint_strs, sense)
        display_results(objective_str, constraint_strs, sense, result)


# ========================================================================
# PAGE 2 — custom user-built LP (word-based variable names)
# ========================================================================
def page_custom():
    st.header("Build your own LP problem")
    st.caption(
        "Use plain, meaningful variable names instead of x, y — e.g. `Utility`, "
        "`RawMaterial`, `FuelGas`. Any word works as a variable."
    )

    st.subheader("1. Objective function")
    sense_label = st.radio("Direction:", ["Maximize", "Minimize"], horizontal=True)
    objective_str = st.text_area(
        "Enter your objective function:",
        placeholder="e.g. 4*Utility + 7*RawMaterial",
        key="objective_input",
    )

    obj_vars = []
    if objective_str.strip():
        try:
            local_dict = build_symbol_locals(objective_str)
            expr = parse_expression(objective_str, local_dict)
            obj_vars = sorted(str(v) for v in expr.free_symbols)
        except Exception as e:
            st.warning(f"Can't parse this yet: {e}")

    if obj_vars:
        st.markdown("**Declared variables so far:** " + ", ".join(f"`{v}`" for v in obj_vars))
    else:
        st.caption("Declared variables so far: none yet — start typing a variable name above.")

    st.subheader("2. Constraints")
    st.caption("One constraint per line, using <=, >=, or ==. Example: Utility + RawMaterial >= 50")
    constraints_text = st.text_area(
        "Enter constraints:",
        placeholder="Utility + RawMaterial >= 50\nUtility <= 30\nRawMaterial <= 40",
        height=140,
        key="constraints_input",
    )
    constraint_strs = [c for c in constraints_text.split("\n") if c.strip()]

    all_vars = set(obj_vars)
    for c_str in constraint_strs:
        all_vars |= extract_identifiers(c_str)
    # drop bare numbers accidentally matched (identifiers regex won't match pure numbers, so this is just variables)
    all_vars = sorted(all_vars)

    if all_vars:
        st.markdown("**All declared variables (objective + constraints):** "
                     + ", ".join(f"`{v}`" for v in all_vars))

    st.caption("Note: all variables are assumed non-negative (≥ 0) by default, as is standard for LPs.")

    if st.button("Check & solve", type="primary"):
        if not objective_str.strip():
            st.error("Please enter an objective function first.")
            return
        sense = "max" if sense_label == "Maximize" else "min"
        result = solve_lp(objective_str, constraint_strs, sense)
        display_results(objective_str, constraint_strs, sense, result)


# ========================================================================
# NAVIGATION + ROUTER
# ========================================================================
if "page" not in st.session_state:
    st.session_state.page = "example"

st.title("📈 Linear Programming Optimizer")

col1, col2 = st.columns(2)
with col1:
    if st.button("📘 Example: Refinery Crude LP", use_container_width=True,
                 type="primary" if st.session_state.page == "example" else "secondary"):
        st.session_state.page = "example"
with col2:
    if st.button("✍️ Build Your Own LP", use_container_width=True,
                 type="primary" if st.session_state.page == "custom" else "secondary"):
        st.session_state.page = "custom"

st.divider()

if st.session_state.page == "example":
    page_example()
else:
    page_custom()
