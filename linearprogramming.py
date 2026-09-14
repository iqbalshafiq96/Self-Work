"""
app.py

Two-page LP optimizer Streamlit app - single file.
Page 1: worked refinery crude-oil purchasing example (readable variable names).
Page 2: user builds their own LP using plain-word variable names (e.g. Utility,
RawMaterial) - declared variables update live as the user types.

Both pages share the exact same solving pipeline (solve_lp) and the exact
same result-display logic, so behavior is consistent between them.

Visualization: results are always shown as an elegant contour plot (feasible
region + objective contour lines + optimal point). A contour plot is only
meaningful in two dimensions, so when a problem has more than two decision
variables, the user picks which two variables to put on the axes; every
other variable is held fixed at its optimal (solved) value, and the plotted
slice passes exactly through the true optimum.
"""

import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from scipy.optimize import linprog

st.set_page_config(page_title="LP Optimizer", page_icon="📈", layout="centered")

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

RESERVED_WORDS = {
    "Min", "Max", "min", "max", "Abs", "abs", "sin", "cos", "tan",
    "exp", "log", "sqrt", "True", "False", "None", "and", "or", "not"
}

# ========================================================================
# VISUAL STYLE - shared palette + rcParams for a clean, professional look
# ========================================================================
PALETTE = {
    "blue_light": "#B5D4F4",
    "teal": "#1D9E75",
    "teal_dark": "#0F6E56",
    "coral_dark": "#993C1D",
    "blue": "#2a78d6",
    "gray_light": "#e1e0d9",
    "text": "#3d3d3a",
    "text_secondary": "#73726c",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": PALETTE["text"],
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.labelcolor": PALETTE["text"],
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.grid": True,
    "grid.color": PALETTE["gray_light"],
    "grid.linewidth": 0.6,
    "xtick.color": PALETTE["text_secondary"],
    "ytick.color": PALETTE["text_secondary"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "legend.fontsize": 9.5,
})


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=3, width=0.6)


# ========================================================================
# LP ENGINE - parsing, linearity check, solving
# ========================================================================
def extract_identifiers(text: str) -> set:
    tokens = set(IDENTIFIER_RE.findall(text))
    return tokens - RESERVED_WORDS


def build_symbol_locals(*texts) -> dict:
    all_tokens = set()
    for text in texts:
        all_tokens.update(extract_identifiers(text))
    return {token: sp.Symbol(token) for token in all_tokens}


def parse_equation_or_inequality(expr_str: str, local_dict: dict):
    expr_str = expr_str.strip()
    if not expr_str:
        return None, None

    if "<=" in expr_str:
        lhs, rhs = expr_str.split("<=", 1)
        rel = "<="
    elif ">=" in expr_str:
        lhs, rhs = expr_str.split(">=", 1)
        rel = ">="
    elif "=" in expr_str:
        lhs, rhs = expr_str.split("=", 1)
        rel = "=="
    else:
        raise ValueError(f"Constraint standard sign missing ('<=', '>=', '='): {expr_str}")

    sym_lhs = sp.sympify(lhs, locals=local_dict)
    sym_rhs = sp.sympify(rhs, locals=local_dict)
    diff = sym_lhs - sym_rhs
    return diff, rel


def solve_lp(objective_str: str, constraints_list: list, sense: str, var_names: list, bounds_dict: dict):
    """
    Universal LP solver. Alongside the solution, returns everything needed to
    redraw the problem geometrically: A_ub/b_ub, A_eq/b_eq, bounds, the
    *original* (non sign-flipped) objective coefficients, and the ordered
    variable names.
    """
    if not var_names:
        return {"success": False, "message": "No variables defined."}

    var_names = sorted(list(var_names))
    sym_vars = [sp.Symbol(v) for v in var_names]
    local_dict = {v: sym_vars[i] for i, v in enumerate(var_names)}

    try:
        obj_expr = sp.sympify(objective_str, locals=local_dict)
    except Exception as e:
        return {"success": False, "message": f"Error parsing objective function: {e}"}

    c_natural = []
    for var in sym_vars:
        coeff = obj_expr.coeff(var)
        if any(v in coeff.free_symbols for v in sym_vars):
            return {"success": False, "message": f"Non-linear term detected in objective for variable {var}."}
        c_natural.append(float(coeff))

    c = list(c_natural)
    if sense.lower() == "maximize":
        c = [-val for val in c]

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    for constr in constraints_list:
        if not constr.strip():
            continue
        try:
            diff, rel = parse_equation_or_inequality(constr, local_dict)
        except Exception as e:
            return {"success": False, "message": f"Error parsing constraint '{constr}': {e}"}

        const_term = float(diff.as_coefficients_dict().get(1, 0))
        coeffs = []
        for var in sym_vars:
            coeff = diff.coeff(var)
            if any(v in coeff.free_symbols for v in sym_vars):
                return {"success": False, "message": f"Non-linear term detected in constraint '{constr}'."}
            coeffs.append(float(coeff))

        if rel == "<=":
            A_ub.append(coeffs)
            b_ub.append(-const_term)
        elif rel == ">=":
            A_ub.append([-val for val in coeffs])
            b_ub.append(const_term)
        elif rel == "==":
            A_eq.append(coeffs)
            b_eq.append(-const_term)

    bounds = [bounds_dict.get(v, (0, None)) for v in var_names]

    res = linprog(
        c=c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method="highs"
    )

    if res.success:
        opt_val = -res.fun if sense.lower() == "maximize" else res.fun
        solution = dict(zip(var_names, res.x))
        return {
            "success": True,
            "fun": opt_val,
            "x": solution,
            "message": res.message,
            "status": res.status,
            "var_names": var_names,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "obj_coeffs": c_natural,
            "sense": sense,
        }
    else:
        return {"success": False, "message": f"Solver failed: {res.message}"}


# ========================================================================
# VISUALIZATION - contour plot, sliced to any chosen pair of variables
# ========================================================================
def slice_2d(result: dict, xn: str, yn: str) -> dict:
    """
    Reduce an N-variable LP result to a 2D slice through the true optimum:
    the two chosen variables (xn, yn) vary, every other variable is fixed at
    its optimal value. Constraint and objective coefficients are adjusted
    accordingly, so the slice's optimal contour value still equals the
    problem's real optimal objective value.
    """
    var_names = result["var_names"]
    ix, iy = var_names.index(xn), var_names.index(yn)
    other_idx = [i for i in range(len(var_names)) if i not in (ix, iy)]
    fixed_vals = {i: result["x"][var_names[i]] for i in other_idx}

    def reduce_row(coeffs, b):
        offset = sum(coeffs[i] * fixed_vals[i] for i in other_idx)
        return [coeffs[ix], coeffs[iy]], b - offset

    A_ub2, b_ub2 = [], []
    for coeffs, b in zip(result["A_ub"], result["b_ub"]):
        c2, b2 = reduce_row(coeffs, b)
        A_ub2.append(c2)
        b_ub2.append(b2)

    A_eq2, b_eq2 = [], []
    for coeffs, b in zip(result["A_eq"], result["b_eq"]):
        c2, b2 = reduce_row(coeffs, b)
        A_eq2.append(c2)
        b_eq2.append(b2)

    bounds2 = [result["bounds"][ix], result["bounds"][iy]]
    c_full = result["obj_coeffs"]
    offset_obj = sum(c_full[i] * fixed_vals[i] for i in other_idx)

    return {
        "xn": xn, "yn": yn,
        "opt_x": result["x"][xn], "opt_y": result["x"][yn],
        "A_ub": A_ub2, "b_ub": b_ub2, "A_eq": A_eq2, "b_eq": b_eq2,
        "bounds": bounds2,
        "c0": c_full[ix], "c1": c_full[iy], "offset": offset_obj,
        "sense": result["sense"], "opt_val": result["fun"],
        "fixed_vals": {var_names[i]: fixed_vals[i] for i in other_idx},
    }


def _feasible_mask(X, Y, A_ub, b_ub, A_eq, b_eq, bounds, tol=1e-6):
    mask = np.ones_like(X, dtype=bool)
    for coeffs, b in zip(A_ub, b_ub):
        mask &= (coeffs[0] * X + coeffs[1] * Y) <= b + tol
    for coeffs, b in zip(A_eq, b_eq):
        mask &= np.abs(coeffs[0] * X + coeffs[1] * Y - b) <= 1e-4 * max(1.0, abs(b))
    lo0, hi0 = bounds[0]
    lo1, hi1 = bounds[1]
    mask &= X >= (lo0 if lo0 is not None else -np.inf) - tol
    mask &= Y >= (lo1 if lo1 is not None else -np.inf) - tol
    if hi0 is not None:
        mask &= X <= hi0 + tol
    if hi1 is not None:
        mask &= Y <= hi1 + tol
    return mask


def _plot_window(A_ub, b_ub, A_eq, b_eq, bounds, opt_x, opt_y, pad=1.35):
    candidates = [max(opt_x, opt_y, 1.0) * 1.5]
    rows = list(zip(A_ub, b_ub)) + list(zip(A_eq, b_eq))
    for coeffs, b in rows:
        a0, a1 = coeffs
        if a0 not in (0, 0.0) and b / a0 > 0:
            candidates.append(b / a0)
        if a1 not in (0, 0.0) and b / a1 > 0:
            candidates.append(b / a1)
    for lo, hi in bounds:
        if hi is not None:
            candidates.append(hi)
    upper = max(candidates) * pad
    return max(upper, 1.0)


def plot_contour(sl: dict):
    """Elegant contour plot for a 2D slice: shaded feasible region, objective
    contour lines, and the optimal point highlighted."""
    xn, yn = sl["xn"], sl["yn"]
    opt_x, opt_y = sl["opt_x"], sl["opt_y"]
    A_ub, b_ub = sl["A_ub"], sl["b_ub"]
    A_eq, b_eq = sl["A_eq"], sl["b_eq"]
    bounds = sl["bounds"]
    c0, c1, offset = sl["c0"], sl["c1"], sl["offset"]
    sense = sl["sense"]
    opt_val = sl["opt_val"]

    upper = _plot_window(A_ub, b_ub, A_eq, b_eq, bounds, opt_x, opt_y)
    grid = np.linspace(0, upper, 400)
    X, Y = np.meshgrid(grid, grid)
    mask = _feasible_mask(X, Y, A_ub, b_ub, A_eq, b_eq, bounds)
    Z = c0 * X + c1 * Y + offset

    fig, ax = plt.subplots(figsize=(7, 5.8), dpi=150)

    ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5],
                colors=[PALETTE["blue_light"]], alpha=0.45)

    for coeffs, b in zip(A_ub, b_ub):
        a0, a1 = coeffs
        if a0 == 0 and a1 == 0:
            continue
        xs = np.array([0, upper])
        if a1 != 0:
            ys = (b - a0 * xs) / a1
        else:
            xs = np.array([b / a0, b / a0])
            ys = np.array([0, upper])
        ax.plot(xs, ys, color=PALETTE["coral_dark"], linestyle=(0, (5, 4)),
                 linewidth=1.3, alpha=0.85)

    feasible_vals = Z[mask]
    if feasible_vals.size > 0:
        lo = np.percentile(feasible_vals, 15)
        levels = np.linspace(lo, opt_val, 4)
        for lvl in levels[:-1]:
            ax.contour(X, Y, Z, levels=[lvl], colors=[PALETTE["teal"]],
                        linewidths=1.1, linestyles=(0, (4, 4)), alpha=0.6)
        ax.contour(X, Y, Z, levels=[opt_val], colors=[PALETTE["teal_dark"]], linewidths=2.2)

    ax.scatter([opt_x], [opt_y], s=70, color=PALETTE["teal_dark"],
               edgecolor="white", linewidth=1.6, zorder=5)
    ax.annotate(
        f"Optimum\n({opt_x:,.2f}, {opt_y:,.2f})\nZ = {opt_val:,.2f}",
        xy=(opt_x, opt_y), xytext=(12, 12), textcoords="offset points",
        fontsize=9.5, color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c3c2b7", lw=0.7),
    )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel(xn)
    ax.set_ylabel(yn)
    ax.set_title(f"Feasible region & objective contours ({sense.lower()} Z)", pad=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    _clean_axes(ax)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["blue_light"],
               markeredgecolor="none", markersize=12, label="Feasible region"),
        Line2D([0], [0], color=PALETTE["coral_dark"], linestyle=(0, (5, 4)), lw=1.3,
               label="Constraint boundary"),
        Line2D([0], [0], color=PALETTE["teal"], linestyle=(0, (4, 4)), lw=1.1,
               label="Objective contour"),
        Line2D([0], [0], color=PALETTE["teal_dark"], lw=2.2, label="Optimal contour"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["teal_dark"],
               markeredgecolor="white", markersize=8, label="Optimal point"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, handlelength=2.2)

    fig.tight_layout()
    st.pyplot(fig)


def display_results(result: dict, sense: str, key_prefix: str = ""):
    """Shared UI rendering logic for LP execution results."""
    if not result["success"]:
        st.error(f"Solver Error: {result['message']}")
        return

    st.success("Optimization completed successfully")
    st.metric(label=f"Optimal objective value ({sense})", value=f"{result['fun']:,.4f}")

    st.subheader("Optimal decision variable values")
    df_res = pd.DataFrame(list(result["x"].items()), columns=["Variable", "Optimal Value"])
    st.dataframe(df_res.style.format({"Optimal Value": "{:,.4f}"}), use_container_width=True)

    st.subheader("Visualization")
    var_names = result["var_names"]

    if len(var_names) < 2:
        st.info("Add at least one more decision variable to visualize a contour plot.")
        return

    if len(var_names) == 2:
        xn, yn = var_names
    else:
        st.caption(
            "This problem has more than two decision variables, so pick which two "
            "to plot — every other variable is held fixed at its optimal value."
        )
        col1, col2 = st.columns(2)
        with col1:
            xn = st.selectbox("X-axis variable", var_names, index=0, key=f"{key_prefix}_xvar")
        remaining = [v for v in var_names if v != xn]
        with col2:
            default_idx = min(1, len(remaining) - 1) if len(remaining) > 1 else 0
            yn = st.selectbox("Y-axis variable", remaining, index=default_idx, key=f"{key_prefix}_yvar")

    sl = slice_2d(result, xn, yn)
    if sl["fixed_vals"]:
        fixed_str = ", ".join(f"{k} = {v:,.2f}" for k, v in sl["fixed_vals"].items())
        st.caption(f"Other variables held at their optimal values: {fixed_str}")
    plot_contour(sl)


# ========================================================================
# NAVIGATION - two pages toggled by buttons
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


# ------------------------------------------------------------------------
# PAGE 1: Worked Refinery Example
# ------------------------------------------------------------------------
def page_example():
    st.header("Refinery crude oil purchasing")
    st.markdown(
        """
A refinery buys crude oil from several suppliers and refines it into gasoline, diesel,
and fuel oil. Each crude costs a different amount per m³ and yields a different mix of
finished products - some crudes are richer in gasoline, others in diesel or fuel oil.

The refinery wants to decide **how much of each crude to buy and process per day** in
order to **maximize Gross Refinery Margin (GRM)** - total product revenue minus crude
purchase cost - while staying within crude supply limits, total daily processing
(throughput) capacity, and how much of each product the market will absorb.

Four crude types are available: **Oman, Tapis, Labuan,** and **Murban.**
"""
    )

    objective_str = (
        "650*(0.35*Oman+0.45*Tapis+0.30*Labuan+0.40*Murban) + "
        "580*(0.40*Oman+0.30*Tapis+0.25*Labuan+0.35*Murban) + "
        "350*(0.15*Oman+0.10*Tapis+0.30*Labuan+0.15*Murban) - "
        "420*Oman - 460*Tapis - 440*Labuan - 450*Murban"
    )
    constraint_items = [
        ("Oman <= 260000", "Oman supply limit — max Oman available per day (m³/day)"),
        ("Tapis <= 45000", "Tapis supply limit"),
        ("Labuan <= 40000", "Labuan supply limit"),
        ("Murban <= 95000", "Murban supply limit"),
        ("Oman + Tapis + Labuan + Murban <= 300000",
         "Throughput capacity — total crude the refinery can process, 300,000 m³/day"),
        ("0.35*Oman + 0.45*Tapis + 0.30*Labuan + 0.40*Murban >= 90000", "Gasoline demand floor"),
        ("0.35*Oman + 0.45*Tapis + 0.30*Labuan + 0.40*Murban <= 130000", "Gasoline demand ceiling"),
        ("0.40*Oman + 0.30*Tapis + 0.25*Labuan + 0.35*Murban >= 60000", "Diesel demand floor"),
        ("0.40*Oman + 0.30*Tapis + 0.25*Labuan + 0.35*Murban <= 90000", "Diesel demand ceiling"),
        ("0.15*Oman + 0.10*Tapis + 0.30*Labuan + 0.15*Murban >= 20000", "Fuel oil minimum"),
    ]
    constraint_strs = [expr for expr, _ in constraint_items]

    with st.expander("Show objective function & constraints", expanded=True):
        st.markdown("**Decision variables:** daily m³ of each crude processed — "
                     "`Oman`, `Tapis`, `Labuan`, `Murban`")

        st.markdown("**Objective (maximize GRM):**")
        st.code(objective_str, language="text")
        st.caption("Revenue from gasoline + diesel + fuel oil, minus crude purchase cost.")

        st.markdown("**Constraints:**")
        for expr, note in constraint_items:
            st.code(expr, language="text")
            st.caption(note)

    var_names = ["Oman", "Tapis", "Labuan", "Murban"]
    bounds_dict = {v: (0, None) for v in var_names}

    if st.button("Solve example", type="primary"):
        st.session_state["result_example"] = solve_lp(
            objective_str=objective_str,
            constraints_list=constraint_strs,
            sense="Maximize",
            var_names=var_names,
            bounds_dict=bounds_dict
        )

    if "result_example" in st.session_state:
        display_results(st.session_state["result_example"], "Maximize", key_prefix="example")


# ------------------------------------------------------------------------
# PAGE 2: User Custom LP Builder
# ------------------------------------------------------------------------
def page_custom():
    st.header("Build your own LP problem")
    st.caption(
        "Use plain, meaningful variable names instead of x, y — e.g. `Utility`, "
        "`RawMaterial`. Any word works as a variable, and the detected list "
        "below updates as you type. With more than two variables, you'll be "
        "able to pick which two to plot after solving."
    )

    col_opt, col_sense = st.columns([3, 1])
    with col_sense:
        sense = st.selectbox("Optimization Sense", ["Maximize", "Minimize"])
    with col_opt:
        obj_input = st.text_input("Objective Function", "40 * Utility + 30 * RawMaterial")

    st.subheader("Constraints")
    st.caption("Enter one constraint per line using `<=`, `>=`, or `=`.")
    constraints_input = st.text_area(
        "Constraints List",
        value="2 * Utility + 1 * RawMaterial <= 100\n1 * Utility + 2 * RawMaterial <= 80",
        height=120
    )

    all_text = obj_input + "\n" + constraints_input
    detected_vars = sorted(list(extract_identifiers(all_text)))

    st.subheader("Variable Bounds")
    if detected_vars:
        st.info(f"Detected Variables ({len(detected_vars)}): " + ", ".join(detected_vars))
        bounds_dict = {}
        cols = st.columns(min(len(detected_vars), 4))
        for i, var in enumerate(detected_vars):
            with cols[i % 4]:
                st.write(f"**{var}**")
                min_val = st.number_input(f"Min ({var})", value=0.0, key=f"min_{var}")
                has_max = st.checkbox(f"Set Max ({var})", key=f"has_max_{var}")
                max_val = st.number_input(f"Max ({var})", value=100.0, key=f"max_{var}") if has_max else None
                bounds_dict[var] = (min_val, max_val)
    else:
        st.warning("No variables detected yet. Type an objective function or constraint above.")
        bounds_dict = {}

    st.divider()

    if st.button("Solve Custom LP", type="primary"):
        constraints_list = [c.strip() for c in constraints_input.split("\n") if c.strip()]
        st.session_state["result_custom"] = solve_lp(
            objective_str=obj_input,
            constraints_list=constraints_list,
            sense=sense,
            var_names=detected_vars,
            bounds_dict=bounds_dict
        )

    if "result_custom" in st.session_state:
        display_results(st.session_state["result_custom"], sense, key_prefix="custom")


# ========================================================================
# ROUTER
# ========================================================================
if st.session_state.page == "example":
    page_example()
else:
    page_custom()
