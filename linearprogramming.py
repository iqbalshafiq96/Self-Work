"""
app.py

Two-page LP optimizer Streamlit app - single file.
Page 1: worked refinery crude-oil purchasing example (readable variable names).
Page 2: user builds their own LP using plain-word variable names (e.g. Utility,
RawMaterial) - declared variables update live as the user types.

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

# Reserved words/functions that SymPy or Python use that shouldn't be treated as user variables
RESERVED_WORDS = {
    "Min", "Max", "min", "max", "Abs", "abs", "sin", "cos", "tan",
    "exp", "log", "sqrt", "True", "False", "None", "and", "or", "not"
}

# ========================================================================
# LP ENGINE - parsing, linearity check, solving
# ========================================================================
def extract_identifiers(text: str) -> set:
    """Find every word-like token in a string (candidate variable names)."""
    tokens = set(IDENTIFIER_RE.findall(text))
    return tokens - RESERVED_WORDS


def build_symbol_locals(*texts) -> dict:
    """Build a dict of plain sympy Symbols for every word found across texts."""
    all_tokens = set()
    for text in texts:
        all_tokens.update(extract_identifiers(text))
    return {token: sp.Symbol(token) for token in all_tokens}


def parse_equation_or_inequality(expr_str: str, local_dict: dict):
    """
    Parses string equations/inequalities into a standard SymPy expression
    normalized to: Expression <= 0, Expression >= 0, or Expression == 0.
    """
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
    Universal LP solver using SymPy for algebraic parsing and SciPy linprog for numerical optimization.
    """
    if not var_names:
        return {"success": False, "message": "No variables defined."}

    # Order variables deterministically
    var_names = sorted(list(var_names))
    sym_vars = [sp.Symbol(v) for v in var_names]
    local_dict = {v: sym_vars[i] for i, v in enumerate(var_names)}

    # Parse Objective
    try:
        obj_expr = sp.sympify(objective_str, locals=local_dict)
    except Exception as e:
        return {"success": False, "message": f"Error parsing objective function: {e}"}

    # Extract objective coefficients (c vector)
    c = []
    for var in sym_vars:
        coeff = obj_expr.coeff(var)
        # Verify linearity (no variable left in the coefficient)
        if any(v in coeff.free_symbols for v in sym_vars):
            return {"success": False, "message": f"Non-linear term detected in objective for variable {var}."}
        c.append(float(coeff))

    if sense.lower() == "maximize":
        c = [-val for val in c]  # linprog minimizes by default

    # Parse Constraints
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    for constr in constraints_list:
        if not constr.strip():
            continue
        try:
            diff, rel = parse_equation_or_inequality(constr, local_dict)
        except Exception as e:
            return {"success": False, "message": f"Error parsing constraint '{constr}': {e}"}

        # Extract constant term and linear coefficients
        const_term = float(diff.as_coefficients_dict().get(1, 0))
        coeffs = []
        for var in sym_vars:
            coeff = diff.coeff(var)
            if any(v in coeff.free_symbols for v in sym_vars):
                return {"success": False, "message": f"Non-linear term detected in constraint '{constr}'."}
            coeffs.append(float(coeff))

        # Re-arrange: coeff*vars + const (rel) 0  =>  coeff*vars (rel) -const
        if rel == "<=":
            A_ub.append(coeffs)
            b_ub.append(-const_term)
        elif rel == ">=":
            A_ub.append([-val for val in coeffs])
            b_ub.append(const_term)
        elif rel == "==":
            A_eq.append(coeffs)
            b_eq.append(-const_term)

    # Format bounds for linprog
    bounds = [bounds_dict.get(v, (0, None)) for v in var_names]

    # Run SciPy linprog
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
            "status": res.status
        }
    else:
        return {"success": False, "message": f"Solver failed: {res.message}"}


def display_results(result: dict, sense: str):
    """Shared UI rendering logic for LP execution results."""
    if result["success"]:
        st.success("Optimization Completed Successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"Optimal Objective Value ({sense})", value=f"{result['fun']:,.4f}")

        st.subheader("Optimal Decision Variable Values")
        df_res = pd.DataFrame(
            list(result["x"].items()),
            columns=["Variable", "Optimal Value"]
        )
        st.dataframe(df_res.style.format({"Optimal Value": "{:,.4f}"}), use_container_width=True)

        # Plot variable allocations bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(df_res["Variable"], df_res["Optimal Value"], color="#4C72B0")
        ax.set_ylabel("Value")
        ax.set_title("Decision Variable Allocations")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.error(f"Solver Error: {result['message']}")


# ========================================================================
# STREAMLIT UI - Navigation and Page Routing
# ========================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Page 1: Worked Refinery Example", "Page 2: Custom LP Builder"])


# ------------------------------------------------------------------------
# PAGE 1: Worked Refinery Example
# ------------------------------------------------------------------------
if page == "Page 1: Worked Refinery Example":
    st.title("🛢️ Refinery Crude Oil Purchasing LP")
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

    st.subheader("Objective function")
    st.latex(
        r"""
        \text{Maximize GRM} =\ 650(0.35\,Oman + 0.45\,Tapis + 0.30\,Labuan + 0.40\,Murban)
        """
    )
    st.caption("# Revenue from gasoline — price $650/m³ × each crude's gasoline yield")
    st.latex(
        r"""
        {}+\ 580(0.40\,Oman + 0.30\,Tapis + 0.25\,Labuan + 0.35\,Murban)
        """
    )
    st.caption("# Revenue from diesel — price $580/m³ × each crude's diesel yield")
    st.latex(
        r"""
        {}+\ 350(0.15\,Oman + 0.10\,Tapis + 0.30\,Labuan + 0.15\,Murban)
        """
    )
    st.caption("# Revenue from fuel oil — price $350/m³ × each crude's fuel oil yield")
    st.latex(
        r"""
        {}-\ 420\,Oman - 460\,Tapis - 440\,Labuan - 450\,Murban
        """
    )
    st.caption("# Crude purchase cost — price per m³ × volume bought, for each crude")

    st.subheader("Constraints")

    st.latex(r"Oman \le 260{,}000 \qquad Tapis \le 45{,}000 \qquad Labuan \le 40{,}000 \qquad Murban \le 95{,}000")
    st.caption("# Crude supply limits — the most each supplier can deliver per day (m³/day)")

    st.latex(r"Oman + Tapis + Labuan + Murban \le 300{,}000")
    st.caption("# Throughput capacity — total crude the refinery can physically process, 300,000 m³/day")

    st.latex(
        r"90{,}000 \le 0.35\,Oman + 0.45\,Tapis + 0.30\,Labuan + 0.40\,Murban \le 130{,}000"
    )
    st.caption("# Gasoline demand band — the market absorbs between 90,000 and 130,000 m³/day")

    st.latex(
        r"60{,}000 \le 0.40\,Oman + 0.30\,Tapis + 0.25\,Labuan + 0.35\,Murban \le 90{,}000"
    )
    st.caption("# Diesel demand band")

    st.latex(r"0.15\,Oman + 0.10\,Tapis + 0.30\,Labuan + 0.15\,Murban \ge 20{,}000")
    st.caption("# Fuel oil minimum — the refinery must produce at least this much fuel oil per day")

    # ---- solver inputs, matching the formulas & comments shown above ----
    objective_str = (
        "650*(0.35*Oman+0.45*Tapis+0.30*Labuan+0.40*Murban) + "   # revenue: gasoline
        "580*(0.40*Oman+0.30*Tapis+0.25*Labuan+0.35*Murban) + "   # revenue: diesel
        "350*(0.15*Oman+0.10*Tapis+0.30*Labuan+0.15*Murban) - "   # revenue: fuel oil
        "420*Oman - 460*Tapis - 440*Labuan - 450*Murban"          # cost: crude purchase
    )
    constraints_list = [
        "Oman <= 260000",                                              # Oman supply limit
        "Tapis <= 45000",                                              # Tapis supply limit
        "Labuan <= 40000",                                             # Labuan supply limit
        "Murban <= 95000",                                             # Murban supply limit
        "Oman + Tapis + Labuan + Murban <= 300000",                    # throughput capacity: 300,000 m3/d
        "0.35*Oman + 0.45*Tapis + 0.30*Labuan + 0.40*Murban >= 90000", # gasoline demand floor
        "0.35*Oman + 0.45*Tapis + 0.30*Labuan + 0.40*Murban <= 130000",# gasoline demand ceiling
        "0.40*Oman + 0.30*Tapis + 0.25*Labuan + 0.35*Murban >= 60000", # diesel demand floor
        "0.40*Oman + 0.30*Tapis + 0.25*Labuan + 0.35*Murban <= 90000", # diesel demand ceiling
        "0.15*Oman + 0.10*Tapis + 0.30*Labuan + 0.15*Murban >= 20000", # fuel oil minimum
    ]
    var_names = ["Oman", "Tapis", "Labuan", "Murban"]
    bounds_dict = {v: (0, None) for v in var_names}

    if st.button("Solve Refinery Optimization", type="primary"):
        res = solve_lp(
            objective_str=objective_str,
            constraints_list=constraints_list,
            sense="Maximize",
            var_names=var_names,
            bounds_dict=bounds_dict
        )
        display_results(res, "Maximize")


# ------------------------------------------------------------------------
# PAGE 2: User Custom LP Builder
# ------------------------------------------------------------------------
else:
    st.title("🛠️ Custom Linear Program Builder")
    st.markdown("Build your own linear program dynamically. Variable names are automatically extracted as you type.")

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

    # Dynamic Live Variable Parsing
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
        res = solve_lp(
            objective_str=obj_input,
            constraints_list=constraints_list,
            sense=sense,
            var_names=detected_vars,
            bounds_dict=bounds_dict
        )
        display_results(res, sense)
