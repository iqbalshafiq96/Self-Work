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

# Reserved words/functions that SymPy or Python use that shouldn't be treated as user variables
RESERVED_WORDS = {
    "Min", "Max", "min", "max", "Abs", "abs", "sin", "cos", "tan", 
    "exp", "log", "sqrt", "True", "False", "None", "and", "or", "not"
}

# ========================================================================
# LP ENGINE — parsing, linearity check, solving
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
# STREAMLIT UI — Navigation and Page Routing
# ========================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Page 1: Worked Refinery Example", "Page 2: Custom LP Builder"])


# ------------------------------------------------------------------------
# PAGE 1: Worked Refinery Example
# ------------------------------------------------------------------------
if page == "Page 1: Worked Refinery Example":
    st.title("🛢️ Refinery Crude Oil Purchasing LP")
    st.markdown("""
    This worked example optimizes crude oil purchasing decisions for a refinery to maximize total profit while meeting production requirements and operational constraints.
    """)

    st.subheader("Problem Definition")
    st.latex(r"\text{Maximize Profit} = 15 \cdot \text{CrudeA} + 20 \cdot \text{CrudeB}")
    
    st.markdown("**Subject to Constraints:**")
    st.latex(r"0.4 \cdot \text{CrudeA} + 0.3 \cdot \text{CrudeB} \ge 120 \quad \text{(Gasoline yield requirement)}")
    st.latex(r"0.2 \cdot \text{CrudeA} + 0.4 \cdot \text{CrudeB} \ge 80 \quad \text{(Heating Oil yield requirement)}")
    st.latex(r"\text{CrudeA} + \text{CrudeB} \le 500 \quad \text{(Refinery processing capacity)}")

    objective_str = "15 * CrudeA + 20 * CrudeB"
    constraints_list = [
        "0.4 * CrudeA + 0.3 * CrudeB >= 120",
        "0.2 * CrudeA + 0.4 * CrudeB >= 80",
        "CrudeA + CrudeB <= 500"
    ]
    var_names = ["CrudeA", "CrudeB"]
    bounds_dict = {"CrudeA": (0, None), "CrudeB": (0, None)}

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
