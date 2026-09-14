import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import linprog

# Color palette for constraints
CONSTRAINT_COLORS = [
    "#e74c3c", "#9b59b6", "#3498db", "#f39c12", 
    "#1abc9c", "#d35400", "#2c3e50", "#8e44ad"
]


def parse_equation_or_inequality(constr_str: str, local_dict: dict):
    """
    Parses a constraint string into a SymPy expression (LHS - RHS)
    and extracts the relational operator (<=, >=, ==).
    """
    constr_str = constr_str.strip()
    if "<=" in constr_str:
        lhs, rhs = constr_str.split("<=")
        rel = "<="
    elif ">=" in constr_str:
        lhs, rhs = constr_str.split(">=")
        rel = ">="
    elif "=" in constr_str:
        lhs, rhs = constr_str.split("=")
        rel = "=="
    elif "==" in constr_str:
        lhs, rhs = constr_str.split("==")
        rel = "=="
    else:
        raise ValueError(f"Constraint '{constr_str}' lacks a valid operator (<=, >=, =).")

    lhs_expr = sp.sympify(lhs, locals=local_dict)
    rhs_expr = sp.sympify(rhs, locals=local_dict)
    diff = lhs_expr - rhs_expr
    return diff, rel


def solve_lp(objective_str: str, constraints_list: list, sense: str, var_names: list, bounds_dict: dict):
    """
    Converts mathematical expressions into standard form matrices for scipy.optimize.linprog.
    Always returns structural matrix data to enable diagnostic plotting even on failure.
    """
    if not var_names:
        return {"success": False, "message": "No decision variables specified."}

    var_names = sorted(list(var_names))
    sym_vars = [sp.Symbol(v) for v in var_names]
    local_dict = {v: sym_vars[i] for i, v in enumerate(var_names)}

    # Parse Objective Function
    try:
        obj_expr = sp.sympify(objective_str, locals=local_dict)
    except Exception as e:
        return {"success": False, "message": f"Error parsing objective function: {e}"}

    c = []
    for var in sym_vars:
        coeff = obj_expr.coeff(var)
        if any(v in coeff.free_symbols for v in sym_vars):
            return {"success": False, "message": f"Non-linear term detected for variable '{var}'."}
        c.append(float(coeff))

    obj_const = float(obj_expr.as_coefficients_dict().get(1, 0))
    c_raw = list(c)

    # scipy.optimize.linprog minimizes by default
    if sense.lower() == "maximize":
        c = [-val for val in c]

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    # Parse Constraints
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

    # Run LP Solver
    res = linprog(
        c=c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method="highs"
    )

    # Build response structure
    solution = dict(zip(var_names, res.x)) if res.x is not None else {v: 0.0 for v in var_names}
    opt_val = ((-res.fun if sense.lower() == "maximize" else res.fun) + obj_const) if res.fun is not None else None

    return {
        "success": res.success,
        "fun": opt_val,
        "x": solution,
        "message": res.message,
        "status": res.status,
        "c": c_raw,
        "obj_const": obj_const,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds,
        "var_names": var_names,
        "raw_constraints": [c for c in constraints_list if c.strip()],
    }


def plot_interactive_contour_lines(result: dict, default_x: str = None, default_y: str = None):
    """
    Generates an interactive 2D objective contour plot with constraint boundaries 
    and shaded feasible region (handles both successful and failed LP solves).
    """
    var_names = result["var_names"]
    
    if len(var_names) < 2:
        st.info("Interactive contour line plots require at least 2 decision variables.")
        return

    default_x_idx = var_names.index(default_x) if default_x in var_names else 0
    
    st.markdown("**2D Projection Controls**")
    col_x, col_y = st.columns(2)
    
    with col_x:
        x_name = st.selectbox("X-Axis Variable", var_names, index=default_x_idx, key="contour_x_var")
    
    y_options = [v for v in var_names if v != x_name]
    default_y_idx = y_options.index(default_y) if default_y in y_options else 0
    
    with col_y:
        y_name = st.selectbox("Y-Axis Variable", y_options, index=default_y_idx, key="contour_y_var")

    x_idx = var_names.index(x_name)
    y_idx = var_names.index(y_name)

    opt_x = result["x"].get(x_name, 0.0)
    opt_y = result["x"].get(y_name, 0.0)

    # Calculate reasonable plotting bounds
    x_max = max(opt_x * 1.5, 100.0) if result["success"] else 120.0
    y_max = max(opt_y * 1.5, 100.0) if result["success"] else 120.0

    x_vals = np.linspace(0, x_max, 250)
    y_vals = np.linspace(0, y_max, 250)
    X, Y = np.meshgrid(x_vals, y_vals)

    fixed_objective_contrib = result.get("obj_const", 0.0)
    c = result["c"]

    fixed_vars_summary = []
    for idx, v_name in enumerate(var_names):
        if idx not in (x_idx, y_idx):
            val = result["x"].get(v_name, 0.0)
            fixed_objective_contrib += c[idx] * val
            fixed_vars_summary.append(f"{v_name} = {val:,.2f}")

    if fixed_vars_summary:
        st.caption(f"ℹ️ Other variables held constant at: **{', '.join(fixed_vars_summary)}**")

    c_x = c[x_idx]
    c_y = c[y_idx]
    Z = c_x * X + c_y * Y + fixed_objective_contrib

    fig = go.Figure()

    # 1. SHADE FEASIBLE REGION
    feasible_mask = np.ones_like(X, dtype=bool)

    A_ub = result.get("A_ub", [])
    b_ub = result.get("b_ub", [])
    if A_ub and b_ub:
        for a, b in zip(A_ub, b_ub):
            eff_b = b
            for v_i in range(len(var_names)):
                if v_i not in (x_idx, y_idx):
                    eff_b -= a[v_i] * result["x"].get(var_names[v_i], 0.0)
            lhs_val = a[x_idx] * X + a[y_idx] * Y
            feasible_mask = feasible_mask & (lhs_val <= eff_b + 1e-5)

    bounds = result.get("bounds", [])
    x_min_b, x_max_b = bounds[x_idx] if x_idx < len(bounds) else (0, None)
    y_min_b, y_max_b = bounds[y_idx] if y_idx < len(bounds) else (0, None)

    if x_min_b is not None:
        feasible_mask &= (X >= x_min_b - 1e-5)
    if x_max_b is not None:
        feasible_mask &= (X <= x_max_b + 1e-5)
    if y_min_b is not None:
        feasible_mask &= (Y >= y_min_b - 1e-5)
    if y_max_b is not None:
        feasible_mask &= (Y <= y_max_b + 1e-5)

    feasible_z = feasible_mask.astype(float)
    feasible_z[~feasible_mask] = np.nan

    fig.add_trace(
        go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=feasible_z,
            showscale=False,
            colorscale=[[0, "rgba(46, 204, 113, 0.25)"], [1, "rgba(46, 204, 113, 0.25)"]],
            hoverinfo="skip",
            name="Feasible Region"
        )
    )

    # 2. CONTOUR LINES & CONSTRAINTS
    fig.add_trace(
        go.Contour(
            x=x_vals,
            y=y_vals,
            z=Z,
            contours_coloring="lines",
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='navy')
            ),
            line=dict(color='#1f77b4', width=1.5, dash='dash'),
            name="Objective Contour",
            hoverinfo="x+y+z"
        )
    )

    raw_constraints = result.get("raw_constraints", [])
    if A_ub and b_ub:
        for idx, (a, b) in enumerate(zip(A_ub, b_ub)):
            eff_b = b
            for v_i in range(len(var_names)):
                if v_i not in (x_idx, y_idx):
                    eff_b -= a[v_i] * result["x"].get(var_names[v_i], 0.0)

            a_x, a_y = a[x_idx], a[y_idx]
            line_color = CONSTRAINT_COLORS[idx % len(CONSTRAINT_COLORS)]
            constr_label = raw_constraints[idx] if idx < len(raw_constraints) else f"Constraint {idx+1}"

            if abs(a_y) > 1e-6:
                y_line = (eff_b - a_x * x_vals) / a_y
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_line,
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        name=f"C{idx+1}: {constr_label}",
                        hoverinfo="x+y"
                    )
                )
            elif abs(a_x) > 1e-6:
                x_val = eff_b / a_x
                fig.add_trace(
                    go.Scatter(
                        x=[x_val, x_val],
                        y=[0, y_max],
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        name=f"C{idx+1}: {constr_label}",
                        hoverinfo="x+y"
                    )
                )

    # 3. OPTIMAL POINT MARKER (Only displayed if LP solve succeeded)
    if result["success"]:
        fig.add_trace(
            go.Scatter(
                x=[opt_x],
                y=[opt_y],
                mode='markers+text',
                marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='black', width=1)),
                text=[f" Optimal ({opt_x:.2f}, {opt_y:.2f})"],
                textposition="top right",
                name="Optimal Solution",
                hoverinfo="x+y"
            )
        )

    plot_title = "Diagnostic Contour Map (Infeasible or Unbounded LP)" if not result["success"] else "Interactive LP Solution & Contour Map"

    fig.update_layout(
        title=plot_title,
        xaxis=dict(title=x_name, range=[0, x_max], showgrid=True, gridcolor='rgba(200,200,200,0.4)'),
        yaxis=dict(title=y_name, range=[0, y_max], showgrid=True, gridcolor='rgba(200,200,200,0.4)'),
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=50, b=120),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(200,200,200,0.6)",
            borderwidth=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_results(result: dict, sense: str, default_x: str = None, default_y: str = None):
    """Renders solution metrics, data tables, and diagnostic/contour visualizations."""
    if result["success"]:
        st.success("Optimization Completed Successfully!")

        col1, _ = st.columns(2)
        with col1:
            st.metric(label=f"Optimal Value ({sense.upper()})", value=f"{result['fun']:,.4f}")

        st.subheader("Optimal Variable Assignments")
        df_res = pd.DataFrame(
            list(result["x"].items()),
            columns=["Variable", "Optimal Value"]
        )
        st.dataframe(df_res.style.format({"Optimal Value": "{:,.4f}"}), use_container_width=True)

        st.subheader("Contour & Feasible Region Map")
        plot_interactive_contour_lines(result, default_x=default_x, default_y=default_y)
    else:
        st.error(f"Solver Failure: {result['message']}")
        st.warning("⚠️ Rendering diagnostic plot below to help inspect constraint conflicts or unbounded directions.")
        plot_interactive_contour_lines(result, default_x=default_x, default_y=default_y)


# Streamlit Application Layout
def main():
    st.set_page_config(page_title="Linear Programming Solver & Visualizer", layout="wide")
    st.title("Linear Programming Solver & Contour Visualizer")

    st.sidebar.header("Problem Formulation")
    sense = st.sidebar.radio("Optimization Goal", ["Maximize", "Minimize"])
    
    obj_input = st.sidebar.text_input("Objective Function", value="3*x1 + 5*x2")
    
    constraints_input = st.sidebar.text_area(
        "Constraints (one per line)",
        value="2*x1 + 3*x2 <= 120\nx1 + 2*x2 <= 70\nx1 >= 0\nx2 >= 0"
    )

    if st.sidebar.button("Solve & Plot", type="primary"):
        constr_list = [c.strip() for c in constraints_input.split("\n") if c.strip()]
        
        # Auto-detect variables
        vars_found = set()
        for constr in constr_list + [obj_input]:
            # Simple word-token extraction for symbols
            for token in constr.replace("<=", " ").replace(">=", " ").replace("=", " ").replace("+", " ").replace("-", " ").replace("*", " ").split():
                if token.isidentifier() and not token.isdigit():
                    vars_found.add(token)
                    
        bounds_dict = {v: (0, None) for v in vars_found}
        
        res = solve_lp(
            objective_str=obj_input,
            constraints_list=constr_list,
            sense=sense,
            var_names=list(vars_found),
            bounds_dict=bounds_dict
        )

        display_results(res, sense)


if __name__ == "__main__":
    main()
