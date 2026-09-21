from scipy.optimize import minimize, Bounds, LinearConstraint, linprog
import os
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Define a consistent palette for indexed constraints
CONSTRAINT_COLORS = [
    "#E74C3C", "#3498DB", "#9B59B6", "#F1C40F", "#E67E22",
    "#1ABC9C", "#34495E", "#D35400", "#27AE60", "#8E44AD"
]

# ----------------------------------------------------------------------
# AI PARSER (GOOGLE GEMINI) - QUADRATIC (QP)
# ----------------------------------------------------------------------
class QPProblemSchema(BaseModel):
    sense: str = Field(description="Optimization sense: 'Maximize' or 'Minimize'")
    objective_function: str = Field(description="Algebraic objective expression without 'Maximize'/'Minimize' prefix. May include quadratic terms such as 'x**2', 'x*y', e.g., '2*x**2 + 3*x*y - y'")
    constraints: list[str] = Field(description="List of LINEAR constraint equations only, using <=, >=, or =, e.g., ['x + y <= 10', 'x >= 0']. Quadratic terms are NOT allowed in constraints.")


def parse_qp_with_gemini(user_prompt: str, api_key: str = None) -> QPProblemSchema:
    resolved_api_key = api_key or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not resolved_api_key:
        raise ValueError("Google API Key not found. Please add GOOGLE_API_KEY to Streamlit Secrets.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=resolved_api_key
    )

    structured_llm = llm.with_structured_output(QPProblemSchema)

    system_prompt = (
        "You are an expert operations research assistant specializing in Quadratic Programming (QP). "
        "Parse the user's natural language problem. Extract decision variables and formulate the algebraic "
        "objective function, which MAY contain quadratic terms (e.g., x**2, x*y, 2*y**2). "
        "Constraints MUST remain strictly LINEAR (no quadratic or cross-product terms in constraints). "
        "Do NOT include unit labels or currency signs in algebraic terms. Standardize variable names using "
        "standard Python identifier names (e.g., x, y, qty1, qty2)."
    )

    return structured_llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])


# ----------------------------------------------------------------------
# QP-SPECIFIC VALIDATION
# ----------------------------------------------------------------------
def check_expression_quadratic(expr_str: str, var_names: list) -> tuple[bool, str]:
    if not expr_str.strip():
        return True, ""

    try:
        sym_dict = {v: sp.Symbol(v) for v in var_names}
        parsed_expr = sp.sympify(expr_str, locals=sym_dict)
    except Exception as e:
        return False, f"Syntax / parsing error: {e}"

    free_symbols = parsed_expr.free_symbols
    if not free_symbols:
        return True, ""

    try:
        ordered_syms = sorted(free_symbols, key=str)
        poly = sp.Poly(parsed_expr, *ordered_syms)
        deg = poly.total_degree()
        if deg > 2:
            return False, (f"Total degree {deg} detected — exceeds the maximum degree of 2 supported by "
                           f"the Quadratic Programming (QP) solver. Only linear and quadratic terms are allowed.")
    except sp.PolynomialError:
        return False, "Non-polynomial or transcendental term detected."

    return True, ""


# ----------------------------------------------------------------------
# QP ENGINE - matrix extraction & solving
# ----------------------------------------------------------------------
def build_quadratic_matrices(obj_expr, sym_vars):
    n = len(sym_vars)
    zero_subs = {v: 0 for v in sym_vars}

    H = sp.hessian(obj_expr, sym_vars)
    Q = np.array(H.evalf()).astype(float).reshape(n, n)

    grad = [sp.diff(obj_expr, v).subs(zero_subs) for v in sym_vars]
    c_vec = np.array([float(g) for g in grad])

    const = float(obj_expr.subs(zero_subs))

    return Q, c_vec, const


def _qp_find_feasible_start(n, A_ub, b_ub, A_eq, b_eq, bounds_list):
    try:
        res0 = linprog(
            c=np.zeros(n),
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None,
            b_eq=b_eq if b_eq else None,
            bounds=bounds_list,
            method="highs"
        )
        if res0.success:
            return res0.x
    except Exception:
        pass

    x0 = []
    for lo, hi in bounds_list:
        lo_v = lo if lo is not None else -1.0
        hi_v = hi if hi is not None else (lo_v + 2.0 if lo is not None else 1.0)
        x0.append((lo_v + hi_v) / 2.0)
    return np.array(x0)


def solve_qp(objective_str: str, constraints_list: list, sense: str, var_names: list, bounds_dict: dict):
    if not var_names:
        return {"success": False, "message": "No variables defined."}

    var_names = sorted(list(var_names))
    sym_vars = [sp.Symbol(v) for v in var_names]
    local_dict = {v: sym_vars[i] for i, v in enumerate(var_names)}
    n = len(var_names)

    is_quad, quad_msg = check_expression_quadratic(objective_str, var_names)
    if not is_quad:
        return {"success": False, "message": f"Unsupported objective function: {quad_msg}"}

    try:
        obj_expr = sp.sympify(objective_str, locals=local_dict)
    except Exception as e:
        return {"success": False, "message": f"Error parsing objective function: {e}"}

    Q_orig, c_orig, const_orig = build_quadratic_matrices(obj_expr, sym_vars)

    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    
    # Bound-mapped constraints list with indexed metadata
    bound_constraints = []

    for idx, constr in enumerate(constraints_list, start=1):
        if not constr.strip():
            continue
        try:
            diff, rel = parse_equation_or_inequality(constr, local_dict)
        except Exception as e:
            return {"success": False, "message": f"Error parsing constraint '{constr}': {e}"}

        is_lin_c, lin_c_msg = check_expression_linearity(str(diff), var_names)
        if not is_lin_c:
            return {"success": False, "message": f"Constraint '{constr}' is non-linear: {lin_c_msg}"}

        const_term = float(diff.as_coefficients_dict().get(1, 0))
        coeffs = [float(diff.coeff(v)) for v in sym_vars]
        color = CONSTRAINT_COLORS[(idx - 1) % len(CONSTRAINT_COLORS)]

        if rel == "<=":
            A_ub.append(coeffs)
            b_ub.append(-const_term)
            bound_constraints.append({
                "index": idx,
                "label": f"C{idx}: {constr}",
                "coeffs": coeffs,
                "b": -const_term,
                "rel": "<=",
                "color": color
            })
        elif rel == ">=":
            A_ub.append([-v for v in coeffs])
            b_ub.append(const_term)
            bound_constraints.append({
                "index": idx,
                "label": f"C{idx}: {constr}",
                "coeffs": [-v for v in coeffs],
                "b": const_term,
                "rel": ">=",
                "color": color
            })
        elif rel == "==":
            A_eq.append(coeffs)
            b_eq.append(-const_term)
            bound_constraints.append({
                "index": idx,
                "label": f"C{idx}: {constr}",
                "coeffs": coeffs,
                "b": -const_term,
                "rel": "==",
                "color": color
            })

    bounds_list = [bounds_dict.get(v, (0, None)) for v in var_names]
    scipy_bounds = Bounds(
        [(-np.inf if b[0] is None else b[0]) for b in bounds_list],
        [(np.inf if b[1] is None else b[1]) for b in bounds_list]
    )

    sense_sign = -1.0 if sense.lower() == "maximize" else 1.0
    Q_used = sense_sign * Q_orig
    c_used = sense_sign * c_orig
    Q_sym = (Q_used + Q_used.T) / 2.0

    def objective_func(x):
        return 0.5 * x @ Q_sym @ x + c_used @ x

    def objective_grad(x):
        return Q_sym @ x + c_used

    def objective_hess(x):
        return Q_sym

    eigvals = np.linalg.eigvalsh(Q_sym)
    is_convex = bool(np.all(eigvals >= -1e-7))

    lin_constraints = []
    if A_ub:
        lin_constraints.append(LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub)))
    if A_eq:
        lin_constraints.append(LinearConstraint(np.array(A_eq), np.array(b_eq), np.array(b_eq)))

    x0 = _qp_find_feasible_start(n, A_ub, b_ub, A_eq, b_eq, bounds_list)

    try:
        res = minimize(
            objective_func,
            x0,
            jac=objective_grad,
            hess=objective_hess,
            method="trust-constr",
            bounds=scipy_bounds,
            constraints=lin_constraints if lin_constraints else None,
            options={"maxiter": 1000, "gtol": 1e-10, "xtol": 1e-12}
        )
    except Exception as e:
        return {"success": False, "message": f"Solver exception: {e}"}

    if not res.success and res.status not in (1, 2):
        return {"success": False, "message": f"Solver failed: {res.message}"}

    opt_val = sense_sign * res.fun + const_orig
    solution = dict(zip(var_names, res.x))

    return {
        "success": True,
        "fun": opt_val,
        "x": solution,
        "message": res.message,
        "status": res.status,
        "Q": Q_orig,
        "c": c_orig,
        "obj_const": const_orig,
        "obj_expr": obj_expr,
        "sym_vars": sym_vars,
        "is_convex": is_convex,
        "eigvals": eigvals,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "bounds": bounds_list,
        "var_names": var_names,
        "raw_constraints": constraints_list,
        "bound_constraints": bound_constraints  # <-- Explicitly passed to visualizer
    }


# ----------------------------------------------------------------------
# QP PLOTTING (Strict Constraint Color Matching)
# ----------------------------------------------------------------------
def plot_interactive_contour_lines_qp(result: dict, default_x: str = None, default_y: str = None):
    var_names = result["var_names"]

    if len(var_names) < 2:
        st.info("Interactive contour line plots require at least 2 decision variables.")
        return

    default_x_idx = var_names.index(default_x) if default_x in var_names else 0

    st.markdown("**2D Projection Settings**")
    col_x, col_y, col_n = st.columns(3)

    with col_x:
        x_name = st.selectbox("X-Axis Variable", var_names, index=default_x_idx, key="contour_x_var_qp")

    y_options = [v for v in var_names if v != x_name]
    default_y_idx = y_options.index(default_y) if default_y in y_options else 0

    with col_y:
        y_name = st.selectbox("Y-Axis Variable", y_options, index=default_y_idx, key="contour_y_var_qp")

    with col_n:
        n_contours = st.number_input(
            "Objective Contours (N)",
            min_value=5, max_value=300, value=120, step=5,
            key="n_contours_input_qp"
        )

    allow_negative = st.checkbox("Allow Negative Axes Ranges", value=False, key="allow_neg_axes_qp")

    x_idx = var_names.index(x_name)
    y_idx = var_names.index(y_name)

    opt_x = result["x"][x_name]
    opt_y = result["x"][y_name]

    view_x_max = max(abs(opt_x) * 1.5, 2.0)
    view_y_max = max(abs(opt_y) * 1.5, 2.0)
    calc_x_max = view_x_max * 10.0
    calc_y_max = view_y_max * 10.0

    fixed_vars_summary = [f"{v} = {result['x'][v]:,.4f}" for idx, v in enumerate(var_names) if idx not in (x_idx, y_idx)]
    if fixed_vars_summary:
        st.caption(f"ℹ️ Other variables held constant at optimal values: **{', '.join(fixed_vars_summary)}**")

    obj_expr = result["obj_expr"]
    sym_vars = result["sym_vars"]
    subs_dict = {sv: result["x"][vn] for vn, sv in zip(var_names, sym_vars) if vn not in (x_name, y_name)}
    expr_2d = obj_expr.subs(subs_dict)
    z_func = sp.lambdify((sp.Symbol(x_name), sp.Symbol(y_name)), expr_2d, "numpy")

    # Reconstruct halfplanes for polygon calculation
    halfplanes = []
    bound_constraints = result.get("bound_constraints", [])
    for c_info in bound_constraints:
        a = c_info["coeffs"]
        eff_b = c_info["b"]
        for v_i in range(len(var_names)):
            if v_i not in (x_idx, y_idx):
                eff_b -= a[v_i] * result["x"][var_names[v_i]]
        halfplanes.append((a[x_idx], a[y_idx], eff_b))

    bounds = result.get("bounds", [])
    x_min_b, x_max_b = bounds[x_idx] if x_idx < len(bounds) else (0, None)
    y_min_b, y_max_b = bounds[y_idx] if y_idx < len(bounds) else (0, None)

    b_x_min = (-calc_x_max if allow_negative else 0.0) if x_min_b is None else x_min_b
    b_x_max = calc_x_max if x_max_b is None else min(x_max_b, calc_x_max)
    b_y_min = (-calc_y_max if allow_negative else 0.0) if y_min_b is None else y_min_b
    b_y_max = calc_y_max if y_max_b is None else min(y_max_b, calc_y_max)

    fig = go.Figure()

    poly_x, poly_y = compute_feasible_polygon_vertices(
        halfplanes, bounds_x=(b_x_min, b_x_max), bounds_y=(b_y_min, b_y_max)
    )

    if poly_x is not None and len(poly_x) > 0:
        fig.add_trace(go.Scatter(
            x=np.append(poly_x, poly_x[0]), y=np.append(poly_y, poly_y[0]),
            fill="toself", fillcolor="rgba(46, 204, 113, 0.25)",
            line=dict(color="rgba(46, 204, 113, 0.6)", width=1),
            name="Feasible Region", hoverinfo="skip"
        ))

    x_min_calc = -calc_x_max if allow_negative else 0
    y_min_calc = -calc_y_max if allow_negative else 0
    x_vals = np.linspace(x_min_calc, calc_x_max, 250)
    y_vals = np.linspace(y_min_calc, calc_y_max, 250)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.asarray(z_func(X, Y), dtype=float)

    contour_line_color = "#1f77b4"
    fig.add_trace(go.Contour(
        x=x_vals, y=y_vals, z=Z, contours_coloring="lines",
        ncontours=int(n_contours),
        contours=dict(showlabels=True, labelfont=dict(size=10, color="navy")),
        line=dict(color=contour_line_color, width=1.5, dash="dash"),
        showscale=False, showlegend=False, hoverinfo="x+y+z"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=contour_line_color, width=1.5, dash="dash"),
        name="Objective Contour", showlegend=True
    ))

    # Plot each constraint strictly matching its mapped index & color
    for c_info in bound_constraints:
        a = c_info["coeffs"]
        eff_b = c_info["b"]
        for v_i in range(len(var_names)):
            if v_i not in (x_idx, y_idx):
                eff_b -= a[v_i] * result["x"][var_names[v_i]]

        a_x, a_y = a[x_idx], a[y_idx]
        line_color = c_info["color"]
        label = c_info["label"]

        if abs(a_y) > 1e-6:
            y_line = (eff_b - a_x * x_vals) / a_y
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_line, mode="lines",
                line=dict(color=line_color, width=2),
                name=label, hoverinfo="x+y"
            ))
        elif abs(a_x) > 1e-6:
            x_val = eff_b / a_x
            fig.add_trace(go.Scatter(
                x=[x_val, x_val], y=[y_min_calc, calc_y_max], mode="lines",
                line=dict(color=line_color, width=2),
                name=label, hoverinfo="x+y"
            ))

    fig.add_trace(go.Scatter(
        x=[opt_x], y=[opt_y], mode="markers+text",
        marker=dict(color="#d62728", size=12, symbol="circle", line=dict(color="black", width=1)),
        text=[f" Optimal ({opt_x:.2f}, {opt_y:.2f})"],
        textposition="top right", name="Optimal Solution", hoverinfo="x+y"
    ))

    x_min_view = None if allow_negative else 0
    y_min_view = None if allow_negative else 0

    fig.update_layout(
        title="",
        xaxis=dict(title=x_name, range=[x_min_view, view_x_max], showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        yaxis=dict(title=y_name, range=[y_min_view, view_y_max], showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=20, b=120),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(200,200,200,0.6)", borderwidth=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
