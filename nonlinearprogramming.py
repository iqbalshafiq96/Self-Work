import os
import re
import sympy as sp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize, Bounds, LinearConstraint, linprog
from scipy.spatial import ConvexHull
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------------------------------------------------
# COLOR PALETTES & CONSTANTS
# ----------------------------------------------------------------------
CONSTRAINT_COLORS = [
    '#e74c3c',  # Red
    '#3498db',  # Blue
    '#9b59b6',  # Purple
    '#f39c12',  # Orange
    '#1abc9c',  # Teal
    '#d35400',  # Rust
    '#2c3e50',  # Dark Slate
    '#8e44ad',  # Deep Purple
    '#27ae60',  # Green
    '#16a085',  # Dark Teal
]

VAR_BADGE_COLORS = [
    {"bg": "#EBF5FB", "text": "#1B4F72", "border": "#A9CCE3"},
    {"bg": "#E8F8F5", "text": "#0E6251", "border": "#A3E4D7"},
    {"bg": "#FEF9E7", "text": "#7D6608", "border": "#F9E79F"},
    {"bg": "#F4ECF7", "text": "#512E5F", "border": "#D2B4DE"},
    {"bg": "#FBEEE6", "text": "#6E2C00", "border": "#EDBB99"},
]

# Standard reserved math/programming words to exclude from variable parsing
RESERVED_WORDS = {
    "max", "min", "maximize", "minimize", "subject", "to", "st", "s.t.",
    "and", "or", "not", "sin", "cos", "tan", "log", "exp", "sqrt", "abs"
}


# ----------------------------------------------------------------------
# PARSING & GEOMETRY HELPERS
# ----------------------------------------------------------------------
def extract_identifiers(text: str) -> set:
    """Extracts valid Pythonic variable names, excluding numbers and math keywords."""
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
    return {t for t in tokens if t.lower() not in RESERVED_WORDS}


def highlight_variables_in_text(text: str, var_names: list) -> str:
    """Wraps detected variables in HTML styled tags for preview rendering."""
    if not var_names:
        return text

    sorted_vars = sorted(var_names, key=len, reverse=True)
    pattern = r'\b(' + '|'.join(map(re.escape, sorted_vars)) + r')\b'

    def replacer(match):
        v = match.group(0)
        idx = var_names.index(v) if v in var_names else 0
        color = VAR_BADGE_COLORS[idx % len(VAR_BADGE_COLORS)]
        return (f'<span style="background-color: {color["bg"]}; color: {color["text"]}; '
                f'border: 1px solid {color["border"]}; padding: 1px 5px; border-radius: 4px; '
                f'font-weight: 600;">{v}</span>')

    return re.sub(pattern, replacer, text)


def parse_equation_or_inequality(line_str: str, local_dict: dict):
    """Parses a string inequality/equation into a SymPy expression (LHS - RHS) and relational sign."""
    if "<=" in line_str:
        parts = line_str.split("<=")
        rel = "<="
    elif ">=" in line_str:
        parts = line_str.split(">=")
        rel = ">="
    elif "=" in line_str:
        parts = line_str.split("=")
        rel = "=="
    else:
        raise ValueError("Missing valid relational operator (<=, >=, or =).")

    lhs = sp.sympify(parts[0], locals=local_dict)
    rhs = sp.sympify(parts[1], locals=local_dict)
    return lhs - rhs, rel


def check_expression_linearity(expr_str: str, var_names: list) -> tuple[bool, str]:
    """Ensures an expression has total polynomial degree <= 1 across all variables."""
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
        if deg > 1:
            return False, f"Degree {deg} polynomial term detected. Constraints must be strictly linear."
    except sp.PolynomialError:
        return False, "Non-polynomial term detected in constraint expression."

    return True, ""


def compute_feasible_polygon_vertices(halfplanes: list, bounds_x: tuple, bounds_y: tuple):
    """
    Computes 2D feasible polygon vertices formed by linear halfplanes A_x*x + A_y*y <= b
    and axis bounding box constraints via half-space intersections.
    """
    all_planes = list(halfplanes)

    if bounds_x[0] is not None:
        all_planes.append((-1.0, 0.0, -bounds_x[0]))
    if bounds_x[1] is not None:
        all_planes.append((1.0, 0.0, bounds_x[1]))
    if bounds_y[0] is not None:
        all_planes.append((0.0, -1.0, -bounds_y[0]))
    if bounds_y[1] is not None:
        all_planes.append((0.0, 1.0, bounds_y[1]))

    intersections = []
    n_planes = len(all_planes)

    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            a1, b1, c1 = all_planes[i]
            a2, b2, c2 = all_planes[j]

            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-9:
                continue

            x_int = (c1 * b2 - c2 * b1) / det
            y_int = (a1 * c2 - a2 * c1) / det

            feasible = True
            for a_k, b_k, c_k in all_planes:
                if a_k * x_int + b_k * y_int > c_k + 1e-7:
                    feasible = False
                    break

            if feasible:
                intersections.append((x_int, y_int))

    if len(intersections) < 3:
        return None, None

    pts = np.unique(np.round(intersections, decimals=8), axis=0)
    if len(pts) < 3:
        return None, None

    try:
        hull = ConvexHull(pts)
        sorted_pts = pts[hull.vertices]
        return sorted_pts[:, 0], sorted_pts[:, 1]
    except Exception:
        return None, None


# ----------------------------------------------------------------------
# AI PARSER (GOOGLE GEMINI) - QUADRATIC (QP)
# ----------------------------------------------------------------------
class QPProblemSchema(BaseModel):
    sense: str = Field(description="Optimization sense: 'Maximize' or 'Minimize'")
    objective_function: str = Field(description="Algebraic objective expression without 'Maximize'/'Minimize' prefix. May include quadratic terms such as 'x**2', 'x*y', e.g., '2*x**2 + 3*x*y - y'")
    constraints: list[str] = Field(description="List of LINEAR constraint equations only, using <=, >=, or =, e.g., ['x + y <= 10', 'x >= 0']. Quadratic terms are NOT allowed in constraints.")


def parse_qp_with_gemini(user_prompt: str, api_key: str = None) -> QPProblemSchema:
    """Extracts QP parameters from natural language using Google AI Studio Gemini API."""
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
    """
    Analyzes an expression using SymPy and allows terms up to TOTAL DEGREE 2
    (linear terms, pure quadratic terms like x**2, and bilinear cross terms like x*y).
    Rejects degree > 2 polynomial terms and any non-polynomial/transcendental terms.
    Returns (is_quadratic_or_lower, message).
    """
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
                           f"the Quadratic Programming (QP) solver. Only linear and quadratic (incl. bilinear "
                           f"cross-product) terms are allowed, e.g., x, x**2, x*y.")
    except sp.PolynomialError:
        return False, "Non-polynomial or transcendental term detected (e.g., trig, log, exp, fractional exponent, or division by a variable)."

    return True, ""


# ----------------------------------------------------------------------
# QP ENGINE - matrix extraction & solving
# ----------------------------------------------------------------------
def build_quadratic_matrices(obj_expr, sym_vars):
    """
    Decomposes a (degree <= 2) SymPy expression into:
        expr(x) = 0.5 * x^T Q x + c^T x + const
    using the exact Hessian (Q) and gradient-at-origin (c).
    Returns (Q: np.ndarray, c: np.ndarray, const: float)
    """
    n = len(sym_vars)
    zero_subs = {v: 0 for v in sym_vars}

    H = sp.hessian(obj_expr, sym_vars)
    Q = np.array(H.evalf()).astype(float).reshape(n, n)

    grad = [sp.diff(obj_expr, v).subs(zero_subs) for v in sym_vars]
    c_vec = np.array([float(g) for g in grad])

    const = float(obj_expr.subs(zero_subs))

    return Q, c_vec, const


def _qp_find_feasible_start(n, A_ub, b_ub, A_eq, b_eq, bounds_list):
    """Phase-1 LP (zero objective) to find any feasible point as a warm start for the QP solver."""
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

    # --- Constraints: must remain strictly LINEAR in QP mode ---
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for constr in constraints_list:
        if not constr.strip():
            continue
        try:
            diff, rel = parse_equation_or_inequality(constr, local_dict)
        except Exception as e:
            return {"success": False, "message": f"Error parsing constraint '{constr}': {e}"}

        is_lin_c, lin_c_msg = check_expression_linearity(str(diff), var_names)
        if not is_lin_c:
            return {"success": False,
                    "message": f"Constraint '{constr}' is non-linear ({lin_c_msg}). "
                               f"QP mode supports quadratic terms in the OBJECTIVE only; "
                               f"constraints must remain linear."}

        const_term = float(diff.as_coefficients_dict().get(1, 0))
        coeffs = [float(diff.coeff(v)) for v in sym_vars]

        if rel == "<=":
            A_ub.append(coeffs)
            b_ub.append(-const_term)
        elif rel == ">=":
            A_ub.append([-v for v in coeffs])
            b_ub.append(const_term)
        elif rel == "==":
            A_eq.append(coeffs)
            b_eq.append(-const_term)

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
        A_eq_np = np.array(A_eq)
        b_eq_np = np.array(b_eq)
        lin_constraints.append(LinearConstraint(A_eq_np, b_eq_np, b_eq_np))

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
        return {"success": False, "message": f"Solver raised an exception: {e}"}

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
    }


# ----------------------------------------------------------------------
# QP PLOTTING (SYNCHRONIZED COLOR MAP)
# ----------------------------------------------------------------------
def plot_interactive_contour_lines_qp(result: dict, default_x: str = None, default_y: str = None):
    """
    QP contour + feasible region plot with explicitly synchronized line & legend colors.
    """
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
            min_value=5,
            max_value=300,
            value=120,
            step=5,
            key="n_contours_input_qp",
            help="Higher values increase contour frequency and produce finer intervals."
        )

    allow_negative = st.checkbox(
        "Allow Negative Axes Ranges",
        value=False,
        key="allow_neg_axes_qp",
        help="If unticked (default), axes will strictly lock to 0 as the minimum value when scrolling or panning."
    )

    x_idx = var_names.index(x_name)
    y_idx = var_names.index(y_name)

    opt_x = result["x"][x_name]
    opt_y = result["x"][y_name]

    view_x_max = max(abs(opt_x) * 1.5, 2.0)
    view_y_max = max(abs(opt_y) * 1.5, 2.0)

    calc_x_max = view_x_max * 10.0
    calc_y_max = view_y_max * 10.0

    fixed_vars_summary = []
    for idx, v_name in enumerate(var_names):
        if idx not in (x_idx, y_idx):
            val = result["x"][v_name]
            fixed_vars_summary.append(f"{v_name} = {val:,.4f}")

    if fixed_vars_summary:
        st.caption(f"ℹ️ Other variables held constant at optimal values: **{', '.join(fixed_vars_summary)}**")

    obj_expr = result["obj_expr"]
    sym_vars = result["sym_vars"]
    subs_dict = {
        sv: result["x"][vn]
        for vn, sv in zip(var_names, sym_vars)
        if vn not in (x_name, y_name)
    }
    x_sym = sp.Symbol(x_name)
    y_sym = sp.Symbol(y_name)
    expr_2d = obj_expr.subs(subs_dict)
    z_func = sp.lambdify((x_sym, y_sym), expr_2d, "numpy")

    halfplanes = []
    A_ub = result.get("A_ub", [])
    b_ub = result.get("b_ub", [])

    if A_ub and b_ub:
        for a, b in zip(A_ub, b_ub):
            eff_b = b
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
        px = np.append(poly_x, poly_x[0])
        py = np.append(poly_y, poly_y[0])

        fig.add_trace(
            go.Scatter(
                x=px, y=py,
                fill="toself",
                fillcolor="rgba(46, 204, 113, 0.25)",
                line=dict(color="rgba(46, 204, 113, 0.6)", width=1),
                name="Feasible Region",
                hoverinfo="skip"
            )
        )

    x_min_calc = -calc_x_max if allow_negative else 0
    y_min_calc = -calc_y_max if allow_negative else 0

    x_vals = np.linspace(x_min_calc, calc_x_max, 250)
    y_vals = np.linspace(y_min_calc, calc_y_max, 250)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.asarray(z_func(X, Y), dtype=float)
    if Z.shape != X.shape:
        Z = np.full_like(X, float(Z))

    contour_line_color = '#1f77b4'

    fig.add_trace(
        go.Contour(
            x=x_vals, y=y_vals, z=Z,
            contours_coloring="lines",
            ncontours=int(n_contours),
            contours=dict(showlabels=True, labelfont=dict(size=10, color='navy')),
            line=dict(color=contour_line_color, width=1.5, dash='dash'),
            showscale=False, showlegend=False, hoverinfo="x+y+z"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=contour_line_color, width=1.5, dash='dash'),
            name="Objective Contour", showlegend=True
        )
    )

    raw_constraints = result.get("raw_constraints", [])
    if A_ub and b_ub:
        for idx, (a, b) in enumerate(zip(A_ub, b_ub)):
            eff_b = b
            for v_i in range(len(var_names)):
                if v_i not in (x_idx, y_idx):
                    eff_b -= a[v_i] * result["x"][var_names[v_i]]

            a_x, a_y = a[x_idx], a[y_idx]
            
            line_color = CONSTRAINT_COLORS[idx % len(CONSTRAINT_COLORS)]
            constr_label = raw_constraints[idx] if idx < len(raw_constraints) else f"Constraint {idx+1}"

            if abs(a_y) > 1e-6:
                y_line = (eff_b - a_x * x_vals) / a_y
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_line, 
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        marker=dict(color=line_color),
                        name=f"C{idx+1}: {constr_label}",
                        showlegend=True,
                        hoverinfo="x+y"
                    )
                )
            elif abs(a_x) > 1e-6:
                x_val = eff_b / a_x
                fig.add_trace(
                    go.Scatter(
                        x=[x_val, x_val], y=[y_min_calc, calc_y_max], 
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        marker=dict(color=line_color),
                        name=f"C{idx+1}: {constr_label}",
                        showlegend=True,
                        hoverinfo="x+y"
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=[opt_x], y=[opt_y], mode='markers+text',
            marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='black', width=1)),
            text=[f" Optimal ({opt_x:.2f}, {opt_y:.2f})"],
            textposition="top right", name="Optimal Solution", hoverinfo="x+y"
        )
    )

    x_min_view = None if allow_negative else 0
    y_min_view = None if allow_negative else 0

    xaxis_config = dict(title=x_name, range=[x_min_view, view_x_max], showgrid=True, gridcolor='rgba(200,200,200,0.4)')
    yaxis_config = dict(title=y_name, range=[y_min_view, view_y_max], showgrid=True, gridcolor='rgba(200,200,200,0.4)')

    if not allow_negative:
        xaxis_config.update(dict(rangemode="nonnegative", minallowed=0))
        yaxis_config.update(dict(rangemode="nonnegative", minallowed=0))

    fig.update_layout(
        title="",
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=20, b=120),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(200,200,200,0.6)", borderwidth=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_results_qp(result: dict, sense: str, default_x: str = None, default_y: str = None):
    if result["success"]:
        st.success("Optimization Completed Successfully!")

        if not result.get("is_convex", True):
            eigvals = result.get("eigvals", [])
            min_eig = float(np.min(eigvals)) if len(eigvals) else 0.0
            st.warning(
                f"⚠️ **Non-convex QP detected** (most negative eigenvalue of the effective Hessian = "
                f"{min_eig:,.4f}). The solution below is a **local** optimum found by a gradient-based "
                f"solver (trust-constr) — global optimality is **not** guaranteed for non-convex problems.",
                icon="⚠️"
            )
        else:
            st.caption("✅ Convexity check passed — the Hessian is positive semi-definite, so this solution is the **global** optimum.")

        col1, _ = st.columns(2)
        with col1:
            st.metric(label=f"Optimal Objective Value ({sense})", value=f"{result['fun']:,.4f}")

        st.subheader("Optimal Decision Variable Values")
        df_res = pd.DataFrame(list(result["x"].items()), columns=["Variable", "Optimal Value"])
        st.dataframe(df_res.style.format({"Optimal Value": "{:,.4f}"}), use_container_width=True)

        st.subheader("Interactive Objective Contour Map & Feasible Area")
        plot_interactive_contour_lines_qp(result, default_x=default_x, default_y=default_y)
    else:
        st.error(f"Solver Error: {result['message']}")


# ----------------------------------------------------------------------
# PAGE LAYOUT
# ----------------------------------------------------------------------
def page_quadratic():
    st.header("QP Problem Statement (Objective Function)")
    st.caption(
        "This solver supports **quadratic objective functions** (e.g., `x**2`, `3*x*y`, `-2*y**2`) "
        "combined with **linear constraints only**. Internally it minimizes/maximizes "
        "0.5·xᵀQx + cᵀx via SciPy's `trust-constr` solver using the exact Hessian/gradient."
    )

    with st.expander("✨ Auto-parse problem statement using Google AI Studio (Gemini)", expanded=True):
        natural_prompt_qp = st.text_area(
            "Describe your Quadratic Programming problem in natural language:",
            placeholder=(
                "A petrochemical plant runs three parallel catalytic reactors, R1, R2 and R3, to convert a single feedstock into a product. "
                "The plant must process exactly 150 kmol/h of total feed, split among the three reactors. The operating cost in USD/h of each "
                "reactor is quadratic in its own feed rate: R1 costs 0.02 times the square of its feed rate plus 3 times its feed rate, "
                "R2 costs 0.03 times the square of its feed rate plus 2 times its feed rate, and R3 costs 0.05 times the square of its feed rate "
                "plus 1 times its feed rate. The reactors convert 90%, 80% and 70% of their feed to product respectively, and the plant must produce "
                "at least 125 kmol/h of product in total. Each kmol of feed processed requires 2 MJ of cooling in R1, 3 MJ in R2 and 4 MJ in R3, "
                "and the plant's cooling system can deliver at most 450 MJ/h. Finally, no reactor can receive a negative feed rate, and the "
                "maximum feed rates are 80 kmol/h for R1, 70 kmol/h for R2 and 60 kmol/h for R3. Determine the feed rate to each reactor that minimizes the total operating cost."
            ),
            height=220,
            key="qp_natural_prompt"
        )
        if st.button("🤖 Parse with Gemini", type="secondary", key="qp_parse_btn"):
            if not natural_prompt_qp.strip():
                st.warning("Please enter a natural language problem statement.")
            else:
                with st.spinner("Parsing problem with Gemini..."):
                    try:
                        parsed_qp = parse_qp_with_gemini(natural_prompt_qp)
                        
                        st.session_state["qp_sense"] = parsed_qp.sense
                        st.session_state["qp_obj_input"] = parsed_qp.objective_function
                        st.session_state["qp_constraints_input"] = "\n".join(parsed_qp.constraints)
                        
                        st.success("Successfully parsed problem statement!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to parse via Gemini API: {e}")

    if "qp_sense" not in st.session_state:
        st.session_state["qp_sense"] = "Minimize"
    if "qp_obj_input" not in st.session_state:
        st.session_state["qp_obj_input"] = "0.02*r1**2 + 3*r1 + 0.03*r2**2 + 2*r2 + 0.05*r3**2 + r3"
    if "qp_constraints_input" not in st.session_state:
        st.session_state["qp_constraints_input"] = (
            "r1 + r2 + r3 = 150\n"
            "0.9*r1 + 0.8*r2 + 0.7*r3 >= 125\n"
            "2*r1 + 3*r2 + 4*r3 <= 450\n"
            "r1 >= 0\n"
            "r1 <= 80\n"
            "r2 >= 0\n"
            "r2 <= 70\n"
            "r3 >= 0\n"
            "r3 <= 60"
        )

    st.caption(
        "Use plain, meaningful variable names — e.g. `x`, `y`, `qty1`. "
        "Quadratic/bilinear terms (`x**2`, `x*y`) are allowed **only** in the objective; "
        "constraints must stay linear."
    )

    col_opt_qp, col_sense_qp = st.columns([3, 1])
    with col_sense_qp:
        sense_qp = st.selectbox(
            "Optimization Sense", 
            ["Maximize", "Minimize"], 
            key="qp_sense"
        )
    with col_opt_qp:
        obj_input_qp = st.text_input(
            "Objective Function", 
            key="qp_obj_input"
        )

    all_text_qp = obj_input_qp + "\n" + st.session_state["qp_constraints_input"]
    detected_vars_qp = sorted(list(extract_identifiers(all_text_qp)))
    is_obj_quad, obj_quad_err = check_expression_quadratic(obj_input_qp, detected_vars_qp)

    if not is_obj_quad:
        st.error(
            f"⚠️ **Unsupported Objective Detected:** The QP solver requires degree ≤ 2.\n\n"
            f"**Reason:** {obj_quad_err}\n\n"
            f"*Please reformulate your objective function so every term has total degree ≤ 2 "
            f"(constants, linear terms like `x`, pure quadratic terms like `x**2`, or bilinear terms like `x*y`).*",
            icon="🚨"
        )

    st.subheader("Constraints")
    st.caption("Enter one **linear** constraint per line using `<=`, `>=`, or `=` (no quadratic terms here).")
    constraints_input_qp = st.text_area(
        "Constraints List",
        height=160,
        key="qp_constraints_input"
    )

    all_text_qp = obj_input_qp + "\n" + constraints_input_qp
    detected_vars_qp = sorted(list(extract_identifiers(all_text_qp)))

    non_linear_constraints = []
    for line in [c.strip() for c in constraints_input_qp.split("\n") if c.strip()]:
        try:
            sym_dict = {v: sp.Symbol(v) for v in detected_vars_qp}
            diff, _ = parse_equation_or_inequality(line, sym_dict)
            if diff is not None:
                is_lin_line, lin_err = check_expression_linearity(str(diff), detected_vars_qp)
                if not is_lin_line:
                    non_linear_constraints.append((line, lin_err))
        except Exception:
            pass

    if non_linear_constraints:
        msgs = "\n".join([f"- `{ln}` → {err}" for ln, err in non_linear_constraints])
        st.error(
            f"⚠️ **Non-Linear Constraint(s) Detected:** QP mode only allows quadratic terms in the "
            f"**objective**; all constraints must be linear.\n\n{msgs}",
            icon="🚨"
        )

    enable_bounds_qp = st.checkbox("Enable Custom Variable Bounds", value=False, key="qp_enable_bounds")
    bounds_dict_qp = {}

    if enable_bounds_qp:
        st.markdown("**Configure Bounds**")
        if detected_vars_qp:
            cols = st.columns(min(len(detected_vars_qp), 4))
            for i, var in enumerate(detected_vars_qp):
                with cols[i % 4]:
                    st.write(f"**{var}**")
                    min_val = st.number_input(f"Min ({var})", value=0.0, key=f"qp_min_{var}")
                    has_max = st.checkbox(f"Set Max ({var})", key=f"qp_has_max_{var}")
                    max_val = st.number_input(f"Max ({var})", value=100.0, key=f"qp_max_{var}") if has_max else None
                    bounds_dict_qp[var] = (min_val, max_val)
        else:
            st.warning("No variables detected yet to configure bounds.")
    else:
        bounds_dict_qp = {var: (0.0, None) for var in detected_vars_qp}

    st.divider()

    if detected_vars_qp:
        badge_spans = []
        for i, var in enumerate(detected_vars_qp):
            color = VAR_BADGE_COLORS[i % len(VAR_BADGE_COLORS)]
            style = (
                f"background-color: {color['bg']}; color: {color['text']}; "
                f"border: 1px solid {color['border']}; padding: 3px 9px; border-radius: 6px; "
                f"font-weight: 600; font-size: 0.9em; display: inline-block; margin-right: 4px;"
            )
            badge_spans.append(f'<span style="{style}">{var}</span>')

        badges_html = " ".join(badge_spans)
        st.markdown(f"**Recognized Variables:** {badges_html}", unsafe_allow_html=True)

        with st.expander("Optimization Problem Statement Preview", expanded=True):
            highlighted_obj = highlight_variables_in_text(obj_input_qp, detected_vars_qp)
            st.markdown(
                f"**Parsed Objective:** {sense_qp} &nbsp; <code>{highlighted_obj}</code>",
                unsafe_allow_html=True,
            )

            lines = [c.strip() for c in constraints_input_qp.split("\n") if c.strip()]
            if lines:
                st.markdown("**Parsed Constraints:**")
                for idx, line in enumerate(lines, 1):
                    h_line = highlight_variables_in_text(line, detected_vars_qp)
                    st.markdown(f"&nbsp;&nbsp;**C{idx}:** <code>{h_line}</code>", unsafe_allow_html=True)
    else:
        st.markdown("*No variables detected yet.*")

    st.divider()

    solve_disabled_qp = (not is_obj_quad) or bool(non_linear_constraints)
    if st.button("Solve Quadratic Program", type="primary", disabled=solve_disabled_qp, key="qp_solve_btn"):
        constraints_list_qp = [c.strip() for c in constraints_input_qp.split("\n") if c.strip()]
        st.session_state.result_qp = solve_qp(
            objective_str=obj_input_qp,
            constraints_list=constraints_list_qp,
            sense=sense_qp,
            var_names=detected_vars_qp,
            bounds_dict=bounds_dict_qp
        )

    if st.session_state.result_qp is not None:
        default_x_qp = detected_vars_qp[0] if len(detected_vars_qp) > 0 else None
        default_y_qp = detected_vars_qp[1] if len(detected_vars_qp) > 1 else None
        display_results_qp(
            st.session_state.result_qp,
            sense_qp,
            default_x=default_x_qp,
            default_y=default_y_qp
        )


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if "result_qp" not in st.session_state:
        st.session_state.result_qp = None

    page_quadratic()
