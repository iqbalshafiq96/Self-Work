import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
import plotly.graph_objects as go
import requests
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="LP Optimizer", layout="centered")

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

RESERVED_WORDS = {
    "Min", "Max", "min", "max", "Abs", "abs", "sin", "cos", "tan",
    "exp", "log", "sqrt", "True", "False", "None", "and", "or", "not"
}

CONSTRAINT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

VAR_BADGE_COLORS = [
    {"bg": "#e1f5fe", "text": "#0288d1", "border": "#81d4fa"},
    {"bg": "#f3e5f5", "text": "#7b1fa2", "border": "#ce93d8"},
    {"bg": "#e8f5e9", "text": "#388e3c", "border": "#a5d6a7"},
    {"bg": "#fff3e0", "text": "#f57c00", "border": "#ffcc80"},
    {"bg": "#fce4ec", "text": "#c2185b", "border": "#f48fb1"},
    {"bg": "#e0f2f1", "text": "#00796b", "border": "#80cbc4"},
    {"bg": "#fffde7", "text": "#fbc02d", "border": "#fff59d"},
    {"bg": "#efebe9", "text": "#5d4037", "border": "#bcaaa4"},
]

if "result_example" not in st.session_state:
    st.session_state.result_example = None
if "result_custom" not in st.session_state:
    st.session_state.result_custom = None


# ========================================================================
# LLM SCHEMA & AI PARSER (GOOGLE GEMINI)
# ========================================================================
class LPProblemSchema(BaseModel):
    sense: str = Field(description="Optimization sense: 'Maximize' or 'Minimize'")
    objective_function: str = Field(description="Algebraic objective expression without 'Maximize' or 'Minimize' prefix, e.g., '12*fracKerosene - 12*fracAGO'")
    constraints: list[str] = Field(description="List of constraint equations using <=, >=, or =, e.g., ['6*fracKerosene - 8*fracAGO <= 2.5', 'fracKerosene <= 0.75']")


def parse_lp_with_gemini(user_prompt: str, api_key: str = None) -> LPProblemSchema:
    """Extracts LP parameters from natural language using Google AI Studio Gemini API."""
    resolved_api_key = api_key or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not resolved_api_key:
        raise ValueError("Google API Key not found. Please add GOOGLE_API_KEY to Streamlit Secrets.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=resolved_api_key
    )

    structured_llm = llm.with_structured_output(LPProblemSchema)

    system_prompt = (
        "You are an expert operations research assistant. Parse the user's natural language linear programming problem. "
        "Extract decision variables, formulate the algebraic objective function, and construct clean constraint equations. "
        "Do NOT include unit labels or currency signs in algebraic terms. Standardize variable names using standard Python identifier names (e.g., fracKerosene, fracAGO, x, y)."
    )

    return structured_llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])


# ========================================================================
# LP ENGINE - parsing, linearity check, solving
# ========================================================================
def extract_identifiers(text: str) -> set:
    tokens = set(IDENTIFIER_RE.findall(text))
    return tokens - RESERVED_WORDS


def check_expression_linearity(expr_str: str, var_names: list) -> tuple[bool, str]:
    """
    Analyzes an expression using SymPy to detect higher-order terms, variable products,
    or non-linear functions (e.g. x^2, x*y, sin(x), sqrt(x)).
    Returns (is_linear, message).
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

    for sym in free_symbols:
        try:
            poly = sp.Poly(parsed_expr, sym)
            if poly.degree(sym) > 1:
                return False, f"Higher-order polynomial term detected for variable '{sym.name}' (degree = {poly.degree(sym)})."
        except sp.PolynomialError:
            return False, f"Non-polynomial or non-linear transcendental term (e.g., trig, log, exp, fractional exponent) detected containing variable '{sym.name}'."

    if len(free_symbols) > 1:
        try:
            poly = sp.Poly(parsed_expr, *list(free_symbols))
            if poly.total_degree() > 1:
                return False, f"Multiplicative interaction term between variables detected (total degree = {poly.total_degree()})."
        except sp.PolynomialError:
            return False, "Non-linear term or cross-variable multiplication detected."

    return True, ""


def highlight_variables_in_text(text: str, detected_vars: list) -> str:
    """Replaces variable occurrences in text with HTML styled badges."""
    if not text or not detected_vars:
        return text

    sorted_vars = sorted(detected_vars, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(v) for v in sorted_vars) + r")\b")

    def replacer(match):
        var = match.group(0)
        idx = detected_vars.index(var)
        color = VAR_BADGE_COLORS[idx % len(VAR_BADGE_COLORS)]
        style = (
            f"background-color: {color['bg']}; "
            f"color: {color['text']}; "
            f"border: 1px solid {color['border']}; "
            f"padding: 1px 6px; "
            f"border-radius: 4px; "
            f"font-weight: 600; "
            f"font-family: monospace;"
        )
        return f'<span style="{style}">{var}</span>'

    return pattern.sub(replacer, text)


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
    if not var_names:
        return {"success": False, "message": "No variables defined."}

    var_names = sorted(list(var_names))
    sym_vars = [sp.Symbol(v) for v in var_names]
    local_dict = {v: sym_vars[i] for i, v in enumerate(var_names)}

    is_lin, lin_msg = check_expression_linearity(objective_str, var_names)
    if not is_lin:
        return {"success": False, "message": f"Non-linear objective function: {lin_msg}"}

    try:
        obj_expr = sp.sympify(objective_str, locals=local_dict)
    except Exception as e:
        return {"success": False, "message": f"Error parsing objective function: {e}"}

    c = []
    for var in sym_vars:
        coeff = obj_expr.coeff(var)
        if any(v in coeff.free_symbols for v in sym_vars):
            return {"success": False, "message": f"Non-linear term detected in objective for variable {var}."}
        c.append(float(coeff))

    obj_const = float(obj_expr.as_coefficients_dict().get(1, 0))
    c_raw = list(c)

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
        opt_val = (-res.fun if sense.lower() == "maximize" else res.fun) + obj_const
        solution = dict(zip(var_names, res.x))
        return {
            "success": True,
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
            "raw_constraints": constraints_list,
        }
    else:
        return {"success": False, "message": f"Solver failed: {res.message}"}


def compute_feasible_polygon_vertices(halfplanes, bounds_x, bounds_y):
    all_planes = list(halfplanes)
    all_planes.append((1.0, 0.0, bounds_x[1]))    # x <= x_max
    all_planes.append((-1.0, 0.0, -bounds_x[0]))  # x >= x_min
    all_planes.append((0.0, 1.0, bounds_y[1]))    # y <= y_max
    all_planes.append((0.0, -1.0, -bounds_y[0]))  # y >= y_min

    intersections = []
    num_planes = len(all_planes)

    for i in range(num_planes):
        for j in range(i + 1, num_planes):
            a1, b1, c1 = all_planes[i]
            a2, b2, c2 = all_planes[j]

            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-9:
                continue

            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det

            feasible = True
            for a, b, c in all_planes:
                if a * x + b * y > c + 1e-6:
                    feasible = False
                    break

            if feasible:
                intersections.append((x, y))

    if not intersections:
        return None, None

    pts = np.unique(np.round(intersections, 6), axis=0)
    if len(pts) < 3:
        return None, None

    try:
        hull = ConvexHull(pts)
        ordered_pts = pts[hull.vertices]
        return ordered_pts[:, 0], ordered_pts[:, 1]
    except Exception:
        return None, None


def plot_interactive_contour_lines(result: dict, default_x: str = None, default_y: str = None):
    var_names = result["var_names"]

    if len(var_names) < 2:
        st.info("Interactive contour line plots require at least 2 decision variables.")
        return

    default_x_idx = var_names.index(default_x) if default_x in var_names else 0

    st.markdown("**2D Projection Settings**")
    col_x, col_y, col_n = st.columns(3)

    with col_x:
        x_name = st.selectbox("X-Axis Variable", var_names, index=default_x_idx, key="contour_x_var")

    y_options = [v for v in var_names if v != x_name]
    default_y_idx = y_options.index(default_y) if default_y in y_options else 0

    with col_y:
        y_name = st.selectbox("Y-Axis Variable", y_options, index=default_y_idx, key="contour_y_var")

    with col_n:
        n_contours = st.number_input(
            "Objective Contours (N)", 
            min_value=5, 
            max_value=300, 
            value=60, 
            step=5, 
            key="n_contours_input",
            help="Higher values increase contour frequency and produce finer intervals."
        )

    allow_negative = st.checkbox(
        "Allow Negative Axes Ranges", 
        value=False, 
        key="allow_neg_axes",
        help="If unticked (default), axes will strictly lock to 0 as the minimum value when scrolling or panning."
    )

    x_idx = var_names.index(x_name)
    y_idx = var_names.index(y_name)

    opt_x = result["x"][x_name]
    opt_y = result["x"][y_name]

    view_x_max = max(opt_x * 1.5, 2.0)
    view_y_max = max(opt_y * 1.5, 2.0)

    calc_x_max = view_x_max * 10.0
    calc_y_max = view_y_max * 10.0

    fixed_objective_contrib = result.get("obj_const", 0.0)
    c = result["c"]

    fixed_vars_summary = []
    for idx, v_name in enumerate(var_names):
        if idx not in (x_idx, y_idx):
            val = result["x"][v_name]
            fixed_objective_contrib += c[idx] * val
            fixed_vars_summary.append(f"{v_name} = {val:,.2f}")

    if fixed_vars_summary:
        st.caption(f"ℹ️ Other variables held constant at optimal values: **{', '.join(fixed_vars_summary)}**")

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

    b_x_min = 0.0 if x_min_b is None else x_min_b
    b_x_max = calc_x_max if x_max_b is None else min(x_max_b, calc_x_max)
    b_y_min = 0.0 if y_min_b is None else y_min_b
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
                x=px,
                y=py,
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
    Z = c[x_idx] * X + c[y_idx] * Y + fixed_objective_contrib

    fig.add_trace(
        go.Contour(
            x=x_vals,
            y=y_vals,
            z=Z,
            contours_coloring="lines",
            ncontours=int(n_contours),
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='navy')
            ),
            line=dict(color='#1f77b4', width=1.5, dash='dash'),
            showscale=False,
            showlegend=False,
            hoverinfo="x+y+z"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='#1f77b4', width=1.5, dash='dash'),
            name="Objective Contour",
            showlegend=True
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
                        y=[y_min_calc, calc_y_max],
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        name=f"C{idx+1}: {constr_label}",
                        hoverinfo="x+y"
                    )
                )

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

    x_min_view = None if allow_negative else 0
    y_min_view = None if allow_negative else 0

    xaxis_config = dict(
        title=x_name, 
        range=[x_min_view, view_x_max], 
        showgrid=True, 
        gridcolor='rgba(200,200,200,0.4)'
    )
    yaxis_config = dict(
        title=y_name, 
        range=[y_min_view, view_y_max], 
        showgrid=True, 
        gridcolor='rgba(200,200,200,0.4)'
    )

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
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(200,200,200,0.6)",
            borderwidth=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_results(result: dict, sense: str, default_x: str = None, default_y: str = None):
    if result["success"]:
        st.success("Optimization Completed Successfully!")

        col1, _ = st.columns(2)
        with col1:
            st.metric(label=f"Optimal Objective Value ({sense})", value=f"{result['fun']:,.4f}")

        st.subheader("Optimal Decision Variable Values")
        df_res = pd.DataFrame(
            list(result["x"].items()),
            columns=["Variable", "Optimal Value"]
        )
        st.dataframe(df_res.style.format({"Optimal Value": "{:,.4f}"}), use_container_width=True)

        st.subheader("Interactive Objective Contour Map & Feasible Area")
        plot_interactive_contour_lines(result, default_x=default_x, default_y=default_y)
    else:
        st.error(f"Solver Error: {result['message']}")


# ========================================================================
# PAGES / VIEWS
# ========================================================================
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
        st.session_state.result_example = solve_lp(
            objective_str=objective_str,
            constraints_list=constraint_strs,
            sense="Maximize",
            var_names=var_names,
            bounds_dict=bounds_dict
        )

    if st.session_state.result_example is not None:
        display_results(
            st.session_state.result_example, 
            sense="Maximize", 
            default_x="Oman", 
            default_y="Murban"
        )


def page_custom():
    st.header("LP Problem Statement (Objective Function)")

    # --- AI PARSER SECTION ---
    with st.expander("✨ Auto-parse problem statement using Google AI Studio (Gemini)", expanded=True):
        natural_prompt = st.text_area(
            "Describe your Linear Programming problem in natural language:",
            placeholder=(
                "A crude distillation column splits incoming crude oil into fractions by boiling range, "
                "and two of the boundaries between adjacent products are \"swing cuts\" whose draw point "
                "can be adjusted: a 3,000 barrel-per-day swing cut sitting between Heavy Naphtha and Kerosene, "
                "and a 4,000 barrel-per-day swing cut sitting between Diesel and Atmospheric Gas Oil (AGO); "
                "let fracKerosene be the fraction of the first swing cut routed to Kerosene rather than Heavy Naphtha, "
                "and fracAGO be the fraction of the second swing cut routed to AGO rather than Diesel, both fractions "
                "constrained between 0 and 1, and the goal is to choose fracKerosene and fracAGO to maximize the "
                "incremental daily margin in thousands of dollars per day, given by 12 times fracKerosene minus 12 times "
                "fracAGO, since Kerosene sells at a $4-per-barrel premium over Heavy Naphtha while Diesel sells at a "
                "$3-per-barrel premium over AGO, subject to the requirement that 6 times fracKerosene minus 8 times "
                "fracAGO not exceed 2.5 to respect the downstream hydrotreater's throughput capacity, that 4 times "
                "fracKerosene minus 3 times fracAGO not exceed 2.6 to respect the column's reboiler and duty capacity, "
                "that fracKerosene not exceed 0.75 to keep the Kerosene product within its freeze-point specification, "
                "and that fracAGO be at least 0.20 to meet a contracted minimum production requirement for AGO."
            ),
            height=320
        )
        if st.button("🤖 Parse with Gemini", type="secondary"):
            if not natural_prompt.strip():
                st.warning("Please enter a natural language problem statement.")
            else:
                with st.spinner("Parsing problem with Gemini..."):
                    try:
                        parsed = parse_lp_with_gemini(natural_prompt)
                        st.session_state["parsed_sense"] = parsed.sense
                        st.session_state["parsed_obj"] = parsed.objective_function
                        st.session_state["parsed_constraints"] = "\n".join(parsed.constraints)
                        st.success("Successfully parsed problem statement!")
                    except Exception as e:
                        st.error(f"Failed to parse via Gemini API: {e}")

    default_sense = st.session_state.get("parsed_sense", "Maximize")
    default_obj = st.session_state.get("parsed_obj", "12 * fracKerosene - 12 * fracAGO")
    default_constraints = st.session_state.get(
        "parsed_constraints",
        "6 * fracKerosene - 8 * fracAGO <= 2.5\n"
        "4 * fracKerosene - 3 * fracAGO <= 2.6\n"
        "fracKerosene <= 0.75\n"
        "fracAGO >= 0.20\n"
        "fracKerosene >= 0\n"
        "fracAGO <= 1"
    )

    st.caption(
        "Use plain, meaningful variable names — e.g. `fracKerosene`, `fracAGO`, `x`, `y`. "
        "Any word works as a variable, and the detected list below updates as you type."
    )

    col_opt, col_sense = st.columns([3, 1])
    with col_sense:
        sense_index = 0 if default_sense.lower() == "maximize" else 1
        sense = st.selectbox("Optimization Sense", ["Maximize", "Minimize"], index=sense_index)
    with col_opt:
        obj_input = st.text_input("Objective Function", value=default_obj)

    # --- NON-LINEARITY DETECTION ENGINE FOR OBJECTIVE FUNCTION ---
    all_text = obj_input + "\n" + default_constraints
    detected_vars = sorted(list(extract_identifiers(all_text)))
    is_obj_linear, obj_lin_err = check_expression_linearity(obj_input, detected_vars)

    if not is_obj_linear:
        st.error(
            f"⚠️ **Non-Linear Objective Detected:** The HiGHS Simplex LP solver cannot solve this function.\n\n"
            f"**Reason:** {obj_lin_err}\n\n"
            f"*Please reformulate your objective function as a strictly linear expression (degree = 1 with linear coefficients).*",
            icon="🚨"
        )

    st.subheader("Constraints")
    st.caption("Enter one constraint per line using `<=`, `>=`, or `=`.")
    constraints_input = st.text_area(
        "Constraints List",
        value=default_constraints,
        height=160
    )

    all_text = obj_input + "\n" + constraints_input
    detected_vars = sorted(list(extract_identifiers(all_text)))

    enable_bounds = st.checkbox("Enable Custom Variable Bounds", value=False)
    bounds_dict = {}

    if enable_bounds:
        st.markdown("**Configure Bounds**")
        if detected_vars:
            cols = st.columns(min(len(detected_vars), 4))
            for i, var in enumerate(detected_vars):
                with cols[i % 4]:
                    st.write(f"**{var}**")
                    min_val = st.number_input(f"Min ({var})", value=0.0, key=f"min_{var}")
                    has_max = st.checkbox(f"Set Max ({var})", key=f"has_max_{var}")
                    max_val = st.number_input(f"Max ({var})", value=100.0, key=f"max_{var}") if has_max else None
                    bounds_dict[var] = (min_val, max_val)
        else:
            st.warning("No variables detected yet to configure bounds.")
    else:
        bounds_dict = {var: (0.0, None) for var in detected_vars}

    st.divider()

    if detected_vars:
        badge_spans = []
        for i, var in enumerate(detected_vars):
            color = VAR_BADGE_COLORS[i % len(VAR_BADGE_COLORS)]
            style = (
                f"background-color: {color['bg']}; "
                f"color: {color['text']}; "
                f"border: 1px solid {color['border']}; "
                f"padding: 3px 9px; "
                f"border-radius: 6px; "
                f"font-weight: 600; "
                f"font-size: 0.9em; "
                f"display: inline-block; "
                f"margin-right: 4px;"
            )
            badge_spans.append(f'<span style="{style}">{var}</span>')

        badges_html = " ".join(badge_spans)
        st.markdown(f"**Recognized Variables:** {badges_html}", unsafe_allow_html=True)

        with st.expander("Optimization Problem Statement Preview", expanded=True):
            highlighted_obj = highlight_variables_in_text(obj_input, detected_vars)
            st.markdown(
                f"**Parsed Objective:** {sense} &nbsp; <code>{highlighted_obj}</code>",
                unsafe_allow_html=True,
            )

            lines = [c.strip() for c in constraints_input.split("\n") if c.strip()]
            if lines:
                st.markdown("**Parsed Constraints:**")
                for idx, line in enumerate(lines, 1):
                    h_line = highlight_variables_in_text(line, detected_vars)
                    st.markdown(
                        f"&nbsp;&nbsp;**C{idx}:** <code>{h_line}</code>",
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown("*No variables detected yet.*")

    st.divider()

    if st.button("Solve Custom LP", type="primary", disabled=not is_obj_linear):
        constraints_list = [c.strip() for c in constraints_input.split("\n") if c.strip()]
        st.session_state.result_custom = solve_lp(
            objective_str=obj_input,
            constraints_list=constraints_list,
            sense=sense,
            var_names=detected_vars,
            bounds_dict=bounds_dict
        )

    if st.session_state.result_custom is not None:
        display_results(
            st.session_state.result_custom, 
            sense,
            default_x="fracKerosene",
            default_y="fracAGO"
        )


# ========================================================================
# MAIN ROUTER & TOP NAVIGATION BUTTONS
# ========================================================================
if "page" not in st.session_state:
    st.session_state.page = "custom"  # Default view

st.title("Linear Programming Optimizer")

# 3 Button Tabs Navigation
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📘 Example: Refinery Crude LP", use_container_width=True,
                 type="primary" if st.session_state.page == "example" else "secondary"):
        st.session_state.page = "example"
with col2:
    if st.button("✍️ Build your own LP", use_container_width=True,
                 type="primary" if st.session_state.page == "custom" else "secondary"):
        st.session_state.page = "custom"
with col3:
    if st.button("🌐 Non-Linear Optimizer - Coming Soon)", use_container_width=True,
                 type="primary" if st.session_state.page == "app2" else "secondary"):
        st.session_state.page = "app2"

st.divider()

# Page Execution Logic
if st.session_state.page == "example":
    page_example()
elif st.session_state.page == "custom":
    page_custom()
elif st.session_state.page == "app2":
    # Replace this URL with your raw GitHub App2.py file URL
    # Format: https://raw.githubusercontent.com/<USERNAME>/<REPO>/<BRANCH>/App2.py
    GITHUB_APP2_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/App2.py"
    
    try:
        response = requests.get(GITHUB_APP2_URL)
        if response.status_code == 200:
            app2_code = response.text
            # Dynamically execute App2.py code within this tab scope
            exec(app2_code)
        else:
            st.error(f"Failed to fetch App2.py from GitHub. HTTP Status: {response.status_code}")
    except Exception as e:
        st.error(f"Error loading App2.py from GitHub: {e}")
