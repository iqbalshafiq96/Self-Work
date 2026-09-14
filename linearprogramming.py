import re
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
import plotly.graph_objects as go
from scipy.optimize import linprog

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

# Color palette for unique variable badges
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
# LP ENGINE - parsing, linearity check, solving
# ========================================================================
def extract_identifiers(text: str) -> set:
    tokens = set(IDENTIFIER_RE.findall(text))
    return tokens - RESERVED_WORDS


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


def plot_interactive_contour_lines(result: dict, default_x: str = None, default_y: str = None):
    """Generate an interactive 2D objective contour plot with shaded feasible region."""
    var_names = result["var_names"]

    if len(var_names) < 2:
        st.info("Interactive contour line plots require at least 2 decision variables.")
        return

    default_x_idx = var_names.index(default_x) if default_x in var_names else 0

    st.markdown("**2D Projection Settings**")
    col_x, col_y = st.columns(2)

    with col_x:
        x_name = st.selectbox("X-Axis Variable", var_names, index=default_x_idx, key="contour_x_var")

    y_options = [v for v in var_names if v != x_name]
    default_y_idx = y_options.index(default_y) if default_y in y_options else 0

    with col_y:
        y_name = st.selectbox("Y-Axis Variable", y_options, index=default_y_idx, key="contour_y_var")

    x_idx = var_names.index(x_name)
    y_idx = var_names.index(y_name)

    opt_x = result["x"][x_name]
    opt_y = result["x"][y_name]

    x_max = max(opt_x * 1.5, 10.0)
    y_max = max(opt_y * 1.5, 10.0)

    x_vals = np.linspace(0, x_max, 250)
    y_vals = np.linspace(0, y_max, 250)
    X, Y = np.meshgrid(x_vals, y_vals)

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

    c_x = c[x_idx]
    c_y = c[y_idx]
    Z = c_x * X + c_y * Y + fixed_objective_contrib

    fig = go.Figure()

    # -------------------------------------------------------------------------
    # 1. SHADE FEASIBLE REGION
    # -------------------------------------------------------------------------
    feasible_mask = np.ones_like(X, dtype=bool)

    A_ub = result.get("A_ub", [])
    b_ub = result.get("b_ub", [])
    if A_ub and b_ub:
        for a, b in zip(A_ub, b_ub):
            eff_b = b
            for v_i in range(len(var_names)):
                if v_i not in (x_idx, y_idx):
                    eff_b -= a[v_i] * result["x"][var_names[v_i]]
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

    # Solid green overlay for feasible area
    fig.add_trace(
        go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=feasible_z,
            showscale=False,
            colorscale=[[0, "rgba(46, 204, 113, 0.25)"], [1, "rgba(46, 204, 113, 0.25)"]],
            hoverinfo="skip",
            name="Feasible Region",
            showlegend=True
        )
    )

    # -------------------------------------------------------------------------
    # 2. CONTOUR LINES & CONSTRAINTS
    # -------------------------------------------------------------------------
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
            showscale=False,
            showlegend=False,
            hoverinfo="x+y+z"
        )
    )

    # Explicit legend item for Objective Contour Line to match dashed line style
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
                        y=[0, y_max],
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        name=f"C{idx+1}: {constr_label}",
                        hoverinfo="x+y"
                    )
                )

    # -------------------------------------------------------------------------
    # 3. OPTIMAL POINT MARKER
    # -------------------------------------------------------------------------
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

    fig.update_layout(
        title="",
        xaxis=dict(title=x_name, range=[0, x_max], showgrid=True, gridcolor='rgba(200,200,200,0.4)'),
        yaxis=dict(title=y_name, range=[0, y_max], showgrid=True, gridcolor='rgba(200,200,200,0.4)'),
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
# ROUTER
# ========================================================================
if "page" not in st.session_state:
    st.session_state.page = "example"

st.title("Linear Programming Optimizer")

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
    st.header("Build your own LP problem")
    st.caption(
        "Use plain, meaningful variable names instead of x, y — e.g. `Utility`, "
        "`RawMaterial`. Any word works as a variable, and the detected list "
        "below updates as you type."
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

    # Parse all variables present across objective function and constraint text boxes
    all_text = obj_input + "\n" + constraints_input
    detected_vars = sorted(list(extract_identifiers(all_text)))

    # Render uniquely colored badges directly under both dialog boxes
    if detected_vars:
        badge_spans = []
        for i, var in enumerate(detected_vars):
            color = VAR_BADGE_COLORS[i % len(VAR_BADGE_COLORS)]
            style = (
                f"background-color: {color['bg']}; "
                f"color: {color['text']}; "
                f"border: 1px solid {color['border']}; "
                "padding: 3px 9px; "
                "border-radius: 6px; "
                "font-weight: 600; "
                "font-size: 0.9em; "
                "display: inline-block; "
                "margin-right: 4px;"
            )
            badge_spans.append(f'<span style="{style}">{var}</span>')

        badges_html = " ".join(badge_spans)
        st.markdown(f"**Recognized Variables:** {badges_html}", unsafe_allow_html=True)
    else:
        st.markdown("*No variables detected yet.*")

    st.subheader("Variable Bounds")
    if detected_vars:
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
        st.session_state.result_custom = solve_lp(
            objective_str=obj_input,
            constraints_list=constraints_list,
            sense=sense,
            var_names=detected_vars,
            bounds_dict=bounds_dict
        )

    if st.session_state.result_custom is not None:
        display_results(st.session_state.result_custom, sense)


if st.session_state.page == "example":
    page_example()
else:
    page_custom()
