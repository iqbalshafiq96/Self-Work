from scipy.optimize import minimize, Bounds, LinearConstraint, linprog
import os
import sympy as sp
import numpy as np
import plotly.graphobjects as go
import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

----------------------------------------------------------------------
# AI PARSER (GOOGLE GEMINI) - QUADRATIC (QP)
----------------------------------------------------------------------
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
        model="gemini-1.5-flash",
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

----------------------------------------------------------------------
# QP-SPECIFIC VALIDATION (degree <= 2 allowed; reuses sp already imported)
----------------------------------------------------------------------
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

----------------------------------------------------------------------
# QP ENGINE - matrix extraction & solving
----------------------------------------------------------------------
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


def qp_find_feasible_start(n, Aub, bub, Aeq, beq, bounds_list):
    """Phase-1 LP (zero objective) to find any feasible point as a warm start for the QP solver."""
    try:
        res0 = linprog(
            c=np.zeros(n),
            A_ub=Aub if Aub else None,
            b_ub=bub if bub else None,
            A_eq=Aeq if Aeq else None,
            b_eq=beq if beq else None,
            bounds=bounds_list,
            method="highs"
        )
        if res0.success:
            return res0.x
    except Exception:
        pass

    x0 = []
    for lo, hi in bounds_list:
        lov = lo if lo is not None else -1.0
        hiv = hi if hi is not None else (lov + 2.0 if lo is not None else 1.0)
        x0.append((lov + hiv) / 2.0)
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

    Aub, bub, Aeq, beq = [], [], [], []
    ub_orig_idx, eq_orig_idx = [], []

    for orig_i, constr in enumerate(constraints_list):
        if not constr.strip():
            continue
        try:
            diff, rel = parse_equation_or_inequality(constr, local_dict)
        except Exception as e:
            return {"success": False, "message": f"Error parsing constraint '{constr}': {e}"}

        is_linc, linc_msg = check_expression_linearity(str(diff), var_names)
        if not is_linc:
            return {"success": False,
                    "message": f"Constraint '{constr}' is non-linear ({linc_msg}). "
                               f"QP mode supports quadratic terms in the OBJECTIVE only; "
                               f"constraints must remain linear."}

        const_term = float(diff.as_coefficients_dict().get(1, 0))
        coeffs = [float(diff.coeff(v)) for v in sym_vars]

        if rel == "<=":
            Aub.append(coeffs)
            bub.append(-const_term)
            ub_orig_idx.append(orig_i)
        elif rel == ">=":
            Aub.append([-v for v in coeffs])
            bub.append(const_term)
            ub_orig_idx.append(orig_i)
        elif rel in ("==", "="):
            Aeq.append(coeffs)
            beq.append(-const_term)
            eq_orig_idx.append(orig_i)

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
    if Aub:
        lin_constraints.append(LinearConstraint(np.array(Aub), -np.inf, np.array(bub)))
    if Aeq:
        Aeq_np = np.array(Aeq)
        beq_np = np.array(beq)
        lin_constraints.append(LinearConstraint(Aeq_np, beq_np, beq_np))

    x0 = qp_find_feasible_start(n, Aub, bub, Aeq, beq, bounds_list)

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
        "objconst": const_orig,
        "objexpr": obj_expr,
        "symvars": sym_vars,
        "isconvex": is_convex,
        "eigvals": eigvals,
        "Aub": Aub,
        "bub": bub,
        "Aeq": Aeq,
        "beq": beq,
        "bounds": bounds_list,
        "varnames": var_names,
        "rawconstraints": constraints_list,
        "uborigidx": ub_orig_idx,
        "eqorigidx": eq_orig_idx,
    }

----------------------------------------------------------------------
# QP PLOTTING
----------------------------------------------------------------------
def plot_interactive_contour_lines_qp(result: dict, defaultx: str = None, defaulty: str = None):
    varnames = result["varnames"]

    if len(varnames) < 2:
        st.info("Interactive contour line plots require at least 2 decision variables.")
        return

    default_x_idx = varnames.index(defaultx) if defaultx in varnames else 0

    st.markdown("### 2D Projection Settings")
    colx, coly, coln = st.columns(3)

    with colx:
        xname = st.selectbox("X-Axis Variable", varnames, index=default_x_idx, key="contourxvarqp")

    yoptions = [v for v in varnames if v != xname]
    default_y_idx = yoptions.index(defaulty) if defaulty in yoptions else 0

    with coly:
        yname = st.selectbox("Y-Axis Variable", yoptions, index=default_y_idx, key="contouryvarqp")

    with coln:
        ncontours = st.number_input(
            "Objective Contours (N)",
            min_value=5,
            max_value=300,
            value=120,
            step=5,
            key="ncontoursinputqp",
            help="Higher values increase contour frequency and produce finer intervals."
        )

    allow_negative = st.checkbox(
        "Allow Negative Axes Ranges",
        value=False,
        key="allownegaxesqp",
        help="If unticked (default), axes will strictly lock to 0 as the minimum value when scrolling or panning."
    )

    xidx = varnames.index(xname)
    yidx = varnames.index(yname)

    optx = result["x"][xname]
    opty = result["x"][yname]

    view_xmax = max(abs(optx) * 1.5, 2.0)
    view_ymax = max(abs(opty) * 1.5, 2.0)

    calc_xmax = view_xmax * 10.0
    calc_ymax = view_ymax * 10.0

    fixed_vars_summary = []
    for idx, vname in enumerate(varnames):
        if idx not in (xidx, yidx):
            val = result["x"][vname]
            fixed_vars_summary.append(f"{vname} = {val:,.4f}")

    if fixed_vars_summary:
        st.caption(f"ℹ️ Other variables held constant at optimal values: {', '.join(fixed_vars_summary)}")

    objexpr = result["objexpr"]
    symvars = result["symvars"]
    subs_dict = {
        sv: result["x"][vn]
        for vn, sv in zip(varnames, symvars)
        if vn not in (xname, yname)
    }
    xsym = sp.Symbol(xname)
    ysym = sp.Symbol(yname)
    expr2d = objexpr.subs(subs_dict)
    zfunc = sp.lambdify((xsym, ysym), expr2d, "numpy")

    halfplanes = []
    Aub = result.get("Aub", [])
    bub = result.get("bub", [])

    if Aub and bub:
        for a, b in zip(Aub, bub):
            effb = b
            for vi in range(len(varnames)):
                if vi not in (xidx, yidx):
                    effb -= a[vi] * result["x"][varnames[vi]]
            halfplanes.append((a[xidx], a[yidx], effb))

    bounds = result.get("bounds", [])
    xminb, xmaxb = bounds[xidx] if xidx < len(bounds) else (0, None)
    yminb, ymaxb = bounds[yidx] if yidx < len(bounds) else (0, None)

    bxmin = (-calc_xmax if allow_negative else 0.0) if xminb is None else xminb
    bxmax = calc_xmax if xmaxb is None else min(xmaxb, calc_xmax)
    bymin = (-calc_ymax if allow_negative else 0.0) if yminb is None else yminb
    bymax = calc_ymax if ymaxb is None else min(ymaxb, calc_ymax)

    fig = go.Figure()

    polyx, polyy = compute_feasible_polygon_vertices(
        halfplanes, boundsx=(bxmin, bxmax), boundsy=(bymin, bymax)
    )

    if polyx is not None and len(polyx) > 0:
        px = np.append(polyx, polyx[0])
        py = np.append(polyy, polyy[0])

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

    xmin_calc = -calc_xmax if allow_negative else 0
    ymin_calc = -calc_ymax if allow_negative else 0

    xvals = np.linspace(xmin_calc, calc_xmax, 250)
    yvals = np.linspace(ymin_calc, calc_ymax, 250)
    X, Y = np.meshgrid(xvals, yvals)

    Z = np.asarray(zfunc(X, Y), dtype=float)
    if Z.shape != X.shape:
        Z = np.full_like(X, float(Z))

    contour_line_color = '#1f77b4'

    fig.add_trace(
        go.Contour(
            x=xvals, y=yvals, z=Z,
            contours_coloring="lines",
            ncontours=int(ncontours),
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

    raw_constraints = result.get("rawconstraints", [])
    ub_orig_idx = result.get("uborigidx", list(range(len(Aub))))

    if Aub and bub:
        for rowi, (a, b) in enumerate(zip(Aub, bub)):
            origi = ub_orig_idx[rowi] if rowi < len(ub_orig_idx) else rowi

            effb = b
            for vi in range(len(varnames)):
                if vi not in (xidx, yidx):
                    effb -= a[vi] * result["x"][varnames[vi]]

            ax, ay = a[xidx], a[yidx]
            line_color = CONSTRAINT_COLORS[origi % len(CONSTRAINT_COLORS)]
            constr_label = raw_constraints[origi] if origi < len(raw_constraints) else f"Constraint {origi + 1}"

            if abs(ay) > 1e-6:
                yline = (effb - ax * xvals) / ay
                fig.add_trace(
                    go.Scatter(
                        x=xvals, y=yline, mode='lines',
                        line=dict(color=line_color, width=2),
                        name=f"C{origi + 1}: {constr_label}", hoverinfo="x+y"
                    )
                )
            elif abs(ax) > 1e-6:
                xval = effb / ax
                fig.add_trace(
                    go.Scatter(
                        x=[xval, xval], y=[ymin_calc, calc_ymax], mode='lines',
                        line=dict(color=line_color, width=2),
                        name=f"C{origi + 1}: {constr_label}", hoverinfo="x+y"
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=[optx], y=[opty], mode='markers+text',
            marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='black', width=1)),
            text=[f" Optimal ({optx:.2f}, {opty:.2f})"],
            textposition="top right", name="Optimal Solution", hoverinfo="x+y"
        )
    )

    xmin_view = None if allow_negative else 0
    ymin_view = None if allow_negative else 0

    xaxis_config = dict(title=xname, range=[xmin_view, view_xmax], showgrid=True, gridcolor='rgba(200,200,200,0.4)')
    yaxis_config = dict(title=yname, range=[ymin_view, view_ymax], showgrid=True, gridcolor='rgba(200,200,200,0.4)')

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


def display_results_qp(result: dict, sense: str, defaultx: str = None, defaulty: str = None):
    if result["success"]:
        st.success("Optimization Completed Successfully!")

        if not result.get("isconvex", True):
            eigvals = result.get("eigvals", [])
            min_eig = float(np.min(eigvals)) if len(eigvals) else 0.0
            st.warning(
                f"⚠️ Non-convex QP detected (most negative eigenvalue of the effective Hessian = "
                f"{min_eig:,.4f}). The solution below is a local optimum found by a gradient-based "
                f"solver (trust-constr) — global optimality is not guaranteed for non-convex problems.",
                icon="⚠️"
            )
        else:
            st.caption("✅ Convexity check passed — the Hessian is positive semi-definite, so this solution is the global optimum.")

        col1, _ = st.columns(2)
        with col1:
            st.metric(label=f"Optimal Objective Value ({sense})", value=f"{result['fun']:,.4f}")

        st.subheader("Optimal Decision Variable Values")
        dfres = pd.DataFrame(list(result["x"].items()), columns=["Variable", "Optimal Value"])
        st.dataframe(dfres.style.format({"Optimal Value": "{:,.4f}"}), use_container_width=True)

        st.subheader("Interactive Objective Contour Map & Feasible Area")
        plot_interactive_contour_lines_qp(result, defaultx=defaultx, defaulty=defaulty)
    else:
        st.error(f"Solver Error: {result['message']}")

----------------------------------------------------------------------
# PAGE FUNCTION
----------------------------------------------------------------------
def page_quadratic():
    st.header("QP Problem Statement (Objective Function)")
    st.caption(
        "This solver supports quadratic objective functions (e.g., x**2, 3*x*y, -2*y**2) "
        "combined with linear constraints only. Internally it minimizes/maximizes "
        "0.5·xᵀQx + cᵀx via SciPy's trust-constr solver using the exact Hessian/gradient."
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
            key="qpnaturalprompt"
        )
        if st.button("🤖 Parse with Gemini", type="secondary", key="qpparsebtn"):
            if not natural_prompt_qp.strip():
                st.warning("Please enter a natural language problem statement.")
            else:
                with st.spinner("Parsing problem with Gemini…"):
                    try:
                        parsed_qp = parse_qp_with_gemini(natural_prompt_qp)

                        st.session_state["qpsense"] = parsed_qp.sense
                        st.session_state["qpobjinput"] = parsed_qp.objective_function
                        st.session_state["qpconstraintsinput"] = "\n".join(parsed_qp.constraints)

                        st.success("Successfully parsed problem statement!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to parse via Gemini API: {e}")

    if "qpsense" not in st.session_state:
        st.session_state["qpsense"] = "Minimize"
    if "qpobjinput" not in st.session_state:
        st.session_state["qpobjinput"] = "0.02*r1**2 + 3*r1 + 0.03*r2**2 + 2*r2 + 0.05*r3**2 + r3"
    if "qpconstraintsinput" not in st.session_state:
        st.session_state["qpconstraintsinput"] = (
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
        "Use plain, meaningful variable names — e.g. x, y, qty1. "
        "Quadratic/bilinear terms (x**2, x*y) are allowed *only* in the objective; "
        "constraints must stay linear."
    )

    coloptqp, colsenseqp = st.columns([3, 1])
    with colsenseqp:
        senseqp = st.selectbox(
            "Optimization Sense",
            ["Maximize", "Minimize"],
            key="qpsense"
        )
    with coloptqp:
        objinputqp = st.text_input(
            "Objective Function",
            key="qpobjinput"
        )

    alltextqp = objinputqp + "\n" + st.session_state["qpconstraintsinput"]
    detectedvarsqp = sorted(list(extract_identifiers(alltextqp)))
    isobjquad, objquaderr = check_expression_quadratic(objinputqp, detectedvarsqp)

    if not isobjquad:
        st.error(
            f"⚠️ Unsupported Objective Detected: The QP solver requires degree ≤ 2.\n\n"
            f"Reason: {objquaderr}\n\n"
            f"Please reformulate your objective function so every term has total degree ≤ 2 "
            f"(constants, linear terms like x, pure quadratic terms like x**2, or bilinear terms like x*y).",
            icon="🚨"
        )

    st.subheader("Constraints")
    st.caption("Enter one linear constraint per line using <=, >=, or = (no quadratic terms here).")
    constraintsinputqp = st.text_area(
        "Constraints List",
        height=160,
        key="qpconstraintsinput"
    )

    alltextqp = objinputqp + "\n" + constraintsinputqp
    detectedvarsqp = sorted(list(extract_identifiers(alltextqp)))

    nonlinearconstraints = []
    for line in [c.strip() for c in constraintsinputqp.split("\n") if c.strip()]:
        try:
            symdict = {v: sp.Symbol(v) for v in detectedvarsqp}
            diff, _ = parse_equation_or_inequality(line, symdict)
            if diff is not None:
                islinline, linerr = check_expression_linearity(str(diff), detectedvarsqp)
                if not islinline:
                    nonlinearconstraints.append((line, linerr))
        except Exception:
            pass

    if nonlinearconstraints:
        msgs = "\n".join([f"- {ln} → {err}" for ln, err in nonlinearconstraints])
        st.error(
            f"⚠️ Non-Linear Constraint(s) Detected: QP mode only allows quadratic terms in the "
            f"objective; all constraints must be linear.\n\n{msgs}",
            icon="🚨"
        )

    enableboundsqp = st.checkbox("Enable Custom Variable Bounds", value=False, key="qpenablebounds")
    boundsdictqp = {}

    if enableboundsqp:
        st.markdown("Configure Bounds")
        if detectedvarsqp:
            cols = st.columns(min(len(detectedvarsqp), 4))
            for i, var in enumerate(detectedvarsqp):
                with cols[i % 4]:
                    st.write(f"{var}")
                    minval = st.number_input(f"Min ({var})", value=0.0, key=f"qpmin{var}")
                    hasmax = st.checkbox(f"Set Max ({var})", key=f"qphasmax{var}")
                    maxval = st.number_input(f"Max ({var})", value=100.0, key=f"qpmax{var}") if hasmax else None
                    boundsdictqp[var] = (minval, maxval)
        else:
            st.warning("No variables detected yet to configure bounds.")
    else:
        boundsdictqp = {var: (0.0, None) for var in detectedvarsqp}

    st.divider()

    if detectedvarsqp:
        badgespans = []
        for i, var in enumerate(detectedvarsqp):
            color = VAR_BADGE_COLORS[i % len(VAR_BADGE_COLORS)]
            style = (
                f"background-color: {color['bg']}; color: {color['text']}; "
                f"border: 1px solid {color['border']}; padding: 3px 9px; border-radius: 6px; "
                f"font-weight: 600; font-size: 0.9em; display: inline-block; margin-right: 4px;"
            )
            badgespans.append(f'<span style="{style}">{var}</span>')

        badgeshtml = " ".join(badgespans)
        st.markdown(f"Recognized Variables: {badgeshtml}", unsafe_allow_html=True)

        with st.expander("Optimization Problem Statement Preview", expanded=True):
            highlighted_obj = highlight_variables_in_text(objinputqp, detectedvarsqp)
            st.markdown(
                f"Parsed Objective: **{senseqp}** &nbsp; {highlighted_obj}",
                unsafe_allow_html=True,
            )

            lines = [c.strip() for c in constraintsinputqp.split("\n") if c.strip()]
            if lines:
                st.markdown("Parsed Constraints:")
                for idx, line in enumerate(lines, 1):
                    h_line = highlight_variables_in_text(line, detectedvarsqp)
                    st.markdown(f"&nbsp;&nbsp;**C{idx}:** {h_line}", unsafe_allow_html=True)
    else:
        st.markdown("No variables detected yet.")

    st.divider()

    solvedisabledqp = (not isobjquad) or bool(nonlinearconstraints)
    if st.button("Solve Quadratic Program", type="primary", disabled=solvedisabledqp, key="qpsolvebtn"):
        constraintslistqp = [c.strip() for c in constraintsinputqp.split("\n") if c.strip()]
        st.session_state.resultqp = solve_qp(
            objective_str=objinputqp,
            constraints_list=constraintslistqp,
            sense=senseqp,
            var_names=detectedvarsqp,
            bounds_dict=boundsdictqp
        )

    if st.session_state.resultqp is not None:
        defaultxqp = detectedvarsqp[0] if len(detectedvarsqp) > 0 else None
        defaultyqp = detectedvarsqp[1] if len(detectedvarsqp) > 1 else None
        display_results_qp(
            st.session_state.resultqp,
            senseqp,
            defaultx=defaultxqp,
            defaulty=defaultyqp
        )

----------------------------------------------------------------------
# ENTRY POINT
----------------------------------------------------------------------
if "resultqp" not in st.session_state:
    st.session_state.resultqp = None

page_quadratic()
