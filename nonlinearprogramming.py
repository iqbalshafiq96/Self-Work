from scipy.optimize import minimize, Bounds, LinearConstraint, linprog
import os
import sympy as sp
import numpy as np
import plotly.graphobjects as go
import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from langchaingoogle_genai import ChatGoogleGenerativeAI

----------------------------------------------------------------------
AI PARSER (GOOGLE GEMINI) - QUADRATIC (QP)
----------------------------------------------------------------------
class QPProblemSchema(BaseModel):
    sense: str = Field(description="Optimization sense: 'Maximize' or 'Minimize'")
    objectivefunction: str = Field(description="Algebraic objective expression without 'Maximize'/'Minimize' prefix. May include quadratic terms such as 'x2', 'xy', e.g., '2x2 + 3xy - y'")
    constraints: list[str] = Field(description="List of LINEAR constraint equations only, using <=, >=, or =, e.g., ['x + y <= 10', 'x >= 0']. Quadratic terms are NOT allowed in constraints.")


def parseqpwithgemini(userprompt: str, apikey: str = None) -> QPProblemSchema:
    """Extracts QP parameters from natural language using Google AI Studio Gemini API."""
    resolvedapikey = apikey or st.secrets.get("GOOGLEAPIKEY") or os.getenv("GOOGLEAPIKEY")

    if not resolvedapikey:
        raise ValueError("Google API Key not found. Please add GOOGLEAPIKEY to Streamlit Secrets.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        googleapikey=resolvedapikey
    )

    structuredllm = llm.withstructuredoutput(QPProblemSchema)

    systemprompt = (
        "You are an expert operations research assistant specializing in Quadratic Programming (QP). "
        "Parse the user's natural language problem. Extract decision variables and formulate the algebraic "
        "objective function, which MAY contain quadratic terms (e.g., x2, xy, 2y2). "
        "Constraints MUST remain strictly LINEAR (no quadratic or cross-product terms in constraints). "
        "Do NOT include unit labels or currency signs in algebraic terms. Standardize variable names using "
        "standard Python identifier names (e.g., x, y, qty1, qty2)."
    )

    return structuredllm.invoke([
        ("system", systemprompt),
        ("user", userprompt)
    ])

----------------------------------------------------------------------
QP-SPECIFIC VALIDATION (degree <= 2 allowed; reuses sp already imported)
----------------------------------------------------------------------
def checkexpressionquadratic(exprstr: str, varnames: list) -> tuple[bool, str]:
    """
    Analyzes an expression using SymPy and allows terms up to TOTAL DEGREE 2
    (linear terms, pure quadratic terms like x2, and bilinear cross terms like x*y).
    Rejects degree > 2 polynomial terms and any non-polynomial/transcendental terms.
    Returns (isquadraticorlower, message).
    """
    if not exprstr.strip():
        return True, ""

    try:
        symdict = {v: sp.Symbol(v) for v in varnames}
        parsedexpr = sp.sympify(exprstr, locals=symdict)
    except Exception as e:
        return False, f"Syntax / parsing error: {e}"

    freesymbols = parsedexpr.freesymbols
    if not freesymbols:
        return True, ""

    try:
        orderedsyms = sorted(freesymbols, key=str)
        poly = sp.Poly(parsedexpr, *orderedsyms)
        deg = poly.totaldegree()
        if deg > 2:
            return False, (f"Total degree {deg} detected — exceeds the maximum degree of 2 supported by "
                           f"the Quadratic Programming (QP) solver. Only linear and quadratic (incl. bilinear "
                           f"cross-product) terms are allowed, e.g., x, x2, x*y.")
    except sp.PolynomialError:
        return False, "Non-polynomial or transcendental term detected (e.g., trig, log, exp, fractional exponent, or division by a variable)."

    return True, ""

----------------------------------------------------------------------
QP ENGINE - matrix extraction & solving
----------------------------------------------------------------------
def buildquadraticmatrices(objexpr, symvars):
    """
    Decomposes a (degree <= 2) SymPy expression into:
        expr(x) = 0.5 * x^T Q x + c^T x + const
    using the exact Hessian (Q) and gradient-at-origin (c).
    Returns (Q: np.ndarray, c: np.ndarray, const: float)
    """
    n = len(symvars)
    zerosubs = {v: 0 for v in symvars}

    H = sp.hessian(objexpr, symvars)
    Q = np.array(H.evalf()).astype(float).reshape(n, n)

    grad = [sp.diff(objexpr, v).subs(zerosubs) for v in symvars]
    cvec = np.array([float(g) for g in grad])

    const = float(objexpr.subs(zerosubs))

    return Q, cvec, const


def qpfindfeasiblestart(n, Aub, bub, Aeq, beq, boundslist):
    """Phase-1 LP (zero objective) to find any feasible point as a warm start for the QP solver."""
    try:
        res0 = linprog(
            c=np.zeros(n),
            Aub=Aub if Aub else None,
            bub=bub if bub else None,
            Aeq=Aeq if Aeq else None,
            beq=beq if beq else None,
            bounds=boundslist,
            method="highs"
        )
        if res0.success:
            return res0.x
    except Exception:
        pass

    x0 = []
    for lo, hi in boundslist:
        lov = lo if lo is not None else -1.0
        hiv = hi if hi is not None else (lov + 2.0 if lo is not None else 1.0)
        x0.append((lov + hiv) / 2.0)
    return np.array(x0)


def solveqp(objectivestr: str, constraintslist: list, sense: str, varnames: list, boundsdict: dict):
    if not varnames:
        return {"success": False, "message": "No variables defined."}

    varnames = sorted(list(varnames))
    symvars = [sp.Symbol(v) for v in varnames]
    localdict = {v: symvars[i] for i, v in enumerate(varnames)}
    n = len(varnames)

    isquad, quadmsg = checkexpressionquadratic(objectivestr, varnames)
    if not isquad:
        return {"success": False, "message": f"Unsupported objective function: {quadmsg}"}

    try:
        objexpr = sp.sympify(objectivestr, locals=localdict)
    except Exception as e:
        return {"success": False, "message": f"Error parsing objective function: {e}"}

    Qorig, corig, constorig = buildquadraticmatrices(objexpr, symvars)

    # --- Constraints: must remain strictly LINEAR in QP mode ---
    # NOTE (BUGFIX): We now track the ORIGINAL index of every constraint line (as it
    # appears in constraints_list / the "Parsed Constraints Preview") alongside each
    # row we append to Aub/bub. Equality constraints are routed to Aeq and are
    # therefore absent from Aub — without recording the original index, the plot
    # function's enumerate(A_ub) position silently drifts out of sync with
    # raw_constraints, causing the legend color/number for a constraint to no longer
    # match the line actually drawn on the chart (e.g. "C5" shown in one color while the
    # line for the real C5 equation is drawn in another). Recording ub_orig_idx lets
    # the plot always look up the correct label AND the correct color using the same
    # original constraint number everywhere (preview, legend, and chart).
    Aub, bub, Aeq, beq = [], [], [], []
    uborigidx, eqorigidx = [], []

    for origi, constr in enumerate(constraintslist):
        if not constr.strip():
            continue
        try:
            diff, rel = parseequationorinequality(constr, localdict)
        except Exception as e:
            return {"success": False, "message": f"Error parsing constraint '{constr}': {e}"}

        islinc, lincmsg = checkexpressionlinearity(str(diff), varnames)
        if not islinc:
            return {"success": False,
                    "message": f"Constraint '{constr}' is non-linear ({lincmsg}). "
                               f"QP mode supports quadratic terms in the OBJECTIVE only; "
                               f"constraints must remain linear."}

        constterm = float(diff.ascoefficientsdict().get(1, 0))
        coeffs = [float(diff.coeff(v)) for v in symvars]

        if rel == "<=":
            Aub.append(coeffs)
            bub.append(-constterm)
            uborigidx.append(origi)
        elif rel == ">=":
            Aub.append([-v for v in coeffs])
            bub.append(constterm)
            uborigidx.append(origi)
        elif rel == "==":
            Aeq.append(coeffs)
            beq.append(-constterm)
            eqorigidx.append(origi)

    boundslist = [boundsdict.get(v, (0, None)) for v in varnames]
    scipybounds = Bounds(
        [(-np.inf if b[0] is None else b[0]) for b in boundslist],
        [(np.inf if b[1] is None else b[1]) for b in boundslist]
    )

    sensesign = -1.0 if sense.lower() == "maximize" else 1.0
    Qused = sensesign * Qorig
    cused = sensesign * corig
    Qsym = (Qused + Qused.T) / 2.0

    def objectivefunc(x):
        return 0.5 * x @ Qsym @ x + cused @ x

    def objectivegrad(x):
        return Qsym @ x + cused

    def objectivehess(x):
        return Qsym

    eigvals = np.linalg.eigvalsh(Qsym)
    isconvex = bool(np.all(eigvals >= -1e-7))

    linconstraints = []
    if Aub:
        linconstraints.append(LinearConstraint(np.array(Aub), -np.inf, np.array(bub)))
    if Aeq:
        Aeqnp = np.array(Aeq)
        beqnp = np.array(beq)
        linconstraints.append(LinearConstraint(Aeqnp, beqnp, beqnp))

    x0 = qpfindfeasiblestart(n, Aub, bub, Aeq, beq, boundslist)

    try:
        res = minimize(
            objectivefunc,
            x0,
            jac=objectivegrad,
            hess=objectivehess,
            method="trust-constr",
            bounds=scipybounds,
            constraints=linconstraints if linconstraints else None,
            options={"maxiter": 1000, "gtol": 1e-10, "xtol": 1e-12}
        )
    except Exception as e:
        return {"success": False, "message": f"Solver raised an exception: {e}"}

    if not res.success and res.status not in (1, 2):
        return {"success": False, "message": f"Solver failed: {res.message}"}

    optval = sensesign * res.fun + constorig
    solution = dict(zip(varnames, res.x))

    return {
        "success": True,
        "fun": optval,
        "x": solution,
        "message": res.message,
        "status": res.status,
        "Q": Qorig,
        "c": corig,
        "objconst": constorig,
        "objexpr": objexpr,
        "symvars": symvars,
        "isconvex": isconvex,
        "eigvals": eigvals,
        "Aub": Aub,
        "bub": bub,
        "Aeq": Aeq,
        "beq": beq,
        "bounds": boundslist,
        "varnames": varnames,
        "rawconstraints": constraintslist,
        "uborigidx": uborigidx,   # NEW: maps each Aub row -> its original constraint index
        "eqorigidx": eqorigidx,   # NEW: same, for A_eq rows (kept for completeness/future use)
    }

----------------------------------------------------------------------
QP PLOTTING (reuses computefeasiblepolygon_vertices from app.py)
----------------------------------------------------------------------
def plotinteractivecontourlinesqp(result: dict, defaultx: str = None, defaulty: str = None):
    """
    QP contour + feasible region plot. The objective may be quadratic, so the 2D
    projection Z-surface is derived via exact SymPy substitution (fixing all other
    variables at their optimal values) rather than a linear formula.
    """
    varnames = result["varnames"]

    if len(varnames) < 2:
        st.info("Interactive contour line plots require at least 2 decision variables.")
        return

    defaultxidx = varnames.index(defaultx) if defaultx in varnames else 0

    st.markdown("2D Projection Settings")
    colx, coly, coln = st.columns(3)

    with colx:
        xname = st.selectbox("X-Axis Variable", varnames, index=defaultxidx, key="contourxvarqp")

    yoptions = [v for v in varnames if v != xname]
    defaultyidx = yoptions.index(defaulty) if defaulty in yoptions else 0

    with coly:
        yname = st.selectbox("Y-Axis Variable", yoptions, index=defaultyidx, key="contouryvarqp")

    with coln:
        ncontours = st.numberinput(
            "Objective Contours (N)",
            minvalue=5,
            maxvalue=300,
            value=120,
            step=5,
            key="ncontoursinputqp",
            help="Higher values increase contour frequency and produce finer intervals."
        )

    allownegative = st.checkbox(
        "Allow Negative Axes Ranges",
        value=False,
        key="allownegaxesqp",
        help="If unticked (default), axes will strictly lock to 0 as the minimum value when scrolling or panning."
    )

    xidx = varnames.index(xname)
    yidx = varnames.index(yname)

    optx = result["x"][xname]
    opty = result["x"][yname]

    viewxmax = max(abs(optx) * 1.5, 2.0)
    viewymax = max(abs(opty) * 1.5, 2.0)

    calcxmax = viewxmax * 10.0
    calcymax = viewymax * 10.0

    fixedvarssummary = []
    for idx, vname in enumerate(varnames):
        if idx not in (xidx, yidx):
            val = result["x"][vname]
            fixedvarssummary.append(f"{vname} = {val:,.4f}")

    if fixedvarssummary:
        st.caption(f"ℹ️ Other variables held constant at optimal values: {', '.join(fixedvarssummary)}")

    objexpr = result["objexpr"]
    symvars = result["symvars"]
    subsdict = {
        sv: result["x"][vn]
        for vn, sv in zip(varnames, symvars)
        if vn not in (xname, yname)
    }
    xsym = sp.Symbol(xname)
    ysym = sp.Symbol(yname)
    expr2d = objexpr.subs(subsdict)
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

    bxmin = (-calcxmax if allownegative else 0.0) if xminb is None else xminb
    bxmax = calcxmax if xmaxb is None else min(xmaxb, calcxmax)
    bymin = (-calcymax if allownegative else 0.0) if yminb is None else yminb
    bymax = calcymax if ymaxb is None else min(ymaxb, calcymax)

    fig = go.Figure()

    polyx, polyy = computefeasiblepolygonvertices(
        halfplanes, boundsx=(bxmin, bxmax), boundsy=(bymin, bymax)
    )

    if polyx is not None and len(polyx) > 0:
        px = np.append(polyx, polyx[0])
        py = np.append(polyy, polyy[0])

        fig.addtrace(
            go.Scatter(
                x=px, y=py,
                fill="toself",
                fillcolor="rgba(46, 204, 113, 0.25)",
                line=dict(color="rgba(46, 204, 113, 0.6)", width=1),
                name="Feasible Region",
                hoverinfo="skip"
            )
        )

    xmincalc = -calcxmax if allownegative else 0
    ymincalc = -calcymax if allownegative else 0

    xvals = np.linspace(xmincalc, calcxmax, 250)
    yvals = np.linspace(ymincalc, calcymax, 250)
    X, Y = np.meshgrid(xvals, yvals)

    Z = np.asarray(zfunc(X, Y), dtype=float)
    if Z.shape != X.shape:
        Z = np.fulllike(X, float(Z))

    # Unified line color for contours and legend proxy
    contourlinecolor = '#1f77b4'

    fig.addtrace(
        go.Contour(
            x=xvals, y=yvals, z=Z,
            contourscoloring="lines",
            ncontours=int(ncontours),
            contours=dict(showlabels=True, labelfont=dict(size=10, color='navy')),
            line=dict(color=contourlinecolor, width=1.5, dash='dash'),
            showscale=False, showlegend=False, hoverinfo="x+y+z"
        )
    )

    fig.addtrace(
        go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=contourlinecolor, width=1.5, dash='dash'),
            name="Objective Contour", showlegend=True
        )
    )

    # ------------------------------------------------------------------
    # BUGFIX: constraint color/label alignment
    # ------------------------------------------------------------------
    # Previously this loop used enumerate(zip(A_ub, b_ub)) and indexed BOTH the
    # color (CONSTRAINT_COLORS[idx]) and the label (raw_constraints[idx]) using
    # the position within Aub only (inequalities only). Since equality constraints
    # are filtered out of Aub (they live in Aeq instead) but raw_constraints
    # contains the FULL original list, any equality constraint occurring before an
    # inequality caused every following Aub row to read the WRONG label from
    # raw_constraints, while the trace's line_color (still based on the Aub
    # position) no longer matched the color that same constraint's number would get
    # in the "Parsed Constraints Preview" above the chart. That's exactly what
    # produced "C5 shown as indigo in the legend, but drawn in red on the chart".
    #
    # Fix: use ub_orig_idx (the constraint's true position in the original,
    # unfiltered list) for BOTH the color lookup and the label/name, so the number,
    # color, and equation text are always consistent between the legend, the plotted
    # line, and the "Parsed Constraints Preview".
    rawconstraints = result.get("rawconstraints", [])
    uborigidx = result.get("uborigidx", list(range(len(Aub))))

    if Aub and bub:
        for rowi, (a, b) in enumerate(zip(Aub, bub)):
            origi = uborigidx[rowi] if rowi < len(uborigidx) else rowi

            effb = b
            for vi in range(len(varnames)):
                if vi not in (xidx, yidx):
                    effb -= a[vi] * result["x"][varnames[vi]]

            ax, ay = a[xidx], a[yidx]
            linecolor = CONSTRAINTCOLORS[origi % len(CONSTRAINTCOLORS)]
            constrlabel = raw¨C319Ci] if orig¨C320Cconstraints) else f"Constraint {orig¨C321Cy) > 1e-6:
                yline = (effb - ax * xvals) / ay
                fig.addtrace(
                    go.Scatter(
                        x=xvals, y=yline, mode='lines',
                        line=dict(color=linecolor, width=2),
                        name=f"C{origi + 1}: {constrlabel}", hoverinfo="x+y"
                    )
                )
            elif abs(ax) > 1e-6:
                xval = effb / ax
                fig.addtrace(
                    go.Scatter(
                        x=[xval, xval], y=[ymincalc, calcymax], mode='lines',
                        line=dict(color=linecolor, width=2),
                        name=f"C{origi + 1}: {constrlabel}", hoverinfo="x+y"
                    )
                )

    fig.addtrace(
        go.Scatter(
            x=[optx], y=[opty], mode='markers+text',
            marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='black', width=1)),
            text=[f" Optimal ({optx:.2f}, {opty:.2f})"],
            textposition="top right", name="Optimal Solution", hoverinfo="x+y"
        )
    )

    xminview = None if allownegative else 0
    yminview = None if allownegative else 0

    xaxisconfig = dict(title=xname, range=[xminview, viewxmax], showgrid=True, gridcolor='rgba(200,200,200,0.4)')
    yaxisconfig = dict(title=yname, range=[yminview, viewymax], showgrid=True, gridcolor='rgba(200,200,200,0.4)')

    if not allownegative:
        xaxisconfig.update(dict(rangemode="nonnegative", minallowed=0))
        yaxisconfig.update(dict(rangemode="nonnegative", minallowed=0))

    fig.updatelayout(
        title="",
        xaxis=xaxisconfig,
        yaxis=yaxisconfig,
        template="plotlywhite",
        height=600,
        margin=dict(l=40, r=40, t=20, b=120),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(200,200,200,0.6)", borderwidth=1
        ),
    )

    st.plotlychart(fig, usecontainerwidth=True)


def displayresultsqp(result: dict, sense: str, defaultx: str = None, defaulty: str = None):
    if result["success"]:
        st.success("Optimization Completed Successfully!")

        if not result.get("isconvex", True):
            eigvals = result.get("eigvals", [])
            mineig = float(np.min(eigvals)) if len(eigvals) else 0.0
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
        st.dataframe(dfres.style.format({"Optimal Value": "{:,.4f}"}), usecontainerwidth=True)

        st.subheader("Interactive Objective Contour Map & Feasible Area")
        plotinteractivecontourlinesqp(result, defaultx=defaultx, defaulty=defaulty)
    else:
        st.error(f"Solver Error: {result['message']}")

----------------------------------------------------------------------
PAGE (mirrors page_custom()'s architecture, for quadratic objectives)
----------------------------------------------------------------------
def pagequadratic():
    st.header("QP Problem Statement (Objective Function)")
    st.caption(
        "This solver supports quadratic objective functions (e.g., x**2, 3*x*y, -2*y**2) "
        "combined with linear constraints only. Internally it minimizes/maximizes "
        "0.5·xᵀQx + cᵀx via SciPy's trust-constr solver using the exact Hessian/gradient."
    )

    with st.expander("✨ Auto-parse problem statement using Google AI Studio (Gemini)", expanded=True):
        naturalpromptqp = st.textarea(
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
            if not naturalpromptqp.strip():
                st.warning("Please enter a natural language problem statement.")
            else:
                with st.spinner("Parsing problem with Gemini…"):
                    try:
                        parsedqp = parseqpwithgemini(naturalpromptqp)

                        # Bind parsed output directly to the widget keys
                        st.sessionstate["qpsense"] = parsedqp.sense
                        st.sessionstate["qpobjinput"] = parsedqp.objectivefunction
                        st.sessionstate["qpconstraintsinput"] = "\n".join(parsedqp.constraints)

                        st.success("Successfully parsed problem statement!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to parse via Gemini API: {e}")

    # Set initial default values if key does not exist
    if "qpsense" not in st.sessionstate:
        st.sessionstate["qpsense"] = "Minimize"
    if "qpobjinput" not in st.sessionstate:
        st.sessionstate["qpobjinput"] = "0.02r12 + 3r1 + 0.03r22 + 2r2 + 0.05r32 + r3"
    if "qpconstraintsinput" not in st.sessionstate:
        st.sessionstate["qpconstraintsinput"] = (
            "r1 + r2 + r3 = 150\n"
            "0.9r1 + 0.8r2 + 0.7r3 >= 125\n"
            "2r1 + 3r2 + 4*r3 <= 450\n"
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
        objinputqp = st.textinput(
            "Objective Function",
            key="qpobjinput"
        )

    alltextqp = objinputqp + "\n" + st.sessionstate["qpconstraintsinput"]
    detectedvarsqp = sorted(list(extractidentifiers(alltextqp)))
    isobjquad, objquaderr = checkexpressionquadratic(objinputqp, detectedvarsqp)

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
    constraintsinputqp = st.textarea(
        "Constraints List",
        height=160,
        key="qpconstraintsinput"
    )

    alltextqp = objinputqp + "\n" + constraintsinputqp
    detectedvarsqp = sorted(list(extractidentifiers(alltextqp)))

    nonlinearconstraints = []
    for line in [c.strip() for c in constraintsinputqp.split("\n") if c.strip()]:
        try:
            symdict = {v: sp.Symbol(v) for v in detectedvarsqp}
            diff, _ = parseequationorinequality(line, symdict)
            if diff is not None:
                islinline, linerr = checkexpressionlinearity(str(diff), detectedvarsqp)
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
                    minval = st.numberinput(f"Min ({var})", value=0.0, key=f"qpmin{var}")
                    hasmax = st.checkbox(f"Set Max ({var})", key=f"qphasmax{var}")
                    maxval = st.numberinput(f"Max ({var})", value=100.0, key=f"qpmax{var}") if hasmax else None
                    boundsdictqp[var] = (minval, maxval)
        else:
            st.warning("No variables detected yet to configure bounds.")
    else:
        boundsdictqp = {var: (0.0, None) for var in detectedvarsqp}

    st.divider()

    if detectedvarsqp:
        badgespans = []
        for i, var in enumerate(detectedvarsqp):
            color = VARBADGECOLORS[i % len(VARBADGECOLORS)]
            style = (
                f"background-color: {color['bg']}; color: {color['text']}; "
                f"border: 1px solid {color['border']}; padding: 3px 9px; border-radius: 6px; "
                f"font-weight: 600; font-size: 0.9em; display: inline-block; margin-right: 4px;"
            )
            badgespans.append(f'{var}')

        badgeshtml = " ".join(badgespans)
        st.markdown(f"Recognized Variables: {badgeshtml}", unsafeallowhtml=True)

        with st.expander("Optimization Problem Statement Preview", expanded=True):
            highlightedobj = highlightvariablesintext(objinputqp, detectedvarsqp)
            st.markdown(
                f"Parsed Objective: {senseqp}   {highlighted_obj}",
                unsafeallowhtml=True,
            )

            lines = [c.strip() for c in constraintsinputqp.split("\n") if c.strip()]
            if lines:
                st.markdown("Parsed Constraints:")
                for idx, line in enumerate(lines, 1):
                    hline = highlightvariablesintext(line, detectedvarsqp)
                    st.markdown(f"  C{idx}: {h_line}", unsafeallowhtml=True)
    else:
        st.markdown("No variables detected yet.")

    st.divider()

    solvedisabledqp = (not isobjquad) or bool(nonlinearconstraints)
    if st.button("Solve Quadratic Program", type="primary", disabled=solvedisabledqp, key="qpsolvebtn"):
        constraintslistqp = [c.strip() for c in constraintsinputqp.split("\n") if c.strip()]
        st.sessionstate.resultqp = solveqp(
            objectivestr=objinputqp,
            constraintslist=constraintslistqp,
            sense=senseqp,
            varnames=detectedvarsqp,
            boundsdict=boundsdictqp
        )

    if st.sessionstate.resultqp is not None:
        defaultxqp = detectedvarsqp[0] if len(detectedvarsqp) > 0 else None
        defaultyqp = detectedvarsqp[1] if len(detectedvarsqp) > 1 else None
        displayresultsqp(
            st.sessionstate.resultqp,
            senseqp,
            defaultx=defaultxqp,
            defaulty=defaultyqp
        )

----------------------------------------------------------------------
ENTRY POINT — this runs immediately when app.py exec()'s this file's
text inside the "app2" tab branch.
----------------------------------------------------------------------
if "resultqp" not in st.sessionstate:
    st.sessionstate.resultqp = None

page_quadratic()
