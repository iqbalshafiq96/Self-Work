import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# -----------------------------------------------------------------------------
# Color Palette for Constraints
# -----------------------------------------------------------------------------
CONSTRAINTCOLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# -----------------------------------------------------------------------------
# Feasible Region Polygon Utility
# -----------------------------------------------------------------------------
def computefeasiblepolygonvertices(halfplanes, boundsx=(0, 100), boundsy=(0, 100)):
    """
    Computes vertices of the 2D feasible region bounded by linear constraints
    (a*x + b*y <= c) and bounding boxes.
    """
    xmin, xmax = boundsx
    ymin, ymax = boundsy

    # Define bounding box as half-planes (a, b, c) -> a*x + b*y <= c
    allplanes = list(halfplanes)
    allplanes.extend([
        (-1.0, 0.0, -xmin),
        (1.0, 0.0, xmax),
        (0.0, -1.0, -ymin),
        (0.0, 1.0, ymax)
    ])

    intersections = []
    numplanes = len(allplanes)

    for i in range(numplanes):
        a1, b1, c1 = allplanes[i]
        for j in range(i + 1, numplanes):
            a2, b2, c2 = allplanes[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-9:
                continue

            xi = (c1 * b2 - c2 * b1) / det
            yi = (a1 * c2 - a2 * c1) / det

            # Verify if point satisfies all half-planes
            feasible = True
            for a, b, c in allplanes:
                if a * xi + b * yi > c + 1e-6:
                    feasible = False
                    break

            if feasible:
                intersections.append((xi, yi))

    if len(intersections) < 3:
        return None, None

    pts = np.array(intersections)
    try:
        hull = ConvexHull(pts)
        hullpts = pts[hull.vertices]
        return hullpts[:, 0], hullpts[:, 1]
    except Exception:
        return None, None


# -----------------------------------------------------------------------------
# Quadratic Programming Solver
# -----------------------------------------------------------------------------
def solveqp(objstr: str, constrlist: list, boundsdict: dict):
    """
    Parses and solves a Quadratic Programming problem using SymPy and Scipy.
    """
    allstrs = [objstr] + constrlist
    allvars = set()

    for s in allstrs:
        parsed = sp.sympify(s)
        allvars.update(parsed.free_symbols)

    symvars = sorted(list(allvars), key=lambda x: x.name)
    varnames = [v.name for v in symvars]
    n = len(symvars)

    objexpr = sp.sympify(objstr)

    # Calculate Hessian (Q) and Gradient vector (c)
    Q = np.zeros((n, n))
    c_vec = np.zeros(n)

    for i in range(n):
        for j in range(n):
            Q[i, j] = float(sp.diff(sp.diff(objexpr, symvars[i]), symvars[j]))
        # Constant factor evaluation for linear terms
        grad_i = sp.diff(objexpr, symvars[i])
        c_vec[i] = float(grad_i.subs({v: 0 for v in symvars}))

    def qpobj(x):
        return 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(c_vec, x)

    def qpgrad(x):
        return np.dot(Q, x) + c_vec

    Aub, bub = [], []
    rawconstraints = []

    for cstr in constrlist:
        if "<=" in cstr:
            lhs, rhs = cstr.split("<=")
            expr = sp.sympify(lhs) - sp.sympify(rhs)
        elif ">=" in cstr:
            lhs, rhs = cstr.split(">=")
            expr = sp.sympify(rhs) - sp.sympify(lhs)
        else:
            continue

        rawconstraints.append(cstr)
        a_row = [float(sp.diff(expr, v)) for v in symvars]
        b_val = float(-expr.subs({v: 0 for v in symvars}))
        Aub.append(a_row)
        bub.append(b_val)

    bounds = [boundsdict.get(vn, (0, None)) for vn in varnames]
    x0 = np.zeros(n)

    constraints = []
    if Aub:
        Aub_arr = np.array(Aub)
        bub_arr = np.array(bub)
        constraints.append({'type': 'ineq', 'fun': lambda x: bub_arr - np.dot(Aub_arr, x)})

    res = minimize(qpobj, x0, jac=qpgrad, bounds=bounds, constraints=constraints, method='SLSQP')

    return {
        "status": res.status,
        "message": res.message,
        "x": dict(zip(varnames, res.x)),
        "fun": res.fun,
        "varnames": varnames,
        "symvars": symvars,
        "objexpr": objexpr,
        "Q": Q,
        "c_vec": c_vec,
        "Aub": Aub,
        "bub": bub,
        "bounds": bounds,
        "rawconstraints": rawconstraints,
        "uborigidx": list(range(len(Aub)))
    }


# -----------------------------------------------------------------------------
# Display Results
# -----------------------------------------------------------------------------
def displayresultsqp(result: dict):
    if result["status"] == 0:
        st.success("Optimal solution found successfully!")
        st.metric("Optimal Objective Value", f"{result['fun']:,.4f}")

        st.markdown("### Decision Variables")
        cols = st.columns(len(result["x"]))
        for idx, (vn, val) in enumerate(result["x"].items()):
            cols[idx].metric(vn, f"{val:,.4f}")

        # Matrix diagnostics
        mineig = np.min(np.linalg.eigvals(result["Q"]))
        if mineig > 1e-6:
            st.caption(" Positive Definite Hessian (Strictly Convex Objective)")
        elif mineig >= -1e-6:
            st.caption(" Positive Semi-Definite Hessian (Convex Objective)")
        else:
            st.caption(" Non-Convex Hessian Detected")
    else:
        st.error(f"Optimization failed: {result['message']}")


# -----------------------------------------------------------------------------
# Plotting Function (Fixed)
# -----------------------------------------------------------------------------
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

    st.markdown("### 2D Projection Settings")
    colx, coly, coln = st.columns(3)

    with colx:
        xname = st.selectbox("X-Axis Variable", varnames, index=defaultxidx, key="contourxvarqp")

    yoptions = [v for v in varnames if v != xname]
    defaultyidx = yoptions.index(defaulty) if defaulty in yoptions else 0

    with coly:
        yname = st.selectbox("Y-Axis Variable", yoptions, index=defaultyidx, key="contouryvarqp")

    with coln:
        ncontours = st.number_input(
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

    xmincalc = -calcxmax if allownegative else 0
    ymincalc = -calcymax if allownegative else 0

    xvals = np.linspace(xmincalc, calcxmax, 250)
    yvals = np.linspace(ymincalc, calcymax, 250)
    X, Y = np.meshgrid(xvals, yvals)

    Z = np.asarray(zfunc(X, Y), dtype=float)
    if Z.shape != X.shape:
        Z = np.full_like(X, float(Z))

    contourlinecolor = '#1f77b4'

    fig.add_trace(
        go.Contour(
            x=xvals, y=yvals, z=Z,
            contours_coloring="lines",
            ncontours=int(ncontours),
            contours=dict(showlabels=True, labelfont=dict(size=10, color='navy')),
            line=dict(color=contourlinecolor, width=1.5, dash='dash'),
            showscale=False, showlegend=False, hoverinfo="x+y+z"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=contourlinecolor, width=1.5, dash='dash'),
            name="Objective Contour", showlegend=True
        )
    )

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
            constrlabel = rawconstraints[origi] if origi < len(rawconstraints) else f"Constraint {origi + 1}"

            if abs(ay) > 1e-6:
                yline = (effb - ax * xvals) / ay
                fig.add_trace(
                    go.Scatter(
                        x=xvals, y=yline, mode='lines',
                        line=dict(color=linecolor, width=2),
                        name=f"C{origi + 1}: {constrlabel}", hoverinfo="x+y"
                    )
                )
            elif abs(ax) > 1e-6:
                xval = effb / ax
                fig.add_trace(
                    go.Scatter(
                        x=[xval, xval], y=[ymincalc, calcymax], mode='lines',
                        line=dict(color=linecolor, width=2),
                        name=f"C{origi + 1}: {constrlabel}", hoverinfo="x+y"
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

    xminview = None if allownegative else 0
    yminview = None if allownegative else 0

    xaxisconfig = dict(title=xname, range=[xminview, viewxmax], showgrid=True, gridcolor='rgba(200,200,200,0.4)')
    yaxisconfig = dict(title=yname, range=[yminview, viewymax], showgrid=True, gridcolor='rgba(200,200,200,0.4)')

    if not allownegative:
        xaxisconfig.update(dict(rangemode="nonnegative", minallowed=0))
        yaxisconfig.update(dict(rangemode="nonnegative", minallowed=0))

    fig.update_layout(
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

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Streamlit Page UI
# -----------------------------------------------------------------------------
def pagequadratic():
    st.title("Quadratic Programming Solver")
    st.markdown("Solve optimization problems with a quadratic objective function and linear constraints.")

    default_obj = "2*x1**2 + x2**2 + x1*x2 - 8*x1 - 6*x2"
    default_constrs = "x1 + x2 <= 4\nx1 - 2*x2 <= 2"

    objinput = st.text_input("Objective Function (Minimize)", value=default_obj)
    constrinput = st.text_area("Linear Constraints (one per line)", value=default_constrs)

    constrs = [c.strip() for c in constrinput.split("\n") if c.strip()]

    if st.button("Solve QP Problem", type="primary"):
        try:
            res = solveqp(objinput, constrs, boundsdict={})
            st.session_state["qp_result"] = res
        except Exception as e:
            st.error(f"Error solving QP problem: {e}")

    if "qp_result" in st.session_state:
        res = st.session_state["qp_result"]
        displayresultsqp(res)
        st.markdown("---")
        plotinteractivecontourlinesqp(res)


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="QP Optimization", layout="wide")
    pagequadratic()
