import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Steam Letdown System",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1117;
    color: #e6edf3;
}

/* Main title */
.main-title {
    font-size: 30px;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 0px;
}

.subtitle {
    color: #8b949e;
    font-size: 14px;
    margin-bottom: 20px;
}

/* KPI cards */
.kpi {
    background: linear-gradient(145deg, #111a23, #0d141c);
    border: 1px solid #263341;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}

.kpi-title {
    color: #8b949e;
    font-size: 12px;
    text-transform: uppercase;
}

.kpi-value {
    color: #58a6ff;
    font-size: 27px;
    font-weight: 700;
}

.kpi-unit {
    color: #8b949e;
    font-size: 12px;
}

/* Process equipment */
.equipment {
    background-color: #121b24;
    border: 1px solid #30404f;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

.equipment-title {
    color: #f0f6fc;
    font-weight: 700;
}

.status-running {
    color: #3fb950;
    font-weight: 700;
}

.status-warning {
    color: #d29922;
    font-weight: 700;
}

.status-alarm {
    color: #f85149;
    font-weight: 700;
}

/* Section */
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #f0f6fc;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* Pipe */
.pipe {
    height: 8px;
    background: #4ea1ff;
    border-radius: 4px;
    margin-top: 50px;
}

/* Flow arrow */
.arrow {
    font-size: 35px;
    color: #58a6ff;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">STEAM LETDOWN & DESUPERHEATING SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-time thermodynamic monitoring & control simulation</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT / CONTROL PANEL
# ============================================================

with st.sidebar:

    st.header("⚙️ OPERATING INPUTS")

    st.subheader("HP Steam")

    steam_flow = st.number_input(
        "Steam Flow (tph)",
        min_value=1.0,
        max_value=200.0,
        value=107.0,
        step=1.0
    )

    inlet_pressure = st.number_input(
        "Inlet Pressure (barg)",
        min_value=1.0,
        max_value=100.0,
        value=50.0,
        step=0.5
    )

    inlet_temp = st.number_input(
        "Inlet Temperature (°C)",
        min_value=100.0,
        max_value=600.0,
        value=419.0,
        step=1.0
    )

    st.divider()

    st.subheader("Letdown Valve")

    valve_opening = st.slider(
        "Valve Opening (%)",
        min_value=0,
        max_value=100,
        value=35
    )

    st.divider()

    st.subheader("Spray Water")

    spray_temp = st.number_input(
        "Spray Water Temperature (°C)",
        min_value=20.0,
        max_value=200.0,
        value=90.0
    )

    spray_flow = st.slider(
        "Spray Water Flow (tph)",
        min_value=0.0,
        max_value=15.0,
        value=3.0,
        step=0.1
    )


# ============================================================
# SIMPLE PROCESS MODEL
# ============================================================

# Approximate pressure drop model
# Higher valve opening → higher downstream pressure

outlet_pressure = 1.0 + (
    inlet_pressure - 1.0
) * (valve_opening / 100) ** 0.55

# Approximate desuperheating effect
# More water → lower outlet temperature

base_temp = inlet_temp - 120 * (valve_opening / 100)

temp_reduction = spray_flow * 18

outlet_temp = base_temp - temp_reduction

# Prevent unrealistic values
outlet_temp = max(outlet_temp, 120)

# Simple velocity / flow indication
steam_flow_m3h = steam_flow * 20


# ============================================================
# STATUS
# ============================================================

if outlet_temp > 200:
    status = "HIGH TEMPERATURE"
    status_class = "status-warning"
elif outlet_pressure < 2:
    status = "LOW PRESSURE"
    status_class = "status-warning"
else:
    status = "NORMAL"
    status_class = "status-running"


# ============================================================
# KPI ROW
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">HP Steam Flow</div>
            <div class="kpi-value">{steam_flow:.1f}</div>
            <div class="kpi-unit">TPH</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">Valve Position</div>
            <div class="kpi-value">{valve_opening}%</div>
            <div class="kpi-unit">CONTROL VALVE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">LP Pressure</div>
            <div class="kpi-value">{outlet_pressure:.2f}</div>
            <div class="kpi-unit">BARG</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">LP Temperature</div>
            <div class="kpi-value">{outlet_temp:.1f}</div>
            <div class="kpi-unit">°C</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">System Status</div>
            <div class="kpi-value" style="font-size:20px">
                {status}
            </div>
            <div class="kpi-unit">LETdown SYSTEM</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PROCESS GRAPHIC
# ============================================================

st.markdown(
    '<div class="section-title">PROCESS OVERVIEW</div>',
    unsafe_allow_html=True
)


# Use columns to construct process graphic
p1, p2, p3, p4, p5 = st.columns(
    [1.4, 0.35, 1.5, 0.35, 1.8]
)


# ------------------------------------------------------------
# HP STEAM
# ------------------------------------------------------------

with p1:

    st.markdown("""
    <div class="equipment">

    <div class="equipment-title">
    🔥 HP STEAM
    </div>

    <hr>

    <b>50.0 barg</b><br>
    <span style="color:#58a6ff;font-size:22px">
    419 °C
    </span>

    <br><br>

    Flow<br>
    <b>107.0 TPH</b>

    <br><br>

    <span class="status-running">
    ● AVAILABLE
    </span>

    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# ARROW
# ------------------------------------------------------------

with p2:

    st.markdown(
        '<div class="arrow">➜</div>',
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# CONTROL VALVE
# ------------------------------------------------------------

with p3:

    st.markdown(f"""
    <div class="equipment">

    <div class="equipment-title">
    CONTROL VALVE
    </div>

    <br>

    <div style="
        width:120px;
        height:65px;
        margin:auto;
        border-top:5px solid #58a6ff;
        border-bottom:5px solid #58a6ff;
        position:relative;
    ">

        <div style="
            position:absolute;
            left:45%;
            top:5px;
            width:0;
            height:0;
            border-left:12px solid transparent;
            border-right:12px solid transparent;
            border-top:25px solid #d0d7de;
        ">
        </div>

        <div style="
            position:absolute;
            left:45%;
            bottom:5px;
            width:0;
            height:0;
            border-left:12px solid transparent;
            border-right:12px solid transparent;
            border-bottom:25px solid #d0d7de;
        ">
        </div>

    </div>

    <br>

    <b>{valve_opening}% OPEN</b>

    <br><br>

    <span style="color:#58a6ff">
    ΔP = {inlet_pressure-outlet_pressure:.1f} bar
    </span>

    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# ARROW
# ------------------------------------------------------------

with p4:

    st.markdown(
        '<div class="arrow">➜</div>',
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# DESUPERHEATER
# ------------------------------------------------------------

with p5:

    st.markdown(f"""
    <div class="equipment">

    <div class="equipment-title">
    💧 SPRAY DESUPERHEATER
    </div>

    <br>

    <div style="
        border:3px solid #58a6ff;
        border-radius:50%;
        width:100px;
        height:100px;
        margin:auto;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:30px;
    ">
    💧
    </div>

    <br>

    Spray Water<br>

    <b>{spray_flow:.1f} TPH</b>

    <br><br>

    Outlet<br>

    <b>{outlet_pressure:.2f} barg</b><br>

    <span style="color:#58a6ff;font-size:22px">
    {outlet_temp:.1f} °C
    </span>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SPRAY WATER LINE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

w1, w2, w3 = st.columns([3, 1, 3])

with w1:
    st.markdown(
        """
        <div style="
        text-align:right;
        color:#58a6ff;
        font-weight:bold;
        ">
        💧 SPRAY WATER<br>
        70 barg / 90°C
        </div>
        """,
        unsafe_allow_html=True
    )

with w2:
    st.markdown(
        """
        <div style="
        text-align:center;
        color:#58a6ff;
        font-size:28px;
        ">
        ↓
        </div>
        """,
        unsafe_allow_html=True
    )

with w3:
    st.markdown(
        f"""
        <div style="
        text-align:left;
        color:#58a6ff;
        font-weight:bold;
        ">
        INJECTION<br>
        {spray_flow:.1f} TPH
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CONTROL LOOP
# ============================================================

st.markdown(
    '<div class="section-title">CONTROL LOOP</div>',
    unsafe_allow_html=True
)

ctrl1, ctrl2, ctrl3 = st.columns(3)

with ctrl1:

    st.metric(
        "PIC — Outlet Pressure",
        f"{outlet_pressure:.2f} barg"
    )

with ctrl2:

    st.metric(
        "TIC — Outlet Temperature",
        f"{outlet_temp:.1f} °C"
    )

with ctrl3:

    st.metric(
        "Spray Water Flow",
        f"{spray_flow:.1f} TPH"
    )


# ============================================================
# TREND
# ============================================================

st.markdown(
    '<div class="section-title">PROCESS TREND</div>',
    unsafe_allow_html=True
)

time = np.arange(0, 60)

pressure_trend = (
    outlet_pressure
    + 0.15 * np.sin(time / 5)
)

temperature_trend = (
    outlet_temp
    + 3 * np.sin(time / 6)
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=time,
        y=pressure_trend,
        name="Pressure",
        mode="lines"
    )
)

fig.add_trace(
    go.Scatter(
        x=time,
        y=temperature_trend,
        name="Temperature",
        mode="lines",
        yaxis="y2"
    )
)

fig.update_layout(

    height=350,

    template="plotly_dark",

    paper_bgcolor="#0b1117",

    plot_bgcolor="#0b1117",

    xaxis=dict(
        title="Time (min)",
        gridcolor="#263341"
    ),

    yaxis=dict(
        title="Pressure (barg)",
        gridcolor="#263341"
    ),

    yaxis2=dict(
        title="Temperature (°C)",
        overlaying="y",
        side="right"
    ),

    legend=dict(
        orientation="h",
        y=1.1
    ),

    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)
