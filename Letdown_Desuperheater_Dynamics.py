import time

from iapws import IAPWS97
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Real-Time Desuperheater Dynamics",
    page_icon="💨",
    layout="wide",
)

st.title("💨 Real-Time Desuperheater Dynamic Simulation by Iqbal SHERPA 20260807")

# --- INITIALIZE SESSION STATE FOR REAL-TIME SIMULATION ---
if "time_history" not in st.session_state:
    st.session_state.time_history = [0.0]
    st.session_state.temp_history = [180.0]
    st.session_state.outlet_flow_history = [100.0]
    st.session_state.spray_flow_history = [2.0]
    st.session_state.inlet_flow_history = [98.0]
    st.session_state.sim_time = 0.0

# --- SIDEBAR INPUTS ---
st.sidebar.header("🕹️ Simulation Real-Time Controls")
is_running = st.sidebar.toggle("Run Live Simulation", value=True)
tau = st.sidebar.slider(
    "Response Time Constant τ (s)", 1.0, 30.0, 8.0, help="Higher = slower response"
)
dt = st.sidebar.slider("Step Delay Δt (s)", 0.1, 1.0, 0.2)

if st.sidebar.button("Reset Dynamic Trends"):
    st.session_state.time_history = [0.0]
    st.session_state.temp_history = [180.0]
    st.session_state.outlet_flow_history = [100.0]
    st.session_state.spray_flow_history = [2.0]
    st.session_state.inlet_flow_history = [98.0]
    st.session_state.sim_time = 0.0

st.sidebar.header("1. Operating Configuration")
Pressure_Unit_Type = st.sidebar.selectbox(
    "Pressure Unit Type",
    [
        "Bar Gauge (barG)",
        "Bar Absolute (barA)",
        "Megapascals Gauge (MPaG)",
        "Megapascals Absolute (MPaA)",
    ],
)

High_Pressure_Inlet_Steam_Pressure = st.sidebar.number_input(
    "Inlet Pressure", value=50.0
)
High_Pressure_Inlet_Steam_Temperature_Degrees_Celsius = (
    st.sidebar.number_input("Inlet Temp (°C)", value=419.0)
)

Outlet_Temperature_Calculation_Mode = st.sidebar.radio(
    "Calculation Mode",
    [
        "INPUT - Specify Target Outlet Temperature",
        "CALC - Calculate Outlet Temperature from Spray Flow",
    ],
)
is_calc_mode = (
    Outlet_Temperature_Calculation_Mode
    == "CALC - Calculate Outlet Temperature from Spray Flow"
)

Desuperheater_Outlet_Steam_Pressure = st.sidebar.number_input(
    "Outlet Pressure", value=4.6
)
Desuperheater_Outlet_Steam_Target_Temperature_Degrees_Celsius = (
    st.sidebar.number_input(
        "Target Outlet Temp (°C)", value=160.0, disabled=is_calc_mode
    )
)

Spray_Feedwater_Inlet_Pressure = st.sidebar.number_input(
    "Feedwater Pressure", value=70.0
)
Spray_Feedwater_Inlet_Temperature_Degrees_Celsius = st.sidebar.number_input(
    "Feedwater Temp (°C)", value=90.0
)

Specified_Spray_Feedwater_Mass_Flow_Rate_Tons_Per_Hour = (
    st.sidebar.number_input(
        "Specified Spray Flow (t/h)", value=2.35, disabled=not is_calc_mode
    )
)

Mass_Flow_Rate_Basis = st.sidebar.selectbox(
    "Basis", ["Inlet Steam Flow Rate", "Outlet Target Steam Flow Rate"]
)
Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour = st.sidebar.number_input(
    "Specified Steam Flow (t/h)", value=107.0
)

# --- STEADY-STATE TARGET CALCULATION ---
ATMOSPHERIC_PRESSURE_BAR = 1.01325
ATMOSPHERIC_PRESSURE_MEGAPASCALS = 0.101325

if Pressure_Unit_Type == "Bar Gauge (barG)":
    p_in_mpaa = (
        High_Pressure_Inlet_Steam_Pressure + ATMOSPHERIC_PRESSURE_BAR
    ) / 10.0
    p_out_mpaa = (
        Desuperheater_Outlet_Steam_Pressure + ATMOSPHERIC_PRESSURE_BAR
    ) / 10.0
    p_fw_mpaa = (
        Spray_Feedwater_Inlet_Pressure + ATMOSPHERIC_PRESSURE_BAR
    ) / 10.0
elif Pressure_Unit_Type == "Bar Absolute (barA)":
    p_in_mpaa = High_Pressure_Inlet_Steam_Pressure / 10.0
    p_out_mpaa = Desuperheater_Outlet_Steam_Pressure / 10.0
    p_fw_mpaa = Spray_Feedwater_Inlet_Pressure / 10.0
elif Pressure_Unit_Type == "Megapascals Gauge (MPaG)":
    p_in_mpaa = (
        High_Pressure_Inlet_Steam_Pressure + ATMOSPHERIC_PRESSURE_MEGAPASCALS
    )
    p_out_mpaa = (
        Desuperheater_Outlet_Steam_Pressure + ATMOSPHERIC_PRESSURE_MEGAPASCALS
    )
    p_fw_mpaa = (
        Spray_Feedwater_Inlet_Pressure + ATMOSPHERIC_PRESSURE_MEGAPASCALS
    )
else:
    p_in_mpaa = High_Pressure_Inlet_Steam_Pressure
    p_out_mpaa = Desuperheater_Outlet_Steam_Pressure
    p_fw_mpaa = Spray_Feedwater_Inlet_Pressure

# Enthalpies via IAPWS97
enthalpy_steam_inlet = IAPWS97(
    P=p_in_mpaa,
    T=High_Pressure_Inlet_Steam_Temperature_Degrees_Celsius + 273.15,
).h
enthalpy_feedwater_inlet = IAPWS97(
    P=p_fw_mpaa,
    T=Spray_Feedwater_Inlet_Temperature_Degrees_Celsius + 273.15,
).h

if is_calc_mode:
    target_fw_flow = Specified_Spray_Feedwater_Mass_Flow_Rate_Tons_Per_Hour
    if Mass_Flow_Rate_Basis == "Inlet Steam Flow Rate":
        target_inlet_flow = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        target_outlet_flow = target_inlet_flow + target_fw_flow
    else:
        target_outlet_flow = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        target_inlet_flow = target_outlet_flow - target_fw_flow

    enthalpy_steam_outlet = (
        (target_inlet_flow * enthalpy_steam_inlet)
        + (target_fw_flow * enthalpy_feedwater_inlet)
    ) / target_outlet_flow
    target_temp_outlet = (
        IAPWS97(P=p_out_mpaa, h=enthalpy_steam_outlet).T - 273.15
    )
else:
    target_temp_outlet = (
        Desuperheater_Outlet_Steam_Target_Temperature_Degrees_Celsius
    )
    enthalpy_steam_outlet = IAPWS97(
        P=p_out_mpaa, T=target_temp_outlet + 273.15
    ).h

    if Mass_Flow_Rate_Basis == "Inlet Steam Flow Rate":
        target_inlet_flow = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        target_fw_flow = (
            target_inlet_flow
            * (enthalpy_steam_outlet - enthalpy_steam_inlet)
            / (enthalpy_feedwater_inlet - enthalpy_steam_outlet)
        )
        target_outlet_flow = target_inlet_flow + target_fw_flow
    else:
        target_outlet_flow = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        target_inlet_flow = (
            target_outlet_flow
            * (enthalpy_steam_outlet - enthalpy_feedwater_inlet)
            / (enthalpy_steam_inlet - enthalpy_feedwater_inlet)
        )
        target_fw_flow = target_outlet_flow - target_inlet_flow

saturation_temp = IAPWS97(P=p_out_mpaa, x=0).T - 273.15

# --- NUMERICAL INTEGRATION STEP (FIRST-ORDER LAG) ---
if is_running:
    st.session_state.sim_time += dt

    # Euler integration step for single-time-constant process dynamics:
    # dy/dt = (y_target - y_current) / tau
    curr_temp = st.session_state.temp_history[-1]
    curr_inlet_flow = st.session_state.inlet_flow_history[-1]
    curr_spray_flow = st.session_state.spray_flow_history[-1]

    new_temp = curr_temp + (dt / tau) * (target_temp_outlet - curr_temp)
    new_inlet_flow = curr_inlet_flow + (dt / tau) * (
        target_inlet_flow - curr_inlet_flow
    )
    new_spray_flow = curr_spray_flow + (dt / tau) * (
        target_fw_flow - curr_spray_flow
    )
    new_outlet_flow = new_inlet_flow + new_spray_flow

    # Append to rolling history buffers (keep last 300 points)
    st.session_state.time_history.append(st.session_state.sim_time)
    st.session_state.temp_history.append(new_temp)
    st.session_state.inlet_flow_history.append(new_inlet_flow)
    st.session_state.spray_flow_history.append(new_spray_flow)
    st.session_state.outlet_flow_history.append(new_outlet_flow)

    if len(st.session_state.time_history) > 300:
        st.session_state.time_history.pop(0)
        st.session_state.temp_history.pop(0)
        st.session_state.inlet_flow_history.pop(0)
        st.session_state.spray_flow_history.pop(0)
        st.session_state.outlet_flow_history.pop(0)

# --- REAL-TIME DISPLAY & METRICS ---
c1, c2, c3 = st.columns(3)
c1.metric(
    "Current / Target Temp",
    f"{st.session_state.temp_history[-1]:.1f} °C",
    f"Target: {target_temp_outlet:.1f} °C",
)
c2.metric(
    "Current / Target Spray Flow",
    f"{st.session_state.spray_flow_history[-1]:.2f} t/h",
    f"Target: {target_fw_flow:.2f} t/h",
)
c3.metric(
    "Current Outlet Flow",
    f"{st.session_state.outlet_flow_history[-1]:.2f} t/h",
    f"Target: {target_outlet_flow:.2f} t/h",
)

# --- RENDER DYNAMIC PLOTS ---
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(
        "Live Temperature Response (°C)",
        "Live Mass Flow Rates Response (t/h)",
    ),
)

fig.add_trace(
    go.Scatter(
        x=st.session_state.time_history,
        y=st.session_state.temp_history,
        mode="lines",
        name="Outlet Temp (°C)",
        line=dict(color="#00A896", width=2.5),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=st.session_state.time_history,
        y=[saturation_temp] * len(st.session_state.time_history),
        mode="lines",
        name="Saturation Limit (°C)",
        line=dict(color="#E63946", dash="dash"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=st.session_state.time_history,
        y=st.session_state.outlet_flow_history,
        mode="lines",
        name="Outlet Steam Flow (t/h)",
        line=dict(color="#1D3557", width=2),
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=st.session_state.time_history,
        y=st.session_state.spray_flow_history,
        mode="lines",
        name="Spray Water Flow (t/h)",
        line=dict(color="#457B9D", width=2),
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=500,
    template="plotly_white",
    font=dict(family="Segoe UI, Aptos, Arial", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)

fig.update_xaxes(title_text="Simulation Time (s)", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# --- STREAMLIT RE-RUN LOOP ---
if is_running:
    time.sleep(dt)
    st.rerun()
