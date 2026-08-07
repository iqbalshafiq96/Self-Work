import time

from iapws import IAPWS97
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Real-Time Desuperheater Lead-Lag Dynamics",
    page_icon="💨",
    layout="wide",
)

st.title(
    "💨 Real-Time Desuperheater Dynamic Simulation (Lead-Lag Enthalpy Model) by Iqbal SHERPA 20260807"
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Dynamic & Lead-Lag Controls")
is_running = st.sidebar.toggle("Run Live Simulation", value=True)

tau_steam = st.sidebar.slider(
    "Steam Flow Time Constant τ_steam (s)",
    1.0,
    30.0,
    5.0,
    help="Speed of steam flow response to step changes.",
)

tau_spray = st.sidebar.slider(
    "Feedwater Flow Time Constant τ_spray (s)",
    1.0,
    30.0,
    12.0,
    help="Speed of spray water valve response. Set τ_spray > τ_steam to see temperature overshoot!",
)

tau_thermal = st.sidebar.slider(
    "Thermal Mixing/Sensor Lag τ_thermal (s)",
    0.5,
    10.0,
    2.0,
    help="Thermal inertia/mixing lag inside the pipe line.",
)

dt = st.sidebar.slider("Step Delay Δt (s)", 0.1, 1.0, 0.2)

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


# --- INITIALIZE / RESET SESSION STATE AT STEADY STATE ---
def reset_to_steady_state():
    st.session_state.time_history = [0.0]
    st.session_state.temp_history = [target_temp_outlet]
    st.session_state.outlet_flow_history = [target_outlet_flow]
    st.session_state.spray_flow_history = [target_fw_flow]
    st.session_state.inlet_flow_history = [target_inlet_flow]
    st.session_state.sim_time = 0.0


if "time_history" not in st.session_state:
    reset_to_steady_state()

if st.sidebar.button("Reset Dynamic Trends"):
    reset_to_steady_state()

# --- NUMERICAL INTEGRATION WITH LEAD-LAG ENTHALPY BALANCE ---
if is_running:
    st.session_state.sim_time += dt

    curr_inlet_flow = st.session_state.inlet_flow_history[-1]
    curr_spray_flow = st.session_state.spray_flow_history[-1]
    curr_temp = st.session_state.temp_history[-1]

    # 1. Independent first-order lag updates for flows
    new_inlet_flow = curr_inlet_flow + (dt / tau_steam) * (
        target_inlet_flow - curr_inlet_flow
    )
    new_spray_flow = curr_spray_flow + (dt / tau_spray) * (
        target_fw_flow - curr_spray_flow
    )
    new_outlet_flow = new_inlet_flow + new_spray_flow

    # 2. Instantaneous Enthalpy Balance from dynamic transient flow rates
    if new_outlet_flow > 0:
        dynamic_outlet_enthalpy = (
            (new_inlet_flow * enthalpy_steam_inlet)
            + (new_spray_flow * enthalpy_feedwater_inlet)
        ) / new_outlet_flow
        instantaneous_temp = (
            IAPWS97(P=p_out_mpaa, h=dynamic_outlet_enthalpy).T - 273.15
        )
    else:
        instantaneous_temp = curr_temp

    # 3. Apply thermal lag to temperature reading (pipe wall/sensor thermal mass)
    new_temp = curr_temp + (dt / tau_thermal) * (
        instantaneous_temp - curr_temp
    )

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

# Lead-Lag Ratio Indicator Banner
lead_lag_ratio = tau_spray / tau_steam
if lead_lag_ratio > 1.2:
    st.warning(
        f"⚠️ Water Lags Steam (τ_spray/τ_steam = {lead_lag_ratio:.2f}):"
        " Expect transient temperature OVERSHOOT (upshoot) during flow"
        " increases!"
    )
elif lead_lag_ratio < 0.8:
    st.info(
        f"ℹ️ Water Leads Steam (τ_spray/τ_steam = {lead_lag_ratio:.2f}):"
        " Expect transient temperature UNDERSHOOT (dip) during flow increases."
    )

# --- RENDER DYNAMIC PLOTS ---
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "Live Temperature Transient Response (°C)",
        "Live Outlet Steam Mass Flow Rate (t/h)",
        "Live Spray Feedwater Mass Flow Rate (t/h)",
    ),
)

# Row 1: Outlet Temperature
fig.add_trace(
    go.Scatter(
        x=st.session_state.time_history,
        y=st.session_state.temp_history,
        mode="lines",
        name="Outlet Temp (°C)",
        line=dict(color="#008080", width=2.5),  # Teal
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
        line=dict(color="#E63946", dash="dash"),  # Crimson
    ),
    row=1,
    col=1,
)

# Row 2: Outlet Steam Flow
fig.add_trace(
    go.Scatter(
        x=st.session_state.time_history,
        y=st.session_state.outlet_flow_history,
        mode="lines",
        name="Outlet Steam Flow (t/h)",
        line=dict(color="#4169E1", width=2.5),  # Royal Blue
    ),
    row=2,
    col=1,
)

# Row 3: Spray Water Flow
fig.add_trace(
    go.Scatter(
        x=st.session_state.time_history,
        y=st.session_state.spray_flow_history,
        mode="lines",
        name="Spray Water Flow (t/h)",
        line=dict(color="#E67E22", width=2.5),  # Deep Orange
    ),
    row=3,
    col=1,
)

fig.update_layout(
    height=680,
    template="plotly_white",
    font=dict(family="Segoe UI, Aptos, Arial", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)

fig.update_xaxes(title_text="Simulation Time (s)", row=3, col=1)
st.plotly_chart(fig, use_container_width=True)

# --- STREAMLIT RE-RUN LOOP ---
if is_running:
    time.sleep(dt)
    st.rerun()
