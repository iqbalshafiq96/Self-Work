import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from iapws import IAPWS97

st.set_page_config(
    page_title="Desuperheater Dynamic Model", page_icon="💨", layout="wide"
)

st.title(
    "💨 Desuperheater Letdown Dynamic Response Model by Iqbal SHERPA _20260708"
)

# --- SIDEBAR INPUTS ---
st.sidebar.header("Dynamic Response Settings")
simulation_time = st.sidebar.slider("Simulation Time (seconds)", 30, 300, 120)
time_constant = st.sidebar.slider(
    "Thermal Time Constant τ (seconds)", 1, 30, 10
)

st.sidebar.header("Configuration")
Pressure_Unit_Type = st.sidebar.selectbox(
    "Pressure Unit Type",
    [
        "Bar Gauge (barG)",
        "Bar Absolute (barA)",
        "Megapascals Gauge (MPaG)",
        "Megapascals Absolute (MPaA)",
    ],
)

st.sidebar.header("1. High-Pressure Inlet Steam")
High_Pressure_Inlet_Steam_Pressure = st.sidebar.number_input(
    "Inlet Pressure", value=50.0
)
High_Pressure_Inlet_Steam_Temperature_Degrees_Celsius = (
    st.sidebar.number_input("Inlet Temp (°C)", value=419.0)
)

st.sidebar.header("2. Desuperheater Outlet Parameters & Mode")
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

st.sidebar.header("3. Spray Feedwater Parameters")
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

st.sidebar.header("4. Flow Rate Basis")
Mass_Flow_Rate_Basis = st.sidebar.selectbox(
    "Basis", ["Inlet Steam Flow Rate", "Outlet Target Steam Flow Rate"]
)
Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour = st.sidebar.number_input(
    "Specified Steam Flow (t/h)", value=107.0
)

# Initial Baseline Baseline / Target Values
st.sidebar.header("5. Baseline / Step Initial Values")
initial_temp = st.sidebar.number_input(
    "Initial Outlet Temp (°C)", value=180.0, disabled=is_calc_mode
)
initial_spray = st.sidebar.number_input(
    "Initial Spray Flow (t/h)", value=1.0, disabled=not is_calc_mode
)
initial_steam_flow = st.sidebar.number_input(
    "Initial Steam Flow (t/h)", value=90.0
)

# --- STEADY-STATE CALCULATION LOGIC ---
ATMOSPHERIC_PRESSURE_MEGAPASCALS = 0.101325
ATMOSPHERIC_PRESSURE_BAR = 1.01325

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

# Pressure Display Conversions
p_in_bara, p_in_barg = p_in_mpaa * 10.0, (
    p_in_mpaa * 10.0
) - ATMOSPHERIC_PRESSURE_BAR
p_out_bara, p_out_barg = p_out_mpaa * 10.0, (
    p_out_mpaa * 10.0
) - ATMOSPHERIC_PRESSURE_BAR
p_fw_bara, p_fw_barg = p_fw_mpaa * 10.0, (
    p_fw_mpaa * 10.0
) - ATMOSPHERIC_PRESSURE_BAR

temperature_steam_inlet = High_Pressure_Inlet_Steam_Temperature_Degrees_Celsius
temperature_feedwater_inlet = (
    Spray_Feedwater_Inlet_Temperature_Degrees_Celsius
)

enthalpy_steam_inlet = IAPWS97(
    P=p_in_mpaa, T=temperature_steam_inlet + 273.15
).h
enthalpy_feedwater_inlet = IAPWS97(
    P=p_fw_mpaa, T=temperature_feedwater_inlet + 273.15
).h

# Calculate Steady-State Target Conditions
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

    outlet_state = IAPWS97(P=p_out_mpaa, h=enthalpy_steam_outlet)
    target_temp_outlet = outlet_state.T - 273.15
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

# Saturation Properties
saturated_liquid = IAPWS97(P=p_out_mpaa, x=0)
saturation_temp = saturated_liquid.T - 273.15
superheat_margin = target_temp_outlet - saturation_temp

# --- DYNAMIC TIME-SERIES SIMULATION ---
time_vector = np.linspace(0, simulation_time, 200)

# First order exponential lag response: Y(t) = Y_initial + (Y_target - Y_initial) * (1 - exp(-t/tau))
temp_outlet_dynamic = initial_temp + (target_temp_outlet - initial_temp) * (
    1 - np.exp(-time_vector / time_constant)
)

if is_calc_mode:
    # Spray flow changes, driving outlet temperature
    fw_flow_dynamic = initial_spray + (target_fw_flow - initial_spray) * (
        1 - np.exp(-time_vector / time_constant)
    )
    inlet_flow_dynamic = initial_steam_flow + (
        target_inlet_flow - initial_steam_flow
    ) * (1 - np.exp(-time_vector / time_constant))
    outlet_flow_dynamic = inlet_flow_dynamic + fw_flow_dynamic
else:
    # Target temperature changes, driving required spray flow
    inlet_flow_dynamic = initial_steam_flow + (
        target_inlet_flow - initial_steam_flow
    ) * (1 - np.exp(-time_vector / time_constant))
    fw_flow_dynamic = initial_spray + (target_fw_flow - initial_spray) * (
        1 - np.exp(-time_vector / time_constant)
    )
    outlet_flow_dynamic = inlet_flow_dynamic + fw_flow_dynamic

# --- DISPLAY UI COMPONENTS ---
c1, c2, c3 = st.columns(3)
c1.metric("Steady-State Inlet Flow", f"{target_inlet_flow:.2f} t/h")
c2.metric("Steady-State Spray Flow", f"{target_fw_flow:.2f} t/h")
c3.metric("Steady-State Outlet Flow", f"{target_outlet_flow:.2f} t/h")

st.markdown("---")

# Dynamic Response Plots (Plotly Subplots)
st.subheader("📊 Dynamic Step Response Plot")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(
        "Outlet Steam Temperature Dynamic Trajectory",
        "Mass Flow Rates Trajectory",
    ),
)

# Temperature Plot
fig.add_trace(
    go.Scatter(
        x=time_vector,
        y=temp_outlet_dynamic,
        mode="lines",
        name="Outlet Temp (°C)",
        line=dict(color="#00A896", width=2.5),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=time_vector,
        y=[saturation_temp] * len(time_vector),
        mode="lines",
        name="Saturation Temp (°C)",
        line=dict(color="#E63946", dash="dash"),
    ),
    row=1,
    col=1,
)

# Flow Plot
fig.add_trace(
    go.Scatter(
        x=time_vector,
        y=outlet_flow_dynamic,
        mode="lines",
        name="Outlet Steam Flow (t/h)",
        line=dict(color="#1D3557", width=2),
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=time_vector,
        y=fw_flow_dynamic,
        mode="lines",
        name="Spray Water Flow (t/h)",
        line=dict(color="#457B9D", width=2),
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=time_vector,
        y=inlet_flow_dynamic,
        mode="lines",
        name="Inlet Steam Flow (t/h)",
        line=dict(color="#A8DADC", width=2),
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=550,
    hovermode="x unified",
    template="plotly_white",
    font=dict(family="Segoe UI, Aptos, Arial", size=12),
)

fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
fig.update_yaxes(title_text="Flow Rate (t/h)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# Safety Alerts
if superheat_margin < 0:
    st.error(
        "CRITICAL ALERT: Steady-state temperature is below saturation! Liquid"
        " droplets present."
    )
elif superheat_margin < 2.0:
    st.warning(
        "WARNING: Low superheat margin (< 2.0 °C)! Risk of incomplete"
        " vaporization."
    )
else:
    st.success(
        f"System State: SUPERHEATED STEAM (Margin: {superheat_margin:.2f} °C)"
    )

# Summary Process Table
st.subheader("Process Results Breakdown")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### Pressure & Thermal Summary")
    st.write(f"**Inlet Pressure:** {p_in_barg:.2f} barG | {p_in_mpaa:.3f} MPaA")
    st.write(
        f"**Outlet Pressure:** {p_out_barg:.2f} barG | {p_out_mpaa:.3f} MPaA"
    )
    st.write(
        f"**Target Outlet Temp:** {target_temp_outlet:.2f} °C (Sat Temp:"
        f" {saturation_temp:.2f} °C)"
    )

with col_right:
    st.markdown("##### Enthalpy & Mass Balance")
    st.write(f"**Inlet Enthalpy:** {enthalpy_steam_inlet:.2f} kJ/kg")
    st.write(f"**Spray Water Enthalpy:** {enthalpy_feedwater_inlet:.2f} kJ/kg")
    st.write(f"**Outlet Enthalpy:** {enthalpy_steam_outlet:.2f} kJ/kg")
