from iapws import IAPWS97
import streamlit as st

st.set_page_config(
    page_title="Desuperheater Calculator", page_icon="💨", layout="wide"
)

st.title("💨 Desuperheater Letdown Mass & Energy Balance by Iqbal SHERPA _20260708")

# --- SIDEBAR INPUTS ---
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

st.sidebar.header("2. Desuperheater Outlet Parameters")
Desuperheater_Outlet_Steam_Pressure = st.sidebar.number_input(
    "Outlet Pressure", value=4.6
)
Outlet_Temperature_Calculation_Mode = st.sidebar.radio(
    "Calculation Mode",
    [
        "INPUT - Specify Target Outlet Temperature",
        "CALC - Calculate Outlet Temperature from Spray Flow",
    ],
)

Desuperheater_Outlet_Steam_Target_Temperature_Degrees_Celsius = (
    st.sidebar.number_input("Target Outlet Temp (°C)", value=160.0)
)

st.sidebar.header("3. Spray Feedwater Parameters")
Spray_Feedwater_Inlet_Pressure = st.sidebar.number_input(
    "Feedwater Pressure", value=70.0
)
Spray_Feedwater_Inlet_Temperature_Degrees_Celsius = st.sidebar.number_input(
    "Feedwater Temp (°C)", value=90.0
)
Specified_Spray_Feedwater_Mass_Flow_Rate_Tons_Per_Hour = (
    st.sidebar.number_input("Specified Spray Flow (t/h)", value=2.35)
)

st.sidebar.header("4. Flow Rate Basis")
Mass_Flow_Rate_Basis = st.sidebar.selectbox(
    "Basis", ["Inlet Steam Flow Rate", "Outlet Target Steam Flow Rate"]
)
Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour = st.sidebar.number_input(
    "Specified Steam Flow (t/h)", value=107.0
)

# --- CALCULATION LOGIC ---
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

p_in_mpag = p_in_mpaa - ATMOSPHERIC_PRESSURE_MEGAPASCALS
p_out_mpag = p_out_mpaa - ATMOSPHERIC_PRESSURE_MEGAPASCALS
p_fw_mpag = p_fw_mpaa - ATMOSPHERIC_PRESSURE_MEGAPASCALS

temperature_steam_inlet = High_Pressure_Inlet_Steam_Temperature_Degrees_Celsius
temperature_feedwater_inlet = (
    Spray_Feedwater_Inlet_Temperature_Degrees_Celsius
)

# Inlet Specific Enthalpies via IAPWS-IF97
enthalpy_steam_inlet = IAPWS97(
    P=p_in_mpaa, T=temperature_steam_inlet + 273.15
).h
enthalpy_feedwater_inlet = IAPWS97(
    P=p_fw_mpaa, T=temperature_feedwater_inlet + 273.15
).h

is_calc_mode = (
    Outlet_Temperature_Calculation_Mode
    == "CALC - Calculate Outlet Temperature from Spray Flow"
)

if is_calc_mode:
    mass_flow_feedwater_inlet = (
        Specified_Spray_Feedwater_Mass_Flow_Rate_Tons_Per_Hour
    )
    if Mass_Flow_Rate_Basis == "Inlet Steam Flow Rate":
        mass_flow_steam_inlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_outlet = (
            mass_flow_steam_inlet + mass_flow_feedwater_inlet
        )
    else:
        mass_flow_steam_outlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_inlet = (
            mass_flow_steam_outlet - mass_flow_feedwater_inlet
        )

    enthalpy_steam_outlet = (
        (mass_flow_steam_inlet * enthalpy_steam_inlet)
        + (mass_flow_feedwater_inlet * enthalpy_feedwater_inlet)
    ) / mass_flow_steam_outlet

    outlet_state = IAPWS97(P=p_out_mpaa, h=enthalpy_steam_outlet)
    temperature_steam_outlet = outlet_state.T - 273.15
else:
    temperature_steam_outlet = (
        Desuperheater_Outlet_Steam_Target_Temperature_Degrees_Celsius
    )
    enthalpy_steam_outlet = IAPWS97(
        P=p_out_mpaa, T=temperature_steam_outlet + 273.15
    ).h

    if Mass_Flow_Rate_Basis == "Inlet Steam Flow Rate":
        mass_flow_steam_inlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_feedwater_inlet = (
            mass_flow_steam_inlet
            * (enthalpy_steam_outlet - enthalpy_steam_inlet)
            / (enthalpy_feedwater_inlet - enthalpy_steam_outlet)
        )
        mass_flow_steam_outlet = (
            mass_flow_steam_inlet + mass_flow_feedwater_inlet
        )
    else:
        mass_flow_steam_outlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_inlet = (
            mass_flow_steam_outlet
            * (enthalpy_steam_outlet - enthalpy_feedwater_inlet)
            / (enthalpy_steam_inlet - enthalpy_feedwater_inlet)
        )
        mass_flow_feedwater_inlet = (
            mass_flow_steam_outlet - mass_flow_steam_inlet
        )

# Saturation Properties
saturated_liquid = IAPWS97(P=p_out_mpaa, x=0)
saturation_temp = saturated_liquid.T - 273.15
superheat_margin = temperature_steam_outlet - saturation_temp

if superheat_margin > 0.1:
    outlet_steam_condition = "SUPERHEATED STEAM"
elif abs(superheat_margin) <= 0.1:
    outlet_steam_condition = "SATURATED STEAM (Dry Saturated)"
else:
    outlet_steam_condition = "WET STEAM (Two-Phase Liquid and Vapor)"

pressure_drop_bar = (p_in_mpaa - p_out_mpaa) * 10.0

# --- DISPLAY STREAMLIT UI ---

# Top Summary KPI Cards
c1, c2, c3 = st.columns(3)
c1.metric("Inlet Steam Flow", f"{mass_flow_steam_inlet:.2f} t/h")
c2.metric("Spray Water Flow", f"{mass_flow_feedwater_inlet:.2f} t/h")
c3.metric("Outlet Steam Flow", f"{mass_flow_steam_outlet:.2f} t/h")

st.markdown("---")

# Safety Alert Banners
if superheat_margin < 0:
    st.error(
        "CRITICAL ALERT: Outlet temperature is below saturation! Liquid"
        " droplets will be present in the steam line."
    )
elif superheat_margin < 2.0:
    st.warning(
        "WARNING: Low superheat margin (< 2.0 °C)! High risk of incomplete"
        " vaporization and water carryover."
    )
else:
    st.success(f"System State: {outlet_steam_condition}")

# Detailed Results Table
st.subheader("Process Results Breakdown")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### Pressure & Thermal Summary")
    st.write(
        f"**Inlet Pressure:** {p_in_barg:.2f} barG | {p_in_bara:.2f} barA |"
        f" {p_in_mpaa:.3f} MPaA"
    )
    st.write(
        f"**Outlet Pressure:** {p_out_barg:.2f} barG | {p_out_bara:.2f} barA |"
        f" {p_out_mpaa:.3f} MPaA"
    )
    st.write(
        f"**Spray Pressure:** {p_fw_barg:.2f} barG | {p_fw_bara:.2f} barA |"
        f" {p_fw_mpaa:.3f} MPaA"
    )
    st.write(f"**Steam Pressure Drop:** {pressure_drop_bar:.2f} bar")
    st.write(
        f"**Resulting Outlet Temp:** {temperature_steam_outlet:.2f} °C"
    )
    st.write(
        f"**Outlet Saturation Temp:** {saturation_temp:.2f} °C"
    )
    st.write(
        f"**Superheat Margin:** {superheat_margin:.2f} °C"
    )

with col_right:
    st.markdown("##### Enthalpy & Mass Balance")
    st.write(
        f"**Inlet Steam Enthalpy:** {enthalpy_steam_inlet:.2f} kJ/kg"
    )
    st.write(
        f"**Spray Water Enthalpy:** {enthalpy_feedwater_inlet:.2f} kJ/kg"
    )
    st.write(
        f"**Outlet Steam Enthalpy:** {enthalpy_steam_outlet:.2f} kJ/kg"
    )
    st.write(
        f"**Inlet Steam Mass Flow:** {mass_flow_steam_inlet:.2f} t/h"
    )
    st.write(
        f"**Spray Water Mass Flow:** {mass_flow_feedwater_inlet:.2f} t/h"
    )
    st.write(
        f"**Outlet Steam Mass Flow:** {mass_flow_steam_outlet:.2f} t/h"
    )
