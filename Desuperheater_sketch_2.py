import xml.etree.ElementTree as ET
from iapws import IAPWS97
import streamlit as st

# ----------------------------------------------------------------------
# COLOR PALETTE CONFIGURATION
# ----------------------------------------------------------------------
HP_COLOR = "#800020"  # Professional Maroon for HP Steam
HP_GLOW = "#A52A2A"  # Soft Warm Maroon Glow for HP Steam

FW_COLOR = "#0EA5E9"  # Vivid sky blue for feedwater spray
FW_GLOW = "#38BDF8"  # Cyan glow for feedwater

LP_COLOR = "#64748B"  # Low-Pressure Steam Line Color
LP_GLOW = "#94A3B8"  # Low-Pressure Steam Glow

EQUIP_COLOR = "#D97706"  # Amber for control valve & desuperheater outline
EQUIP_GLOW = "#F59E0B"  # Bright amber glow for active equipment


# ----------------------------------------------------------------------
# ANIMATED PROCESS SVG ENGINE
# ----------------------------------------------------------------------
def build_animated_process_svg(
    p_in,
    t_in,
    m_in,
    p_fw,
    t_fw,
    m_fw,
    p_out,
    t_out,
    m_out,
    t_sat,
    t_margin,
    p_unit,
):
    # Dynamic particle speed calculation based on mass flow rates
    dur_inlet = max(1.0, min(6.0, 150.0 / max(m_in, 1.0)))
    dur_valve = max(0.8, min(4.0, 100.0 / max(m_in, 1.0)))
    dur_fw = max(0.8, min(4.0, 10.0 / max(m_fw, 0.1)))
    dur_outlet = max(1.0, min(6.0, 150.0 / max(m_out, 1.0)))

    svg = f"""
    <svg
        width="100%"
        height="330"
        viewBox="0 0 1500 330"
        xmlns="http://www.w3.org/2000/svg"
        preserveAspectRatio="xMidYMid meet"
    >

    <defs>
        <!-- HP STEAM GLOW -->
        <filter id="hpSteamGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>

        <!-- LP STEAM GLOW -->
        <filter id="lpSteamGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>

        <!-- FEEDWATER GLOW -->
        <filter id="waterGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>

        <!-- EQUIPMENT GLOW -->
        <filter id="equipmentGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="5" result="blur"/>
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>

        <!-- PROCESS LINE GLOW -->
        <filter id="lineGlow" x="-20%" y="-100%" width="140%" height="300%">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>

        <!-- HP STEAM PARTICLE GRADIENT -->
        <radialGradient id="hpSteamParticle">
            <stop offset="0%" stop-color="#FFFFFF" stop-opacity="1"/>
            <stop offset="40%" stop-color="{HP_GLOW}" stop-opacity="0.95"/>
            <stop offset="100%" stop-color="{HP_COLOR}" stop-opacity="0"/>
        </radialGradient>

        <!-- LP STEAM PARTICLE GRADIENT -->
        <radialGradient id="lpSteamParticle">
            <stop offset="0%" stop-color="#FFFFFF" stop-opacity="1"/>
            <stop offset="40%" stop-color="{LP_GLOW}" stop-opacity="0.95"/>
            <stop offset="100%" stop-color="{LP_COLOR}" stop-opacity="0"/>
        </radialGradient>

        <!-- WATER PARTICLE GRADIENT -->
        <radialGradient id="waterParticle">
            <stop offset="0%" stop-color="#FFFFFF" stop-opacity="1"/>
            <stop offset="40%" stop-color="{FW_GLOW}" stop-opacity="1"/>
            <stop offset="100%" stop-color="{FW_COLOR}" stop-opacity="0"/>
        </radialGradient>

        <!-- WATER ARROW MARKER ONLY -->
        <marker id="waterArrow" markerWidth="12" markerHeight="12" refX="10" refY="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 Z" fill="{FW_COLOR}"/>
        </marker>
    </defs>

    <!-- LAYER 1: BACK - PIPING SECTIONS -->
    <!-- HP Steam Pipe (Professional Maroon) -->
    <path d="M60 210 H800" stroke="{HP_COLOR}" stroke-width="9" opacity="0.16" filter="url(#lineGlow)" />
    <path d="M60 210 H800" stroke="{HP_COLOR}" stroke-width="5" stroke-linecap="round" />

    <!-- LP Steam Pipe (Blue) -->
    <path d="M800 210 H1450" stroke="{LP_COLOR}" stroke-width="9" opacity="0.16" filter="url(#lineGlow)" />
    <path d="M800 210 H1450" stroke="{LP_COLOR}" stroke-width="5" stroke-linecap="round" />

    <!-- PARTICLES -->
    <!-- HP Inlet Steam Particles (Maroon) -->
    <g filter="url(#hpSteamGlow)">
        <circle r="7" fill="url(#hpSteamParticle)">
            <animateMotion dur="{dur_inlet:.2f}s" repeatCount="indefinite" path="M60 210 H490"/>
        </circle>
        <circle r="5" fill="url(#hpSteamParticle)">
            <animateMotion dur="{dur_inlet:.2f}s" begin="-1.2s" repeatCount="indefinite" path="M60 210 H490"/>
        </circle>
        <circle r="4" fill="url(#hpSteamParticle)">
            <animateMotion dur="{dur_inlet:.2f}s" begin="-2.5s" repeatCount="indefinite" path="M60 210 H490"/>
        </circle>

        <!-- Valve-to-Desuperheater Section -->
        <circle r="7" fill="url(#hpSteamParticle)">
            <animateMotion dur="{dur_valve:.2f}s" repeatCount="indefinite" path="M490 210 H800"/>
        </circle>
        <circle r="5" fill="url(#hpSteamParticle)">
            <animateMotion dur="{dur_valve:.2f}s" begin="-1s" repeatCount="indefinite" path="M490 210 H800"/>
        </circle>
    </g>

    <!-- LP Outlet Steam Particles (Blue) -->
    <g filter="url(#lpSteamGlow)">
        <circle r="7" fill="url(#lpSteamParticle)">
            <animateMotion dur="{dur_outlet:.2f}s" repeatCount="indefinite" path="M800 210 H1450"/>
        </circle>
        <circle r="5" fill="url(#lpSteamParticle)">
            <animateMotion dur="{dur_outlet:.2f}s" begin="-1.5s" repeatCount="indefinite" path="M800 210 H1450"/>
        </circle>
        <circle r="4" fill="url(#lpSteamParticle)">
            <animateMotion dur="{dur_outlet:.2f}s" begin="-3s" repeatCount="indefinite" path="M800 210 H1450"/>
        </circle>
    </g>

    <!-- LAYER 2: FOREGROUND - PRESSURE CONTROL VALVE -->
    <g transform="translate(490,210)">
        <!-- Solid White Background Fill to Occlude Pipe Line -->
        <polygon points="-35,-30 0,0 -35,30" fill="#FFFFFF"/>
        <polygon points="35,-30 0,0 35,30" fill="#FFFFFF"/>
        <circle cx="0" cy="-70" r="17" fill="#FFFFFF"/>
        
        <!-- Foreground Valve Outlines & Actuator (Centered) -->
        <g filter="url(#equipmentGlow)">
            <polygon points="-35,-30 0,0 -35,30" fill="#FFFFFF" fill-opacity="0.9" stroke="{EQUIP_COLOR}" stroke-width="3"/>
            <polygon points="35,-30 0,0 35,30" fill="#FFFFFF" fill-opacity="0.9" stroke="{EQUIP_COLOR}" stroke-width="3"/>
            <line x1="0" y1="0" x2="0" y2="-53" stroke="{EQUIP_COLOR}" stroke-width="3"/>
            <circle cx="0" cy="-70" r="17" fill="#FFFFFF" stroke="{EQUIP_COLOR}" stroke-width="3"/>
            <circle cx="0" cy="-70" r="20" fill="none" stroke="{EQUIP_GLOW}" stroke-width="2" opacity="0">
                <animate attributeName="r" values="18;28;18" dur="2.5s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.7;0;0.7" dur="2.5s" repeatCount="indefinite"/>
            </circle>
        </g>
    </g>
    <text x="490" y="260" text-anchor="middle" fill="#334155" font-family="Segoe UI, sans-serif" font-size="16" font-weight="600">Isenthalpic Expansion</text>

    <!-- LAYER 3: FOREGROUND - DESUPERHEATER BODY -->
    <g transform="translate(800,210)">
        <!-- Solid White Background Fill -->
        <path d="M-105 -42 L-35 -16 L35 -16 L105 -42 L105 42 L35 16 L-35 16 L-105 42 Z" fill="#FFFFFF"/>
        
        <g filter="url(#equipmentGlow)">
            <path d="M-105 -42 L-35 -16 L35 -16 L105 -42 L105 42 L35 16 L-35 16 L-105 42 Z" fill="#FFFFFF" fill-opacity="0.95" stroke="{EQUIP_COLOR}" stroke-width="3" stroke-linejoin="round"/>
            <line x1="-35" y1="-16" x2="35" y2="-16" stroke="{EQUIP_COLOR}" stroke-width="2"/>
            <line x1="-35" y1="16" x2="35" y2="16" stroke="{EQUIP_COLOR}" stroke-width="2"/>
            <line x1="0" y1="-75" x2="0" y2="-5" stroke="{FW_COLOR}" stroke-width="3"/>
            <circle cx="0" cy="-5" r="6" fill="{FW_COLOR}" filter="url(#waterGlow)"/>

            <!-- SPRAY PARTICLES -->
            <g filter="url(#waterGlow)">
                <circle r="5" fill="url(#waterParticle)">
                    <animateMotion dur="1.0s" repeatCount="indefinite" path="M0 -5 L35 5"/>
                </circle>
                <circle r="4" fill="url(#waterParticle)">
                    <animateMotion dur="1.2s" begin="-0.4s" repeatCount="indefinite" path="M0 -5 L45 -8"/>
                </circle>
                <circle r="3" fill="url(#waterParticle)">
                    <animateMotion dur="0.9s" begin="-0.2s" repeatCount="indefinite" path="M0 -5 L40 12"/>
                </circle>
            </g>
        </g>
    </g>
    <text x="800" y="265" text-anchor="middle" fill="#334155" font-family="Segoe UI, sans-serif" font-size="16" font-weight="600">Desuperheater</text>

    <!-- FEEDWATER PIPE -->
    <path d="M800 65 V205" stroke="{FW_COLOR}" stroke-width="5" stroke-linecap="round"/>
    <path d="M800 65 V205" stroke="{FW_COLOR}" stroke-width="10" stroke-linecap="round" opacity="0.15" filter="url(#lineGlow)"/>
    <path d="M800 100 V150" stroke="{FW_COLOR}" stroke-width="2" marker-end="url(#waterArrow)"/>

    <!-- FEEDWATER PARTICLES -->
    <g filter="url(#waterGlow)">
        <circle r="7" fill="url(#waterParticle)">
            <animateMotion dur="{dur_fw:.2f}s" repeatCount="indefinite" path="M800 65 V205"/>
        </circle>
        <circle r="5" fill="url(#waterParticle)">
            <animateMotion dur="{dur_fw:.2f}s" begin="-0.65s" repeatCount="indefinite" path="M800 65 V205"/>
        </circle>
        <circle r="4" fill="url(#waterParticle)">
            <animateMotion dur="{dur_fw:.2f}s" begin="-1.3s" repeatCount="indefinite" path="M800 65 V205"/>
        </circle>
    </g>

    <!-- PROCESS LABELS -->
    <g font-family="Segoe UI, sans-serif" font-size="15">
        <!-- Inlet Steam Text (Professional Maroon) -->
        <text x="60" y="100" font-weight="bold" fill="{HP_COLOR}">High Pressure Steam Line</text>
        <text x="60" y="125" fill="{HP_COLOR}">Flow: {m_in:.2f} t/h</text>
        <text x="60" y="150" fill="{HP_COLOR}">Press: {p_in:.2f} {p_unit}</text>
        <text x="60" y="175" fill="{HP_COLOR}">Temp: {t_in:.1f} °C</text>

        <!-- Feedwater Text -->
        <text x="830" y="35" font-weight="bold" fill="{FW_COLOR}">Feedwater Spray Line</text>
        <text x="830" y="60" fill="{FW_COLOR}">Flow: {m_fw:.2f} t/h</text>
        <text x="830" y="85" fill="{FW_COLOR}">Press: {p_fw:.2f} {p_unit}</text>
        <text x="830" y="110" fill="{FW_COLOR}">Temp: {t_fw:.1f} °C</text>

        <!-- Outlet Steam Text (Superheat Margin aligned at y=175 to match HP Temp) -->
        <text x="1150" y="50" font-weight="bold" fill="{LP_COLOR}">Low Pressure Steam Line</text>
        <text x="1150" y="75" fill="{LP_COLOR}">Flow: {m_out:.2f} t/h</text>
        <text x="1150" y="100" fill="{LP_COLOR}">Press: {p_out:.2f} {p_unit}</text>
        <text x="1150" y="125" fill="{LP_COLOR}">Temp: {t_out:.1f} °C</text>
        <text x="1150" y="150" fill="{LP_COLOR}">Sat Temp: {t_sat:.1f} °C</text>
        <text x="1150" y="175" fill="{LP_COLOR}">Superheat Margin: {t_margin:.1f} °C</text>
    </g>

    </svg>
    """
    return svg


# ----------------------------------------------------------------------
# STREAMLIT APPLICATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Desuperheater Calculator", page_icon="💨", layout="wide"
)

st.title("💨 Desuperheater Letdown Mass & Energy Balance")
st.caption(
    "Developed by Iqbal SHERPA 20260708. Contact me for further information"
    " @iqbalshafiq96@gmail.com"
)

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
        "Target Outlet Temp (°C)", value=158.0, disabled=is_calc_mode
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
    unit_label = "barG"
elif Pressure_Unit_Type == "Bar Absolute (barA)":
    p_in_mpaa = High_Pressure_Inlet_Steam_Pressure / 10.0
    p_out_mpaa = Desuperheater_Outlet_Steam_Pressure / 10.0
    p_fw_mpaa = Spray_Feedwater_Inlet_Pressure / 10.0
    unit_label = "barA"
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
    unit_label = "MPaG"
else:
    p_in_mpaa = High_Pressure_Inlet_Steam_Pressure
    p_out_mpaa = Desuperheater_Outlet_Steam_Pressure
    p_fw_mpaa = Spray_Feedwater_Inlet_Pressure
    unit_label = "MPaA"

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

if is_calc_mode:
    mass_flow_feedwater_inlet = (
        Specified_Spray_Feedwater_Mass_Flow_Rate_Tons_Per_Hour
    )
    if Mass_Flow_Rate_Basis == "Inlet Steam Flow Rate":
        mass_flow_steam_inlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_outlet = mass_flow_steam_inlet + mass_flow_feedwater_inlet
    else:
        mass_flow_steam_outlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_inlet = mass_flow_steam_outlet - mass_flow_feedwater_inlet

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
        mass_flow_steam_outlet = mass_flow_steam_inlet + mass_flow_feedwater_inlet
    else:
        mass_flow_steam_outlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_inlet = (
            mass_flow_steam_outlet
            * (enthalpy_steam_outlet - enthalpy_feedwater_inlet)
            / (enthalpy_steam_inlet - enthalpy_feedwater_inlet)
        )
        mass_flow_feedwater_inlet = mass_flow_steam_outlet - mass_flow_steam_inlet

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

# --- RENDER ANIMATED PROCESS FLOW DIAGRAM ---
svg_data = build_animated_process_svg(
    p_in=High_Pressure_Inlet_Steam_Pressure,
    t_in=temperature_steam_inlet,
    m_in=mass_flow_steam_inlet,
    p_fw=Spray_Feedwater_Inlet_Pressure,
    t_fw=temperature_feedwater_inlet,
    m_fw=mass_flow_feedwater_inlet,
    p_out=Desuperheater_Outlet_Steam_Pressure,
    t_out=temperature_steam_outlet,
    m_out=mass_flow_steam_outlet,
    t_sat=saturation_temp,
    t_margin=superheat_margin,
    p_unit=unit_label,
)

col_left, col_center, col_right = st.columns([0.05, 0.90, 0.05])
with col_center:
    st.components.v1.html(
        f'<div style="display:flex;justify-content:center;width:100%;">{svg_data}</div>',
        height=335,
    )

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
    st.write(f"**Resulting Outlet Temp:** {temperature_steam_outlet:.2f} °C")
    st.write(f"**Outlet Saturation Temp:** {saturation_temp:.2f} °C")
    st.write(f"**Superheat Margin:** {superheat_margin:.2f} °C")

with col_right:
    st.markdown("##### Enthalpy & Mass Balance")
    st.write(f"**Inlet Steam Enthalpy:** {enthalpy_steam_inlet:.2f} kJ/kg")
    st.write(f"**Spray Water Enthalpy:** {enthalpy_feedwater_inlet:.2f} kJ/kg")
    st.write(f"**Outlet Steam Enthalpy:** {enthalpy_steam_outlet:.2f} kJ/kg")
    st.write(f"**Inlet Steam Mass Flow:** {mass_flow_steam_inlet:.2f} t/h")
    st.write(f"**Spray Water Mass Flow:** {mass_flow_feedwater_inlet:.2f} t/h")
    st.write(f"**Outlet Steam Mass Flow:** {mass_flow_steam_outlet:.2f} t/h")
