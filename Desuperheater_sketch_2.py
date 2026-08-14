import time
from iapws import IAPWS97
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Real-Time Desuperheater Lead-Lag Dynamics",
    page_icon="💨",
    layout="wide",
)

st.title("💨 Real-Time Desuperheater Dynamic Simulation (Lead-Lag Enthalpy Model)")
st.caption(
    "Developed by Iqbal SHERPA 20260807. Contact me for further information"
    " @iqbalshafiq96@gmail.com"
)

# ----------------------------------------------------------------------
# COLOR PALETTE CONFIGURATION & COLOR INTERPOLATION HELPER
# ----------------------------------------------------------------------
HP_COLOR = "#800020"  # Professional Maroon for HP Steam
HP_GLOW = "#A52A2A"   # Soft Warm Maroon Glow for HP Steam

FW_COLOR = "#0EA5E9"  # Vivid sky blue for feedwater spray
FW_GLOW = "#38BDF8"   # Cyan glow for feedwater

EQUIP_COLOR = "#D97706"  # Amber for control valve & desuperheater outline
EQUIP_GLOW = "#F59E0B"  # Bright amber glow for active equipment

# Base LP line colors
LP_COLOR = "#64748B"  # Low-Pressure Steam Line Color
LP_GLOW = "#94A3B8"   # Low-Pressure Steam Glow


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def interpolate_color(color_a, color_b, factor):
    """Interpolate linearly between two hex colors by factor [0.0, 1.0]"""
    factor = max(0.0, min(1.0, factor))
    rgb_a = hex_to_rgb(color_a)
    rgb_b = hex_to_rgb(color_b)
    res_rgb = tuple(
        rgb_a[i] + (rgb_b[i] - rgb_a[i]) * factor for i in range(3)
    )
    return rgb_to_hex(res_rgb)


def calculate_lp_colors(superheat_margin):
    """Dynamically calculates the LP Steam Pipe & Particle Color spectrum:
    - Margin > 8.0°C to 20.0°C: Transitions from Silver (#64748B) to HP Maroon (#800020).
    - Margin < 2.0°C to 0.0°C: Transitions from Silver (#64748B) to Feedwater Blue (#0EA5E9).
    - Margin between 2.0°C and 8.0°C: Remains neutral Silver (#64748B).
    """
    if superheat_margin > 8.0:
        factor = (superheat_margin - 8.0) / (20.0 - 8.0)
        lp_color = interpolate_color(LP_COLOR, HP_COLOR, factor)
        lp_glow = interpolate_color(LP_GLOW, HP_GLOW, factor)
    elif superheat_margin < 2.0:
        factor = (2.0 - superheat_margin) / (2.0 - 0.0)
        lp_color = interpolate_color(LP_COLOR, FW_COLOR, factor)
        lp_glow = interpolate_color(LP_GLOW, FW_GLOW, factor)
    else:
        lp_color = LP_COLOR
        lp_glow = LP_GLOW

    return lp_color, lp_glow


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
    target_m_fw,
    p_out,
    t_out,
    target_t_out,
    m_out,
    target_m_out,
    t_sat,
    t_margin,
    p_unit,
):
    # Dynamic particle speed calculation based on mass flow rates
    dur_inlet = max(1.0, min(6.0, 150.0 / max(m_in, 1.0)))
    dur_valve = max(0.8, min(4.0, 100.0 / max(m_in, 1.0)))
    dur_fw = max(0.8, min(4.0, 10.0 / max(m_fw, 0.1)))
    dur_outlet = max(1.0, min(6.0, 150.0 / max(m_out, 1.0)))

    # Compute dynamic LP colors based on superheat margin spectrum
    lp_color, lp_glow = calculate_lp_colors(t_margin)

    svg = f"""
    <svg
        width="100%"
        height="350"
        viewBox="0 0 1500 350"
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
            <stop offset="40%" stop-color="{lp_glow}" stop-opacity="0.95"/>
            <stop offset="100%" stop-color="{lp_color}" stop-opacity="0"/>
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

    <!-- PIPING SECTIONS -->
    <path d="M60 210 H800" stroke="{HP_COLOR}" stroke-width="9" opacity="0.16" filter="url(#lineGlow)" />
    <path d="M60 210 H800" stroke="{HP_COLOR}" stroke-width="5" stroke-linecap="round" />

    <path d="M800 210 H1450" stroke="{lp_color}" stroke-width="9" opacity="0.25" filter="url(#lineGlow)" />
    <path d="M800 210 H1450" stroke="{lp_color}" stroke-width="5" stroke-linecap="round" />

    <!-- PARTICLES -->
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

        <circle r="7" fill="url(#hpSteamParticle)">
            <animateMotion dur="{dur_valve:.2f}s" repeatCount="indefinite" path="M490 210 H800"/>
        </circle>
        <circle r="5" fill="url(#hpSteamParticle)">
            <animateMotion dur="{dur_valve:.2f}s" begin="-1s" repeatCount="indefinite" path="M490 210 H800"/>
        </circle>
    </g>

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

    <!-- PRESSURE CONTROL VALVE -->
    <g transform="translate(490,210)">
        <polygon points="-35,-30 0,0 -35,30" fill="#FFFFFF"/>
        <polygon points="35,-30 0,0 35,30" fill="#FFFFFF"/>
        <circle cx="0" cy="-70" r="17" fill="#FFFFFF"/>
        
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

    <!-- DESUPERHEATER BODY -->
    <g transform="translate(800,210)">
        <path d="M-105 -42 L-35 -16 L35 -16 L105 -42 L105 42 L35 16 L-35 16 L-105 42 Z" fill="#FFFFFF"/>
        
        <g filter="url(#equipmentGlow)">
            <path d="M-105 -42 L-35 -16 L35 -16 L105 -42 L105 42 L35 16 L-35 16 L-105 42 Z" fill="#FFFFFF" fill-opacity="0.95" stroke="{EQUIP_COLOR}" stroke-width="3" stroke-linejoin="round"/>
            <line x1="-35" y1="-16" x2="35" y2="-16" stroke="{EQUIP_COLOR}" stroke-width="2"/>
            <line x1="-35" y1="16" x2="35" y2="16" stroke="{EQUIP_COLOR}" stroke-width="2"/>
            <line x1="0" y1="-75" x2="0" y2="-5" stroke="{FW_COLOR}" stroke-width="3"/>
            <circle cx="0" cy="-5" r="6" fill="{FW_COLOR}" filter="url(#waterGlow)"/>

            <g filter="url(#waterGlow)">
                <circle r="5" fill="url(#waterParticle)">
                    <animateMotion dur="1.0s" repeatCount="indefinite" path="M0 -5 L85 8"/>
                </circle>
                <circle r="4" fill="url(#waterParticle)">
                    <animateMotion dur="1.2s" begin="-0.4s" repeatCount="indefinite" path="M0 -5 L90 -6"/>
                </circle>
                <circle r="3" fill="url(#waterParticle)">
                    <animateMotion dur="0.9s" begin="-0.2s" repeatCount="indefinite" path="M0 -5 L80 14"/>
                </circle>
            </g>
        </g>
    </g>

    <!-- EQUIPMENT LABELS -->
    <text x="490" y="265" text-anchor="middle" fill="#334155" font-family="Segoe UI, sans-serif" font-size="16" font-weight="600">Isenthalpic Expansion</text>
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
        <text x="60" y="100" font-weight="bold" fill="{HP_COLOR}">High Pressure Steam Line</text>
        <text x="60" y="125" fill="{HP_COLOR}">Flow: {m_in:.2f} t/h</text>
        <text x="60" y="150" fill="{HP_COLOR}">Press: {p_in:.2f} {p_unit}</text>
        <text x="60" y="175" fill="{HP_COLOR}">Temp: {t_in:.1f} degC</text>

        <text x="830" y="30" font-weight="bold" fill="{FW_COLOR}">Feedwater Spray Line</text>
        <text x="830" y="55" fill="{FW_COLOR}">Flow: {m_fw:.2f} t/h (Target {target_m_fw:.2f} t/h)</text>
        <text x="830" y="80" fill="{FW_COLOR}">Press: {p_fw:.2f} {p_unit}</text>
        <text x="830" y="105" fill="{FW_COLOR}">Temp: {t_fw:.1f} degC</text>

        <text x="1150" y="30" font-weight="bold" fill="{lp_color}">LP Pressure Steam Line</text>
        <text x="1150" y="55" fill="{lp_color}">Flow: {m_out:.2f} t/h (Target {target_m_out:.2f} t/h)</text>
        <text x="1150" y="80" fill="{lp_color}">Press: {p_out:.2f} {p_unit}</text>
        <text x="1150" y="105" fill="{lp_color}">Temp: {t_out:.1f} degC (Target {target_t_out:.1f} degC)</text>
        <text x="1150" y="130" fill="{lp_color}">Sat Temp: {t_sat:.1f} degC</text>
        <text x="1150" y="155" fill="{lp_color}">Superheat Margin: {t_margin:.1f} degC</text>
    </g>

    </svg>
    """
    return svg


# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Dynamic & Lead-Lag Controls")
is_running = st.sidebar.toggle("Run Live Simulation", value=True)

tau_steam = st.sidebar.slider(
    "Steam Flow Time Constant τ_steam (s)",
    1.0,
    30.0,
    2.19,
    step=0.01,
    help="Speed of steam flow response to step changes.",
)

tau_spray = st.sidebar.slider(
    "Feedwater Flow Time Constant τ_spray (s)",
    1.0,
    30.0,
    3.46,
    step=0.01,
    help=(
        "Speed of spray water valve response. Set τ_spray > τ_steam to see"
        " temperature overshoot!"
    ),
)

tau_thermal = st.sidebar.slider(
    "Thermal Mixing/Sensor Lag τ_thermal (s)",
    0.1,
    10.0,
    0.73,
    step=0.01,
    help="Thermal inertia/mixing lag inside the pipe line.",
)

dt = st.sidebar.slider("Step Delay Δt (s)", 0.05, 1.0, 0.2, step=0.01)

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
    "Specified Steam Flow (t/h)", value=10.0
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


# --- ISOLATED SIMULATION & RENDER FRAGMENT ---
@st.fragment(run_every=dt if is_running else None)
def render_simulation_fragment():
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

        # 3. Apply thermal lag to temperature reading
        new_temp = curr_temp + (dt / tau_thermal) * (
            instantaneous_temp - curr_temp
        )

        # Append to rolling history buffers
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

    # Render Animated Letdown Station P&ID SVG via st.components.v1.html
    current_outlet_temp = st.session_state.temp_history[-1]
    current_margin = current_outlet_temp - saturation_temp

    svg_code = build_animated_process_svg(
        p_in=High_Pressure_Inlet_Steam_Pressure,
        t_in=High_Pressure_Inlet_Steam_Temperature_Degrees_Celsius,
        m_in=st.session_state.inlet_flow_history[-1],
        p_fw=Spray_Feedwater_Inlet_Pressure,
        t_fw=Spray_Feedwater_Inlet_Temperature_Degrees_Celsius,
        m_fw=st.session_state.spray_flow_history[-1],
        target_m_fw=target_fw_flow,
        p_out=Desuperheater_Outlet_Steam_Pressure,
        t_out=current_outlet_temp,
        target_t_out=target_temp_outlet,
        m_out=st.session_state.outlet_flow_history[-1],
        target_m_out=target_outlet_flow,
        t_sat=saturation_temp,
        t_margin=current_margin,
        p_unit=Pressure_Unit_Type.split("(")[-1].replace(")", ""),
    )

    components.html(svg_code, height=360)

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

    fig.add_trace(
        go.Scatter(
            x=st.session_state.time_history,
            y=st.session_state.temp_history,
            mode="lines",
            name="Outlet Temp (°C)",
            line=dict(color="#008080", width=2.5),
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
            line=dict(color="#4169E1", width=2.5),
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
            line=dict(color="#E67E22", width=2.5),
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


# Execute the fragment
render_simulation_fragment()
