import math
from iapws import IAPWS97
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------
# COLOUR PALETTE (Using Matplotlib theme defaults for text/lines)
# ----------------------------------------------------------------------
STEAM_COLOR = "#708090"  # neutral pipe color (slate grey)
FW_COLOR = "#1e88e5"  # feedwater (soft sky blue)
EQUIP_COLOR = "#d97706"  # warm amber for valve / venturi outline
EQUIP_FILL = "none"  # transparent fill to match any theme background

LW_PIPE = 4.2
LW_EQUIP = 2.0


def build_figure(
    p_in,
    t_in,
    m_in,
    p_fw,
    t_fw,
    m_fw,
    p_out,
    t_out,
    m_out,
    p_unit,
    figsize=(15, 6),
):
    fig, ax = plt.subplots(figsize=figsize)

    # Set background to transparent so it blends with Streamlit theme
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Inherit standard theme text color dynamically
    text_color = plt.rcParams.get("text.color", "currentColor")

    ax.set_xlim(0, 16)
    ax.set_ylim(1.5, 8.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def pipe(x1, y1, x2, y2, color=STEAM_COLOR, lw=LW_PIPE, zorder=2):
        ax.add_line(
            mlines.Line2D(
                [x1, x2],
                [y1, y2],
                color=color,
                lw=lw,
                solid_capstyle="round",
                zorder=zorder,
            )
        )

    def flow_arrow(x, y, dx, dy, color=STEAM_COLOR):
        ax.annotate(
            "",
            xy=(x + dx, y + dy),
            xytext=(x, y),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.8, mutation_scale=15
            ),
            zorder=3,
        )

    def control_valve(x, y, tag_lines, size=0.34):
        s = size
        top, bot = (x, y + s), (x, y - s)
        left, right = (x - s, y), (x + s, y)
        for tri_pts in ([left, top, bot], [right, top, bot]):
            ax.add_patch(
                patches.Polygon(
                    tri_pts,
                    closed=True,
                    facecolor=EQUIP_FILL,
                    edgecolor=EQUIP_COLOR,
                    lw=LW_EQUIP,
                    zorder=4,
                )
            )
        ax.add_line(
            mlines.Line2D(
                [x, x],
                [y + s, y + s + 0.38],
                color=EQUIP_COLOR,
                lw=LW_EQUIP,
                zorder=4,
            )
        )
        ax.add_patch(
            patches.Circle(
                (x, y + s + 0.6),
                0.22,
                facecolor=EQUIP_FILL,
                edgecolor=EQUIP_COLOR,
                lw=LW_EQUIP,
                zorder=4,
            )
        )
        ax.text(
            x,
            y - s - 0.28,
            tag_lines,
            ha="center",
            va="top",
            color=text_color,
            fontsize=10,
            zorder=5,
            linespacing=1.35,
        )

    def venturi_desuperheater(
        cx, cy, w=4.2, r_in=0.62, r_throat=0.22, throat_w=0.7
    ):
        x0 = cx - w / 2
        x1 = cx - throat_w / 2
        x2 = cx + throat_w / 2
        x3 = cx + w / 2

        top = [
            (x0, cy + r_in),
            (x1, cy + r_throat),
            (x2, cy + r_throat),
            (x3, cy + r_in),
        ]
        bot = [
            (x3, cy - r_in),
            (x2, cy - r_throat),
            (x1, cy - r_throat),
            (x0, cy - r_in),
        ]
        outline = top + bot
        ax.add_patch(
            patches.Polygon(
                outline,
                closed=True,
                facecolor=EQUIP_FILL,
                edgecolor=EQUIP_COLOR,
                lw=LW_EQUIP,
                zorder=4,
                joinstyle="round",
            )
        )

        flow_arrow(cx - 0.28, cy, 0.56, 0, color=STEAM_COLOR)

        nozzle_w = 0.26
        nozzle = patches.Polygon(
            [
                (cx - nozzle_w, cy + r_throat + 0.55),
                (cx + nozzle_w, cy + r_throat + 0.55),
                (cx + 0.06, cy + r_throat + 0.05),
                (cx - 0.06, cy + r_throat + 0.05),
            ],
            closed=True,
            facecolor=EQUIP_FILL,
            edgecolor=FW_COLOR,
            lw=LW_EQUIP,
            zorder=5,
        )
        ax.add_patch(nozzle)

        tip_y = cy + r_throat + 0.02
        for ang, ln in [(-26, 0.20), (-9, 0.24), (9, 0.24), (26, 0.20)]:
            rad = math.radians(ang)
            dx, dy = ln * math.sin(rad), -ln * math.cos(rad)
            ax.add_line(
                mlines.Line2D(
                    [cx, cx + dx],
                    [tip_y, tip_y + dy],
                    color=FW_COLOR,
                    lw=1.3,
                    zorder=6,
                )
            )

        ax.text(
            cx,
            cy - r_in - 0.42,
            "SPRAY DESUPERHEATER",
            ha="center",
            va="top",
            color=text_color,
            fontsize=10,
            zorder=5,
        )

        return x0, x3, cy + r_throat + 0.55

    def label(x, y, txt, color=text_color, fs=10, ha="left"):
        ax.text(
            x,
            y,
            txt,
            color=color,
            fontsize=fs,
            ha=ha,
            va="center",
            zorder=6,
            linespacing=1.35,
        )

    Y = 4.6

    # ------------------------------------------------------------------
    # HP Steam Line (Inlet)
    # ------------------------------------------------------------------
    pipe(0.6, Y, 4.1, Y)
    flow_arrow(1.6, Y, 0.8, 0)
    inlet_txt = (
        f"High Pressure Steam Line\nFlow: {m_in:.2f} t/h\nPress: {p_in:.2f}"
        f" {p_unit}\nTemp: {t_in:.1f} °C"
    )
    label(0.6, Y + 0.95, inlet_txt, fs=9.5)

    # ------------------------------------------------------------------
    # Pressure Control Valve
    # ------------------------------------------------------------------
    pcv_x = 5.0
    control_valve(pcv_x, Y, "Pressure Control Valve\n(PCV)")
    pipe(4.1, Y, pcv_x - 0.34, Y)
    pipe(pcv_x + 0.34, Y, 6.1, Y)
    flow_arrow(5.65, Y, 0.35, 0)

    # ------------------------------------------------------------------
    # Venturi Spray Desuperheater
    # ------------------------------------------------------------------
    vessel_x = 8.7
    v_in, v_out, nozzle_top = venturi_desuperheater(vessel_x, Y)
    pipe(6.1, Y, v_in, Y)

    # Feedwater Spray Line
    fw_top = 7.35
    pipe(vessel_x, fw_top, vessel_x, nozzle_top, color=FW_COLOR)
    flow_arrow(vessel_x, 7.0, 0, -0.42, color=FW_COLOR)
    pipe(6.6, fw_top, vessel_x, fw_top, color=FW_COLOR)
    flow_arrow(7.2, fw_top, 0.4, 0, color=FW_COLOR)
    fw_txt = (
        f"Feedwater Spray Line\nFlow: {m_fw:.2f} t/h\nPress: {p_fw:.2f}"
        f" {p_unit}\nTemp: {t_fw:.1f} °C"
    )
    label(6.6, fw_top + 0.75, fw_txt, color=FW_COLOR, fs=9.5)

    # ------------------------------------------------------------------
    # LP Steam Line (Outlet) - Starts exactly at desuperheater exit flange (v_out)
    # ------------------------------------------------------------------
    pipe(v_out, Y, 15.4, Y)
    flow_arrow(14.2, Y, 0.8, 0)
    outlet_txt = (
        f"Low Pressure Steam Line\nFlow: {m_out:.2f} t/h\nPress: {p_out:.2f}"
        f" {p_unit}\nTemp: {t_out:.1f} °C"
    )
    label(11.8, Y + 0.95, outlet_txt, fs=9.5)

    # Outer frame
    frame = patches.FancyBboxPatch(
        (0.18, 1.8),
        15.64,
        6.8,
        boxstyle="round,pad=0,rounding_size=0.18",
        fill=False,
        edgecolor=STEAM_COLOR,
        lw=1.0,
    )
    ax.add_patch(frame)

    fig.tight_layout()
    return fig


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

# Enthalpies via IAPWS-IF97
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

# --- RENDER DYNAMIC PROCESS FLOW DIAGRAM ---
fig = build_figure(
    p_in=High_Pressure_Inlet_Steam_Pressure,
    t_in=temperature_steam_inlet,
    m_in=mass_flow_steam_inlet,
    p_fw=Spray_Feedwater_Inlet_Pressure,
    t_fw=temperature_feedwater_inlet,
    m_fw=mass_flow_feedwater_inlet,
    p_out=Desuperheater_Outlet_Steam_Pressure,
    t_out=temperature_steam_outlet,
    m_out=mass_flow_steam_outlet,
    p_unit=unit_label,
)
st.pyplot(fig, use_container_width=True)

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
