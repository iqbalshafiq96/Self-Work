import io
import math
from iapws import IAPWS97
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="PRDS / Desuperheater Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Pressure Reducing & Desuperheating Station (PRDS) Balance")

# ----------------------------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------------------------
st.sidebar.header("Input Parameters")

unit_choice = st.sidebar.radio("Pressure Unit", ["bara", "barg", "MPa", "kPa"])

# Pressure Conversion to Absolute Bar
if unit_choice == "barg":
    p_in_raw = st.sidebar.number_input(
        "HP Steam Pressure (barg)", value=40.0, step=1.0
    )
    p_fw_raw = st.sidebar.number_input(
        "Feedwater Pressure (barg)", value=45.0, step=1.0
    )
    p_out_raw = st.sidebar.number_input(
        "LP Steam Pressure (barg)", value=10.0, step=0.5
    )
    p_in = p_in_raw + 1.01325
    p_fw = p_fw_raw + 1.01325
    p_out = p_out_raw + 1.01325
    unit_label = "barg"
elif unit_choice == "MPa":
    p_in_raw = st.sidebar.number_input(
        "HP Steam Pressure (MPa)", value=4.1, step=0.1
    )
    p_fw_raw = st.sidebar.number_input(
        "Feedwater Pressure (MPa)", value=4.6, step=0.1
    )
    p_out_raw = st.sidebar.number_input(
        "LP Steam Pressure (MPa)", value=1.1, step=0.05
    )
    p_in = p_in_raw * 10.0
    p_fw = p_fw_raw * 10.0
    p_out = p_out_raw * 10.0
    unit_label = "MPa"
elif unit_choice == "kPa":
    p_in_raw = st.sidebar.number_input(
        "HP Steam Pressure (kPa)", value=4100.0, step=100.0
    )
    p_fw_raw = st.sidebar.number_input(
        "Feedwater Pressure (kPa)", value=4600.0, step=100.0
    )
    p_out_raw = st.sidebar.number_input(
        "LP Steam Pressure (kPa)", value=1100.0, step=50.0
    )
    p_in = p_in_raw / 100.0
    p_fw = p_fw_raw / 100.0
    p_out = p_out_raw / 100.0
    unit_label = "kPa"
else:  # bara
    p_in_raw = st.sidebar.number_input(
        "HP Steam Pressure (bara)", value=41.0, step=1.0
    )
    p_fw_raw = st.sidebar.number_input(
        "Feedwater Pressure (bara)", value=46.0, step=1.0
    )
    p_out_raw = st.sidebar.number_input(
        "LP Steam Pressure (bara)", value=11.0, step=0.5
    )
    p_in = p_in_raw
    p_fw = p_fw_raw
    p_out = p_out_raw
    unit_label = "bara"

# Temperature Inputs (°C)
temperature_steam_inlet = st.sidebar.number_input(
    "HP Steam Temperature (°C)", value=400.0, step=5.0
)
temperature_feedwater_inlet = st.sidebar.number_input(
    "Feedwater Temperature (°C)", value=105.0, step=5.0
)
temperature_steam_outlet = st.sidebar.number_input(
    "LP Steam Target Temperature (°C)", value=210.0, step=5.0
)

# Flow Basis Calculation Choice
calc_mode = st.sidebar.radio(
    "Calculation Target",
    ["Given HP Steam Flow", "Given Target LP Steam Flow"],
)

if calc_mode == "Given HP Steam Flow":
    mass_flow_steam_inlet = st.sidebar.number_input(
        "HP Steam Flow (t/h)", value=50.0, step=1.0
    )
    mass_flow_target_lp = None
else:
    mass_flow_target_lp = st.sidebar.number_input(
        "Target LP Steam Flow (t/h)", value=60.0, step=1.0
    )
    mass_flow_steam_inlet = None

# Variable mapping for internal consistency
High_Pressure_Inlet_Steam_Pressure = p_in_raw
Spray_Feedwater_Inlet_Pressure = p_fw_raw
Desuperheater_Outlet_Steam_Pressure = p_out_raw

# ----------------------------------------------------------------------
# THERMODYNAMIC CALCULATIONS (IAPWS-IF97)
# ----------------------------------------------------------------------
p_in_MPa = p_in / 10.0
p_fw_MPa = p_fw / 10.0
p_out_MPa = p_out / 10.0

# Convert °C to Kelvin for IAPWS97
t_in_K = temperature_steam_inlet + 273.15
t_fw_K = temperature_feedwater_inlet + 273.15
t_out_K = temperature_steam_outlet + 273.15

# Calculate Specific Enthalpies (kJ/kg)
hp_steam = IAPWS97(P=p_in_MPa, T=t_in_K)
h_in = hp_steam.h

fw_water = IAPWS97(P=p_fw_MPa, T=t_fw_K)
h_fw = fw_water.h

lp_steam = IAPWS97(P=p_out_MPa, T=t_out_K)
h_out = lp_steam.h

# Saturation Temperature check at outlet pressure
lp_sat = IAPWS97(P=p_out_MPa, x=1)
t_sat_out = lp_sat.T - 273.15

# Energy & Mass Balance Calculations
# Mass balance: m_in + m_fw = m_out
# Energy balance: m_in * h_in + m_fw * h_fw = m_out * h_out
if mass_flow_steam_inlet is not None:
    # Calculate required spray and final output based on HP inlet flow
    mass_flow_feedwater_inlet = mass_flow_steam_inlet * (
        (h_in - h_out) / (h_out - h_fw)
    )
    mass_flow_steam_outlet = (
        mass_flow_steam_inlet + mass_flow_feedwater_inlet
    )
else:
    # Calculate required HP inlet and spray based on target LP outlet flow
    mass_flow_steam_inlet = mass_flow_target_lp * (
        (h_out - h_fw) / (h_in - h_fw)
    )
    mass_flow_feedwater_inlet = mass_flow_target_lp - mass_flow_steam_inlet
    mass_flow_steam_outlet = mass_flow_target_lp


# ----------------------------------------------------------------------
# PFD DIAGRAM RENDERER
# ----------------------------------------------------------------------
STEAM_COLOR = "#64748B"  # Slate grey
FW_COLOR = "#0EA5E9"  # Sky blue
EQUIP_COLOR = "#D97706"  # Amber
EQUIP_FILL = "none"

LW_PIPE = 4.2
LW_EQUIP = 2.0


def build_svg_figure(
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
    figsize=(14, 3.4),
):
    plt.rcParams.update(
        {
            "font.sans-serif": ["Segoe UI", "Aptos", "Arial", "DejaVu Sans"],
            "font.family": "sans-serif",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    text_color = "#334155"

    ax.set_xlim(0.4, 15.6)
    ax.set_ylim(3.4, 9.2)
    ax.set_aspect("equal")
    ax.axis("off")

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
            fontsize=11.5,
            fontweight="bold",
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

        spray_origin_x = cx
        spray_origin_y = cy
        spray_len = 0.35

        ax.add_line(
            mlines.Line2D(
                [spray_origin_x, spray_origin_x + spray_len],
                [spray_origin_y, spray_origin_y],
                color=FW_COLOR,
                linestyle="--",
                lw=1.2,
                zorder=5,
            )
        )
        ax.add_line(
            mlines.Line2D(
                [spray_origin_x, spray_origin_x + spray_len],
                [spray_origin_y, spray_origin_y + 0.18],
                color=FW_COLOR,
                linestyle="--",
                lw=1.2,
                zorder=5,
            )
        )
        ax.add_line(
            mlines.Line2D(
                [spray_origin_x, spray_origin_x + spray_len],
                [spray_origin_y, spray_origin_y - 0.18],
                color=FW_COLOR,
                linestyle="--",
                lw=1.2,
                zorder=5,
            )
        )

        ax.text(
            cx,
            cy - 0.62,
            "Desuperheater",
            ha="center",
            va="top",
            color=text_color,
            fontsize=11.5,
            fontweight="bold",
            zorder=5,
        )

        return x0, x3

    def label(x, y, txt, color=text_color, fs=11.5, ha="left"):
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

    pipe(0.6, Y, 4.1, Y)
    flow_arrow(1.6, Y, 0.8, 0)
    inlet_txt = (
        f"High Pressure Steam Line\nFlow: {m_in:.2f} t/h\nPress: {p_in:.2f}"
        f" {p_unit}\nTemp: {t_in:.1f} °C"
    )
    label(0.6, Y + 1.25, inlet_txt)

    pcv_x = 5.0
    control_valve(pcv_x, Y, "Isenthalpic Expansion")
    pipe(4.1, Y, pcv_x - 0.34, Y)
    pipe(pcv_x + 0.34, Y, 6.1, Y)
    flow_arrow(5.65, Y, 0.35, 0)

    vessel_x = 8.7
    v_in, v_out = venturi_desuperheater(vessel_x, Y)
    pipe(6.1, Y, v_in, Y)

    fw_top = 7.35
    pipe(vessel_x, fw_top, vessel_x, Y, color=FW_COLOR, zorder=3)
    flow_arrow(vessel_x, 6.2, 0, -0.42, color=FW_COLOR)
    pipe(6.6, fw_top, vessel_x, fw_top, color=FW_COLOR)
    flow_arrow(7.2, fw_top, 0.4, 0, color=FW_COLOR)
    fw_txt = (
        f"Feedwater Spray Line\nFlow: {m_fw:.2f} t/h\nPress: {p_fw:.2f}"
        f" {p_unit}\nTemp: {t_fw:.1f} °C"
    )
    label(6.6, fw_top + 0.95, fw_txt, color=FW_COLOR)

    pipe(v_out, Y, 15.4, Y)
    flow_arrow(14.2, Y, 0.8, 0)
    outlet_txt = (
        f"Low Pressure Steam Line\nFlow: {m_out:.2f} t/h\nPress: {p_out:.2f}"
        f" {p_unit}\nTemp: {t_out:.1f} °C"
    )
    label(11.8, Y + 1.25, outlet_txt)

    svg_buffer = io.StringIO()
    fig.savefig(
        svg_buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig)

    return svg_buffer.getvalue()


# ----------------------------------------------------------------------
# STREAMLIT MAIN INTERFACE DISPLAY
# ----------------------------------------------------------------------
st.subheader("Process Flow Diagram")

svg_data = build_svg_figure(
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

# Centered layout using tightly defined columns to control scale
col_left, col_center, col_right = st.columns([0.28, 0.44, 0.28])
with col_center:
    st.image(svg_data, use_container_width=True)

st.divider()

# Summary Metrics Display
st.subheader("Mass & Energy Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("HP Steam Flow", f"{mass_flow_steam_inlet:.2f} t/h")
c2.metric("Spray Water Flow", f"{mass_flow_feedwater_inlet:.2f} t/h")
c3.metric("LP Steam Flow", f"{mass_flow_steam_outlet:.2f} t/h")
c4.metric(
    "Superheat Degree",
    f"{(temperature_steam_outlet - t_sat_out):.1f} °C",
    delta=f"Sat Temp: {t_sat_out:.1f} °C",
    delta_color="off",
)

# Detailed Property Summary Table
st.markdown("##### Detailed Stream Enthalpy Breakdown")
st.table(
    {
        "Stream": ["HP Steam Inlet", "Feedwater Spray", "LP Steam Outlet"],
        f"Pressure ({unit_label})": [
            f"{p_in_raw:.2f}",
            f"{p_fw_raw:.2f}",
            f"{p_out_raw:.2f}",
        ],
        "Temperature (°C)": [
            f"{temperature_steam_inlet:.1f}",
            f"{temperature_feedwater_inlet:.1f}",
            f"{temperature_steam_outlet:.1f}",
        ],
        "Specific Enthalpy (kJ/kg)": [
            f"{h_in:.2f}",
            f"{h_fw:.2f}",
            f"{h_out:.2f}",
        ],
        "Mass Flow Rate (t/h)": [
            f"{mass_flow_steam_inlet:.2f}",
            f"{mass_flow_feedwater_inlet:.2f}",
            f"{mass_flow_steam_outlet:.2f}",
        ],
    }
)
