import io
import math
from iapws import IAPWS97
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------
# COLOUR PALETTE & STYLE CONFIGURATION
# ----------------------------------------------------------------------
STEAM_COLOR = "#64748B"  # Slate grey for steam piping
FW_COLOR = "#0EA5E9"  # Vivid sky blue for feedwater spray
EQUIP_COLOR = "#D97706"  # Amber for control valve & desuperheater outline
EQUIP_FILL = "none"  # Transparent background integration

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
    # Set high-DPI font parameters for technical legibility (Scaled up for Streamlit alignment)
    plt.rcParams.update(
        {
            "font.sans-serif": ["Segoe UI", "Aptos", "Arial", "DejaVu Sans"],
            "font.family": "sans-serif",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Set background to fully transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Technical text color
    text_color = "#334155"

    # Tight bounding limits
    ax.set_xlim(0.4, 15.6)
    ax.set_ylim(3.4, 9.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Drawing Helper Functions
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
        # Actuator Stem
        ax.add_line(
            mlines.Line2D(
                [x, x],
                [y + s, y + s + 0.38],
                color=EQUIP_COLOR,
                lw=LW_EQUIP,
                zorder=4,
            )
        )
        # Actuator Diaphragm/Dome
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

        # Atomized Spray Cone lines
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

    # HP Steam Line (Inlet)
    pipe(0.6, Y, 4.1, Y)
    flow_arrow(1.6, Y, 0.8, 0)
    inlet_txt = (
        f"High Pressure Steam Line\nFlow: {m_in:.2f} t/h\nPress: {p_in:.2f}"
        f" {p_unit}\nTemp: {t_in:.1f} °C"
    )
    label(0.6, Y + 1.25, inlet_txt)

    # Pressure Control Valve
    pcv_x = 5.0
    control_valve(pcv_x, Y, "Isenthalpic Expansion")
    pipe(4.1, Y, pcv_x - 0.34, Y)
    pipe(pcv_x + 0.34, Y, 6.1, Y)
    flow_arrow(5.65, Y, 0.35, 0)

    # Venturi Spray Desuperheater
    vessel_x = 8.7
    v_in, v_out = venturi_desuperheater(vessel_x, Y)
    pipe(6.1, Y, v_in, Y)

    # Feedwater Spray Line
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

    # LP Steam Line (Outlet)
    pipe(v_out, Y, 15.4, Y)
    flow_arrow(14.2, Y, 0.8, 0)
    outlet_txt = (
        f"Low Pressure Steam Line\nFlow: {m_out:.2f} t/h\nPress: {p_out:.2f}"
        f" {p_unit}\nTemp: {t_out:.1f} °C"
    )
    label(11.8, Y + 1.25, outlet_txt)

    # Render Matplotlib figure to SVG memory buffer
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
# STREAMLIT DISPLAY BLOCK (Resized layout ratio)
# ----------------------------------------------------------------------
# Render SVG Process Flow Diagram centered and reduced in width
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

# Narrowed column ratio scales down the diagram width while keeping the aspect ratio
col_left, col_center, col_right = st.columns([0.28, 0.44, 0.28])
with col_center:
    st.image(svg_data, use_container_width=True)
