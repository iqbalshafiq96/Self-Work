import io
import re
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

LW_PIPE = 3.36
LW_EQUIP = 1.6


def inject_particle_effects(svg_xml):
    """Injects CSS animations and dynamic SVG elements for steam & spray particle effects."""
    css_and_particles = """
    <style>
        @keyframes steam-flow {
            0% { transform: translateX(0px); opacity: 0; }
            20% { opacity: 0.85; }
            80% { opacity: 0.85; }
            100% { transform: translateX(120px); opacity: 0; }
        }
        @keyframes spray-inject {
            0% { transform: translateY(0px) scale(0.6); opacity: 0; }
            30% { opacity: 0.9; }
            100% { transform: translateY(45px) translateX(35px) scale(1.4); opacity: 0; }
        }
        @keyframes atomized-mix {
            0% { transform: translateX(0px) translateY(0px); opacity: 0; }
            25% { opacity: 0.8; }
            100% { transform: translateX(90px) translateY(var(--dy)); opacity: 0; }
        }

        .particle-steam {
            animation: steam-flow 2.4s infinite linear;
            fill: #94A3B8;
        }
        .particle-spray {
            animation: spray-inject 1.2s infinite ease-in;
            fill: #0EA5E9;
        }
        .particle-mix {
            animation: atomized-mix 1.8s infinite ease-out;
            fill: #38BDF8;
        }
    </style>
    <g id="particle-layer">
        <!-- HP Steam Particles -->
        <circle cx="120" cy="220" r="2.5" class="particle-steam" style="animation-delay: 0.0s;" />
        <circle cx="120" cy="223" r="2.0" class="particle-steam" style="animation-delay: 0.8s;" />
        <circle cx="120" cy="217" r="1.8" class="particle-steam" style="animation-delay: 1.6s;" />

        <!-- Feedwater Spray Droplets (Vertical Inject) -->
        <circle cx="568" cy="80" r="2.2" class="particle-spray" style="animation-delay: 0.0s;" />
        <circle cx="568" cy="80" r="1.8" class="particle-spray" style="animation-delay: 0.4s;" />
        <circle cx="568" cy="80" r="2.5" class="particle-spray" style="animation-delay: 0.8s;" />

        <!-- Mixing / Atomized Steam-Water Particles (Post Venturi) -->
        <circle cx="575" cy="220" r="2.2" class="particle-mix" style="--dy: -6px; animation-delay: 0.1s;" />
        <circle cx="575" cy="220" r="1.5" class="particle-mix" style="--dy: 5px; animation-delay: 0.5s;" />
        <circle cx="575" cy="220" r="2.8" class="particle-mix" style="--dy: -2px; animation-delay: 0.9s;" />
        <circle cx="575" cy="220" r="1.8" class="particle-mix" style="--dy: 8px; animation-delay: 1.3s;" />
    </g>
    """
    # Insert particle layer right before closing </svg> tag
    return svg_xml.replace("</svg>", f"{css_and_particles}\n</svg>")


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
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    text_color = "#334155"

    ax.set_xlim(0.4, 15.6)
    ax.set_ylim(3.6, 9.0)
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
                arrowstyle="-|>",
                color=color,
                lw=1.44,
                mutation_scale=12,
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
            fontsize=8.0,
            fontweight="medium",
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
                lw=0.96,
                zorder=5,
            )
        )
        ax.add_line(
            mlines.Line2D(
                [spray_origin_x, spray_origin_x + spray_len],
                [spray_origin_y, spray_origin_y + 0.18],
                color=FW_COLOR,
                linestyle="--",
                lw=0.96,
                zorder=5,
            )
        )
        ax.add_line(
            mlines.Line2D(
                [spray_origin_x, spray_origin_x + spray_len],
                [spray_origin_y, spray_origin_y - 0.18],
                color=FW_COLOR,
                linestyle="--",
                lw=0.96,
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
            fontsize=8.0,
            fontweight="medium",
            zorder=5,
        )

        return x0, x3

    def label(x, y, txt, color=text_color, fs=7.6, ha="left"):
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

    # HP Steam Line
    pipe(0.6, Y, 4.1, Y)
    flow_arrow(1.6, Y, 0.8, 0)
    inlet_txt = (
        f"High Pressure Steam Line\nFlow: {m_in:.2f} t/h\nPress: {p_in:.2f}"
        f" {p_unit}\nTemp: {t_in:.1f} °C"
    )
    label(0.6, Y + 1.15, inlet_txt)

    # Control Valve
    pcv_x = 5.0
    control_valve(pcv_x, Y, "Isenthalpic Expansion")
    pipe(4.1, Y, pcv_x - 0.34, Y)
    pipe(pcv_x + 0.34, Y, 6.1, Y)
    flow_arrow(5.65, Y, 0.35, 0)

    # Desuperheater
    vessel_x = 8.7
    v_in, v_out = venturi_desuperheater(vessel_x, Y)
    pipe(6.1, Y, v_in, Y)

    # Feedwater Line
    fw_top = 7.35
    pipe(vessel_x, fw_top, vessel_x, Y, color=FW_COLOR, zorder=3)
    flow_arrow(vessel_x, 6.2, 0, -0.42, color=FW_COLOR)
    pipe(6.6, fw_top, vessel_x, fw_top, color=FW_COLOR)
    flow_arrow(7.2, fw_top, 0.4, 0, color=FW_COLOR)
    fw_txt = (
        f"Feedwater Spray Line\nFlow: {m_fw:.2f} t/h\nPress: {p_fw:.2f}"
        f" {p_unit}\nTemp: {t_fw:.1f} °C"
    )
    label(6.6, fw_top + 0.85, fw_txt, color=FW_COLOR)

    # LP Steam Line
    pipe(v_out, Y, 15.4, Y)
    flow_arrow(14.2, Y, 0.8, 0)
    outlet_txt = (
        f"Low Pressure Steam Line\nFlow: {m_out:.2f} t/h\nPress: {p_out:.2f}"
        f" {p_unit}\nTemp: {t_out:.1f} °C"
    )
    label(11.8, Y + 1.15, outlet_txt)

    # Save to buffer
    svg_buffer = io.StringIO()
    fig.savefig(
        svg_buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig)

    # Inject CSS/SVG animation effects before returning
    raw_svg = svg_buffer.getvalue()
    return inject_particle_effects(raw_svg)
