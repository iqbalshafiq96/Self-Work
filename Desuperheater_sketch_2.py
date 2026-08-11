"""
app.py
======
Streamlit front-end and diagram generator for the Letdown Steam System process graphic.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this file (app.py) and requirements.txt to your GitHub repo.
    2. On share.streamlit.io, point a new app at app.py in that repo.
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import streamlit as st

# ----------------------------------------------------------------------
# COLOUR PALETTE - softer / more formal than a raw DCS mimic
# ----------------------------------------------------------------------
BG_COLOR      = "#0f2438"   # deep slate blue
STEAM_COLOR   = "#eef3f7"   # near-white steam pipe
FW_COLOR      = "#4fb6e6"   # feedwater (soft sky blue)
EQUIP_COLOR   = "#f0b429"   # warm amber for valve / venturi outline
EQUIP_FILL    = "#16324a"   # subtle fill inside equipment shapes
TEXT_COLOR    = "#f2f6f9"
SUBTEXT_COLOR = "#9fb6c9"
TAG_EDGE      = "#3d5a73"

LW_PIPE  = 4.2
LW_EQUIP = 2.0


def build_figure(figsize=(15, 8.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def pipe(x1, y1, x2, y2, color=STEAM_COLOR, lw=LW_PIPE, zorder=2):
        ax.add_line(mlines.Line2D([x1, x2], [y1, y2], color=color, lw=lw,
                                   solid_capstyle="round", zorder=zorder))

    def flow_arrow(x, y, dx, dy, color=STEAM_COLOR):
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                     mutation_scale=15), zorder=3)

    def control_valve(x, y, tag_lines, size=0.34):
        s = size
        top, bot = (x, y + s), (x, y - s)
        left, right = (x - s, y), (x + s, y)
        for tri_pts in ([left, top, bot], [right, top, bot]):
            ax.add_patch(patches.Polygon(tri_pts, closed=True, facecolor=EQUIP_FILL,
                                         edgecolor=EQUIP_COLOR, lw=LW_EQUIP, zorder=4))
        ax.add_line(mlines.Line2D([x, x], [y + s, y + s + 0.38], color=EQUIP_COLOR,
                                   lw=LW_EQUIP, zorder=4))
        ax.add_patch(patches.Circle((x, y + s + 0.6), 0.22, facecolor=EQUIP_FILL,
                                     edgecolor=EQUIP_COLOR, lw=LW_EQUIP, zorder=4))
        ax.text(x, y - s - 0.28, tag_lines, ha="center", va="top", color=TEXT_COLOR,
                fontsize=10, fontweight="bold", zorder=5, linespacing=1.35)

    def venturi_desuperheater(cx, cy, w=4.2, r_in=0.62, r_throat=0.22, throat_w=0.7):
        """Converging - throat - diverging venturi section, drawn as a
        filled outline, with the feedwater spray nozzle entering at the
        throat."""
        x0 = cx - w / 2          # inlet full-bore
        x1 = cx - throat_w / 2   # start of throat
        x2 = cx + throat_w / 2   # end of throat
        x3 = cx + w / 2          # outlet full-bore

        top = [(x0, cy + r_in), (x1, cy + r_throat), (x2, cy + r_throat), (x3, cy + r_in)]
        bot = [(x3, cy - r_in), (x2, cy - r_throat), (x1, cy - r_throat), (x0, cy - r_in)]
        outline = top + bot
        ax.add_patch(patches.Polygon(outline, closed=True, facecolor=EQUIP_FILL,
                                     edgecolor=EQUIP_COLOR, lw=LW_EQUIP, zorder=4,
                                     joinstyle="round"))

        # centreline steam flow arrow through the throat
        flow_arrow(cx - 0.28, cy, 0.56, 0, color=STEAM_COLOR)

        # spray nozzle body sitting on top of the throat
        nozzle_w = 0.26
        nozzle = patches.Polygon(
            [(cx - nozzle_w, cy + r_throat + 0.55),
             (cx + nozzle_w, cy + r_throat + 0.55),
             (cx + 0.06, cy + r_throat + 0.05),
             (cx - 0.06, cy + r_throat + 0.05)],
            closed=True, facecolor=EQUIP_FILL, edgecolor=FW_COLOR, lw=LW_EQUIP, zorder=5)
        ax.add_patch(nozzle)

        # atomised spray fan, fanning downward from the nozzle tip into the throat
        tip_y = cy + r_throat + 0.02
        for ang, ln in [(-26, 0.20), (-9, 0.24), (9, 0.24), (26, 0.20)]:
            rad = math.radians(ang)
            dx, dy = ln * math.sin(rad), -ln * math.cos(rad)
            ax.add_line(mlines.Line2D([cx, cx + dx], [tip_y, tip_y + dy],
                                       color=FW_COLOR, lw=1.3, zorder=6))

        ax.text(cx, cy - r_in - 0.42, "SPRAY DESUPERHEATER\n(Venturi Type)", ha="center",
                va="top", color=TEXT_COLOR, fontsize=10.5, fontweight="bold",
                zorder=5, linespacing=1.4)

        return x0, x3, cy + r_throat + 0.55  # inlet x, outlet x, nozzle top y

    def label(x, y, txt, color=TEXT_COLOR, fs=11, weight="bold", ha="left", style="normal"):
        ax.text(x, y, txt, color=color, fontsize=fs, fontweight=weight, ha=ha,
                va="center", zorder=6, style=style)

    def tag_box(x, y, txt, color=SUBTEXT_COLOR, edge=TAG_EDGE):
        ax.text(x, y, txt, color=color, fontsize=8.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=edge, lw=1),
                zorder=6, linespacing=1.3)

    # ------------------------------------------------------------------
    # Title block
    # ------------------------------------------------------------------
    ax.text(8, 8.75, "Letdown Steam System", ha="center", va="center",
            color=TEXT_COLOR, fontsize=19, fontweight="bold")
    ax.text(8, 8.32, "High Pressure to Low Pressure Steam Letdown with Spray Desuperheating",
            ha="center", va="center", color=SUBTEXT_COLOR, fontsize=11, style="italic")

    Y = 4.6

    # ------------------------------------------------------------------
    # HP steam line (inlet)
    # ------------------------------------------------------------------
    pipe(0.6, Y, 4.1, Y)
    flow_arrow(1.6, Y, 0.8, 0)
    label(0.6, Y + 0.55, "High Pressure Steam Line", fs=11.5)
    tag_box(1.9, Y - 0.55, "HP Steam\n~ 40 barg")

    # ------------------------------------------------------------------
    # Pressure control valve
    # ------------------------------------------------------------------
    pcv_x = 5.0
    control_valve(pcv_x, Y, "Pressure Control Valve\n(PCV)")
    pipe(4.1, Y, pcv_x - 0.34, Y)
    pipe(pcv_x + 0.34, Y, 6.1, Y)
    flow_arrow(5.65, Y, 0.35, 0)

    # ------------------------------------------------------------------
    # Venturi spray desuperheater
    # ------------------------------------------------------------------
    vessel_x = 8.7
    v_in, v_out, nozzle_top = venturi_desuperheater(vessel_x, Y)
    pipe(6.1, Y, v_in, Y)
    pipe(v_out, Y, 10.3, Y)
    flow_arrow(10.0, Y, 0.28, 0)

    # Feedwater spray supply line down to the nozzle
    fw_top = 7.35
    pipe(vessel_x, fw_top, vessel_x, nozzle_top, color=FW_COLOR)
    flow_arrow(vessel_x, 7.0, 0, -0.42, color=FW_COLOR)
    pipe(6.6, fw_top, vessel_x, fw_top, color=FW_COLOR)
    flow_arrow(7.2, fw_top, 0.4, 0, color=FW_COLOR)
    label(6.6, fw_top + 0.42, "Feedwater Spray Line", color=FW_COLOR, fs=11.5)
    tag_box(7.35, fw_top - 0.42, "BFW Supply", edge="#2e6a86")

    # ------------------------------------------------------------------
    # LP steam line (outlet)
    # ------------------------------------------------------------------
    pipe(10.3, Y, 15.4, Y)
    flow_arrow(14.2, Y, 0.8, 0)
    label(12.05, Y + 0.55, "Low Pressure Steam Line", fs=11.5)
    tag_box(13.7, Y - 0.55, "LP Steam\n~ 12 barg")

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    leg_x, leg_y = 0.6, 1.55
    legend_items = [
        (STEAM_COLOR, "Steam Line"),
        (FW_COLOR, "Feedwater / Spray Line"),
        (EQUIP_COLOR, "Equipment Outline"),
    ]
    for i, (c, txt) in enumerate(legend_items):
        yy = leg_y - i * 0.42
        ax.add_line(mlines.Line2D([leg_x, leg_x + 0.5], [yy, yy], color=c, lw=3.2))
        ax.text(leg_x + 0.68, yy, txt, color=TEXT_COLOR, fontsize=9.5, va="center")

    # Outer frame, rounded, soft
    frame = patches.FancyBboxPatch((0.18, 0.18), 15.64, 8.84,
                                    boxstyle="round,pad=0,rounding_size=0.18",
                                    fill=False, edgecolor="#2c4a63", lw=1.4)
    ax.add_patch(frame)

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# STREAMLIT INTERFACE
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Letdown Steam System",
    page_icon="♨️",
    layout="wide",
)

# Match page background to figure background
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {BG_COLOR}; }}
        [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}
    </style>
    """,
    unsafe_allow_html=True,
)

fig = build_figure()
st.pyplot(fig, use_container_width=True)

with st.expander("About this system"):
    st.markdown(
        """
        This graphic shows a typical **HP-to-LP steam letdown station**:

        - **High Pressure Steam Line** — supplies steam from the upstream
          high pressure header.
        - **Pressure Control Valve (PCV)** — reduces steam pressure from
          HP to the LP setpoint.
        - **Spray Desuperheater (venturi type)** — feedwater is injected
          at the throat of a venturi, where high steam velocity promotes
          rapid atomisation and mixing, cooling the steam toward
          saturation.
        - **Low Pressure Steam Line** — carries the reduced-pressure,
          desuperheated steam onward to the LP distribution header.
        """
    )
