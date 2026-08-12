import io
import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# COLOUR PALETTE & STYLE CONFIGURATION
# ----------------------------------------------------------------------

STEAM_COLOR = "#64748B"       # Slate grey for steam piping
STEAM_GLOW = "#E2E8F0"        # Bright steam particle

FW_COLOR = "#0EA5E9"          # Feedwater
FW_GLOW = "#38BDF8"           # Bright feedwater particle

EQUIP_COLOR = "#D97706"       # Amber equipment
EQUIP_FILL = "none"

LW_PIPE = 3.36
LW_EQUIP = 1.6


# ======================================================================
# OPTIMIZED ANIMATION HELPER
# ======================================================================

def add_flow_animation(
    svg,
    steam_segments,
    feedwater_segments,
):
    """
    Adds GPU-accelerated animated glowing flows to the SVG.
    Uses CSS stroke-dasharray and keyframes instead of heavy SMIL animateMotion.
    """

    animation_svg = f"""
    <style>
        /* Modern, GPU-accelerated glowing flow effects */
        .steam-flow-line {{
            stroke: {STEAM_GLOW};
            stroke-dasharray: 6, 12;
            stroke-linecap: round;
            animation: steamFlow 1.2s linear infinite;
            filter: drop-shadow(0px 0px 2.5px {STEAM_GLOW});
            opacity: 0.85;
        }}

        .water-flow-line {{
            stroke: {FW_GLOW};
            stroke-dasharray: 5, 10;
            stroke-linecap: round;
            animation: waterFlow 0.9s linear infinite;
            filter: drop-shadow(0px 0px 3px {FW_GLOW});
            opacity: 0.9;
        }}

        /* Keyframes for smooth continuous directional flow */
        @keyframes steamFlow {{
            0% {{ stroke-dashoffset: 36; }}
            100% {{ stroke-dashoffset: 0; }}
        }}

        @keyframes waterFlow {{
            0% {{ stroke-dashoffset: 30; }}
            100% {{ stroke-dashoffset: 0; }}
        }}
    </style>

    <g id="animated-flow-layer">
    """

    # --------------------------------------------------------------
    # STEAM FLOW LINES
    # --------------------------------------------------------------
    for x1, y1, x2, y2, duration, _ in steam_segments:
        animation_svg += f"""
        <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" 
              class="steam-flow-line" stroke-width="2.4" 
              style="animation-duration: {duration:.2f}s;" />
        """

    # --------------------------------------------------------------
    # FEEDWATER FLOW LINES
    # --------------------------------------------------------------
    for x1, y1, x2, y2, duration, _ in feedwater_segments:
        animation_svg += f"""
        <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" 
              class="water-flow-line" stroke-width="2.2" 
              style="animation-duration: {duration:.2f}s;" />
        """

    animation_svg += "</g>\n"

    # Insert animation before SVG closing tag
    svg = svg.replace(
        "</svg>",
        animation_svg + "</svg>"
    )

    return svg


# ======================================================================
# MAIN FIGURE
# ======================================================================

def build_svg_figure(
    p_in=100.0,
    t_in=500.0,
    m_in=50.0,
    p_fw=110.0,
    t_fw=180.0,
    m_fw=5.0,
    p_out=40.0,
    t_out=350.0,
    m_out=55.0,
    p_unit="bar",
    figsize=(14, 3.4),
):

    # --------------------------------------------------------------
    # Font configuration
    # --------------------------------------------------------------

    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Segoe UI",
                "Aptos",
                "Arial",
                "DejaVu Sans",
            ],
            "font.family": "sans-serif",
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    text_color = "#334155"

    # --------------------------------------------------------------
    # ORIGINAL COORDINATE SYSTEM
    # --------------------------------------------------------------

    ax.set_xlim(0.4, 15.6)
    ax.set_ylim(3.6, 9.0)

    ax.set_aspect("equal")
    ax.axis("off")

    # --------------------------------------------------------------
    # DRAWING HELPERS
    # --------------------------------------------------------------

    def pipe(
        x1,
        y1,
        x2,
        y2,
        color=STEAM_COLOR,
        lw=LW_PIPE,
        zorder=2,
    ):
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

    def flow_arrow(
        x,
        y,
        dx,
        dy,
        color=STEAM_COLOR,
    ):
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

    def control_valve(
        x,
        y,
        tag_lines,
        size=0.34,
    ):
        s = size

        top = (x, y + s)
        bot = (x, y - s)

        left = (x - s, y)
        right = (x + s, y)

        for tri_pts in (
            [left, top, bot],
            [right, top, bot],
        ):
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

        # Actuator stem
        ax.add_line(
            mlines.Line2D(
                [x, x],
                [
                    y + s,
                    y + s + 0.38,
                ],
                color=EQUIP_COLOR,
                lw=LW_EQUIP,
                zorder=4,
            )
        )

        # Actuator dome
        ax.add_patch(
            patches.Circle(
                (
                    x,
                    y + s + 0.6,
                ),
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
        cx,
        cy,
        w=4.2,
        r_in=0.62,
        r_throat=0.22,
        throat_w=0.7,
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

        # Spray cone
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

    def label(
        x,
        y,
        txt,
        color=text_color,
        fs=7.6,
        ha="left",
    ):
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

    # ==================================================================
    # PROCESS DRAWING
    # ==================================================================

    Y = 4.6

    # HP STEAM LINE
    pipe(0.6, Y, 4.1, Y)
    flow_arrow(1.6, Y, 0.8, 0)
    inlet_txt = (
        f"High Pressure Steam Line\n"
        f"Flow: {m_in:.2f} t/h\n"
        f"Press: {p_in:.2f} {p_unit}\n"
        f"Temp: {t_in:.1f} °C"
    )
    label(0.6, Y + 1.15, inlet_txt)

    # PCV
    pcv_x = 5.0
    control_valve(pcv_x, Y, "Isenthalpic Expansion")
    pipe(4.1, Y, pcv_x - 0.34, Y)
    pipe(pcv_x + 0.34, Y, 6.1, Y)
    flow_arrow(5.65, Y, 0.35, 0)

    # DESUPERHEATER
    vessel_x = 8.7
    v_in, v_out = venturi_desuperheater(vessel_x, Y)
    pipe(6.1, Y, v_in, Y)

    # FEEDWATER
    fw_top = 7.35
    pipe(vessel_x, fw_top, vessel_x, Y, color=FW_COLOR, zorder=3)
    flow_arrow(vessel_x, 6.2, 0, -0.42, color=FW_COLOR)
    pipe(6.6, fw_top, vessel_x, fw_top, color=FW_COLOR)
    flow_arrow(7.2, fw_top, 0.4, 0, color=FW_COLOR)

    fw_txt = (
        f"Feedwater Spray Line\n"
        f"Flow: {m_fw:.2f} t/h\n"
        f"Press: {p_fw:.2f} {p_unit}\n"
        f"Temp: {t_fw:.1f} °C"
    )
    label(6.6, fw_top + 0.85, fw_txt, color=FW_COLOR)

    # LP STEAM
    pipe(v_out, Y, 15.4, Y)
    flow_arrow(14.2, Y, 0.8, 0)
    outlet_txt = (
        f"Low Pressure Steam Line\n"
        f"Flow: {m_out:.2f} t/h\n"
        f"Press: {p_out:.2f} {p_unit}\n"
        f"Temp: {t_out:.1f} °C"
    )
    label(11.8, Y + 1.15, outlet_txt)

    # ==================================================================
    # RENDER MATPLOTLIB SVG
    # ==================================================================

    svg_buffer = io.StringIO()
    fig.savefig(
        svg_buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig)
    svg = svg_buffer.getvalue()

    # ==================================================================
    # CONVERT COORDINATES
    # ==================================================================

    def convert_point(x, y):
        px, py = ax.transData.transform((x, y))
        svg_height = fig.get_figheight() * fig.dpi
        py = svg_height - py
        return px, py

    viewbox_match = re.search(r'viewBox="([^"]+)"', svg)
    if viewbox_match:
        vb = [float(x) for x in viewbox_match.group(1).split()]
        vb_x, vb_y, vb_w, vb_h = vb
    else:
        vb_x = 0
        vb_y = 0

    steam_segments = []
    feedwater_segments = []

    def make_segment(x1, y1, x2, y2, duration, offset):
        p1 = convert_point(x1, y1)
        p2 = convert_point(x2, y2)
        return (
            p1[0] - vb_x,
            p1[1] - vb_y,
            p2[0] - vb_x,
            p2[1] - vb_y,
            duration,
            offset,
        )

    # --------------------------------------------------------------
    # GENERATE SEGMENTS
    # --------------------------------------------------------------
    steam_segments.append(make_segment(0.6, Y, 4.1, Y, 3.0, 0.0))
    steam_segments.append(make_segment(5.34, Y, 6.1, Y, 1.0, 0.0))
    steam_segments.append(make_segment(6.1, Y, v_in, Y, 1.5, 0.0))
    steam_segments.append(make_segment(v_out, Y, 15.4, Y, 3.8, 0.0))

    feedwater_segments.append(make_segment(6.6, fw_top, vessel_x, fw_top, 1.0, 0.0))
    feedwater_segments.append(make_segment(vessel_x, fw_top, vessel_x, Y, 1.7, 0.0))

    # Add optimized SVG animation
    svg = add_flow_animation(svg, steam_segments, feedwater_segments)

    return svg


# ======================================================================
# EXECUTION / SAVE TO FILE
# ======================================================================

if __name__ == "__main__":
    final_svg = build_svg_figure()
    
    # Save the output to an SVG file
    with open("optimized_desuperheater.svg", "w", encoding="utf-8") as f:
        f.write(final_svg)
        
    print("Done bos! SVG file saved as 'optimized_desuperheater.svg'")
