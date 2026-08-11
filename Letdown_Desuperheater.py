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
    figsize=(15, 4.2),  # Reduced height ratio to match tightened bounds
):
    fig, ax = plt.subplots(figsize=figsize)

    # Set background to transparent so it blends with Streamlit theme
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Inherit standard theme text color dynamically
    text_color = plt.rcParams.get("text.color", "currentColor")

    ax.set_xlim(0, 16)
    # Tighter vertical limits to remove top/bottom white space
    ax.set_ylim(3.2, 8.5)
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

        # Atomized Spray Cone
        spray_y_top = cy + r_throat + 0.05
        spray_y_bot = cy - r_throat + 0.05
        ax.add_line(
            mlines.Line2D(
                [cx, cx - 0.35],
                [spray_y_top, spray_y_bot],
                color=FW_COLOR,
                linestyle="--",
                lw=1.2,
                zorder=5,
            )
        )
        ax.add_line(
            mlines.Line2D(
                [cx, cx],
                [spray_y_top, spray_y_bot],
                color=FW_COLOR,
                linestyle="--",
                lw=1.2,
                zorder=5,
            )
        )
        ax.add_line(
            mlines.Line2D(
                [cx, cx + 0.35],
                [spray_y_top, spray_y_bot],
                color=FW_COLOR,
                linestyle="--",
                lw=1.2,
                zorder=5,
            )
        )

        # Label aligned on same y-axis height as control valve label
        ax.text(
            cx,
            cy - 0.62,
            "Desuperheater",
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
    control_valve(pcv_x, Y, "Isenthalpic Expansion")
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
    # LP Steam Line (Outlet)
    # ------------------------------------------------------------------
    pipe(v_out, Y, 15.4, Y)
    flow_arrow(14.2, Y, 0.8, 0)
    outlet_txt = (
        f"Low Pressure Steam Line\nFlow: {m_out:.2f} t/h\nPress: {p_out:.2f}"
        f" {p_unit}\nTemp: {t_out:.1f} °C"
    )
    label(11.8, Y + 0.95, outlet_txt, fs=9.5)

    # Zero-margin layout to strip surrounding whitespace
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig
