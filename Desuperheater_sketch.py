import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st


def draw_letdown_system():
    # Set global formal font settings
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # --- PROCESS PIPING (P&ID Standard Lines) ---
    # Inlet Steam Line
    ax.plot([0.5, 3.0], [3.0, 3.0], color="#111111", lw=2)
    ax.annotate(
        "",
        xy=(1.8, 3.0),
        xytext=(1.2, 3.0),
        arrowprops=dict(
            arrowstyle="-|>", mutation_scale=12, color="#111111", lw=1.5
        ),
    )

    # Intermediate Steam Line (Valve to Venturi)
    ax.plot([4.0, 6.5], [3.0, 3.0], color="#111111", lw=2)
    ax.annotate(
        "",
        xy=(5.4, 3.0),
        xytext=(4.8, 3.0),
        arrowprops=dict(
            arrowstyle="-|>", mutation_scale=12, color="#111111", lw=1.5
        ),
    )

    # Outlet Steam Line
    ax.plot([8.5, 11.5], [3.0, 3.0], color="#111111", lw=2)
    ax.annotate(
        "",
        xy=(10.2, 3.0),
        xytext=(9.6, 3.0),
        arrowprops=dict(
            arrowstyle="-|>", mutation_scale=12, color="#111111", lw=1.5
        ),
    )

    # Feedwater Line
    ax.plot([7.5, 7.5], [3.3, 5.2], color="#005f73", lw=2)
    ax.annotate(
        "",
        xy=(7.5, 3.8),
        xytext=(7.5, 4.4),
        arrowprops=dict(
            arrowstyle="-|>", mutation_scale=12, color="#005f73", lw=1.5
        ),
    )

    # --- P&ID DIAPHRAGM CONTROL VALVE (CV-101) ---
    body_left = patches.Polygon(
        [[3.0, 2.5], [3.0, 3.5], [3.5, 3.0]],
        color="white",
        ec="#111111",
        lw=1.8,
        zorder=3,
    )
    body_right = patches.Polygon(
        [[4.0, 2.5], [4.0, 3.5], [3.5, 3.0]],
        color="white",
        ec="#111111",
        lw=1.8,
        zorder=3,
    )
    ax.add_patch(body_left)
    ax.add_patch(body_right)

    # Stem and Actuator
    ax.plot([3.5, 3.5], [3.0, 4.3], color="#111111", lw=1.8, zorder=3)
    actuator = patches.Wedge(
        (3.5, 4.3),
        r=0.45,
        theta1=0,
        theta2=180,
        color="white",
        ec="#111111",
        lw=1.8,
        zorder=3,
    )
    ax.add_patch(actuator)

    # --- VENTURI DESUPERHEATER (No Fill / Hollow Line Drawing) ---
    # Top Outer Profile
    venturi_top_x = [6.5, 6.9, 7.3, 7.7, 8.1, 8.5]
    venturi_top_y = [3.0, 3.7, 3.3, 3.3, 3.7, 3.0]
    ax.plot(venturi_top_x, venturi_top_y, color="#111111", lw=1.8, zorder=3)

    # Bottom Outer Profile
    venturi_bottom_x = [6.5, 6.9, 7.3, 7.7, 8.1, 8.5]
    venturi_bottom_y = [3.0, 2.3, 2.7, 2.7, 2.3, 3.0]
    ax.plot(
        venturi_bottom_x, venturi_bottom_y, color="#111111", lw=1.8, zorder=3
    )

    # Internal Spray Nozzle Tip
    nozzle_tip = patches.Polygon(
        [[7.38, 3.3], [7.62, 3.3], [7.5, 3.05]],
        color="white",
        ec="#005f73",
        lw=1.5,
        zorder=4,
    )
    ax.add_patch(nozzle_tip)

    # Internal Spray Cone
    ax.plot(
        [7.5, 7.25], [3.05, 2.8], color="#005f73", ls=":", lw=1.2, zorder=4
    )
    ax.plot([7.5, 7.5], [3.05, 2.75], color="#005f73", ls=":", lw=1.2, zorder=4)
    ax.plot(
        [7.5, 7.75], [3.05, 2.8], color="#005f73", ls=":", lw=1.2, zorder=4
    )

    # --- FORMAL TECHNICAL LABELS ---
    label_style = {
        "fontsize": 9,
        "fontfamily": "sans-serif",
        "fontweight": "semibold",
    }

    # Process Points
    ax.text(
        0.5,
        3.25,
        "HP STEAM INLET\n(HP-01)",
        color="#ae2012",
        ha="left",
        **label_style,
    )
    ax.text(
        7.5,
        5.4,
        "FEEDWATER INLET\n(FW-01)",
        color="#005f73",
        ha="center",
        **label_style,
    )
    ax.text(
        11.5,
        3.25,
        "DESUPERHEATED STEAM\n(LP-01)",
        color="#0a9396",
        ha="right",
        **label_style,
    )

    # Equipment Tags
    ax.text(
        3.5,
        1.6,
        "PRESSURE REDUCING VALVE\n[PCV-101]",
        ha="center",
        color="#111111",
        **label_style,
    )
    ax.text(
        7.5,
        1.6,
        "VENTURI DESUPERHEATER\n[DS-101]",
        ha="center",
        color="#111111",
        **label_style,
    )

    plt.tight_layout()
    return fig


st.title("Letdown Steam System - Process Schematic")
fig = draw_letdown_system()
st.pyplot(fig)
