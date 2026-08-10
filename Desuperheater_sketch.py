import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st


def draw_letdown_system_probe():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Main Steam Line (Continuous straight pipe)
    ax.plot([0.5, 11.5], [3.5, 3.5], color="black", lw=2)  # Top pipe wall
    ax.plot([0.5, 11.5], [2.5, 2.5], color="black", lw=2)  # Bottom pipe wall

    # Steam Flow Arrows
    ax.annotate(
        "",
        xy=(2.5, 3.0),
        xytext=(0.5, 3.0),
        arrowprops=dict(arrowstyle="->", lw=2, color="crimson"),
    )
    ax.annotate(
        "",
        xy=(6.5, 3.0),
        xytext=(4.0, 3.0),
        arrowprops=dict(arrowstyle="->", lw=2, color="orange"),
    )
    ax.annotate(
        "",
        xy=(11.5, 3.0),
        xytext=(8.5, 3.0),
        arrowprops=dict(arrowstyle="->", lw=2, color="dodgerblue"),
    )

    # --- CONTROL VALVE (P&ID Diaphragm) ---
    body_left = patches.Polygon(
        [[3.0, 2.5], [3.0, 3.5], [3.5, 3.0]], color="white", ec="black", lw=2
    )
    body_right = patches.Polygon(
        [[4.0, 2.5], [4.0, 3.5], [3.5, 3.0]], color="white", ec="black", lw=2
    )
    ax.add_patch(body_left)
    ax.add_patch(body_right)
    ax.plot([3.5, 3.5], [3.0, 4.2], color="black", lw=2)
    actuator = patches.Wedge(
        (3.5, 4.2), r=0.4, theta1=0, theta2=180, color="white", ec="black", lw=2
    )
    ax.add_patch(actuator)

    # --- SPRAY NOZZLE PROBE DESUPERHEATER ---
    # Flange connection on top of the main line
    ax.plot([7.1, 7.9], [3.6, 3.6], color="black", lw=3)  # Flange base

    # Feedwater Line & Probe Pipe entering steam line
    ax.annotate(
        "",
        xy=(7.5, 3.1),
        xytext=(7.5, 5.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="teal"),
    )

    # Injection Probe Body (Extension inside pipe)
    ax.plot([7.4, 7.4], [3.5, 2.9], color="black", lw=1.5)
    ax.plot([7.6, 7.6], [3.5, 2.9], color="black", lw=1.5)

    # Spray Nozzle Tip (Facing downstream/right)
    nozzle = patches.Polygon(
        [[7.6, 3.1], [7.6, 2.7], [7.9, 2.9]], color="white", ec="teal", lw=1.5
    )
    ax.add_patch(nozzle)

    # Atomized Spray Cone (Downstream dispersion)
    ax.plot(
        [7.9, 8.8], [2.9, 3.4], color="teal", linestyle=":", linewidth=1.5
    )
    ax.plot([7.9, 9.0], [2.9, 2.9], color="teal", linestyle=":", linewidth=1.5)
    ax.plot(
        [7.9, 8.8], [2.9, 2.4], color="teal", linestyle=":", linewidth=1.5
    )

    # Text Labels
    ax.text(
        0.5, 3.8, "HP Steam In", fontsize=10, color="crimson", fontweight="bold"
    )
    ax.text(
        3.5,
        1.8,
        "Control Valve\n(Pneumatic)",
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        7.5,
        5.7,
        "Feedwater In",
        fontsize=10,
        color="teal",
        fontweight="bold",
        ha="center",
    )
    ax.text(
        7.5,
        1.6,
        "Spray Probe Desuperheater",
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        10.0,
        3.8,
        "Outlet Steam",
        fontsize=10,
        color="dodgerblue",
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


st.title("Letdown Steam System - Spray Probe Design")
fig = draw_letdown_system_probe()
st.pyplot(fig)
