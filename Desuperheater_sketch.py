import matplotlib.patches as patches
import matplotlib.pyplot as plt
import streamlit as st


def draw_letdown_system():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Main Steam Line (Inlet to Valve, Valve to Venturi, Venturi Outlet)
    ax.annotate(
        "",
        xy=(3.0, 3.0),
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

    # --- DETAILED VENTURI DESUPERHEATER (Hollow / No Fill) ---
    # Top Outer Wall Profile (Inlet, Converging Section, Throat, Diverging Section, Outlet)
    venturi_top_x = [6.5, 7.0, 7.4, 7.6, 8.0, 8.5]
    venturi_top_y = [3.8, 3.8, 3.3, 3.3, 3.8, 3.8]
    ax.plot(venturi_top_x, venturi_top_y, color="black", lw=2)

    # Bottom Outer Wall Profile (Mirrored)
    venturi_bottom_x = [6.5, 7.0, 7.4, 7.6, 8.0, 8.5]
    venturi_bottom_y = [2.2, 2.2, 2.7, 2.7, 2.2, 2.2]
    ax.plot(venturi_bottom_x, venturi_bottom_y, color="black", lw=2)

    # Internal Venturi Throat Contour Lines (Inner Nozzle Walls)
    ax.plot(
        [6.8, 7.3, 7.7, 8.2], [3.6, 3.2, 3.2, 3.6], color="gray", lw=1.5, ls="--"
    )
    ax.plot(
        [6.8, 7.3, 7.7, 8.2], [2.4, 2.8, 2.8, 2.4], color="gray", lw=1.5, ls="--"
    )

    # Feedwater Line Pipe into Throat
    ax.annotate(
        "",
        xy=(7.5, 3.0),
        xytext=(7.5, 5.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="teal"),
    )

    # Internal Spray Nozzle Tip (At Venturi Throat)
    nozzle_tip = patches.Polygon(
        [[7.35, 3.15], [7.65, 3.15], [7.5, 2.95]],
        color="white",
        ec="teal",
        lw=1.5,
    )
    ax.add_patch(nozzle_tip)

    # Atomized Spray Representation (Dotted spray cone)
    ax.plot(
        [7.5, 7.2], [2.95, 2.65], color="teal", linestyle=":", linewidth=1.5
    )
    ax.plot(
        [7.5, 7.5], [2.95, 2.6], color="teal", linestyle=":", linewidth=1.5
    )
    ax.plot(
        [7.5, 7.8], [2.95, 2.65], color="teal", linestyle=":", linewidth=1.5
    )

    # Text Labels
    ax.text(
        0.5, 3.3, "HP Steam In", fontsize=10, color="crimson", fontweight="bold"
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
        "Venturi Desuperheater",
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        10.0,
        3.3,
        "Outlet Steam",
        fontsize=10,
        color="dodgerblue",
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


st.title("Letdown Steam System Custom Sketch")
fig = draw_letdown_system()
st.pyplot(fig)
