import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

def draw_letdown_system():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Main Steam Line
    ax.annotate("", xy=(3, 3), xytext=(0.5, 3), arrowprops=dict(arrowstyle="->", lw=2, color="crimson"))
    ax.annotate("", xy=(6.5, 3), xytext=(4, 3), arrowprops=dict(arrowstyle="->", lw=2, color="orange"))
    ax.annotate("", xy=(11.5, 3), xytext=(8.5, 3), arrowprops=dict(arrowstyle="->", lw=2, color="dodgerblue"))

    # Feedwater Line
    ax.annotate("", xy=(7.5, 3.2), xytext=(7.5, 5.5), arrowprops=dict(arrowstyle="->", lw=2, color="teal"))

    # Control Valve (Bowtie shape)
    valve_x = [3, 4, 3, 4, 3]
    valve_y = [2.5, 3.5, 3.5, 2.5, 2.5]
    ax.plot(valve_x, valve_y, color="black", lw=2)
    ax.plot([3.5, 3.5], [3, 4], color="black", lw=2) # Valve stem
    ax.add_patch(patches.Circle((3.5, 4.2), 0.2, color="black", fill=False, lw=2)) # Actuator

    # Venturi Desuperheater Body (Converging-diverging nozzle)
    venturi_x = [6.5, 7.2, 7.8, 8.5, 8.5, 7.8, 7.2, 6.5]
    venturi_y = [2.2, 2.7, 2.7, 2.2, 3.8, 3.3, 3.3, 3.8]
    ax.fill(venturi_x, venturi_y, color="#e0e0e0", ec="black", lw=2)

    # Text Labels
    ax.text(0.5, 3.3, "HP Steam In", fontsize=10, color="crimson", fontweight="bold")
    ax.text(3.0, 1.8, "Control Valve", fontsize=10, fontweight="bold")
    ax.text(7.5, 5.7, "Feedwater In", fontsize=10, color="teal", fontweight="bold", ha="center")
    ax.text(7.5, 1.6, "Venturi Desuperheater", fontsize=10, fontweight="bold", ha="center")
    ax.text(10.0, 3.3, "Outlet Steam", fontsize=10, color="dodgerblue", fontweight="bold")

    plt.tight_layout()
    return fig

st.title("Letdown Steam System Custom Sketch")
fig = draw_letdown_system()
st.pyplot(fig)
