import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Involute Gear Mesh", layout="wide")
st.title("Involute Gear Mesh Analysis")

# Sidebar Inputs
st.sidebar.header("Gear Geometry")
module = st.sidebar.number_input("Module (m)", min_value=0.5, value=2.0, step=0.5)
pressure_angle_deg = st.sidebar.number_input(
    "Pressure Angle (°)", min_value=14.5, max_value=25.0, value=20.0, step=0.5
)
n_pinion = st.sidebar.number_input(
    "Pinion Teeth Count (N1)", min_value=8, value=19, step=1
)
n_gear = st.sidebar.number_input(
    "Gear Teeth Count (N2)", min_value=8, value=57, step=1
)

alpha = np.radians(pressure_angle_deg)


def generate_tooth_profile(N, m, alpha):
    """Generates a single involute tooth profile coordinates centered at origin."""
    r_pitch = (N * m) / 2.0
    r_base = r_pitch * np.cos(alpha)
    r_addendum = r_pitch + m
    r_dedendum = r_pitch - 1.25 * m

    # Maximum roll angle theta at addendum circle
    theta_max = np.sqrt(max(0, (r_addendum / r_base) ** 2 - 1))
    theta_vals = np.linspace(0, theta_max, 40)

    # Right side involute curve
    x_inv = r_base * (np.sin(theta_vals) - theta_vals * np.cos(theta_vals))
    y_inv = r_base * (np.cos(theta_vals) + theta_vals * np.sin(theta_vals))

    # Thickness angle at pitch circle to center tooth symmetrically
    inv_alpha = np.tan(alpha) - alpha
    tooth_thickness_angle = (np.pi / (2 * N)) + inv_alpha

    # Rotate right side into position
    x_right = x_inv * np.cos(
        tooth_thickness_angle / 2
    ) + y_inv * np.sin(tooth_thickness_angle / 2)
    y_right = -x_inv * np.sin(
        tooth_thickness_angle / 2
    ) + y_inv * np.cos(tooth_thickness_angle / 2)

    # Mirror for left side curve
    x_left = -x_right[::-1]
    y_left = y_right[::-1]

    # Combine full tooth outline
    x_tooth = np.concatenate([x_left, x_right])
    y_tooth = np.concatenate([y_left, y_right])

    return x_tooth, y_tooth, r_pitch


# Generate Pinion & Gear Tooth Outlines
x_p_single, y_p_single, r_p = generate_tooth_profile(n_pinion, module, alpha)
x_g_single, y_g_single, r_g = generate_tooth_profile(n_gear, module, alpha)

center_dist = r_p + r_g

# Plotting
fig = go.Figure()

# Plot Top 3 Teeth for Pinion (Mesh zone near pitch point)
num_teeth_to_show = 3
tooth_pitch_angle = 2 * np.pi / n_pinion

for i in range(
    -num_teeth_to_show // 2 + 1, num_teeth_to_show // 2 + 1
):
    rot = i * tooth_pitch_angle
    x_rot = x_p_single * np.cos(rot) - y_p_single * np.sin(rot)
    y_rot = x_p_single * np.sin(rot) + y_p_single * np.cos(rot)
    fig.add_trace(
        go.Scatter(
            x=x_rot,
            y=y_rot,
            mode="lines",
            fill="toself",
            name=f"Pinion Tooth {i}",
            line=dict(color="teal"),
        )
    )

# Plot Top 3 Teeth for Gear (Positioned vertically below Pinion)
tooth_pitch_angle_g = 2 * np.pi / n_gear
for i in range(
    -num_teeth_to_show // 2 + 1, num_teeth_to_show // 2 + 1
):
    # Offset by half-pitch to mesh properly
    rot = (i + 0.5) * tooth_pitch_angle_g
    x_rot = x_g_single * np.cos(rot) - y_g_single * np.sin(rot)
    y_rot = -(x_g_single * np.sin(rot) + y_g_single * np.cos(rot)) + center_dist

    fig.add_trace(
        go.Scatter(
            x=x_rot,
            y=y_rot,
            mode="lines",
            fill="toself",
            name=f"Gear Tooth {i}",
            line=dict(color="royalblue"),
        )
    )

# Pitch Point Reference Indicator
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[r_p],
        mode="markers",
        name="Pitch Point",
        marker=dict(size=12, color="crimson", symbol="cross"),
    )
)

fig.update_layout(
    title="Involute Tooth Mesh Profile (Local Mesh Zone)",
    xaxis_title="X (mm)",
    yaxis_title="Y (mm)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    template="plotly_white",
    height=550,
)

st.plotly_chart(fig, use_container_width=True)
