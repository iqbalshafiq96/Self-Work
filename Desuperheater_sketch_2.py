import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gear Mesh Analysis", layout="wide")

st.title("Gear Mesh & Frequency Analysis")

# Sidebar - Inputs
st.sidebar.header("Operating Parameters")
unit = st.sidebar.radio("Driver Speed Unit", ["RPM", "Hz"])
driver_speed_input = st.sidebar.number_input(
    f"Driver Speed ({unit})", min_value=1.0, value=1500.0, step=10.0
)

st.sidebar.header("Gear Geometry")
n_pinion = st.sidebar.number_input(
    "Pinion Teeth Count (N_pinion)", min_value=1, value=19, step=1
)
n_gear = st.sidebar.number_input(
    "Gear Teeth Count (N_gear)", min_value=1, value=57, step=1
)

# Core Computations
fr_driver = (
    driver_speed_input if unit == "Hz" else driver_speed_input / 60.0
)  # Driver speed in Hz
gear_ratio = n_gear / n_pinion
gmf = fr_driver * n_pinion  # GMF based on driver teeth

# Layout Columns
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Frequency Calculations")

    st.markdown(f"**Driver Running Speed ($f_r$):** `{fr_driver:.2f} Hz`")
    st.markdown(f"**Gear Ratio ($i$):** `{gear_ratio:.3f}`")
    st.markdown(f"**Fundamental GMF:** `{gmf:.2f} Hz`")

    st.markdown("### GMF Harmonics & Sidebands (Hz)")

    # Generate Sideband Table
    harmonics = [1, 2, 3]
    sb_orders = range(1, 6)

    table_data = []
    for h in harmonics:
        center = h * gmf
        row = {"Harmonic": f"{h}x GMF ({center:.1f} Hz)"}
        for sb in sb_orders:
            lower = center - (sb * fr_driver)
            upper = center + (sb * fr_driver)
            row[f"SB -{sb}"] = f"{lower:.1f}"
            row[f"SB +{sb}"] = f"{upper:.1f}"
        table_data.append(row)

    st.dataframe(table_data, use_container_width=True)

with col2:
    st.subheader("Mesh Region Schematic")

    # Geometry Setup for Visualization (Pitch Circles)
    module = 1.0  # Normalized module
    r_pinion = (n_pinion * module) / 2.0
    r_gear = (n_gear * module) / 2.0
    center_dist = r_pinion + r_gear

    # Focus on top mesh point region (arc segment)
    theta = np.linspace(-np.pi / 6, np.pi / 6, 100)  # ~60 degree arc

    # Pinion Arc (centered at origin 0,0)
    x_p = r_pinion * np.sin(theta)
    y_p = r_pinion * np.cos(theta)

    # Gear Arc (centered below pinion at 0, -center_dist)
    x_g = r_gear * np.sin(theta)
    y_g = -center_dist + r_gear * np.cos(theta)

    fig = go.Figure()

    # Plot Pitch Circles
    fig.add_trace(
        go.Scatter(
            x=x_p,
            y=y_p,
            mode="lines",
            name="Pinion Pitch Arc",
            line=dict(color="teal", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_g,
            y=y_g,
            mode="lines",
            name="Gear Pitch Arc",
            line=dict(color="royalblue", width=3),
        )
    )

    # Tangent / Pitch Point
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[r_pinion],
            mode="markers",
            name="Pitch Point",
            marker=dict(size=10, color="red"),
        )
    )

    fig.update_layout(
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)
