import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gear Mesh & Spectrum Analysis", layout="wide")

st.title("Gear Mesh & Frequency Spectrum Analysis")

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
gmf = fr_driver * n_pinion  # Fundamental GMF

st.sidebar.header("Spectrum Simulation Controls")
# Dynamic default for Fmax to cover up to 3x GMF initially
default_fmax = float(np.ceil(3.5 * gmf / 100.0) * 100.0) if gmf > 0 else 2000.0
fmax = st.sidebar.slider(
    "Spectrum Fmax (Hz)",
    min_value=100.0,
    max_value=10000.0,
    value=max(500.0, default_fmax),
    step=50.0,
)

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
        height=320,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

# Full-width Spectrum Plot Section
st.subheader("Simulated Vibration Spectrum")

# Peak lists for plotting
freq_peaks = []
amp_peaks = []
labels = []

# Base Noise Floor
noise_floor = 0.02

# Generate peaks for higher harmonic orders dynamically (up to order 10) to support larger Fmax values
max_harmonic_order = int(np.ceil(fmax / gmf)) + 1 if gmf > 0 else 3

for h in range(1, max_harmonic_order + 1):
    center = h * gmf
    # Decreasing base amplitude for higher harmonics
    base_amp = 1.0 / (h**0.7)

    # Add GMF Center Peak
    if center <= fmax:
        freq_peaks.append(center)
        amp_peaks.append(base_amp)
        labels.append(f"GMF {h}")

    # Add Sidebands (up to 5 orders)
    for sb in range(1, 6):
        sb_amp = base_amp * (0.35 / (sb**0.8))  # Sideband attenuation factor

        lower_f = center - (sb * fr_driver)
        upper_f = center + (sb * fr_driver)

        if 0 <= lower_f <= fmax:
            freq_peaks.append(lower_f)
            amp_peaks.append(sb_amp)
            labels.append("")

        if 0 <= upper_f <= fmax:
            freq_peaks.append(upper_f)
            amp_peaks.append(sb_amp)
            labels.append("")

fig_spec = go.Figure()

# Plot spectral line stems
for f_peak, a_peak in zip(freq_peaks, amp_peaks):
    fig_spec.add_trace(
        go.Scatter(
            x=[f_peak, f_peak],
            y=[noise_floor, a_peak],
            mode="lines",
            line=dict(color="#008080", width=2),
            showlegend=False,
            hoverinfo="x+y",
        )
    )

# Add GMF Markers & Labels (GMF 1, GMF 2, GMF 3, etc.)
gmf_freqs = [f for f, lbl in zip(freq_peaks, labels) if lbl.startswith("GMF")]
gmf_amps = [a for a, lbl in zip(amp_peaks, labels) if lbl.startswith("GMF")]
gmf_text = [lbl for lbl in labels if lbl.startswith("GMF")]

fig_spec.add_trace(
    go.Scatter(
        x=gmf_freqs,
        y=gmf_amps,
        mode="markers+text",
        text=gmf_text,
        textposition="top center",
        marker=dict(color="crimson", size=8),
        name="GMF Harmonics",
    )
)

fig_spec.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="Amplitude (g / peak)",
    xaxis=dict(range=[0, fmax]),
    yaxis=dict(range=[0, 1.3]),
    template="plotly_white",
    height=400,
    margin=dict(l=40, r=20, t=30, b=40),
)

st.plotly_chart(fig_spec, use_container_width=True)
