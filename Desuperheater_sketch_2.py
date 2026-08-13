import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gear Mesh Analysis", layout="wide")

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
fr_driver_hz = (
    driver_speed_input if unit == "Hz" else driver_speed_input / 60.0
)  # Driver speed in Hz
fr_driver_rpm = fr_driver_hz * 60.0
gear_ratio = n_gear / n_pinion
gmf_hz = fr_driver_hz * n_pinion  # Fundamental GMF in Hz

st.sidebar.header("Spectrum Simulation Controls")
# Select X-axis Display Unit
spectrum_unit = st.sidebar.radio(
    "Spectrum X-Axis Unit", ["Hz", "RPM", "Orders"]
)

# Set Default and Max Fmax based on Unit
if spectrum_unit == "Hz":
    default_fmax = (
        float(np.ceil(3.5 * gmf_hz / 100.0) * 100.0) if gmf_hz > 0 else 2000.0
    )
    fmax = st.sidebar.slider(
        "Spectrum Fmax (Hz)",
        min_value=100.0,
        max_value=10000.0,
        value=max(500.0, default_fmax),
        step=50.0,
    )
    scale_factor = 1.0
    unit_label = "Hz"
elif spectrum_unit == "RPM":
    default_fmax = (
        float(np.ceil(3.5 * gmf_hz * 60 / 1000.0) * 1000.0)
        if gmf_hz > 0
        else 120000.0
    )
    fmax = st.sidebar.slider(
        "Spectrum Fmax (RPM)",
        min_value=1000.0,
        max_value=600000.0,
        value=max(10000.0, default_fmax),
        step=1000.0,
    )
    scale_factor = 60.0
    unit_label = "RPM"
else:  # Orders (normalized to Driver 1x running speed)
    default_fmax = (
        float(np.ceil(3.5 * n_pinion)) if n_pinion > 0 else 100.0
    )
    fmax = st.sidebar.slider(
        "Spectrum Fmax (Orders)",
        min_value=5.0,
        max_value=200.0,
        value=max(20.0, default_fmax),
        step=1.0,
    )
    scale_factor = 1.0 / fr_driver_hz if fr_driver_hz > 0 else 1.0
    unit_label = "Orders (x fr)"

# Layout Columns
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Frequency Calculations")

    st.markdown(
        f"**Driver Running Speed ($f_r$):** `{fr_driver_hz:.2f} Hz` /"
        f" `{fr_driver_rpm:.0f} RPM`"
    )
    st.markdown(f"**Gear Ratio ($i$):** `{gear_ratio:.3f}`")
    st.markdown(
        f"**Fundamental GMF:** `{gmf_hz:.2f} Hz` /"
        f" `{gmf_hz*60:.0f} RPM` (`{n_pinion:.0f}x` Orders)"
    )

    st.markdown("### GMF Harmonics & Sidebands (Hz)")

    # Generate Sideband Table (Sideband 1 and 2)
    harmonics = [1, 2, 3]
    sb_orders = [1, 2]

    table_data = []
    for h in harmonics:
        center = h * gmf_hz
        row = {"Harmonic": f"{h}x GMF ({center:.1f} Hz)"}
        for sb in sb_orders:
            lower = center - (sb * fr_driver_hz)
            upper = center + (sb * fr_driver_hz)
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
    theta = np.linspace(-np.pi / 6, np.pi / 6, 100)

    x_p = r_pinion * np.sin(theta)
    y_p = r_pinion * np.cos(theta)

    x_g = r_gear * np.sin(theta)
    y_g = -center_dist + r_gear * np.cos(theta)

    fig = go.Figure()

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

freq_peaks_hz = []
amp_peaks = []
labels = []

noise_floor = 0.02

# 1. Add 1x and 2x Running Speed Peaks
for order in [1, 2]:
    f_run = order * fr_driver_hz
    freq_peaks_hz.append(f_run)
    amp_peaks.append(0.6 / order)
    labels.append(f"{order}x fr")

# 2. Add GMF Harmonics and Sidebands (up to 2 orders)
fmax_hz = fmax / scale_factor
max_harmonic_order = int(np.ceil(fmax_hz / gmf_hz)) + 1 if gmf_hz > 0 else 3

tick_vals = [0]
tick_text = ["0"]

# Add 1x and 2x Running Speed Ticks
for order in [1, 2]:
    val = (order * fr_driver_hz) * scale_factor
    if val <= fmax:
        tick_vals.append(val)
        if spectrum_unit == "Orders":
            tick_text.append(f"{order}x fr")
        else:
            tick_text.append(f"{order}x fr\n({val:.1f})")

# Add GMF Harmonics Ticks & Peaks
for h in range(1, max_harmonic_order + 1):
    center_hz = h * gmf_hz
    center_scaled = center_hz * scale_factor
    base_amp = 1.0 / (h**0.7)

    if center_scaled <= fmax:
        tick_vals.append(center_scaled)
        if spectrum_unit == "Orders":
            tick_text.append(f"GMF {h}\n({h*n_pinion:.0f}x)")
        else:
            tick_text.append(f"GMF {h}\n({center_scaled:.1f})")

        freq_peaks_hz.append(center_hz)
        amp_peaks.append(base_amp)
        labels.append(f"GMF {h}")

    # Sidebands restricted to 2 orders (SB 1 & SB 2)
    for sb in [1, 2]:
        sb_amp = base_amp * (0.35 / (sb**0.8))

        lower_f = center_hz - (sb * fr_driver_hz)
        upper_f = center_hz + (sb * fr_driver_hz)

        if lower_f >= 0 and (lower_f * scale_factor) <= fmax:
            freq_peaks_hz.append(lower_f)
            amp_peaks.append(sb_amp)
            labels.append("")

        if upper_f >= 0 and (upper_f * scale_factor) <= fmax:
            freq_peaks_hz.append(upper_f)
            amp_peaks.append(sb_amp)
            labels.append("")

# Scale x-axis coordinates according to selected unit
x_peaks = [f * scale_factor for f in freq_peaks_hz]

fig_spec = go.Figure()

# Plot spectral line stems
for x_p, a_p in zip(x_peaks, amp_peaks):
    if x_p <= fmax:
        fig_spec.add_trace(
            go.Scatter(
                x=[x_p, x_p],
                y=[noise_floor, a_p],
                mode="lines",
                line=dict(color="#008080", width=2),
                showlegend=False,
                hoverinfo="x+y",
            )
        )

# Marker grouping for GMF vs Running Speed (1x, 2x)
gmf_x = [x for x, lbl in zip(x_peaks, labels) if lbl.startswith("GMF")]
gmf_amps = [a for a, lbl in zip(amp_peaks, labels) if lbl.startswith("GMF")]
gmf_text = [lbl for lbl in labels if lbl.startswith("GMF")]

fr_x = [x for x, lbl in zip(x_peaks, labels) if lbl.endswith("fr")]
fr_amps = [a for a, lbl in zip(amp_peaks, labels) if lbl.endswith("fr")]
fr_text = [lbl for lbl in labels if lbl.endswith("fr")]

# Highlight Running Speeds (1x, 2x)
fig_spec.add_trace(
    go.Scatter(
        x=fr_x,
        y=fr_amps,
        mode="markers+text",
        text=fr_text,
        textposition="top center",
        marker=dict(color="orange", size=8),
        name="1x / 2x Running Speed",
    )
)

# Highlight GMF Peaks
fig_spec.add_trace(
    go.Scatter(
        x=gmf_x,
        y=gmf_amps,
        mode="markers+text",
        text=gmf_text,
        textposition="top center",
        marker=dict(color="crimson", size=8),
        name="GMF Harmonics",
    )
)

fig_spec.update_layout(
    xaxis_title=f"Frequency ({unit_label})",
    yaxis_title="Amplitude (g / peak)",
    xaxis=dict(
        range=[0, fmax],
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
    ),
    yaxis=dict(range=[0, 1.3]),
    template="plotly_white",
    height=440,
    margin=dict(l=40, r=20, t=30, b=40),
)

st.plotly_chart(fig_spec, use_container_width=True)
