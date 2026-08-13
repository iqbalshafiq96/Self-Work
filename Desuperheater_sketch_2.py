import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gear Mesh Analysis", layout="wide")

st.title("Gear Mesh & Frequency Spectrum Analysis")

# Sidebar - Gear Geometry First
st.sidebar.header("Gear Geometry")
n_gear = st.sidebar.number_input(
    "Gear Teeth Count (N_gear)", min_value=1, value=57, step=1
)
n_pinion = st.sidebar.number_input(
    "Pinion Teeth Count (N_pinion)", min_value=1, value=19, step=1
)

# Sidebar - Operating Parameters
st.sidebar.header("Operating Parameters")
unit = st.sidebar.radio("Driver Speed Unit", ["RPM", "Hz"])
driver_speed_input = st.sidebar.number_input(
    f"Driver Speed ({unit})", min_value=1.0, value=1500.0, step=10.0
)

# Gear Orientation Selection
orientation_options = [
    "Speed Increaser (Driver = Gear)",
    "Speed Reducer (Driver = Pinion)",
]
orientation = st.sidebar.selectbox(
    "Gearbox Orientation",
    orientation_options,
    index=0,
    help="Determines driven speed based on driver attachment."
)

# Core Computations
fr_driver_hz = (
    driver_speed_input if unit == "Hz" else driver_speed_input / 60.0
)
fr_driver_rpm = fr_driver_hz * 60.0
gear_ratio = n_gear / n_pinion

if "Reducer" in orientation:
    fr_driven_hz = fr_driver_hz / gear_ratio
    gmf_hz = fr_driver_hz * n_pinion
else:
    fr_driven_hz = fr_driver_hz * gear_ratio
    gmf_hz = fr_driver_hz * n_gear

fr_driven_rpm = fr_driven_hz * 60.0

st.sidebar.metric(
    label="Calculated Driven Speed",
    value=f"{fr_driven_rpm:.1f} RPM",
    delta=f"{fr_driven_hz:.2f} Hz",
    delta_color="off"
)

st.sidebar.header("Spectrum Simulation Controls")
spectrum_unit = st.sidebar.radio(
    "Spectrum X-Axis Unit", ["Hz", "RPM", "Orders"]
)

if spectrum_unit == "Hz":
    scale_factor = 1.0
    unit_label = "Hz"
elif spectrum_unit == "RPM":
    scale_factor = 60.0
    unit_label = "RPM"
else:
    scale_factor = 1.0 / fr_driver_hz if fr_driver_hz > 0 else 1.0
    unit_label = "Orders"

optimize_option = st.sidebar.selectbox(
    "Optimize Fmax Target",
    ["Manual Slider", "1x GMF (+ Sidebands)", "2x GMF (+ Sidebands)", "3x GMF (+ Sidebands)"],
    help="Selecting a GMF target automatically calculates the ideal Fmax range with headroom for sidebands."
)

gmf_headroom_multiplier = {
    "1x GMF (+ Sidebands)": 1.15,
    "2x GMF (+ Sidebands)": 2.15,
    "3x GMF (+ Sidebands)": 3.15,
}

if optimize_option != "Manual Slider":
    calc_fmax_hz = gmf_hz * gmf_headroom_multiplier[optimize_option]
    fmax = float(calc_fmax_hz * scale_factor)
    st.sidebar.info(
        f"**Fmax Auto-Optimized:** `{fmax:.1f} {unit_label}`\n\n"
        f"*(Targeting `{optimize_option[:2]}` at `{gmf_hz * int(optimize_option[0]):.1f} Hz`)*"
    )
else:
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
    else:
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

# Resolution & Recording Duration Calculations
fmax_hz = fmax / scale_factor
req_delta_f = fr_driver_hz / 2.5 if fr_driver_hz > 0 else 1.0
min_fft_lines = int(np.ceil(fmax_hz / req_delta_f)) if req_delta_f > 0 else 800

standard_lines = [400, 800, 1600, 3200, 6400, 12800]
recommended_lines = next(
    (l for l in standard_lines if l >= min_fft_lines), 12800
)
actual_delta_f = fmax_hz / recommended_lines if recommended_lines > 0 else 0.0

# Time per recording block calculation
recording_time_sec = recommended_lines / fmax_hz if fmax_hz > 0 else 0.0

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Frequency Calculations")

    st.markdown(
        f"**Driver Running Speed ($f_{{driver}}$):** `{fr_driver_hz:.2f} Hz` / `{fr_driver_rpm:.0f} RPM`"
    )
    st.markdown(
        f"**Driven Running Speed ($f_{{driven}}$):** `{fr_driven_hz:.2f} Hz` / `{fr_driven_rpm:.0f} RPM`"
    )
    st.markdown(f"**Gear Ratio ($i$):** `{gear_ratio:.3f}`")
    st.markdown(
        f"**Fundamental GMF:** `{gmf_hz:.2f} Hz` / `{gmf_hz*60:.0f} RPM`"
    )

    st.markdown("### GMF Harmonics & Sidebands (Hz)")

    harmonics = [1, 2, 3]
    sb_orders = [1, 2]

    table_data = []
    for h in harmonics:
        center = h * gmf_hz
        row = {"Harmonic": f"{h}x GMF ({center:.1f} Hz)"}
        for sb in sb_orders:
            lower = center - (sb * fr_driver_hz)
            upper = center + (sb * fr_driver_hz)
            row[f"SB -{sb}"] = f"{lower:.1f}" if lower > 0 else "N/A"
            row[f"SB +{sb}"] = f"{upper:.1f}"
        table_data.append(row)

    st.dataframe(table_data, use_container_width=True)

    # Resolution Guidance & Recording Duration Box
    st.info(
        f"**Required Spectral Resolution Guidance:**\n"
        f"* **Max Allowed Resolution Step ($\Delta f$):** `{req_delta_f:.2f} Hz`\n"
        f"* **Recommended Lines ($N_{{lines}}$):** `{recommended_lines:,} Lines` at $F_{{max}} = {fmax_hz:.0f}\\text{{ Hz}}$\n"
        f"* **Achieved Resolution:** `{actual_delta_f:.3f} Hz/line` (Sidebands clearly resolvable)\n"
        f"* **Recording Duration per Time Block ($T_{{record}}$):** `{recording_time_sec:.2f} s`"
    )

with col2:
    st.subheader("Mesh Region Schematic")

    module = 1.0
    r_pinion = (n_pinion * module) / 2.0
    r_gear = (n_gear * module) / 2.0
    center_dist = r_pinion + r_gear

    theta = np.linspace(-np.pi / 6, np.pi / 6, 100)

    x_p = r_pinion * np.sin(theta)
    y_p = r_pinion * np.cos(theta)

    x_g = r_gear * np.sin(theta)
    y_g = -center_dist + r_gear * np.cos(theta)

    fig = go.Figure()

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
            x=x_p,
            y=y_p,
            mode="lines",
            name="Pinion Pitch Arc",
            line=dict(color="teal", width=3),
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

noise_floor = 0.02

x_1x, amp_1x, lbl_1x = [], [], []
x_2x, amp_2x, lbl_2x = [], [], []
x_gmf, amp_gmf, lbl_gmf = [], [], []
x_sb, amp_sb, lbl_sb = [], [], []

for order in [1, 2]:
    val_hz = order * fr_driver_hz
    val_scaled = val_hz * scale_factor
    amp = 0.6 / order

    if 0 < val_scaled <= fmax:
        if order == 1:
            x_1x.append(val_scaled)
            amp_1x.append(amp)
            lbl_1x.append("1x")
        else:
            x_2x.append(val_scaled)
            amp_2x.append(amp)
            lbl_2x.append("2x")

max_harmonic_order = int(np.ceil(fmax_hz / gmf_hz)) + 1 if gmf_hz > 0 else 3

for h in range(1, max_harmonic_order + 1):
    center_hz = h * gmf_hz
    center_scaled = center_hz * scale_factor
    base_amp = 1.0 / (h**0.7)

    if 0 < center_scaled <= fmax:
        x_gmf.append(center_scaled)
        amp_gmf.append(base_amp)
        lbl_gmf.append(f"GMF {h}")

    for sb in [1, 2]:
        sb_amp = base_amp * (0.35 / (sb**0.8))

        lower_scaled = (center_hz - (sb * fr_driver_hz)) * scale_factor
        upper_scaled = (center_hz + (sb * fr_driver_hz)) * scale_factor

        if 0 < lower_scaled <= fmax:
            x_sb.append(lower_scaled)
            amp_sb.append(sb_amp)
            lbl_sb.append(f"GMF{h} -{sb}x")

        if 0 < upper_scaled <= fmax:
            x_sb.append(upper_scaled)
            amp_sb.append(sb_amp)
            lbl_sb.append(f"GMF{h} +{sb}x")

fig_spec = go.Figure()

all_x = x_1x + x_2x + x_gmf + x_sb
all_amps = amp_1x + amp_2x + amp_gmf + amp_sb
all_labels = lbl_1x + lbl_2x + lbl_gmf + lbl_sb

for x_p, a_p in zip(all_x, all_amps):
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

if x_1x:
    fig_spec.add_trace(
        go.Scatter(
            x=x_1x,
            y=amp_1x,
            mode="markers+text",
            text=lbl_1x,
            textposition="top center",
            textfont=dict(color="darkorange", size=11, family="Segoe UI"),
            marker=dict(color="darkorange", size=9),
            cliponaxis=False,
            name="1x Running Speed",
        )
    )

if x_2x:
    fig_spec.add_trace(
        go.Scatter(
            x=x_2x,
            y=amp_2x,
            mode="markers+text",
            text=lbl_2x,
            textposition="top center",
            textfont=dict(color="purple", size=11, family="Segoe UI"),
            marker=dict(color="purple", size=9),
            cliponaxis=False,
            name="2x Running Speed",
        )
    )

if x_gmf:
    fig_spec.add_trace(
        go.Scatter(
            x=x_gmf,
            y=amp_gmf,
            mode="markers+text",
            text=lbl_gmf,
            textposition="top center",
            textfont=dict(color="crimson", size=11, family="Segoe UI"),
            marker=dict(color="crimson", size=9),
            cliponaxis=False,
            name="GMF Harmonics",
        )
    )

if all_amps:
    max_idx = int(np.argmax(all_amps))
    top_x = all_x[max_idx]
    top_amp = all_amps[max_idx]
    top_label = all_labels[max_idx]

    fig_spec.add_trace(
        go.Scatter(
            x=[top_x],
            y=[top_amp],
            mode="markers",
            marker=dict(
                size=14,
                color="red",
                symbol="diamond",
                line=dict(color="black", width=1.5)
            ),
            name="Peak Amplitude",
            showlegend=True
        )
    )

    fig_spec.add_annotation(
        x=top_x,
        y=top_amp,
        text=f"<b>Max Peak: {top_label}</b><br>{top_x:.1f} {unit_label} | {top_amp:.2f} g",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        ax=0,
        ay=-45,
        bordercolor="red",
        borderwidth=1.5,
        borderpad=4,
        bgcolor="rgba(255, 255, 255, 0.9)",
        font=dict(size=11, family="Segoe UI", color="black")
    )

fig_spec.update_layout(
    xaxis_title=f"Frequency ({unit_label})",
    yaxis_title="Amplitude (g / peak)",
    xaxis=dict(
        range=[0, fmax],
        nticks=5,
        zeroline=False,
    ),
    yaxis=dict(range=[0, max(all_amps, default=1.0) * 1.35]),
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=20, t=40, b=40),
)

st.plotly_chart(fig_spec, use_container_width=True)
