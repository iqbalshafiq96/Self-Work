import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Gear Mesh Analysis", layout="wide")

st.title("Gear Mesh & Frequency Spectrum Analysis")

# Sidebar - Gear Geometry
st.sidebar.header("Gear Geometry")
n_gear = st.sidebar.number_input(
    "Gear Teeth Count (N_gear)", min_value=1, value=57, step=1
)
n_pinion = st.sidebar.number_input(
    "Pinion Teeth Count (N_pinion)", min_value=1, value=19, step=1
)

# Gear Natural Frequency Input with Unit Dropdown inline
col_fn1, col_fn2 = st.sidebar.columns([2, 1])

with col_fn1:
    fn_input = st.number_input(
        "Gear Natural Freq (f_n)",
        min_value=0.0,
        value=0.0,
        step=50.0,
        help="Set to 0 if not specified.",
    )

with col_fn2:
    fn_unit = st.selectbox("Unit", ["Hz", "RPM"], index=0, key="fn_unit_select")

# Convert fn input to Hz internally (0 = not specified)
fn_hz = fn_input if fn_unit == "Hz" else fn_input / 60.0

# Sidebar - Operating Parameters
st.sidebar.header("Operating Parameters")

col_spd1, col_spd2 = st.sidebar.columns([2, 1])

with col_spd1:
    driver_speed_input = st.number_input(
        "Driver Speed", min_value=1.0, value=1500.0, step=10.0
    )

with col_spd2:
    unit = st.selectbox("Unit", ["RPM", "Hz"], index=0, key="driver_speed_unit_select")

orientation_options = [
    "Speed Increaser (Driver = Gear)",
    "Speed Reducer (Driver = Pinion)",
]
orientation = st.sidebar.selectbox(
    "Gearbox Orientation",
    orientation_options,
    index=0,
    help="Determines driven speed based on driver attachment.",
)

# Core Speed Computations
fr_driver_hz = (
    driver_speed_input if unit == "Hz" else driver_speed_input / 60.0
)
gear_ratio = n_gear / n_pinion

if "Reducer" in orientation:
    f_pinion_hz = fr_driver_hz
    f_gear_hz = fr_driver_hz / gear_ratio
    gmf_hz = f_pinion_hz * n_pinion
else:
    f_gear_hz = fr_driver_hz
    f_pinion_hz = fr_driver_hz * gear_ratio
    gmf_hz = f_gear_hz * n_gear

fr_driven_hz = f_gear_hz if "Reducer" in orientation else f_pinion_hz

st.sidebar.metric(
    label="Calculated Driven Speed",
    value=f"{fr_driven_hz * 60.0:.1f} RPM",
    delta=f"{fr_driven_hz:.2f} Hz",
    delta_color="off",
)

st.sidebar.header("Spectrum Simulation Controls")

# Added Component Amplitude Controls
col_amp1, col_amp2 = st.sidebar.columns(2)
with col_amp1:
    amp_1x_gear = st.number_input(
        "1x Gear Amp (g)",
        min_value=0.00,
        max_value=5.00,
        value=0.50,
        step=0.05,
        help="Set to 0 to disable 1x Gear component.",
    )
with col_amp2:
    amp_1x_pinion = st.number_input(
        "1x Pinion Amp (g)",
        min_value=0.00,
        max_value=5.00,
        value=0.50,
        step=0.05,
        help="Set to 0 to disable 1x Pinion component.",
    )

spectrum_unit = st.sidebar.radio(
    "Spectrum X-Axis Unit", ["Hz", "RPM", "Orders (Driver)"]
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
    [
        "Manual Slider",
        "1x GMF (+ Sidebands)",
        "2x GMF (+ Sidebands)",
        "3x GMF (+ Sidebands)",
    ],
    help="Selecting a GMF target automatically calculates the ideal Fmax range with headroom for sidebands.",
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
            float(np.ceil(3.5 * gmf_hz / 100.0) * 100.0)
            if gmf_hz > 0
            else 2000.0
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

# Time Waveform Simulation Controls in Sidebar
st.sidebar.header("Time Waveform Controls")
twf_revolutions = st.sidebar.slider(
    "Plot Duration (Driver Shaft Revolutions)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5,
)
twf_noise_level = st.sidebar.slider(
    "Noise Level (g RMS)",
    min_value=0.00,
    max_value=0.10,
    value=0.00,
    step=0.005,
)

# Main Panel - Frequency Calculations Section
st.subheader("Frequency Calculations")

st.markdown(
    f"**Gear Running Speed ($f_{{gear}}$):** `{f_gear_hz:.2f} Hz` / `{f_gear_hz*60:.0f} RPM`"
)
st.markdown(
    f"**Pinion Running Speed ($f_{{pinion}}$):** `{f_pinion_hz:.2f} Hz` / `{f_pinion_hz*60:.0f} RPM`"
)
st.markdown(f"**Gear Ratio ($i$):** `{gear_ratio:.3f}`")
st.markdown(f"**Fundamental GMF:** `{gmf_hz:.2f} Hz` / `{gmf_hz*60:.0f} RPM`")

if fn_hz > 0:
    fn_display = fn_hz * scale_factor
    st.markdown(
        f"**Gear Natural Frequency ($f_n$):** `{fn_display:.1f} {unit_label}` ({fn_hz:.2f} Hz)"
    )

st.markdown("### GMF Harmonics & Sidebands")

col_sb1, col_sb2 = st.columns([1, 1])
with col_sb1:
    sideband_source = st.radio(
        "Sideband Reference Source:",
        ["Gear Side", "Pinion Side", "Both Sides"],
        horizontal=True,
        help="Select which shaft running speed to use for sideband modulation calculations.",
    )
with col_sb2:
    selected_orders = st.multiselect(
        "Select Sideband Multipliers (k):",
        options=[1, 2, 3, 4, 5],
        default=[1, 2],
        help="Order multiplier (k) for calculating ±k*f_shaft sidebands.",
    )

mod_freqs = []
if sideband_source in ["Gear Side", "Both Sides"]:
    mod_freqs.append(("Gear", f_gear_hz))
if sideband_source in ["Pinion Side", "Both Sides"]:
    mod_freqs.append(("Pinion", f_pinion_hz))

min_mod_freq = (
    min([freq for _, freq in mod_freqs]) if mod_freqs else fr_driver_hz
)
req_delta_f = min_mod_freq / 2.5 if min_mod_freq > 0 else 1.0

fmax_hz = fmax / scale_factor
min_fft_lines = int(np.ceil(fmax_hz / req_delta_f)) if req_delta_f > 0 else 800

standard_lines = [400, 800, 1600, 3200, 6400, 12800]
recommended_lines = next(
    (l for l in standard_lines if l >= min_fft_lines), 12800
)
actual_delta_f = fmax_hz / recommended_lines if recommended_lines > 0 else 0.0
recording_time_sec = recommended_lines / fmax_hz if fmax_hz > 0 else 0.0

harmonics = [1, 2, 3]
sb_orders = sorted(selected_orders)

table_data = []
resonance_warning = False

for h in harmonics:
    center = h * gmf_hz
    row = {"Harmonic": f"{h}x GMF ({center:.1f} Hz)"}

    if fn_hz > 0 and abs(center - fn_hz) / fn_hz <= 0.05:
        resonance_warning = True

    if not sb_orders or not mod_freqs:
        row["Status"] = "No sidebands or source selected"
    else:
        for source_name, freq in mod_freqs:
            prefix = "G" if source_name == "Gear" else "P"
            for k in sb_orders:
                lower = center - (k * freq)
                upper = center + (k * freq)

                if (
                    fn_hz > 0
                    and (
                        abs(lower - fn_hz) / fn_hz <= 0.05
                        or abs(upper - fn_hz) / fn_hz <= 0.05
                    )
                ):
                    resonance_warning = True

                row[f"{prefix} SB -{k}"] = (
                    f"{lower:.1f}" if lower > 0 else "N/A"
                )
                row[f"{prefix} SB +{k}"] = f"{upper:.1f}"
    table_data.append(row)

st.dataframe(table_data, use_container_width=True)

if resonance_warning:
    st.warning(
        f"**Potential Resonance Alert:** Gear Natural Frequency ($f_n = {fn_hz:.1f}\\text{{ Hz}}$) "
        f"is within 5% of a GMF harmonic or primary sideband! Expect amplified excitation."
    )

# Resolution Guidance Box
st.info(
    f"**Required Spectral Resolution Guidance:**\n"
    f"* **Target Modulating Frequency ($f_{{mod}}$):** `{min_mod_freq:.2f} Hz` (Based on {sideband_source})\n"
    f"* **Max Allowed Resolution Step ($\Delta f = f_{{mod}} / 2.5$):** `{req_delta_f:.2f} Hz`\n"
    f"* **Recommended Lines ($N_{{lines}}$):** `{recommended_lines:,} Lines` at $F_{{max}} = {fmax_hz:.0f}\\text{{ Hz}}$\n"
    f"* **Achieved Resolution:** `{actual_delta_f:.3f} Hz/line` (Sidebands clearly resolvable)\n"
    f"* **Recording Duration ($T_{{record}}$):** `{recording_time_sec:.2f} seconds` per block"
)

# Spectrum Plot Section
st.subheader("Simulated Vibration Spectrum")

noise_floor = 0.02
x_1x, amp_1x, lbl_1x = [], [], []
x_gmf, amp_gmf, lbl_gmf = [], [], []
x_sb, amp_sb = [], []
tick_vals = []

# Dynamic 1x Speeds handling with user-defined amplitudes
gear_scaled = f_gear_hz * scale_factor
if 0 < gear_scaled <= fmax and amp_1x_gear > 0:
    tick_vals.append(gear_scaled)
    x_1x.append(gear_scaled)
    amp_1x.append(amp_1x_gear)
    lbl_1x.append("1x Gear")

pinion_scaled = f_pinion_hz * scale_factor
if 0 < pinion_scaled <= fmax and amp_1x_pinion > 0:
    tick_vals.append(pinion_scaled)
    x_1x.append(pinion_scaled)
    amp_1x.append(amp_1x_pinion)
    lbl_1x.append("1x Pinion")

max_harmonic_order = int(np.ceil(fmax_hz / gmf_hz)) + 1 if gmf_hz > 0 else 3

for h in range(1, max_harmonic_order + 1):
    center_hz = h * gmf_hz
    center_scaled = center_hz * scale_factor
    base_amp = 1.0 / (h**0.7)

    if 0 < center_scaled <= fmax:
        tick_vals.append(center_scaled)
        x_gmf.append(center_scaled)
        amp_gmf.append(base_amp)
        lbl_gmf.append(f"GMF {h}")

    for source_name, freq in mod_freqs:
        for k in sb_orders:
            sb_amp = base_amp * (0.3 / (k**0.8))
            lower_scaled = (center_hz - (k * freq)) * scale_factor
            upper_scaled = (center_hz + (k * freq)) * scale_factor

            if 0 < lower_scaled <= fmax:
                x_sb.append(lower_scaled)
                amp_sb.append(sb_amp)

            if 0 < upper_scaled <= fmax:
                x_sb.append(upper_scaled)
                amp_sb.append(sb_amp)

fn_amp = amp_1x[0] if amp_1x else 0.5

fig_spec = go.Figure()

all_x = x_1x + x_gmf + x_sb
all_amps = amp_1x + amp_gmf + amp_sb

if fn_hz > 0:
    fn_scaled = fn_hz * scale_factor
    if 0 < fn_scaled <= fmax:
        all_x.append(fn_scaled)
        all_amps.append(fn_amp)

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
            mode="markers",
            marker=dict(color="darkorange", size=9),
            cliponaxis=False,
            name="1x Running Speeds",
        )
    )
    for x_p, a_p, txt in zip(x_1x, amp_1x, lbl_1x):
        fig_spec.add_annotation(
            x=x_p,
            y=a_p + 0.05,
            text=txt,
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="darkorange", size=11, family="Segoe UI"),
        )

if x_gmf:
    fig_spec.add_trace(
        go.Scatter(
            x=x_gmf,
            y=amp_gmf,
            mode="markers",
            marker=dict(color="crimson", size=9),
            cliponaxis=False,
            name="GMF Harmonics",
        )
    )
    for x_p, a_p, txt in zip(x_gmf, amp_gmf, lbl_gmf):
        fig_spec.add_annotation(
            x=x_p,
            y=a_p + 0.05,
            text=txt,
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="crimson", size=11, family="Segoe UI"),
        )

if fn_hz > 0:
    fn_scaled = fn_hz * scale_factor
    if 0 < fn_scaled <= fmax:
        fn_low = fn_scaled * 0.95
        fn_high = fn_scaled * 1.05

        fig_spec.add_vrect(
            x0=fn_low,
            x1=fn_high,
            fillcolor="purple",
            opacity=0.15,
            layer="below",
            line_width=1,
            line_dash="dot",
            line_color="purple",
            annotation_text="±5% fn Band",
            annotation_position="top left",
            annotation_font=dict(size=10, color="purple"),
        )

        fig_spec.add_trace(
            go.Scatter(
                x=[fn_scaled],
                y=[fn_amp],
                mode="markers",
                marker=dict(
                    color="gold",
                    size=12,
                    symbol="diamond",
                    line=dict(color="purple", width=2),
                ),
                name="Natural Frequency (f_n)",
            )
        )
        fig_spec.add_annotation(
            x=fn_scaled,
            y=fn_amp + 0.05,
            text=f"f_n ({fn_scaled:.1f})",
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="purple", size=11, family="Segoe UI"),
        )
        tick_vals.append(fn_scaled)

clean_ticks = sorted(list(set([t for t in tick_vals if t > 0])))
max_y = max(all_amps) + 0.5 if all_amps else 1.65

fig_spec.update_layout(
    xaxis_title=f"Frequency ({unit_label})",
    yaxis_title="Amplitude (g / peak)",
    xaxis=dict(
        range=[min(clean_ticks) * 0.8 if clean_ticks else 1.0, fmax],
        tickmode="array",
        tickvals=clean_ticks,
        tickformat=".1f",
        zeroline=False,
    ),
    yaxis=dict(range=[0, max(1.65, max_y)]),
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=20, t=60, b=40),
)

st.plotly_chart(fig_spec, use_container_width=True)

# Time Waveform Simulation Section
st.subheader("Simulated Time Waveform")

t_max = twf_revolutions / fr_driver_hz if fr_driver_hz > 0 else 0.1
fs = max(10000.0, 10.0 * fmax_hz)
t = np.linspace(0, t_max, int(fs * t_max), endpoint=False)

signal = np.zeros_like(t)

# 1. Add Running Speeds ONLY if amplitude > 0 and within active Fmax display range
if amp_1x_gear > 0 and (0 < f_gear_hz * scale_factor <= fmax):
    signal += amp_1x_gear * np.sin(2 * np.pi * f_gear_hz * t)

if amp_1x_pinion > 0 and (0 < f_pinion_hz * scale_factor <= fmax):
    signal += amp_1x_pinion * np.sin(2 * np.pi * f_pinion_hz * t + np.pi / 4)

# 2. Add GMF Harmonics & Modulation Envelopes ONLY if within active Fmax display range
for h in range(1, max_harmonic_order + 1):
    center_hz = h * gmf_hz
    if not (0 < center_hz * scale_factor <= fmax):
        continue

    base_amp = 1.0 / (h**0.7)

    mod_envelope = np.ones_like(t)
    for source_name, freq in mod_freqs:
        for k in sb_orders:
            lower_hz = center_hz - (k * freq)
            upper_hz = center_hz + (k * freq)

            sb_in_bounds = (0 < lower_hz * scale_factor <= fmax) or (
                0 < upper_hz * scale_factor <= fmax
            )
            if sb_in_bounds:
                depth = (0.3 / (k**0.8)) * 2.0
                mod_envelope += depth * np.cos(2 * np.pi * k * freq * t)

    signal += base_amp * mod_envelope * np.sin(2 * np.pi * center_hz * t)

# 3. Add Natural Frequency component ONLY if active in display range
if fn_hz > 0 and (0 < fn_hz * scale_factor <= fmax):
    signal += fn_amp * np.sin(2 * np.pi * fn_hz * t)

# 4. Add random noise floor ONLY if explicitly enabled (> 0.00)
if twf_noise_level > 0.0:
    np.random.seed(42)
    signal += np.random.normal(0, twf_noise_level, size=len(t))

fig_twf = go.Figure()

fig_twf.add_trace(
    go.Scatter(
        x=t * 1000.0,
        y=signal,
        mode="lines",
        line=dict(color="#008080", width=1),
        name="Vibration Waveform",
    )
)

fig_twf.update_layout(
    xaxis_title="Time (ms)",
    yaxis_title="Acceleration (g)",
    template="plotly_white",
    height=400,
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(zeroline=False),
)

st.plotly_chart(fig_twf, use_container_width=True)
