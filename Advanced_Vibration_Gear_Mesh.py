import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Multi-Stage Gear Mesh Analysis", layout="wide")
st.title("Multi-Stage Gear Mesh & Frequency Spectrum Analysis")

# --- 1. PRESET DEFINITIONS & STATE MANAGEMENT ---
PRESETS = {
    "User Custom": {
        "num_stages": 1,
        "n_gear_1": 57,
        "n_pinion_1": 19,
        "n_gear_2": 60,
        "n_pinion_2": 20,
        "n_gear_3": 50,
        "n_pinion_3": 25,
        "n_gear_4": 40,
        "n_pinion_4": 20,
        "fn_input": 0.0,
        "fn_unit": "Hz",
        "driver_speed_input": 1500.0,
        "unit": "RPM",
        "orientation": "Speed Increaser (Driver = Gear)",
        "amp_1x_gear": 0.20,
        "amp_1x_pinion": 0.10,
        "amp_gmf_1": 1.50,
        "amp_fn": 0.00,
        "sideband_source": "Gear Side",
        "selected_orders": [1, 2],
        "twf_ref_shaft": "Gear (1x)",
        "twf_revolutions": 6.0,
        "twf_noise_level": 0.000,
        "optimize_option": "1x GMF (+ Sidebands)",
    },
    "Scenario 1: Unbalance / Misalignment": {
        "num_stages": 1,
        "n_gear_1": 57,
        "n_pinion_1": 19,
        "fn_input": 0.0,
        "fn_unit": "Hz",
        "driver_speed_input": 1800.0,
        "unit": "RPM",
        "orientation": "Speed Increaser (Driver = Gear)",
        "amp_1x_gear": 0.80,
        "amp_1x_pinion": 0.20,
        "amp_gmf_1": 0.30,
        "amp_fn": 0.00,
        "sideband_source": "Gear Side",
        "selected_orders": [1],
        "twf_ref_shaft": "Gear (1x)",
        "twf_revolutions": 5.0,
        "twf_noise_level": 0.010,
        "optimize_option": "Manual Slider",
    },
    "Scenario 2: Resonant Gear Excitation": {
        "num_stages": 1,
        "n_gear_1": 57,
        "n_pinion_1": 19,
        "fn_input": 475.0,
        "fn_unit": "Hz",
        "driver_speed_input": 1500.0,
        "unit": "RPM",
        "orientation": "Speed Reducer (Driver = Pinion)",
        "amp_1x_gear": 0.10,
        "amp_1x_pinion": 0.10,
        "amp_gmf_1": 1.50,
        "amp_fn": 2.00,
        "sideband_source": "Gear Side",
        "selected_orders": [1, 2],
        "twf_ref_shaft": "Gear (1x)",
        "twf_revolutions": 5.0,
        "twf_noise_level": 0.020,
        "optimize_option": "Manual Slider",
    },
    "Scenario 3: Broken Tooth (Speed Increaser)": {
        "num_stages": 1,
        "n_gear_1": 57,
        "n_pinion_1": 19,
        "fn_input": 0.0,
        "fn_unit": "Hz",
        "driver_speed_input": 1500.0,
        "unit": "RPM",
        "orientation": "Speed Increaser (Driver = Gear)",
        "amp_1x_gear": 0.20,
        "amp_1x_pinion": 0.10,
        "amp_gmf_1": 1.50,
        "amp_fn": 0.00,
        "sideband_source": "Gear Side",
        "selected_orders": [1, 2],
        "twf_ref_shaft": "Gear (1x)",
        "twf_revolutions": 6.0,
        "twf_noise_level": 0.000,
        "optimize_option": "1x GMF (+ Sidebands)",
    },
}

defaults = PRESETS["User Custom"]
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "preset_select" not in st.session_state:
    st.session_state["preset_select"] = "User Custom"


def on_preset_change():
    selected = st.session_state["preset_select"]
    if selected in PRESETS and PRESETS[selected] is not None:
        p = PRESETS[selected]
        for k, v in p.items():
            st.session_state[k] = v


def on_input_change():
    st.session_state["preset_select"] = "User Custom"


# --- 2. SIDEBAR CONTROLS ---

st.sidebar.header("Gearbox Configuration")

num_stages = st.sidebar.selectbox(
    "Number of Gear Stages",
    options=[1, 2, 3, 4],
    index=st.session_state["num_stages"] - 1,
    key="num_stages",
    on_change=on_input_change,
)

stage_gears = []
for idx in range(1, num_stages + 1):
    st.sidebar.markdown(f"**Stage {idx} Geometry**")
    col_g, col_p = st.sidebar.columns(2)
    with col_g:
        n_g = st.number_input(
            f"Stage {idx} Gear Teeth",
            min_value=1,
            value=int(st.session_state.get(f"n_gear_{idx}", 50)),
            step=1,
            key=f"n_gear_{idx}",
            on_change=on_input_change,
        )
    with col_p:
        n_p = st.number_input(
            f"Stage {idx} Pinion Teeth",
            min_value=1,
            value=int(st.session_state.get(f"n_pinion_{idx}", 20)),
            step=1,
            key=f"n_pinion_{idx}",
            on_change=on_input_change,
        )
    stage_gears.append({"n_gear": n_g, "n_pinion": n_p})

col_fn1, col_fn2 = st.sidebar.columns([2, 1])
with col_fn1:
    fn_input = st.number_input(
        "Gear Natural Freq (f_n)",
        min_value=0.0,
        value=float(st.session_state["fn_input"]),
        step=50.0,
        key="fn_input",
        on_change=on_input_change,
        help="Set to 0 if not specified.",
    )
with col_fn2:
    fn_unit = st.selectbox(
        "Unit",
        ["Hz", "RPM"],
        key="fn_unit",
        on_change=on_input_change,
    )

fn_hz = fn_input if fn_unit == "Hz" else fn_input / 60.0

st.sidebar.header("Operating Parameters")
col_spd1, col_spd2 = st.sidebar.columns([2, 1])

with col_spd1:
    driver_speed_input = st.number_input(
        "Driver Speed",
        min_value=1.0,
        value=float(st.session_state["driver_speed_input"]),
        step=10.0,
        key="driver_speed_input",
        on_change=on_input_change,
    )

with col_spd2:
    unit = st.selectbox(
        "Unit",
        ["RPM", "Hz"],
        key="unit",
        on_change=on_input_change,
    )

orientation_options = [
    "Speed Increaser (Driver = Gear)",
    "Speed Reducer (Driver = Pinion)",
]
orientation = st.sidebar.selectbox(
    "Gearbox Orientation",
    orientation_options,
    key="orientation",
    on_change=on_input_change,
)

# Core Multi-Stage Speed Computations
fr_driver_hz = driver_speed_input if unit == "Hz" else driver_speed_input / 60.0
is_reducer = "Reducer" in orientation

stage_results = []
current_input_speed = fr_driver_hz
overall_ratio = 1.0

for idx, stage in enumerate(stage_gears):
    ratio = stage["n_gear"] / stage["n_pinion"]
    if is_reducer:
        # Pinion drives Gear
        f_in = current_input_speed
        f_out = current_input_speed / ratio
        gmf = f_in * stage["n_pinion"]
        overall_ratio *= ratio
    else:
        # Gear drives Pinion
        f_in = current_input_speed
        f_out = current_input_speed * ratio
        gmf = f_in * stage["n_gear"]
        overall_ratio *= ratio

    stage_results.append({
        "Stage": idx + 1,
        "Input Speed (Hz)": f_in,
        "Output Speed (Hz)": f_out,
        "Ratio": ratio,
        "GMF (Hz)": gmf,
    })
    current_input_speed = f_out

fr_driven_hz = current_input_speed
gmf_hz = stage_results[0]["GMF (Hz)"]  # Primary GMF for Stage 1

f_gear_hz = stage_results[0]["Input Speed (Hz)"] if is_reducer else stage_results[0]["Output Speed (Hz)"]
f_pinion_hz = stage_results[0]["Output Speed (Hz)"] if is_reducer else stage_results[0]["Input Speed (Hz)"]

st.sidebar.metric(
    label="Final Driven Speed",
    value=f"{fr_driven_hz * 60.0:.1f} RPM",
    delta=f"{fr_driven_hz:.2f} Hz",
    delta_color="off",
)

st.sidebar.markdown("---")

# Spectrum Simulation Controls
st.sidebar.header("Spectrum Simulation Controls")

st.sidebar.selectbox(
    "Simulation Preset",
    options=list(PRESETS.keys()),
    key="preset_select",
    on_change=on_preset_change,
    help="Selecting a scenario loads pre-configured parameters.",
)

col_amp1, col_amp2 = st.sidebar.columns(2)
with col_amp1:
    amp_1x_gear = st.number_input(
        "1x Gear Amp (g)",
        min_value=0.00,
        max_value=5.00,
        value=float(st.session_state["amp_1x_gear"]),
        step=0.05,
        key="amp_1x_gear",
        on_change=on_input_change,
    )
    amp_gmf_1 = st.number_input(
        "GMF 1x Base Amp (g)",
        min_value=0.00,
        max_value=5.00,
        value=float(st.session_state["amp_gmf_1"]),
        step=0.05,
        key="amp_gmf_1",
        on_change=on_input_change,
    )

with col_amp2:
    amp_1x_pinion = st.number_input(
        "1x Pinion Amp (g)",
        min_value=0.00,
        max_value=5.00,
        value=float(st.session_state["amp_1x_pinion"]),
        step=0.05,
        key="amp_1x_pinion",
        on_change=on_input_change,
    )
    amp_fn = st.number_input(
        "Gear Fn Amp (g)",
        min_value=0.00,
        max_value=5.00,
        value=float(st.session_state["amp_fn"]),
        step=0.05,
        key="amp_fn",
        on_change=on_input_change,
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
    key="optimize_option",
    on_change=on_input_change,
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
        f"*(Targeting Stage 1 `{optimize_option[:2]}` at `{gmf_hz * int(optimize_option[0]):.1f} Hz`)*"
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
        n_p1 = stage_gears[0]["n_pinion"]
        default_fmax = float(np.ceil(3.5 * n_p1)) if n_p1 > 0 else 100.0
        fmax = st.sidebar.slider(
            "Spectrum Fmax (Orders)",
            min_value=5.0,
            max_value=200.0,
            value=max(20.0, default_fmax),
            step=1.0,
        )

# Time Waveform Controls
st.sidebar.header("Time Waveform Controls")
twf_ref_shaft = st.sidebar.selectbox(
    "Reference Shaft for Revolutions",
    ["Gear (1x)", "Pinion (1x)", "Driver Shaft"],
    key="twf_ref_shaft",
    on_change=on_input_change,
)
twf_revolutions = st.sidebar.slider(
    "Plot Duration (Shaft Revolutions)",
    min_value=1.0,
    max_value=20.0,
    value=float(st.session_state["twf_revolutions"]),
    step=0.5,
    key="twf_revolutions",
    on_change=on_input_change,
)
twf_noise_level = st.sidebar.slider(
    "Noise Level (g RMS)",
    min_value=0.00,
    max_value=0.10,
    value=float(st.session_state["twf_noise_level"]),
    step=0.005,
    key="twf_noise_level",
    on_change=on_input_change,
)

# --- 3. MAIN PANEL ANALYSIS & MULTI-STAGE TABLE ---

st.subheader("Frequency Calculations")

st.markdown(f"**Overall Gearbox Ratio:** `{overall_ratio:.3f}`")
st.markdown(
    f"**Input Driver Speed:** `{fr_driver_hz:.2f} Hz` / `{fr_driver_hz * 60:.0f} RPM`"
)
st.markdown(
    f"**Final Driven Speed:** `{fr_driven_hz:.2f} Hz` / `{fr_driven_hz * 60:.0f} RPM`"
)

# Display stage-by-stage calculations
stage_summary = []
for res in stage_results:
    stage_summary.append({
        "Stage": f"Stage {res['Stage']}",
        "Input Speed (RPM)": f"{res['Input Speed (Hz)'] * 60:.1f}",
        "Output Speed (RPM)": f"{res['Output Speed (Hz)'] * 60:.1f}",
        "Stage Ratio": f"{res['Ratio']:.3f}",
        "GMF (Hz)": f"{res['GMF (Hz)']:.2f}",
        "GMF (CPM/RPM)": f"{res['GMF (Hz)'] * 60:.0f}",
    })

st.markdown("### Stage Breakdown")
st.table(stage_summary)

if fn_hz > 0:
    fn_display = fn_hz * scale_factor
    st.markdown(
        f"**Gear Natural Frequency ($f_n$):** `{fn_display:.1f} {unit_label}` ({fn_hz:.2f} Hz)"
    )

st.markdown("### GMF Harmonics & Sidebands (Stage 1 Reference)")

col_sb1, col_sb2 = st.columns([1, 1])
with col_sb1:
    sideband_source = st.radio(
        "Sideband Reference Source:",
        ["Gear Side", "Pinion Side", "Both Sides"],
        key="sideband_source",
        on_change=on_input_change,
        horizontal=True,
    )
with col_sb2:
    selected_orders = st.multiselect(
        "Select Sideband Multipliers (k):",
        options=[1, 2, 3, 4, 5],
        key="selected_orders",
        on_change=on_input_change,
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
        f"is within 5% of a GMF harmonic or primary sideband!"
    )

st.info(
    f"**Required Spectral Resolution Guidance:**\n"
    f"* **Target Modulating Frequency ($f_{{mod}}$):** `{min_mod_freq:.2f} Hz` ({sideband_source})\n"
    f"* **Max Allowed Step ($\Delta f = f_{{mod}} / 2.5$):** `{req_delta_f:.2f} Hz`\n"
    f"* **Recommended Lines ($N_{{lines}}$):** `{recommended_lines:,} Lines` at $F_{{max}} = {fmax_hz:.0f}\\text{{ Hz}}$\n"
    f"* **Achieved Resolution:** `{actual_delta_f:.3f} Hz/line` | **Duration ($T_{{record}}$):** `{recording_time_sec:.2f} s`"
)

# --- 4. SPECTRUM PLOT ---

st.subheader("Simulated Vibration Spectrum")

noise_floor = 0.02
x_1x, amp_1x, lbl_1x = [], [], []
x_gmf, amp_gmf, lbl_gmf = [], [], []
x_sb, amp_sb = [], []
tick_vals = []

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

if amp_gmf_1 > 0:
    for h in range(1, max_harmonic_order + 1):
        center_hz = h * gmf_hz
        center_scaled = center_hz * scale_factor
        base_amp = amp_gmf_1 / (h**0.7)

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

fig_spec = go.Figure()

all_x = x_1x + x_gmf + x_sb
all_amps = amp_1x + amp_gmf + amp_sb

if fn_hz > 0 and amp_fn > 0:
    fn_scaled = fn_hz * scale_factor
    if 0 < fn_scaled <= fmax:
        all_x.append(fn_scaled)
        all_amps.append(amp_fn)

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
            font=dict(color="darkorange", size=11),
        )

if x_gmf:
    fig_spec.add_trace(
        go.Scatter(
            x=x_gmf,
            y=amp_gmf,
            mode="markers",
            marker=dict(color="crimson", size=9),
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
            font=dict(color="crimson", size=11),
        )

if fn_hz > 0 and amp_fn > 0:
    fn_scaled = fn_hz * scale_factor
    if 0 < fn_scaled <= fmax:
        fig_spec.add_vrect(
            x0=fn_scaled * 0.95,
            x1=fn_scaled * 1.05,
            fillcolor="purple",
            opacity=0.15,
            layer="below",
            line_dash="dot",
            line_color="purple",
        )
        fig_spec.add_trace(
            go.Scatter(
                x=[fn_scaled],
                y=[amp_fn],
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
            y=amp_fn + 0.05,
            text=f"f_n ({fn_scaled:.1f})",
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="purple", size=11),
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

# --- 5. TIME WAVEFORM PLOT (SYNTHESIS INTEGRATED FROM CODE A) ---

st.subheader("Simulated Time Waveform")

# Determine Reference Shaft Speed for TWF Time Base
if twf_ref_shaft == "Gear (1x)":
    ref_freq_hz = f_gear_hz
elif twf_ref_shaft == "Pinion (1x)":
    ref_freq_hz = f_pinion_hz
else:
    ref_freq_hz = fr_driver_hz

t_max = twf_revolutions / ref_freq_hz if ref_freq_hz > 0 else 0.1
fs = max(10000.0, 10.0 * fmax_hz)
t = np.linspace(0, t_max, int(fs * t_max), endpoint=False)

signal = np.zeros_like(t)

# 1x Shaft Fundamental Components
if amp_1x_gear > 0 and (0 < f_gear_hz * scale_factor <= fmax):
    signal += amp_1x_gear * np.sin(2 * np.pi * f_gear_hz * t)

if amp_1x_pinion > 0 and (0 < f_pinion_hz * scale_factor <= fmax):
    signal += amp_1x_pinion * np.sin(2 * np.pi * f_pinion_hz * t + np.pi / 4)

# GMF Harmonics and Sideband Amplitude Modulation Synthesis
if amp_gmf_1 > 0:
    for h in range(1, max_harmonic_order + 1):
        center_hz = h * gmf_hz
        if not (0 < center_hz * scale_factor <= fmax):
            continue

        base_amp = amp_gmf_1 / (h**0.7)
        mod_envelope = np.ones_like(t)

        for source_name, freq in mod_freqs:
            for k in sb_orders:
                lower_hz = center_hz - (k * freq)
                upper_hz = center_hz + (k * freq)

                if (0 < lower_hz * scale_factor <= fmax) or (
                    0 < upper_hz * scale_factor <= fmax
                ):
                    depth = (0.3 / (k**0.8)) * 2.0
                    mod_envelope += depth * np.cos(2 * np.pi * k * freq * t)

        signal += base_amp * mod_envelope * np.sin(2 * np.pi * center_hz * t)

# Gear Natural Frequency Structural Response
if fn_hz > 0 and amp_fn > 0 and (0 < fn_hz * scale_factor <= fmax):
    signal += amp_fn * np.sin(2 * np.pi * fn_hz * t)

# Gaussian Noise Synthesis
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
