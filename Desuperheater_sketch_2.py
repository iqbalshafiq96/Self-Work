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

# Driver Speed Input with Unit Dropdown inline directly on the right
col_spd1, col_spd2 = st.sidebar.columns([2, 1])

with col_spd1:
    driver_speed_input = st.number_input(
        "Driver Speed", min_value=1.0, value=1500.0, step=10.0
    )

with col_spd2:
    unit = st.selectbox("Unit", ["RPM", "Hz"], index=0, key="driver_speed_unit_select")

# Gear Orientation Selection
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

# Main Panel - Frequency Calculations Section
st.subheader("Frequency Calculations")

st.markdown(
    f"**Pinion Running Speed ($f_{{pinion}}$):** `{f_pinion_hz:.2f} Hz` / `{f_pinion_hz*60:.0f} RPM`"
)
st.markdown(
    f"**Gear Running Speed ($f_{{gear}}$):** `{f_gear_hz:.2f} Hz` / `{f_gear_hz*60:.0f} RPM`"
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
        ["Pinion Side", "Gear Side", "Both Sides"],
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
if sideband_source in ["Pinion Side", "Both Sides"]:
    mod_freqs.append(("Pinion", f_pinion_hz))
if sideband_source in ["Gear Side", "Both Sides"]:
    mod_freqs.append(("Gear", f_gear_hz))

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
            prefix = "P" if source_name == "Pinion" else "G"
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

for name, val_hz in [("Pinion", f_pinion_hz), ("Gear", f_gear_hz)]:
    val_scaled = val_hz * scale_factor
    if 0 < val_scaled <= fmax:
        tick_vals.append(val_scaled)
        x_1x.append(val_scaled)
        amp_1x.append(0.5)
        lbl_1x.append(f"1x {name}")

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

fig_spec = go.Figure()

all_x = x_1x + x_gmf + x_sb
all_amps = amp_1x + amp_gmf + amp_sb

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
            textposition="middle right",
            textfont=dict(color="darkorange", size=11, family="Segoe UI"),
            textangle=-90,
            marker=dict(color="darkorange", size=9),
            cliponaxis=False,
            name="1x Running Speeds",
        )
    )

if x_gmf:
    fig_spec.add_trace(
        go.Scatter(
            x=x_gmf,
            y=amp_gmf,
            mode="markers+text",
            text=lbl_gmf,
            textposition="middle right",
            textfont=dict(color="crimson", size=11, family="Segoe UI"),
            textangle=-90,
            marker=dict(color="crimson", size=9),
            cliponaxis=False,
            name="GMF Harmonics",
        )
    )

# Highlight Natural Frequency (f_n) on Spectrum if specified (> 0)
if fn_hz > 0:
    fn_scaled = fn_hz * scale_factor
    if 0 < fn_scaled <= fmax:
        fig_spec.add_trace(
            go.Scatter(
                x=[fn_scaled],
                y=[1.2],
                mode="markers+text",
                text=[f"f_n ({fn_scaled:.1f})"],
                textposition="middle right",
                textfont=dict(color="purple", size=11, family="Segoe UI"),
                textangle=-90,
                marker=dict(
                    color="gold",
                    size=12,
                    symbol="diamond",
                    line=dict(color="purple", width=2),
                ),
                name="Natural Frequency (f_n)",
            )
        )
        tick_vals.append(fn_scaled)

clean_ticks = sorted(list(set([t for t in tick_vals if t > 0])))

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
    yaxis=dict(range=[0, 1.65]),
    template="plotly_white",
    height=480,
    margin=dict(l=40, r=20, t=60, b=40),
)

st.plotly_chart(fig_spec, use_container_width=True)
