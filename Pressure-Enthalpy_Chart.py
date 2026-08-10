import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import CoolProp.CoolProp as CP

# --- Page Configuration ---
st.set_page_config(
    page_title="Refrigeration Cycle Analyzer",
    page_icon="❄️",
    layout="wide"
)

# Custom CSS for a modern, digital dashboard look
st.markdown("""
<style>
    /* Dark theme background tweaks */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Card containers for state point summary */
    .metric-card {
        background: #1e222d;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2e3545;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 20px;
        color: #58a6ff;
        font-weight: 700;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    .metric-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Modern, dark-mode Matplotlib styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'SF Pro Text', 'Segoe UI', 'DejaVu Sans', 'Arial'],
    'font.weight': 'normal',
    'figure.facecolor': '#161b22',
    'axes.facecolor': '#0d1117',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'axes.titlecolor': '#f0f6fc',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'text.color': '#c9d1d9'
})

P_ATM_BAR = 1.01325  # Standard atmospheric pressure in bar

# --- Helper Functions ---
def convert_to_bara(pressure_val, unit):
    """Converts input pressure from user units to absolute bar (bara)."""
    if unit == 'barg':
        return pressure_val + P_ATM_BAR
    elif unit == 'bara':
        return pressure_val
    elif unit == 'kpag':
        return (pressure_val / 100.0) + P_ATM_BAR
    elif unit == 'kpaa':
        return pressure_val / 100.0
    return pressure_val

def get_saturation_curve(fluid, num_points=120):
    """Generates saturation liquid and vapor curves using CoolProp."""
    try:
        # Fixed: Use 'P_min' instead of 'Pmin' to prevent CoolProp parsing errors
        p_crit = CP.PropsSI('Pcrit', fluid) / 1e5  # in bara
        p_min = CP.PropsSI('P_min', fluid) / 1e5   # in bara
        
        # Logarithmic spacing gives better density at low pressures
        p_start = max(p_min, 0.1)
        p_end = p_crit * 0.98
        pressures = np.geomspace(p_start, p_end, num_points)
        
        sat_liq_h, sat_vap_h = [], []
        valid_p = []
        
        for p in pressures:
            p_pa = p * 1e5
            try:
                h_l = CP.PropsSI('H', 'P', p_pa, 'Q', 0, fluid) / 1000.0  # kJ/kg
                h_v = CP.PropsSI('H', 'P', p_pa, 'Q', 1, fluid) / 1000.0  # kJ/kg
                sat_liq_h.append(h_l)
                sat_vap_h.append(h_v)
                valid_p.append(p)
            except Exception:
                continue
                
        return sat_liq_h, sat_vap_h, valid_p
    except Exception as e:
        st.error(f"Error calculating saturation curves: {e}")
        return [], [], []

def get_point_props(p_bara, t_degc, fluid):
    """Calculates Enthalpy (kJ/kg) and Saturation Temperature (°C) at given P & T."""
    p_pa = p_bara * 1e5
    t_k = t_degc + 273.15
    
    h_kj = CP.PropsSI('H', 'P', p_pa, 'T', t_k, fluid) / 1000.0
    t_sat_k = CP.PropsSI('T', 'P', p_pa, 'Q', 0, fluid)
    t_sat_c = t_sat_k - 273.15
    
    return h_kj, t_sat_c

# --- Header Section ---
st.title("❄️ Digital Chiller Cycle Analyzer")
st.caption("Interactive Pressure-Enthalpy ($P-h$) Diagram & Thermodynamic State Calculation")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Settings & Inputs")
    
    refrigerant_choice = st.selectbox(
        "Refrigerant",
        ["Ammonia", "R134a"],
        index=0
    )
    fluid_map = {"Ammonia": "Ammonia", "R134a": "R134a"}
    fluid = fluid_map[refrigerant_choice]
    
    p_unit = st.selectbox(
        "Pressure Unit",
        ["barg", "bara", "kpag", "kpaa"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Process Parameters")
    
    p_suction_in = st.number_input(f"Suction Pressure ({p_unit})", value=2.0, step=0.1)
    t_suction_in = st.number_input("Suction Temperature (°C)", value=-10.0, step=0.5)
    
    p_discharge_in = st.number_input(f"Discharge Pressure ({p_unit})", value=12.0, step=0.1)
    t_discharge_in = st.number_input("Discharge Temperature (°C)", value=90.0, step=0.5)
    
    t_condenser_in = st.number_input("Condenser Outlet Temp (°C)", value=30.0, step=0.5)

# --- Computations ---
try:
    CP.set_reference_state(fluid, 'IIR')

    p_suction = convert_to_bara(p_suction_in, p_unit)
    p_discharge = convert_to_bara(p_discharge_in, p_unit)

    # State Points
    s1_h, s1_tsat = get_point_props(p_suction, t_suction_in, fluid)
    s1_sh = t_suction_in - s1_tsat

    s2_h, s2_tsat = get_point_props(p_discharge, t_discharge_in, fluid)
    s2_sh = t_discharge_in - s2_tsat

    s3_h, s3_tsat = get_point_props(p_discharge, t_condenser_in, fluid)

    s4_h = s3_h
    s4_tsat = CP.PropsSI('T', 'P', p_suction * 1e5, 'Q', 0, fluid) - 273.15

    sat_liq_h, sat_vap_h, sat_p = get_saturation_curve(fluid)

    # --- Modern Dark Plotting ---
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    # Saturation Envelopes with modern subtle colors
    ax.plot(sat_liq_h, sat_p, color='#1f6beb', linestyle='--', linewidth=1.8, label='Liquid Saturation')
    ax.plot(sat_vap_h, sat_p, color='#da3633', linestyle='--', linewidth=1.8, label='Vapor Saturation')

    # Refrigeration Cycle Path
    h_vals = [s1_h, s2_h, s3_h, s4_h]
    p_vals = [p_suction, p_discharge, p_discharge, p_suction]
    
    h_loop = h_vals + [h_vals[0]]
    p_loop = p_vals + [p_vals[0]]

    # Glow effect line under the main loop
    ax.plot(h_loop, p_loop, color='#238636', alpha=0.3, linewidth=6)
    # Core loop line
    ax.plot(h_loop, p_loop, color='#3fb950', linewidth=2.2, marker='o', 
            markersize=6, markerfacecolor='#56d364', markeredgecolor='#ffffff', label='Refrigeration Cycle')

    # Dynamic Arrows
    for k in range(len(h_vals)):
        mid_h = (h_loop[k] + h_loop[k+1]) / 2
        mid_p = (p_loop[k] + p_loop[k+1]) / 2
        dh = (h_loop[k+1] - h_loop[k]) * 0.012
        dp = (p_loop[k+1] - p_loop[k]) * 0.012
        ax.annotate('', xy=(mid_h + dh, mid_p + dp), xytext=(mid_h, mid_p),
                    arrowprops=dict(arrowstyle='->', color='#ffffff', lw=1.5, mutation_scale=12))

    # Clean Dark Callout Cards on Plot
    labels = [
        (f"S1: Suction\n{p_suction:.2f} bara | {s1_h:.1f} kJ/kg\nSH: {s1_sh:.1f} K", (12, -15), 'left', 'top'),
        (f"S2: Discharge\n{p_discharge:.2f} bara | {s2_h:.1f} kJ/kg\nSH: {s2_sh:.1f} K", (12, 10), 'left', 'bottom'),
        (f"S3: Condenser\n{p_discharge:.2f} bara | {s3_h:.1f} kJ/kg", (-12, 10), 'right', 'bottom'),
        (f"S4: Expansion\n{p_suction:.2f} bara | {s4_h:.1f} kJ/kg", (-12, -15), 'right', 'top')
    ]

    for j, (text, offset, ha, va) in enumerate(labels):
        ax.annotate(text, xy=(h_vals[j], p_vals[j]), xytext=offset,
                    textcoords='offset points', fontsize=8.5, fontweight='medium',
                    color='#f0f6fc', ha=ha, va=va,
                    bbox=dict(facecolor='#161b22', alpha=0.9, edgecolor='#30363d', boxstyle='round,pad=0.4'))

    # Axis Formatting & Scaling
    h_min = min(sat_liq_h + h_vals) - 60
    h_max = max(sat_vap_h + h_vals) + 120
    p_max = max(sat_p + p_vals) * 1.15

    ax.set_xlim(max(0, h_min), h_max)
    ax.set_ylim(0, p_max)
    ax.set_xlabel('Enthalpy (kJ/kg)', fontsize=10, labelpad=8)
    ax.set_ylabel('Pressure (bara)', fontsize=10, labelpad=8)
    ax.set_title(f'P-h Diagram ({refrigerant_choice})', fontsize=12, pad=12, loc='left')
    ax.grid(True, which="both")
    
    # Custom Legend
    legend = ax.legend(loc='upper right', frameon=True, facecolor='#161b22', edgecolor='#30363d', fontsize=8.5)
    for text in legend.get_texts():
        text.set_color('#c9d1d9')

    plt.tight_layout()
    st.pyplot(fig)

    # --- Digital Metric Cards Display ---
    st.markdown("### 📊 State Points Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S1 - Suction</div>
            <div class="metric-value">{s1_h:.1f} <span style="font-size:13px;">kJ/kg</span></div>
            <div class="metric-sub">P: {p_suction:.2f} bara | T: {t_suction_in:.1f}°C</div>
            <div class="metric-sub">Superheat: <b>{s1_sh:.1f} K</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S2 - Discharge</div>
            <div class="metric-value">{s2_h:.1f} <span style="font-size:13px;">kJ/kg</span></div>
            <div class="metric-sub">P: {p_discharge:.2f} bara | T: {t_discharge_in:.1f}°C</div>
            <div class="metric-sub">Superheat: <b>{s2_sh:.1f} K</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S3 - Condenser Outlet</div>
            <div class="metric-value">{s3_h:.1f} <span style="font-size:13px;">kJ/kg</span></div>
            <div class="metric-sub">P: {p_discharge:.2f} bara | T: {t_condenser_in:.1f}°C</div>
            <div class="metric-sub">Sat Temp: {s3_tsat:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S4 - Expansion</div>
            <div class="metric-value">{s4_h:.1f} <span style="font-size:13px;">kJ/kg</span></div>
            <div class="metric-sub">P: {p_suction:.2f} bara | Isenthalpic</div>
            <div class="metric-sub">Sat Temp: {s4_tsat:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Calculation Error: Verify that input pressures and temperatures fall within a valid thermodynamic range. Details: {e}")
