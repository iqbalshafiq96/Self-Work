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

# Custom CSS for a clean, modern, high-contrast light theme
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #FAFAFA;
        font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
    }
    
    /* Elegant Metric Cards for Light Mode */
    .metric-card {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 12px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 300;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        color: #111827;
        font-weight: 300;
        font-family: 'Helvetica Neue Light', 'Segoe UI Light', sans-serif;
    }
    .metric-sub {
        font-size: 12px;
        color: #4B5563;
        margin-top: 4px;
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# Classy, Light-Theme Matplotlib styling with thin typography
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue Light', 'Helvetica Neue', 'Segoe UI Light', 'Segoe UI', 'Arial'],
    'font.weight': 'light',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FAFAFA',
    'axes.edgecolor': '#E5E7EB',
    'axes.labelcolor': '#374151',
    'axes.titlecolor': '#111827',
    'axes.labelweight': 'light',
    'axes.titleweight': 'light',
    'xtick.color': '#6B7280',
    'ytick.color': '#6B7280',
    'grid.color': '#E5E7EB',
    'grid.linestyle': ':',
    'grid.alpha': 0.8,
    'text.color': '#111827'
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
        p_crit = CP.PropsSI('Pcrit', fluid) / 1e5  # in bara
        p_min = CP.PropsSI('P_min', fluid) / 1e5   # in bara
        
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
st.title("Refrigeration Cycle Analyzer")
st.caption("Interactive Pressure-Enthalpy ($P-h$) Diagram & Thermodynamic Calculations")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Settings & Inputs")
    
    refrigerant_choice = st.selectbox(
        "Refrigerant",
        ["Ammonia", "R134a"],
        index=0  # Pre-selected to Ammonia
    )
    fluid_map = {"Ammonia": "Ammonia", "R134a": "R134a"}
    fluid = fluid_map[refrigerant_choice]
    
    p_unit = st.selectbox(
        "Pressure Unit",
        ["barg", "bara", "kpag", "kpaa"],
        index=1  # Pre-selected to bara
    )
    
    st.markdown("---")
    st.subheader("Process Parameters")
    
    # Defaults set to your initial condition parameters
    p_suction_in = st.number_input(f"Suction Pressure ({p_unit})", value=7.41, step=0.1, format="%.2f")
    t_suction_in = st.number_input("Suction Temperature (°C)", value=15.60, step=0.1, format="%.2f")
    
    p_discharge_in = st.number_input(f"Discharge Pressure ({p_unit})", value=19.95, step=0.1, format="%.2f")
    t_discharge_in = st.number_input("Discharge Temperature (°C)", value=52.44, step=0.1, format="%.2f")
    
    t_condenser_in = st.number_input("Condenser Outlet Temp (°C)", value=44.04, step=0.1, format="%.2f")

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

    # --- Light & Classy Plotting ---
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    # Saturation Envelopes
    ax.plot(sat_liq_h, sat_p, color='#0284C7', linestyle='--', linewidth=1.2, label='Liquid Saturation')
    ax.plot(sat_vap_h, sat_p, color='#E11D48', linestyle='--', linewidth=1.2, label='Vapor Saturation')

    # Refrigeration Cycle Path
    h_vals = [s1_h, s2_h, s3_h, s4_h]
    p_vals = [p_suction, p_discharge, p_discharge, p_suction]
    
    h_loop = h_vals + [h_vals[0]]
    p_loop = p_vals + [p_vals[0]]

    # Core cycle loop line
    ax.plot(h_loop, p_loop, color='#059669', linewidth=1.8, marker='o', 
            markersize=5, markerfacecolor='#10B981', markeredgecolor='#FFFFFF', markeredgewidth=1, label='Refrigeration Cycle')

    # Mid-segment Directional Arrows
    for k in range(len(h_vals)):
        mid_h = (h_loop[k] + h_loop[k+1]) / 2
        mid_p = (p_loop[k] + p_loop[k+1]) / 2
        dh = (h_loop[k+1] - h_loop[k]) * 0.012
        dp = (p_loop[k+1] - p_loop[k]) * 0.012
        ax.annotate('', xy=(mid_h + dh, mid_p + dp), xytext=(mid_h, mid_p),
                    arrowprops=dict(arrowstyle='->', color='#059669', lw=1.2, mutation_scale=12))

    # Clean Callout Cards on Plot
    labels = [
        (f"S1: Suction\n{p_suction:.2f} bara | {s1_h:.1f} kJ/kg\nSH: {s1_sh:.1f} K", (12, -15), 'left', 'top'),
        (f"S2: Discharge\n{p_discharge:.2f} bara | {s2_h:.1f} kJ/kg\nSH: {s2_sh:.1f} K", (12, 10), 'left', 'bottom'),
        (f"S3: Condenser\n{p_discharge:.2f} bara | {s3_h:.1f} kJ/kg", (-12, 10), 'right', 'bottom'),
        (f"S4: Expansion\n{p_suction:.2f} bara | {s4_h:.1f} kJ/kg", (-12, -15), 'right', 'top')
    ]

    for j, (text, offset, ha, va) in enumerate(labels):
        ax.annotate(text, xy=(h_vals[j], p_vals[j]), xytext=offset,
                    textcoords='offset points', fontsize=8, fontweight='light',
                    color='#1F2937', ha=ha, va=va,
                    bbox=dict(facecolor='#FFFFFF', alpha=0.9, edgecolor='#E5E7EB', boxstyle='round,pad=0.4'))

    # Axis Limits: Cap Y-max at Discharge Pressure (19.95 bara) + 5.0 bar = 24.95 bara
    h_min = min(sat_liq_h + h_vals) - 60
    h_max = max(sat_vap_h + h_vals) + 120
    
    max_specified_pressure = max(p_vals)
    p_max_limit = max_specified_pressure + 5.0

    ax.set_xlim(max(0, h_min), h_max)
    ax.set_ylim(0, p_max_limit)
    
    ax.set_xlabel('Enthalpy (kJ/kg)', fontsize=9.5, labelpad=8)
    ax.set_ylabel('Pressure (bara)', fontsize=9.5, labelpad=8)
    ax.set_title(f'P-h Diagram ({refrigerant_choice})', fontsize=11, pad=12, loc='left')
    ax.grid(True, which="both")
    
    # Custom Legend
    legend = ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB', fontsize=8)
    for text in legend.get_texts():
        text.set_color('#374151')

    plt.tight_layout()
    st.pyplot(fig)

    # --- Metric Cards Display ---
    st.markdown("### State Points Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S1 - Suction</div>
            <div class="metric-value">{s1_h:.1f} <span style="font-size:12px;">kJ/kg</span></div>
            <div class="metric-sub">P: {p_suction:.2f} bara | T: {t_suction_in:.1f}°C</div>
            <div class="metric-sub">Superheat: <b>{s1_sh:.1f} K</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S2 - Discharge</div>
            <div class="metric-value">{s2_h:.1f} <span style="font-size:12px;">kJ/kg</span></div>
            <div class="metric-sub">P: {p_discharge:.2f} bara | T: {t_discharge_in:.1f}°C</div>
            <div class="metric-sub">Superheat: <b>{s2_sh:.1f} K</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S3 - Condenser Outlet</div>
            <div class="metric-value">{s3_h:.1f} <span style="font-size:12px;">kJ/kg</span></div>
            <div class="metric-sub">P: {p_discharge:.2f} bara | T: {t_condenser_in:.1f}°C</div>
            <div class="metric-sub">Sat Temp: {s3_tsat:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S4 - Expansion</div>
            <div class="metric-value">{s4_h:.1f} <span style="font-size:12px;">kJ/kg</span></div>
            <div class="metric-sub">P: {p_suction:.2f} bara | Isenthalpic</div>
            <div class="metric-sub">Sat Temp: {s4_tsat:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Calculation Error: Verify that input pressures and temperatures fall within a valid thermodynamic range. Details: {e}")
