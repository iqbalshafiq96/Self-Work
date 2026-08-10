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

# Custom CSS for a refined, modern light aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1a1a1a;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Clean Minimalist Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 22px;
        color: #0f172a;
        font-weight: 400;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 12px;
        color: #64748b;
        font-weight: 300;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Elegant, clean, and light Matplotlib styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Segoe UI', 'Avenir', 'Inter', 'Arial'],
    'font.weight': '300',
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 0.8,
    'axes.labelcolor': '#334155',
    'axes.titlecolor': '#0f172a',
    'axes.labelweight': '300',
    'axes.titleweight': '400',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'grid.color': '#f1f5f9',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'grid.alpha': 1.0,
    'text.color': '#1e293b'
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
        p_crit = CP.PropsSI('Pcrit', fluid) / 1e5
        p_min = CP.PropsSI('P_min', fluid) / 1e5
        
        p_start = max(p_min, 0.1)
        p_end = p_crit * 0.98
        pressures = np.geomspace(p_start, p_end, num_points)
        
        sat_liq_h, sat_vap_h = [], []
        valid_p = []
        
        for p in pressures:
            p_pa = p * 1e5
            try:
                h_l = CP.PropsSI('H', 'P', p_pa, 'Q', 0, fluid) / 1000.0
                h_v = CP.PropsSI('H', 'P', p_pa, 'Q', 1, fluid) / 1000.0
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
st.title("Refrigeration Cycle Analysis")
st.caption("Pressure-Enthalpy ($P-h$) thermodynamic diagram & process calculations")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Configuration")
    
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
    st.subheader("Process Operating Points")
    
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

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)

    # Saturation Envelopes (Thin, understated lines)
    ax.plot(sat_liq_h, sat_p, color='#0284c7', linestyle='--', linewidth=1.2, label='Liquid Saturation')
    ax.plot(sat_vap_h, sat_p, color='#e11d48', linestyle='--', linewidth=1.2, label='Vapor Saturation')

    # Cycle Loop
    h_vals = [s1_h, s2_h, s3_h, s4_h]
    p_vals = [p_suction, p_discharge, p_discharge, p_suction]
    
    h_loop = h_vals + [h_vals[0]]
    p_loop = p_vals + [p_vals[0]]

    # Sleek Cycle Path
    ax.plot(h_loop, p_loop, color='#0f172a', linewidth=1.5, marker='o', 
            markersize=5, markerfacecolor='#0f172a', markeredgecolor='#ffffff', markeredgewidth=1.2, label='Cycle Path')

    # Mid-segment directional arrows
    for k in range(len(h_vals)):
        mid_h = (h_loop[k] + h_loop[k+1]) / 2
        mid_p = (p_loop[k] + p_loop[k+1]) / 2
        dh = (h_loop[k+1] - h_loop[k]) * 0.012
        dp = (p_loop[k+1] - p_loop[k]) * 0.012
        ax.annotate('', xy=(mid_h + dh, mid_p + dp), xytext=(mid_h, mid_p),
                    arrowprops=dict(arrowstyle='->', color='#0f172a', lw=1.2, mutation_scale=10))

    # Refined Callout Boxes
    labels = [
        (f"S1: Suction\n{p_suction:.2f} bara | {s1_h:.1f} kJ/kg\nSH: {s1_sh:.1f} K", (10, -12), 'left', 'top'),
        (f"S2: Discharge\n{p_discharge:.2f} bara | {s2_h:.1f} kJ/kg\nSH: {s2_sh:.1f} K", (10, 10), 'left', 'bottom'),
        (f"S3: Condenser\n{p_discharge:.2f} bara | {s3_h:.1f} kJ/kg", (-10, 10), 'right', 'bottom'),
        (f"S4: Expansion\n{p_suction:.2f} bara | {s4_h:.1f} kJ/kg", (-10, -12), 'right', 'top')
    ]

    for j, (text, offset, ha, va) in enumerate(labels):
        ax.annotate(text, xy=(h_vals[j], p_vals[j]), xytext=offset,
                    textcoords='offset points', fontsize=8, fontweight='300',
                    color='#334155', ha=ha, va=va,
                    bbox=dict(facecolor='#ffffff', alpha=0.9, edgecolor='#e2e8f0', boxstyle='round,pad=0.4', lw=0.8))

    # Axis Limits & Scaling
    h_min = min(sat_liq_h + h_vals) - 60
    h_max = max(sat_vap_h + h_vals) + 120
    
    # Set Y max exactly +5 bar above the specified discharge pressure
    p_max_limit = p_discharge + 5.0

    ax.set_xlim(max(0, h_min), h_max)
    ax.set_ylim(0, p_max_limit)
    
    ax.set_xlabel('Enthalpy (kJ/kg)', fontsize=9.5, fontweight='300', labelpad=8)
    ax.set_ylabel('Pressure (bara)', fontsize=9.5, fontweight='300', labelpad=8)
    ax.set_title(f'P-h Diagram — {refrigerant_choice}', fontsize=11, fontweight='400', pad=12, loc='left')
    ax.grid(True, which="both")
    
    # Clean, borderless legend
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=8)

    # Remove top and right spines for a modern minimalist look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    # --- Minimalist Metric Display Cards ---
    st.markdown("##### State Point Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S1 Suction</div>
            <div class="metric-value">{s1_h:.1f} <span style="font-size:12px; color:#64748b;">kJ/kg</span></div>
            <div class="metric-sub">{p_suction:.2f} bara · {t_suction_in:.1f}°C</div>
            <div class="metric-sub">Superheat: {s1_sh:.1f} K</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S2 Discharge</div>
            <div class="metric-value">{s2_h:.1f} <span style="font-size:12px; color:#64748b;">kJ/kg</span></div>
            <div class="metric-sub">{p_discharge:.2f} bara · {t_discharge_in:.1f}°C</div>
            <div class="metric-sub">Superheat: {s2_sh:.1f} K</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S3 Condenser Outlet</div>
            <div class="metric-value">{s3_h:.1f} <span style="font-size:12px; color:#64748b;">kJ/kg</span></div>
            <div class="metric-sub">{p_discharge:.2f} bara · {t_condenser_in:.1f}°C</div>
            <div class="metric-sub">Sat Temp: {s3_tsat:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">S4 Expansion</div>
            <div class="metric-value">{s4_h:.1f} <span style="font-size:12px; color:#64748b;">kJ/kg</span></div>
            <div class="metric-sub">{p_suction:.2f} bara · Isenthalpic</div>
            <div class="metric-sub">Sat Temp: {s4_tsat:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Calculation Error: Please verify the input values. Details: {e}")
