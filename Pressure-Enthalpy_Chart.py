import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import CoolProp.CoolProp as CP

# --- Global Page Config & Styling ---
st.set_page_config(page_title="Refrigeration Cycle Analyzer", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# Set global Matplotlib params
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
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

def get_saturation_curve(fluid, num_points=100):
    """Generates saturation liquid and vapor curves using CoolProp."""
    try:
        p_crit = CP.PropsSI('Pcrit', fluid) / 1e5  # in bara
        p_min = CP.PropsSI('Pmin', fluid) / 1e5    # in bara
        
        # Avoid exact critical point to prevent convergence errors
        pressures = np.linspace(max(p_min, 0.1), p_crit * 0.98, num_points)
        
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

# --- Streamlit UI Layout ---
st.title("🧊 Chiller Refrigeration Cycle Analyzer ($P-h$ Diagram)")

with st.sidebar:
    st.header("1. Options & Units")
    
    refrigerant_choice = st.selectbox(
        "Select Refrigerant",
        ["Ammonia", "R134a"],
        index=0
    )
    # CoolProp internal fluid mapping
    fluid_map = {"Ammonia": "Ammonia", "R134a": "R134a"}
    fluid = fluid_map[refrigerant_choice]
    
    p_unit = st.selectbox(
        "Pressure Unit",
        ["barg", "bara", "kpag", "kpaa"],
        index=0
    )
    
    st.header("2. Cycle Parameters")
    
    # Defaults sensible for Ammonia (barg)
    p_suction_in = st.number_input(f"Suction Pressure ({p_unit})", value=2.0, step=0.1)
    t_suction_in = st.number_input("Suction Temperature (°C)", value=-10.0, step=0.5)
    
    p_discharge_in = st.number_input(f"Discharge Pressure ({p_unit})", value=12.0, step=0.1)
    t_discharge_in = st.number_input("Discharge Temperature (°C)", value=90.0, step=0.5)
    
    t_condenser_in = st.number_input("Condenser Outlet Temperature (°C)", value=30.0, step=0.5)

# --- Perform Calculations ---
try:
    # Set IIR reference state for standard refrigeration conventions
    CP.set_reference_state(fluid, 'IIR')

    # Convert pressures to absolute bar (bara)
    p_suction = convert_to_bara(p_suction_in, p_unit)
    p_discharge = convert_to_bara(p_discharge_in, p_unit)

    # State Point Calculations
    # S1: Suction
    s1_h, s1_tsat = get_point_props(p_suction, t_suction_in, fluid)
    s1_sh = t_suction_in - s1_tsat

    # S2: Discharge
    s2_h, s2_tsat = get_point_props(p_discharge, t_discharge_in, fluid)
    s2_sh = t_discharge_in - s2_tsat

    # S3: Condenser Outlet (Discharge Pressure, Condenser Temp)
    s3_h, s3_tsat = get_point_props(p_discharge, t_condenser_in, fluid)

    # S4: Expansion (Suction Pressure, Isenthalpic from S3)
    s4_h = s3_h
    s4_tsat = CP.PropsSI('T', 'P', p_suction * 1e5, 'Q', 0, fluid) - 273.15

    # Saturation Lines
    sat_liq_h, sat_vap_h, sat_p = get_saturation_curve(fluid)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    # Plot Saturation Curves
    ax.plot(sat_liq_h, sat_p, 'b--', label='Liquid Saturation Line', lw=1.5)
    ax.plot(sat_vap_h, sat_p, 'r--', label='Vapor Saturation Line', lw=1.5)

    # Cycle Loop Coordinates
    h_vals = [s1_h, s2_h, s3_h, s4_h]
    p_vals = [p_suction, p_discharge, p_discharge, p_suction]
    
    h_loop = h_vals + [h_vals[0]]
    p_loop = p_vals + [p_vals[0]]

    # Plot Loop Line and Points
    ax.plot(h_loop, p_loop, 'k-', marker='o', markersize=5, alpha=0.8, lw=2, label='Refrigeration Cycle')

    # Add Mid-segment Arrows
    for k in range(len(h_vals)):
        mid_h = (h_loop[k] + h_loop[k+1]) / 2
        mid_p = (p_loop[k] + p_loop[k+1]) / 2
        dh = (h_loop[k+1] - h_loop[k]) * 0.015
        dp = (p_loop[k+1] - p_loop[k]) * 0.015
        ax.annotate('', xy=(mid_h + dh, mid_p + dp), xytext=(mid_h, mid_p),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.8, mutation_scale=15))

    # Dynamic Annotations for State Points
    labels = [
        (f"S1: Suction\nP: {p_suction:.2f} bara\nT: {t_suction_in:.2f} °C\nh: {s1_h:.2f} kJ/kg\nSuperheat: {s1_sh:.2f} K", (10, -15), 'left', 'top'),
        (f"S2: Discharge\nP: {p_discharge:.2f} bara\nT: {t_discharge_in:.2f} °C\nh: {s2_h:.2f} kJ/kg\nSuperheat: {s2_sh:.2f} K", (10, 10), 'left', 'bottom'),
        (f"S3: Condenser Outlet\nP: {p_discharge:.2f} bara\nT: {t_condenser_in:.2f} °C\nh: {s3_h:.2f} kJ/kg", (-10, 10), 'right', 'bottom'),
        (f"S4: Expansion\nP: {p_suction:.2f} bara\nh: {s4_h:.2f} kJ/kg", (-10, -15), 'right', 'top')
    ]

    for j, (text, offset, ha, va) in enumerate(labels):
        ax.annotate(text, xy=(h_vals[j], p_vals[j]), xytext=offset,
                    textcoords='offset points', fontsize=8, fontweight='bold',
                    ha=ha, va=va, bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'))

    # Axis Limits & Formatting
    h_min = min(sat_liq_h + h_vals) - 50
    h_max = max(sat_vap_h + h_vals) + 100
    p_max = max(sat_p + p_vals) + 2

    ax.set_xlim(max(0, h_min), h_max)
    ax.set_ylim(0, p_max)
    ax.set_xlabel('Enthalpy (kJ/kg)', fontsize=11)
    ax.set_ylabel('Pressure (bara)', fontsize=11)
    ax.set_title(f'Pressure-Enthalpy (P-h) Diagram for {refrigerant_choice}', fontsize=13, pad=12)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(loc='upper right')

    # Display Plot in Streamlit
    st.pyplot(fig)

    # --- Summary Data Table ---
    st.subheader("State Points Summary")
    summary_df = {
        "State Point": ["S1 (Suction)", "S2 (Discharge)", "S3 (Condenser Outlet)", "S4 (Expansion)"],
        "Pressure (bara)": [p_suction, p_discharge, p_discharge, p_suction],
        "Temperature (°C)": [t_suction_in, t_discharge_in, t_condenser_in, "N/A (Two-Phase)"],
        "Enthalpy (kJ/kg)": [f"{s1_h:.2f}", f"{s2_h:.2f}", f"{s3_h:.2f}", f"{s4_h:.2f}"],
        "Sat Temp (°C)": [f"{s1_tsat:.2f}", f"{s2_tsat:.2f}", f"{s3_tsat:.2f}", f"{s4_tsat:.2f}"],
        "Superheat (K)": [f"{s1_sh:.2f}", f"{s2_sh:.2f}", "-", "-"]
    }
    st.dataframe(summary_df, use_container_width=True)

except Exception as e:
    st.error(f"Calculation Error: Please ensure input temperatures and pressures are physically valid for {refrigerant_choice}. Details: {e}")
