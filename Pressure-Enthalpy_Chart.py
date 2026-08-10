import streamlit as st
import numpy as np
import CoolProp.CoolProp as CP
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Refrigeration Cycle Analyzer",
    page_icon="❄️",
    layout="wide"
)

# Custom CSS for a clean metric cards display
st.markdown("""
<style>
    .stApp {
        background-color: #FAFAFA;
        font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
    }
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
    }
    .metric-sub {
        font-size: 12px;
        color: #4B5563;
        margin-top: 4px;
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

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
        index=0
    )
    fluid_map = {"Ammonia": "Ammonia", "R134a": "R134a"}
    fluid = fluid_map[refrigerant_choice]
    
    p_unit = st.selectbox(
        "Pressure Unit",
        ["barg", "bara", "kpag", "kpaa"],
        index=1
    )
    
    st.markdown("---")
    st.subheader("Process Parameters")
    
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

    # --- Plotly Interactive Chart ---
    fig = go.Figure()

    # 1. Saturation Curves
    fig.add_trace(go.Scatter(
        x=sat_liq_h, y=sat_p,
        mode='lines',
        name='Liquid Saturation',
        line=dict(color='#0284C7', width=1.5, dash='dash'),
        hovertemplate='Enthalpy: %{x:.1f} kJ/kg<br>Pressure: %{y:.2f} bara<extra>Saturated Liquid</extra>'
    ))

    fig.add_trace(go.Scatter(
        x=sat_vap_h, y=sat_p,
        mode='lines',
        name='Vapor Saturation',
        line=dict(color='#E11D48', width=1.5, dash='dash'),
        hovertemplate='Enthalpy: %{x:.1f} kJ/kg<br>Pressure: %{y:.2f} bara<extra>Saturated Vapor</extra>'
    ))

    # 2. Cycle Loop Path
    h_vals = [s1_h, s2_h, s3_h, s4_h]
    p_vals = [p_suction, p_discharge, p_discharge, p_suction]
    h_loop = h_vals + [h_vals[0]]
    p_loop = p_vals + [p_vals[0]]

    fig.add_trace(go.Scatter(
        x=h_loop, y=p_loop,
        mode='lines+markers',
        name='Refrigeration Cycle',
        line=dict(color='#059669', width=2.5),
        marker=dict(size=8, color='#10B981', line=dict(color='#FFFFFF', width=1.5)),
        hoverinfo='skip'
    ))

    # 3. State Point Markers & Annotations
    point_labels = ["S1: Suction", "S2: Discharge", "S3: Condenser", "S4: Expansion"]
    point_texts = [
        f"<b>S1: Suction</b><br>{p_suction:.2f} bara | {s1_h:.1f} kJ/kg<br>SH: {s1_sh:.1f} K",
        f"<b>S2: Discharge</b><br>{p_discharge:.2f} bara | {s2_h:.1f} kJ/kg<br>SH: {s2_sh:.1f} K",
        f"<b>S3: Condenser</b><br>{p_discharge:.2f} bara | {s3_h:.1f} kJ/kg",
        f"<b>S4: Expansion</b><br>{p_suction:.2f} bara | {s4_h:.1f} kJ/kg"
    ]
    
    # Text offset positions for annotations
    text_positions = ["top right", "top right", "top left", "bottom left"]

    for idx in range(4):
        fig.add_trace(go.Scatter(
            x=[h_vals[idx]], y=[p_vals[idx]],
            mode='markers+text',
            name=point_labels[idx],
            text=[f"  S{idx+1}  "],
            textposition=text_positions[idx],
            marker=dict(size=10, color='#059669'),
            hovertemplate=f"%{{text}}<br>Enthalpy: %{{x:.1f}} kJ/kg<br>Pressure: %{{y:.2f}} bara<extra>{point_labels[idx]}</extra>"
        ))

        # Callout Annotations
        fig.add_annotation(
            x=h_vals[idx],
            y=p_vals[idx],
            text=point_texts[idx],
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor="#6B7280",
            ax=45 if idx in [0, 1] else -45,
            ay=-35 if idx in [1, 2] else 35,
            bgcolor="#FFFFFF",
            bordercolor="#E5E7EB",
            borderwidth=1,
            borderpad=4,
            font=dict(size=11, color="#1F2937")
        )

    # Axis Limits Setup
    h_min = max(0, min(sat_liq_h + h_vals) - 60)
    h_max = max(sat_vap_h + h_vals) + 120
    p_max_limit = max(p_vals) + 5.0

    # Layout Configuration
    fig.update_layout(
        title=dict(text=f'P-h Diagram ({refrigerant_choice})', font=dict(size=16, color='#111827')),
        xaxis=dict(
            title='Enthalpy (kJ/kg)',
            range=[h_min, h_max],
            showgrid=True,
            gridcolor='#E5E7EB',
            zeroline=False
        ),
        yaxis=dict(
            title='Pressure (bara)',
            range=[0, p_max_limit],
            showgrid=True,
            gridcolor='#E5E7EB',
            zeroline=False
        ),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='#FFFFFF',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E5E7EB',
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=60, b=50),
        height=550,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

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
