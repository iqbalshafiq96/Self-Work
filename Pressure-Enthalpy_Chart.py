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

# Custom CSS for Light Theme Metric Cards
st.markdown("""
<style>
    .stApp {
        background-color: #FAFAFA;
        font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 11px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 20px;
        color: #111827;
        font-weight: 600;
    }
    .metric-sub {
        font-size: 11px;
        color: #4B5563;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

P_ATM_BAR = 1.01325  # Atmospheric pressure in bar

# --- Helper Functions ---
def convert_to_bara(pressure_val, unit):
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
    try:
        p_crit = CP.PropsSI('Pcrit', fluid) / 1e5
        p_min = CP.PropsSI('P_min', fluid) / 1e5
        p_start = max(p_min, 0.1)
        p_end = p_crit * 0.98
        pressures = np.geomspace(p_start, p_end, num_points)
        
        sat_liq_h, sat_vap_h, valid_p = [], [], []
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
    p_pa = p_bara * 1e5
    t_k = t_degc + 273.15
    h_kj = CP.PropsSI('H', 'P', p_pa, 'T', t_k, fluid) / 1000.0
    t_sat_k = CP.PropsSI('T', 'P', p_pa, 'Q', 0, fluid)
    t_sat_c = t_sat_k - 273.15
    return h_kj, t_sat_c

def calculate_cycle_points(p_suc_in, t_suc_in, p_dis_in, t_dis_in, t_cond_in, p_unit, fluid):
    p_suction = convert_to_bara(p_suc_in, p_unit)
    p_discharge = convert_to_bara(p_dis_in, p_unit)

    s1_h, s1_tsat = get_point_props(p_suction, t_suc_in, fluid)
    s1_sh = t_suc_in - s1_tsat

    s2_h, s2_tsat = get_point_props(p_discharge, t_dis_in, fluid)
    s2_sh = t_dis_in - s2_tsat

    s3_h, s3_tsat = get_point_props(p_discharge, t_cond_in, fluid)
    s4_h = s3_h
    s4_tsat = CP.PropsSI('T', 'P', p_suction * 1e5, 'Q', 0, fluid) - 273.15

    q_in = s1_h - s4_h
    w_in = s2_h - s1_h
    cop = q_in / w_in if w_in != 0 else 0

    return {
        'p_suction': p_suction, 'p_discharge': p_discharge,
        's1_h': s1_h, 's1_sh': s1_sh, 't_suc': t_suc_in, 's1_tsat': s1_tsat,
        's2_h': s2_h, 's2_sh': s2_sh, 't_dis': t_dis_in, 's2_tsat': s2_tsat,
        's3_h': s3_h, 's3_tsat': s3_tsat, 't_cond': t_cond_in,
        's4_h': s4_h, 's4_tsat': s4_tsat,
        'cop': cop, 'q_in': q_in, 'w_in': w_in
    }

# --- Header Section ---
st.title("Refrigeration Cycle Analyzer")
st.caption("Interactive Pressure-Enthalpy ($P-h$) Diagram & Thermodynamic Calculations")
st.caption("Developed by Iqbal SHERPA_20260810")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Analysis Mode")
    analysis_mode = st.radio(
        "Select Operating Mode",
        ["Single Profile", "Compare Profiles"],
        index=0,
        help="Choose whether to analyze one profile or overlay two profiles for comparison."
    )
    
    show_callouts = st.checkbox("Show Graph Callouts", value=True, help="Toggle on-graph annotations for state points.")
    
    st.markdown("---")
    st.header("2. Fluid & Units")
    refrigerant_choice = st.selectbox("Refrigerant", ["Ammonia", "R134a"], index=0)
    fluid = refrigerant_choice
    p_unit = st.selectbox("Pressure Unit", ["barg", "bara", "kpag", "kpaa"], index=1)

    st.markdown("---")
    
    if analysis_mode == "Single Profile":
        st.subheader("Process Parameters")
        p_suc_A = st.number_input(f"Suction Pressure ({p_unit})", value=7.41, step=0.1, key="p_suc_A")
        t_suc_A = st.number_input("Suction Temp (°C)", value=15.60, step=0.1, key="t_suc_A")
        p_dis_A = st.number_input(f"Discharge Pressure ({p_unit})", value=19.95, step=0.1, key="p_dis_A")
        t_dis_A = st.number_input("Discharge Temp (°C)", value=52.44, step=0.1, key="t_dis_A")
        t_cond_A = st.number_input("Condenser Outlet Temp (°C)", value=44.04, step=0.1, key="t_cond_A")
    else:
        tab_a, tab_b = st.tabs(["Profile A (Primary)", "Profile B (Compare)"])
        with tab_a:
            p_suc_A = st.number_input(f"Suction Press ({p_unit})", value=7.41, step=0.1, key="p_suc_A_m")
            t_suc_A = st.number_input("Suction Temp (°C)", value=15.60, step=0.1, key="t_suc_A_m")
            p_dis_A = st.number_input(f"Discharge Press ({p_unit})", value=19.95, step=0.1, key="p_dis_A_m")
            t_dis_A = st.number_input("Discharge Temp (°C)", value=52.44, step=0.1, key="t_dis_A_m")
            t_cond_A = st.number_input("Condenser Temp (°C)", value=44.04, step=0.1, key="t_cond_A_m")

        with tab_b:
            p_suc_B = st.number_input(f"Suction Press ({p_unit})", value=6.50, step=0.1, key="p_suc_B")
            t_suc_B = st.number_input("Suction Temp (°C)", value=12.00, step=0.1, key="t_suc_B")
            p_dis_B = st.number_input(f"Discharge Press ({p_unit})", value=21.50, step=0.1, key="p_dis_B")
            t_dis_B = st.number_input("Discharge Temp (°C)", value=60.00, step=0.1, key="t_dis_B")
            t_cond_B = st.number_input("Condenser Temp (°C)", value=46.00, step=0.1, key="t_cond_B")

# --- Computations ---
try:
    CP.set_reference_state(fluid, 'IIR')
    
    # Calculate Profile A
    prof_A = calculate_cycle_points(
        p_suc_in=p_suc_A,
        t_suc_in=t_suc_A,
        p_dis_in=p_dis_A,
        t_dis_in=t_dis_A,
        t_cond_in=t_cond_A,
        p_unit=p_unit,
        fluid=fluid
    )
    
    # Calculate Profile B if in comparison mode
    prof_B = None
    if analysis_mode == "Compare Profiles":
        prof_B = calculate_cycle_points(
            p_suc_in=p_suc_B,
            t_suc_in=t_suc_B,
            p_dis_in=p_dis_B,
            t_dis_in=t_dis_B,
            t_cond_in=t_cond_B,
            p_unit=p_unit,
            fluid=fluid
        )

    sat_liq_h, sat_vap_h, sat_p = get_saturation_curve(fluid)

    # --- Plotly Figure ---
    fig = go.Figure()

    # Saturation Curves
    fig.add_trace(go.Scatter(
        x=sat_liq_h, y=sat_p, mode='lines', name='Liquid Saturation',
        line=dict(color='#0284C7', width=1.5, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=sat_vap_h, y=sat_p, mode='lines', name='Vapor Saturation',
        line=dict(color='#E11D48', width=1.5, dash='dash')
    ))

    # Helper function to plot profile loop
    def add_profile_trace(fig, prof, name, line_color, marker_color, dash_style='solid'):
        h_vals = [prof['s1_h'], prof['s2_h'], prof['s3_h'], prof['s4_h']]
        p_vals = [prof['p_suction'], prof['p_discharge'], prof['p_discharge'], prof['p_suction']]
        h_loop = h_vals + [h_vals[0]]
        p_loop = p_vals + [p_vals[0]]

        fig.add_trace(go.Scatter(
            x=h_loop, y=p_loop,
            mode='lines+markers',
            name=name,
            line=dict(color=line_color, width=2.5, dash=dash_style),
            marker=dict(size=8, color=marker_color, line=dict(color='#FFFFFF', width=1)),
            hovertemplate='Enthalpy: %{x:.1f} kJ/kg<br>Pressure: %{y:.2f} bara'
        ))

    # Add Profile A
    add_profile_trace(fig, prof_A, "Profile A", "#059669", "#10B981")

    # Add Profile B if comparing
    if prof_B:
        add_profile_trace(fig, prof_B, "Profile B", "#D97706", "#F59E0B", dash_style='dot')

    # Helper function to add annotated callouts with mirrored direction options
    def add_profile_annotations(fig, prof, prefix="", text_color="#10B981", mirror=False):
        # Direction multiplier: +1 for Profile A, -1 for Profile B (mirrored)
        m = -1 if mirror else 1

        # Point 1: Suction
        fig.add_annotation(
            x=prof['s1_h'], y=prof['p_suction'],
            text=f"<b>{prefix}S1 (Suction)</b><br>P: {prof['p_suction']:.2f} bara<br>T: {prof['t_suc']:.1f}°C<br>Superheat: {prof['s1_sh']:.1f} K",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=text_color,
            ax=55 * m, ay=55 * m,
            bordercolor=text_color, borderwidth=1, borderpad=4, bgcolor="#FFFFFF", opacity=0.9,
            font=dict(size=10, color="#111827")
        )
        # Point 2: Discharge
        fig.add_annotation(
            x=prof['s2_h'], y=prof['p_discharge'],
            text=f"<b>{prefix}S2 (Discharge)</b><br>P: {prof['p_discharge']:.2f} bara<br>T: {prof['t_dis']:.1f}°C<br>Superheat: {prof['s2_sh']:.1f} K",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=text_color,
            ax=55 * m, ay=-55 * m,
            bordercolor=text_color, borderwidth=1, borderpad=4, bgcolor="#FFFFFF", opacity=0.9,
            font=dict(size=10, color="#111827")
        )
        # Point 3: Condenser Outlet
        subcool = prof['s3_tsat'] - prof['t_cond']
        subcool_str = f"Subcooled: {subcool:.1f} K" if subcool > 0 else "Sat Liquid"
        fig.add_annotation(
            x=prof['s3_h'], y=prof['p_discharge'],
            text=f"<b>{prefix}S3 (Cond. Out)</b><br>P: {prof['p_discharge']:.2f} bara<br>T: {prof['t_cond']:.1f}°C<br>{subcool_str}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=text_color,
            ax=-55 * m, ay=-55 * m,
            bordercolor=text_color, borderwidth=1, borderpad=4, bgcolor="#FFFFFF", opacity=0.9,
            font=dict(size=10, color="#111827")
        )
        # Point 4: Expansion Outlet
        fig.add_annotation(
            x=prof['s4_h'], y=prof['p_suction'],
            text=f"<b>{prefix}S4 (Exp. Out)</b><br>P: {prof['p_suction']:.2f} bara<br>T_sat: {prof['s4_tsat']:.1f}°C",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=text_color,
            ax=-55 * m, ay=55 * m,
            bordercolor=text_color, borderwidth=1, borderpad=4, bgcolor="#FFFFFF", opacity=0.9,
            font=dict(size=10, color="#111827")
        )

    # Attach annotations if toggle is enabled
    if show_callouts:
        # Profile A standard offsets
        add_profile_annotations(fig, prof_A, prefix="A-" if prof_B else "", text_color="#059669", mirror=False)
        # Profile B mirrored offsets
        if prof_B:
            add_profile_annotations(fig, prof_B, prefix="B-", text_color="#D97706", mirror=True)

    # Axis Formatting
    all_h = sat_liq_h + [prof_A['s1_h'], prof_A['s2_h'], prof_A['s3_h'], prof_A['s4_h']]
    all_p = [prof_A['p_suction'], prof_A['p_discharge']]
    if prof_B:
        all_h += [prof_B['s1_h'], prof_B['s2_h'], prof_B['s3_h'], prof_B['s4_h']]
        all_p += [prof_B['p_suction'], prof_B['p_discharge']]

    fig.update_layout(
        title=dict(text=f'P-h Diagram ({refrigerant_choice}) - Mode: {analysis_mode}', font=dict(size=16)),
        xaxis=dict(title='Enthalpy (kJ/kg)', range=[max(0, min(all_h) - 100), max(all_h) + 140], showgrid=True, gridcolor='#E5E7EB'),
        yaxis=dict(title='Pressure (bara)', range=[0, max(all_p) + 8.0], showgrid=True, gridcolor='#E5E7EB'),
        plot_bgcolor='#FAFAFA', paper_bgcolor='#FFFFFF',
        legend=dict(orientation='h', y=1.05, x=1, xanchor='right'),
        height=620, margin=dict(l=60, r=40, t=60, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Summary Metrics Display ---
    st.markdown("### Thermodynamic Performance")
    
    if analysis_mode == "Single Profile":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("COP (Cooling)", f"{prof_A['cop']:.2f}")
        c2.metric("Refrigeration Effect", f"{prof_A['q_in']:.1f} kJ/kg")
        c3.metric("Compressor Work", f"{prof_A['w_in']:.1f} kJ/kg")
        c4.metric("Suction Superheat", f"{prof_A['s1_sh']:.1f} K")

        st.markdown("### State Points Summary (Profile A)")
        sc1, sc2, sc3, sc4 = st.columns(4)
        subcool_val_A = prof_A['s3_tsat'] - prof_A['t_cond']
        subcool_card_str = f"Subcooled: <b>{subcool_val_A:.1f} K</b>" if subcool_val_A > 0 else "Sat Liquid"
        
        with sc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">S1 - Suction</div>
                <div class="metric-value">{prof_A['s1_h']:.1f} <span style="font-size:12px;">kJ/kg</span></div>
                <div class="metric-sub">P: {prof_A['p_suction']:.2f} bara | T: {prof_A['t_suc']:.1f}°C</div>
                <div class="metric-sub">Superheat: <b>{prof_A['s1_sh']:.1f} K</b></div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">S2 - Discharge</div>
                <div class="metric-value">{prof_A['s2_h']:.1f} <span style="font-size:12px;">kJ/kg</span></div>
                <div class="metric-sub">P: {prof_A['p_discharge']:.2f} bara | T: {prof_A['t_dis']:.1f}°C</div>
                <div class="metric-sub">Superheat: <b>{prof_A['s2_sh']:.1f} K</b></div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">S3 - Condenser Outlet</div>
                <div class="metric-value">{prof_A['s3_h']:.1f} <span style="font-size:12px;">kJ/kg</span></div>
                <div class="metric-sub">P: {prof_A['p_discharge']:.2f} bara | T: {prof_A['t_cond']:.1f}°C</div>
                <div class="metric-sub">{subcool_card_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">S4 - Expansion</div>
                <div class="metric-value">{prof_A['s4_h']:.1f} <span style="font-size:12px;">kJ/kg</span></div>
                <div class="metric-sub">P: {prof_A['p_suction']:.2f} bara | Isenthalpic</div>
                <div class="metric-sub">Sat Temp: {prof_A['s4_tsat']:.1f}°C</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("#### Profile Comparison Metrics")
        m1, m2, m3, m4 = st.columns(4)
        
        cop_diff = prof_B['cop'] - prof_A['cop']
        q_diff = prof_B['q_in'] - prof_A['q_in']
        w_diff = prof_B['w_in'] - prof_A['w_in']

        m1.metric("COP (Cooling)", f"{prof_A['cop']:.2f}", delta=f"{cop_diff:+.2f} (Profile B)")
        m2.metric("Refrigeration Effect", f"{prof_A['q_in']:.1f} kJ/kg", delta=f"{q_diff:+.1f} kJ/kg")
        m3.metric("Compressor Work", f"{prof_A['w_in']:.1f} kJ/kg", delta=f"{w_diff:+.1f} kJ/kg")
        m4.metric("Discharge Pressure", f"{prof_A['p_discharge']:.2f} bara", delta=f"{prof_B['p_discharge'] - prof_A['p_discharge']:+.2f} bara")

except Exception as e:
    st.error(f"Calculation Error: Please review thermodynamic parameters. Details: {e}")
