import CoolProp.CoolProp as CP
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Refrigeration Cycle Analyzer", page_icon="❄️", layout="wide"
)

# Custom CSS for Light Theme Metric Cards & Subtitles
st.markdown(
    """
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
    .tsat-subtitle {
        font-size: 12px;
        color: #6B7280;
        margin-top: -12px;
        margin-bottom: 8px;
        font-weight: 400;
    }
</style>
""",
    unsafe_allow_html=True,
)

P_ATM_BAR = 1.01325  # Atmospheric pressure in bar


# --- Helper Functions ---
def convert_to_bara(pressure_val, unit):
    if unit == "barg":
        return pressure_val + P_ATM_BAR
    elif unit == "bara":
        return pressure_val
    elif unit == "kpag":
        return (pressure_val / 100.0) + P_ATM_BAR
    elif unit == "kpaa":
        return pressure_val / 100.0
    return pressure_val


def get_saturation_curve(fluid, num_points=120):
    try:
        p_crit = CP.PropsSI("Pcrit", fluid) / 1e5
        p_min = CP.PropsSI("P_min", fluid) / 1e5
        p_start = max(p_min, 0.1)
        p_end = p_crit * 0.98
        pressures = np.geomspace(p_start, p_end, num_points)

        sat_liq_h, sat_vap_h, valid_p = [], [], []
        for p in pressures:
            p_pa = p * 1e5
            try:
                h_l = CP.PropsSI("H", "P", p_pa, "Q", 0, fluid) / 1000.0
                h_v = CP.PropsSI("H", "P", p_pa, "Q", 1, fluid) / 1000.0
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

    # Enthalpy (kJ/kg)
    h_kj = CP.PropsSI("H", "P", p_pa, "T", t_k, fluid) / 1000.0

    # Heat Capacities (kJ/kg·K)
    cp_kj = CP.PropsSI("Cpmass", "P", p_pa, "T", t_k, fluid) / 1000.0
    cv_kj = CP.PropsSI("Cvmass", "P", p_pa, "T", t_k, fluid) / 1000.0

    # Saturation Temperature (°C)
    t_sat_k = CP.PropsSI("T", "P", p_pa, "Q", 0, fluid)
    t_sat_c = t_sat_k - 273.15

    return h_kj, cp_kj, cv_kj, t_sat_c


def calc_tsat_ui(p_val, unit, fl):
    """Calculates saturation temperature in °C directly for input subtitles."""
    try:
        p_bara = convert_to_bara(p_val, unit)
        p_pa = p_bara * 1e5
        t_sat_k = CP.PropsSI("T", "P", p_pa, "Q", 0, fl)
        return f"Sat Temp: {t_sat_k - 273.15:.2f} °C"
    except Exception:
        return "Sat Temp: N/A"


def calculate_cycle_points(
    p_suc_in,
    t_suc_in,
    p_dis_in,
    t_dis_in,
    t_cond_in,
    p_unit,
    fluid,
    duty_mode="Not Specified",
    duty_kw_val=0.0,
    flowrate_kghr_val=0.0,
):
    p_suction = convert_to_bara(p_suc_in, p_unit)
    p_discharge = convert_to_bara(p_dis_in, p_unit)

    # State 1 (Suction)
    s1_h, s1_cp, s1_cv, s1_tsat = get_point_props(p_suction, t_suc_in, fluid)
    s1_sh = t_suc_in - s1_tsat

    # State 2 (Discharge)
    s2_h, s2_cp, s2_cv, s2_tsat = get_point_props(p_discharge, t_dis_in, fluid)
    s2_sh = t_dis_in - s2_tsat

    # State 3 (Condenser Outlet)
    s3_h, s3_cp, s3_cv, s3_tsat = get_point_props(
        p_discharge, t_cond_in, fluid
    )

    # State 4 (Expansion Valve Outlet)
    s4_h = s3_h
    s4_tsat = CP.PropsSI("T", "P", p_suction * 1e5, "Q", 0, fluid) - 273.15

    q_in = s1_h - s4_h
    w_in = s2_h - s1_h
    cop = q_in / w_in if w_in != 0 else 0

    # Duty & Flowrate Calculation
    evap_duty_kw = None
    m_flow_kghr = None

    if duty_mode == "Specify kW":
        evap_duty_kw = duty_kw_val
        if q_in > 0:
            m_flow_kghr = (evap_duty_kw * 3600.0) / q_in
    elif duty_mode == "Specify Refrigeration Flowrate":
        m_flow_kghr = flowrate_kghr_val
        evap_duty_kw = (m_flow_kghr * q_in) / 3600.0

    # Isentropic Efficiency calculation using S2 Cp/Cv
    t1_k = t_suc_in + 273.15
    t2_k = t_dis_in + 273.15
    k_s2 = s2_cp / s2_cv

    if (t2_k - t1_k) != 0 and k_s2 > 0:
        eta_isen = (
            (t1_k / (t2_k - t1_k))
            * (((p_discharge / p_suction) ** ((k_s2 - 1.0) / k_s2)) - 1.0)
            * 100.0
        )
    else:
        eta_isen = 0.0

    # Discharge Superheat Heat Load (kJ/kg) & Power (kW)
    h_s2_sat_vap = (
        CP.PropsSI("H", "P", p_discharge * 1e5, "Q", 1, fluid) / 1000.0
    )
    dis_sh_energy = s2_h - h_s2_sat_vap
    w_comp_energy = w_in

    dis_sh_kw = None
    w_comp_kw = None
    if m_flow_kghr is not None and m_flow_kghr > 0:
        dis_sh_kw = (dis_sh_energy * m_flow_kghr) / 3600.0
        w_comp_kw = (w_comp_energy * m_flow_kghr) / 3600.0

    # Compression Ratio
    cr = p_discharge / p_suction

    return {
        "p_suction": p_suction,
        "p_discharge": p_discharge,
        "s1_h": s1_h,
        "s1_cp": s1_cp,
        "s1_cv": s1_cv,
        "s1_sh": s1_sh,
        "t_suc": t_suc_in,
        "s1_tsat": s1_tsat,
        "s2_h": s2_h,
        "s2_cp": s2_cp,
        "s2_cv": s2_cv,
        "s2_sh": s2_sh,
        "t_dis": t_dis_in,
        "s2_tsat": s2_tsat,
        "s3_h": s3_h,
        "s3_cp": s3_cp,
        "s3_cv": s3_cv,
        "s3_tsat": s3_tsat,
        "t_cond": t_cond_in,
        "s4_h": s4_h,
        "s4_tsat": s4_tsat,
        "cop": cop,
        "q_in": q_in,
        "w_in": w_in,
        "eta_isen": eta_isen,
        "dis_sh_energy": dis_sh_energy,
        "w_comp_energy": w_comp_energy,
        "dis_sh_kw": dis_sh_kw,
        "w_comp_kw": w_comp_kw,
        "cr": cr,
        "evap_duty_kw": evap_duty_kw,
        "m_flow_kghr": m_flow_kghr,
    }


# --- Header Section ---
st.title("Refrigeration Cycle Analyzer")
st.caption(
    "Developed by Iqbal SHERPA 20260810. Contact me for further information"
    " @iqbalshafiq96@gmail.com"
)
st.caption(
    "Interactive Pressure-Enthalpy ($P-h$) Diagram & Thermodynamic Calculations"
)

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Analysis Mode")
    analysis_mode = st.radio(
        "Select Operating Mode",
        ["Single Profile", "Compare Profiles"],
        index=1,
        help="Choose whether to analyze one profile or overlay two profiles for comparison.",
    )

    show_callouts = st.checkbox(
        "Show Graph Callouts",
        value=True,
        help="Toggle on-graph annotations for state points.",
    )

    st.markdown("---")
    st.header("2. Fluid & Units")
    refrigerant_choice = st.selectbox(
        "Refrigerant", ["Ammonia", "R134a"], index=0
    )
    fluid = refrigerant_choice
    p_unit = st.selectbox(
        "Pressure Unit", ["barg", "bara", "kpag", "kpaa"], index=1
    )

    energy_unit = "kJ/kg"

    st.markdown("---")

    duty_options = [
        "Not Specified",
        "Specify kW",
        "Specify Refrigeration Flowrate",
    ]

    if analysis_mode == "Single Profile":
        st.subheader("Process Parameters")
        p_suc_A = st.number_input(
            f"Suction Pressure ({p_unit})", value=7.41, step=0.1, key="p_suc_A"
        )

        t_suc_A = st.number_input(
            "Suction Temp (°C)", value=15.60, step=0.1, key="t_suc_A"
        )
        t_sat_suc_A_str = calc_tsat_ui(p_suc_A, p_unit, fluid)
        st.markdown(
            f'<div class="tsat-subtitle">{t_sat_suc_A_str}</div>',
            unsafe_allow_html=True,
        )

        p_dis_A = st.number_input(
            f"Discharge Pressure ({p_unit})", value=19.95, step=0.1, key="p_dis_A"
        )

        t_dis_A = st.number_input(
            "Discharge Temp (°C)", value=52.44, step=0.1, key="t_dis_A"
        )
        t_sat_dis_A_str = calc_tsat_ui(p_dis_A, p_unit, fluid)
        st.markdown(
            f'<div class="tsat-subtitle">{t_sat_dis_A_str}</div>',
            unsafe_allow_html=True,
        )

        t_cond_A = st.number_input(
            "Condenser Outlet Temp (°C)", value=44.04, step=0.1, key="t_cond_A"
        )

        st.markdown("**Evaporator Duty Specification**")
        duty_mode_A = st.selectbox(
            "Evaporator Duty Mode", duty_options, index=0, key="duty_mode_A"
        )
        duty_kw_A = 0.0
        flowrate_kghr_A = 0.0
        if duty_mode_A == "Specify kW":
            duty_kw_A = st.number_input(
                "Evaporator Duty (kW)", value=500.0, step=10.0, key="duty_kw_A"
            )
        elif duty_mode_A == "Specify Refrigeration Flowrate":
            flowrate_kghr_A = st.number_input(
                "Flowrate (kg/hr)", value=1500.0, step=50.0, key="flowrate_kghr_A"
            )

    else:
        tab_a, tab_b = st.tabs(["Profile A (Primary)", "Profile B (Compare)"])
        with tab_a:
            p_suc_A = st.number_input(
                f"Suction Press ({p_unit})", value=7.41, step=0.1, key="p_suc_A_m"
            )

            t_suc_A = st.number_input(
                "Suction Temp (°C)", value=15.60, step=0.1, key="t_suc_A_m"
            )
            t_sat_suc_A_str = calc_tsat_ui(p_suc_A, p_unit, fluid)
            st.markdown(
                f'<div class="tsat-subtitle">{t_sat_suc_A_str}</div>',
                unsafe_allow_html=True,
            )

            p_dis_A = st.number_input(
                f"Discharge Press ({p_unit})", value=19.95, step=0.1, key="p_dis_A_m"
            )

            t_dis_A = st.number_input(
                "Discharge Temp (°C)", value=52.44, step=0.1, key="t_dis_A_m"
            )
            t_sat_dis_A_str = calc_tsat_ui(p_dis_A, p_unit, fluid)
            st.markdown(
                f'<div class="tsat-subtitle">{t_sat_dis_A_str}</div>',
                unsafe_allow_html=True,
            )

            t_cond_A = st.number_input(
                "Condenser Temp (°C)", value=44.04, step=0.1, key="t_cond_A_m"
            )

            st.markdown("**Evaporator Duty Specification**")
            duty_mode_A = st.selectbox(
                "Evaporator Duty Mode", duty_options, index=0, key="duty_mode_A_m"
            )
            duty_kw_A = 0.0
            flowrate_kghr_A = 0.0
            if duty_mode_A == "Specify kW":
                duty_kw_A = st.number_input(
                    "Evaporator Duty (kW)", value=500.0, step=10.0, key="duty_kw_A_m"
                )
            elif duty_mode_A == "Specify Refrigeration Flowrate":
                flowrate_kghr_A = st.number_input(
                    "Flowrate (kg/hr)", value=1500.0, step=50.0, key="flowrate_kghr_A_m"
                )

        with tab_b:
            p_suc_B = st.number_input(
                f"Suction Press ({p_unit})", value=6.50, step=0.1, key="p_suc_B"
            )

            t_suc_B = st.number_input(
                "Suction Temp (°C)", value=12.00, step=0.1, key="t_suc_B"
            )
            t_sat_suc_B_str = calc_tsat_ui(p_suc_B, p_unit, fluid)
            st.markdown(
                f'<div class="tsat-subtitle">{t_sat_suc_B_str}</div>',
                unsafe_allow_html=True,
            )

            p_dis_B = st.number_input(
                f"Discharge Press ({p_unit})", value=21.50, step=0.1, key="p_dis_B"
            )

            t_dis_B = st.number_input(
                "Discharge Temp (°C)", value=60.00, step=0.1, key="t_dis_B"
            )
            t_sat_dis_B_str = calc_tsat_ui(p_dis_B, p_unit, fluid)
            st.markdown(
                f'<div class="tsat-subtitle">{t_sat_dis_B_str}</div>',
                unsafe_allow_html=True,
            )

            t_cond_B = st.number_input(
                "Condenser Temp (°C)", value=46.00, step=0.1, key="t_cond_B"
            )

            st.markdown("**Evaporator Duty Specification**")
            duty_mode_B = st.selectbox(
                "Evaporator Duty Mode", duty_options, index=0, key="duty_mode_B"
            )
            duty_kw_B = 0.0
            flowrate_kghr_B = 0.0
            if duty_mode_B == "Specify kW":
                duty_kw_B = st.number_input(
                    "Evaporator Duty (kW)", value=500.0, step=10.0, key="duty_kw_B"
                )
            elif duty_mode_B == "Specify Refrigeration Flowrate":
                flowrate_kghr_B = st.number_input(
                    "Flowrate (kg/hr)", value=1500.0, step=50.0, key="flowrate_kghr_B"
                )

# --- Computations ---
try:
    CP.set_reference_state(fluid, "IIR")

    # Calculate Profile A
    prof_A = calculate_cycle_points(
        p_suc_in=p_suc_A,
        t_suc_in=t_suc_A,
        p_dis_in=p_dis_A,
        t_dis_in=t_dis_A,
        t_cond_in=t_cond_A,
        p_unit=p_unit,
        fluid=fluid,
        duty_mode=duty_mode_A,
        duty_kw_val=duty_kw_A,
        flowrate_kghr_val=flowrate_kghr_A,
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
            fluid=fluid,
            duty_mode=duty_mode_B,
            duty_kw_val=duty_kw_B,
            flowrate_kghr_val=flowrate_kghr_B,
        )

    sat_liq_h, sat_vap_h, sat_p = get_saturation_curve(fluid)

    # --- Plotly Figure ---
    fig = go.Figure()

    # Saturation Curves
    fig.add_trace(
        go.Scatter(
            x=sat_liq_h,
            y=sat_p,
            mode="lines",
            name="Liquid Saturation",
            line=dict(color="#0284C7", width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sat_vap_h,
            y=sat_p,
            mode="lines",
            name="Vapor Saturation",
            line=dict(color="#E11D48", width=1.5, dash="dash"),
        )
    )

    # Helper function to plot profile loop
    def add_profile_trace(
        fig, prof, name, line_color, marker_color, dash_style="solid"
    ):
        h_vals = [prof["s1_h"], prof["s2_h"], prof["s3_h"], prof["s4_h"]]
        p_vals = [
            prof["p_suction"],
            prof["p_discharge"],
            prof["p_discharge"],
            prof["p_suction"],
        ]
        h_loop = h_vals + [h_vals[0]]
        p_loop = p_vals + [p_vals[0]]

        fig.add_trace(
            go.Scatter(
                x=h_loop,
                y=p_loop,
                mode="lines+markers",
                name=name,
                line=dict(color=line_color, width=2.5, dash=dash_style),
                marker=dict(
                    size=8, color=marker_color, line=dict(color="#FFFFFF", width=1)
                ),
                hovertemplate=(
                    "Enthalpy: %{x:.1f} kJ/kg<br>Pressure: %{y:.2f} bara"
                ),
            )
        )

    # Add Profile A
    add_profile_trace(fig, prof_A, "Profile A", "#059669", "#10B981")

    # Add Profile B if comparing
    if prof_B:
        add_profile_trace(
            fig,
            prof_B,
            "Profile B",
            "#D97706",
            "#F59E0B",
            dash_style="dot",
        )

    # Helper function to add annotated callouts
    def add_profile_annotations(
        fig, prof, prefix="", text_color="#10B981", vert_mirror=False
    ):
        x_m = -1 if vert_mirror else 1

        fig.add_annotation(
            x=prof["s1_h"],
            y=prof["p_suction"],
            text=(
                f"<b>{prefix}S1 (Suction)</b><br>P: {prof['p_suction']:.2f}"
                f" bara<br>T: {prof['t_suc']:.1f}°C<br>Superheat:"
                f" {prof['s1_sh']:.1f} K<br>Cp: {prof['s1_cp']:.1f}"
                f" kJ/kg·K<br>Cv: {prof['s1_cv']:.1f} kJ/kg·K"
            ),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=text_color,
            ax=60 * x_m,
            ay=55,
            bordercolor=text_color,
            borderwidth=1,
            borderpad=4,
            bgcolor="#FFFFFF",
            opacity=0.9,
            font=dict(size=10, color="#111827"),
        )
        fig.add_annotation(
            x=prof["s2_h"],
            y=prof["p_discharge"],
            text=(
                f"<b>{prefix}S2 (Discharge)</b><br>P: {prof['p_discharge']:.2f}"
                f" bara<br>T: {prof['t_dis']:.1f}°C<br>Superheat:"
                f" {prof['s2_sh']:.1f} K<br>Cp: {prof['s2_cp']:.1f}"
                f" kJ/kg·K<br>Cv: {prof['s2_cv']:.1f} kJ/kg·K"
            ),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=text_color,
            ax=60 * x_m,
            ay=-55,
            bordercolor=text_color,
            borderwidth=1,
            borderpad=4,
            bgcolor="#FFFFFF",
            opacity=0.9,
            font=dict(size=10, color="#111827"),
        )
        subcool = prof["s3_tsat"] - prof["t_cond"]
        subcool_str = (
            f"Subcooled: {subcool:.1f} K" if subcool > 0 else "Sat Liquid"
        )
        fig.add_annotation(
            x=prof["s3_h"],
            y=prof["p_discharge"],
            text=(
                f"<b>{prefix}S3 (Cond. Out)</b><br>P: {prof['p_discharge']:.2f}"
                f" bara<br>T: {prof['t_cond']:.1f}°C<br>{subcool_str}"
            ),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=text_color,
            ax=-60 * x_m,
            ay=-55,
            bordercolor=text_color,
            borderwidth=1,
            borderpad=4,
            bgcolor="#FFFFFF",
            opacity=0.9,
            font=dict(size=10, color="#111827"),
        )
        fig.add_annotation(
            x=prof["s4_h"],
            y=prof["p_suction"],
            text=(
                f"<b>{prefix}S4 (Exp. Out)</b><br>P: {prof['p_suction']:.2f}"
                f" bara<br>T_sat: {prof['s4_tsat']:.1f}°C"
            ),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=text_color,
            ax=-60 * x_m,
            ay=55,
            bordercolor=text_color,
            borderwidth=1,
            borderpad=4,
            bgcolor="#FFFFFF",
            opacity=0.9,
            font=dict(size=10, color="#111827"),
        )

    if show_callouts:
        add_profile_annotations(
            fig,
            prof_A,
            prefix="A-" if prof_B else "",
            text_color="#059669",
            vert_mirror=False,
        )
        if prof_B:
            add_profile_annotations(
                fig,
                prof_B,
                prefix="B-",
                text_color="#D97706",
                vert_mirror=True,
            )

    all_h = sat_liq_h + [
        prof_A["s1_h"],
        prof_A["s2_h"],
        prof_A["s3_h"],
        prof_A["s4_h"],
    ]
    all_p = [prof_A["p_suction"], prof_A["p_discharge"]]
    if prof_B:
        all_h += [
            prof_B["s1_h"],
            prof_B["s2_h"],
            prof_B["s3_h"],
            prof_B["s4_h"],
        ]
        all_p += [prof_B["p_suction"], prof_B["p_discharge"]]

    fig.update_layout(
        title=dict(
            text=f"P-h Diagram ({refrigerant_choice}) - Mode: {analysis_mode}",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Enthalpy (kJ/kg)",
            range=[max(0, min(all_h) - 100), max(all_h) + 160],
            showgrid=True,
            gridcolor="#E5E7EB",
        ),
        yaxis=dict(
            title="Pressure (bara)",
            range=[0, max(all_p) + 8.0],
            showgrid=True,
            gridcolor="#E5E7EB",
        ),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        height=620,
        margin=dict(l=60, r=40, t=60, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Summary Metrics Display ---
    st.markdown("### Thermodynamic Performance")

    if analysis_mode == "Single Profile":
        dis_sh_lbl = (
            f"Discharge SH ({prof_A['dis_sh_kw']:.1f} kW)"
            if prof_A["dis_sh_kw"] is not None
            else f"Discharge SH ({energy_unit})"
        )
        w_comp_lbl = (
            f"Comp Work ({prof_A['w_comp_kw']:.1f} kW)"
            if prof_A["w_comp_kw"] is not None
            else f"Comp Work ({energy_unit})"
        )

        r1_c1, r1_c2, r1_c3, r1_c4, r1_c5, r1_c6 = st.columns(6)
        r1_c1.metric("Isentropic Efficiency", f"{prof_A['eta_isen']:.1f} %")
        r1_c2.metric("Suction Superheat", f"{prof_A['s1_sh']:.1f} K")
        r1_c3.metric("Discharge Superheat", f"{prof_A['s2_sh']:.1f} K")
        r1_c4.metric(
            dis_sh_lbl, f"{prof_A['dis_sh_energy']:.1f} {energy_unit}"
        )
        r1_c5.metric(
            w_comp_lbl, f"{prof_A['w_comp_energy']:.1f} {energy_unit}"
        )
        r1_c6.metric("Compression Ratio", f"{prof_A['cr']:.2f}")

        duty_str_A = (
            f"{prof_A['evap_duty_kw']:.1f} kW"
            if prof_A["evap_duty_kw"] is not None
            else "N/A"
        )
        flow_str_A = (
            f"{prof_A['m_flow_kghr']:.1f} kg/hr"
            if prof_A["m_flow_kghr"] is not None
            else "N/A"
        )

        r2_c1, r2_c2, _, _, _, _ = st.columns(6)
        r2_c1.metric("Evaporator Duty", duty_str_A)
        r2_c2.metric("Refrig. Flowrate", flow_str_A)

    else:
        st.markdown("#### Profile Comparison Metrics")
        st.caption(
            "Displaying **Profile B** baseline values; deltas in brackets show"
            " shift relative to **Profile A**."
        )

        eta_diff = prof_B["eta_isen"] - prof_A["eta_isen"]
        suc_sh_diff = prof_B["s1_sh"] - prof_A["s1_sh"]
        dis_sh_diff = prof_B["s2_sh"] - prof_A["s2_sh"]
        dis_sh_energy_diff = prof_B["dis_sh_energy"] - prof_A["dis_sh_energy"]
        w_comp_diff = prof_B["w_comp_energy"] - prof_A["w_comp_energy"]
        cr_diff = prof_B["cr"] - prof_A["cr"]

        r1_m1, r1_m2, r1_m3, r1_m4, r1_m5, r1_m6 = st.columns(6)

        r1_m1.metric(
            "Isentropic Efficiency",
            f"{prof_B['eta_isen']:.1f} %",
            delta=f"{eta_diff:+.1f} %",
            delta_color="normal",
        )
        r1_m2.metric(
            "Suction Superheat",
            f"{prof_B['s1_sh']:.1f} K",
            delta=f"{suc_sh_diff:+.1f} K",
            delta_color="inverse",
        )
        r1_m3.metric(
            "Discharge Superheat",
            f"{prof_B['s2_sh']:.1f} K",
            delta=f"{dis_sh_diff:+.1f} K",
            delta_color="inverse",
        )

        # 4. Discharge Superheat (kW if available, else kJ/kg)
        if (
            prof_B["dis_sh_kw"] is not None
            and prof_A["dis_sh_kw"] is not None
        ):
            sh_kw_diff = prof_B["dis_sh_kw"] - prof_A["dis_sh_kw"]
            r1_m4.metric(
                "Discharge Superheat (kW)",
                f"{prof_B['dis_sh_kw']:.1f} kW",
                delta=f"{sh_kw_diff:+.1f} kW",
                delta_color="inverse",
            )
        else:
            r1_m4.metric(
                f"Discharge SH ({energy_unit})",
                f"{prof_B['dis_sh_energy']:.1f} {energy_unit}",
                delta=f"{dis_sh_energy_diff:+.1f} {energy_unit}",
                delta_color="inverse",
            )

        # 5. Compressor Work (kW if available, else kJ/kg)
        if (
            prof_B["w_comp_kw"] is not None
            and prof_A["w_comp_kw"] is not None
        ):
            w_kw_diff = prof_B["w_comp_kw"] - prof_A["w_comp_kw"]
            r1_m5.metric(
                "Compressor Work (kW)",
                f"{prof_B['w_comp_kw']:.1f} kW",
                delta=f"{w_kw_diff:+.1f} kW",
                delta_color="inverse",
            )
        else:
            r1_m5.metric(
                f"Compressor Work ({energy_unit})",
                f"{prof_B['w_comp_energy']:.1f} {energy_unit}",
                delta=f"{w_comp_diff:+.1f} {energy_unit}",
                delta_color="inverse",
            )

        # 6. Compression Ratio
        r1_m6.metric(
            "Compression Ratio",
            f"{prof_B['cr']:.2f}",
            delta=f"{cr_diff:+.2f}",
            delta_color="off",
        )

        # --- Row 2 (Remaining 2 KPIs) ---
        r2_m1, r2_m2, _, _, _, _ = st.columns(6)

        # 7. Evaporator Duty
        if (
            prof_B["evap_duty_kw"] is not None
            and prof_A["evap_duty_kw"] is not None
        ):
            duty_diff = prof_B["evap_duty_kw"] - prof_A["evap_duty_kw"]
            r2_m1.metric(
                "Evaporator Duty",
                f"{prof_B['evap_duty_kw']:.1f} kW",
                delta=f"{duty_diff:+.1f} kW",
                delta_color="normal",
            )
        else:
            val_b = (
                f"{prof_B['evap_duty_kw']:.1f} kW"
                if prof_B["evap_duty_kw"] is not None
                else "N/A"
            )
            r2_m1.metric("Evaporator Duty", val_b, delta=None)

        # 8. Refrigerant Flowrate
        if (
            prof_B["m_flow_kghr"] is not None
            and prof_A["m_flow_kghr"] is not None
        ):
            flow_diff = prof_B["m_flow_kghr"] - prof_A["m_flow_kghr"]
            r2_m2.metric(
                "Refrig. Flowrate",
                f"{prof_B['m_flow_kghr']:.1f} kg/hr",
                delta=f"{flow_diff:+.1f} kg/hr",
                delta_color="inverse",
            )
        else:
            val_b_flow = (
                f"{prof_B['m_flow_kghr']:.1f} kg/hr"
                if prof_B["m_flow_kghr"] is not None
                else "N/A"
            )
            r2_m2.metric("Refrig. Flowrate", val_b_flow, delta=None)

except Exception as e:
    st.error(
        "Calculation Error: Please review thermodynamic parameters. Details:"
        f" {e}"
    )
