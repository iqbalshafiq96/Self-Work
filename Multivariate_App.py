import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f, norm
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Multivariate PCA Analysis", layout="wide")
st.title("Multivariate Statistical Process Control (MSPC)")
st.caption(
    "Developed by Iqbal SHERPA 20260824. Contact me for further information"
    " @iqbalshafiq96@gmail.com"
)

# Case Study Reference Image & Data URLs
IMAGE_URL = "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case.png"

# ---------------------------------------------------------
# PHASE 1 - SELECTABLE BASELINE (NOC) DATASETS
# ---------------------------------------------------------
# Users can pick which Normal Operating Condition (NOC) dataset is used to
# build the Phase 1 baseline model. Everything downstream (normalization,
# correlation, eigen decomposition, PC scores, Hotelling T2 / SPE limits,
# and consequently ALL of Phase 2 online monitoring) is recalculated from
# whichever dataset is selected here.
NOC_DATASET_URLS = {
    "NOC6_1 (Default)": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv",
    "NOC_Z3700": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Z3700.csv",
}

# Dataset Map for Phase 2 Cases
CASE_URLS = {
    "Case 0": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_0.csv",
    "Case 1": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_1.csv",
    "Case 2": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_2.csv",
    "Case 3": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_3.csv",
    "Case 10": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_10.csv",
}

@st.cache_data
def load_data(url):
    """Loads CSV data from GitHub, auto-converting URL, handling BOM, and detecting delimiter."""
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    
    df = pd.read_csv(url, sep=None, engine='python', encoding='utf-8-sig')
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '')
    return df

# ---------------------------------------------------------
# WORKFLOW NAVIGATION & STATE INITIALIZATION
# ---------------------------------------------------------
if "phase_selection" not in st.session_state:
    st.session_state.phase_selection = "Phase 1: Offline Modelling and Monitoring Setup"

if "noc_dataset_selection" not in st.session_state:
    st.session_state.noc_dataset_selection = "NOC6_1 (Default)"

p1_active = st.session_state.phase_selection.startswith("Phase 1")
p2_active = st.session_state.phase_selection.startswith("Phase 2")

col_btn1, col_btn2, _ = st.columns([1, 1, 2])

with col_btn1:
    if st.button(
        "📊 Phase 1: Offline Setup", 
        type="primary" if p1_active else "secondary", 
        use_container_width=True
    ):
        st.session_state.phase_selection = "Phase 1: Offline Modelling and Monitoring Setup"
        st.rerun()

with col_btn2:
    if st.button(
        "🚨 Phase 2: Online Monitoring", 
        type="primary" if p2_active else "secondary", 
        use_container_width=True
    ):
        st.session_state.phase_selection = "Phase 2: Online Monitoring and Fault Detection"
        st.rerun()

# Gray horizontal bar
st.markdown("---")

# Header positioned below horizontal bar & above navigation tabs
if p1_active:
    st.header("Phase 1: Offline Modelling and Monitoring Setup")
elif p2_active:
    st.header("Phase 2: Real-time Online Fault Detection")

# ---------------------------------------------------------
# BASELINE (NOC) DATASET SELECTOR
# ---------------------------------------------------------
# Placed above the shared core calculation block (and visible regardless of
# phase) so that switching it recomputes the Phase 1 baseline model AND
# propagates through to Phase 2 online monitoring automatically.
sel_col1, sel_col2 = st.columns([1.4, 2.6])
with sel_col1:
    st.selectbox(
        "📂 Phase 1 Baseline (NOC) Dataset:",
        options=list(NOC_DATASET_URLS.keys()),
        key="noc_dataset_selection",
        help="Choose which Normal Operating Condition dataset is used to build the Phase 1 baseline model. "
             "Switching this recalculates the correlation matrix, eigen decomposition, PC scores, "
             "Hotelling T² / SPE limits for Phase 1, and all Phase 2 online monitoring results."
    )
with sel_col2:
    st.caption(
        f"Currently active baseline: **{st.session_state.noc_dataset_selection}**  \n"
        f"Source: `{NOC_DATASET_URLS[st.session_state.noc_dataset_selection]}`"
    )

st.markdown("---")

phase_selection = st.session_state.phase_selection
selected_noc_label = st.session_state.noc_dataset_selection
selected_noc_url = NOC_DATASET_URLS[selected_noc_label]

try:
    # ---------------------------------------------------------
    # SHARED CORE CALCULATION (Phase 1 Baseline Model Setup)
    # ---------------------------------------------------------
    raw_df = load_data(selected_noc_url)
    raw_df.index = raw_df.index + 1
    numeric_df = raw_df.select_dtypes(include=[np.number]).dropna()

    # Fit Scaler on Phase 1 Base Model
    scaler = StandardScaler()
    normalized_array = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(normalized_array, columns=numeric_df.columns, index=numeric_df.index)

    # Correlation Matrix
    corr_matrix = normalized_df.corr()
    mask_strictly_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    lower_corr = corr_matrix.copy()
    lower_corr[mask_strictly_upper] = np.nan

    # Eigen Decomposition
    raw_eigenvalues, raw_eigenvectors = np.linalg.eig(corr_matrix.values)
    eigenvalues = np.real(raw_eigenvalues)
    eigenvectors = np.real(raw_eigenvectors)
    
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    pc_names = [f"PC{i+1}" for i in range(len(sorted_eigenvalues))]
    eigenvector_df = pd.DataFrame(sorted_eigenvectors, index=numeric_df.columns, columns=pc_names)

    diag_eigen_matrix = np.full((len(sorted_eigenvalues), len(sorted_eigenvalues)), np.nan)
    np.fill_diagonal(diag_eigen_matrix, sorted_eigenvalues)
    eigenvalue_matrix_df = pd.DataFrame(diag_eigen_matrix, index=pc_names, columns=pc_names)

    # Threshold Selection (< 80%)
    var_explained = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cum_var_explained = np.cumsum(var_explained)

    num_components_selected = int(np.sum(cum_var_explained < 0.80))
    selected_pc_names = pc_names[:num_components_selected]
    selected_eigenvectors = sorted_eigenvectors[:, :num_components_selected]
    selected_eigenvalues = sorted_eigenvalues[:num_components_selected]
    selected_eigenvector_df = eigenvector_df[selected_pc_names]

    bar_colors = ['#EF553B' if val < 0.80 else '#636EFA' for val in cum_var_explained]

    # Phase 1 Scores, Hotelling T2, and SPE Residual Calculations
    if num_components_selected > 0:
        pc_scores_array = np.dot(normalized_df.values, selected_eigenvector_df.values)
        pc_scores_df = pd.DataFrame(pc_scores_array, columns=selected_pc_names, index=normalized_df.index)
        
        # Hotelling T2
        t2_components = (pc_scores_array ** 2) / selected_eigenvalues
        t2_scores = np.sum(t2_components, axis=1)
        
        t2_summary_df = pc_scores_df.copy()
        t2_summary_df['Hotelling_T2'] = t2_scores
        
        n = len(normalized_df)
        A = num_components_selected
        df1 = A
        df2 = n - A
        
        if n > A:
            multiplier = (A * (n - 1)) / (n - A)
            f_val_05 = f.ppf(1 - 0.05, df1, df2)
            f_val_01 = f.ppf(1 - 0.01, df1, df2)
            
            t2_warning_limit = multiplier * f_val_05
            t2_control_limit = multiplier * f_val_01
        else:
            multiplier = f_val_05 = f_val_01 = np.nan
            t2_warning_limit = t2_control_limit = np.nan

        # SPE Calculation: PVt = PC_scores * Transposed_Selected_Eigenvectors
        pvt_array = np.dot(pc_scores_array, selected_eigenvector_df.values.T)
        pvt_df = pd.DataFrame(pvt_array, columns=normalized_df.columns, index=normalized_df.index)

        # Residual E Matrix: E = Normalized Data - PVt
        e_matrix_df = normalized_df - pvt_df

        # SPE is sum of squares of each row in E matrix
        spe_scores = np.sum(e_matrix_df.values ** 2, axis=1)
        spe_summary_df = pd.DataFrame({'SPE': spe_scores}, index=normalized_df.index)

        # ---------------------------------------------------------
        # SPE THRESHOLD (Jackson & Mudholkar Approximation via Z-Table)
        # ---------------------------------------------------------
        unselected_eigenvalues = sorted_eigenvalues[num_components_selected:]

        if len(unselected_eigenvalues) > 0:
            theta1 = np.sum(unselected_eigenvalues ** 1)
            theta2 = np.sum(unselected_eigenvalues ** 2)
            theta3 = np.sum(unselected_eigenvalues ** 3)

            if theta1 > 0 and theta2 > 0:
                h0 = 1 - (2 * theta1 * theta3) / (3 * (theta2 ** 2))
                
                # Z-values from Standard Normal Distribution
                z_val_05 = norm.ppf(1 - 0.05)  # 1.6449 (95%)
                z_val_01 = norm.ppf(1 - 0.01)  # 2.3263 (99%)

                def calc_spe_limit(z_score):
                    term1 = (z_score * np.sqrt(2 * theta2 * (h0 ** 2))) / theta1
                    term2 = (theta2 * h0 * (h0 - 1)) / (theta1 ** 2)
                    return theta1 * ((1 + term1 + term2) ** (1 / h0))

                spe_warning_limit = calc_spe_limit(z_val_05)
                spe_control_limit = calc_spe_limit(z_val_01)
            else:
                theta1 = theta2 = theta3 = h0 = z_val_05 = z_val_01 = np.nan
                spe_warning_limit = spe_control_limit = np.nan
        else:
            theta1 = theta2 = theta3 = h0 = z_val_05 = z_val_01 = np.nan
            spe_warning_limit = spe_control_limit = 0.0

    else:
        pc_scores_df = pd.DataFrame()
        t2_summary_df = pd.DataFrame()
        e_matrix_df = pd.DataFrame()
        pvt_df = pd.DataFrame()
        spe_summary_df = pd.DataFrame()
        t2_warning_limit = t2_control_limit = np.nan
        spe_warning_limit = spe_control_limit = np.nan
        theta1 = theta2 = theta3 = h0 = z_val_05 = z_val_01 = np.nan
        df1 = df2 = n = A = multiplier = 0
        f_val_05 = f_val_01 = np.nan

    # =========================================================
    # PHASE 1: OFFLINE MODELLING AND MONITORING SETUP
    # =========================================================
    if phase_selection == "Phase 1: Offline Modelling and Monitoring Setup":
        tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Case Study Reference",
            "Raw Data", 
            "Normalized Data", 
            "Correlation Matrix", 
            "Eigenvalue Matrix", 
            "Selected Eigenvector matrix",
            "Principal Component Computation",
            "Hotelling T2 Calculation",
            "Squared Prediction Error (SPE)"
        ], key="phase1_tabs")

        with tab0:
            st.subheader("Case Study Reference Diagram")
            st.caption("Process schematic detailing sensor tag locations across the system.")
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                st.image(IMAGE_URL, use_container_width=True)

        with tab1:
            st.subheader("Raw Data Table")
            st.caption(
                f"Baseline Normal Operating Conditions (NOC) training dataset collected from historical plant operation. "
                f"Active dataset: **{selected_noc_label}**."
            )
            st.dataframe(raw_df, use_container_width=True)

        with tab2:
            st.subheader("Normalized Data Table")
            st.caption("Auto-scaled dataset with zero mean and unit variance ($Z = \\frac{X - \\mu}{\\sigma}$) to eliminate parameter unit biases.")
            st.dataframe(normalized_df, use_container_width=True)

        with tab3:
            st.subheader("Correlation Matrix & Bivariate Parameter Inspection")
            st.markdown(
                "The **Correlation Matrix** displays linear interactions between operating parameters. "
                "Values near **+1.0** indicate strong positive linear correlation, **-1.0** indicate strong negative correlation, and **0.0** indicate no linear relationship."
            )
            left_col_corr, right_col_corr = st.columns([1.4, 1])
            
            with left_col_corr:
                st.markdown("**Correlation Matrix (Lower Triangle & Diagonal)**")
                fig_corr = px.imshow(
                    lower_corr,
                    labels=dict(color="Correlation"),
                    x=lower_corr.columns,
                    y=lower_corr.index,
                    color_continuous_scale="RdBu",
                    zmin=-1, zmax=1,
                    text_auto=".2f"
                )
                fig_corr.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    height=580,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                st.dataframe(lower_corr, use_container_width=True)

            with right_col_corr:
                st.markdown("**Interactive Parameter Scatter Plot**")
                cols_list = list(numeric_df.columns)
                c_x, c_y = st.columns(2)
                with c_x:
                    param_x = st.selectbox("X-Axis Parameter", cols_list, index=0, key="p1_param_x")
                with c_y:
                    default_y_idx = 1 if len(cols_list) > 1 else 0
                    param_y = st.selectbox("Y-Axis Parameter", cols_list, index=default_y_idx, key="p1_param_y")

                current_corr = numeric_df[param_x].corr(numeric_df[param_y])
                st.caption(f"**Pearson Correlation ({param_x} vs {param_y}):** `{current_corr:.4f}`")

                fig_scatter = px.scatter(
                    numeric_df,
                    x=param_x,
                    y=param_y,
                    hover_data=[numeric_df.index]
                )

                valid_pts = numeric_df[[param_x, param_y]].dropna()
                if len(valid_pts) > 1:
                    slope, intercept = np.polyfit(valid_pts[param_x], valid_pts[param_y], 1)
                    line_x = np.array([valid_pts[param_x].min(), valid_pts[param_x].max()])
                    line_y = slope * line_x + intercept
                    fig_scatter.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Trendline", line=dict(color="red", width=2)))

                fig_scatter.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    height=480,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with tab4:
            st.subheader("Eigenvalue Matrix & Cumulative Variance Analysis")
            st.markdown(
                "An **Eigenvalue** ($\lambda$) represents the variance captured along its corresponding **Eigenvector** (the new principal component axis shift). "
                "Formula: $\\mathbf{R}\\mathbf{v} = \\lambda \\mathbf{v}$, where $\\mathbf{R}$ is the correlation matrix and $\\mathbf{v}$ is the eigenvector."
            )
            left_col, right_col = st.columns([1.4, 1])
            
            with left_col:
                st.markdown("**Eigenvalue Matrix (Diagonal Only)**")
                fig_eigen_val = px.imshow(
                    eigenvalue_matrix_df,
                    labels=dict(color="Eigenvalue"),
                    x=eigenvalue_matrix_df.columns,
                    y=eigenvalue_matrix_df.index,
                    color_continuous_scale="Viridis",
                    text_auto=".3f"
                )
                fig_eigen_val.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    height=580,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_eigen_val, use_container_width=True)
                st.dataframe(eigenvalue_matrix_df, use_container_width=True)

            with right_col:
                st.markdown("**Cumulative Eigenvalue Distribution Plot**")
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Bar(x=pc_names, y=var_explained.tolist(), name="Individual Variance", marker_color=bar_colors))
                fig_cum.add_trace(go.Scatter(x=pc_names, y=cum_var_explained.tolist(), name="Cumulative Variance", mode="lines+markers", line=dict(color="#2CA02C", width=2), marker=dict(color=bar_colors, size=8)))
                fig_cum.add_shape(type="line", x0=-0.5, y0=0.8, x1=len(pc_names)-0.5, y1=0.8, line=dict(color="Red", width=2, dash="dash"))

                fig_cum.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    xaxis_title="Principal Components",
                    yaxis_title="Explained Variance Ratio",
                    yaxis=dict(range=[0, 1.05]),
                    height=580,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_cum, use_container_width=True)

        with tab5:
            st.subheader(f"Selected Eigenvector Matrix ({num_components_selected} Components <80% Cumulative Variance)")
            st.caption("Directional loading vectors ($P$) defining the coordinate rotation from original variable space into the principal component subspace.")
            if num_components_selected > 0:
                fig_selected_eigen = px.imshow(
                    selected_eigenvector_df,
                    labels=dict(color="Weight"),
                    x=selected_eigenvector_df.columns,
                    y=selected_eigenvector_df.index,
                    color_continuous_scale="Viridis",
                    text_auto=".3f"
                )
                fig_selected_eigen.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    height=600,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_selected_eigen, use_container_width=True)
                st.dataframe(selected_eigenvector_df, use_container_width=True)
            else:
                st.warning("No Principal Components are strictly below the 80% threshold.")

        with tab6:
            st.subheader("Principal Component Scores Computation")
            st.markdown("\n\n")
            st.markdown(
                "**PC Scores** ($T$) represent the transformed coordinates of process samples projected onto the lower-dimensional principal component subspace.  \n\n"
                "Formula: $T = Z \\cdot P_A$ *(where $Z$ is Normalized Data and $P_A$ is the Selected Eigenvector Matrix)*"
            )
            
            if num_components_selected > 0:
                left_col_pc, right_col_pc = st.columns([1.4, 1])
                
                with left_col_pc:
                    st.markdown("**Principal Component Scores Matrix**")
                    fig_pc_scores = px.imshow(
                        pc_scores_df,
                        labels=dict(color="Score"),
                        x=pc_scores_df.columns,
                        y=pc_scores_df.index,
                        color_continuous_scale="RdBu",
                        text_auto=".3f"
                    )
                    fig_pc_scores.update_layout(
                        font_family="Source Sans Pro, sans-serif",
                        height=580,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_pc_scores, use_container_width=True)
                    st.dataframe(pc_scores_df, use_container_width=True)
                
                with right_col_pc:
                    st.markdown("**PC Scores Scatter Inspection**")
                    pc_cols = list(pc_scores_df.columns)
                    c1_pc, c2_pc = st.columns(2)
                    with c1_pc:
                        pc_x = st.selectbox("X-Axis PC Score", pc_cols, index=0, key="p1_pc_x")
                    with c2_pc:
                        default_pc_y = 1 if len(pc_cols) > 1 else 0
                        pc_y = st.selectbox("Y-Axis PC Score", pc_cols, index=default_pc_y, key="p1_pc_y")

                    fig_pc_scatter = px.scatter(
                        pc_scores_df,
                        x=pc_x,
                        y=pc_y,
                        hover_data=[pc_scores_df.index],
                        title=f"Projection Space: {pc_x} vs {pc_y}"
                    )
                    fig_pc_scatter.update_layout(
                        font_family="Source Sans Pro, sans-serif",
                        height=480,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_pc_scatter, use_container_width=True)

        with tab7:
            st.subheader("Hotelling T² Statistical Limits & Control Chart")
            st.markdown("\n\n")
            st.markdown(
                "**Hotelling $T^2$** measures the **distance of sample points from the coordinate center (origin) within the PC model subspace**. "
                "It detects systematic, systemic shifts in process operation.  \n\n"
                "Formulas: $T^2_i = \\sum_{a=1}^{A} \\frac{t_{ia}^2}{\\lambda_a}$ | "
                "Upper Control Limit: $T^2_{\\alpha} = \\frac{A(n-1)}{n-A} F_{A, n-A, \\alpha}$"
            )
            
            if num_components_selected > 0:
                left_col_t2, right_col_t2 = st.columns([1.2, 1.2])
                
                with left_col_t2:
                    st.markdown("**Sample Hotelling T² Scores**")
                    st.dataframe(
                        t2_summary_df.style.background_gradient(
                            subset=['Hotelling_T2'], 
                            cmap='YlOrRd'
                        ).format("{:.4f}"),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    st.markdown("### Formula Breakdown & Calculated Metrics")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.write(f"**Samples (n):** `{n}`")
                        st.write(f"**Components (A):** `{A}`")
                        st.write(f"**Degrees of Freedom (df1, df2):** `({df1}, {df2})`")
                        st.write(f"**Multiplier [A(n-1)/(n-A)]:** `{multiplier:.4f}`")
                    
                    with metric_col2:
                        st.write(f"**F({df1}, {df2}, 0.05):** `{f_val_05:.4f}`")
                        st.write(f"**F({df1}, {df2}, 0.01):** `{f_val_01:.4f}`")
                        st.write(f"**Warning Limit (alpha = 0.05):** `{t2_warning_limit:.4f}`")
                        st.write(f"**Control Limit (alpha = 0.01):** `{t2_control_limit:.4f}`")

                with right_col_t2:
                    st.markdown("**Hotelling T² Control Chart**")
                    fig_t2 = go.Figure()
                    
                    fig_t2.add_trace(go.Scatter(
                        x=t2_summary_df.index,
                        y=t2_summary_df['Hotelling_T2'],
                        mode='lines+markers',
                        name='Hotelling T²',
                        line=dict(color='#1F77B4', width=2),
                        marker=dict(size=6)
                    ))
                    
                    if not np.isnan(t2_warning_limit):
                        fig_t2.add_shape(
                            type="line",
                            x0=t2_summary_df.index.min(),
                            y0=t2_warning_limit,
                            x1=t2_summary_df.index.max(),
                            y1=t2_warning_limit,
                            line=dict(color="Orange", width=2, dash="dot")
                        )
                        fig_t2.add_annotation(
                            x=t2_summary_df.index.max(),
                            y=t2_warning_limit,
                            text=f"Warning Limit (alpha=0.05): {t2_warning_limit:.2f}",
                            showarrow=False,
                            yshift=10,
                            font=dict(color="Orange")
                        )

                    if not np.isnan(t2_control_limit):
                        fig_t2.add_shape(
                            type="line",
                            x0=t2_summary_df.index.min(),
                            y0=t2_control_limit,
                            x1=t2_summary_df.index.max(),
                            y1=t2_control_limit,
                            line=dict(color="Red", width=2, dash="dash")
                        )
                        fig_t2.add_annotation(
                            x=t2_summary_df.index.max(),
                            y=t2_control_limit,
                            text=f"Control Limit (alpha=0.01): {t2_control_limit:.2f}",
                            showarrow=False,
                            yshift=10,
                            font=dict(color="Red")
                        )

                    fig_t2.update_layout(
                        font_family="Source Sans Pro, sans-serif",
                        xaxis_title="Sample Index",
                        yaxis_title="Hotelling T² Score",
                        height=580,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_t2, use_container_width=True)

            else:
                st.warning("No Principal Components selected to compute Hotelling T².")

        with tab8:
            st.subheader("Squared Prediction Error (SPE) Statistical Limits & Control Chart")
            st.markdown("\n\n")
            st.markdown(
                "**Squared Prediction Error (SPE / Q-statistic)** measures the **consistency of correlation structure** by evaluating residuals perpendicular to the model subspace. "
                "It detects abnormal events or new disturbances that break baseline parameter correlations.  \n\n"
                "**Formulas:**  \n"
                "• $PV^T = T \\cdot P_A^T$ *(Reconstructed Normal Data)*  \n"
                "• $E = Z - PV^T$ *(Residual Matrix)*  \n"
                "• $\\text{SPE}_i = \\sum_{j=1}^{p} E_{ij}^2$ *(Row-wise sum of squared residuals)*  \n"
                "• **Jackson-Mudholkar Limit Formula:**  \n"
                "  $Q_{\\alpha} = \\theta_1 \\left[ 1 + \\frac{z_{\\alpha} \\sqrt{2 \\theta_2 h_0^2}}{\\theta_1} + \\frac{\\theta_2 h_0 (h_0 - 1)}{\\theta_1^2} \\right]^{\\frac{1}{h_0}}$  \n"
                "  *(where $\\theta_i = \\sum_{j=A+1}^{p} \\lambda_j^i$ for unselected eigenvalues, $h_0 = 1 - \\frac{2 \\theta_1 \\theta_3}{3 \\theta_2^2}$)*"
            )

            if num_components_selected > 0:
                left_col_spe, right_col_spe = st.columns([1.2, 1.2])

                with left_col_spe:
                    st.markdown("**Sample SPE Scores & Residual Matrix $E$**")
                    st.dataframe(
                        spe_summary_df.style.background_gradient(
                            subset=['SPE'], 
                            cmap='YlOrRd'
                        ).format("{:.4f}"),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    st.markdown("### Formula Breakdown & Calculated Metrics")
                    
                    spe_metric_col1, spe_metric_col2 = st.columns(2)
                    with spe_metric_col1:
                        st.write(f"**Unselected Eigenvalues (p - A):** `{len(unselected_eigenvalues)}`")
                        st.write(f"**Theta 1 (θ₁):** `{theta1:.4f}`")
                        st.write(f"**Theta 2 (θ₂):** `{theta2:.4f}`")
                        st.write(f"**Theta 3 (θ₃):** `{theta3:.4f}`")
                        st.write(f"**Exponent (h₀):** `{h0:.4f}`")
                    
                    with spe_metric_col2:
                        st.write(f"**Z(0.05) [95%]:** `{z_val_05:.4f}`")
                        st.write(f"**Z(0.01) [99%]:** `{z_val_01:.4f}`")
                        st.write(f"**Warning Limit (alpha = 0.05):** `{spe_warning_limit:.4f}`")
                        st.write(f"**Control Limit (alpha = 0.01):** `{spe_control_limit:.4f}`")

                with right_col_spe:
                    st.markdown("**SPE Control Chart**")
                    fig_spe = go.Figure()
                    
                    fig_spe.add_trace(go.Scatter(
                        x=spe_summary_df.index,
                        y=spe_summary_df['SPE'],
                        mode='lines+markers',
                        name='SPE Score',
                        line=dict(color='#2CA02C', width=2),
                        marker=dict(size=6)
                    ))

                    if not np.isnan(spe_warning_limit):
                        fig_spe.add_shape(
                            type="line",
                            x0=spe_summary_df.index.min(),
                            y0=spe_warning_limit,
                            x1=spe_summary_df.index.max(),
                            y1=spe_warning_limit,
                            line=dict(color="Orange", width=2, dash="dot")
                        )
                        fig_spe.add_annotation(
                            x=spe_summary_df.index.max(),
                            y=spe_warning_limit,
                            text=f"Warning Limit (alpha=0.05): {spe_warning_limit:.2f}",
                            showarrow=False,
                            yshift=10,
                            font=dict(color="Orange")
                        )

                    if not np.isnan(spe_control_limit):
                        fig_spe.add_shape(
                            type="line",
                            x0=spe_summary_df.index.min(),
                            y0=spe_control_limit,
                            x1=spe_summary_df.index.max(),
                            y1=spe_control_limit,
                            line=dict(color="Red", width=2, dash="dash")
                        )
                        fig_spe.add_annotation(
                            x=spe_summary_df.index.max(),
                            y=spe_control_limit,
                            text=f"Control Limit (alpha=0.01): {spe_control_limit:.2f}",
                            showarrow=False,
                            yshift=10,
                            font=dict(color="Red")
                        )

                    fig_spe.update_layout(
                        font_family="Source Sans Pro, sans-serif",
                        xaxis_title="Sample Index",
                        yaxis_title="SPE Score",
                        height=580,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_spe, use_container_width=True)

            else:
                st.warning("No Principal Components selected to compute Residual E Table and SPE.")

    # =========================================================
    # PHASE 2: ONLINE MONITORING AND FAULT DETECTION
    # =========================================================
    elif phase_selection == "Phase 2: Online Monitoring and Fault Detection":

        st.caption(f"Phase 2 model is derived from the Phase 1 baseline dataset: **{selected_noc_label}**.")

        # Selectbox to pick between registered Online Cases
        selected_case_label = st.selectbox(
            "Select Online Case Study Dataset:",
            options=list(CASE_URLS.keys()),
            key="phase2_case_select"
        )
        
        target_case_url = CASE_URLS[selected_case_label]
        case_raw_df = load_data(target_case_url)
        case_raw_df.index = case_raw_df.index + 1
        
        if "Timestamp" in case_raw_df.columns:
            time_axis = case_raw_df["Timestamp"]
            numeric_case_df = case_raw_df.drop(columns=["Timestamp"]).select_dtypes(include=[np.number])
        else:
            time_axis = case_raw_df.index
            numeric_case_df = case_raw_df.select_dtypes(include=[np.number])

        numeric_case_df = numeric_case_df[numeric_df.columns]

        case_norm_array = scaler.transform(numeric_case_df)
        case_norm_df = pd.DataFrame(case_norm_array, columns=numeric_df.columns, index=time_axis)

        if num_components_selected > 0:
            case_pc_scores_array = np.dot(case_norm_df.values, selected_eigenvector_df.values)
            case_pc_scores_df = pd.DataFrame(case_pc_scores_array, columns=selected_pc_names, index=time_axis)
            
            case_t2_components = (case_pc_scores_array ** 2) / selected_eigenvalues
            case_t2_scores = np.sum(case_t2_components, axis=1)
            
            case_t2_summary_df = case_pc_scores_df.copy()
            case_t2_summary_df['Hotelling_T2'] = case_t2_scores

            # SPE Calculation (Phase 2 Online): PVt = PC_scores * Transposed_Selected_Eigenvectors
            case_pvt_array = np.dot(case_pc_scores_array, selected_eigenvector_df.values.T)
            case_pvt_df = pd.DataFrame(case_pvt_array, columns=numeric_df.columns, index=time_axis)

            # Residual E Matrix: E = Normalized Case Data - PVt
            case_e_matrix_df = case_norm_df - case_pvt_df

            # SPE is sum of squares of each row in E matrix
            case_spe_scores = np.sum(case_e_matrix_df.values ** 2, axis=1)
            case_spe_summary_df = pd.DataFrame({'SPE': case_spe_scores}, index=time_axis)
            
            # Native Streamlit tabs matching Phase 1 structure with key persistence
            p2_tab1, p2_tab2, p2_tab3, p2_tab4, p2_tab5 = st.tabs([
                "Case Data",
                "Normalized Data",
                "Principal Component Scores",
                "Hotelling T2 Online Control Chart",
                "SPE Online Control Chart"
            ], key="phase2_tabs")

            with p2_tab1:
                st.subheader(f"Phase 2 Raw Data ({selected_case_label})")
                st.caption("Real-time incoming plant data points under evaluation for potential operational anomalies.")
                st.dataframe(case_raw_df, use_container_width=True)

            with p2_tab2:
                st.subheader(f"Phase 2 Normalized Data ({selected_case_label} - Using Phase 1 Parameters)")
                st.caption("Online process data scaled using Phase 1 historical mean ($\mu_1$) and standard deviation ($\sigma_1$).")
                st.dataframe(case_norm_df, use_container_width=True)

            with p2_tab3:
                st.subheader(f"Phase 2 Online PC Scores ({selected_case_label})")
                st.markdown(
                    "Real-time projections of online operational state onto Phase 1 principal directions.  \n"
                    "Formula: $T_{\\text{new}} = Z_{\\text{new}} \\cdot P_{\\text{Phase 1}}$"
                )
                st.dataframe(case_pc_scores_df, use_container_width=True)

            with p2_tab4:
                st.subheader(f"Phase 2 Hotelling T² Online Control Chart ({selected_case_label})")
                st.caption(f"Evaluated against Phase 1 Limits (Warning Limit: {t2_warning_limit:.4f} | Control Limit: {t2_control_limit:.4f})")
                
                fig_online_t2 = go.Figure()

                x_start_p2 = time_axis.iloc[0]
                x_end_p2 = time_axis.iloc[-1]

                fig_online_t2.add_trace(go.Scatter(
                    x=time_axis,
                    y=case_t2_scores,
                    mode='lines+markers',
                    name='Online Hotelling T²',
                    line=dict(color='#1F77B4', width=2),
                    marker=dict(size=6)
                ))

                if not np.isnan(t2_warning_limit):
                    fig_online_t2.add_shape(
                        type="line",
                        x0=x_start_p2,
                        y0=t2_warning_limit,
                        x1=x_end_p2,
                        y1=t2_warning_limit,
                        line=dict(color="Orange", width=2, dash="dot")
                    )
                    fig_online_t2.add_annotation(
                        x=x_end_p2,
                        y=t2_warning_limit,
                        text=f"Phase 1 Warning Limit (0.05): {t2_warning_limit:.2f}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="Orange")
                    )

                if not np.isnan(t2_control_limit):
                    fig_online_t2.add_shape(
                        type="line",
                        x0=x_start_p2,
                        y0=t2_control_limit,
                        x1=x_end_p2,
                        y1=t2_control_limit,
                        line=dict(color="Red", width=2, dash="dash")
                    )
                    fig_online_t2.add_annotation(
                        x=x_end_p2,
                        y=t2_control_limit,
                        text=f"Phase 1 Control Limit (0.01): {t2_control_limit:.2f}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="Red")
                    )

                is_string_time = isinstance(x_start_p2, str)
                
                # Dynamic Y-axis upper limit (+15% above Control Limit)
                y_max_limit = t2_control_limit * 1.15 if not np.isnan(t2_control_limit) else None

                fig_online_t2.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    xaxis=dict(
                        title="Sample Time / Timestamp Index",
                        type='category' if is_string_time else None
                    ),
                    yaxis=dict(
                        title="Hotelling T² Score",
                        range=[0, y_max_limit] if y_max_limit else None
                    ),
                    height=580,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_online_t2, use_container_width=True)

                # ---------------------------------------------------------
                # T² CONTRIBUTION PLOT FOR SELECTED SAMPLE OR TIMEFRAME AVERAGE
                # ---------------------------------------------------------
                st.markdown("---")
                st.subheader(f"Fault Diagnosis: Variable Contribution Analysis ({selected_case_label})")
                st.caption("Decomposes out-of-control $T^2$ excursions back to individual original process variables to identify root causes.")

                AVG_LABEL = "Average (All Samples / Timeframe)"
                sample_options = [AVG_LABEL] + time_axis.tolist()

                # Explicit Session State Key to preserve selected timestamp
                selected_sample_id = st.selectbox(
                    "Select Sample / Timestamp to Diagnose:",
                    options=sample_options,
                    key="selected_sample_id_key"
                )

                N_samples = len(case_norm_df)
                p_vars = len(numeric_df.columns)
                
                all_term_matrices = np.zeros((N_samples, num_components_selected, p_vars))
                
                for i in range(N_samples):
                    s_i = case_pc_scores_array[i]
                    x_i = case_norm_df.iloc[i].values
                    all_term_matrices[i] = (s_i[:, np.newaxis] / selected_eigenvalues[:, np.newaxis]) * \
                                           (x_i[np.newaxis, :] * selected_eigenvector_df.values.T)

                all_sample_contributions = np.sum(all_term_matrices, axis=1)

                if selected_sample_id == AVG_LABEL:
                    variable_contributions = np.mean(all_sample_contributions, axis=0)
                    plot_title = f"Top 5 Contributing Factors (Average Across {selected_case_label})"
                    table_header = f"**Top 5 Root-Cause Ranking ({selected_case_label} Average)**"
                else:
                    sample_idx_loc = time_axis.tolist().index(selected_sample_id)
                    variable_contributions = all_sample_contributions[sample_idx_loc]
                    plot_title = f"Top 5 Contributing Factors for Sample `{selected_sample_id}`"
                    table_header = f"**Top 5 Root-Cause Ranking (`{selected_sample_id}`)**"

                contrib_df = pd.DataFrame({
                    'Variable': numeric_df.columns,
                    'Absolute_Contribution': np.abs(variable_contributions),
                    'Directional_Contribution': variable_contributions
                }).sort_values(by='Absolute_Contribution', ascending=False)

                top_5_df = contrib_df.head(5)

                col_top5_chart, col_top5_table = st.columns([1.3, 1])

                with col_top5_chart:
                    fig_contrib = px.bar(
                        top_5_df,
                        x='Absolute_Contribution',
                        y='Variable',
                        orientation='h',
                        title=plot_title,
                        labels={'Absolute_Contribution': 'Absolute T² Contribution Score', 'Variable': 'Parameter'},
                        color='Absolute_Contribution',
                        color_continuous_scale='Reds'
                    )
                    fig_contrib.update_layout(
                        font_family="Source Sans Pro, sans-serif",
                        yaxis=dict(autorange="reversed"),
                        height=380,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_contrib, use_container_width=True)

                with col_top5_table:
                    st.markdown(table_header)
                    st.dataframe(
                        top_5_df[['Variable', 'Absolute_Contribution', 'Directional_Contribution']].style.format({
                            'Absolute_Contribution': '{:.4f}',
                            'Directional_Contribution': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                
                st.markdown("---")
                st.markdown(f"**All Online Samples T² Summary ({selected_case_label})**")
                st.dataframe(
                    case_t2_summary_df.style.background_gradient(
                        subset=['Hotelling_T2'], 
                        cmap='YlOrRd'
                    ).format("{:.4f}"),
                    use_container_width=True
                )

            with p2_tab5:
                st.subheader(f"Phase 2 SPE Online Control Chart ({selected_case_label})")
                st.caption(f"Evaluated against Phase 1 Limits (Warning Limit: {spe_warning_limit:.4f} | Control Limit: {spe_control_limit:.4f})")

                fig_online_spe = go.Figure()

                x_start_p2_spe = time_axis.iloc[0]
                x_end_p2_spe = time_axis.iloc[-1]

                fig_online_spe.add_trace(go.Scatter(
                    x=time_axis,
                    y=case_spe_scores,
                    mode='lines+markers',
                    name='Online SPE',
                    line=dict(color='#2CA02C', width=2),
                    marker=dict(size=6)
                ))

                if not np.isnan(spe_warning_limit):
                    fig_online_spe.add_shape(
                        type="line",
                        x0=x_start_p2_spe,
                        y0=spe_warning_limit,
                        x1=x_end_p2_spe,
                        y1=spe_warning_limit,
                        line=dict(color="Orange", width=2, dash="dot")
                    )
                    fig_online_spe.add_annotation(
                        x=x_end_p2_spe,
                        y=spe_warning_limit,
                        text=f"Phase 1 Warning Limit (0.05): {spe_warning_limit:.2f}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="Orange")
                    )

                if not np.isnan(spe_control_limit):
                    fig_online_spe.add_shape(
                        type="line",
                        x0=x_start_p2_spe,
                        y0=spe_control_limit,
                        x1=x_end_p2_spe,
                        y1=spe_control_limit,
                        line=dict(color="Red", width=2, dash="dash")
                    )
                    fig_online_spe.add_annotation(
                        x=x_end_p2_spe,
                        y=spe_control_limit,
                        text=f"Phase 1 Control Limit (0.01): {spe_control_limit:.2f}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="Red")
                    )

                is_string_time_spe = isinstance(x_start_p2_spe, str)

                # Dynamic Y-axis upper limit (+15% above Control Limit)
                y_max_limit_spe = spe_control_limit * 1.15 if not np.isnan(spe_control_limit) else None

                fig_online_spe.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    xaxis=dict(
                        title="Sample Time / Timestamp Index",
                        type='category' if is_string_time_spe else None
                    ),
                    yaxis=dict(
                        title="SPE Score",
                        range=[0, y_max_limit_spe] if y_max_limit_spe else None
                    ),
                    height=580,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_online_spe, use_container_width=True)

                # ---------------------------------------------------------
                # SPE RESIDUAL CONTRIBUTION PLOT FOR SELECTED SAMPLE OR TIMEFRAME AVERAGE
                # ---------------------------------------------------------
                st.markdown("---")
                st.subheader(f"Fault Diagnosis: SPE Residual Contribution Analysis ({selected_case_label})")
                st.caption("Decomposes out-of-control SPE excursions back to individual original process variables to identify root causes.")

                AVG_LABEL_SPE = "Average (All Samples / Timeframe)"
                sample_options_spe = [AVG_LABEL_SPE] + time_axis.tolist()

                # Explicit Session State Key to preserve selected timestamp
                selected_sample_id_spe = st.selectbox(
                    "Select Sample / Timestamp to Diagnose:",
                    options=sample_options_spe,
                    key="selected_sample_id_key_spe"
                )

                # Residual contribution per variable = E_ij^2, since SPE_i = sum_j E_ij^2
                all_spe_contributions = case_e_matrix_df.values ** 2

                if selected_sample_id_spe == AVG_LABEL_SPE:
                    spe_variable_contributions = np.mean(all_spe_contributions, axis=0)
                    plot_title_spe = f"Top 5 Contributing Factors (Average Across {selected_case_label})"
                    table_header_spe = f"**Top 5 Root-Cause Ranking ({selected_case_label} Average)**"
                else:
                    sample_idx_loc_spe = time_axis.tolist().index(selected_sample_id_spe)
                    spe_variable_contributions = all_spe_contributions[sample_idx_loc_spe]
                    plot_title_spe = f"Top 5 Contributing Factors for Sample `{selected_sample_id_spe}`"
                    table_header_spe = f"**Top 5 Root-Cause Ranking (`{selected_sample_id_spe}`)**"

                contrib_df_spe = pd.DataFrame({
                    'Variable': numeric_df.columns,
                    'Absolute_Contribution': np.abs(spe_variable_contributions),
                    'Directional_Contribution': case_e_matrix_df.mean(axis=0).values if selected_sample_id_spe == AVG_LABEL_SPE else case_e_matrix_df.iloc[sample_idx_loc_spe].values
                }).sort_values(by='Absolute_Contribution', ascending=False)

                top_5_df_spe = contrib_df_spe.head(5)

                col_top5_chart_spe, col_top5_table_spe = st.columns([1.3, 1])

                with col_top5_chart_spe:
                    fig_contrib_spe = px.bar(
                        top_5_df_spe,
                        x='Absolute_Contribution',
                        y='Variable',
                        orientation='h',
                        title=plot_title_spe,
                        labels={'Absolute_Contribution': 'Absolute SPE Contribution Score', 'Variable': 'Parameter'},
                        color='Absolute_Contribution',
                        color_continuous_scale='Greens'
                    )
                    fig_contrib_spe.update_layout(
                        font_family="Source Sans Pro, sans-serif",
                        yaxis=dict(autorange="reversed"),
                        height=380,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_contrib_spe, use_container_width=True)

                with col_top5_table_spe:
                    st.markdown(table_header_spe)
                    st.dataframe(
                        top_5_df_spe[['Variable', 'Absolute_Contribution', 'Directional_Contribution']].style.format({
                            'Absolute_Contribution': '{:.4f}',
                            'Directional_Contribution': '{:.4f}'
                        }),
                        use_container_width=True
                    )

                st.markdown("---")
                st.markdown(f"**All Online Samples SPE Summary ({selected_case_label})**")
                st.dataframe(
                    case_spe_summary_df.style.background_gradient(
                        subset=['SPE'], 
                        cmap='YlOrRd'
                    ).format("{:.4f}"),
                    use_container_width=True
                )
        else:
            st.warning("No Principal Components were retained from Phase 1 setup.")

except Exception as e:
    st.error(f"Error executing multivariate analysis: {e}")
