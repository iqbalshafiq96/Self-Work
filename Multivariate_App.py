import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Multivariate Process Monitoring", layout="wide")
st.title("Multivariate Statistical Process Control (MSPC)")

# URLs
IMAGE_URL = "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case.png"
GITHUB_PHASE1_CSV = "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv"
GITHUB_PHASE2_CSV = "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case0.csv"

@st.cache_data
def load_csv_data(url):
    """Loads CSV data from GitHub, auto-converting URL and auto-detecting delimiter."""
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    df = pd.read_csv(url, sep=None, engine='python')
    return df

# Initialize Session State Variables if not present
if "phase1_completed" not in st.session_state:
    st.session_state.phase1_completed = False

# Sidebar Navigation / Mode Selection
st.sidebar.title("Navigation & Phase Selection")
phase_selection = st.sidebar.radio(
    "Select Workflow Phase:",
    ["Phase 1: Offline Modelling & Training", "Phase 2: Online Monitoring & Fault Detection"]
)

st.sidebar.markdown("---")
if st.session_state.phase1_completed:
    st.sidebar.success("Phase 1 Baseline: Established")
    st.sidebar.caption(f"Selected Components (A): {st.session_state.A}")
    st.sidebar.caption(f"Control Limit (alpha=0.01): {st.session_state.t2_control_limit:.2f}")
    st.sidebar.caption(f"Warning Limit (alpha=0.05): {st.session_state.t2_warning_limit:.2f}")
else:
    st.sidebar.warning("Phase 1 Baseline: Not Set")

# ==============================================================================
# PHASE 1: OFFLINE MODELLING AND MONITORING SETUP
# ==============================================================================
if phase_selection == "Phase 1: Offline Modelling & Training":
    st.header("Phase 1: Offline Modelling and Monitoring Setup")
    
    try:
        raw_df = load_csv_data(GITHUB_PHASE1_CSV)
        raw_df.index = raw_df.index + 1
        numeric_df = raw_df.select_dtypes(include=[np.number]).dropna()

        # 1. Normalization & Save Scaler Statistics
        scaler = StandardScaler()
        normalized_array = scaler.fit_transform(numeric_df)
        normalized_df = pd.DataFrame(normalized_array, columns=numeric_df.columns, index=numeric_df.index)

        # Save Phase 1 normalization parameters to session state
        st.session_state.phase1_mean = scaler.mean_
        st.session_state.phase1_scale = scaler.scale_
        st.session_state.feature_columns = list(numeric_df.columns)

        # 2. Correlation Matrix
        corr_matrix = normalized_df.corr()
        mask_strictly_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        lower_corr = corr_matrix.copy()
        lower_corr[mask_strictly_upper] = np.nan

        # 3. Eigen Decomposition
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

        # 4. Variance Filter (<80%)
        var_explained = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        cum_var_explained = np.cumsum(var_explained)

        num_components_selected = int(np.sum(cum_var_explained < 0.80))
        selected_pc_names = pc_names[:num_components_selected]
        selected_eigenvectors = sorted_eigenvectors[:, :num_components_selected]
        selected_eigenvalues = sorted_eigenvalues[:num_components_selected]
        selected_eigenvector_df = eigenvector_df[selected_pc_names]

        bar_colors = ['#EF553B' if val < 0.80 else '#636EFA' for val in cum_var_explained]

        # 5. Hotelling T^2 Limits Calculation
        if num_components_selected > 0:
            pc_scores_array = np.dot(normalized_df.values, selected_eigenvector_df.values)
            pc_scores_df = pd.DataFrame(pc_scores_array, columns=selected_pc_names, index=normalized_df.index)
            
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

                # Save calculated Phase 1 values into session state for Phase 2
                st.session_state.selected_eigenvectors = selected_eigenvector_df
                st.session_state.selected_eigenvalues = selected_eigenvalues
                st.session_state.num_components_selected = A
                st.session_state.A = A
                st.session_state.t2_warning_limit = t2_warning_limit
                st.session_state.t2_control_limit = t2_control_limit
                st.session_state.phase1_completed = True
            else:
                multiplier = f_val_05 = f_val_01 = t2_warning_limit = t2_control_limit = np.nan
        else:
            pc_scores_df = t2_summary_df = pd.DataFrame()
            t2_warning_limit = t2_control_limit = multiplier = df1 = df2 = n = A = 0
            f_val_05 = f_val_01 = np.nan

        # Render Tabs for Phase 1
        tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Case Study Reference", "Raw Data", "Normalized Data", "Correlation Matrix",
            "Eigenvalue Matrix", "Selected Eigenvector matrix", "Principal Component Computation", "Hotelling T2 Calculation"
        ])

        with tab0:
            st.subheader("Case Study Reference Diagram")
            st.image(IMAGE_URL, use_container_width=True)

        with tab1:
            st.subheader("Phase 1 Raw Data Table")
            st.dataframe(raw_df, use_container_width=True)

        with tab2:
            st.subheader("Phase 1 Normalized Data Table")
            st.dataframe(normalized_df, use_container_width=True)

        with tab3:
            st.subheader("Correlation Matrix & Bivariate Parameter Inspection")
            c1, c2 = st.columns([1.4, 1])
            with c1:
                fig_corr = px.imshow(lower_corr, labels=dict(color="Correlation"), color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f")
                st.plotly_chart(fig_corr, use_container_width=True)
                st.dataframe(lower_corr, use_container_width=True)
            with c2:
                cols_list = list(numeric_df.columns)
                px_var = st.selectbox("X Parameter", cols_list, index=0, key="p1_x")
                py_var = st.selectbox("Y Parameter", cols_list, index=1 if len(cols_list) > 1 else 0, key="p1_y")
                fig_scat = px.scatter(numeric_df, x=px_var, y=py_var)
                st.plotly_chart(fig_scat, use_container_width=True)

        with tab4:
            st.subheader("Eigenvalue Matrix & Cumulative Variance Analysis")
            c1, c2 = st.columns([1.4, 1])
            with c1:
                fig_ev = px.imshow(eigenvalue_matrix_df, color_continuous_scale="Viridis", text_auto=".3f")
                st.plotly_chart(fig_ev, use_container_width=True)
                st.dataframe(eigenvalue_matrix_df, use_container_width=True)
            with c2:
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Bar(x=pc_names, y=var_explained.tolist(), marker_color=bar_colors, name="Variance"))
                fig_cum.add_trace(go.Scatter(x=pc_names, y=cum_var_explained.tolist(), mode="lines+markers", line=dict(color="#2CA02C"), name="Cumulative"))
                fig_cum.add_shape(type="line", x0=-0.5, y0=0.8, x1=len(pc_names)-0.5, y1=0.8, line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig_cum, use_container_width=True)

        with tab5:
            st.subheader(f"Selected Eigenvector Matrix ({num_components_selected} PCs <80% Cumulative Variance)")
            st.dataframe(selected_eigenvector_df, use_container_width=True)

        with tab6:
            st.subheader("Principal Component Scores Computation")
            st.markdown("**PC Scores = Normalized Data * Selected Eigenvectors**")
            st.dataframe(pc_scores_df, use_container_width=True)

        with tab7:
            st.subheader("Hotelling T² Statistical Limits & Control Chart")
            st.markdown("**T² Limit Formula:** T²_alpha = [ A * (n - 1) / (n - A) ] * F(A, n - A, alpha)")
            c1, c2 = st.columns([1.2, 1.2])
            with c1:
                st.dataframe(t2_summary_df.style.background_gradient(subset=['Hotelling_T2'], cmap='YlOrRd').format("{:.4f}"), use_container_width=True)
                st.write(f"**Warning Limit (alpha = 0.05):** `{t2_warning_limit:.4f}`")
                st.write(f"**Control Limit (alpha = 0.01):** `{t2_control_limit:.4f}`")
            with c2:
                fig_t2 = go.Figure()
                fig_t2.add_trace(go.Scatter(x=t2_summary_df.index, y=t2_summary_df['Hotelling_T2'], mode='lines+markers', name='Hotelling T²'))
                fig_t2.add_shape(type="line", x0=t2_summary_df.index.min(), y0=t2_warning_limit, x1=t2_summary_df.index.max(), y1=t2_warning_limit, line=dict(color="Orange", dash="dot"))
                fig_t2.add_shape(type="line", x0=t2_summary_df.index.min(), y0=t2_control_limit, x1=t2_summary_df.index.max(), y1=t2_control_limit, line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig_t2, use_container_width=True)

    except Exception as e:
        st.error(f"Phase 1 Processing Error: {e}")

# ==============================================================================
# PHASE 2: ONLINE MONITORING AND FAULT DETECTION
# ==============================================================================
else:
    st.header("Phase 2: Online Monitoring and Fault Detection")

    if not st.session_state.phase1_completed:
        st.info("Phase 1 baseline model parameters not found. Please run **Phase 1: Offline Modelling & Training** first.")
    else:
        try:
            # Load Phase 2 Case Data
            phase2_raw_df = load_csv_data(GITHUB_PHASE2_CSV)
            
            # Identify Timestamp and Process Variable Columns
            if 'Timestamp' in phase2_raw_df.columns:
                timestamp_col = phase2_raw_df['Timestamp']
                numeric_phase2_df = phase2_raw_df.drop(columns=['Timestamp'])
            else:
                timestamp_col = phase2_raw_df.index
                numeric_phase2_df = phase2_raw_df.select_dtypes(include=[np.number])

            # Ensure columns match Phase 1 exactly
            feature_cols = st.session_state.feature_columns
            numeric_phase2_df = numeric_phase2_df[feature_cols]

            # 1. Normalize Phase 2 Data using Phase 1 Mean and Standard Deviation
            phase2_norm_array = (numeric_phase2_df.values - st.session_state.phase1_mean) / st.session_state.phase1_scale
            phase2_norm_df = pd.DataFrame(phase2_norm_array, columns=feature_cols, index=phase2_raw_df.index)

            # 2. Calculate PC Scores using Phase 1 Selected Eigenvectors
            selected_eigenvectors = st.session_state.selected_eigenvectors
            selected_eigenvalues = st.session_state.selected_eigenvalues
            
            phase2_scores_array = np.dot(phase2_norm_df.values, selected_eigenvectors.values)
            phase2_scores_df = pd.DataFrame(
                phase2_scores_array,
                columns=selected_eigenvectors.columns,
                index=phase2_raw_df.index
            )

            # 3. Compute Hotelling T^2 for Phase 2 Data
            p2_t2_components = (phase2_scores_array ** 2) / selected_eigenvalues
            p2_t2_scores = np.sum(p2_t2_components, axis=1)

            p2_t2_summary_df = phase2_scores_df.copy()
            p2_t2_summary_df['Hotelling_T2'] = p2_t2_scores
            if 'Timestamp' in phase2_raw_df.columns:
                p2_t2_summary_df.insert(0, 'Timestamp', timestamp_col)

            # Retrieve Control and Warning Limits established in Phase 1
            t2_warning_limit = st.session_state.t2_warning_limit
            t2_control_limit = st.session_state.t2_control_limit

            # Status Flags (Fault Detection)
            p2_t2_summary_df['Status'] = 'Normal'
            p2_t2_summary_df.loc[p2_t2_summary_df['Hotelling_T2'] > t2_warning_limit, 'Status'] = 'Warning'
            p2_t2_summary_df.loc[p2_t2_summary_df['Hotelling_T2'] > t2_control_limit, 'Status'] = 'Fault Detected'

            # Phase 2 Output Tabs
            tab2_0, tab2_1, tab2_2, tab2_3, tab2_4 = st.tabs([
                "Case Data (Raw)",
                "Normalized Online Data",
                "Online PC Scores",
                "Online Hotelling T2 & Status",
                "Online Fault Detection Chart"
            ])

            with tab2_0:
                st.subheader("Phase 2 Online Raw Case Data")
                st.dataframe(phase2_raw_df, use_container_width=True)

            with tab2_1:
                st.subheader("Phase 2 Normalized Data (Scaled via Phase 1 Parameters)")
                st.caption("Standardized using Phase 1 Mean and Standard Deviation vectors.")
                st.dataframe(phase2_norm_df, use_container_width=True)

            with tab2_2:
                st.subheader("Online PC Scores Computation")
                st.markdown("**Online PC Scores = Phase 2 Normalized Data * Phase 1 Selected Eigenvectors**")
                st.dataframe(phase2_scores_df, use_container_width=True)

            with tab2_3:
                st.subheader("Online Hotelling T² Calculation & Fault Status")
                
                # Highlight fault rows in red and warning rows in orange
                def highlight_status(val):
                    if val == 'Fault Detected':
                        return 'background-color: #ff4b4b; color: white; font-weight: bold;'
                    elif val == 'Warning':
                        return 'background-color: #ffa500; color: black; font-weight: bold;'
                    return ''

                st.dataframe(
                    p2_t2_summary_df.style.applymap(highlight_status, subset=['Status']).format({'Hotelling_T2': '{:.4f}'}),
                    use_container_width=True
                )

            with tab2_4:
                st.subheader("Real-Time Process Monitoring & Control Chart")
                
                x_axis = timestamp_col if 'Timestamp' in phase2_raw_df.columns else p2_t2_summary_df.index
                
                fig_online_t2 = go.Figure()

                # Hotelling T2 Line
                fig_online_t2.add_trace(go.Scatter(
                    x=x_axis,
                    y=p2_t2_summary_df['Hotelling_T2'],
                    mode='lines+markers',
                    name='Online T²',
                    line=dict(color='#1F77B4', width=2),
                    marker=dict(size=6)
                ))

                # Warning Limit (Phase 1)
                fig_online_t2.add_shape(
                    type="line", x0=x_axis.iloc[0], y0=t2_warning_limit, x1=x_axis.iloc[-1], y1=t2_warning_limit,
                    line=dict(color="Orange", width=2, dash="dot")
                )
                fig_online_t2.add_annotation(
                    x=x_axis.iloc[-1], y=t2_warning_limit, text=f"Phase 1 Warning Limit: {t2_warning_limit:.2f}",
                    showarrow=False, yshift=10, font=dict(color="Orange")
                )

                # Control Limit (Phase 1)
                fig_online_t2.add_shape(
                    type="line", x0=x_axis.iloc[0], y0=t2_control_limit, x1=x_axis.iloc[-1], y1=t2_control_limit,
                    line=dict(color="Red", width=2, dash="dash")
                )
                fig_online_t2.add_annotation(
                    x=x_axis.iloc[-1], y=t2_control_limit, text=f"Phase 1 Control Limit: {t2_control_limit:.2f}",
                    showarrow=False, yshift=10, font=dict(color="Red")
                )

                # Highlight Out-of-Control Fault Points
                faults = p2_t2_summary_df[p2_t2_summary_df['Status'] == 'Fault Detected']
                if not faults.empty:
                    fault_x = timestamp_col.iloc[faults.index] if 'Timestamp' in phase2_raw_df.columns else faults.index
                    fig_online_t2.add_trace(go.Scatter(
                        x=fault_x,
                        y=faults['Hotelling_T2'],
                        mode='markers',
                        name='Fault (Exceeds Limit)',
                        marker=dict(color='Red', size=10, symbol='x')
                    ))

                fig_online_t2.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    xaxis_title="Timestamp / Sample Index",
                    yaxis_title="Hotelling T² Score",
                    height=580,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_online_t2, use_container_width=True)

        except Exception as e:
            st.error(f"Phase 2 Execution Error: {e}")
