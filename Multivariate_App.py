import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import f
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Multivariate PCA Analysis", layout="wide")
st.title("Multivariate PCA Analysis")

# Case Study Reference Image
IMAGE_URL = "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case.png"

# Direct GitHub link
GITHUB_CSV_URL = "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv"

@st.cache_data
def load_data(url):
    """Loads CSV data from GitHub, auto-converting URL and detecting delimiter."""
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    
    # Auto-detect delimiters (handles semicolons vs commas)
    df = pd.read_csv(url, sep=None, engine='python')
    return df

try:
    # 1. Load Data
    raw_df = load_data(GITHUB_CSV_URL)
    
    # Set DataFrame index to start at 1 instead of 0
    raw_df.index = raw_df.index + 1
    
    # Keep numeric columns for PCA calculations
    numeric_df = raw_df.select_dtypes(include=[np.number])

    # 3. Perform Normalization (StandardScaler: mean=0, std=1)
    scaler = StandardScaler()
    normalized_array = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(normalized_array, columns=numeric_df.columns, index=numeric_df.index)

    # 4. Perform Correlation Matrix Analysis (Lower Triangle INCLUDING Diagonal)
    corr_matrix = normalized_df.corr()
    
    # Mask strictly upper triangle (k=1 preserves diagonal 1s)
    mask_strictly_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    lower_corr = corr_matrix.copy()
    lower_corr[mask_strictly_upper] = np.nan

    # 5. Transform Correlation Matrix into Eigenvector matrix and Eigenvalues
    raw_eigenvalues, raw_eigenvectors = np.linalg.eig(corr_matrix.values)
    
    # Convert complex values to real numbers to avoid JSON serialization error
    eigenvalues = np.real(raw_eigenvalues)
    eigenvectors = np.real(raw_eigenvectors)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    # Full eigenvector matrix
    pc_names = [f"PC{i+1}" for i in range(len(sorted_eigenvalues))]
    eigenvector_df = pd.DataFrame(
        sorted_eigenvectors,
        index=numeric_df.columns,
        columns=pc_names
    )

    # Create Diagonal Eigenvalue Matrix (Diagonal values only, others NaN/Null)
    diag_eigen_matrix = np.full((len(sorted_eigenvalues), len(sorted_eigenvalues)), np.nan)
    np.fill_diagonal(diag_eigen_matrix, sorted_eigenvalues)
    eigenvalue_matrix_df = pd.DataFrame(
        diag_eigen_matrix,
        index=pc_names,
        columns=pc_names
    )

    # 6. Cumulative Eigenvalue Calculation & Strictly Below 80% Threshold Filter
    var_explained = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cum_var_explained = np.cumsum(var_explained)

    # Select components strictly BELOW the 80% threshold (< 0.80)
    num_components_selected = int(np.sum(cum_var_explained < 0.80))
    selected_pc_names = pc_names[:num_components_selected]
    selected_eigenvectors = sorted_eigenvectors[:, :num_components_selected]
    selected_eigenvalues = sorted_eigenvalues[:num_components_selected]
    selected_eigenvector_df = eigenvector_df[selected_pc_names]

    # Dynamic color mapping: Red for strictly <80% threshold, Blue for >=80%
    bar_colors = [
        '#EF553B' if val < 0.80 else '#636EFA' 
        for val in cum_var_explained
    ]

    # 7. Matrix Multiplication: Normalized Data x Selected Eigenvectors
    if num_components_selected > 0:
        pc_scores_array = np.dot(normalized_df.values, selected_eigenvector_df.values)
        pc_scores_df = pd.DataFrame(
            pc_scores_array,
            columns=selected_pc_names,
            index=normalized_df.index
        )
        
        # --- Hotelling T^2 Computation ---
        # Component-wise T2 terms: (t_ij^2) / lambda_j
        t2_components = (pc_scores_array ** 2) / selected_eigenvalues
        
        # T^2 Statistic per sample (Sum across selected PCs)
        t2_scores = np.sum(t2_components, axis=1)
        
        # Combine PC Scores with T^2 Column
        t2_summary_df = pc_scores_df.copy()
        t2_summary_df['Hotelling_T2'] = t2_scores
        
        # Parameters for Statistical Control Limits
        n = len(normalized_df)
        A = num_components_selected
        df1 = A
        df2 = n - A
        
        if n > A:
            # F values for alpha = 0.05 and alpha = 0.01
            f_val_05 = f.ppf(1 - 0.05, df1, df2)
            f_val_01 = f.ppf(1 - 0.01, df1, df2)
            
            # Warning Limit (alpha = 0.05)
            t2_warning_limit = (A * (n - 1) / (n - A)) * f_val_05
            
            # Control Limit (alpha = 0.01)
            t2_control_limit = (A * (n - 1) / (n - A)) * f_val_01
        else:
            f_val_05 = f_val_01 = np.nan
            t2_warning_limit = t2_control_limit = np.nan
            
    else:
        pc_scores_df = pd.DataFrame()
        t2_summary_df = pd.DataFrame()
        t2_warning_limit = t2_control_limit = np.nan
        df1 = df2 = n = A = 0
        f_val_05 = f_val_01 = np.nan

    # --- Streamlit Tabs Output ---
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Case Study Reference",
        "Raw Data", 
        "Normalized Data", 
        "Correlation Matrix", 
        "Eigenvalue Matrix", 
        "Selected Eigenvector matrix",
        "Principal Component Computation",
        "Hotelling T2 Calculation"
    ])

    # Tab 0: Case Study Reference
    with tab0:
        st.subheader("Case Study Reference Diagram")
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.image(IMAGE_URL, use_container_width=True)

    # Tab 1: Raw Data
    with tab1:
        st.subheader("Raw Data Table")
        st.dataframe(raw_df, use_container_width=True)

    # Tab 2: Normalized Data
    with tab2:
        st.subheader("Normalized Data Table")
        st.dataframe(normalized_df, use_container_width=True)

    # Tab 3: Correlation Matrix Analysis & Interactive Scatter Plot Side-by-Side
    with tab3:
        st.subheader("Correlation Matrix & Bivariate Parameter Inspection")
        
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
                xaxis_showgrid=False,
                yaxis_showgrid=False,
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
                param_x = st.selectbox("X-Axis Parameter", cols_list, index=0)
            with c_y:
                default_y_idx = 1 if len(cols_list) > 1 else 0
                param_y = st.selectbox("Y-Axis Parameter", cols_list, index=default_y_idx)

            current_corr = numeric_df[param_x].corr(numeric_df[param_y])
            st.caption(f"**Pearson Correlation ({param_x} vs {param_y}):** `{current_corr:.4f}`")

            fig_scatter = px.scatter(
                numeric_df,
                x=param_x,
                y=param_y,
                hover_data=[numeric_df.index],
                labels={param_x: param_x, param_y: param_y}
            )

            valid_pts = numeric_df[[param_x, param_y]].dropna()
            if len(valid_pts) > 1:
                x_vals = valid_pts[param_x]
                y_vals = valid_pts[param_y]
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                
                line_x = np.array([x_vals.min(), x_vals.max()])
                line_y = slope * line_x + intercept
                
                fig_scatter.add_trace(
                    go.Scatter(
                        x=line_x, 
                        y=line_y, 
                        mode="lines", 
                        name="Trendline", 
                        line=dict(color="red", width=2)
                    )
                )

            fig_scatter.update_layout(
                font_family="Source Sans Pro, sans-serif",
                height=480,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Tab 4: Eigenvalue Matrix & Cumulative Distribution Plot Side-by-Side
    with tab4:
        st.subheader("Eigenvalue Matrix & Cumulative Variance Analysis")
        
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
            
            fig_cum.add_trace(go.Bar(
                x=pc_names, 
                y=var_explained.tolist(), 
                name="Individual Variance",
                marker_color=bar_colors
            ))
            
            fig_cum.add_trace(go.Scatter(
                x=pc_names, 
                y=cum_var_explained.tolist(), 
                name="Cumulative Variance", 
                mode="lines+markers",
                line=dict(color="#2CA02C", width=2),
                marker=dict(color=bar_colors, size=8)
            ))
            
            fig_cum.add_shape(
                type="line",
                x0=-0.5,
                y0=0.8,
                x1=len(pc_names)-0.5,
                y1=0.8,
                line=dict(color="Red", width=2, dash="dash")
            )

            fig_cum.update_layout(
                font_family="Source Sans Pro, sans-serif",
                xaxis_title="Principal Components",
                yaxis_title="Explained Variance Ratio",
                yaxis=dict(range=[0, 1.05]),
                height=580,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_cum, use_container_width=True)

    # Tab 5: Selected Eigenvector Matrix
    with tab5:
        st.subheader(f"Selected Eigenvector Matrix ({num_components_selected} Components strictly <80% Cumulative Variance)")
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

    # Tab 6: Principal Component Computation
    with tab6:
        st.subheader("Principal Component Scores Computation")
        
        st.markdown(
            """
            <div style="font-family: 'Source Sans Pro', sans-serif; font-size: 1.1rem; margin-bottom: 1.2rem; color: #31333F;">
                <strong>PC Scores</strong> = <strong>Normalized Data</strong> &times; <strong>Selected Eigenvectors</strong>
            </div>
            """, 
            unsafe_allow_html=True
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
                
                st.caption("Principal Component Scores Matrix Data Table")
                st.dataframe(pc_scores_df, use_container_width=True)
            
            with right_col_pc:
                st.markdown("**PC Scores Scatter Inspection**")
                pc_cols = list(pc_scores_df.columns)
                
                c1_pc, c2_pc = st.columns(2)
                with c1_pc:
                    pc_x = st.selectbox("X-Axis PC Score", pc_cols, index=0)
                with c2_pc:
                    default_pc_y = 1 if len(pc_cols) > 1 else 0
                    pc_y = st.selectbox("Y-Axis PC Score", pc_cols, index=default_pc_y)

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
        else:
            st.warning("No Principal Components available (0 components strictly below the 80% cumulative threshold).")

    # Tab 7: Hotelling T^2 Calculation
    with tab7:
        st.subheader("Hotelling T² Calculation per Sample")
        
        st.markdown(
            """
            <div style="font-family: 'Source Sans Pro', sans-serif; font-size: 1.05rem; margin-bottom: 1.2rem; color: #31333F;">
                <strong>T² Limit Formula:</strong> &nbsp;
                <em>T²<sub>&alpha;</sub> = [ A &times; (n - 1) / (n - A) ] &times; F<sub>(A, n-A, &alpha;)</sub></em>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        if num_components_selected > 0:
            left_col_t2, right_col_t2 = st.columns([1.3, 1])
            
            # Left Side: Data Table with PC Scores and T2 Column
            with left_col_t2:
                st.markdown("**PC Scores Matrix with Computed Hotelling T²**")
                
                st.dataframe(
                    t2_summary_df.style.background_gradient(
                        subset=['Hotelling_T2'], 
                        cmap='YlOrRd'
                    ).format("{:.4f}"),
                    use_container_width=True
                )
                
                if not np.isnan(t2_control_limit):
                    st.caption(f"**Sample Count (n):** `{n}` | **Selected Components (A):** `{A}`")
                    st.caption(f"**F({df1}, {df2}, 0.05):** `{f_val_05:.4f}` &nbsp;|&nbsp; **Warning Limit (95%):** `{t2_warning_limit:.4f}`")
                    st.caption(f"**F({df1}, {df2}, 0.01):** `{f_val_01:.4f}` &nbsp;|&nbsp; **Control Limit (99%):** `{t2_control_limit:.4f}`")

            # Right Side: Hotelling T2 Chart with Warning & Control Limits
            with right_col_t2:
                st.markdown("**Hotelling T² Control Chart**")
                
                fig_t2 = go.Figure()
                
                # T2 values line/scatter plot
                fig_t2.add_trace(go.Scatter(
                    x=t2_summary_df.index,
                    y=t2_summary_df['Hotelling_T2'],
                    mode='lines+markers',
                    name='Hotelling T²',
                    line=dict(color='#1F77B4', width=2),
                    marker=dict(size=6)
                ))
                
                # Warning Limit Line (alpha = 0.05)
                if not np.isnan(t2_warning_limit):
                    fig_t2.add_shape(
                        type="line",
                        x0=t2_summary_df.index.min(),
                        y0=t2_warning_limit,
                        x1=t2_summary_df.index.max(),
                        y1=t2_warning_limit,
                        line=dict(color="Orange", width=2, dash="dot"),
                    )
                    fig_t2.add_annotation(
                        x=t2_summary_df.index.max(),
                        y=t2_warning_limit,
                        text=f"Warning Limit (95%): {t2_warning_limit:.2f}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="Orange")
                    )

                # Control Limit Line (alpha = 0.01)
                if not np.isnan(t2_control_limit):
                    fig_t2.add_shape(
                        type="line",
                        x0=t2_summary_df.index.min(),
                        y0=t2_control_limit,
                        x1=t2_summary_df.index.max(),
                        y1=t2_control_limit,
                        line=dict(color="Red", width=2, dash="dash"),
                    )
                    fig_t2.add_annotation(
                        x=t2_summary_df.index.max(),
                        y=t2_control_limit,
                        text=f"Control Limit (99%): {t2_control_limit:.2f}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="Red")
                    )

                fig_t2.update_layout(
                    font_family="Source Sans Pro, sans-serif",
                    xaxis_title="Sample Index",
                    yaxis_title="Hotelling T² Score",
                    height=520,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_t2, use_container_width=True)

        else:
            st.warning("No Principal Components selected (<80% variance threshold) to calculate Hotelling T².")

except Exception as e:
    st.error(f"Error loading data: {e}")
