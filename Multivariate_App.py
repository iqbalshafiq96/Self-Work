import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    selected_eigenvector_df = eigenvector_df[selected_pc_names]

    # --- Streamlit Tabs Output ---
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Case Study Reference",
        "Raw Data", 
        "Normalized Data", 
        "Correlation Matrix", 
        "Eigenvalue Matrix", 
        "Selected Eigenvector matrix"
    ])

    # Tab 0: Case Study Reference (Scaled to 80% width centered)
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

    # Tab 3: Interactive Correlation Matrix (Lower Triangle + Diagonal 1s)
    with tab3:
        st.subheader("Correlation Matrix")
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
            height=600
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Tab 4: Combined Eigenvalue Matrix & Cumulative Distribution Plot Side-by-Side
    with tab4:
        st.subheader("Eigenvalue Matrix & Cumulative Variance Analysis")
        
        left_col, right_col = st.columns(2)
        
        # Left Side: Eigenvalue Matrix Heatmap & Table
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
                height=450
            )
            st.plotly_chart(fig_eigen_val, use_container_width=True)
            st.dataframe(eigenvalue_matrix_df, use_container_width=True)

        # Right Side: Cumulative Eigenvalue Distribution Plot
        with right_col:
            st.markdown("**Cumulative Eigenvalue Distribution Plot**")
            fig_cum = go.Figure()
            
            # Individual variance bar plot
            fig_cum.add_trace(go.Bar(
                x=pc_names, 
                y=var_explained.tolist(), 
                name="Individual Variance"
            ))
            
            # Cumulative variance line plot
            fig_cum.add_trace(go.Scatter(
                x=pc_names, 
                y=cum_var_explained.tolist(), 
                name="Cumulative Variance", 
                mode="lines+markers"
            ))
            
            # 80% Threshold Line
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
                height=450
            )
            st.plotly_chart(fig_cum, use_container_width=True)

    # Tab 5: Selected Eigenvector Matrix (Strictly Below 80% Variance Threshold)
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
                height=600
            )
            st.plotly_chart(fig_selected_eigen, use_container_width=True)
            st.dataframe(selected_eigenvector_df, use_container_width=True)
        else:
            st.warning("No Principal Components are strictly below the 80% threshold.")

except Exception as e:
    st.error(f"Error loading data: {e}")
