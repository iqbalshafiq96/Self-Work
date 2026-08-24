import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Multivariate PCA Analysis", layout="wide")
st.title("Multivariate PCA Analysis")

# Define GitHub raw file URL
GITHUB_CSV_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPOSITORY/main/Multivariate_NOC6.csv"

@st.cache_data
def load_data(url):
    """Loads CSV data from GitHub."""
    return pd.read_csv(url)

try:
    # 1. Load Data
    raw_df = load_data(GITHUB_CSV_URL)
    
    # Clean non-numeric columns if any exist
    numeric_df = raw_df.select_dtypes(include=[np.number])

    # 3. Perform Normalization (StandardScaler: mean=0, std=1)
    scaler = StandardScaler()
    normalized_array = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(normalized_array, columns=numeric_df.columns)

    # 4. Perform Correlation Matrix Analysis (Lower Triangle Only)
    corr_matrix = normalized_df.corr()
    
    # Mask the upper triangle and diagonal for lower-triangle display
    mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool))
    lower_corr = corr_matrix.copy()
    lower_corr[mask_upper] = np.nan

    # 5. Transform Correlation Matrix into Eigenvector and Eigenvalue
    # Using correlation matrix for eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    # Format eigenvectors into DataFrame
    eigenvector_df = pd.DataFrame(
        sorted_eigenvectors,
        index=numeric_df.columns,
        columns=[f"PC{i+1}" for i in range(len(sorted_eigenvalues))]
    )

    # 6. Cumulative Eigenvalue Calculation
    var_explained = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cum_var_explained = np.cumsum(var_explained)

    # --- Streamlit Tabs Output ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Raw Data", 
        "Normalized Data", 
        "Correlation Matrix", 
        "Eigen vector matrix", 
        "Cumulative eigen value distribution plot"
    ])

    # Tab 1: Raw Data
    with tab1:
        st.subheader("Raw Data Table")
        st.dataframe(raw_df, use_container_width=True)

    # Tab 2: Normalized Data
    with tab2:
        st.subheader("Normalized Data Table (Standardized)")
        st.dataframe(normalized_df, use_container_width=True)

    # Tab 3: Interactive Correlation Matrix
    with tab3:
        st.subheader("Lower Triangular Correlation Matrix")
        
        # Interactive Heatmap via Plotly
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

    # Tab 4: Interactive Eigenvector Matrix
    with tab4:
        st.subheader("Eigenvector Matrix")
        
        # Interactive Heatmap for Eigenvectors
        fig_eigen = px.imshow(
            eigenvector_df,
            labels=dict(color="Weight"),
            x=eigenvector_df.columns,
            y=eigenvector_df.index,
            color_continuous_scale="Viridis",
            text_auto=".3f"
        )
        fig_eigen.update_layout(
            font_family="Source Sans Pro, sans-serif",
            height=600
        )
        st.plotly_chart(fig_eigen, use_container_width=True)
        st.dataframe(eigenvector_df, use_container_width=True)

    # Tab 5: Cumulative Eigenvalue Distribution Plot
    with tab5:
        st.subheader("Cumulative Eigenvalue / Variance Plot")
        
        components = [f"PC{i+1}" for i in range(len(sorted_eigenvalues))]
        
        fig_cum = go.Figure()
        
        # Individual variance bar chart
        fig_cum.add_trace(go.Bar(
            x=components, 
            y=var_explained, 
            name="Individual Variance"
        ))
        
        # Cumulative variance line chart
        fig_cum.add_trace(go.Scatter(
            x=components, 
            y=cum_var_explained, 
            name="Cumulative Variance", 
            mode="lines+markers"
        ))
        
        # 80% Threshold Line
        fig_cum.add_shape(
            type="line",
            x0=-0.5,
            y0=0.8,
            x1=len(components)-0.5,
            y1=0.8,
            line=dict(color="Red", width=2, dash="dash")
        )

        fig_cum.update_layout(
            font_family="Source Sans Pro, sans-serif",
            xaxis_title="Principal Components",
            yaxis_title="Explained Variance Ratio",
            yaxis=dict(range=[0, 1.05]),
            height=500
        )
        
        st.plotly_chart(fig_cum, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure to replace `GITHUB_CSV_URL` with your exact GitHub raw link.")
