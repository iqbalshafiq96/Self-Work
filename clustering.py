import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Equipment Health Monitoring & Residuals", layout="wide")

st.title("Machinery Health Monitoring: Euclidean vs. Mahalanobis Residuals")
st.markdown(
    "Detect equipment deviations by calculating operating residuals—the distance between "
    "real-time operating data points (**Flow, Suction Pressure, Discharge Temp, Motor Load**) "
    "and baseline mode centroids."
)

st.sidebar.header("Plant Simulation Controls")
n_samples = st.sidebar.slider("Operating Points per Baseline Mode", 3, 10, 5)
noise_level = st.sidebar.slider("Process Noise", 0.5, 3.0, 1.0)

@st.cache_data
def generate_equipment_data(n, noise):
    np.random.seed(42)
    
    # Define 3 Normal Baseline Operating Clusters (Mode Centroids)
    # Features: [Flow (m3/h), Suction Press (bara), Discharge Temp (degC), Load (%)]
    mode1_mean = [120.0, 3.20, 65.0, 45.0]  # Turndown Mode
    mode2_mean = [250.0, 4.00, 95.0, 78.0]  # Base Load Mode
    mode3_mean = [340.0, 4.80, 125.0, 95.0] # Peak Load Mode
    
    centroids = np.array([mode1_mean, mode2_mean, mode3_mean])
    
    cov1 = np.diag([25*noise, 0.04*noise, 9*noise, 16*noise])
    cov2 = np.diag([36*noise, 0.09*noise, 16*noise, 25*noise])
    cov3 = np.diag([49*noise, 0.16*noise, 25*noise, 16*noise])
    
    m1 = np.random.multivariate_normal(mode1_mean, cov1, n)
    m2 = np.random.multivariate_normal(mode2_mean, cov2, n)
    m3 = np.random.multivariate_normal(mode3_mean, cov3, n)
    
    X = np.vstack([m1, m2, m3])
    # Renamed Tag to Set
    sets = [f"Set_{i+1:02d}" for i in range(3 * n)]
    cols = ['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct']
    df = pd.DataFrame(np.round(X, 2), columns=cols, index=sets)
    
    # 1. Euclidean Residuals (Distance to closest centroid)
    dist_euc = cdist(df[cols], centroids, metric='euclidean')
    df['Euclidean_Cluster'] = np.argmin(dist_euc, axis=1) + 1
    df['Euclidean_Residual'] = np.min(dist_euc, axis=1).round(2)
    
    # 2. Mahalanobis Residuals (Scales variances and covariances)
    cov_pooled = np.cov(df[cols].values, rowvar=False)
    inv_cov = np.linalg.pinv(cov_pooled)
    dist_mah = cdist(df[cols], centroids, metric='mahalanobis', VI=inv_cov)
    df['Mahalanobis_Cluster'] = np.argmin(dist_mah, axis=1) + 1
    df['Mahalanobis_Residual'] = np.min(dist_mah, axis=1).round(2)
    
    # Flag Anomaly if residual exceeds threshold
    df['Euc_Health_Status'] = np.where(df['Euclidean_Residual'] > 12.0, "DEVIATION", "NORMAL")
    df['Mah_Health_Status'] = np.where(df['Mahalanobis_Residual'] > 2.2, "DEVIATION", "NORMAL")
    
    return df, centroids

df, centroids = generate_equipment_data(n_samples, noise_level)

with st.expander("View Full Operating Data Sets", expanded=False):
    st.dataframe(df, use_container_width=True)

tab_euc, tab_mah = st.tabs(["Tab 1: Euclidean Residuals", "Tab 2: Mahalanobis Residuals"])

# --- TAB 1: EUCLIDEAN ---
with tab_euc:
    st.header("Euclidean Distance Residual Analysis")
    st.warning("⚠️ Unscaled: Flow (0-350) and Temp (0-130) dominate the residual. Pressure variations (0.1 bara) are practically ignored.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Equipment Residual Table")
        st.dataframe(
            df[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct', 'Euclidean_Residual', 'Euc_Health_Status']],
            use_container_width=True
        )
        
    with col2:
        st.subheader("Euclidean Residuals by Data Set")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['red' if s == 'DEVIATION' else 'navy' for s in df['Euc_Health_Status']]
        ax.bar(df.index, df['Euclidean_Residual'], color=colors)
        ax.axhline(12.0, color='red', linestyle='--', label='Anomaly Threshold')
        ax.set_ylabel("Euclidean Distance (Residual)")
        ax.set_xticklabels(df.index, rotation=45)
        ax.legend()
        st.pyplot(fig)

# --- TAB 2: MAHALANOBIS ---
with tab_mah:
    st.header("Mahalanobis Distance Residual Analysis")
    st.success("✅ Covariance-Weighted: Normalizes differences in engineering units, allowing equal sensitivity across Pressure, Flow, and Temp.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Equipment Residual Table")
        st.dataframe(
            df[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct', 'Mahalanobis_Residual', 'Mah_Health_Status']],
            use_container_width=True
        )
        
    with col2:
        st.subheader("Mahalanobis Residuals by Data Set")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['red' if s == 'DEVIATION' else 'teal' for s in df['Mah_Health_Status']]
        ax.bar(df.index, df['Mahalanobis_Residual'], color=colors)
        ax.axhline(2.2, color='red', linestyle='--', label='Anomaly Threshold')
        ax.set_ylabel("Mahalanobis Distance (Residual)")
        ax.set_xticklabels(df.index, rotation=45)
        ax.legend()
        st.pyplot(fig)
