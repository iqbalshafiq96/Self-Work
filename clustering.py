import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Equipment Residual Health Monitor", layout="wide")

st.title("Machinery Health Monitoring: Fixed Baseline vs. Real-Time Testing")
st.markdown(
    "**Step 1:** Establish fixed baseline operating centroids from historical training data.  \n"
    "**Step 2:** Evaluate incoming real-time operating sets against the fixed baseline to compute health residuals."
)

# ---------------------------------------------------------
# STEP 1: FIX THE TRAINING BASELINE
# ---------------------------------------------------------
st.sidebar.header("1. Baseline Setup (Training Phase)")
baseline_seed = st.sidebar.number_input("Historical Baseline Seed", value=42, step=1)

@st.cache_data
def train_baseline_model(seed):
    np.random.seed(seed)
    # Define 3 baseline operating modes [Flow, Press, Temp, Load]
    m1 = np.random.multivariate_normal([120.0, 3.20, 65.0, 45.0], np.diag([25, 0.04, 9, 16]), 50)
    m2 = np.random.multivariate_normal([250.0, 4.00, 95.0, 78.0], np.diag([36, 0.09, 16, 25]), 50)
    m3 = np.random.multivariate_normal([340.0, 4.80, 125.0, 95.0], np.diag([49, 0.16, 25, 16]), 50)
    
    training_data = np.vstack([m1, m2, m3])
    cols = ['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct']
    
    # Extract Fixed Centroids
    centroids = np.array([m1.mean(axis=0), m2.mean(axis=0), m3.mean(axis=0)])
    
    # Extract Fixed Inverse Covariance Matrix (from training baseline)
    cov_matrix = np.cov(training_data, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    return centroids, inv_cov_matrix, cols

centroids, inv_cov, cols = train_baseline_model(baseline_seed)

st.sidebar.subheader("2. Real-Time Condition Controls")
inject_anomaly = st.sidebar.checkbox("Inject Artificial Pressure Drop (Anomaly)")

# ---------------------------------------------------------
# STEP 2: GENERATE CURRENT TEST DATA (ONLINE PHASE)
# ---------------------------------------------------------
def generate_current_test_data(inject_fault):
    np.random.seed(101) # Different seed for new operating data
    
    # Current test operating points across modes
    t1 = np.random.multivariate_normal([120.0, 3.20, 65.0, 45.0], np.diag([25, 0.04, 9, 16]), 3)
    t2 = np.random.multivariate_normal([250.0, 4.00, 95.0, 78.0], np.diag([36, 0.09, 16, 25]), 3)
    t3 = np.random.multivariate_normal([340.0, 4.80, 125.0, 95.0], np.diag([49, 0.16, 25, 16]), 3)
    
    if inject_fault:
        # Simulate a 0.8 bara suction pressure drop on Set_05 (Mode 2) without changing Flow/Temp
        t2[1, 1] -= 0.85
    
    test_X = np.vstack([t1, t2, t3])
    sets = [f"Set_{i+1:02d}" for i in range(len(test_X))]
    df = pd.DataFrame(np.round(test_X, 2), columns=cols, index=sets)
    
    # Calculate Residuals using the FIXED Training Baseline
    dist_euc = cdist(df[cols], centroids, metric='euclidean')
    df['Euclidean_Cluster'] = np.argmin(dist_euc, axis=1) + 1
    df['Euclidean_Residual'] = np.min(dist_euc, axis=1).round(2)
    
    dist_mah = cdist(df[cols], centroids, metric='mahalanobis', VI=inv_cov)
    df['Mahalanobis_Cluster'] = np.argmin(dist_mah, axis=1) + 1
    df['Mahalanobis_Residual'] = np.min(dist_mah, axis=1).round(2)
    
    df['Euc_Health_Status'] = np.where(df['Euclidean_Residual'] > 12.0, "DEVIATION", "NORMAL")
    df['Mah_Health_Status'] = np.where(df['Mahalanobis_Residual'] > 2.5, "DEVIATION", "NORMAL")
    
    return df

df_current = generate_current_test_data(inject_anomaly)

with st.expander("View Fixed Training Centroids & Current Test Data", expanded=True):
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.write("**Fixed Baseline Centroids (Training)**")
        st.dataframe(pd.DataFrame(np.round(centroids, 2), columns=cols, index=["Mode 1", "Mode 2", "Mode 3"]))
    with col_c2:
        st.write("**Current Test Data Sets (Incoming)**")
        st.dataframe(df_current[cols])

# ---------------------------------------------------------
# OUTPUT TABS
# ---------------------------------------------------------
tab_euc, tab_mah = st.tabs(["Tab 1: Euclidean Residuals", "Tab 2: Mahalanobis Residuals"])

with tab_euc:
    st.header("Euclidean Residuals (Unscaled Distance)")
    st.caption("Distance evaluated against fixed baseline centroids without variance normalization.")
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(df_current[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct', 'Euclidean_Residual', 'Euc_Health_Status']], use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['red' if s == 'DEVIATION' else 'navy' for s in df_current['Euc_Health_Status']]
        ax.bar(df_current.index, df_current['Euclidean_Residual'], color=colors)
        ax.axhline(12.0, color='red', linestyle='--', label='Threshold')
        ax.set_ylabel("Euclidean Residual")
        ax.set_xticklabels(df_current.index, rotation=45)
        ax.legend()
        st.pyplot(fig)

with tab_mah:
    st.header("Mahalanobis Residuals (Covariance-Weighted Distance)")
    st.caption("Distance evaluated against fixed baseline using historical inverse covariance matrix.")
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(df_current[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct', 'Mahalanobis_Residual', 'Mah_Health_Status']], use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['red' if s == 'DEVIATION' else 'teal' for s in df_current['Mah_Health_Status']]
        ax.bar(df_current.index, df_current['Mahalanobis_Residual'], color=colors)
        ax.axhline(2.5, color='red', linestyle='--', label='Threshold')
        ax.set_ylabel("Mahalanobis Residual")
        ax.set_xticklabels(df_current.index, rotation=45)
        ax.legend()
        st.pyplot(fig)
