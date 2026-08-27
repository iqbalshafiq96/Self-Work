import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Equipment Residual Health & Contribution", layout="wide")

st.title("Machinery Health Monitoring: Residuals & Feature Contributions")
st.markdown(
    "**Step 1:** Train fixed baseline operating centroids from historical data.  \n"
    "**Step 2:** Compute real-time residuals and break down **variable contributions** to identify root causes."
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
    
    # Extract Fixed Centroids & Inverse Covariance Matrix
    centroids = np.array([m1.mean(axis=0), m2.mean(axis=0), m3.mean(axis=0)])
    cov_matrix = np.cov(training_data, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    return centroids, inv_cov_matrix, cols

centroids, inv_cov, cols = train_baseline_model(baseline_seed)

st.sidebar.subheader("2. Real-Time Condition Controls")
inject_anomaly = st.sidebar.checkbox("Inject Artificial Pressure Drop (Set_05)", value=True)

# ---------------------------------------------------------
# STEP 2: CALCULATE RESIDUALS AND CONTRIBUTIONS
# ---------------------------------------------------------
def generate_and_analyze_test_data(inject_fault):
    np.random.seed(101)
    
    t1 = np.random.multivariate_normal([120.0, 3.20, 65.0, 45.0], np.diag([25, 0.04, 9, 16]), 3)
    t2 = np.random.multivariate_normal([250.0, 4.00, 95.0, 78.0], np.diag([36, 0.09, 16, 25]), 3)
    t3 = np.random.multivariate_normal([340.0, 4.80, 125.0, 95.0], np.diag([49, 0.16, 25, 16]), 3)
    
    if inject_fault:
        # Induce a pressure anomaly on Set_05
        t2[1, 1] -= 0.85
    
    test_X = np.vstack([t1, t2, t3])
    sets = [f"Set_{i+1:02d}" for i in range(len(test_X))]
    df = pd.DataFrame(np.round(test_X, 2), columns=cols, index=sets)
    
    # --- 1. Euclidean Calculations & Unscaled Contributions ---
    dist_euc = cdist(df[cols], centroids, metric='euclidean')
    df['Euc_Cluster'] = np.argmin(dist_euc, axis=1)
    df['Euclidean_Residual'] = np.min(dist_euc, axis=1).round(2)
    
    euc_top_contribs = []
    for idx, row in df.iterrows():
        c_idx = int(row['Euc_Cluster'])
        diff = np.abs(row[cols].values - centroids[c_idx])
        top_var = cols[np.argmax(diff)]
        euc_top_contribs.append(top_var)
    df['Euc_Top_Contributor'] = euc_top_contribs
    df['Euc_Cluster'] = df['Euc_Cluster'] + 1
    
    # --- 2. Mahalanobis Calculations & Covariance-Weighted Contributions ---
    dist_mah = cdist(df[cols], centroids, metric='mahalanobis', VI=inv_cov)
    df['Mah_Cluster'] = np.argmin(dist_mah, axis=1)
    df['Mahalanobis_Residual'] = np.min(dist_mah, axis=1).round(2)
    
    mah_top_contribs = []
    mah_contrib_matrices = []
    
    for idx, row in df.iterrows():
        c_idx = int(row['Mah_Cluster'])
        diff = row[cols].values - centroids[c_idx]
        
        # Contribution of each variable j to Mahalanobis distance squared
        contribs = diff * np.dot(inv_cov, diff)
        contribs_abs = np.abs(contribs)
        
        top_var = cols[np.argmax(contribs_abs)]
        mah_top_contribs.append(top_var)
        mah_contrib_matrices.append(contribs_abs)
        
    df['Mah_Top_Contributor'] = mah_top_contribs
    df['Mah_Cluster'] = df['Mah_Cluster'] + 1
    
    df['Euc_Status'] = np.where(df['Euclidean_Residual'] > 12.0, "DEVIATION", "NORMAL")
    df['Mah_Status'] = np.where(df['Mahalanobis_Residual'] > 2.5, "DEVIATION", "NORMAL")
    
    # FIX: Explicitly cast mah_contrib_matrices to np.array before np.round
    contrib_df = pd.DataFrame(
        np.round(np.array(mah_contrib_matrices), 2), 
        columns=[f"{c}_contrib" for c in cols], 
        index=sets
    )
    
    return pd.concat([df, contrib_df], axis=1)

df_analysis = generate_and_analyze_test_data(inject_anomaly)

with st.expander("View Fixed Training Centroids", expanded=False):
    st.dataframe(pd.DataFrame(np.round(centroids, 2), columns=cols, index=["Mode 1", "Mode 2", "Mode 3"]))

# ---------------------------------------------------------
# OUTPUT TABS
# ---------------------------------------------------------
tab_euc, tab_mah = st.tabs(["Tab 1: Euclidean Residuals & Top Contributor", "Tab 2: Mahalanobis Residuals & Top Contributor"])

with tab_euc:
    st.header("Euclidean Residual Analysis")
    st.warning("⚠️ Unscaled Contribution: Variable with highest raw value difference (usually Flow or Temp) always dominates.")
    
    st.dataframe(
        df_analysis[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct', 'Euclidean_Residual', 'Euc_Top_Contributor', 'Euc_Status']],
        use_container_width=True
    )

with tab_mah:
    st.header("Mahalanobis Residual Analysis & Root Cause Identification")
    st.success("✅ Covariance-Weighted Contribution: Identifies true mechanical deviations (e.g., Pressure drop) regardless of engineering units.")
    
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.subheader("Residual & Highest Contributor Table")
        st.dataframe(
            df_analysis[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct', 'Mahalanobis_Residual', 'Mah_Top_Contributor', 'Mah_Status']],
            use_container_width=True
        )
        
    with col2:
        st.subheader("Mahalanobis Residuals by Data Set")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['red' if s == 'DEVIATION' else 'teal' for s in df_analysis['Mah_Status']]
        ax.bar(df_analysis.index, df_analysis['Mahalanobis_Residual'], color=colors)
        ax.axhline(2.5, color='red', linestyle='--', label='Threshold')
        ax.set_ylabel("Mahalanobis Residual")
        ax.set_xticklabels(df_analysis.index, rotation=45)
        ax.legend()
        st.pyplot(fig)

    st.subheader("Variable Contribution Breakdown (Mahalanobis)")
    st.caption("Detailed score showing how much each variable contributed to the total squared distance.")
    st.dataframe(
        df_analysis[['Flow_m3h_contrib', 'Press_bara_contrib', 'Temp_degC_contrib', 'Load_pct_contrib', 'Mah_Top_Contributor']],
        use_container_width=True
    )
