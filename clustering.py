import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis, euclidean

st.set_page_config(page_title="Equipment Residual Health & 3D Mapping", layout="wide")

st.title("Machinery Health Monitoring: Residuals, Contributions & 3D Mapping")
st.markdown(
    "**Step 1:** Train fixed baseline operating centroids from historical data.  \n"
    "**Step 2:** Select an operating set and choose a reference cluster to inspect distance and feature contributions."
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
    m1 = np.random.multivariate_normal([120.0, 3.20, 65.0, 45.0], np.diag([25, 0.04, 9, 16]), 100)
    m2 = np.random.multivariate_normal([250.0, 4.00, 95.0, 78.0], np.diag([36, 0.09, 16, 25]), 100)
    m3 = np.random.multivariate_normal([340.0, 4.80, 125.0, 95.0], np.diag([49, 0.16, 25, 16]), 100)
    
    training_data = np.vstack([m1, m2, m3])
    cluster_labels = np.array([0]*100 + [1]*100 + [2]*100)
    cols = ['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct']
    
    # Extract Fixed Centroids & Inverse Covariance Matrix
    centroids = np.array([m1.mean(axis=0), m2.mean(axis=0), m3.mean(axis=0)])
    cov_matrix = np.cov(training_data, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    return training_data, cluster_labels, centroids, inv_cov_matrix, cols

train_X, cluster_labels, centroids, inv_cov, cols = train_baseline_model(baseline_seed)

pct_cols = [f"{c}_pct_contrib" for c in cols]
raw_cols = [f"{c}_raw_contrib" for c in cols]

st.sidebar.subheader("2. Real-Time Condition Controls")
inject_anomaly = st.sidebar.checkbox("Inject Artificial Pressure Drop (Set_05)", value=True)

# ---------------------------------------------------------
# STEP 2: CALCULATE RESIDUALS AND BOTH CONTRIB TYPES
# ---------------------------------------------------------
def generate_test_data(inject_fault):
    np.random.seed(101)
    
    t1 = np.random.multivariate_normal([120.0, 3.20, 65.0, 45.0], np.diag([25, 0.04, 9, 16]), 3)
    t2 = np.random.multivariate_normal([250.0, 4.00, 95.0, 78.0], np.diag([36, 0.09, 16, 25]), 3)
    t3 = np.random.multivariate_normal([340.0, 4.80, 125.0, 95.0], np.diag([49, 0.16, 25, 16]), 3)
    
    if inject_fault:
        # Induce a pressure anomaly on Set_05
        t2[1, 1] -= 0.85
    
    test_X = np.vstack([t1, t2, t3])
    sets = [f"Set_{i+1:02d}" for i in range(len(test_X))]
    return pd.DataFrame(np.round(test_X, 2), columns=cols, index=sets)

df_test = generate_test_data(inject_anomaly)

with st.expander("View Fixed Training Centroids", expanded=False):
    st.dataframe(pd.DataFrame(np.round(centroids, 2), columns=cols, index=["Mode 1", "Mode 2", "Mode 3"]))

# Helper function to compute Mahalanobis & Feature Contributions against a chosen centroid
def calculate_metrics_for_centroid(df, target_centroid):
    m_residuals, e_residuals = [], []
    raw_contribs, pct_contribs, top_contribs = [], [], []
    
    for idx, row in df[cols].iterrows():
        x = row.values
        diff = x - target_centroid
        
        # Distances
        d_mah = mahalanobis(x, target_centroid, inv_cov)
        d_euc = euclidean(x, target_centroid)
        
        m_residuals.append(round(d_mah, 2))
        e_residuals.append(round(d_euc, 2))
        
        # Contributions
        c_raw = np.abs(diff * np.dot(inv_cov, diff))
        total_sq = np.sum(c_raw)
        c_pct = (c_raw / total_sq * 100.0) if total_sq > 0 else np.zeros_like(c_raw)
        
        raw_contribs.append(c_raw)
        pct_contribs.append(c_pct)
        top_contribs.append(cols[np.argmax(c_pct)])
        
    res_df = df.copy()
    res_df['Mahalanobis_Residual'] = m_residuals
    res_df['Euclidean_Residual'] = e_residuals
    res_df['Mah_Top_Contributor'] = top_contribs
    res_df['Mah_Status'] = np.where(res_df['Mahalanobis_Residual'] > 2.5, "DEVIATION", "NORMAL")
    
    raw_df = pd.DataFrame(np.round(raw_contribs, 2), columns=raw_cols, index=df.index)
    pct_df = pd.DataFrame(np.round(pct_contribs, 2), columns=pct_cols, index=df.index)
    
    return pd.concat([res_df, raw_df, pct_df], axis=1)

# Default dataframe referenced to nearest cluster (for Tab 1 & Tab 2)
df_analysis = calculate_metrics_for_centroid(df_test, centroids[1]) # Default Mode 2 for quick view

# ---------------------------------------------------------
# OUTPUT TABS
# ---------------------------------------------------------
tab_euc, tab_mah, tab_3d = st.tabs([
    "Tab 1: Euclidean Residuals", 
    "Tab 2: Mahalanobis & Root Cause", 
    "Tab 3: Interactive 3D Cluster Mapping"
])

with tab_euc:
    st.header("Euclidean Residual Analysis")
    st.warning("⚠️ Unscaled Contribution: Variable with highest raw value difference (usually Flow or Temp) always dominates.")
    
    st.dataframe(
        df_analysis[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct', 'Euclidean_Residual', 'Mah_Top_Contributor', 'Mah_Status']],
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

    st.subheader("Variable Contribution Breakdown")
    st.caption("Raw distance squared score and normalized percentage contribution per parameter.")
    st.dataframe(
        df_analysis[raw_cols + pct_cols + ['Mah_Top_Contributor']],
        use_container_width=True
    )

with tab_3d:
    st.header("3D Operating Space Visualization")
    st.caption("Rotate and zoom the 3D space to see where the selected point lies relative to historical baseline clouds.")
    
    col_controls, col_plot = st.columns([1.2, 2.8])
    
    with col_controls:
        st.markdown("### Point Inspector & Reference Setup")
        
        # 1. Select Data Set
        selected_set = st.selectbox("1. Select Operating Set:", df_test.index, index=4)
        
        # 2. Manual Dropdown for Cluster Reference Selection
        cluster_options = {
            "Mode 1 (Low Load)": 0,
            "Mode 2 (Normal Operating)": 1,
            "Mode 3 (Peak Load)": 2
        }
        selected_mode_label = st.selectbox("2. Reference Cluster Mode:", list(cluster_options.keys()), index=1)
        ref_cluster_idx = cluster_options[selected_mode_label]
        target_centroid = centroids[ref_cluster_idx]
        
        # Recalculate metrics based on selected cluster
        df_tab3 = calculate_metrics_for_centroid(df_test, target_centroid)
        selected_row = df_tab3.loc[selected_set]
        status = selected_row['Mah_Status']
        
        st.markdown("---")
        st.metric(label=f"Mahalanobis Residual (vs {selected_mode_label[:6]})", value=selected_row['Mahalanobis_Residual'])
        st.metric(label="Top Root Cause", value=selected_row['Mah_Top_Contributor'])
        
        if status == "DEVIATION":
            st.error(f"Status: {status}")
        else:
            st.success(f"Status: {status}")
            
        st.markdown("**Feature Contribution Breakdown:**")
        
        # Dual-representation Table (Raw Score + Percent)
        breakdown_df = pd.DataFrame({
            'Variable': cols,
            'Raw Contrib (D²)': [selected_row[c] for c in raw_cols],
            'Contrib (%)': [f"{selected_row[c]:.1f}%" for c in pct_cols]
        })
        st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

    with col_plot:
        fig_3d = go.Figure()
        
        # 1. Baseline Clusters
        mode_colors = ['#008080', '#4169E1', '#8A2BE2']
        for i in range(3):
            mask = cluster_labels == i
            fig_3d.add_trace(go.Scatter3d(
                x=train_X[mask, 0],
                y=train_X[mask, 1],
                z=train_X[mask, 2],
                mode='markers',
                name=f'Baseline Mode {i+1}',
                marker=dict(size=3, color=mode_colors[i], opacity=0.20),
                hoverinfo='none'
            ))

        # 2. Baseline Centroids
        fig_3d.add_trace(go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode='markers+text',
            name='Mode Centroids',
            text=[f"C{i+1}" for i in range(3)],
            textposition="top center",
            marker=dict(size=8, color='black', symbol='diamond')
        ))

        # 3. Distance Vector to SELECTED Centroid
        sp_x = selected_row['Flow_m3h']
        sp_y = selected_row['Press_bara']
        sp_z = selected_row['Temp_degC']
        c_x, c_y, c_z = target_centroid[:3]
        
        fig_3d.add_trace(go.Scatter3d(
            x=[sp_x, c_x],
            y=[sp_y, c_y],
            z=[sp_z, c_z],
            mode='lines',
            name=f'Vector to C{ref_cluster_idx+1}',
            line=dict(color='crimson' if status == 'DEVIATION' else 'black', width=4, dash='dash')
        ))

        # Highlight Selected Set
        fig_3d.add_trace(go.Scatter3d(
            x=[sp_x],
            y=[sp_y],
            z=[sp_z],
            mode='markers+text',
            name=selected_set,
            text=[selected_set],
            textposition="top center",
            marker=dict(
                size=12,
                color='red' if status == 'DEVIATION' else 'limegreen',
                symbol='circle',
                line=dict(color='black', width=2)
            ),
            hovertemplate=f"<b>{selected_set}</b><br>" +
                          "Flow: %{x} m³/h<br>" +
                          "Press: %{y} bara<br>" +
                          "Temp: %{z} °C<br>" +
                          f"Residual (vs C{ref_cluster_idx+1}): {selected_row['Mahalanobis_Residual']}<extra></extra>"
        ))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title='Flow (m³/h)', backgroundcolor="#F8F9FA"),
                yaxis=dict(title='Pressure (bara)', backgroundcolor="#F8F9FA"),
                zaxis=dict(title='Temperature (°C)', backgroundcolor="#F8F9FA"),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=650,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig_3d, use_container_width=True)
