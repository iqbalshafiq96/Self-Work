import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Equipment Residual Health & 3D Mapping", layout="wide")

st.title("Machinery Health Monitoring: Residuals, Contributions & 3D Mapping")
st.markdown(
    "**Step 1:** Train fixed baseline operating centroids from historical data.  \n"
    "**Step 2:** Compute real-time residuals, isolate feature contributions, and map points in 3D cluster space."
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
        contribs_abs = np.abs(contribs).tolist()  # Explicit conversion to float list
        
        top_var = cols[np.argmax(contribs_abs)]
        mah_top_contribs.append(top_var)
        mah_contrib_matrices.append(contribs_abs)
        
    df['Mah_Top_Contributor'] = mah_top_contribs
    df['Mah_Cluster'] = df['Mah_Cluster'] + 1
    
    df['Euc_Status'] = np.where(df['Euclidean_Residual'] > 12.0, "DEVIATION", "NORMAL")
    df['Mah_Status'] = np.where(df['Mahalanobis_Residual'] > 2.5, "DEVIATION", "NORMAL")
    
    # Cast to float matrix to prevent Python 3.14/NumPy 2.x object array errors
    contrib_df = pd.DataFrame(
        np.round(np.array(mah_contrib_matrices, dtype=float), 2), 
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
tab_euc, tab_mah, tab_3d = st.tabs([
    "Tab 1: Euclidean Residuals", 
    "Tab 2: Mahalanobis & Root Cause", 
    "Tab 3: Interactive 3D Cluster Mapping"
])

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

with tab_3d:
    st.header("3D Operating Space Visualization")
    st.caption("Rotate and zoom the 3D space to see where the selected point lies relative to historical baseline clouds.")
    
    col_controls, col_plot = st.columns([1, 3])
    
    with col_controls:
        st.markdown("### Point Inspector")
        selected_set = st.selectbox("Select Operating Set:", df_analysis.index, index=0)
        
        selected_row = df_analysis.loc[selected_set]
        assigned_mode = selected_row['Mah_Cluster']
        status = selected_row['Mah_Status']
        
        st.metric(label="Assigned Mode", value=f"Mode {assigned_mode}")
        st.metric(label="Mahalanobis Residual", value=selected_row['Mahalanobis_Residual'])
        st.metric(label="Top Root Cause", value=selected_row['Mah_Top_Contributor'])
        
        if status == "DEVIATION":
            st.error(f"Status: {status}")
        else:
            st.success(f"Status: {status}")
            
        st.write("**Operating Parameters:**")
        st.json(selected_row[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct']].to_dict())

    with col_plot:
        fig_3d = go.Figure()
        
        # 1. Add Training Baseline Cluster Clouds
        mode_colors = ['#008080', '#4169E1', '#8A2BE2']  # Petronas Teal, Royal Blue, Blue Violet
        for i in range(3):
            mask = cluster_labels == i
            fig_3d.add_trace(go.Scatter3d(
                x=train_X[mask, 0],  # Flow
                y=train_X[mask, 1],  # Pressure
                z=train_X[mask, 2],  # Temperature
                mode='markers',
                name=f'Baseline Mode {i+1}',
                marker=dict(size=3, color=mode_colors[i], opacity=0.25),
                hoverinfo='none'
            ))

        # 2. Add Baseline Centroids
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

        # 3. Add Selected Point Highlight & Distance Vector
        sp_x = selected_row['Flow_m3h']
        sp_y = selected_row['Press_bara']
        sp_z = selected_row['Temp_degC']
        c_idx = int(assigned_mode) - 1
        c_x, c_y, c_z = centroids[c_idx, :3]
        
        # Line connecting selected point to its assigned centroid
        fig_3d.add_trace(go.Scatter3d(
            x=[sp_x, c_x],
            y=[sp_y, c_y],
            z=[sp_z, c_z],
            mode='lines',
            name='Distance Vector',
            line=dict(color='crimson' if status == 'DEVIATION' else 'black', width=4, dash='dash')
        ))

        # Highlight Marker for Selected Set
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
                          f"Residual: {selected_row['Mahalanobis_Residual']}<extra></extra>"
        ))

        # Camera & Axes Styling
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
