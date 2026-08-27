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
    "**Step 2:** Compute real-time residuals, isolate percentage feature contributions, and map points in 3D cluster space."
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

# Pre-define suffix lists so column indexing never fails
pct_cols = [f"{c}_pct_contrib" for c in cols]
raw_cols = [f"{c}_raw_contrib" for c in cols]

st.sidebar.subheader("2. Real-Time Condition Controls")
inject_anomaly = st.sidebar.checkbox("Inject Artificial Pressure Drop (Set_05)", value=True)

# ---------------------------------------------------------
# STEP 2: CALCULATE RESIDUALS AND PERCENT CONTRIBUTIONS
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
    
    # --- 2. Mahalanobis Calculations & Covariance-Weighted Percent Contributions ---
    dist_mah = cdist(df[cols], centroids, metric='mahalanobis', VI=inv_cov)
    df['Mah_Cluster'] = np.argmin(dist_mah, axis=1)
    df['Mahalanobis_Residual'] = np.min(dist_mah, axis=1).round(2)
    
    mah_top_contribs = []
    mah_contrib_raw_matrices = []
    mah_contrib_pct_matrices = []
    
    for idx, row in df.iterrows():
        c_idx = int(row['Mah_Cluster'])
        diff = row[cols].values - centroids[c_idx]
        
        # Absolute contribution of each variable j to squared distance
        contribs_abs = np.abs(diff * np.dot(inv_cov, diff))
        total_dist_sq = np.sum(contribs_abs)
        
        # Calculate percentage contribution
        if total_dist_sq > 0:
            contribs_pct = (contribs_abs / total_dist_sq) * 100.0
        else:
            contribs_pct = np.zeros_like(contribs_abs)
            
        top_var = cols[np.argmax(contribs_pct)]
        mah_top_contribs.append(top_var)
        mah_contrib_raw_matrices.append(contribs_abs)
        mah_contrib_pct_matrices.append(contribs_pct)
        
    df['Mah_Top_Contributor'] = mah_top_contribs
    df['Mah_Cluster'] = df['Mah_Cluster'] + 1
    
    df['Euc_Status'] = np.where(df['Euclidean_Residual'] > 12.0, "DEVIATION", "NORMAL")
    df['Mah_Status'] = np.where(df['Mahalanobis_Residual'] > 2.5, "DEVIATION", "NORMAL")
    
    # Raw contribution score dataframe
    raw_contrib_df = pd.DataFrame(
        np.round(np.array(mah_contrib_raw_matrices, dtype=float), 2), 
        columns=raw_cols, 
        index=sets
    )

    # Percentage contribution dataframe
    pct_contrib_df = pd.DataFrame(
        np.round(np.array(mah_contrib_pct_matrices, dtype=float), 2), 
        columns=pct_cols, 
        index=sets
    )
    
    return pd.concat([df, raw_contrib_df, pct_contrib_df], axis=1)

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

    st.subheader("Variable Contribution Breakdown (%)")
    st.caption("Percentage contribution of each parameter to the total Mahalanobis distance squared.")
    
    # Safe indexing using pre-defined pct_cols list
    st.dataframe(
        df_analysis[pct_cols + ['Mah_Top_Contributor']],
        use_container_width=True
    )

with tab_3d:
    st.header("3D Operating Space Visualization")
    st.caption("Rotate and zoom the 3D space to see where the selected point lies relative to historical baseline clouds.")
    
    col_controls, col_plot = st.columns([1.1, 2.9])
    
    with col_controls:
        st.markdown("### Point Inspector")
        selected_set = st.selectbox("Select Operating Set:", df_analysis.index, index=4)
        
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
            
        st.markdown("**Contribution Breakdown (%):**")
        
        pct_values = [selected_row[c] for c in pct_cols]
        feature_colors = ['#008080', '#D9534F', '#4169E1', '#F0AD4E']
        
        fig_bar = go.Figure()
        for idx, col_name in enumerate(cols):
            fig_bar.add_trace(go.Bar(
                y=[selected_set],
                x=[pct_values[idx]],
                name=col_name,
                orientation='h',
                marker=dict(color=feature_colors[idx]),
                hovertemplate=f"{col_name}: %{{x:.1f}}%<extra></extra>"
            ))
            
        fig_bar.update_layout(
            barmode='stack',
            height=130,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title="Contribution %", range=[0, 100], showgrid=False),
            yaxis=dict(showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.8, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

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
                marker=dict(size=3, color=mode_colors[i], opacity=0.25),
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

        # 3. Selected Point Vector
        sp_x = selected_row['Flow_m3h']
        sp_y = selected_row['Press_bara']
        sp_z = selected_row['Temp_degC']
        c_idx = int(assigned_mode) - 1
        c_x, c_y, c_z = centroids[c_idx, :3]
        
        fig_3d.add_trace(go.Scatter3d(
            x=[sp_x, c_x],
            y=[sp_y, c_y],
            z=[sp_z, c_z],
            mode='lines',
            name='Distance Vector',
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
                          f"Residual: {selected_row['Mahalanobis_Residual']}<extra></extra>"
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
