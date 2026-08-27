import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Equipment Residual Health & 3D Clusters", layout="wide")

st.title("Machinery Health Monitoring: 3D Cluster Cloud & Point Mapping")
st.markdown(
    "**Step 1:** Train fixed baseline operating centroids from historical data.  \n"
    "**Step 2:** Select a data set to see its position relative to the 3D operating cluster clouds."
)

# ---------------------------------------------------------
# STEP 1: FIX THE TRAINING BASELINE
# ---------------------------------------------------------
st.sidebar.header("1. Baseline Setup (Training Phase)")
baseline_seed = st.sidebar.number_input("Historical Baseline Seed", value=42, step=1)

@st.cache_data
def train_baseline_model(seed):
    np.random.seed(seed)
    # Generate 3D/4D operating clusters [Flow, Press, Temp, Load]
    m1 = np.random.multivariate_normal([120.0, 3.20, 65.0, 45.0], np.diag([25, 0.04, 9, 16]), 100)
    m2 = np.random.multivariate_normal([250.0, 4.00, 95.0, 78.0], np.diag([36, 0.09, 16, 25]), 100)
    m3 = np.random.multivariate_normal([340.0, 4.80, 125.0, 95.0], np.diag([49, 0.16, 25, 16]), 100)
    
    training_data = np.vstack([m1, m2, m3])
    cluster_labels = np.array([0]*100 + [1]*100 + [2]*100)
    cols = ['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct']
    
    centroids = np.array([m1.mean(axis=0), m2.mean(axis=0), m3.mean(axis=0)])
    cov_matrix = np.cov(training_data, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    return training_data, cluster_labels, centroids, inv_cov_matrix, cols

train_X, cluster_labels, centroids, inv_cov, cols = train_baseline_model(baseline_seed)

st.sidebar.subheader("2. Real-Time Condition Controls")
inject_anomaly = st.sidebar.checkbox("Inject Artificial Pressure Drop (Set_05)", value=True)

# ---------------------------------------------------------
# STEP 2: CALCULATE RESIDUALS AND TEST DATA
# ---------------------------------------------------------
def generate_test_data(inject_fault):
    np.random.seed(101)
    
    t1 = np.random.multivariate_normal([120.0, 3.20, 65.0, 45.0], np.diag([25, 0.04, 9, 16]), 3)
    t2 = np.random.multivariate_normal([250.0, 4.00, 95.0, 78.0], np.diag([36, 0.09, 16, 25]), 3)
    t3 = np.random.multivariate_normal([340.0, 4.80, 125.0, 95.0], np.diag([49, 0.16, 25, 16]), 3)
    
    if inject_fault:
        t2[1, 1] -= 0.85 # Pressure drop on Set_05
    
    test_X = np.vstack([t1, t2, t3])
    sets = [f"Set_{i+1:02d}" for i in range(len(test_X))]
    df = pd.DataFrame(np.round(test_X, 2), columns=cols, index=sets)
    
    # Distance to Centroids
    dist_mah = cdist(df[cols], centroids, metric='mahalanobis', VI=inv_cov)
    df['Mah_Cluster'] = np.argmin(dist_mah, axis=1)
    df['Mahalanobis_Residual'] = np.min(dist_mah, axis=1).round(2)
    df['Status'] = np.where(df['Mahalanobis_Residual'] > 2.5, "DEVIATION", "NORMAL")
    
    return df

df_test = generate_test_data(inject_anomaly)

# ---------------------------------------------------------
# INTERACTIVE 3D CLUSTER CLOUD PLOT
# ---------------------------------------------------------
st.subheader("3D Operating Space Visualization")

col_controls, col_plot = st.columns([1, 3])

with col_controls:
    st.markdown("### Point Inspector")
    selected_set = st.selectbox("Select Operating Set:", df_test.index, index=0)
    
    selected_row = df_test.loc[selected_set]
    assigned_mode = selected_row['Mah_Cluster'] + 1
    status = selected_row['Status']
    
    st.metric(label="Assigned Mode", value=f"Mode {assigned_mode}")
    st.metric(label="Mahalanobis Residual", value=selected_row['Mahalanobis_Residual'])
    
    if status == "DEVIATION":
        st.error(f"Status: {status}")
    else:
        st.success(f"Status: {status}")
        
    st.write("**Operating Parameters:**")
    st.json(selected_row[['Flow_m3h', 'Press_bara', 'Temp_degC', 'Load_pct']].to_dict())

with col_plot:
    # Build 3D Scatter Figure
    fig = go.Figure()
    
    # 1. Add Training Baseline Cluster Clouds
    mode_colors = ['#008080', '#4169E1', '#8A2BE2'] # Teal, Royal Blue, Blue Violet
    for i in range(3):
        mask = cluster_labels == i
        fig.add_trace(go.Scatter3d(
            x=train_X[mask, 0], # Flow
            y=train_X[mask, 1], # Pressure
            z=train_X[mask, 2], # Temperature
            mode='markers',
            name=f'Baseline Mode {i+1}',
            marker=dict(size=3, color=mode_colors[i], opacity=0.25),
            hoverinfo='none'
        ))

    # 2. Add Baseline Centroids
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers+text',
        name='Mode Centroids',
        text=[f"C{i+1}" for i in range(3)],
        textposition="top center",
        marker=dict(size=8, color='black', symbol='diamond')
    ))

    # 3. Add Selected Point Highlight
    sp_x = selected_row['Flow_m3h']
    sp_y = selected_row['Press_bara']
    sp_z = selected_row['Temp_degC']
    c_idx = int(selected_row['Mah_Cluster'])
    c_x, c_y, c_z = centroids[c_idx, :3]
    
    # Line connecting selected point to its assigned centroid
    fig.add_trace(go.Scatter3d(
        x=[sp_x, c_x],
        y=[sp_y, c_y],
        z=[sp_z, c_z],
        mode='lines',
        name='Distance Vector',
        line=dict(color='crimson' if status == 'DEVIATION' else 'black', width=4, dash='dash')
    ))

    # Highlight Marker for Selected Set
    fig.add_trace(go.Scatter3d(
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
                      "Flow: %{x} m3/h<br>" +
                      "Press: %{y} bara<br>" +
                      "Temp: %{z} °C<br>" +
                      f"Residual: {selected_row['Mahalanobis_Residual']}<extra></extra>"
    ))

    # Camera & Axes Styling
    fig.update_layout(
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

    st.plotly_chart(fig, use_container_width=True)
