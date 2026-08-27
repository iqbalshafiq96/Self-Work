import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# Page Config
st.set_page_config(page_title="Model Clustering Comparison", layout="wide")

st.title("Model Benchmarking: Euclidean vs. Mahalanobis Clustering")
st.markdown(
    "Compare how distance metrics evaluate model performance across correlated features "
    "(e.g., Accuracy vs. Latency)."
)

# Sidebar Options
st.sidebar.header("Dataset Configuration")
n_per_cluster = st.sidebar.slider("Models per Tier", min_value=3, max_value=8, value=4)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

@st.cache_data
def generate_model_data(n, seed):
    np.random.seed(seed)
    
    # Generate 3 model tiers (Edge, Mid, Foundation)
    cov1 = [[4, 15], [15, 80]]
    c1 = np.random.multivariate_normal([75, 20], cov1, n)

    cov2 = [[5, 25], [25, 150]]
    c2 = np.random.multivariate_normal([85, 80], cov2, n)

    cov3 = [[3, 20], [20, 180]]
    c3 = np.random.multivariate_normal([95, 180], cov3, n)

    X = np.vstack([c1, c2, c3])
    names = [f"Model_{chr(65+i)}" for i in range(3 * n)]
    
    df = pd.DataFrame(np.round(X, 2), columns=['Accuracy', 'Latency'], index=names)
    
    # 1. Euclidean Clustering
    dist_euc = pdist(df[['Accuracy', 'Latency']], metric='euclidean')
    Z_euc = linkage(dist_euc, method='ward')
    df['Cluster_Euclidean'] = fcluster(Z_euc, t=3, criterion='maxclust')
    
    # 2. Mahalanobis Clustering
    cov_matrix = np.cov(df[['Accuracy', 'Latency']].values, rowvar=False)
    inv_cov = np.linalg.pinv(cov_matrix)
    dist_mah = pdist(df[['Accuracy', 'Latency']].values, metric='mahalanobis', VI=inv_cov)
    Z_mah = linkage(dist_mah, method='average')
    df['Cluster_Mahalanobis'] = fcluster(Z_mah, t=3, criterion='maxclust')
    
    return df, squareform(dist_euc), squareform(dist_mah)

df, mat_euc, mat_mah = generate_model_data(n_per_cluster, random_seed)

# Show raw data section
with st.expander("View Raw Benchmark Dataset", expanded=False):
    st.dataframe(df, use_container_width=True)

# Main Output Tabs
tab_euc, tab_mah = st.tabs(["Euclidean Clustering", "Mahalanobis Clustering"])

# --- TAB 1: EUCLIDEAN ---
with tab_euc:
    st.header("Euclidean Distance Clustering")
    st.caption("Treats all dimensions independently; highly sensitive to scale differences (e.g., Latency vs Accuracy).")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Cluster Assignments")
        st.dataframe(df[['Accuracy', 'Latency', 'Cluster_Euclidean']], use_container_width=True)
        
    with col2:
        st.subheader("Scatter Visualization")
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(df['Latency'], df['Accuracy'], c=df['Cluster_Euclidean'], cmap='viridis', s=80)
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Euclidean Clusters")
        st.pyplot(fig)

    st.subheader("Pairwise Distance Matrix (Euclidean)")
    st.dataframe(pd.DataFrame(mat_euc, index=df.index, columns=df.index).round(2), use_container_width=True)

# --- TAB 2: MAHALANOBIS ---
with tab_mah:
    st.header("Mahalanobis Distance Clustering")
    st.caption("Accounts for feature variance and inter-metric correlation (covariance matrix inverse).")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Cluster Assignments")
        st.dataframe(df[['Accuracy', 'Latency', 'Cluster_Mahalanobis']], use_container_width=True)
        
    with col2:
        st.subheader("Scatter Visualization")
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(df['Latency'], df['Accuracy'], c=df['Cluster_Mahalanobis'], cmap='plasma', s=80)
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Mahalanobis Clusters")
        st.pyplot(fig)

    st.subheader("Pairwise Distance Matrix (Mahalanobis)")
    st.dataframe(pd.DataFrame(mat_mah, index=df.index, columns=df.index).round(2), use_container_width=True)
