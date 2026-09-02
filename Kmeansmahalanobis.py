import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="AVEVA OMR - Nearest Cluster OMI (%) Engine",
    layout="wide",
)

st.title("AVEVA OMR Engine: Nearest Cluster Assignment & OMI (%)")
st.caption("Dynamic Euclidean/Mahalanobis Nearest-Cluster Matching with Percentage OMI Normalization")

# ---------------------------------------------------------
# AVEVA OMR NEAREST-CLUSTER ENGINE
# ---------------------------------------------------------
class AVEVAOMRNearestClusterEngine:
    def __init__(self, n_clusters: int = 3, shrink_factor: float = 1e-3):
        self.n_clusters = n_clusters
        self.shrink_factor = shrink_factor
        self.clusters = {}
        self.feature_cols = []
        self.scaler = None

    def fit_baseline_clusters(self, X_raw: pd.DataFrame, feature_cols: list, percentile: float = 99.0):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw[feature_cols])

        # Subdivide training data into operating profile clusters using mini-batch / k-means allocation
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        for c_id in range(self.n_clusters):
            mask = cluster_labels == c_id
            X_c_scaled = X_scaled[mask]
            X_c_raw = X_raw[feature_cols].iloc[mask].values

            n_samples, p_features = X_c_scaled.shape

            z_centroid = np.mean(X_c_scaled, axis=0)
            raw_centroid = np.mean(X_c_raw, axis=0)

            # Ledoit-Wolf Precision Matrix
            try:
                lw = LedoitWolf()
                cov_matrix = lw.fit(X_c_scaled).covariance_
            except Exception:
                cov_matrix = np.cov(X_c_scaled, rowvar=False)

            ridge = np.eye(p_features) * (self.shrink_factor * np.trace(cov_matrix) / max(1, p_features))
            reg_cov = cov_matrix + ridge

            U, S, Vt = np.linalg.svd(reg_cov)
            max_s = np.max(S)
            S_inv = np.array([1.0 / s if (s / max_s) > 1e-5 else 0.0 for s in S])
            precision_matrix = np.dot(Vt.T, np.dot(np.diag(S_inv), U.T))

            # Raw Mahalanobis Distances for Baseline Cluster Calibration
            diffs = X_c_scaled - z_centroid
            dist_sq = np.sum(np.dot(diffs, precision_matrix) * diffs, axis=1)
            raw_distances = np.sqrt(np.maximum(0.0, dist_sq)) / np.sqrt(p_features)

            # Statistical 99% Threshold (100% Benchmark baseline)
            threshold_abs = np.percentile(raw_distances, percentile)

            self.clusters[c_id] = {
                "z_centroid": z_centroid,
                "raw_centroid": raw_centroid,
                "precision": precision_matrix,
                "threshold_abs": threshold_abs,
                "p_features": p_features,
                "sample_count": n_samples,
            }

    def score_live_sample(self, raw_sample: np.ndarray):
        z_sample = self.scaler.transform(raw_sample.reshape(1, -1))[0]

        # 1. Identify Nearest Cluster Centroid in Scaled Feature Space
        best_cluster_id = 0
        min_euclidean_dist = float("inf")

        for c_id, cl baseline in self.clusters.items():
            dist = np.linalg.norm(z_sample - baseline["z_centroid"])
            if dist < min_euclidean_dist:
                min_euclidean_dist = dist
                best_cluster_id = c_id

        # 2. Evaluate Mahalanobis Distance against assigned Nearest Cluster
        cl = self.clusters[best_cluster_id]
        z_centroid = cl["z_centroid"]
        raw_centroid = cl["raw_centroid"]
        precision = cl["precision"]
        p_features = cl["p_features"]
        threshold_abs = cl["threshold_abs"]

        diff = z_sample - z_centroid
        raw_mahal_sq = float(np.dot(np.dot(diff, precision), diff.T))
        raw_mahal_dist = np.sqrt(max(0.0, raw_mahal_sq)) / np.sqrt(p_features)

        # 3. Calculate Distance in Percentage (%) Relative to Calibrated Threshold (100% = Baseline Alarm Boundary)
        omi_pct = (raw_mahal_dist / (threshold_abs + 1e-6)) * 100.0

        # Sensor Residual Calculations
        raw_predicted = raw_centroid
        raw_residuals = raw_sample - raw_predicted
        pct_residuals = (raw_residuals / (np.abs(raw_predicted) + 1e-6)) * 100.0
        std_residuals = raw_residuals / self.scaler.scale_

        return {
            "nearest_cluster": best_cluster_id,
            "raw_mahal_dist": raw_mahal_dist,
            "threshold_abs": threshold_abs,
            "OMI_pct": omi_pct,
            "Is_Alert": omi_pct > 100.0,
            "raw_predicted": raw_predicted,
            "raw_residuals": raw_residuals,
            "pct_residuals": pct_residuals,
            "std_residuals": std_residuals,
        }

# ---------------------------------------------------------
# DATA LOADER & PIPELINE UTILS
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_csv(url):
    df = pd.read_csv(url)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    return df

DATASET_MAP = {
    "Baseline Dataset 1 (NOC6_1 & Case_0)": {
        "train": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv",
        "test": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_0.csv",
    },
    "Turbine Baseline Dataset 2 (NOC_Turbine & Case_Turbine)": {
        "train": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Turbine_r1.csv",
        "test": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Turbine.csv",
    },
}

selected_dataset_name = st.sidebar.selectbox("Select Asset Datasets:", list(DATASET_MAP.keys()))
n_clusters_input = st.sidebar.slider("Operating State Clusters (k):", 1, 5, 3)

TRAIN_URL = DATASET_MAP[selected_dataset_name]["train"]
TEST_URL = DATASET_MAP[selected_dataset_name]["test"]

# ---------------------------------------------------------
# APPLICATION WORKFLOW
# ---------------------------------------------------------
raw_train_df = load_and_clean_csv(TRAIN_URL)
raw_test_df = load_and_clean_csv(TEST_URL)
feature_cols = [c for c in raw_train_df.select_dtypes(include=[np.number]).columns if c.lower() not in ["timestamp", "time", "date"]]

st.write(f"Ingested **{len(feature_cols)}** Process Sensors across **{raw_train_df.shape[0]}** Training Samples.")

if st.button("Calibrate Nearest-Cluster OMR Baseline"):
    engine = AVEVAOMRNearestClusterEngine(n_clusters=n_clusters_input)
    engine.fit_baseline_clusters(X_raw=raw_train_df, feature_cols=feature_cols)
    st.session_state["omr_engine"] = engine
    st.success("Nearest Cluster Model Trained Successfully!")

if "omr_engine" in st.session_state:
    engine = st.session_state["omr_engine"]
    
    # Process Live Data Stream
    eval_results = []
    for i in range(len(raw_test_df)):
        sample = raw_test_df[feature_cols].iloc[i].values
        res = engine.score_live_sample(sample)
        eval_results.append({
            "Sample": i,
            "Cluster": f"Cluster {res['nearest_cluster']}",
            "OMI (%)": res["OMI_pct"],
            "Status": "ALARM BREACH (>100%)" if res["Is_Alert"] else "Normal (≤100%)",
        })

    results_df = pd.DataFrame(eval_results)

    # Chart OMI % Trend
    st.subheader("Asset Health Evaluation: Overall Model Index Distance (%)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df["Sample"], y=results_df["OMI (%)"], mode="lines", name="OMI (%) Distance", line=dict(color="#008080")))
    fig.add_trace(go.Scatter(x=results_df["Sample"], y=[100.0]*len(results_df), mode="lines", name="100% Alarm Threshold", line=dict(color="red", dash="dash")))

    alarms = results_df[results_df["OMI (%)"] > 100.0]
    if not alarms.empty:
        fig.add_trace(go.Scatter(x=alarms["Sample"], y=alarms["OMI (%)"], mode="markers", name="Breach Point", marker=dict(color="red", size=6, symbol="x")))

    fig.update_layout(xaxis_title="Sample Index", yaxis_title="OMI Distance (% of Baseline Threshold)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown Table for Single Point
    st.subheader("Nearest-Cluster Sensor Residual Breakdown (%)")
    sample_to_inspect = st.selectbox("Select Sample Index to Inspect:", results_df["Sample"].tolist())
    
    raw_sample = raw_test_df[feature_cols].iloc[sample_to_inspect].values
    diag_res = engine.score_live_sample(raw_sample)

    diag_df = pd.DataFrame({
        "Sensor Tag": feature_cols,
        "Actual Value (y)": raw_sample,
        "Nearest Cluster Target (ŷ)": diag_res["raw_predicted"],
        "Raw Residual (y - ŷ)": diag_res["raw_residuals"],
        "Sensor Residual (%)": diag_res["pct_residuals"],
        "Normalized Deviation (σ)": diag_res["std_residuals"],
    }).sort_values(by="Normalized Deviation (σ)", key=abs, ascending=False)

    st.write(f"Assigned to **Cluster {diag_res['nearest_cluster']}** | Computed OMI: **{diag_res['OMI_pct']:.2f}%**")
    st.dataframe(diag_df.style.format({
        "Actual Value (y)": "{:.4f}",
        "Nearest Cluster Target (ŷ)": "{:.4f}",
        "Raw Residual (y - ŷ)": "{:+.4f}",
        "Sensor Residual (%)": "{:+.2f}%",
        "Normalized Deviation (σ)": "{:+.2f}σ",
    }), use_container_width=True)
