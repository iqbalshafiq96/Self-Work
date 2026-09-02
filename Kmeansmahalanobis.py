import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="AVEVA OMR - Auto-Optimized OMI (%) Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AVEVA OMR Engine: Auto-Optimized Cluster Assignment & OMI (%)")
st.caption(
    "Automated $k$ Selection via Silhouette Scoring, Nearest-Cluster Matching, and Percentage Deviations"
)


# ---------------------------------------------------------
# AUTOMATED K OPTIMIZER
# ---------------------------------------------------------
def find_optimal_k(X_scaled: np.ndarray, max_k: int = 10) -> tuple[int, dict]:
    """
    Evaluates Silhouette Scores across a range of k (2 to max_k)
    to automatically determine the optimal cluster count.
    """
    n_samples = X_scaled.shape[0]
    # Bound max_k so clusters don't become excessively small
    upper_bound = min(max_k, max(2, n_samples // 10))
    
    if upper_bound < 2:
        return 1, {1: 1.0}

    scores = {}
    best_k = 2
    best_score = -1.0

    for k in range(2, upper_bound + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Silhouette score requires at least 2 distinct clusters in output
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

    return best_k, scores


# ---------------------------------------------------------
# AVEVA OMR NEAREST-CLUSTER ENGINE CLASS
# ---------------------------------------------------------
class AVEVAOMRNearestClusterEngine:
    """
    AVEVA PRiSM / OMR Nearest-Cluster Empirical Engine with dynamic k optimization.
    """
    def __init__(self, shrink_factor: float = 1e-3):
        self.shrink_factor = shrink_factor
        self.clusters = {}
        self.feature_cols = []
        self.scaler = None
        self.optimal_k = 1
        self.silhouette_scores = {}

    def fit_baseline_clusters(
        self,
        X_raw: pd.DataFrame,
        feature_cols: list,
        user_k: int = 0,
        percentile: float = 99.0,
    ):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw[feature_cols])

        # 1. Automatic k optimization if user_k is 0 (Auto Mode)
        if user_k <= 0:
            best_k, scores = find_optimal_k(X_scaled, max_k=10)
            self.optimal_k = best_k
            self.silhouette_scores = scores
        else:
            self.optimal_k = user_k
            self.silhouette_scores = {}

        # 2. Partition baseline dataset using selected k
        if self.optimal_k > 1:
            kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
        else:
            cluster_labels = np.zeros(len(X_raw), dtype=int)

        self.clusters = {}
        for c_id in np.unique(cluster_labels):
            mask = cluster_labels == c_id
            X_c_scaled = X_scaled[mask]
            X_c_raw = X_raw[feature_cols].iloc[mask].values

            n_samples, p_features = X_c_scaled.shape

            z_centroid = np.mean(X_c_scaled, axis=0)
            raw_centroid = np.mean(X_c_raw, axis=0)

            # Ledoit-Wolf Covariance & Precision Matrix
            try:
                lw = LedoitWolf()
                cov_matrix = lw.fit(X_c_scaled).covariance_
            except Exception:
                cov_matrix = np.cov(X_c_scaled, rowvar=False)

            ridge = np.eye(p_features) * (
                self.shrink_factor * np.trace(cov_matrix) / max(1, p_features)
            )
            reg_cov = cov_matrix + ridge

            U, S, Vt = np.linalg.svd(reg_cov)
            max_s = np.max(S)
            S_inv = np.array([1.0 / s if (s / max_s) > 1e-5 else 0.0 for s in S])
            precision_matrix = np.dot(Vt.T, np.dot(np.diag(S_inv), U.T))

            # Raw Mahalanobis Distances for Cluster Baseline
            diffs = X_c_scaled - z_centroid
            dist_sq = np.sum(np.dot(diffs, precision_matrix) * diffs, axis=1)
            raw_distances = np.sqrt(np.maximum(0.0, dist_sq)) / np.sqrt(p_features)

            # Statistical Threshold (100% Benchmark Limit for this Cluster)
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

        # Identify Nearest Cluster Centroid
        best_cluster_id = 0
        min_euclidean_dist = float("inf")

        for c_id, baseline in self.clusters.items():
            dist = np.linalg.norm(z_sample - baseline["z_centroid"])
            if dist < min_euclidean_dist:
                min_euclidean_dist = dist
                best_cluster_id = c_id

        # Evaluate Distance against assigned Nearest Cluster
        cl = self.clusters[best_cluster_id]
        z_centroid = cl["z_centroid"]
        raw_centroid = cl["raw_centroid"]
        precision = cl["precision"]
        p_features = cl["p_features"]
        threshold_abs = cl["threshold_abs"]

        diff = z_sample - z_centroid
        raw_mahal_sq = float(np.dot(np.dot(diff, precision), diff.T))
        raw_mahal_dist = np.sqrt(max(0.0, raw_mahal_sq)) / np.sqrt(p_features)

        # OMI Distance in Percentage (%) [100% = Baseline Alarm Limit]
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
# DATA INGESTION PIPELINE
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_csv(url):
    df = pd.read_csv(url)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    return df


# Sidebar configuration
st.sidebar.header("Asset Baseline Configuration")

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

selected_dataset_name = st.sidebar.selectbox(
    "Select Asset Dataset Pair:", options=list(DATASET_MAP.keys()), index=0
)

cluster_mode = st.sidebar.radio(
    "Cluster Optimization Mode:",
    options=["Automatic (Silhouette Score)", "Manual Override"],
    index=0,
)

if cluster_mode == "Manual Override":
    manual_k = st.sidebar.slider(
        "Select Cluster Count (k):",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )
else:
    manual_k = 0  # 0 flags automatic search in engine

percentile_thresh = st.sidebar.slider(
    "Baseline Alarm Percentile Limit (%):",
    min_value=95.0,
    max_value=99.9,
    value=99.0,
    step=0.1,
)

TRAIN_URL = DATASET_MAP[selected_dataset_name]["train"]
TEST_URL = DATASET_MAP[selected_dataset_name]["test"]

# Load Data
try:
    raw_train_df = load_and_clean_csv(TRAIN_URL)
    raw_test_df = load_and_clean_csv(TEST_URL)

    feature_cols = [
        c
        for c in raw_train_df.select_dtypes(include=[np.number]).columns
        if c.lower() not in ["timestamp", "time", "date"]
    ]

    st.success(
        f"Ingested **{len(feature_cols)}** Process Tag Sensors across **{raw_train_df.shape[0]}** Training Samples."
    )
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Build / Calibrate Model
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("Calibrate Baseline Model", type="primary"):
        engine = AVEVAOMRNearestClusterEngine()
        engine.fit_baseline_clusters(
            X_raw=raw_train_df,
            feature_cols=feature_cols,
            user_k=manual_k,
            percentile=percentile_thresh,
        )
        st.session_state["omr_engine"] = engine
        st.session_state["active_dataset_name"] = selected_dataset_name
        st.success("Model Calibrated!")

# ---------------------------------------------------------
# LIVE EVALUATION & MONITORING
# ---------------------------------------------------------
if "omr_engine" in st.session_state:
    if st.session_state.get("active_dataset_name") != selected_dataset_name:
        st.warning("Dataset selection changed. Please click **Calibrate Baseline Model** again.")
    else:
        engine = st.session_state["omr_engine"]

        # Optimization Summary Banner
        if engine.silhouette_scores:
            st.info(
                f"Automated Optimization Selected **k = {engine.optimal_k}** operating clusters "
                f"(Peak Silhouette Score: **{engine.silhouette_scores[engine.optimal_k]:.4f}**)."
            )
            
            # Silhouette Score Evaluation Chart
            score_df = pd.DataFrame(
                list(engine.silhouette_scores.items()), columns=["k", "Silhouette Score"]
            )
            fig_scores = px.line(
                score_df,
                x="k",
                y="Silhouette Score",
                markers=True,
                title="Cluster Optimization Curve (Silhouette Method)",
            )
            fig_scores.add_vline(
                x=engine.optimal_k, line_dash="dash", line_color="green", annotation_text="Optimal k"
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        # Score Live Stream
        eval_results = []
        for i in range(len(raw_test_df)):
            sample = raw_test_df[feature_cols].iloc[i].values
            res = engine.score_live_sample(sample)
            eval_results.append(
                {
                    "Sample": i,
                    "Cluster": f"Cluster {res['nearest_cluster']}",
                    "OMI (%)": res["OMI_pct"],
                    "Status": "ALARM BREACH (>100%)"
                    if res["Is_Alert"]
                    else "Normal (≤100%)",
                }
            )

        results_df = pd.DataFrame(eval_results)

        st.markdown("---")
        st.subheader("1. Asset Overall Model Index (OMI) Trend (%)")

        # Key Metrics Row
        total_samples = len(results_df)
        total_alarms = (results_df["OMI (%)"] > 100.0).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Evaluated Timestamps", total_samples)
        m2.metric("Normal Operating Range (≤100%)", total_samples - total_alarms)
        m3.metric(
            "OMI Alarm Breaches (>100%)",
            total_alarms,
            delta=f"{round((total_alarms/total_samples)*100, 1)}% Off-Normal",
            delta_color="inverse",
        )

        # Plotly OMI Chart
        fig_omi = go.Figure()
        fig_omi.add_trace(
            go.Scatter(
                x=results_df["Sample"],
                y=results_df["OMI (%)"],
                mode="lines",
                name="Overall Model Index (%)",
                line=dict(color="#008080", width=1.5),
            )
        )
        fig_omi.add_trace(
            go.Scatter(
                x=results_df["Sample"],
                y=[100.0] * len(results_df),
                mode="lines",
                name="100% Alarm Threshold",
                line=dict(color="red", dash="dash", width=2),
            )
        )

        alarms = results_df[results_df["OMI (%)"] > 100.0]
        if not alarms.empty:
            fig_omi.add_trace(
                go.Scatter(
                    x=alarms["Sample"],
                    y=alarms["OMI (%)"],
                    mode="markers",
                    name="Threshold Breach",
                    marker=dict(color="red", size=6, symbol="x"),
                )
            )

        fig_omi.update_layout(
            xaxis_title="Sample Index",
            yaxis_title="OMI Distance (% of Baseline Limit)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_omi, use_container_width=True)

        # ---------------------------------------------------------
        # SENSOR RESIDUAL & CATCH DIAGNOSTICS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("2. Sensor Catch & Percentage Deviation Breakdown")

        all_samples = results_df["Sample"].tolist()
        alarm_samples = results_df[results_df["Status"] == "ALARM BREACH (>100%)"]["Sample"].tolist()
        default_idx = all_samples.index(alarm_samples[0]) if alarm_samples else 0

        sample_to_inspect = st.selectbox(
            "Select Sample Timestamp to Inspect:",
            options=all_samples,
            index=default_idx,
            format_func=lambda x: f"Sample #{x} {'⚠️ [ALARM]' if x in alarm_samples else '✅ [Normal]'}",
        )

        # Single point diagnostics
        raw_sample = raw_test_df[feature_cols].iloc[sample_to_inspect].values
        diag_res = engine.score_live_sample(raw_sample)

        diag_df = pd.DataFrame(
            {
                "Sensor Tag": feature_cols,
                "Actual Value (y)": raw_sample,
                "Nearest Cluster Target (ŷ)": diag_res["raw_predicted"],
                "Raw Residual (y - ŷ)": diag_res["raw_residuals"],
                "Sensor Residual (%)": diag_res["pct_residuals"],
                "Normalized Deviation (σ)": diag_res["std_residuals"],
                "Abs Deviation (|σ|)": np.abs(diag_res["std_residuals"]),
            }
        ).sort_values(by="Abs Deviation (|σ|)", ascending=False)

        st.info(
            f"Sample **#{sample_to_inspect}** mapped to **Cluster {diag_res['nearest_cluster']}** | Calculated OMI: **{diag_res['OMI_pct']:.2f}%**"
        )

        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            fig_pct = px.bar(
                diag_df.head(10),
                x="Sensor Residual (%)",
                y="Sensor Tag",
                orientation="h",
                title="Top Sensor Residuals (%)",
                color="Sensor Residual (%)",
                color_continuous_scale="RdBu_r",
            )
            fig_pct.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_pct, use_container_width=True)

        with c_chart2:
            fig_sigma = px.bar(
                diag_df.head(10),
                x="Normalized Deviation (σ)",
                y="Sensor Tag",
                orientation="h",
                title="Normalized Standard Deviations (σ)",
                color="Normalized Deviation (σ)",
                color_continuous_scale="Reds",
            )
            fig_sigma.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_sigma, use_container_width=True)

        st.write("**Full Tag Residual Breakdown Table:**")
        st.dataframe(
            diag_df.drop(columns=["Abs Deviation (|σ|)"]).style.format(
                {
                    "Actual Value (y)": "{:.4f}",
                    "Nearest Cluster Target (ŷ)": "{:.4f}",
                    "Raw Residual (y - ŷ)": "{:+.4f}",
                    "Sensor Residual (%)": "{:+.2f}%",
                    "Normalized Deviation (σ)": "{:+.2f}σ",
                }
            ),
            use_container_width=True,
            height=300,
        )
else:
    st.info("Click **Calibrate Baseline Model** above to train the empirical engine.")
