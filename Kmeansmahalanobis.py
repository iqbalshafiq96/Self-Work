import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="AVEVA OMR - Correlation Matrix & 10% Benchmark Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AVEVA OMR Engine: Correlation-Based Nearest Cluster (10% Scale)")
st.caption(
    "Empirical Pattern Matching via Standardization, Correlation Inversion (R⁻¹), and 10% OMR Alarm Scaling"
)


# ---------------------------------------------------------
# AUTOMATED K OPTIMIZER
# ---------------------------------------------------------
def find_optimal_k(X_scaled: np.ndarray, max_k: int = 20) -> tuple[int, dict]:
    n_samples = X_scaled.shape[0]
    upper_bound = min(max_k, max(2, n_samples // 5))

    if upper_bound < 2:
        return 1, {1: 1.0}

    scores = {}
    best_k = 2
    best_score = -1.0

    for k in range(2, upper_bound + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

    return best_k, scores


# ---------------------------------------------------------
# AVEVA CORRELATION MATRIX NEAREST-CLUSTER ENGINE
# ---------------------------------------------------------
class AVEVACorrelationClusterEngine:
    """
    AVEVA OMR Empirical Engine using:
    1. Standardization (z-score)
    2. Cluster Correlation Matrix R (via Covariance of Standardized Data)
    3. Inverse Correlation Matrix (R^-1) Mahalanobis Distance
    4. 10% Benchmark Scaling (Normal <= 10%, Alarm > 10%)
    """

    def __init__(self, ridge_factor: float = 1e-4):
        self.ridge_factor = ridge_factor
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

        # Step 1: Standardization (z-scores)
        X_scaled = self.scaler.fit_transform(X_raw[feature_cols])

        # Step 2: Optimal K search (Auto vs Manual)
        if user_k <= 0:
            best_k, scores = find_optimal_k(X_scaled, max_k=20)
            self.optimal_k = best_k
            self.silhouette_scores = scores
        else:
            self.optimal_k = user_k
            self.silhouette_scores = {}

        # Partition baseline dataset
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

            # Step 3: Correlation Matrix R of the cluster
            if n_samples > 1:
                R = np.corrcoef(X_c_scaled, rowvar=False)
            else:
                R = np.eye(p_features)

            # Handle edge case where corrcoef returns scalar or NaN
            if np.ndim(R) == 0 or np.isnan(R).any():
                R = np.eye(p_features)

            # High dynamic ridge adjustment for sparse/small clusters
            adaptive_ridge = max(self.ridge_factor, 1e-2 if n_samples < p_features else self.ridge_factor)
            R_reg = R + adaptive_ridge * np.eye(p_features)
            R_inv = np.linalg.pinv(R_reg)

            # Distance calculated using R_inv
            diffs = X_c_scaled - z_centroid
            dist_sq = np.sum(np.dot(diffs, R_inv) * diffs, axis=1)
            raw_distances = np.sqrt(np.maximum(0.0, dist_sq))

            # Step 4: 99th percentile baseline boundary
            d_99 = np.percentile(raw_distances, percentile) if len(raw_distances) > 0 else 1.0

            self.clusters[c_id] = {
                "z_centroid": z_centroid,
                "raw_centroid": raw_centroid,
                "R_inv": R_inv,
                "d_99": d_99,
                "p_features": p_features,
                "sample_count": n_samples,
            }

    def score_live_sample(self, raw_sample: np.ndarray):
        z_sample = self.scaler.transform(raw_sample.reshape(1, -1))[0]

        # Step 5: Find Nearest Cluster & Distance using Correlation Matrix R_inv
        min_distance = float("inf")
        best_cluster_id = 0

        for c_id, cl in self.clusters.items():
            diff = z_sample - cl["z_centroid"]
            d_m = np.sqrt(
                np.maximum(0.0, float(np.dot(np.dot(diff, cl["R_inv"]), diff.T)))
            )

            if d_m < min_distance:
                min_distance = d_m
                best_cluster_id = c_id

        # Target nearest cluster reference
        cl = self.clusters[best_cluster_id]
        d_99 = cl["d_99"]
        raw_centroid = cl["raw_centroid"]

        # Step 6: OMR Percentage Scaling
        omr_pct = (min_distance / (d_99 + 1e-6)) * 10.0

        # Residual Calculations
        raw_residuals = raw_sample - raw_centroid
        pct_residuals = (raw_residuals / (np.abs(raw_centroid) + 1e-6)) * 100.0
        std_residuals = raw_residuals / self.scaler.scale_

        return {
            "nearest_cluster": best_cluster_id,
            "raw_mahal_dist": min_distance,
            "d_99_threshold": d_99,
            "OMR_pct": omr_pct,
            "Is_Alert": omr_pct > 10.0,
            "raw_predicted": raw_centroid,
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


st.sidebar.header("Asset Baseline Configuration")

DATASET_MAP = {
    "Baseline Dataset 1 (NOC6_1 & Case_0)": {
        "train": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv",
        "test": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_0.csv",
    },
    "Turbine Baseline Dataset 2 (NOC_Turbine & Case_Turbine)": {
        "train": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Turbine.csv",
        "test": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Turbine.csv",
    },
}

selected_dataset_name = st.sidebar.selectbox(
    "Select Asset Dataset Pair:", options=list(DATASET_MAP.keys()), index=1
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
        max_value=30,  # Increased upper limit
        value=3,
        step=1,
    )
else:
    manual_k = 0

percentile_thresh = st.sidebar.slider(
    "Baseline 10% Scale Boundary Percentile:",
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
        f"Ingested **{len(feature_cols)}** Tag Sensors across **{raw_train_df.shape[0]}** Training Baseline Timestamps."
    )
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Build / Calibrate Model
col_btn1, _ = st.columns([1, 4])
with col_btn1:
    if st.button("Calibrate Baseline Model", type="primary"):
        engine = AVEVACorrelationClusterEngine()
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
        st.warning("Dataset selection changed. Click **Calibrate Baseline Model** again.")
    else:
        engine = st.session_state["omr_engine"]

        if engine.silhouette_scores:
            st.info(
                f"Automated Optimization Selected **k = {engine.optimal_k}** operating clusters "
                f"(Peak Silhouette Score: **{engine.silhouette_scores[engine.optimal_k]:.4f}**)."
            )

        # Score Live Data
        eval_results = []
        for i in range(len(raw_test_df)):
            sample = raw_test_df[feature_cols].iloc[i].values
            res = engine.score_live_sample(sample)
            eval_results.append(
                {
                    "Sample": i,
                    "Cluster": f"Cluster {res['nearest_cluster']}",
                    "OMR (%)": res["OMR_pct"],
                    "Status": "ALARM BREACH (>10%)"
                    if res["Is_Alert"]
                    else "Normal (≤10%)",
                }
            )

        results_df = pd.DataFrame(eval_results)

        st.markdown("---")
        st.subheader("1. Overall Model Residual (OMR) Trend (%)")

        total_samples = len(results_df)
        total_alarms = (results_df["OMR (%)"] > 10.0).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Evaluated Timestamps", total_samples)
        m2.metric("Normal Operating Range (≤10%)", total_samples - total_alarms)
        m3.metric(
            "OMR Alarm Breaches (>10%)",
            total_alarms,
            delta=f"{round((total_alarms/total_samples)*100, 1)}% Off-Normal",
            delta_color="inverse",
        )

        # Plotly OMR Trend Chart
        fig_omr = go.Figure()
        fig_omr.add_trace(
            go.Scatter(
                x=results_df["Sample"],
                y=results_df["OMR (%)"],
                mode="lines",
                name="Overall Model Residual (%)",
                line=dict(color="#008080", width=1.5),
            )
        )
        fig_omr.add_trace(
            go.Scatter(
                x=results_df["Sample"],
                y=[10.0] * len(results_df),
                mode="lines",
                name="10% Alarm Threshold",
                line=dict(color="red", dash="dash", width=2),
            )
        )

        alarms = results_df[results_df["OMR (%)"] > 10.0]
        if not alarms.empty:
            fig_omr.add_trace(
                go.Scatter(
                    x=alarms["Sample"],
                    y=alarms["OMR (%)"],
                    mode="markers",
                    name="Threshold Breach (>10%)",
                    marker=dict(color="red", size=6, symbol="x"),
                )
            )

        fig_omr.update_layout(
            xaxis_title="Sample Index",
            yaxis_title="OMR (%) [10% = Baseline Alarm Boundary]",
            hovermode="x unified",
        )
        st.plotly_chart(fig_omr, use_container_width=True)

        # ---------------------------------------------------------
        # SENSOR RESIDUAL DIAGNOSTICS
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("2. Sensor Residual & Catch Diagnostics")

        all_samples = results_df["Sample"].tolist()
        alarm_samples = results_df[results_df["Status"] == "ALARM BREACH (>10%)"]["Sample"].tolist()
        default_idx = all_samples.index(alarm_samples[0]) if alarm_samples else 0

        sample_to_inspect = st.selectbox(
            "Select Timestamp to Inspect:",
            options=all_samples,
            index=default_idx,
            format_func=lambda x: f"Sample #{x} {'⚠️ [ALARM >10%]' if x in alarm_samples else '✅ [Normal ≤10%]'}",
        )

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
            f"Sample **#{sample_to_inspect}** mapped to **Cluster {diag_res['nearest_cluster']}** | Calculated OMR: **{diag_res['OMR_pct']:.2f}%**"
        )

        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            fig_pct = px.bar(
                diag_df.head(10),
                x="Sensor Residual (%)",
                y="Sensor Tag",
                orientation="h",
                title="Top 10 Sensor Residuals (%)",
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
                title="Top 10 Normalized Deviations (σ)",
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
