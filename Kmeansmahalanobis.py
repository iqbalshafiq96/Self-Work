import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gg
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Rotating Asset Reliability Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Rotating Asset Reliability Monitor")
st.caption(
    "Regime-based Clustering & Multi-Dimensional Mahalanobis Residual Scoring"
)

# ---------------------------------------------------------
# SIDEBAR DATASET SELECTION
# ---------------------------------------------------------
st.sidebar.header("Data Source Configuration")

DATASET_MAP = {
    "Default Baseline Dataset (NOC6_1 & Case_0)": {
        "train": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv",
        "test": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_0.csv",
    },
    "Turbine Baseline Dataset (NOC_Turbine & Case_Turbine)": {
        "train": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Turbine.csv",
        "test": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Turbine.csv",
    },
}

selected_dataset_name = st.sidebar.selectbox(
    "Select Baseline & Test Pair:",
    options=list(DATASET_MAP.keys()),
    index=0,
)

TRAIN_URL = DATASET_MAP[selected_dataset_name]["train"]
TEST_URL = DATASET_MAP[selected_dataset_name]["test"]

# ---------------------------------------------------------
# TOP-LEVEL PHASE SELECTION
# ---------------------------------------------------------
phase = st.radio(
    "Select Workflow Phase:",
    options=["Phase 1: Offline Training", "Phase 2: Online Monitoring"],
    horizontal=True,
)

st.markdown("---")

# ---------------------------------------------------------
# PHASE 1: OFFLINE TRAINING
# ---------------------------------------------------------
if phase == "Phase 1: Offline Training":
    st.header("Phase 1: Offline Baseline Training")

    t_data, t_cluster, t_metrics, t_summary = st.tabs(
        [
            "1. Data Ingestion & Preprocessing",
            "2. Operational Regime Clustering",
            "3. Regime-Specific Metrics & Radius",
            "4. Pipeline Training Summary",
        ]
    )

    # STEP 1: DATA INGESTION
    with t_data:
        st.subheader("Step 1: Baseline Data Load & Standardization")
        st.write(f"Active Training Source: `{TRAIN_URL}`")

        @st.cache_data
        def load_train_data(url):
            return pd.read_csv(url)

        try:
            raw_train_df = load_train_data(TRAIN_URL)

            numeric_cols = raw_train_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            feature_cols = [c for c in numeric_cols if c.lower() != "timestamp"]

            st.success(
                f"Data loaded successfully: {raw_train_df.shape[0]} rows × {len(feature_cols)} process features."
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**Feature Selection:**")
                st.dataframe(pd.DataFrame({"Tag Name": feature_cols}), height=250)
            with col2:
                st.write("**Raw Data Preview:**")
                st.dataframe(raw_train_df.head(), height=250)

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(raw_train_df[feature_cols])

            st.session_state["raw_train_df"] = raw_train_df
            st.session_state["feature_cols"] = feature_cols
            st.session_state["scaler"] = scaler
            st.session_state["scaled_train"] = scaled_train
            st.session_state["active_dataset_name"] = selected_dataset_name

        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

    # STEP 2: CLUSTERING
    with t_cluster:
        st.subheader("Step 2: Operational Regime Partitioning")

        if "scaled_train" not in st.session_state:
            st.info("Please load and scale data in Step 1 first.")
        else:
            scaled_train = st.session_state["scaled_train"]

            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                algo_choice = st.selectbox(
                    "Select Clustering Algorithm:",
                    options=["KMeans", "GaussianMixture (GMM)"],
                )
            with col_cfg2:
                n_clusters = st.slider(
                    "Select Number of Operational Regimes (K):",
                    min_value=2,
                    max_value=10,
                    value=4,
                )

            if st.button("Execute Clustering"):
                if algo_choice == "KMeans":
                    model = KMeans(
                        n_clusters=n_clusters, random_state=42, n_init=10
                    )
                    cluster_labels = model.fit_predict(scaled_train)
                else:
                    model = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type="full",
                        random_state=42,
                    )
                    cluster_labels = model.fit(scaled_train).predict(
                        scaled_train
                    )

                st.session_state["cluster_model"] = model
                st.session_state["algo_choice"] = algo_choice
                st.session_state["cluster_labels"] = cluster_labels
                st.session_state["n_clusters"] = n_clusters

                st.success(
                    f"Clustering complete. Identified {n_clusters} regimes across baseline dataset."
                )

            if "cluster_labels" in st.session_state:
                cluster_labels = st.session_state["cluster_labels"]

                counts = (
                    pd.Series(cluster_labels)
                    .value_counts()
                    .reset_index(name="Count")
                )
                counts.columns = ["Regime", "Count"]
                fig_dist = px.bar(
                    counts,
                    x="Regime",
                    y="Count",
                    color="Regime",
                    title="Data Distribution Across Operating Regimes",
                    text_auto=True,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

    # STEP 3: REGIME METRICS & RADIUS
    with t_metrics:
        st.subheader(
            "Step 3: Calculating Local Centroids, Precision Matrices & Thresholds"
        )

        if "cluster_labels" not in st.session_state:
            st.info("Please complete operational clustering in Step 2 first.")
        else:
            scaled_train = st.session_state["scaled_train"]
            raw_train_df = st.session_state["raw_train_df"]
            feature_cols = st.session_state["feature_cols"]
            cluster_labels = st.session_state["cluster_labels"]
            n_clusters = st.session_state["n_clusters"]
            model = st.session_state["cluster_model"]
            algo_choice = st.session_state["algo_choice"]

            percentile_thresh = st.slider(
                "Select Radius Threshold Percentile:",
                min_value=90.0,
                max_value=99.9,
                value=99.0,
                step=0.1,
            )

            if st.button("Compute Regime Metrics & Covariance"):
                registry = {}

                for k in range(n_clusters):
                    cluster_mask = cluster_labels == k
                    cluster_data = scaled_train[cluster_mask]
                    raw_cluster_data = raw_train_df[feature_cols].iloc[cluster_mask]

                    if algo_choice == "KMeans":
                        centroid = model.cluster_centers_[k]
                    else:
                        centroid = model.means_[k]

                    # Centroid in raw unscaled space for deviation calculation
                    raw_centroid = raw_cluster_data.mean().values

                    lw = LedoitWolf()
                    lw.fit(cluster_data)
                    precision_matrix = lw.precision_

                    diffs = cluster_data - centroid
                    d_m_train = np.sqrt(
                        np.sum(np.dot(diffs, precision_matrix) * diffs, axis=1)
                    )

                    threshold_R = np.percentile(d_m_train, percentile_thresh)

                    registry[k] = {
                        "centroid": centroid,
                        "raw_centroid": raw_centroid,
                        "precision": precision_matrix,
                        "threshold_R": threshold_R,
                        "sample_count": len(cluster_data),
                        "max_d_m": d_m_train.max(),
                        "mean_d_m": d_m_train.mean(),
                    }

                st.session_state["registry"] = registry
                st.session_state["percentile_thresh"] = percentile_thresh
                st.session_state["is_trained"] = True
                st.success(
                    "Local regime covariance matrices inverted and boundary radii established!"
                )

            if "registry" in st.session_state:
                registry = st.session_state["registry"]
                regime_summary = []

                for k, v in registry.items():
                    regime_summary.append(
                        {
                            "Regime": k,
                            "Baseline Samples": v["sample_count"],
                            "Mean Mahalanobis Dist": round(v["mean_d_m"], 4),
                            "Max Mahalanobis Dist": round(v["max_d_m"], 4),
                            "Threshold Radius (R_k)": round(
                                v["threshold_R"], 4
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(regime_summary), use_container_width=True
                )

    # STEP 4: SUMMARY
    with t_summary:
        st.subheader("Step 4: Baseline Model Registry Status")

        if st.session_state.get("is_trained", False):
            st.success(
                "Model Pipeline is Fully Trained and Ready for Online Monitoring!"
            )

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Feature Dimensions", len(st.session_state["feature_cols"]))
            m_col2.metric("Operating Regimes", st.session_state["n_clusters"])
            m_col3.metric("Clustering Engine", st.session_state["algo_choice"])
            m_col4.metric(
                "Boundary Percentile",
                f"{st.session_state['percentile_thresh']}%",
            )

            st.write("---")
            st.write(
                "**Action:** Switch top selection button to **Phase 2: Online Monitoring**."
            )
        else:
            st.warning(
                "Training pipeline is incomplete. Please run Step 3 to register regime metrics."
            )

# ---------------------------------------------------------
# PHASE 2: ONLINE MONITORING
# ---------------------------------------------------------
else:
    st.header("Phase 2: Online Monitoring & Anomaly Scoring")

    if not st.session_state.get("is_trained", False):
        st.error(
            "No trained baseline model detected. Please complete 'Phase 1: Offline Training' first."
        )
    elif st.session_state.get("active_dataset_name") != selected_dataset_name:
        st.warning(
            f"Dataset configuration changed to **{selected_dataset_name}**. "
            "Please return to **Phase 1: Offline Training** to retrain the baseline model."
        )
    else:
        t_live_data, t_predict, t_score, t_dashboard = st.tabs(
            [
                "1. Live Data Ingestion",
                "2. Regime Prediction",
                "3. Mahalanobis & Residual Scoring",
                "4. Anomaly Alert Dashboard",
            ]
        )

        # STEP 1: LIVE DATA INGESTION
        with t_live_data:
            st.subheader("Step 1: Load Real-Time / Injected Test Data")
            st.write(f"Active Test Source: `{TEST_URL}`")

            @st.cache_data
            def load_test_data(url):
                return pd.read_csv(url)

            try:
                raw_test_df = load_test_data(TEST_URL)
                feature_cols = st.session_state["feature_cols"]

                st.success(
                    f"Test dataset loaded: {raw_test_df.shape[0]} samples."
                )
                st.dataframe(raw_test_df.head(), height=250)

                scaler = st.session_state["scaler"]
                scaled_test = scaler.transform(raw_test_df[feature_cols])

                st.session_state["raw_test_df"] = raw_test_df
                st.session_state["scaled_test"] = scaled_test

            except Exception as e:
                st.error(f"Failed to load test dataset: {e}")

        # STEP 2: REGIME PREDICTION
        with t_predict:
            st.subheader("Step 2: Assign Live Points to Baseline Regimes")

            if "scaled_test" not in st.session_state:
                st.info("Please load test data in Step 1 first.")
            else:
                scaled_test = st.session_state["scaled_test"]
                model = st.session_state["cluster_model"]

                live_clusters = model.predict(scaled_test)
                st.session_state["live_clusters"] = live_clusters

                st.success(
                    "All incoming data points mapped to closest operating regimes."
                )

                fig_regimes = px.line(
                    x=np.arange(len(live_clusters)),
                    y=live_clusters,
                    labels={"x": "Sample Index", "y": "Active Regime ID"},
                    title="Active Operational Regime Assignment Over Time",
                )
                fig_regimes.update_traces(mode="lines+markers")
                st.plotly_chart(fig_regimes, use_container_width=True)

        # STEP 3: MAHALANOBIS & RESIDUAL SCORING
        with t_score:
            st.subheader(
                "Step 3: Calculate Local Mahalanobis Distance & Edge Residuals"
            )

            if "live_clusters" not in st.session_state:
                st.info("Please complete regime prediction in Step 2 first.")
            else:
                scaled_test = st.session_state["scaled_test"]
                live_clusters = st.session_state["live_clusters"]
                registry = st.session_state["registry"]

                results = []

                for i, point in enumerate(scaled_test):
                    k = live_clusters[i]
                    centroid = registry[k]["centroid"]
                    precision = registry[k]["precision"]
                    threshold_R = registry[k]["threshold_R"]

                    diff = point - centroid
                    d_m = np.sqrt(np.dot(np.dot(diff, precision), diff.T))

                    d_edge = max(0.0, d_m - threshold_R)

                    results.append(
                        {
                            "Sample": i,
                            "Assigned_Regime": k,
                            "Mahalanobis_Distance": d_m,
                            "Regime_Threshold": threshold_R,
                            "Edge_Residual": d_edge,
                            "Alarm_Status": "FAULT / ANOMALY"
                            if d_edge > 0
                            else "Normal",
                        }
                    )

                results_df = pd.DataFrame(results)
                st.session_state["results_df"] = results_df
                st.success("Distance computations complete!")

                st.dataframe(results_df, height=300, use_container_width=True)

        # STEP 4: ANOMALY ALERT DASHBOARD (ENHANCED WITH ROOT CAUSE ATTRIBUTION)
        with t_dashboard:
            st.subheader("Step 4: Machine Reliability & Anomaly Dashboard")

            if "results_df" not in st.session_state:
                st.info("Please execute scoring calculations in Step 3 first.")
            else:
                results_df = st.session_state["results_df"]

                total_samples = len(results_df)
                total_anomalies = (
                    results_df["Alarm_Status"] == "FAULT / ANOMALY"
                ).sum()
                normal_samples = total_samples - total_anomalies

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Evaluation Samples", total_samples)
                c2.metric(
                    "Normal Operations",
                    normal_samples,
                    delta="Normal",
                    delta_color="normal",
                )
                c3.metric(
                    "Detected Anomalies (Breaches)",
                    total_anomalies,
                    delta=f"{round((total_anomalies/total_samples)*100, 1)}% Faults",
                    delta_color="inverse",
                )

                st.markdown("---")

                # Trend Plot
                fig_trend = gg.Figure()
                fig_trend.add_trace(
                    gg.Scatter(
                        x=results_df["Sample"],
                        y=results_df["Mahalanobis_Distance"],
                        mode="lines",
                        name="Mahalanobis Distance (d_M)",
                        line=dict(color="blue", width=1.5),
                    )
                )

                fig_trend.add_trace(
                    gg.Scatter(
                        x=results_df["Sample"],
                        y=results_df["Regime_Threshold"],
                        mode="lines",
                        name="Regime Radius Threshold (R_k)",
                        line=dict(color="orange", dash="dash", width=2),
                    )
                )

                faults = results_df[results_df["Edge_Residual"] > 0]
                if not faults.empty:
                    fig_trend.add_trace(
                        gg.Scatter(
                            x=faults["Sample"],
                            y=faults["Mahalanobis_Distance"],
                            mode="markers",
                            name="Anomaly / Fault Breach",
                            marker=dict(color="red", size=6, symbol="x"),
                        )
                    )

                fig_trend.update_layout(
                    title="Live Mahalanobis Distance vs. Regime Threshold (R_k)",
                    xaxis_title="Sample Index / Time",
                    yaxis_title="Distance Score",
                    hovermode="x unified",
                )

                st.plotly_chart(fig_trend, use_container_width=True)

                # Edge Residual Plot
                fig_edge = px.area(
                    results_df,
                    x="Sample",
                    y="Edge_Residual",
                    title="Edge Residual Score (d_edge = max(0, d_M - R_k))",
                    color_discrete_sequence=["red"],
                )
                st.plotly_chart(fig_edge, use_container_width=True)

                # ---------------------------------------------------------
                # NEW SECTION: ROOT CAUSE ATTRIBUTION & DEVIATION ANALYSIS
                # ---------------------------------------------------------
                st.markdown("---")
                st.subheader("Root Cause & Feature Deviation Inspector")
                st.caption(
                    "Select a sample point to diagnose feature-level deviations from the predicted regime baseline."
                )

                feature_cols = st.session_state["feature_cols"]
                raw_test_df = st.session_state["raw_test_df"]
                scaled_test = st.session_state["scaled_test"]
                registry = st.session_state["registry"]
                scaler = st.session_state["scaler"]

                col_sel1, col_sel2 = st.columns([1, 2])

                with col_sel1:
                    fault_samples = results_df[
                        results_df["Alarm_Status"] == "FAULT / ANOMALY"
                    ]["Sample"].tolist()

                    if fault_samples:
                        sample_to_inspect = st.selectbox(
                            "Select Detected Anomaly Sample Index:",
                            options=fault_samples,
                            index=0,
                        )
                    else:
                        sample_to_inspect = st.number_input(
                            "Select Any Sample Index to Inspect:",
                            min_value=0,
                            max_value=total_samples - 1,
                            value=0,
                        )

                # Extract data for selected sample
                sample_row = results_df.iloc[sample_to_inspect]
                assigned_k = int(sample_row["Assigned_Regime"])
                d_m_val = sample_row["Mahalanobis_Distance"]
                r_k_val = sample_row["Regime_Threshold"]
                status = sample_row["Alarm_Status"]

                with col_sel2:
                    st.write(
                        f"**Inspection Summary for Sample `{sample_to_inspect}`:**"
                    )
                    st.write(
                        f"• Mapped Operating Regime: **Regime {assigned_k}**  \n"
                        f"• Mahalanobis Distance ($d_M$): `{d_m_val:.4f}` | Threshold ($R_k$): `{r_k_val:.4f}`  \n"
                        f"• Status: **:{'red' if status == 'FAULT / ANOMALY' else 'green'}[{status}]**"
                    )

                # Calculation of feature-level deviations
                z_sample = scaled_test[sample_to_inspect]
                z_centroid = registry[assigned_k]["centroid"]

                # Standard deviations and means of baseline features
                std_devs = scaler.scale_
                means = scaler.mean_

                raw_actuals = raw_test_df[feature_cols].iloc[sample_to_inspect].values
                raw_predicted = (
                    registry[assigned_k]["raw_centroid"]
                    if "raw_centroid" in registry[assigned_k]
                    else (z_centroid * std_devs + means)
                )

                raw_delta = raw_actuals - raw_predicted
                scaled_abs_dev = np.abs(z_sample - z_centroid)
                std_deviations = (raw_actuals - raw_predicted) / std_devs

                diag_df = pd.DataFrame(
                    {
                        "Feature Tag": feature_cols,
                        "Actual Value": raw_actuals,
                        "Predicted Baseline Mean": raw_predicted,
                        "Absolute Delta (Δ)": raw_delta,
                        "Deviations (Std Devs σ)": std_deviations,
                        "Contribution Score (|z - μ_k|)": scaled_abs_dev,
                    }
                ).sort_values(by="Contribution Score (|z - μ_k|)", ascending=False)

                # Display Top Contributor Visuals
                d_col1, d_col2 = st.columns(2)

                with d_col1:
                    fig_contrib = px.bar(
                        diag_df.head(10),
                        x="Contribution Score (|z - μ_k|)",
                        y="Feature Tag",
                        orientation="h",
                        title=f"Top 10 Feature Contributors (Sample #{sample_to_inspect})",
                        color="Contribution Score (|z - μ_k|)",
                        color_continuous_scale="Reds",
                    )
                    fig_contrib.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_contrib, use_container_width=True)

                with d_col2:
                    fig_dev = px.bar(
                        diag_df.head(10),
                        x="Deviations (Std Devs σ)",
                        y="Feature Tag",
                        orientation="h",
                        title=f"Feature Offsets in Standard Deviations (σ)",
                        color="Deviations (Std Devs σ)",
                        color_continuous_scale="RdBu_r",
                    )
                    fig_dev.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_dev, use_container_width=True)

                # Detailed Table Breakdown
                st.write("**Full Feature Deviation & Contribution Breakdown:**")
                st.dataframe(
                    diag_df.style.format(
                        {
                            "Actual Value": "{:.4f}",
                            "Predicted Baseline Mean": "{:.4f}",
                            "Absolute Delta (Δ)": "{:+.4f}",
                            "Deviations (Std Devs σ)": "{:+.2f}σ",
                            "Contribution Score (|z - μ_k|)": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    height=300,
                )
