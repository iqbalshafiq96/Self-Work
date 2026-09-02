import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="Machine Reliability & Anomaly Detection", layout="wide"
)

st.title("Industrial Machinery Reliability & Anomaly Diagnostics")

# Initialize Session State
if "registry" not in st.session_state:
    st.session_state["registry"] = {}

# Tabs setup
t_data, t_regime, t_scoring, t_dashboard = st.tabs(
    [
        "Step 1: Data Preparation",
        "Step 2: Operating Regime Clustering",
        "Step 3: Mahalanobis Distance Scoring",
        "Step 4: Machine Reliability & Anomaly Dashboard",
    ]
)

# -----------------------------------------------------------------------------
# STEP 1: DATA PREPARATION
# -----------------------------------------------------------------------------
with t_data:
    st.subheader("Step 1: Data Preparation & Inspection")

    uploaded_file = st.file_uploader(
        "Upload Plant Machinery Dataset (CSV)", type=["csv"]
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state["raw_df"] = raw_df

        st.write("**Dataset Preview:**")
        st.dataframe(raw_df.head(), use_container_width=True)

        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()

        feature_cols = st.multiselect(
            "Select Features / Sensor Tags for Analysis:",
            options=numeric_cols,
            default=numeric_cols,
        )

        if feature_cols:
            st.session_state["feature_cols"] = feature_cols

            # Train/Test Split Slider
            split_idx = st.slider(
                "Select Baseline Training Sample Count:",
                min_value=int(len(raw_df) * 0.2),
                max_value=int(len(raw_df) * 0.9),
                value=int(len(raw_df) * 0.6),
            )

            raw_train_df = raw_df.iloc[:split_idx].copy()
            raw_test_df = raw_df.iloc[split_idx:].copy().reset_index(drop=True)

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(raw_train_df[feature_cols])
            scaled_test = scaler.transform(raw_test_df[feature_cols])

            st.session_state["raw_train_df"] = raw_train_df
            st.session_state["raw_test_df"] = raw_test_df
            st.session_state["scaled_train"] = scaled_train
            st.session_state["scaled_test"] = scaled_test
            st.session_state["scaler"] = scaler

            st.success(
                f"Data split successfully: {len(raw_train_df)} Training Samples | {len(raw_test_df)} Test Evaluation Samples"
            )


# -----------------------------------------------------------------------------
# STEP 2: OPERATING REGIME CLUSTERING
# -----------------------------------------------------------------------------
with t_regime:
    st.subheader("Step 2: Operating Regime Clustering (k-Means)")

    if "scaled_train" not in st.session_state:
        st.info("Please complete Step 1: Data Preparation first.")
    else:
        n_clusters = st.slider(
            "Select Number of Operating Regimes (k):",
            min_value=1,
            max_value=10,
            value=3,
        )

        if st.button("Fit Operating Regimes"):
            scaled_train = st.session_state["scaled_train"]
            raw_train_df = st.session_state["raw_train_df"]
            feature_cols = st.session_state["feature_cols"]

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            train_clusters = kmeans.fit_predict(scaled_train)

            registry = {}
            for k in range(n_clusters):
                cluster_mask = train_clusters == k
                cluster_samples = scaled_train[cluster_mask]

                if len(cluster_samples) > 0:
                    centroid = kmeans.cluster_centers_[k]
                    cov_matrix = np.cov(cluster_samples, rowvar=False)

                    # Regularization for stability
                    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
                    inv_cov = np.linalg.pinv(cov_matrix)

                    # Compute training Mahalanobis distances to set threshold
                    diff = cluster_samples - centroid
                    m_distances = np.sqrt(
                        np.sum(np.dot(diff, inv_cov) * diff, axis=1)
                    )
                    r_k = np.percentile(m_distances, 99)

                    raw_centroid = raw_train_df[feature_cols][
                        cluster_mask
                    ].mean().values

                    registry[k] = {
                        "centroid": centroid,
                        "raw_centroid": raw_centroid,
                        "inv_cov": inv_cov,
                        "threshold_R_k": r_k,
                        "sample_count": len(cluster_samples),
                    }

            st.session_state["kmeans"] = kmeans
            st.session_state["registry"] = registry
            st.success(f"Successfully trained {n_clusters} Operating Regimes.")


# -----------------------------------------------------------------------------
# STEP 3: MAHALANOBIS DISTANCE SCORING
# -----------------------------------------------------------------------------
with t_scoring:
    st.subheader("Step 3: Test Dataset Scoring")

    if "registry" not in st.session_state or not st.session_state["registry"]:
        st.info("Please fit operating regimes in Step 2 first.")
    else:
        if st.button("Execute Scoring on Test Data"):
            scaled_test = st.session_state["scaled_test"]
            registry = st.session_state["registry"]
            kmeans = st.session_state["kmeans"]

            assigned_regimes = kmeans.predict(scaled_test)

            m_distances = []
            r_k_thresholds = []
            edge_residuals = []
            alarm_statuses = []

            for i, z_i in enumerate(scaled_test):
                k = assigned_regimes[i]
                centroid = registry[k]["centroid"]
                inv_cov = registry[k]["inv_cov"]
                r_k = registry[k]["threshold_R_k"]

                diff = z_i - centroid
                d_m = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
                d_edge = max(0.0, d_m - r_k)

                m_distances.append(d_m)
                r_k_thresholds.append(r_k)
                edge_residuals.append(d_edge)
                alarm_statuses.append(
                    "FAULT / ANOMALY" if d_edge > 0 else "NORMAL"
                )

            results_df = pd.DataFrame(
                {
                    "Sample": np.arange(len(scaled_test)),
                    "Assigned_Regime": assigned_regimes,
                    "Mahalanobis_Distance": m_distances,
                    "Regime_Threshold": r_k_thresholds,
                    "Edge_Residual": edge_residuals,
                    "Alarm_Status": alarm_statuses,
                }
            )

            st.session_state["results_df"] = results_df
            st.success("Scoring execution complete.")
            st.dataframe(results_df.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# STEP 4: MACHINE RELIABILITY & ANOMALY DASHBOARD
# -----------------------------------------------------------------------------
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
        # ROOT CAUSE ATTRIBUTION & FULL DEVIATION INSPECTOR
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("Root Cause & Full Feature Deviation Inspector")
        st.caption(
            "Inspect feature-level deviations from the predicted regime baseline across all features."
        )

        feature_cols = st.session_state["feature_cols"]
        raw_test_df = st.session_state["raw_test_df"]
        scaled_test = st.session_state["scaled_test"]
        registry = st.session_state["registry"]
        scaler = st.session_state["scaler"]

        col_sel1, col_sel2 = st.columns([1, 2])

        with col_sel1:
            filter_mode = st.radio(
                "Sample Selection Mode:",
                options=["Detected Anomalies Only", "All Samples"],
                horizontal=True,
            )

            fault_samples = results_df[
                results_df["Alarm_Status"] == "FAULT / ANOMALY"
            ]["Sample"].tolist()

            if filter_mode == "Detected Anomalies Only" and fault_samples:
                sample_to_inspect = st.selectbox(
                    "Select Anomaly Sample Index:",
                    options=fault_samples,
                    index=0,
                )
            elif filter_mode == "Detected Anomalies Only" and not fault_samples:
                st.warning("No anomalies detected in test dataset.")
                sample_to_inspect = 0
            else:
                sample_to_inspect = st.number_input(
                    "Select Any Sample Index:",
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

        # Complete DataFrame including ALL features without truncation
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

        # Dynamic chart height based on tag count
        dynamic_chart_height = max(400, len(feature_cols) * 28)

        # Display All Feature Contributor Visuals
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            fig_contrib = px.bar(
                diag_df,
                x="Contribution Score (|z - μ_k|)",
                y="Feature Tag",
                orientation="h",
                title=f"Feature Contribution Ranking ({len(feature_cols)} Features)",
                color="Contribution Score (|z - μ_k|)",
                color_continuous_scale="Reds",
                height=dynamic_chart_height,
            )
            fig_contrib.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )
            st.plotly_chart(fig_contrib, use_container_width=True)

        with d_col2:
            fig_dev = px.bar(
                diag_df,
                x="Deviations (Std Devs σ)",
                y="Feature Tag",
                orientation="h",
                title=f"Feature Deviations in Standard Deviations (σ)",
                color="Deviations (Std Devs σ)",
                color_continuous_scale="RdBu_r",
                height=dynamic_chart_height,
            )
            fig_dev.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_dev, use_container_width=True)

        # Detailed Table Breakdown (All Features)
        st.write(
            f"**Complete Feature Breakdown ({len(feature_cols)} Process Tags):**"
        )
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
            height=max(300, min(800, len(feature_cols) * 35)),
        )
