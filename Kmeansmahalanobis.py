import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="AVEVA OMR - Rotating Asset Predictive Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Minimalist Styling
st.title("AVEVA Operational Model & Reliability (OMR) Engine")
st.caption(
    "Empirical Pattern Recognition (EPR) & Multi-Sensor Overall Model Index (OMI) Scoring"
)

# ---------------------------------------------------------
# AVEVA OMR EMPIRICAL SCORING ENGINE
# ---------------------------------------------------------
class AVEVAOMREngine:
    """
    AVEVA PRiSM / OMR Engine implementation.
    Calculates empirical baseline predictions, individual sensor residuals,
    and regularized Hotelling's T^2 / Mahalanobis Overall Model Index (OMI).
    """
    def __init__(self, shrink_factor: float = 1e-3):
        self.shrink_factor = shrink_factor
        self.profiles = {}
        self.feature_cols = []
        self.scaler = None

    def fit_empirical_baseline(self, X_raw: pd.DataFrame, feature_cols: list, profile_col: str = None, percentile: float = 99.0):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw[feature_cols])
        
        # If no explicit operating profile column is provided, treat whole dataset as Primary State (Mode 1)
        if profile_col is None or profile_col not in X_raw.columns:
            profile_ids = np.zeros(len(X_raw), dtype=int)
        else:
            profile_ids = X_raw[profile_col].values

        unique_profiles = np.unique(profile_ids)

        for p_id in unique_profiles:
            mask = profile_ids == p_id
            X_prof_scaled = X_scaled[mask]
            X_prof_raw = X_raw[feature_cols].iloc[mask].values

            n_samples, p_features = X_prof_scaled.shape

            # Empirical Center State Vector
            z_centroid = np.mean(X_prof_scaled, axis=0)
            raw_centroid = np.mean(X_prof_raw, axis=0)

            # Regularized Precision Matrix (Ledoit-Wolf)
            try:
                lw = LedoitWolf()
                cov_matrix = lw.fit(X_prof_scaled).covariance_
            except Exception:
                cov_matrix = np.cov(X_prof_scaled, rowvar=False)

            ridge = np.eye(p_features) * (self.shrink_factor * np.trace(cov_matrix) / max(1, p_features))
            reg_cov = cov_matrix + ridge

            # Inverse SVD for precision matrix stabilization
            U, S, Vt = np.linalg.svd(reg_cov)
            max_s = np.max(S)
            S_inv = np.array([1.0 / s if (s / max_s) > 1e-5 else 0.0 for s in S])
            precision_matrix = np.dot(Vt.T, np.dot(np.diag(S_inv), U.T))

            # Baseline OMI (Dimension-Normalized Mahalanobis Distance)
            diffs = X_prof_scaled - z_centroid
            omi_sq = np.sum(np.dot(diffs, precision_matrix) * diffs, axis=1)
            omi_norm = np.sqrt(np.maximum(0.0, omi_sq)) / np.sqrt(p_features)

            omi_threshold = np.percentile(omi_norm, percentile)

            self.profiles[p_id] = {
                "z_centroid": z_centroid,
                "raw_centroid": raw_centroid,
                "precision": precision_matrix,
                "omi_threshold": omi_threshold,
                "p_features": p_features,
                "sample_count": n_samples,
            }

    def score_live_sample(self, raw_sample: np.ndarray, active_profile: int = 0):
        if active_profile not in self.profiles:
            active_profile = list(self.profiles.keys())[0]

        prof = self.profiles[active_profile]
        z_sample = self.scaler.transform(raw_sample.reshape(1, -1))[0]
        
        z_centroid = prof["z_centroid"]
        raw_centroid = prof["raw_centroid"]
        precision = prof["precision"]
        p_features = prof["p_features"]

        # Predicted Baseline Vector (y_hat)
        raw_predicted = raw_centroid

        # Sensor Residuals (y - y_hat)
        raw_residuals = raw_sample - raw_predicted

        # Normalized Residuals in Standard Deviations
        std_devs = self.scaler.scale_
        std_residuals = raw_residuals / std_devs

        # Compute Overall Model Index (OMI)
        diff = z_sample - z_centroid
        omi_sq = np.dot(np.dot(diff, precision), diff.T)
        omi_sq = max(0.0, float(omi_sq))
        omi_val = np.sqrt(omi_sq) / np.sqrt(p_features)

        omi_thresh = prof["omi_threshold"]
        omi_residual = max(0.0, omi_val - omi_thresh)

        return {
            "active_profile": active_profile,
            "OMI": omi_val,
            "OMI_Threshold": omi_thresh,
            "OMI_Residual": omi_residual,
            "Is_Alert": omi_residual > 0,
            "raw_predicted": raw_predicted,
            "raw_residuals": raw_residuals,
            "std_residuals": std_residuals,
        }


# ---------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_csv(url):
    df = pd.read_csv(url)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------
# SIDEBAR SETUP
# ---------------------------------------------------------
st.sidebar.header("AVEVA OMR Data Sources")

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
    "Select Asset Baseline & Live Case:",
    options=list(DATASET_MAP.keys()),
    index=0,
)

TRAIN_URL = DATASET_MAP[selected_dataset_name]["train"]
TEST_URL = DATASET_MAP[selected_dataset_name]["test"]

# Top-level workflow tabs
phase = st.radio(
    "Workflow Mode:",
    options=["1. Baseline Model Setup & Training", "2. Online Asset Monitoring & Diagnostics"],
    horizontal=True,
)

st.markdown("---")

# ---------------------------------------------------------
# PHASE 1: MODEL SETUP & TRAINING
# ---------------------------------------------------------
if phase == "1. Baseline Model Setup & Training":
    st.header("Phase 1: AVEVA OMR Baseline Model Definition")

    t_data, t_profile, t_train = st.tabs(
        [
            "1. Asset Tag Selection",
            "2. Operational Profile Definition",
            "3. OMI Threshold Calibration",
        ]
    )

    with t_data:
        st.subheader("Asset Sensor Ingestion")
        st.write(f"Training Baseline Dataset: `{TRAIN_URL}`")

        try:
            raw_train_df = load_and_clean_csv(TRAIN_URL)
            numeric_cols = raw_train_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c.lower() not in ["timestamp", "time", "date"]]

            st.success(f"Loaded {raw_train_df.shape[0]} baseline samples across {len(feature_cols)} sensors.")

            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("**Model Inputs (Sensors):**")
                st.dataframe(pd.DataFrame({"Tag Name": feature_cols}), height=250)
            with c2:
                st.write("**Raw Signal Preview:**")
                st.dataframe(raw_train_df.head(), height=250)

            st.session_state["raw_train_df"] = raw_train_df
            st.session_state["feature_cols"] = feature_cols
            st.session_state["active_dataset_name"] = selected_dataset_name

        except Exception as e:
            st.error(f"Error reading asset baseline data: {e}")

    with t_profile:
        st.subheader("Operational Profile Filtering")
        st.write("In AVEVA OMR, operational profiles split baseline data into distinct load levels or modes.")

        if "raw_train_df" in st.session_state:
            feature_cols = st.session_state["feature_cols"]
            raw_train_df = st.session_state["raw_train_df"]

            profile_tag = st.selectbox(
                "Select Operating Mode Variable (Optional):",
                options=["[None - Single Operating Profile]"] + feature_cols,
            )

            if profile_tag != "[None - Single Operating Profile]":
                st.session_state["profile_tag"] = profile_tag
                st.info(f"Filtering profile state using tag: `{profile_tag}`")
            else:
                st.session_state["profile_tag"] = None
                st.info("Operating model configured as single mode profile.")
        else:
            st.warning("Please ingest baseline dataset in Step 1.")

    with t_train:
        st.subheader("Empirical Profile Training & OMI Threshold Calibration")

        if "raw_train_df" not in st.session_state:
            st.warning("Please ingest baseline data in Step 1.")
        else:
            percentile_thresh = st.slider(
                "Set Overall Model Index (OMI) Statistical Alarm Threshold (%):",
                min_value=95.0,
                max_value=99.9,
                value=99.0,
                step=0.1,
            )

            if st.button("Generate Baseline Model & Calculate Calibration"):
                engine = AVEVAOMREngine(shrink_factor=1e-3)
                raw_train_df = st.session_state["raw_train_df"]
                feature_cols = st.session_state["feature_cols"]
                profile_tag = st.session_state.get("profile_tag")

                engine.fit_empirical_baseline(
                    X_raw=raw_train_df,
                    feature_cols=feature_cols,
                    profile_col=profile_tag,
                    percentile=percentile_thresh,
                )

                st.session_state["omr_engine"] = engine
                st.session_state["is_model_trained"] = True
                st.success("AVEVA OMR Empirical Baseline Model generated successfully!")

                # Profile Summary Table
                summary = []
                for pid, prof in engine.profiles.items():
                    summary.append(
                        {
                            "Operating Profile Mode": f"Mode {pid}",
                            "Training Samples": prof["sample_count"],
                            "Monitored Sensors": prof["p_features"],
                            "Calibrated OMI Threshold": round(prof["omi_threshold"], 4),
                        }
                    )
                st.dataframe(pd.DataFrame(summary), use_container_width=True)

# ---------------------------------------------------------
# PHASE 2: ONLINE MONITORING & DIAGNOSTICS
# ---------------------------------------------------------
else:
    st.header("Phase 2: Online Asset Reliability & Catch Diagnostics")

    if not st.session_state.get("is_model_trained", False):
        st.error("No calibrated baseline model detected. Run Phase 1 setup first.")
    elif st.session_state.get("active_dataset_name") != selected_dataset_name:
        st.warning(f"Dataset changed to **{selected_dataset_name}**. Please retrain baseline model in Phase 1.")
    else:
        t_live, t_eval, t_diag = st.tabs(
            [
                "1. Live Data Stream",
                "2. Overall Model Index (OMI) Trend",
                "3. Sensor Catch & Deviation Analysis",
            ]
        )

        with t_live:
            st.subheader("Live Operational Ingestion")
            st.write(f"Active Test Source: `{TEST_URL}`")

            try:
                raw_test_df = load_and_clean_csv(TEST_URL)
                feature_cols = st.session_state["feature_cols"]

                missing = [f for f in feature_cols if f not in raw_test_df.columns]
                if missing:
                    st.error(f"Live dataset missing required model tags: {missing}")
                else:
                    st.success(f"Live data loaded: {raw_test_df.shape[0]} timestamps.")
                    st.dataframe(raw_test_df.head(), height=250)
                    st.session_state["raw_test_df"] = raw_test_df

            except Exception as e:
                st.error(f"Failed to load live data stream: {e}")

        with t_eval:
            st.subheader("Asset Health Evaluation (Overall Model Index)")

            if "raw_test_df" not in st.session_state:
                st.info("Please load live data in Step 1.")
            else:
                raw_test_df = st.session_state["raw_test_df"]
                feature_cols = st.session_state["feature_cols"]
                engine = st.session_state["omr_engine"]
                profile_tag = st.session_state.get("profile_tag")

                eval_results = []
                for i in range(len(raw_test_df)):
                    sample = raw_test_df[feature_cols].iloc[i].values
                    p_id = 0 if (profile_tag is None or profile_tag not in raw_test_df.columns) else int(raw_test_df[profile_tag].iloc[i])
                    
                    res = engine.score_live_sample(sample, active_profile=p_id)
                    eval_results.append(
                        {
                            "Sample": i,
                            "OMI": res["OMI"],
                            "OMI_Threshold": res["OMI_Threshold"],
                            "OMI_Residual": res["OMI_Residual"],
                            "Status": "ALARM / BREACH" if res["Is_Alert"] else "Normal",
                        }
                    )

                results_df = pd.DataFrame(eval_results)
                st.session_state["results_df"] = results_df

                # Top Metrics
                total_samples = len(results_df)
                total_alarms = (results_df["Status"] == "ALARM / BREACH").sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Evaluation Timestamps", total_samples)
                c2.metric("Normal Asset State", total_samples - total_alarms)
                c3.metric(
                    "OMI Alarm Breaches",
                    total_alarms,
                    delta=f"{round((total_alarms/total_samples)*100, 1)}% Off-Normal",
                    delta_color="inverse",
                )

                st.markdown("---")

                # OMI Chart
                fig_omi = go.Figure()
                fig_omi.add_trace(
                    go.Scatter(
                        x=results_df["Sample"],
                        y=results_df["OMI"],
                        mode="lines",
                        name="Overall Model Index (OMI)",
                        line=dict(color="#008080", width=1.5),  # Petronas Teal
                    )
                )
                fig_omi.add_trace(
                    go.Scatter(
                        x=results_df["Sample"],
                        y=results_df["OMI_Threshold"],
                        mode="lines",
                        name="Calibrated Threshold",
                        line=dict(color="red", dash="dash", width=2),
                    )
                )

                alarms = results_df[results_df["Status"] == "ALARM / BREACH"]
                if not alarms.empty:
                    fig_omi.add_trace(
                        go.Scatter(
                            x=alarms["Sample"],
                            y=alarms["OMI"],
                            mode="markers",
                            name="Model Alarm Breach",
                            marker=dict(color="red", size=6, symbol="x"),
                        )
                    )

                fig_omi.update_layout(
                    title="Asset Overall Model Index (OMI) vs. Statistical Threshold",
                    xaxis_title="Sample / Timestamp",
                    yaxis_title="Normalized OMI Distance",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_omi, use_container_width=True)

        with t_diag:
            st.subheader("Sensor Catch & Model Residual Diagnostics")
            st.caption("Diagnose specific tag deviation contributions ($y - \\hat{y}$) driving the OMI alarm.")

            if "results_df" not in st.session_state:
                st.info("Run OMI Evaluation in Step 2 first.")
            else:
                results_df = st.session_state["results_df"]
                raw_test_df = st.session_state["raw_test_df"]
                feature_cols = st.session_state["feature_cols"]
                engine = st.session_state["omr_engine"]
                profile_tag = st.session_state.get("profile_tag")

                all_samples = results_df["Sample"].tolist()
                alarm_samples = results_df[results_df["Status"] == "ALARM / BREACH"]["Sample"].tolist()
                default_idx = all_samples.index(alarm_samples[0]) if alarm_samples else 0

                sample_to_inspect = st.selectbox(
                    "Select Timestamp to Inspect:",
                    options=all_samples,
                    index=default_idx,
                    format_func=lambda x: f"Sample #{x} {'⚠️ [ALARM]' if x in alarm_samples else '✅ [Normal]'}",
                )

                # Execute Single-Point Diagnostics
                raw_sample = raw_test_df[feature_cols].iloc[sample_to_inspect].values
                p_id = 0 if (profile_tag is None or profile_tag not in raw_test_df.columns) else int(raw_test_df[profile_tag].iloc[sample_to_inspect])
                diag_res = engine.score_live_sample(raw_sample, active_profile=p_id)

                # Construct Diagnostic DataFrame
                diag_df = pd.DataFrame(
                    {
                        "Sensor Tag": feature_cols,
                        "Actual Value (y)": raw_sample,
                        "Model Predicted (ŷ)": diag_res["raw_predicted"],
                        "Raw Residual (y - ŷ)": diag_res["raw_residuals"],
                        "Normalized Deviation (σ)": diag_res["std_residuals"],
                        "Absolute Contribution (|σ|)": np.abs(diag_res["std_residuals"]),
                    }
                ).sort_values(by="Absolute Contribution (|σ|)", ascending=False)

                # Visualization layout
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    fig_contrib = px.bar(
                        diag_df.head(10),
                        x="Absolute Contribution (|σ|)",
                        y="Sensor Tag",
                        orientation="h",
                        title=f"Top 10 Offending Sensors (Sample #{sample_to_inspect})",
                        color="Absolute Contribution (|σ|)",
                        color_continuous_scale="Reds",
                    )
                    fig_contrib.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_contrib, use_container_width=True)

                with col_chart2:
                    fig_res = px.bar(
                        diag_df.head(10),
                        x="Normalized Deviation (σ)",
                        y="Sensor Tag",
                        orientation="h",
                        title="Normalized Sensor Residuals (σ)",
                        color="Normalized Deviation (σ)",
                        color_continuous_scale="RdBu_r",
                    )
                    fig_res.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_res, use_container_width=True)

                # Breakdown Table
                st.write("**Complete Sensor Profile Residual Breakdown:**")
                st.dataframe(
                    diag_df.style.format(
                        {
                            "Actual Value (y)": "{:.4f}",
                            "Model Predicted (ŷ)": "{:.4f}",
                            "Raw Residual (y - ŷ)": "{:+.4f}",
                            "Normalized Deviation (σ)": "{:+.2f}σ",
                            "Absolute Contribution (|σ|)": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    height=300,
                )
