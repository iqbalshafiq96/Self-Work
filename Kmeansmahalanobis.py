import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="k-NN Normalized Residual & Diagnostic System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("k-NN Normalized Residual & Diagnostic System")
st.caption(
    "Empirical Pattern Matching via k-Nearest Neighbor Point-to-Point Baseline Alignment."
)


# ---------------------------------------------------------
# POINT-TO-POINT NEAREST NEIGHBOR ENGINE
# ---------------------------------------------------------
class KNNNearestNeighborEngine:
    """
    Model Residual Engine mapping live sample points
    to the single closest baseline timestamp using Euclidean Search 
    for robust target matching, with Residual Mahalanobis scoring.
    """

    def __init__(self):
        self.scaler = None
        self.nn_model_2k = None
        self.nn_lookup_1k = None
        self.feature_cols = []
        self.X_train_raw = None
        self.X_train_scaled = None
        self.cov_inv = None
        self.d_99 = 1.0
        self.metric = "euclidean"

    def fit_baseline_with_progress(
        self,
        X_raw: pd.DataFrame,
        feature_cols: list,
        metric: str = "euclidean",
        percentile: float = 99.0,
        progress_bar=None,
        status_text=None,
    ):
        self.feature_cols = feature_cols
        self.metric = metric.lower()
        self.scaler = StandardScaler()
        self.X_train_raw = X_raw[feature_cols].copy().reset_index(drop=True)
        if status_text:
            status_text.text("Step 1/4: Standardizing feature tags...")
        if progress_bar:
            progress_bar.progress(25)
        time.sleep(0.1)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_raw)
        # Compute Inverse Covariance Matrix for Mahalanobis Residual Scoring
        cov_matrix = np.cov(self.X_train_scaled, rowvar=False)
        # Add regularizing ridge (1e-6) to prevent zero-variance singular inversion
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov_matrix)
        if status_text:
            status_text.text(
                "Step 2/4: Fitting 2-Nearest Neighbor Euclidean Graph for Baseline..."
            )
        if progress_bar:
            progress_bar.progress(50)
        time.sleep(0.1)
        # Always use Euclidean distance for neighbor search to prevent space warping
        self.nn_model_2k = NearestNeighbors(
            n_neighbors=2,
            algorithm="auto",
            metric="euclidean",
        ).fit(self.X_train_scaled)
        if status_text:
            status_text.text(
                f"Step 3/4: Computing {percentile}th percentile scale boundary..."
            )
        if progress_bar:
            progress_bar.progress(75)
        time.sleep(0.1)
        # Calculate calibration baseline distances
        distances, indices = self.nn_model_2k.kneighbors(self.X_train_scaled)
        baseline_dists = []
        for i in range(len(self.X_train_scaled)):
            if self.metric == "mahalanobis":
                # Compute distance between baseline point and its nearest neighbor
                nn_idx = indices[i, 1]
                z_res = self.X_train_scaled[i] - self.X_train_scaled[nn_idx]
                m_dist = np.sqrt(
                    np.maximum(0.0, np.dot(np.dot(z_res, self.cov_inv), z_res.T))
                )
                baseline_dists.append(m_dist)
            else:
                baseline_dists.append(distances[i, 1])
        self.d_99 = max(np.percentile(baseline_dists, percentile), 1e-6)
        if status_text:
            status_text.text("Step 4/4: Finalizing k=1 lookup index...")
        if progress_bar:
            progress_bar.progress(90)
        time.sleep(0.1)
        self.nn_lookup_1k = NearestNeighbors(
            n_neighbors=1,
            algorithm="auto",
            metric="euclidean",
        ).fit(self.X_train_scaled)
        if progress_bar:
            progress_bar.progress(100)
        if status_text:
            status_text.text("Calibration Complete!")

    def score_live_sample(self, raw_sample: np.ndarray):
        z_sample = self.scaler.transform(raw_sample.reshape(1, -1))[0]
        # 1. Physical pattern match: Always use Euclidean distance to find nearest healthy point
        dist_euc, idx = self.nn_lookup_1k.kneighbors(z_sample.reshape(1, -1))
        nearest_idx = int(idx[0][0])
        raw_predicted = self.X_train_raw.iloc[nearest_idx].values
        # 2. Residual Vector Computation
        raw_residuals = raw_sample - raw_predicted
        pct_residuals = (raw_residuals / (np.abs(raw_predicted) + 1e-6)) * 100.0
        std_residuals = raw_residuals / self.scaler.scale_
        # 3. Distance Metric Scoring on the Residual Vector
        if self.metric == "mahalanobis":
            # Mahalanobis distance evaluated on the standardized residual vector
            calculated_dist = float(
                np.sqrt(
                    np.maximum(
                        0.0, np.dot(np.dot(std_residuals, self.cov_inv), std_residuals.T)
                    )
                )
            )
        else:
            calculated_dist = float(dist_euc[0][0])
        mr_pct = (calculated_dist / self.d_99) * 10.0
        return {
            "nearest_baseline_idx": nearest_idx,
            "raw_dist": calculated_dist,
            "d_99_threshold": self.d_99,
            "Model_Residual_pct": mr_pct,
            "Is_Alarm": mr_pct > 5.0,
            "Is_Alert": mr_pct > 10.0,
            "raw_predicted": raw_predicted,
            "raw_residuals": raw_residuals,
            "pct_residuals": pct_residuals,
            "std_residuals": std_residuals,
        }


# ---------------------------------------------------------
# DATA INGESTION PIPELINE
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_csv(file_input):
    df = pd.read_csv(file_input)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    return df


def get_clean_dataset(file_input):
    df = load_and_clean_csv(file_input)
    features = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c.lower() not in ["timestamp", "time", "date"]
    ]
    return df, features


# BASELINE / TRAINING DATASETS
BASELINE_DATASETS = {
    "NOC_Chiller": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Chiller.csv",
    "NOC6_1": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv",
    "NOC_HX": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_hx_normal_operation.csv",
}

# EVALUATION / TEST DATASETS
TEST_DATASETS = {
    "Case_0": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_0.csv",
    "Case_Chiller_Deviation": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Chiller_deviation.csv",
    "Case_Chiller_Highload": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Chiller_highload.csv",
    "Case_Chiller_Motor": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Chiller_motor.csv",
    "Case_HX_Fouling": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_hx_fouling_deterioration.csv",
}

# MAP DESCRIPTIONS FOR EVALUATION DATASETS
CASE_DESCRIPTIONS = {
    "NOC6_1": "Baseline self-evaluation case",
    "NOC_Chiller": "Baseline chiller self-evaluation case",
    "NOC_HX": "Baseline heat exchanger self-evaluation case",
    "Case_0": "Test evaluation dataset 0",
    "Case_Chiller_Deviation": "Chiller operating under standard baseline conditions",
    "Case_Chiller_Highload": "Chiller operating under high thermal load conditions exceeding baseline limits",
    "Case_Chiller_Motor": "Motor system degradation accompanied by elevated stator and bearing temperatures",
    "Case_HX_Fouling": "Heat exchanger operating under fouling deterioration conditions exceeding baseline limits",
}

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Calibrate Baseline",
        "2. Model Residual Trend & Diagnostics",
        "3. Operational Profile",
        "4. Operating Path (All Samples)",
    ]
)

# ---------------------------------------------------------
# TAB 1: CALIBRATE BASELINE
# ---------------------------------------------------------
with tab1:
    st.subheader("Point-to-Point Baseline Calibration")
    st.write(
        "Select or upload a reference baseline model dataset using nearest neighbor matching."
    )
    col_cfg1, col_cfg2 = st.columns(2)
    CUSTOM_BASELINE_KEY = "Upload Custom Baseline CSV..."
    with col_cfg1:
        baseline_options = list(BASELINE_DATASETS.keys()) + [CUSTOM_BASELINE_KEY]
        selected_train_key = st.selectbox(
            "Select Baseline / Training Dataset:",
            options=baseline_options,
            index=0,  # Default to NOC_Chiller
            key="tab1_train_dataset_select",
        )
        uploaded_baseline_file = None
        if selected_train_key == CUSTOM_BASELINE_KEY:
            uploaded_baseline_file = st.file_uploader(
                "Upload Baseline CSV Data:",
                type=["csv"],
                key="tab1_baseline_file_uploader",
                help="Upload a clean numerical CSV containing healthy baseline operation tags.",
            )
        selected_metric = st.radio(
            "Residual Distance Scoring Method:",
            options=["Euclidean", "Mahalanobis"],
            index=0,
            horizontal=True,
            key="tab1_metric_radio",
            help="Euclidean uses standard standardized residual distance. Mahalanobis evaluates residuals against the cross-sensor inverse covariance matrix (Σ^-1).",
        )
    with col_cfg2:
        train_split_pct = st.slider(
            "Training Data Ratio (%):",
            min_value=50,
            max_value=100,
            value=80,
            step=5,
            key="tab1_train_split_slider",
            help="Select percentage of dataset used to calibrate baseline model. Set to 100% to use full baseline dataset.",
        )
        percentile_thresh = st.slider(
            "Baseline Scale Boundary Percentile:",
            min_value=95.0,
            max_value=99.9,
            value=99.0,
            step=0.1,
            key="tab1_percentile_slider",
        )
    with st.expander("📐 Calculation Method Details & Formulations", expanded=False):
        st.markdown(
            """
        ### Robust Distance Architecture

        To prevent artificial fault suppression during parameter degradation, pattern matching is decoupled from residual scoring:

        1. **Target Matching (Euclidean Space):**
           The nearest healthy baseline state $\\mathbf{\\hat{x}}$ is identified using Euclidean distance on $Z$-score standardized variables:
           
           $$d_{\\text{Match}}(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\sum_{i=1}^{p} (z_{x,i} - z_{y,i})^2}$$

        2. **Residual Scoring:**
           Once the true physical target $\\mathbf{\\hat{x}}$ is isolated, the residual vector $\\mathbf{r} = \\mathbf{z}_x - \\mathbf{z}_{\\hat{x}}$ is evaluated using your selected metric:

           * **Euclidean Residual:** $d_{\\text{Residual}} = \\sqrt{\\mathbf{r}^T \\mathbf{r}}$
           * **Mahalanobis Residual:** $d_{\\text{Residual}} = \\sqrt{\\mathbf{r}^T \\mathbf{\\Sigma}^{-1} \\mathbf{r}}$
        """
        )
    # Ingest Data Source based on Selection
    raw_train_df, feature_cols = None, []
    if selected_train_key == CUSTOM_BASELINE_KEY:
        if uploaded_baseline_file is not None:
            try:
                raw_train_df, feature_cols = get_clean_dataset(uploaded_baseline_file)
            except Exception as e:
                st.error(f"Failed to load custom baseline CSV: {e}")
                st.stop()
        else:
            st.info(" Please upload a custom baseline CSV file to proceed.")
            st.stop()
    else:
        try:
            raw_train_df, feature_cols = get_clean_dataset(
                BASELINE_DATASETS[selected_train_key]
            )
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()
    if train_split_pct == 100:
        train_split_df = raw_train_df.copy().reset_index(drop=True)
        test_split_df = pd.DataFrame(columns=raw_train_df.columns)
    else:
        test_size_ratio = (100 - train_split_pct) / 100.0
        train_split_df, test_split_df = train_test_split(
            raw_train_df,
            test_size=test_size_ratio,
            shuffle=True,
            random_state=42,
        )
        train_split_df = train_split_df.reset_index(drop=True)
        test_split_df = test_split_df.reset_index(drop=True)
    c_c1, c_c2, c_c3 = st.columns(3)
    with c_c1:
        st.info(f"**Selected Baseline:** {selected_train_key}")
        st.write(f"- Total Raw Samples: **{raw_train_df.shape[0]}**")
        st.write(f"- {train_split_pct}% Training Baseline: **{train_split_df.shape[0]}**")
        st.write(f"- {100 - train_split_pct}% Evaluation Test Set: **{test_split_df.shape[0]}**")
    with c_c2:
        st.write(f"- **Total Operational Tags:** {len(feature_cols)}")
        st.write(f"- **Percentile Scale Boundary:** {percentile_thresh}%")
    with c_c3:
        st.write(f"- **Residual Metric:** {selected_metric}")
        st.write("- **Match Strategy:** Unwarped Euclidean Search")
    st.markdown("---")
    if st.button("Calibrate Baseline Model", type="primary", use_container_width=True):
        status_text = st.empty()
        progress_bar = st.progress(0)
        engine = KNNNearestNeighborEngine()
        engine.fit_baseline_with_progress(
            X_raw=train_split_df,
            feature_cols=feature_cols,
            metric=selected_metric.lower(),
            percentile=percentile_thresh,
            progress_bar=progress_bar,
            status_text=status_text,
        )
        st.session_state["p2p_engine"] = engine
        st.session_state["active_train_key"] = selected_train_key
        st.session_state["active_feature_cols"] = feature_cols
        st.session_state["active_raw_train_df"] = train_split_df
        st.session_state["active_raw_test_df"] = test_split_df
        st.session_state["active_percentile"] = percentile_thresh
        st.session_state["active_metric"] = selected_metric
        st.session_state["active_train_pct"] = train_split_pct
        st.success("Model Residual Engine Calibrated Successfully!")
    if "p2p_engine" in st.session_state:
        active_key = st.session_state.get("active_train_key")
        active_pct = st.session_state.get("active_percentile")
        active_met = st.session_state.get("active_metric")
        active_split = st.session_state.get("active_train_pct")
        if (
            active_key == selected_train_key
            and active_pct == percentile_thresh
            and active_met == selected_metric
            and active_split == train_split_pct
        ):
            st.success(
                f"Active Baseline Model Ready ({active_key} | {active_split}% Split | {active_met} Metric @ {active_pct}%)."
            )
        else:
            st.info(
                f"Currently Active Model: **{active_key}** ({active_split}% Split | {active_met} Metric @ {active_pct}%). Click 'Calibrate Baseline Model' above to apply changes."
            )

# ---------------------------------------------------------
# TAB 2: MODEL RESIDUAL TREND & DIAGNOSTICS
# ---------------------------------------------------------
with tab2:
    st.subheader("Model Residual Trend (%) & Parameter Diagnostics")
    if "p2p_engine" not in st.session_state:
        st.warning("Please calibrate the baseline model in **Tab 1** first.")
    else:
        engine = st.session_state["p2p_engine"]
        active_train_key = st.session_state["active_train_key"]
        active_train_pct = st.session_state.get("active_train_pct", 80)
        feature_cols = st.session_state["active_feature_cols"]
        raw_train_df = st.session_state["active_raw_train_df"]
        raw_split_test_df = st.session_state["active_raw_test_df"]
        eval_options_map = {}
        raw_test_df = None
        # Check if active baseline is custom uploaded
        if active_train_key == CUSTOM_BASELINE_KEY:
            st.markdown("### 📤 Upload Custom Evaluation Dataset")
            st.info(
                "You calibrated a custom baseline model. Please upload your custom evaluation CSV dataset matching the same feature tags to perform diagnostic comparisons."
            )
            if active_train_pct < 100 and not raw_split_test_df.empty:
                EVAL_HOLDOUT_LABEL = f"Custom Baseline (Holdout {100 - active_train_pct}% Evaluation Set)"
                eval_options_map[EVAL_HOLDOUT_LABEL] = raw_split_test_df
            uploaded_test_file = st.file_uploader(
                "Upload Custom Evaluation CSV:",
                type=["csv"],
                key="tab2_custom_test_uploader",
            )
            if uploaded_test_file is not None:
                try:
                    custom_test_df, test_features = get_clean_dataset(uploaded_test_file)
                    # Verify feature alignment
                    missing_cols = set(feature_cols) - set(test_features)
                    if missing_cols:
                        st.error(
                            f"Uploaded evaluation file is missing required baseline feature tags: {list(missing_cols)}"
                        )
                        st.stop()
                    eval_options_map[uploaded_test_file.name] = custom_test_df
                except Exception as e:
                    st.error(f"Failed to read evaluation CSV: {e}")
                    st.stop()
            if not eval_options_map:
                st.warning("Please upload a custom evaluation CSV above to begin diagnostic evaluation.")
                st.stop()
            selected_eval_label = st.selectbox(
                "Select Evaluation Dataset to Compare Against Calibrated Custom Baseline:",
                options=list(eval_options_map.keys()),
                index=0,
            )
            raw_test_df = eval_options_map[selected_eval_label]
        else:
            # Preset datasets map
            if active_train_pct < 100 and not raw_split_test_df.empty:
                EVAL_HOLDOUT_LABEL = f"{active_train_key} (Holdout {100 - active_train_pct}% Evaluation Set)"
                eval_options_map[EVAL_HOLDOUT_LABEL] = raw_split_test_df
            else:
                EVAL_SELF_LABEL = f"{active_train_key} (Self-Evaluation Full Baseline)"
                eval_options_map[EVAL_SELF_LABEL] = raw_train_df
            if active_train_key == "NOC6_1":
                eval_options_map["Case_0"] = TEST_DATASETS["Case_0"]
            elif active_train_key == "NOC_Chiller":
                for k, v in TEST_DATASETS.items():
                    if "chiller" in k.lower():
                        eval_options_map[k] = v
            elif active_train_key == "NOC_HX":
                for k, v in TEST_DATASETS.items():
                    if "hx" in k.lower():
                        eval_options_map[k] = v
            selected_eval_label = st.selectbox(
                "Select Evaluation / Test Dataset to Compare Against Calibrated Baseline:",
                options=list(eval_options_map.keys()),
                index=0,
                format_func=lambda key: f"{key} — {CASE_DESCRIPTIONS.get(key, 'Evaluation Case')}"
                if key in CASE_DESCRIPTIONS
                else key,
            )
            selected_eval_val = eval_options_map[selected_eval_label]
            if isinstance(selected_eval_val, pd.DataFrame):
                raw_test_df = selected_eval_val
            else:
                try:
                    raw_test_df, _ = get_clean_dataset(selected_eval_val)
                except Exception as e:
                    st.error(f"Failed to load evaluation dataset: {e}")
                    st.stop()
        progress_eval = st.progress(0)
        eval_results = []
        predicted_matrix = []
        all_diag_list = []
        n_samples = len(raw_test_df)
        for i in range(n_samples):
            sample = raw_test_df[feature_cols].iloc[i].values
            res = engine.score_live_sample(sample)
            mr_val = res["Model_Residual_pct"]
            if mr_val > 10.0:
                status_str = "ALERT BREACH (>10%)"
            elif mr_val > 5.0:
                status_str = "ALARM BREACH (5%–10%)"
            else:
                status_str = "Normal (≤5%)"
            eval_results.append(
                {
                    "Sample": i,
                    "Matched Baseline Row": res["nearest_baseline_idx"],
                    "Model Residual (%)": mr_val,
                    "Status": status_str,
                }
            )
            predicted_matrix.append(res["raw_predicted"])
            all_diag_list.append(
                {
                    "Actual Value (y)": sample,
                    "Nearest Baseline Target (ŷ)": res["raw_predicted"],
                    "Raw Residual (y - ŷ)": res["raw_residuals"],
                    "Sensor Residual (%)": res["pct_residuals"],
                    "Normalized Deviation (σ)": res["std_residuals"],
                }
            )
            if i % max(1, n_samples // 10) == 0:
                progress_eval.progress(int((i + 1) / n_samples * 100))
        progress_eval.progress(100)
        results_df = pd.DataFrame(eval_results)
        pred_df = pd.DataFrame(predicted_matrix, columns=feature_cols)
        total_samples = len(results_df)
        total_alarms = (
            (results_df["Model Residual (%)"] > 5.0)
            & (results_df["Model Residual (%)"] <= 10.0)
        ).sum()
        total_alerts = (results_df["Model Residual (%)"] > 10.0).sum()
        total_normal = (results_df["Model Residual (%)"] <= 5.0).sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Evaluated Timestamps", total_samples)
        m2.metric("Normal Operating Range (≤5%)", total_normal)
        m3.metric(
            "Alarm Breaches (5%–10%)",
            total_alarms,
            delta=f"{round((total_alarms / total_samples) * 100, 1)}%",
            delta_color="off",
        )
        m4.metric(
            "Alert Breaches (>10%)",
            total_alerts,
            delta=f"{round((total_alerts / total_samples) * 100, 1)}%",
            delta_color="inverse",
        )
        st.markdown("---")
        col_sidebar, col_main = st.columns([1, 3])
        avg_std_res_all = np.mean(
            [np.abs(d["Normalized Deviation (σ)"]) for d in all_diag_list], axis=0
        )
        top_deviated_idx = np.argsort(avg_std_res_all)[::-1][:5]
        top_deviated_tags = [feature_cols[idx] for idx in top_deviated_idx]
        with col_sidebar:
            st.markdown("### 🎛️ Sensor Selection")
            st.caption("Tick sensors to overlay on the trend plot.")
            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button("Select All", use_container_width=True):
                for f in feature_cols:
                    st.session_state[f"chk_{f}"] = True
            if c_btn2.button("Clear All", use_container_width=True):
                for f in feature_cols:
                    st.session_state[f"chk_{f}"] = False
            if st.button(
                "Top 5 Deviations (σ)", use_container_width=True, type="secondary"
            ):
                for f in feature_cols:
                    st.session_state[f"chk_{f}"] = f in top_deviated_tags
            st.markdown("---")
            selected_tags = []
            for idx, feature in enumerate(feature_cols):
                default_state = True if idx == 0 else False
                if f"chk_{feature}" not in st.session_state:
                    st.session_state[f"chk_{feature}"] = default_state
                if st.checkbox(feature, key=f"chk_{feature}"):
                    selected_tags.append(feature)
        with col_main:
            st.markdown("### 📈 Model Residual (%) Trend")
            fig_mr = go.Figure()
            fig_mr.add_trace(
                go.Scatter(
                    x=results_df["Sample"],
                    y=results_df["Model Residual (%)"],
                    mode="lines",
                    name="Model Residual (%)",
                    line=dict(color="#008080", width=1.5),
                )
            )
            fig_mr.add_trace(
                go.Scatter(
                    x=results_df["Sample"],
                    y=[5.0] * len(results_df),
                    mode="lines",
                    name="5% Alarm Threshold",
                    line=dict(color="orange", dash="dash", width=1.5),
                )
            )
            fig_mr.add_trace(
                go.Scatter(
                    x=results_df["Sample"],
                    y=[10.0] * len(results_df),
                    mode="lines",
                    name="10% Alert Threshold",
                    line=dict(color="red", dash="dash", width=1.5),
                )
            )
            fig_mr.update_layout(
                xaxis_title="Sample Index",
                yaxis_title="Model Residual (%)",
                hovermode="x unified",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_mr, use_container_width=True)
            st.markdown("### 📊 Actual vs. Predicted Parameter Trends")
            if not selected_tags:
                st.info(
                    "👈 Check one or more parameters in the left panel to render actual vs. predicted trends."
                )
            else:
                fig_trends = go.Figure()
                colors = px.colors.qualitative.Plotly
                for i, tag in enumerate(selected_tags):
                    color = colors[i % len(colors)]
                    fig_trends.add_trace(
                        go.Scatter(
                            x=results_df["Sample"],
                            y=raw_test_df[tag],
                            mode="lines",
                            name=f"{tag} (Actual)",
                            line=dict(color=color, width=2),
                        )
                    )
                    fig_trends.add_trace(
                        go.Scatter(
                            x=results_df["Sample"],
                            y=pred_df[tag],
                            mode="lines",
                            name=f"{tag} (Predicted ŷ)",
                            line=dict(color=color, width=1.5, dash="dash"),
                        )
                    )
                fig_trends.update_layout(
                    xaxis_title="Sample Index",
                    yaxis_title="Parameter Value",
                    hovermode="x unified",
                    height=450,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                )
                st.plotly_chart(fig_trends, use_container_width=True)
        st.markdown("---")
        st.subheader("Sensor Diagnostics for Selected Timestamp")
        AVG_LABEL = "Average Sensor Residual"
        sample_options = [AVG_LABEL] + results_df["Sample"].tolist()
        sample_to_inspect = st.selectbox(
            "Select Timestamp to Inspect Sensor Breakdown:",
            options=sample_options,
            index=0,
            format_func=lambda x: AVG_LABEL
            if x == AVG_LABEL
            else f"Sample #{x} (Matched Baseline Row #{results_df.loc[x, 'Matched Baseline Row']})",
        )
        if sample_to_inspect == AVG_LABEL:
            avg_actual = np.mean(
                [d["Actual Value (y)"] for d in all_diag_list], axis=0
            )
            avg_predicted = np.mean(
                [d["Nearest Baseline Target (ŷ)"] for d in all_diag_list], axis=0
            )
            avg_raw_res = np.mean(
                [d["Raw Residual (y - ŷ)"] for d in all_diag_list], axis=0
            )
            avg_pct_res = np.mean(
                [d["Sensor Residual (%)"] for d in all_diag_list], axis=0
            )
            avg_std_res = np.mean(
                [d["Normalized Deviation (σ)"] for d in all_diag_list], axis=0
            )
            avg_mr_pct = results_df["Model Residual (%)"].mean()
            diag_df = pd.DataFrame(
                {
                    "Sensor Tag": feature_cols,
                    "Actual Value (y)": avg_actual,
                    "Nearest Baseline Target (ŷ)": avg_predicted,
                    "Raw Residual (y - ŷ)": avg_raw_res,
                    "Sensor Residual (%)": avg_pct_res,
                    "Normalized Deviation (σ)": avg_std_res,
                    "Abs Deviation (|σ|)": np.abs(avg_std_res),
                }
            ).sort_values(by="Abs Deviation (|σ|)", ascending=False)
            st.info(
                f"Displaying **{AVG_LABEL}** calculated across **{n_samples}** evaluated timestamps | Mean Model Residual: **{avg_mr_pct:.4f}%**"
            )
        else:
            raw_sample = raw_test_df[feature_cols].iloc[sample_to_inspect].values
            diag_res = engine.score_live_sample(raw_sample)
            diag_df = pd.DataFrame(
                {
                    "Sensor Tag": feature_cols,
                    "Actual Value (y)": raw_sample,
                    "Nearest Baseline Target (ŷ)": diag_res["raw_predicted"],
                    "Raw Residual (y - ŷ)": diag_res["raw_residuals"],
                    "Sensor Residual (%)": diag_res["pct_residuals"],
                    "Normalized Deviation (σ)": diag_res["std_residuals"],
                    "Abs Deviation (|σ|)": np.abs(diag_res["std_residuals"]),
                }
            ).sort_values(by="Abs Deviation (|σ|)", ascending=False)
            st.info(
                f"Sample **#{sample_to_inspect}** matched to Baseline Timestamp **#{diag_res['nearest_baseline_idx']}** | Calculated Model Residual: **{diag_res['Model_Residual_pct']:.4f}%**"
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
        st.dataframe(
            diag_df.drop(columns=["Abs Deviation (|σ|)"]).style.format(
                {
                    "Actual Value (y)": "{:.4f}",
                    "Nearest Baseline Target (ŷ)": "{:.4f}",
                    "Raw Residual (y - ŷ)": "{:+.4f}",
                    "Sensor Residual (%)": "{:+.2f}%",
                    "Normalized Deviation (σ)": "{:+.2f}σ",
                }
            ),
            use_container_width=True,
            height=300,
        )
        st.session_state["current_eval_df"] = raw_test_df

# ---------------------------------------------------------
# TAB 3: 3D OPERATIONAL PROFILE
# ---------------------------------------------------------
with tab3:
    st.subheader("3D Space: Live Sample vs. Nearest Baseline Point")
    if "p2p_engine" not in st.session_state:
        st.warning("Please calibrate the baseline model in **Tab 1** first.")
    else:
        engine = st.session_state["p2p_engine"]
        feature_cols = st.session_state["active_feature_cols"]
        raw_train_df = st.session_state["active_raw_train_df"]
        raw_eval_df = st.session_state.get("current_eval_df", raw_train_df)
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            x_tag = st.selectbox("X-Axis Sensor Tag:", options=feature_cols, index=0)
        with col_p2:
            y_tag = st.selectbox(
                "Y-Axis Sensor Tag:",
                options=feature_cols,
                index=min(1, len(feature_cols) - 1),
            )
        with col_p3:
            z_tag = st.selectbox(
                "Z-Axis Sensor Tag:",
                options=feature_cols,
                index=min(2, len(feature_cols) - 1),
            )
        live_sample_idx = st.selectbox(
            "Select Live Timestamp to Overlay in 3D Space:",
            options=list(range(len(raw_eval_df))),
            format_func=lambda i: f"Sample #{i}",
        )
        fig_3d = px.scatter_3d(
            engine.X_train_raw,
            x=x_tag,
            y=y_tag,
            z=z_tag,
            opacity=0.3,
            title="Baseline Space with Live Point Match Vector",
        )
        fig_3d.update_traces(marker=dict(size=2, color="blue"))
        raw_live_sample = raw_eval_df[feature_cols].iloc[live_sample_idx].values
        live_res = engine.score_live_sample(raw_live_sample)
        live_x = raw_eval_df[x_tag].iloc[live_sample_idx]
        live_y = raw_eval_df[y_tag].iloc[live_sample_idx]
        live_z = raw_eval_df[z_tag].iloc[live_sample_idx]
        target_row_raw = live_res["raw_predicted"]
        target_x = target_row_raw[feature_cols.index(x_tag)]
        target_y = target_row_raw[feature_cols.index(y_tag)]
        target_z = target_row_raw[feature_cols.index(z_tag)]
        point_color = "green"
        if live_res["Is_Alert"]:
            point_color = "red"
        elif live_res["Is_Alarm"]:
            point_color = "orange"
        fig_3d.add_trace(
            go.Scatter3d(
                x=[live_x],
                y=[live_y],
                z=[live_z],
                mode="markers",
                name=f"Live Sample #{live_sample_idx}",
                marker=dict(
                    color=point_color,
                    size=8,
                    symbol="diamond",
                ),
            )
        )
        fig_3d.add_trace(
            go.Scatter3d(
                x=[live_x, target_x],
                y=[live_y, target_y],
                z=[live_z, target_z],
                mode="lines+markers",
                name=f"Nearest Baseline Match (Row #{live_res['nearest_baseline_idx']})",
                line=dict(color="orange", width=4),
                marker=dict(size=4, color="orange"),
            )
        )
        fig_3d.update_layout(
            height=700,
            scene=dict(
                xaxis_title=x_tag,
                yaxis_title=y_tag,
                zaxis_title=z_tag,
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: 3D OPERATING PATH (ALL SAMPLES, GRADIENT BY TIME)
# ---------------------------------------------------------
with tab4:
    st.subheader("3D Space: Live Sample vs. Nearest Baseline Point (All Test Samples Overlaid)")
    if "p2p_engine" not in st.session_state:
        st.warning("Please calibrate the baseline model in **Tab 1** first.")
    else:
        engine = st.session_state["p2p_engine"]
        feature_cols = st.session_state["active_feature_cols"]
        raw_train_df = st.session_state["active_raw_train_df"]
        raw_eval_df = st.session_state.get("current_eval_df", raw_train_df)
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            x_tag4 = st.selectbox(
                "X-Axis Sensor Tag:", options=feature_cols, index=0, key="tab4_x_tag"
            )
        with col_q2:
            y_tag4 = st.selectbox(
                "Y-Axis Sensor Tag:",
                options=feature_cols,
                index=min(1, len(feature_cols) - 1),
                key="tab4_y_tag",
            )
        with col_q3:
            z_tag4 = st.selectbox(
                "Z-Axis Sensor Tag:",
                options=feature_cols,
                index=min(2, len(feature_cols) - 1),
                key="tab4_z_tag",
            )
        live_sample_idx4 = st.selectbox(
            "Select Live Timestamp to Overlay in 3D Space:",
            options=list(range(len(raw_eval_df))),
            format_func=lambda i: f"Sample #{i}",
            key="tab4_live_sample_select",
        )
        fig_4d = px.scatter_3d(
            engine.X_train_raw,
            x=x_tag4,
            y=y_tag4,
            z=z_tag4,
            opacity=0.25,
            title="Baseline Space with Full Test Sample Operating Path",
        )
        fig_4d.update_traces(marker=dict(size=2, color="lightgray"))

        # Overlay ALL test/evaluation samples, colored by sample order using a
        # light-green (earliest) to strong-green (latest) gradient.
        n_eval_samples = len(raw_eval_df)
        sample_indices = np.arange(n_eval_samples)
        fig_4d.add_trace(
            go.Scatter3d(
                x=raw_eval_df[x_tag4],
                y=raw_eval_df[y_tag4],
                z=raw_eval_df[z_tag4],
                mode="markers",
                name="Test Samples (Earliest → Latest)",
                marker=dict(
                    size=4,
                    color=sample_indices,
                    colorscale="Greens",
                    cmin=0,
                    cmax=max(n_eval_samples - 1, 1),
                    showscale=True,
                    colorbar=dict(title="Sample Index<br>(Earliest → Latest)"),
                    opacity=0.85,
                ),
                text=[f"Sample #{i}" for i in sample_indices],
                hovertemplate="%{text}<br>"
                + f"{x_tag4}: "
                + "%{x}<br>"
                + f"{y_tag4}: "
                + "%{y}<br>"
                + f"{z_tag4}: "
                + "%{z}<extra></extra>",
            )
        )

        raw_live_sample4 = raw_eval_df[feature_cols].iloc[live_sample_idx4].values
        live_res4 = engine.score_live_sample(raw_live_sample4)
        live_x4 = raw_eval_df[x_tag4].iloc[live_sample_idx4]
        live_y4 = raw_eval_df[y_tag4].iloc[live_sample_idx4]
        live_z4 = raw_eval_df[z_tag4].iloc[live_sample_idx4]
        target_row_raw4 = live_res4["raw_predicted"]
        target_x4 = target_row_raw4[feature_cols.index(x_tag4)]
        target_y4 = target_row_raw4[feature_cols.index(y_tag4)]
        target_z4 = target_row_raw4[feature_cols.index(z_tag4)]
        point_color4 = "green"
        if live_res4["Is_Alert"]:
            point_color4 = "red"
        elif live_res4["Is_Alarm"]:
            point_color4 = "orange"
        fig_4d.add_trace(
            go.Scatter3d(
                x=[live_x4],
                y=[live_y4],
                z=[live_z4],
                mode="markers",
                name=f"Live Sample #{live_sample_idx4}",
                marker=dict(
                    color=point_color4,
                    size=9,
                    symbol="diamond",
                    line=dict(color="black", width=1),
                ),
            )
        )
        fig_4d.add_trace(
            go.Scatter3d(
                x=[live_x4, target_x4],
                y=[live_y4, target_y4],
                z=[live_z4, target_z4],
                mode="lines+markers",
                name=f"Nearest Baseline Match (Row #{live_res4['nearest_baseline_idx']})",
                line=dict(color="orange", width=4),
                marker=dict(size=4, color="orange"),
            )
        )
        fig_4d.update_layout(
            height=700,
            scene=dict(
                xaxis_title=x_tag4,
                yaxis_title=y_tag4,
                zaxis_title=z_tag4,
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        st.plotly_chart(fig_4d, use_container_width=True)
