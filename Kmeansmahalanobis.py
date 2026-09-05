import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
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
    "Empirical Pattern Matching via k-Nearest Neighbor (k=1) Point-to-Point Baseline Alignment."
)


# ---------------------------------------------------------
# POINT-TO-POINT NEAREST NEIGHBOR ENGINE
# ---------------------------------------------------------
class OMRNearestNeighborEngine:
    """
    Model Residual Engine mapping live sample points
    to the single closest baseline timestamp using k-Nearest Neighbors.
    """

    def __init__(self):
        self.scaler = None
        self.nn_model = None
        self.nn_lookup = None
        self.feature_cols = []
        self.X_train_raw = None
        self.X_train_scaled = None
        self.d_99 = 1.0

    def fit_baseline_with_progress(
        self,
        X_raw: pd.DataFrame,
        feature_cols: list,
        percentile: float = 99.0,
        progress_bar=None,
        status_text=None,
    ):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.X_train_raw = X_raw[feature_cols].copy().reset_index(drop=True)

        if status_text:
            status_text.text("Step 1/4: Standardizing feature tags...")
        if progress_bar:
            progress_bar.progress(25)
        time.sleep(0.1)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train_raw)

        if status_text:
            status_text.text("Step 2/4: Fitting 2-Nearest Neighbor graph...")
        if progress_bar:
            progress_bar.progress(50)
        time.sleep(0.1)

        self.nn_model = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(
            self.X_train_scaled
        )

        if status_text:
            status_text.text(
                f"Step 3/4: Computing {percentile}th percentile scale boundary..."
            )
        if progress_bar:
            progress_bar.progress(75)
        time.sleep(0.1)

        distances, _ = self.nn_model.kneighbors(self.X_train_scaled)
        neighbor_dists = distances[:, 1]
        self.d_99 = max(np.percentile(neighbor_dists, percentile), 1e-6)

        if status_text:
            status_text.text("Step 4/4: Finalizing k=1 lookup index...")
        if progress_bar:
            progress_bar.progress(90)
        time.sleep(0.1)

        self.nn_lookup = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
            self.X_train_scaled
        )

        if progress_bar:
            progress_bar.progress(100)
        if status_text:
            status_text.text("Calibration Complete!")

    def score_live_sample(self, raw_sample: np.ndarray):
        z_sample = self.scaler.transform(raw_sample.reshape(1, -1))

        dist, idx = self.nn_lookup.kneighbors(z_sample)
        min_distance = float(dist[0][0])
        nearest_idx = int(idx[0][0])

        raw_predicted = self.X_train_raw.iloc[nearest_idx].values
        mr_pct = (min_distance / self.d_99) * 10.0

        raw_residuals = raw_sample - raw_predicted
        pct_residuals = (raw_residuals / (np.abs(raw_predicted) + 1e-6)) * 100.0
        std_residuals = raw_residuals / self.scaler.scale_

        return {
            "nearest_baseline_idx": nearest_idx,
            "raw_dist": min_distance,
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
def load_and_clean_csv(url):
    df = pd.read_csv(url)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    return df


def get_clean_dataset(url):
    df = load_and_clean_csv(url)
    features = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c.lower() not in ["timestamp", "time", "date"]
    ]
    return df, features


# BASELINE / TRAINING DATASETS
BASELINE_DATASETS = {
    "NOC6_1": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv",
    "NOC_Turbine": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Turbine.csv",
    "NOC_Turbine_r1": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Turbine_r1.csv",
    "NOC_Chiller": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Chiller.csv",
}

# EVALUATION / TEST DATASETS
TEST_DATASETS = {
    "Case_0": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_0.csv",
    "Case_Turbine": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Turbine.csv",
    "Case_Turbine_r1": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Turbine_r1.csv",
    "Case_Chiller": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Chiller.csv",
    "Case_Chiller_Deviation": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Chiller_deviation.csv",
    "Case_Chiller_Highload": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Chiller_highload.csv",
    "Case_Chiller_Motor": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Chiller_motor.csv",
}

tab1, tab2, tab3 = st.tabs(
    [
        "1. Calibrate Baseline",
        "2. Model Residual Trend & Diagnostics",
        "3. 3D Operational Profile",
    ]
)

# ---------------------------------------------------------
# TAB 1: CALIBRATE BASELINE
# ---------------------------------------------------------
with tab1:
    st.subheader("Point-to-Point Baseline Calibration")
    st.write(
        "Select and calibrate a reference baseline model using nearest neighbor matching."
    )

    selected_train_key = st.selectbox(
        "Select Baseline / Training Dataset:",
        options=list(BASELINE_DATASETS.keys()),
        index=1,
        key="tab1_train_dataset_select",
    )

    percentile_thresh = st.slider(
        "Baseline 10% Scale Boundary Percentile:",
        min_value=95.0,
        max_value=99.9,
        value=99.0,
        step=0.1,
        key="tab1_percentile_slider",
    )

    try:
        raw_train_df, feature_cols = get_clean_dataset(
            BASELINE_DATASETS[selected_train_key]
        )
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    c_c1, c_c2 = st.columns(2)
    with c_c1:
        st.info(f"**Selected Baseline Dataset:** {selected_train_key}")
        st.write(f"- Baseline Timestamps: **{raw_train_df.shape[0]}**")

    with c_c2:
        st.write(f"- **Total Operational Tags:** {len(feature_cols)}")
        st.write(f"- **Percentile Scale Boundary:** {percentile_thresh}%")

    st.markdown("---")
    if st.button("Calibrate Baseline Model", type="primary", use_container_width=True):
        status_text = st.empty()
        progress_bar = st.progress(0)

        engine = OMRNearestNeighborEngine()
        engine.fit_baseline_with_progress(
            X_raw=raw_train_df,
            feature_cols=feature_cols,
            percentile=percentile_thresh,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        st.session_state["p2p_engine"] = engine
        st.session_state["active_train_key"] = selected_train_key
        st.session_state["active_feature_cols"] = feature_cols
        st.session_state["active_raw_train_df"] = raw_train_df
        st.session_state["active_percentile"] = percentile_thresh
        st.success("Model Residual Engine Calibrated Successfully!")

    if "p2p_engine" in st.session_state:
        active_key = st.session_state.get("active_train_key")
        active_pct = st.session_state.get("active_percentile")
        if active_key == selected_train_key and active_pct == percentile_thresh:
            st.success(f"Active Baseline Model Ready ({active_key} @ {active_pct}%).")
        else:
            st.info(
                f"Currently Active Model: **{active_key}** (@ {active_pct}%). Click 'Calibrate Baseline Model' above to apply changes."
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
        feature_cols = st.session_state["active_feature_cols"]
        raw_train_df = st.session_state["active_raw_train_df"]

        SELF_EVAL_LABEL = active_train_key

        eval_options_map = {
            SELF_EVAL_LABEL: BASELINE_DATASETS[active_train_key]
        }
        eval_options_map.update(TEST_DATASETS)

        selected_eval_label = st.selectbox(
            "Select Evaluation / Test Dataset to Compare Against Calibrated Baseline:",
            options=list(eval_options_map.keys()),
            index=0,
        )

        selected_eval_url = eval_options_map[selected_eval_label]

        if selected_eval_label == SELF_EVAL_LABEL:
            raw_test_df = raw_train_df
        else:
            try:
                raw_test_df, _ = get_clean_dataset(selected_eval_url)
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

            all_diag_list.append({
                "Actual Value (y)": sample,
                "Nearest Baseline Target (ŷ)": res["raw_predicted"],
                "Raw Residual (y - ŷ)": res["raw_residuals"],
                "Sensor Residual (%)": res["pct_residuals"],
                "Normalized Deviation (σ)": res["std_residuals"]
            })

            if i % max(1, n_samples // 10) == 0:
                progress_eval.progress(int((i + 1) / n_samples * 100))
        progress_eval.progress(100)

        results_df = pd.DataFrame(eval_results)
        pred_df = pd.DataFrame(predicted_matrix, columns=feature_cols)

        total_samples = len(results_df)
        total_alarms = ((results_df["Model Residual (%)"] > 5.0) & (results_df["Model Residual (%)"] <= 10.0)).sum()
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
