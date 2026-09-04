import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="AVEVA OMR - Point-to-Point Nearest Baseline Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AVEVA OMR Engine: Exact Point-to-Point Baseline Matching")
st.caption(
    "Empirical Pattern Matching via k-Nearest Neighbor (k=1). Training Self-Evaluation yields 0% Residual."
)


# ---------------------------------------------------------
# POINT-TO-POINT NEAREST NEIGHBOR ENGINE
# ---------------------------------------------------------
class AVEVAPointToPointEngine:
    """
    AVEVA OMR Engine mapping live samples to the single closest baseline timestamp.
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
            status_text.text(f"Step 3/4: Computing {percentile}th percentile scale boundary...")
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
        omr_pct = (min_distance / self.d_99) * 10.0

        raw_residuals = raw_sample - raw_predicted
        pct_residuals = (raw_residuals / (np.abs(raw_predicted) + 1e-6)) * 100.0
        std_residuals = raw_residuals / self.scaler.scale_

        return {
            "nearest_baseline_idx": nearest_idx,
            "raw_dist": min_distance,
            "d_99_threshold": self.d_99,
            "OMR_pct": omr_pct,
            "Is_Alert": omr_pct > 10.0,
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


AVAILABLE_DATASETS = {
    "NOC6_1 (Normal Baseline 1)": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC6_1.csv",
    "Case_0 (Test Fault 1)": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_0.csv",
    "NOC_Turbine (Turbine Baseline)": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_NOC_Turbine.csv",
    "Case_Turbine (Turbine Test Fault)": "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Multivariate_Case_Turbine.csv",
}

st.sidebar.header("Asset Baseline Configuration")

# Sidebar selects ONLY baseline calibration dataset
selected_train_key = st.sidebar.selectbox(
    "Select Baseline / Training Dataset:",
    options=list(AVAILABLE_DATASETS.keys()),
    index=2,  # Default to NOC_Turbine
)

percentile_thresh = st.sidebar.slider(
    "Baseline 10% Scale Boundary Percentile:",
    min_value=95.0,
    max_value=99.9,
    value=99.0,
    step=0.1,
)

TRAIN_URL = AVAILABLE_DATASETS[selected_train_key]

try:
    raw_train_df = load_and_clean_csv(TRAIN_URL)
    feature_cols = [
        c
        for c in raw_train_df.select_dtypes(include=[np.number]).columns
        if c.lower() not in ["timestamp", "time", "date"]
    ]
    st.sidebar.success(
        f"Ingested **{len(feature_cols)}** Tags across **{raw_train_df.shape[0]}** Baseline Rows."
    )
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(
    [
        "1. Calibrate Baseline",
        "2. OMR Trend & Diagnostics",
        "3. 3D Operational Profile",
    ]
)

# ---------------------------------------------------------
# TAB 1: CALIBRATE BASELINE
# ---------------------------------------------------------
with tab1:
    st.subheader("Point-to-Point Baseline Calibration")
    st.write(
        "Calibrate the baseline model using the selected reference dataset from the sidebar."
    )

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

        engine = AVEVAPointToPointEngine()
        engine.fit_baseline_with_progress(
            X_raw=raw_train_df,
            feature_cols=feature_cols,
            percentile=percentile_thresh,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        st.session_state["p2p_engine"] = engine
        st.session_state["active_train_key"] = selected_train_key
        st.success("Point-to-Point Engine Calibrated Successfully!")

    if (
        "p2p_engine" in st.session_state
        and st.session_state.get("active_train_key") == selected_train_key
    ):
        st.success(f"Active Baseline Model Ready ({selected_train_key}).")

# ---------------------------------------------------------
# TAB 2: OMR TREND & DIAGNOSTICS
# ---------------------------------------------------------
with tab2:
    st.subheader("Overall Model Residual (OMR) Trend (%) & Diagnostics")

    if (
        "p2p_engine" not in st.session_state
        or st.session_state.get("active_train_key") != selected_train_key
    ):
        st.warning("Please calibrate the baseline model in **Tab 1** first.")
    else:
        engine = st.session_state["p2p_engine"]

        # Directly pick or change evaluation data without re-calibrating
        SELF_EVAL_OPTION = f"{selected_train_key} (Self-Evaluation / Baseline)"
        eval_options = [SELF_EVAL_OPTION] + [
            k for k in AVAILABLE_DATASETS.keys() if k != selected_train_key
        ]

        selected_eval_key = st.selectbox(
            "Select Evaluation / Test Dataset to Compare Against Calibrated Baseline:",
            options=eval_options,
            index=0,
        )

        # Map selected eval choice to actual dataframe
        if selected_eval_key == SELF_EVAL_OPTION:
            raw_test_df = raw_train_df
        else:
            raw_test_df = load_and_clean_csv(AVAILABLE_DATASETS[selected_eval_key])

        # Evaluate against current calibrated model
        progress_eval = st.progress(0)
        eval_results = []
        n_samples = len(raw_test_df)

        for i in range(n_samples):
            sample = raw_test_df[feature_cols].iloc[i].values
            res = engine.score_live_sample(sample)
            eval_results.append(
                {
                    "Sample": i,
                    "Matched Baseline Row": res["nearest_baseline_idx"],
                    "OMR (%)": res["OMR_pct"],
                    "Status": "ALARM BREACH (>10%)"
                    if res["Is_Alert"]
                    else "Normal (≤10%)",
                }
            )
            if i % max(1, n_samples // 10) == 0:
                progress_eval.progress(int((i + 1) / n_samples * 100))
        progress_eval.progress(100)

        results_df = pd.DataFrame(eval_results)

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

        fig_omr.update_layout(
            xaxis_title="Sample Index",
            yaxis_title="OMR (%) [10% = Baseline Alarm Boundary]",
            hovermode="x unified",
        )
        st.plotly_chart(fig_omr, use_container_width=True)

        st.markdown("---")
        st.subheader("Sensor Diagnostics for Selected Timestamp")

        all_samples = results_df["Sample"].tolist()
        sample_to_inspect = st.selectbox(
            "Select Timestamp to Inspect Sensor Breakdown:",
            options=all_samples,
            index=0,
            format_func=lambda x: f"Sample #{x} (Matched Baseline Row #{results_df.loc[x, 'Matched Baseline Row']})",
        )

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
            f"Sample **#{sample_to_inspect}** matched to Baseline Timestamp **#{diag_res['nearest_baseline_idx']}** | Calculated OMR: **{diag_res['OMR_pct']:.4f}%**"
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

        # Cache test dataframe in session state for 3D Tab
        st.session_state["current_eval_df"] = raw_test_df

# ---------------------------------------------------------
# TAB 3: 3D OPERATIONAL PROFILE
# ---------------------------------------------------------
with tab3:
    st.subheader("3D Space: Live Sample vs. Nearest Baseline Point")

    if (
        "p2p_engine" not in st.session_state
        or st.session_state.get("active_train_key") != selected_train_key
    ):
        st.warning("Please calibrate the baseline model in **Tab 1** first.")
    else:
        engine = st.session_state["p2p_engine"]
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

        fig_3d.add_trace(
            go.Scatter3d(
                x=[live_x],
                y=[live_y],
                z=[live_z],
                mode="markers",
                name=f"Live Sample #{live_sample_idx}",
                marker=dict(
                    color="red" if live_res["Is_Alert"] else "green",
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
            scene=dict(xaxis_title=x_tag, yaxis_title=y_tag, zaxis_title=z_tag),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        st.plotly_chart(fig_3d, use_container_width=True)
