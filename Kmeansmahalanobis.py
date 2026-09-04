import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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
    Overall Model Residual (OMR) Engine mapping live sample points 
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
        status_container=None,
    ):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.X_train_raw = X_raw[feature_cols].copy().reset_index(drop=True)

        if status_container:
            status_container.write("Step 1/4: Standardizing feature tags...")
        if progress_bar:
            progress_bar.progress(20)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train_raw)

        if status_container:
            status_container.write("Step 2/4: Fitting 2-Nearest Neighbor graph...")
        if progress_bar:
            progress_bar.progress(40)

        self.nn_model = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(
            self.X_train_scaled
        )

        if status_container:
            status_container.write(
                f"Step 3/4: Computing {percentile}th percentile scale boundary..."
            )
        if progress_bar:
            progress_bar.progress(60)

        distances, _ = self.nn_model.kneighbors(self.X_train_scaled)
        neighbor_dists = distances[:, 1]
        self.d_99 = max(np.percentile(neighbor_dists, percentile), 1e-6)

        if status_container:
            status_container.write("Step 4/4: Finalizing k=1 lookup index...")
        if progress_bar:
            progress_bar.progress(80)

        self.nn_lookup = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
            self.X_train_scaled
        )


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
}

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
        # Using st.status ensures the UI stays in a running state until EVERYTHING finishes
        with st.status("Calibrating Baseline Model...", expanded=True) as status:
            progress_bar = st.progress(0)

            engine = OMRNearestNeighborEngine()
            engine.fit_baseline_with_progress(
                X_raw=raw_train_df,
                feature_cols=feature_cols,
                percentile=percentile_thresh,
                progress_bar=progress_bar,
                status_container=status,
            )

            status.write("Finalizing session state and memory synchronization...")
            progress_bar.progress(90)

            # Store in session state
            st.session_state["p2p_engine"] = engine
            st.session_state["active_train_key"] = selected_train_key
            st.session_state["active_feature_cols"] = feature_cols
            st.session_state["active_raw_train_df"] = raw_train_df
            st.session_state["active_percentile"] = percentile_thresh

            progress_bar.progress(100)
            progress_bar.empty()
            
            # The status box turns green ONLY as the context block exits
            status.update(
                label="Overall Model Residual Engine Calibrated Successfully!",
                state="complete",
                expanded=False,
            )

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
# TAB 2: OMR TREND & DIAGNOSTICS
# ---------------------------------------------------------
with tab2:
    st.subheader("Overall Model Residual (OMR) Trend (%) & Diagnostics")

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

        eval_results = []
        n_samples = len(raw_test_df)

        with st.status("Evaluating sample data...", expanded=False) as status_eval:
            progress_eval = st.progress(0)
            for i in range(n_samples):
                sample = raw_test_df[feature_cols].iloc[i].values
                res = engine.score_live_sample(sample)
                
                omr_val = res["OMR_pct"]
                if omr_val > 10.0:
                    status_str = "ALERT BREACH (>10%)"
                elif omr_val > 5.0:
                    status_str = "ALARM BREACH (5%–10%)"
                else:
                    status_str = "Normal (≤5%)"

                eval_results.append(
                    {
                        "Sample": i,
                        "Matched Baseline Row": res["nearest_baseline_idx"],
                        "OMR (%)": omr_val,
                        "Status": status_str,
                    }
                )
                if i % max(1, n_samples // 10) == 0:
                    progress_eval.progress(int((i + 1) / n_samples * 100))
            
            progress_eval.progress(100)
            progress_eval.empty()
            status_eval.update(label="Evaluation Complete!", state="complete")

        results_df = pd.DataFrame(eval_results)

        total_samples = len(results_df)
        total_alarms = ((results_df["OMR (%)"] > 5.0) & (results_df["OMR (%)"] <= 10.0)).sum()
        total_alerts = (results_df["OMR (%)"] > 10.0).sum()
        total_normal = (results_df["OMR (%)"] <= 5.0).sum()

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
                y=[5.0] * len(results_df),
                mode="lines",
                name="5% Alarm Threshold",
                line=dict(color="orange", dash="dash", width=2),
            )
        )
        fig_omr.add_trace(
            go.Scatter(
                x=results_df["Sample"],
                y=[10.0] * len(results_df),
                mode="lines",
                name="10% Alert Threshold",
                line=dict(color="red", dash="dash", width=2),
            )
        )

        fig_omr.update_layout(
            xaxis_title="Sample Index",
            yaxis_title="OMR (%)",
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
