import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_root_cause_inspector(X_test: pd.DataFrame, anomaly_scores: pd.Series, threshold: float):
    """
    Renders the Root Cause & Feature Contribution Inspector section.
    
    Parameters:
    - X_test: Process dataset (DataFrame with feature tags as columns).
    - anomaly_scores: Series of anomaly scores corresponding to X_test index.
    - threshold: Anomaly score decision threshold.
    """
    st.subheader("🔍 Root Cause & Feature Contribution Inspector")
    
    # 1. Baseline Statistics Calculation (Normal operational baseline)
    baseline_mean = X_test.mean()
    baseline_std = X_test.std().replace(0, 1e-6)  # Avoid division by zero
    
    # Identify anomaly sample indices based on threshold
    anomaly_indices = anomaly_scores[anomaly_scores > threshold].index.tolist()
    
    # 2. Inspection Scope & Sample Selection
    if len(X_test) > 0:
        c1, c2 = st.columns([1, 2])
        with c1:
            inspect_mode = st.radio(
                "Select Inspection Scope:",
                ["Detected Anomalies Only", "All Process Samples"],
                horizontal=False
            )
        
        with c2:
            if inspect_mode == "Detected Anomalies Only":
                if len(anomaly_indices) == 0:
                    st.warning("No samples exceed the anomaly threshold. Switching to all samples.")
                    selectable_indices = X_test.index.tolist()
                else:
                    selectable_indices = anomaly_indices
            else:
                selectable_indices = X_test.index.tolist()
                
            sample_to_inspect = st.selectbox(
                "Select Sample ID to Inspect:", 
                selectable_indices
            )

        # 3. Compute Attribution & Deviation Scores for Selected Sample
        sample_row = X_test.loc[sample_to_inspect]
        
        # Calculate standard deviations offset (z-score)
        z_scores = (sample_row - baseline_mean) / baseline_std
        contribution_scores = z_scores.abs()
        
        # Build Diagnostic DataFrame for all features
        diag_df = pd.DataFrame({
            "Feature Tag": X_test.columns,
            "Actual Value": sample_row.values,
            "Baseline Mean (μ)": baseline_mean.values,
            "Deviations (Std Devs σ)": z_scores.values,
            "Contribution Score (|z - μ_k|)": contribution_scores.values
        }).sort_values(by="Contribution Score (|z - μ_k|)", ascending=False).reset_index(drop=True)

        st.markdown("---")

        # 4. Interactive Visualizations (All Features)
        d_col1, d_col2 = st.columns(2)

        # Dynamic height calculation to accommodate all features without overlapping
        dynamic_height = max(420, len(diag_df) * 25)

        with d_col1:
            fig_contrib = px.bar(
                diag_df,
                x="Contribution Score (|z - μ_k|)",
                y="Feature Tag",
                orientation="h",
                title=f"Feature Contribution Scores (Sample #{sample_to_inspect})",
                color="Contribution Score (|z - μ_k|)",
                color_continuous_scale="Reds",
                height=dynamic_height,
            )
            fig_contrib.update_layout(
                yaxis={"categoryorder": "total ascending"},
                font=dict(family="Segoe UI, Arial, sans-serif", size=12),
                margin=dict(l=20, r=20, t=40, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_contrib, use_container_width=True)

        with d_col2:
            fig_dev = px.bar(
                diag_df,
                x="Deviations (Std Devs σ)",
                y="Feature Tag",
                orientation="h",
                title=f"Feature Offsets in Standard Deviations (σ)",
                color="Deviations (Std Devs σ)",
                color_continuous_scale="RdBu_r",
                height=dynamic_height,
            )
            fig_dev.update_layout(
                yaxis={"categoryorder": "total ascending"},
                font=dict(family="Segoe UI, Arial, sans-serif", size=12),
                margin=dict(l=20, r=20, t=40, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_dev, use_container_width=True)

        # 5. Full Feature Diagnostic Table
        st.markdown("### 📋 Complete Feature Breakdown")
        st.dataframe(
            diag_df.style.format({
                "Actual Value": "{:.4f}",
                "Baseline Mean (μ)": "{:.4f}",
                "Deviations (Std Devs σ)": "{:+.2f}σ",
                "Contribution Score (|z - μ_k|)": "{:.4f}"
            }),
            use_container_width=True,
            height=350
        )
    else:
        st.info("No test data available for root cause inspection.")

# ==========================================
# EXAMPLE EXECUTION (FOR TESTING STANDALONE)
# ==========================================
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    
    # Generate mock dataset for demonstration
    np.random.seed(42)
    tags = [f"TAG_{i:02d}_PRESS" for i in range(1, 9)] + [f"TAG_{i:02d}_TEMP" for i in range(1, 9)]
    mock_data = pd.DataFrame(np.random.randn(100, len(tags)), columns=tags)
    
    # Inject synthetic anomaly at sample 45
    mock_data.iloc[45, 2] = 5.2
    mock_data.iloc[45, 10] = -4.8
    
    mock_scores = pd.Series(np.random.uniform(0.1, 0.4, 100))
    mock_scores.iloc[45] = 0.95  # Anomaly score
    
    # Render dashboard
    render_root_cause_inspector(mock_data, mock_scores, threshold=0.70)
