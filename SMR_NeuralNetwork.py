import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

st.set_page_config(page_title="SMR Neural Net Configurator", layout="wide")
st.title("Steam Methane Reforming (SMR) Neural Network Modeling")

# =====================================================================
# 1. GITHUB DATA LOADING & PREPROCESSING (AUTOMATIC NORMALIZATION)
# =====================================================================
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/SMR_Data.csv"
)


@st.cache_data
def load_and_preprocess_smr_data(url_or_path):
    # Fallback to local path if GitHub URL fails or during local dev
    try:
        df = pd.read_csv(url_or_path)
    except Exception:
        df = pd.read_csv("SMR_Data.csv")

    # Extract 3 Inputs and 4 Outputs dynamically from the data
    input_cols = df.columns[:3]
    output_cols = df.columns[3:7]

    X_raw = df[input_cols].values
    Y_raw = df[output_cols].values

    # Z-Score Standardize Inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    # Standardize Outputs
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_raw)

    return df, input_cols, output_cols, X_scaled, Y_scaled, scaler_X, scaler_Y


try:
    (
        df_raw,
        input_names,
        output_names,
        X_norm,
        Y_norm,
        scaler_X,
        scaler_Y,
    ) = load_and_preprocess_smr_data(GITHUB_CSV_URL)
    st.sidebar.success("SMR Data Loaded & Normalized from GitHub!")
except Exception as e:
    st.error(
        f"Failed to load dataset: {e}. Please ensure SMR_Data.csv exists in the repo directory."
    )
    st.stop()

# Dynamic input/output counts inferred directly from loaded SMR data
num_inputs = len(input_names)
num_outputs = len(output_names)


# =====================================================================
# 2. SIDEBAR CONFIGURATION
# =====================================================================
st.sidebar.header("1. Network Architecture")
hidden1_size = st.sidebar.slider("Layer 1 Neurons", 1, 50, 12)
hidden2_size = st.sidebar.slider("Layer 2 Neurons", 0, 50, 6)  # 0 means skip

global_activation = st.sidebar.selectbox(
    "Global Transfer Function (All Layers)",
    ["Tanh (tansig)", "Sigmoid (logsig)", "ReLU"],
)

# Brief descriptive subtitles for each transfer function
activation_descriptions = {
    "Tanh (tansig)": "Outputs zero-centered values between -1 and 1. Great for continuous non-linear process dynamics.",
    "Sigmoid (logsig)": "Outputs values scaled between 0 and 1. Useful for smooth non-linear probability transitions.",
    "ReLU": "Passes positive values directly and zeroes out negative ones. Ideal for deep networks and fast convergence.",
}

st.sidebar.caption(f"ℹ️ {activation_descriptions[global_activation]}")

st.sidebar.header("2. Optimization & Data Options")
lr = st.sidebar.number_input(
    "Learning Rate",
    min_value=0.0001,
    max_value=1.0,
    value=0.01,
    step=0.001,
    format="%.4f",
)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["Adam", "SGD"])
test_ratio = st.sidebar.slider(
    "Test Set Split Ratio", 0.1, 0.4, 0.2, step=0.05
)


# =====================================================================
# 3. PYVIS NETWORK DIAGRAM VISUALIZER
# =====================================================================
def render_pyvis_network(in_dim, h1, h2, out_dim, act_fn):
    max_neurons = max(in_dim, h1, h2, out_dim)
    dynamic_height = max(500, min(max_neurons * 60, 1000))
    total_height = max(300, dynamic_height - 150)

    net = Network(
        height=f"{dynamic_height}px",
        width="100%",
        bgcolor="rgba(0,0,0,0)",
        font_color="white",
        directed=True,
    )

    # Disable physics forces & match Streamlit subtitle font stacks
    net.set_options(
        """
    {
      "nodes": {
        "borderWidth": 2,
        "size": 60,
        "font": { 
          "size": 16, 
          "face": "Source Sans Pro, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif", 
          "color": "#FFFFFF", 
          "bold": true 
        }
      },
      "edges": {
        "color": { "color": "rgba(200, 200, 200, 0.35)", "highlight": "#3498DB" },
        "smooth": { "type": "continuous" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }
      },
      "interaction": { "zoomView": true, "dragView": true },
      "physics": { "enabled": false }
    }
    """
    )

    # Fixed horizontal X coordinates
    x_input = -600
    x_h1 = -200
    x_h2 = 200
    x_output = 600 if h2 > 0 else x_h1 + 400

    input_nodes = [f"L0_N{i}" for i in range(in_dim)]
    h1_nodes = [f"L1_N{i}" for i in range(h1)]
    h2_nodes = [f"L2_N{i}" for i in range(h2)] if h2 > 0 else []
    output_nodes = [f"L3_N{i}" for i in range(out_dim)]

    # Helper function for equal vertical distribution across full canvas height
    def get_equal_y(index, total_count):
        if total_count == 1:
            return 0
        return -total_height / 2 + (index / (total_count - 1)) * total_height

    # 1. Input Layer
    for i, nid in enumerate(input_nodes):
        label_text = (
            f"Input\n{input_names[i]}"
            if i < len(input_names)
            else f"Input\nN{i+1}"
        )
        y_pos = get_equal_y(i, in_dim)
        net.add_node(
            nid,
            label=label_text,
            x=x_input,
            y=y_pos,
            color={"background": "#2C3E50", "border": "#5D6D7E"},
            shape="circle",
        )

    # 2. Hidden Layer 1
    for i, nid in enumerate(h1_nodes):
        y_pos = get_equal_y(i, h1)
        net.add_node(
            nid,
            label=" ",
            x=x_h1,
            y=y_pos,
            color={"background": "#1B4F72", "border": "#3498DB"},
            shape="circle",
        )

    # 3. Hidden Layer 2 (Optional)
    for i, nid in enumerate(h2_nodes):
        y_pos = get_equal_y(i, h2)
        net.add_node(
            nid,
            label=" ",
            x=x_h2,
            y=y_pos,
            color={"background": "#0E6251", "border": "#1ABC9C"},
            shape="circle",
        )

    # 4. Output Layer
    for i, nid in enumerate(output_nodes):
        label_text = (
            f"Output\n{output_names[i]}"
            if i < len(output_names)
            else f"Output\nN{i+1}"
        )
        y_pos = get_equal_y(i, out_dim)
        net.add_node(
            nid,
            label=label_text,
            x=x_output,
            y=y_pos,
            color={"background": "#7E5109", "border": "#F39C12"},
            shape="circle",
        )

    # Edge Connections
    for src in input_nodes:
        for dst in h1_nodes:
            net.add_edge(src, dst)

    if h2 > 0:
        for src in h1_nodes:
            for dst in h2_nodes:
                net.add_edge(src, dst)
        for src in h2_nodes:
            for dst in output_nodes:
                net.add_edge(src, dst)
    else:
        for src in h1_nodes:
            for dst in output_nodes:
                net.add_edge(src, dst)

    html_content = net.generate_html()
    components.html(html_content, height=dynamic_height + 10)


st.subheader("Interactive Architecture Diagram")
render_pyvis_network(
    num_inputs, hidden1_size, hidden2_size, num_outputs, global_activation
)


# =====================================================================
# 4. MODEL CLASS & DATA PARTITIONING
# =====================================================================
class ConfigurableNet(nn.Module):

    def __init__(self, in_dim, h1, h2, act_fn_name, out_dim):
        super().__init__()
        act_map = {
            "Tanh (tansig)": nn.Tanh(),
            "Sigmoid (logsig)": nn.Sigmoid(),
            "ReLU": nn.ReLU(),
        }
        chosen_act = act_map[act_fn_name]

        layers = [nn.Linear(in_dim, h1), chosen_act]
        if h2 > 0:
            layers.extend(
                [nn.Linear(h1, h2), chosen_act, nn.Linear(h2, out_dim)]
            )
        else:
            layers.append(nn.Linear(h1, out_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if "net" not in st.session_state:
    st.session_state.net = None
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []

# Partitioning Data
num_samples = len(X_norm)
split_idx = int(num_samples * (1 - test_ratio))

torch.manual_seed(42)
indices = torch.randperm(num_samples)

train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

X_train, Y_train = X_tensor[train_idx], Y_tensor[train_idx]
X_test, Y_test = X_tensor[test_idx], Y_tensor[test_idx]

st.subheader("Dataset Summary & Partitioning")
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Total SMR Rows", num_samples)
mcol2.metric("Training Samples", X_train.shape[0])
mcol3.metric("Testing Samples", X_test.shape[0])

if st.button("Initialize / Reset Model Architecture"):
    st.session_state.net = ConfigurableNet(
        num_inputs, hidden1_size, hidden2_size, global_activation, num_outputs
    )
    st.session_state.loss_history = []
    st.success("New PyTorch SMR Model initialized!")


# =====================================================================
# 5. WORKFLOW TABS
# =====================================================================
st.divider()
tab_corr, tab1, tab2, tab3 = st.tabs(
    [
        "Data Correlation Matrix",
        "Batch Training Phase",
        "Online Adaptation Phase",
        "Model Testing & Verification",
    ]
)


# --- TAB 0: CORRELATION MATRIX ---
with tab_corr:
    st.write("### SMR Data Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 5))
    corr = df_raw.corr()
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
    )
    st.pyplot(fig)


# --- TAB 1: BATCH TRAINING ---
with tab1:
    st.markdown(
        "Train the model parameters using normalized SMR training inputs (`X_train`, `Y_train`)."
    )
    epochs = st.number_input(
        "Number of Epochs", min_value=10, max_value=5000, value=200
    )

    if st.button("Run Batch Training"):
        if st.session_state.net is None:
            st.warning("Please initialize the model first!")
        else:
            net = st.session_state.net
            optimizer = (
                optim.Adam(net.parameters(), lr=lr)
                if optimizer_choice == "Adam"
                else optim.SGD(net.parameters(), lr=lr)
            )
            criterion = nn.MSELoss()

            progress_bar = st.progress(0)
            chart_place = st.empty()

            net.train()
            for epoch in range(int(epochs)):
                optimizer.zero_grad()
                output = net(X_train)
                loss = criterion(output, Y_train)
                loss.backward()
                optimizer.step()

                st.session_state.loss_history.append(loss.item())
                progress_bar.progress((epoch + 1) / int(epochs))
                chart_place.line_chart(
                    st.session_state.loss_history, y_label="MSE Training Loss"
                )

            st.success(f"Training Complete! Final Loss: {loss.item():.6f}")


# --- TAB 2: ONLINE ADAPTATION ---
with tab2:
    st.markdown(
        "Update model weights step-by-step for incoming streaming process data."
    )
    stream_size = st.number_input(
        "Streaming Samples", min_value=1, max_value=30, value=5
    )

    if st.button("Run Online Adaptation"):
        if st.session_state.net is None:
            st.warning("Please initialize the model first!")
        else:
            net = st.session_state.net
            optimizer = (
                optim.Adam(net.parameters(), lr=lr)
                if optimizer_choice == "Adam"
                else optim.SGD(net.parameters(), lr=lr)
            )
            criterion = nn.MSELoss()

            status_place, chart_place = st.empty(), st.empty()

            net.train()
            for i in range(min(int(stream_size), len(X_test))):
                x_sample, y_sample = X_test[i : i + 1], Y_test[i : i + 1]

                optimizer.zero_grad()
                pred = net(x_sample)
                adapt_loss = criterion(pred, y_sample)
                adapt_loss.backward()
                optimizer.step()

                st.session_state.loss_history.append(adapt_loss.item())
                status_place.text(
                    f"Sample {i+1}/{stream_size} | Loss: {adapt_loss.item():.6f}"
                )
                chart_place.line_chart(
                    st.session_state.loss_history, y_label="MSE Loss"
                )

            st.success("Adaptation complete!")


# --- TAB 3: MODEL TESTING & 4-OUTPUT VERIFICATION ---
with tab3:
    st.markdown(
        "Evaluate actual vs. predicted performance across all **4 Output Variables** (Inverted back to engineering units)."
    )

    if st.button("Evaluate Model on Test Set"):
        if st.session_state.net is None:
            st.warning("Please initialize and train the model first!")
        else:
            net = st.session_state.net
            net.eval()

            with torch.no_grad():
                test_preds_norm = net(X_test).numpy()
                Y_test_norm = Y_test.numpy()

                # Denormalize outputs back to real SMR engineering units
                Y_test_actual = scaler_Y.inverse_transform(Y_test_norm)
                Y_test_pred = scaler_Y.inverse_transform(test_preds_norm)

            st.write("### Output Verification Trends (Actual vs. Predicted)")

            cols = st.columns(2)
            for idx, col_name in enumerate(output_names[:4]):
                with cols[idx % 2]:
                    st.markdown(f"**Output {idx+1}: {col_name}**")
                    chart_data = pd.DataFrame(
                        {
                            "Actual": Y_test_actual[:, idx],
                            "Predicted": Y_test_pred[:, idx],
                        }
                    )
                    st.line_chart(chart_data)

                    # Compute R² score
                    y_t = Y_test_actual[:, idx]
                    y_p = Y_test_pred[:, idx]
                    r2 = 1 - (
                        np.sum((y_t - y_p) ** 2)
                        / (np.sum((y_t - np.mean(y_t)) ** 2) + 1e-8)
                    )
                    st.caption(f"Variable R² Accuracy: {r2:.4f}")
