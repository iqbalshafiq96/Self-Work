import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import graphviz

st.set_page_config(page_title="Neural Net Configurator", layout="wide")
st.title("Neural Network: Interactive Architecture & Testing")

# =====================================================================
# 1. SIDEBAR CONFIGURATION
# =====================================================================
st.sidebar.header("1. Network Architecture")
num_inputs = st.sidebar.number_input("Number of Inputs", min_value=1, max_value=20, value=4)
hidden1_size = st.sidebar.slider("Layer 1 Neurons", 1, 50, 8)
activation1 = st.sidebar.selectbox("Layer 1 Activation (Transfer Fcn)", ["Tanh (tansig)", "Sigmoid (logsig)", "ReLU"])
hidden2_size = st.sidebar.slider("Layer 2 Neurons", 0, 50, 4)  # 0 means skip
num_outputs = st.sidebar.number_input("Number of Outputs", min_value=1, max_value=5, value=1)

st.sidebar.header("2. Optimization & Data Options")
lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["SGD", "Adam"])
test_ratio = st.sidebar.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, step=0.05)


# =====================================================================
# 2. TRUNCATED COMPACT NETWORK VISUALIZER (CLEAN NEURON DOTS)
# =====================================================================
def draw_neural_network(in_dim, h1, h2, out_dim, act_fn, max_display=5):
    dot = graphviz.Digraph(comment="Neural Network Architecture")
    
    # Ultra-compact graph properties
    dot.attr(rankdir="LR", dpi="72", ranksep="0.3", nodesep="0.15")
    dot.attr(
        "node", 
        shape="circle", 
        style="filled", 
        color="#2E86C1", 
        fontcolor="white", 
        fontname="Segoe UI",
        width="0.22",       # Compact node size
        height="0.22",
        fixedsize="true",
        fontsize="7"
    )
    dot.attr("edge", arrowsize="0.3")

    # Helper function to get node representation list (truncates if large)
    def get_layer_nodes(count, prefix):
        if count <= max_display:
            return [(f"{prefix}_{i}", False) for i in range(count)]
        else:
            # Show first 3 nodes, an ellipsis node, and the last node
            nodes = [(f"{prefix}_{i}", False) for i in range(3)]
            nodes.append((f"{prefix}_dots", True))  # True indicates ellipsis node
            nodes.append((f"{prefix}_{count-1}", False))
            return nodes

    # 1. Input Layer
    input_nodes = get_layer_nodes(in_dim, "I")
    with dot.subgraph(name="cluster_input") as c:
        c.attr(color="white", label=f"Input ({in_dim})", fontsize="9")
        for node_id, is_dots in input_nodes:
            label = "..." if is_dots else ""  # Blank for clean neurons
            color = "#7F8C8D" if is_dots else "#34495E"
            c.node(node_id, label, fillcolor=color)

    # 2. Hidden Layer 1
    h1_nodes = get_layer_nodes(h1, "H1")
    with dot.subgraph(name="cluster_h1") as c:
        c.attr(color="white", label=f"Hidden 1 ({h1})\n[{act_fn}]", fontsize="9")
        for node_id, is_dots in h1_nodes:
            label = "..." if is_dots else ""
            color = "#7F8C8D" if is_dots else "#2980B9"
            c.node(node_id, label, fillcolor=color)

    # Connect Input -> Hidden 1
    for i_id, _ in input_nodes:
        for h1_id, _ in h1_nodes:
            dot.edge(i_id, h1_id, color="#BDC3C7", arrowhead="none")

    prev_nodes = h1_nodes

    # 3. Hidden Layer 2 (Optional)
    if h2 > 0:
        h2_nodes = get_layer_nodes(h2, "H2")
        with dot.subgraph(name="cluster_h2") as c:
            c.attr(color="white", label=f"Hidden 2 ({h2})\n[{act_fn}]", fontsize="9")
            for node_id, is_dots in h2_nodes:
                label = "..." if is_dots else ""
                color = "#7F8C8D" if is_dots else "#16A085"
                c.node(node_id, label, fillcolor=color)

        for h1_id, _ in h1_nodes:
            for h2_id, _ in h2_nodes:
                dot.edge(h1_id, h2_id, color="#BDC3C7", arrowhead="none")

        prev_nodes = h2_nodes

    # 4. Output Layer
    output_nodes = get_layer_nodes(out_dim, "O")
    with dot.subgraph(name="cluster_output") as c:
        c.attr(color="white", label=f"Output ({out_dim})\n[Linear]", fontsize="9")
        for node_id, is_dots in output_nodes:
            label = "..." if is_dots else ""
            color = "#7F8C8D" if is_dots else "#D35400"
            c.node(node_id, label, fillcolor=color)

    for p_id, _ in prev_nodes:
        for o_id, _ in output_nodes:
            dot.edge(p_id, o_id, color="#BDC3C7", arrowhead="none")

    return dot


# =====================================================================
# 3. ARCHITECTURE VISUALIZATION & MODEL CLASS
# =====================================================================
st.subheader("Network Diagram & Visual Representation")

# Place diagram inside a bounded column layout
v_col1, v_col2, v_col3 = st.columns([1, 3, 1])
with v_col2:
    net_graph = draw_neural_network(num_inputs, hidden1_size, hidden2_size, num_outputs, activation1)
    st.graphviz_chart(net_graph, use_container_width=True)


class ConfigurableNet(nn.Module):
    def __init__(self, in_dim, h1, h2, act_fn_name, out_dim):
        super().__init__()
        act_map = {
            "Tanh (tansig)": nn.Tanh(),
            "Sigmoid (logsig)": nn.Sigmoid(),
            "ReLU": nn.ReLU()
        }
        
        layers = []
        layers.append(nn.Linear(in_dim, h1))
        layers.append(act_map[act_fn_name])
        
        if h2 > 0:
            layers.append(nn.Linear(h1, h2))
            layers.append(act_map[act_fn_name])
            layers.append(nn.Linear(h2, out_dim))
        else:
            layers.append(nn.Linear(h1, out_dim))
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

if "net" not in st.session_state:
    st.session_state.net = None
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []


# =====================================================================
# 4. DATA CREATION & TRAIN/TEST SPLIT (PURE PYTORCH)
# =====================================================================
st.subheader("Dataset Configuration & Partitioning")
num_samples = st.slider("Total Dataset Size", 50, 1000, 200)

# Generate synthetic dataset
torch.manual_seed(42)
X_raw = torch.randn(num_samples, num_inputs)
T_raw = torch.sin(X_raw[:, :1]) * 2.0 + torch.randn(num_samples, num_outputs) * 0.2

# Partition dataset using PyTorch indices
split_idx = int(num_samples * (1 - test_ratio))
indices = torch.randperm(num_samples)

train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_train, T_train = X_raw[train_idx], T_raw[train_idx]
X_test, T_test = X_raw[test_idx], T_raw[test_idx]

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", num_samples)
col2.metric("Training Samples", X_train.shape[0])
col3.metric("Testing Samples", X_test.shape[0])

if st.button("Initialize / Reset Model Architecture"):
    st.session_state.net = ConfigurableNet(num_inputs, hidden1_size, hidden2_size, activation1, num_outputs)
    st.session_state.loss_history = []
    st.success("New model initialized!")


# =====================================================================
# 5. TRAINING, ADAPTATION & TESTING TABS
# =====================================================================
st.divider()
tab1, tab2, tab3 = st.tabs(["Batch Training Phase", "Online Adaptation Phase", "Model Testing Phase"])

def get_optimizer(model):
    if optimizer_choice == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    return optim.Adam(model.parameters(), lr=lr)


# --- TAB 1: BATCH TRAINING ---
with tab1:
    st.markdown("Train model parameters using the **Training Set** (`X_train`, `T_train`).")
    epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=100)
    
    if st.button("Run Batch Training"):
        if st.session_state.net is None:
            st.warning("Please initialize the model first!")
        else:
            net = st.session_state.net
            optimizer = get_optimizer(net)
            criterion = nn.MSELoss()
            
            progress_bar = st.progress(0)
            chart_place = st.empty()
            
            net.train()
            for epoch in range(int(epochs)):
                optimizer.zero_grad()
                output = net(X_train)
                loss = criterion(output, T_train)
                loss.backward()
                optimizer.step()
                
                st.session_state.loss_history.append(loss.item())
                progress_bar.progress((epoch + 1) / int(epochs))
                chart_place.line_chart(st.session_state.loss_history, y_label="MSE Training Loss")
            
            st.success(f"Training Complete! Final MSE Loss: {loss.item():.6f}")


# --- TAB 2: ONLINE ADAPTATION ---
with tab2:
    st.markdown("Update weights online sample-by-sample on streaming input data.")
    stream_size = st.number_input("Streaming Samples", min_value=1, max_value=50, value=10)
    
    if st.button("Run Online Adaptation"):
        if st.session_state.net is None:
            st.warning("Please initialize the model first!")
        else:
            net = st.session_state.net
            optimizer = get_optimizer(net)
            criterion = nn.MSELoss()
            
            stream_X = torch.randn(int(stream_size), num_inputs)
            stream_T = torch.randn(int(stream_size), num_outputs)
            
            status_place = st.empty()
            chart_place = st.empty()
            
            net.train()
            for i in range(int(stream_size)):
                x_sample = stream_X[i:i+1]
                t_sample = stream_T[i:i+1]
                
                optimizer.zero_grad()
                pred = net(x_sample)
                adapt_loss = criterion(pred, t_sample)
                adapt_loss.backward()
                optimizer.step()
                
                st.session_state.loss_history.append(adapt_loss.item())
                status_place.text(f"Sample {i+1}/{stream_size} | Loss: {adapt_loss.item():.6f}")
                chart_place.line_chart(st.session_state.loss_history, y_label="MSE Loss")
                
            st.success("Adaptation complete!")


# --- TAB 3: MODEL TESTING & EVALUATION ---
with tab3:
    st.markdown("Evaluate performance on unseen data (`X_test`, `T_test`).")
    
    if st.button("Evaluate Model on Test Set"):
        if st.session_state.net is None:
            st.warning("Please initialize and train the model first!")
        else:
            net = st.session_state.net
            
            net.eval()
            criterion = nn.MSELoss()
            
            with torch.no_grad():
                test_predictions = net(X_test)
                test_loss = criterion(test_predictions, T_test).item()
                
                y_true = T_test.numpy().flatten()
                y_pred = test_predictions.numpy().flatten()
                
                target_mean = np.mean(y_true)
                ss_tot = np.sum((y_true - target_mean) ** 2)
                ss_res = np.sum((y_true - y_pred) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Test MSE Loss", f"{test_loss:.6f}")
            mcol2.metric("R² Score (Accuracy)", f"{r2:.4f}")
            
            results_df = pd.DataFrame({
                "Actual Target (T_test)": y_true,
                "Predicted Value (y_pred)": y_pred,
                "Absolute Error": np.abs(y_true - y_pred)
            })
            
            st.write("### Prediction vs Actual Values (Test Data)")
            st.dataframe(results_df.head(10))
            
            st.write("### Actual vs. Predicted Curve")
            st.line_chart(results_df[["Actual Target (T_test)", "Predicted Value (y_pred)"]])
