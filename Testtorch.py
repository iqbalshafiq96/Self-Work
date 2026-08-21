import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="Neural Net Configurator", layout="wide")
st.title("Neural Network: Interactive Architecture & Testing")

# =====================================================================
# 1. SIDEBAR CONFIGURATION
# =====================================================================
st.sidebar.header("1. Network Architecture")
num_inputs = st.sidebar.number_input("Number of Inputs", min_value=1, max_value=20, value=4)
hidden1_size = st.sidebar.slider("Layer 1 Neurons", 1, 50, 8)
hidden2_size = st.sidebar.slider("Layer 2 Neurons", 0, 50, 4)  # 0 means skip
num_outputs = st.sidebar.number_input("Number of Outputs", min_value=1, max_value=5, value=1)

# Global Activation Function applied to all hidden layers
global_activation = st.sidebar.selectbox(
    "Global Transfer Function (All Layers)", 
    ["Tanh (tansig)", "Sigmoid (logsig)", "ReLU"]
)

st.sidebar.header("2. Optimization & Data Options")
lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["SGD", "Adam"])
test_ratio = st.sidebar.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, step=0.05)


# =====================================================================
# 2. PYVIS UNLIMITED NEURON GRAPH VISUALIZER (STRICT SEQUENTIAL FLOW)
# =====================================================================
def render_pyvis_network(in_dim, h1, h2, out_dim, act_fn):
    # Dynamically scale canvas height & vertical node gap based on largest layer
    max_neurons = max(in_dim, h1, h2, out_dim)
    dynamic_height = max(480, min(max_neurons * 45, 1400))
    y_gap = max(40, min(90, 550 // max_neurons))
    
    net = Network(height=f"{dynamic_height}px", width="100%", bgcolor="rgba(0,0,0,0)", font_color="white", directed=True)
    
    # Modern font stack with larger node size (32px) & readable fonts
    net.set_options(f"""
    {{
      "nodes": {{
        "borderWidth": 2,
        "size": 32,
        "font": {{ 
          "size": 14, 
          "face": "Source Sans Pro, -apple-system, BlinkMacSystemFont, Roboto, sans-serif",
          "color": "#FFFFFF"
        }}
      }},
      "edges": {{
        "color": {{ "color": "rgba(200, 200, 200, 0.3)", "highlight": "#3498DB" }},
        "smooth": false,
        "arrows": {{ "to": {{ "enabled": true, "scaleFactor": 0.4 }} }}
      }},
      "physics": {{
        "barnesHut": {{ "gravitationalConstant": -3500, "springLength": 110, "springConstant": 0.03 }},
        "minVelocity": 0.75
      }}
    }}
    """)

    # 1. Define Layer Nodes explicitly
    input_nodes = [f"L0_N{i}" for i in range(in_dim)]
    h1_nodes = [f"L1_N{i}" for i in range(h1)]
    h2_nodes = [f"L2_N{i}" for i in range(h2)] if h2 > 0 else []
    output_nodes = [f"L3_N{i}" for i in range(out_dim)]

    # 2. Add Nodes with explicit horizontal placement
    # Input Layer
    for i, nid in enumerate(input_nodes):
        y = (i - (in_dim - 1) / 2) * y_gap
        net.add_node(
            nid, 
            label=f"Input\nNode {i+1}" if in_dim <= 3 else f"N{i+1}", 
            x=-400, 
            y=y, 
            color={"background": "#2C3E50", "border": "#5D6D7E"}, 
            shape="circle"
        )

    # Hidden Layer 1
    for i, nid in enumerate(h1_nodes):
        y = (i - (h1 - 1) / 2) * y_gap
        net.add_node(
            nid, 
            label=f"H1 [{act_fn}]\nNode {i+1}" if h1 <= 3 else f"N{i+1}", 
            x=-130 if h2 > 0 else 0, 
            y=y, 
            color={"background": "#1B4F72", "border": "#3498DB"}, 
            shape="circle"
        )

    # Hidden Layer 2 (Only created if h2 > 0)
    for i, nid in enumerate(h2_nodes):
        y = (i - (h2 - 1) / 2) * y_gap
        net.add_node(
            nid, 
            label=f"H2 [{act_fn}]\nNode {i+1}" if h2 <= 3 else f"N{i+1}", 
            x=140, 
            y=y, 
            color={"background": "#0E6251", "border": "#1ABC9C"}, 
            shape="circle"
        )

    # Output Layer
    for i, nid in enumerate(output_nodes):
        y = (i - (out_dim - 1) / 2) * y_gap
        net.add_node(
            nid, 
            label=f"Output\nNode {i+1}" if out_dim <= 3 else f"N{i+1}", 
            x=400, 
            y=y, 
            color={"background": "#7E5109", "border": "#F39C12"}, 
            shape="circle"
        )

    # 3. Connect Strictly Sequential (Input -> Layer 1 -> [Layer 2] -> Output)
    # Input -> Hidden Layer 1
    for src in input_nodes:
        for dst in h1_nodes:
            net.add_edge(src, dst)

    # Hidden Layer 1 -> Hidden Layer 2 (if exists) OR Hidden Layer 1 -> Output (if no Hidden Layer 2)
    if h2 > 0:
        for src in h1_nodes:
            for dst in h2_nodes:
                net.add_edge(src, dst)
        
        # Hidden Layer 2 -> Output Layer
        for src in h2_nodes:
            for dst in output_nodes:
                net.add_edge(src, dst)
    else:
        # Hidden Layer 1 -> Output Layer
        for src in h1_nodes:
            for dst in output_nodes:
                net.add_edge(src, dst)

    # Generate HTML content
    html_content = net.generate_html()
    
    # Custom CSS injection: Strips borders, shadows, and forces full background transparency
    custom_css = """
    <style>
        body, html, #mynetwork {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            margin: 0 !important;
            padding: 0 !important;
            font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        div.vis-network {
            border: none !important;
            outline: none !important;
        }
    </style>
    """
    
    # Inject styling into HTML head & strip PyVis inline canvas styles
    html_content = html_content.replace("</head>", f"{custom_css}</head>")
    html_content = html_content.replace("background-color: #0E1117;", "background-color: transparent;")
    
    # Embed component with dynamic height allocation
    components.html(html_content, height=dynamic_height + 10)


# =====================================================================
# 3. ARCHITECTURE VISUALIZATION & MODEL CLASS
# =====================================================================
st.subheader("Interactive Physics Network Diagram")
render_pyvis_network(num_inputs, hidden1_size, hidden2_size, num_outputs, global_activation)


class ConfigurableNet(nn.Module):
    def __init__(self, in_dim, h1, h2, act_fn_name, out_dim):
        super().__init__()
        act_map = {
            "Tanh (tansig)": nn.Tanh(),
            "Sigmoid (logsig)": nn.Sigmoid(),
            "ReLU": nn.ReLU()
        }
        
        chosen_act = act_map[act_fn_name]
        
        layers = []
        layers.append(nn.Linear(in_dim, h1))
        layers.append(chosen_act)
        
        if h2 > 0:
            layers.append(nn.Linear(h1, h2))
            layers.append(chosen_act)
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

torch.manual_seed(42)
X_raw = torch.randn(num_samples, num_inputs)
T_raw = torch.sin(X_raw[:, :1]) * 2.0 + torch.randn(num_samples, num_outputs) * 0.2

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
    st.session_state.net = ConfigurableNet(num_inputs, hidden1_size, hidden2_size, global_activation, num_outputs)
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
