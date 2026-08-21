import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

st.set_page_config(page_title="Neural Net Configurator", layout="wide")
st.title("Neural Network: Batch Training vs. Online Adaptation")

# =====================================================================
# 1. SIDEBAR CONFIGURATION (Layers, Neurons, Transfer Functions)
# =====================================================================
st.sidebar.header("1. Network Architecture")
num_inputs = st.sidebar.number_input("Number of Inputs", min_value=1, max_value=20, value=4)
hidden1_size = st.sidebar.slider("Layer 1 Neurons", 1, 50, 10)
activation1 = st.sidebar.selectbox("Layer 1 Activation (Transfer Fcn)", ["Tanh (tansig)", "Sigmoid (logsig)", "ReLU"])
hidden2_size = st.sidebar.slider("Layer 2 Neurons", 0, 50, 5) # 0 means skip 2nd hidden layer
num_outputs = st.sidebar.number_input("Number of Outputs", min_value=1, max_value=10, value=1)

st.sidebar.header("2. Training & Adaptation Options")
lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001)
optimizer_choice = st.sidebar.selectbox("Optimizer (trainFcn / adaptFcn)", ["SGD", "Adam"])


# =====================================================================
# 2. DYNAMIC PYTORCH MODEL CLASS
# =====================================================================
class ConfigurableNet(nn.Module):
    def __init__(self, in_dim, h1, h2, act_fn_name, out_dim):
        super().__init__()
        act_map = {
            "Tanh (tansig)": nn.Tanh(),
            "Sigmoid (logsig)": nn.Sigmoid(),
            "ReLU": nn.ReLU()
        }
        
        layers = []
        # Layer 1
        layers.append(nn.Linear(in_dim, h1))
        layers.append(act_map[act_fn_name])
        
        # Optional Layer 2
        if h2 > 0:
            layers.append(nn.Linear(h1, h2))
            layers.append(act_map[act_fn_name])
            layers.append(nn.Linear(h2, out_dim))
        else:
            layers.append(nn.Linear(h1, out_dim))
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Initialize Session State to keep model across UI interactions
if "net" not in st.session_state:
    st.session_state.net = None
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []


# =====================================================================
# 3. DATA CREATION & MODEL INSTANTIATION
# =====================================================================
st.subheader("Current Dataset")
num_samples = st.slider("Dataset Size (Batch)", 20, 500, 100)

# Generate synthetic dataset matching sidebar inputs/outputs
X = torch.randn(num_samples, num_inputs)
T = torch.randn(num_samples, num_outputs)

col1, col2 = st.columns(2)

with col1:
    st.write("**Inputs (X preview):**")
    st.dataframe(pd.DataFrame(X.numpy()).head())

with col2:
    st.write("**Targets (T preview):**")
    st.dataframe(pd.DataFrame(T.numpy()).head())

# Initialize Model Button
if st.button("Initialize / Reset Model Architecture"):
    st.session_state.net = ConfigurableNet(num_inputs, hidden1_size, hidden2_size, activation1, num_outputs)
    st.session_state.loss_history = []
    st.success("New model initialized with selected architecture!")


# =====================================================================
# 4. IMPLEMENTATION PHASES: BATCH TRAIN & ONLINE ADAPT
# =====================================================================
st.divider()
tab1, tab2 = st.tabs(["Batch Training Phase (train)", "Online Adaptation Phase (adapt)"])

# Helper function to get optimizer
def get_optimizer(model):
    if optimizer_choice == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    return optim.Adam(model.parameters(), lr=lr)


# --- TAB 1: BATCH TRAINING ---
with tab1:
    st.markdown("Updates weights using the **entire batch** at once over $N$ epochs.")
    epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=50)
    
    if st.button("Run Batch Training (`train`)"):
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
                output = net(X)
                loss = criterion(output, T)
                loss.backward()
                optimizer.step()
                
                st.session_state.loss_history.append(loss.item())
                progress_bar.progress((epoch + 1) / int(epochs))
                
                # Real-time loss plot update
                chart_place.line_chart(st.session_state.loss_history, y_label="MSE Loss")
            
            st.success(f"Final Batch MSE Loss: {loss.item():.6f}")


# --- TAB 2: ONLINE ADAPTATION ---
with tab2:
    st.markdown("Updates weights **sample-by-sample** dynamically in real time.")
    stream_size = st.number_input("Number of Streaming Samples to Process", min_value=1, max_value=50, value=10)
    
    if st.button("Run Online Adaptation (`adapt`)"):
        if st.session_state.net is None:
            st.warning("Please initialize the model first!")
        else:
            net = st.session_state.net
            optimizer = get_optimizer(net)
            criterion = nn.MSELoss()
            
            # Generate new dynamic streaming samples
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
                optimizer.step()  # Weight updated per single sample
                
                st.session_state.loss_history.append(adapt_loss.item())
                status_place.text(f"Processing Streamed Sample {i+1}/{stream_size} | Loss: {adapt_loss.item():.6f}")
                chart_place.line_chart(st.session_state.loss_history, y_label="MSE Loss")
                
            st.success("Online adaptation completed for incoming stream!")
