import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Neural Net Configurator", layout="wide")
st.title("Neural Network: Training, Adaptation & Testing")

# =====================================================================
# 1. SIDEBAR CONFIGURATION
# =====================================================================
st.sidebar.header("1. Network Architecture")
num_inputs = st.sidebar.number_input("Number of Inputs", min_value=1, max_value=20, value=4)
hidden1_size = st.sidebar.slider("Layer 1 Neurons", 1, 50, 10)
activation1 = st.sidebar.selectbox("Layer 1 Activation (Transfer Fcn)", ["Tanh (tansig)", "Sigmoid (logsig)", "ReLU"])
hidden2_size = st.sidebar.slider("Layer 2 Neurons", 0, 50, 5)
num_outputs = st.sidebar.number_input("Number of Outputs", min_value=1, max_value=10, value=1)

st.sidebar.header("2. Optimization & Data Options")
lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["SGD", "Adam"])
test_ratio = st.sidebar.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, step=0.05)


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
# 3. DATA CREATION & TRAIN/TEST SPLIT
# =====================================================================
st.subheader("Dataset Configuration & Partitioning")
num_samples = st.slider("Total Dataset Size", 50, 1000, 200)

# Generate synthetic dataset
X_raw = torch.randn(num_samples, num_inputs)
# Create a target with a deterministic pattern + noise
T_raw = torch.sin(X_raw[:, :1]) * 2.0 + torch.randn(num_samples, num_outputs) * 0.2

# MATLAB equivalent to divideFcn (dividerand)
X_train_np, X_test_np, T_train_np, T_test_np = train_test_split(
    X_raw.numpy(), T_raw.numpy(), test_size=test_ratio, random_state=42
)

# Convert back to PyTorch Tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
T_train = torch.tensor(T_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
T_test = torch.tensor(T_test_np, dtype=torch.float32)

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", num_samples)
col2.metric("Training Samples", X_train.shape[0])
col3.metric("Testing Samples", X_test.shape[0])

# Initialize Model Button
if st.button("Initialize / Reset Model Architecture"):
    st.session_state.net = ConfigurableNet(num_inputs, hidden1_size, hidden2_size, activation1, num_outputs)
    st.session_state.loss_history = []
    st.success("New model initialized!")


# =====================================================================
# 4. TRAINING, ADAPTATION & TESTING TABS
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
            
            net.train() # Set model to training mode
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
            
            # --- EVALUATION PHASE ---
            net.eval() # 1. Set model to evaluation mode
            criterion = nn.MSELoss()
            
            with torch.no_grad(): # 2. Disable gradient calculations for speed and safety
                test_predictions = net(X_test)
                test_loss = criterion(test_predictions, T_test).item()
                
                # Convert predictions to numpy for analysis
                y_true = T_test.numpy().flatten()
                y_pred = test_predictions.numpy().flatten()
                
                # Calculate R-Squared Score
                r2 = r2_score(y_true, y_pred)
            
            # Display Evaluation Metrics
            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Test MSE Loss", f"{test_loss:.6f}")
            mcol2.metric("R² Score (Accuracy)", f"{r2:.4f}")
            
            # Comparison Dataframe
            results_df = pd.DataFrame({
                "Actual Target (T_test)": y_true,
                "Predicted Value (y_pred)": y_pred,
                "Absolute Error": np.abs(y_true - y_pred)
            })
            
            st.write("### Prediction vs Actual Values (Test Data)")
            st.dataframe(results_df.head(10))
            
            # Visual Comparison Chart
            st.write("### Actual vs. Predicted Curve")
            st.line_chart(results_df[["Actual Target (T_test)", "Predicted Value (y_pred)"]])
