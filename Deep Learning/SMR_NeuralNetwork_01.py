import requests
import urllib.parse
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

st.set_page_config(page_title="Neural Net Configurator", layout="wide")
st.title("Neural Network Modeling & Configurator")
st.caption(
    "Developed by Iqbal SHERPA 20260824. Contact me for further information"
    " @iqbalshafiq96@gmail.com")

# =====================================================================
# 1. GITHUB DIRECTORY & DATA LOADING (METADATA-DRIVEN)
# =====================================================================
GITHUB_API_DIR_URL = "https://api.github.com/repos/iqbalshafiq96/Self-Work/contents/Deep%20Learning"

@st.cache_data(ttl=300)
def fetch_csv_list_from_github():
    try:
        response = requests.get(GITHUB_API_DIR_URL)
        if response.status_code == 200:
            files = response.json()
            csv_files = [f["name"] for f in files if f["name"].endswith(".csv")]
            return csv_files
    except Exception:
        pass
    return ["SMR_Data.csv"]

available_csvs = fetch_csv_list_from_github()

st.sidebar.header("0. Dataset Selection")
selected_csv = st.sidebar.selectbox("Select CSV File from GitHub", available_csvs)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Deep%20Learning/"
selected_csv_url = GITHUB_RAW_BASE + urllib.parse.quote(selected_csv)


@st.cache_data
def load_and_preprocess_custom_csv(url_or_path):
    try:
        df_headers = pd.read_csv(url_or_path, nrows=0).columns.tolist()
        df_labels = pd.read_csv(url_or_path, skiprows=1, nrows=1, header=None).values.flatten().tolist()
        
        # Skips Row 2 (label row index 1) so data extraction starts cleanly at Row 3 and below
        df_data = pd.read_csv(url_or_path, skiprows=[1])
        df_data.columns = [str(c).strip() for c in df_data.columns]
        
        input_cols = []
        output_cols = []
        
        start_idx = 1 if "time" in df_headers[0].lower() or "date" in df_headers[0].lower() else 0
        
        for idx in range(start_idx, len(df_headers)):
            col_name = df_headers[idx].strip()
            label_val = str(df_labels[idx]).strip().lower()
            
            if "independent" in label_val or "input" in label_val:
                input_cols.append(col_name)
            elif "dependent" in label_val or "output" in label_val:
                output_cols.append(col_name)
                
        if not input_cols or not output_cols:
            all_data_cols = df_data.columns[1:] if start_idx == 1 else df_data.columns
            input_cols = list(all_data_cols[:3])
            output_cols = list(all_data_cols[3:7])

        X_raw = df_data[input_cols].values.astype(float)
        Y_raw = df_data[output_cols].values.astype(float)

        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)

        scaler_Y = StandardScaler()
        Y_scaled = scaler_Y.fit_transform(Y_raw)

        return df_data, input_cols, output_cols, X_scaled, Y_scaled, scaler_X, scaler_Y
    except Exception as e:
        raise e


try:
    (
        df_raw,
        input_names,
        output_names,
        X_norm,
        Y_norm,
        scaler_X,
        scaler_Y,
    ) = load_and_preprocess_custom_csv(selected_csv_url)
    st.sidebar.success(f"Loaded '{selected_csv}' successfully!")
except Exception as e:
    st.error(f"Failed to load dataset '{selected_csv}': {e}.")
    st.stop()

num_inputs = len(input_names)
num_outputs = len(output_names)


# =====================================================================
# 2. SIDEBAR CONFIGURATION (FLEXIBLE HIDDEN LAYERS)
# =====================================================================
st.sidebar.header("1. Network Architecture")
num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 4, 2)

hidden_layer_sizes = []
default_neurons = [12, 6, 8, 4]
for i in range(num_hidden_layers):
    default_val = default_neurons[i] if i < len(default_neurons) else 6
    neurons = st.sidebar.slider(f"Layer {i+1} Neurons", 1, 50, default_val, key=f"h_layer_{i}")
    hidden_layer_sizes.append(neurons)

global_activation = st.sidebar.selectbox(
    "Global Transfer Function (All Layers)",
    ["Tanh (tansig)", "Sigmoid (logsig)", "ReLU"],
)

activation_descriptions = {
    "Tanh (tansig)": "Outputs zero-centered values between -1 and 1. Great for continuous non-linear process dynamics.",
    "Sigmoid (logsig)": "Outputs values scaled between 0 and 1. Useful for smooth non-linear probability transitions.",
    "ReLU": "Passes positive values directly and zeroes out negative ones. Ideal for deep networks and fast convergence.",
}

st.sidebar.caption(f"ℹ {activation_descriptions[global_activation]}")

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
# 3. AUTOSCALING PYVIS NETWORK DIAGRAM WITH MULTI-LAYER SUPPORT
# =====================================================================
def render_pyvis_network(in_dim, h_sizes, out_dim):
    layer_counts = [in_dim] + h_sizes + [out_dim]
    max_neurons = max(layer_counts)
    dynamic_height = max(550, min(max_neurons * 65, 900))

    net = Network(
        height="100%",
        width="100%",
        bgcolor="rgba(0,0,0,0)",
        font_color="white",
        directed=True,
    )

    net.set_options(
        """
    {
      "nodes": {
        "borderWidth": 2,
        "size": 26,
        "font": { 
          "size": 14, 
          "face": "Segoe UI, Roboto, Helvetica, Arial, sans-serif", 
          "color": "#FFFFFF", 
          "bold": true 
        }
      },
      "edges": {
        "color": { "color": "rgba(200, 200, 200, 0.22)", "highlight": "#F1C40F" },
        "smooth": { "type": "continuous" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.4 } }
      },
      "interaction": { 
        "zoomView": false, 
        "dragView": true,
        "hover": true
      },
      "physics": { "enabled": false }
    }
    """
    )

    num_total_layers = len(layer_counts)
    x_positions = [-600 + i * (1200 / max(1, num_total_layers - 1)) for i in range(num_total_layers)]

    all_layer_nodes = []

    def get_equal_y(index, total_count):
        if total_count == 1:
            return 0
        spread_height = max(350, total_count * 45)
        return -spread_height / 2 + (index / (total_count - 1)) * spread_height

    for layer_idx, count in enumerate(layer_counts):
        current_x = x_positions[layer_idx]
        layer_nodes = []
        
        for i in range(count):
            nid = f"L{layer_idx}_N{i}"
            layer_nodes.append(nid)
            
            if layer_idx == 0:
                label_text = f"Input\n{input_names[i]}" if i < len(input_names) else f"Input\nN{i+1}"
                bg_color, border_color = "#2C3E50", "#5D6D7E"
            elif layer_idx == num_total_layers - 1:
                label_text = f"Output\n{output_names[i]}" if i < len(output_names) else f"Output\nN{i+1}"
                bg_color, border_color = "#7E5109", "#F39C12"
            else:
                label_text = " "
                palette = [("#1B4F72", "#3498DB"), ("#0E6251", "#1ABC9C"), ("#512E5F", "#8E44AD"), ("#78281F", "#E74C3C")]
                bg_color, border_color = palette[(layer_idx - 1) % len(palette)]

            net.add_node(
                nid,
                label=label_text,
                x=current_x,
                y=get_equal_y(i, count),
                color={"background": bg_color, "border": border_color},
                shape="circle",
            )
        all_layer_nodes.append(layer_nodes)

    for l_idx in range(len(all_layer_nodes) - 1):
        for src in all_layer_nodes[l_idx]:
            for dst in all_layer_nodes[l_idx + 1]:
                net.add_edge(src, dst)

    html_content = net.generate_html()

    controls_and_animation_script = """
    <style>
      html, body { width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; border: none !important; outline: none !important; }
      #mynetwork { width: 100% !important; height: 100vh !important; border: none !important; outline: none !important; }
      .diagram-controls {
        position: absolute; top: 15px; right: 20px; z-index: 9999; display: flex; align-items: center; gap: 10px;
        background: rgba(255, 255, 255, 0.25); padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); font-family: Segoe UI, sans-serif; color: #000; font-size: 13px; font-weight: 600;
      }
      .diagram-controls input[type=range] { width: 100px; height: 4px; cursor: pointer; accent-color: #3498DB; background: rgba(0,0,0,0.2); border-radius: 2px; }
      .diagram-btn {
        background: rgba(255, 255, 255, 0.5); color: #000; border: 1px solid rgba(0,0,0,0.25); border-radius: 5px;
        padding: 4px 10px; font-size: 12px; font-family: inherit; font-weight: 700; cursor: pointer; transition: all 0.2s ease;
      }
      .diagram-btn:hover { background: rgba(255, 255, 255, 0.85); border-color: #000; }
    </style>

    <div class="diagram-controls">
      <span>Zoom</span>
      <input type="range" id="zoomSlider" min="10" max="200" value="100">
      <span id="zoomValue" style="min-width: 40px; font-weight: 700;">100%</span>
      <button class="diagram-btn" id="resetZoomBtn" title="Reset view and fit to screen">🏠 Auto-Fit</button>
    </div>

    <script type="text/javascript">
    document.addEventListener("DOMContentLoaded", function() {
        var checkExist = setInterval(function() {
            if (typeof network !== 'undefined') {
                clearInterval(checkExist);
                var zoomSlider = document.getElementById("zoomSlider");
                var zoomValLabel = document.getElementById("zoomValue");
                var resetBtn = document.getElementById("resetZoomBtn");

                function fitDiagramToScreen() {
                    network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
                    setTimeout(function() {
                        var currentScale = network.getScale();
                        var pct = Math.round(currentScale * 100);
                        zoomSlider.value = pct;
                        zoomValLabel.innerText = pct + "%";
                    }, 350);
                }
                fitDiagramToScreen();
                window.addEventListener('resize', function() { network.setSize('100%', '100vh'); fitDiagramToScreen(); });
                zoomSlider.addEventListener("input", function() {
                    var val = parseFloat(this.value);
                    zoomValLabel.innerText = val + "%";
                    network.moveTo({ scale: val / 100.0 });
                });
                resetBtn.addEventListener("click", fitDiagramToScreen);
            }
        }, 100);
    });
    </script>
    </body>
    """
    html_content = html_content.replace("</body>", controls_and_animation_script)
    components.html(html_content, height=dynamic_height)


st.subheader("Interactive Architecture Diagram")
render_pyvis_network(num_inputs, hidden_layer_sizes, num_outputs)


# =====================================================================
# 4. MODEL CLASS & DATA PARTITIONING
# =====================================================================
class ConfigurableNet(nn.Module):
    def __init__(self, in_dim, h_sizes, act_fn_name, out_dim):
        super().__init__()
        act_map = {
            "Tanh (tansig)": nn.Tanh(),
            "Sigmoid (logsig)": nn.Sigmoid(),
            "ReLU": nn.ReLU(),
        }
        chosen_act = act_map[act_fn_name]

        layers = []
        prev_dim = in_dim
        for h_size in h_sizes:
            layers.append(nn.Linear(prev_dim, h_size))
            layers.append(chosen_act)
            prev_dim = h_size
        
        layers.append(nn.Linear(prev_dim, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if "net" not in st.session_state:
    st.session_state.net = None
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Correlation Matrix"

num_samples = len(X_norm)

def repartition_dataset(total_samples, current_test_ratio):
    split_idx = int(total_samples * (1 - current_test_ratio))
    indices = torch.randperm(total_samples)
    return indices[:split_idx], indices[split_idx:]

if "train_idx" not in st.session_state or "test_idx" not in st.session_state:
    st.session_state.train_idx, st.session_state.test_idx = repartition_dataset(num_samples, test_ratio)

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

X_train = X_tensor[st.session_state.train_idx]
Y_train = Y_tensor[st.session_state.train_idx]
X_test = X_tensor[st.session_state.test_idx]
Y_test = Y_tensor[st.session_state.test_idx]

st.subheader("Dataset Summary & Partitioning")
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Total Dataset Rows", num_samples)
mcol2.metric("Training Samples", X_train.shape[0])
mcol3.metric("Testing Samples", X_test.shape[0])

if st.button("Initialize / Reset Model Architecture"):
    st.session_state.train_idx, st.session_state.test_idx = repartition_dataset(num_samples, test_ratio)
    st.session_state.net = ConfigurableNet(
        num_inputs, hidden_layer_sizes, global_activation, num_outputs
    )
    st.session_state.loss_history = []
    st.success("New PyTorch Model initialized with freshly randomized Train/Test sets!")
    st.rerun()


# =====================================================================
# 5. WORKFLOW TABS
# =====================================================================
st.divider()

tab_options = [
    "Data Correlation Matrix",
    "Batch Training Phase",
    "Model Testing & Verification",
]

selected_tab = st.radio(
    "Workflow Navigation",
    options=tab_options,
    index=tab_options.index(st.session_state.active_tab) if st.session_state.active_tab in tab_options else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)


# --- TAB 0: CORRELATION MATRIX (LOWER TRIANGLE ONLY) ---
if selected_tab == "Data Correlation Matrix":
    st.write("### Feature Correlation Matrix (Lower Triangle)")

    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Aptos", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    numeric_df = df_raw.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9, "fontfamily": "sans-serif"},
    )

    ax.tick_params(labelsize=9, colors="#31333F")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    c_left, c_mid, c_right = st.columns([0.1, 0.8, 0.1])
    with c_mid:
        st.pyplot(fig, use_container_width=True)


# --- TAB 1: BATCH TRAINING ---
elif selected_tab == "Batch Training Phase":
    st.markdown("Train model parameters using training inputs (`X_train`, `Y_train`).")
    epochs = st.number_input("Number of Epochs", min_value=10, max_value=5000, value=200)

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


# --- TAB 2: MODEL TESTING & VERIFICATION ---
elif selected_tab == "Model Testing & Verification":
    st.markdown("Evaluate actual vs. predicted performance across all output variables (inverted back to engineering units).")

    if st.button("Evaluate Model on Test Set"):
        if st.session_state.net is None:
            st.warning("Please initialize and train the model first!")
        else:
            net = st.session_state.net
            net.eval()

            with torch.no_grad():
                test_preds_norm = net(X_test).numpy()
                Y_test_norm = Y_test.numpy()

                Y_test_actual = scaler_Y.inverse_transform(Y_test_norm)
                Y_test_pred = scaler_Y.inverse_transform(test_preds_norm)

            st.write("### Output Verification Trends (Actual vs. Predicted)")

            cols = st.columns(2)
            for idx, col_name in enumerate(output_names):
                with cols[idx % 2]:
                    st.markdown(f"**Output {idx+1}: {col_name}**")
                    chart_data = pd.DataFrame(
                        {
                            "Actual": Y_test_actual[:, idx],
                            "Predicted": Y_test_pred[:, idx],
                        }
                    )
                    st.line_chart(chart_data)

                    y_t = Y_test_actual[:, idx]
                    y_p = Y_test_pred[:, idx]
                    r2 = 1 - (
                        np.sum((y_t - y_p) ** 2)
                        / (np.sum((y_t - np.mean(y_t)) ** 2) + 1e-8)
                    )
                    st.caption(f"Variable R² Accuracy: {r2:.4f}")
