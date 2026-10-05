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
import requests
import io
import plotly.figure_factory as ff

st.set_page_config(page_title="Neural Net Configurator", layout="wide")
st.title("Process Neural Network Modeling")
st.caption("Developed by Iqbal SHERPA 20260824. Contact me for further information @iqbalshafiq96@gmail.com")

# =====================================================================
# 0. GITHUB DIRECTORY CSV DISCOVERY & PARSING
# =====================================================================
GITHUB_API_URL = "https://api.github.com/repos/iqbalshafiq96/Self-Work/contents/Deep%20Learning"

@st.cache_data(ttl=600)
def fetch_csv_file_list():
    try:
        response = requests.get(GITHUB_API_URL)
        if response.status_code == 200:
            files = response.json()
            csv_files = [f["name"] for f in files if f["name"].lower().endswith(".csv")]
            return csv_files
    except Exception:
        pass
    return ["SMR_Data.csv"]

available_csvs = fetch_csv_file_list()

st.sidebar.header("0. Dataset Selection")
selected_csv_filename = st.sidebar.selectbox("Select CSV File from GitHub", available_csvs)

# Auto-detect if user switched CSV file and reset session states
if "last_selected_csv" not in st.session_state or st.session_state.last_selected_csv != selected_csv_filename:
    st.session_state.last_selected_csv = selected_csv_filename
    st.session_state.net = None
    st.session_state.loss_history = []
    if "train_idx" in st.session_state:
        del st.session_state.train_idx
    if "test_idx" in st.session_state:
        del st.session_state.test_idx

encoded_filename = selected_csv_filename.replace(" ", "%20")
GITHUB_CSV_URL = f"https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Deep%20Learning/{encoded_filename}"

@st.cache_data
def load_and_preprocess_custom_csv(url_or_path):
    try:
        preview_df = pd.read_csv(url_or_path, nrows=3, header=None)
    except Exception:
        preview_df = pd.read_csv("SMR_Data.csv", nrows=3, header=None)

    col_names = preview_df.iloc[0].values[1:]  # Skip timestamp column (index 0)
    col_types = preview_df.iloc[1].values[1:]  # 'Independent' or 'Dependent'

    try:
        full_df = pd.read_csv(url_or_path, header=None, skiprows=2)
    except Exception:
        full_df = pd.read_csv("SMR_Data.csv", header=None, skiprows=2)

    data_df = full_df.iloc[:, 1:].copy()
    data_df.columns = col_names
    data_df = data_df.apply(pd.to_numeric, errors="coerce").dropna()

    input_names = [col_names[i] for i, t in enumerate(col_types) if str(t).strip().lower() == "independent"]
    output_names = [col_names[i] for i, t in enumerate(col_types) if str(t).strip().lower() == "dependent"]

    if not input_names or not output_names:
        input_names = list(data_df.columns[:3])
        output_names = list(data_df.columns[3:7])

    X_raw = data_df[input_names].values
    Y_raw = data_df[output_names].values

    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    Y_raw = np.nan_to_num(Y_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_raw)

    return data_df, input_names, output_names, X_scaled, Y_scaled, scaler_X, scaler_Y

try:
    (
        df_raw,
        input_names,
        output_names,
        X_norm,
        Y_norm,
        scaler_X,
        scaler_Y,
    ) = load_and_preprocess_custom_csv(GITHUB_CSV_URL)
    st.sidebar.success(f"Loaded '{selected_csv_filename}' successfully!")
except Exception as e:
    st.error(f"Failed to load dataset: {e}. Please ensure valid format.")
    st.stop()

num_inputs = len(input_names)
num_outputs = len(output_names)


# =====================================================================
# 1. SIDEBAR CONFIGURATION (UP TO 3 HIDDEN LAYERS)
# =====================================================================
st.sidebar.header("1. Network Architecture")
hidden1_size = st.sidebar.slider("Layer 1 Neurons", 0, 50, 12)
hidden2_size = st.sidebar.slider("Layer 2 Neurons", 0, 50, 6)
hidden3_size = st.sidebar.slider("Layer 3 Neurons", 0, 50, 0)

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
# 2. AUTOSCALING PYVIS NETWORK DIAGRAM
# =====================================================================
def render_pyvis_network(in_dim, h1, h2, h3, out_dim, act_fn):
    active_h = [h for h in [h1, h2, h3] if h > 0]
    max_neurons = max([in_dim, out_dim] + active_h if active_h else [in_dim, out_dim])
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
        "size": 28,
        "font": { 
          "size": 15, 
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

    layers_list = [in_dim] + active_h + [out_dim]
    num_layer_cols = len(layers_list)
    x_coords = np.linspace(-600, 600, num_layer_cols)

    input_nodes = [f"L0_N{i}" for i in range(in_dim)]
    layer_node_groups = [input_nodes]

    for idx, h_val in enumerate(active_h):
        layer_node_groups.append([f"L{idx+1}_N{i}" for i in range(h_val)])

    output_layer_idx = len(layer_node_groups)
    output_nodes = [f"L{output_layer_idx}_N{i}" for i in range(out_dim)]
    layer_node_groups.append(output_nodes)

    def get_equal_y(index, total_count):
        if total_count == 1:
            return 0
        spread_height = max(350, total_count * 50)
        return -spread_height / 2 + (index / (total_count - 1)) * spread_height

    for col_idx, nodes in enumerate(layer_node_groups):
        x_pos = x_coords[col_idx]
        for i, nid in enumerate(nodes):
            y_pos = get_equal_y(i, len(nodes))
            if col_idx == 0:
                label_text = f"Input\n{input_names[i]}" if i < len(input_names) else f"Input\nN{i+1}"
                bg, border = "#2C3E50", "#5D6D7E"
            elif col_idx == num_layer_cols - 1:
                label_text = f"Output\n{output_names[i]}" if i < len(output_names) else f"Output\nN{i+1}"
                bg, border = "#7E5109", "#F39C12"
            else:
                label_text = " "
                bg, border = "#1B4F72", "#3498DB"

            net.add_node(
                nid,
                label=label_text,
                x=x_pos,
                y=y_pos,
                color={"background": bg, "border": border},
                shape="circle",
            )

    for col_idx in range(len(layer_node_groups) - 1):
        src_layer = layer_node_groups[col_idx]
        dst_layer = layer_node_groups[col_idx + 1]
        for src in src_layer:
            for dst in dst_layer:
                net.add_edge(src, dst)

    html_content = net.generate_html()

    controls_and_animation_script = """
    <style>
      html, body {
        width: 100%;
        height: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
        border: none !important;
        outline: none !important;
      }
      #mynetwork {
        width: 100% !important;
        height: 100vh !important;
        border: none !important;
        outline: none !important;
      }
      .diagram-controls {
        position: absolute;
        top: 15px;
        right: 20px;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.25);
        padding: 6px 14px;
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        font-family: Segoe UI, -apple-system, Roboto, sans-serif;
        color: #000000;
        font-size: 13px;
        font-weight: 600;
      }
      .diagram-controls input[type=range] {
        width: 100px;
        height: 4px;
        cursor: pointer;
        accent-color: #3498DB;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 2px;
      }
      .diagram-btn {
        background: rgba(255, 255, 255, 0.5);
        color: #000000;
        border: 1px solid rgba(0, 0, 0, 0.25);
        border-radius: 5px;
        padding: 4px 10px;
        font-size: 12px;
        font-family: inherit;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.2s ease;
      }
      .diagram-btn:hover {
        background: rgba(255, 255, 255, 0.85);
        border-color: #000000;
      }
    </style>

    <div class="diagram-controls">
      <span style="color: #000000;">Zoom</span>
      <input type="range" id="zoomSlider" min="10" max="200" value="100">
      <span id="zoomValue" style="min-width: 40px; font-weight: 700; color: #000000;">100%</span>
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
                    network.fit({
                        animation: { duration: 300, easingFunction: "easeInOutQuad" }
                    });
                    setTimeout(function() {
                        var currentScale = network.getScale();
                        var pct = Math.round(currentScale * 100);
                        zoomSlider.value = pct;
                        zoomValLabel.innerText = pct + "%";
                    }, 350);
                }

                fitDiagramToScreen();

                window.addEventListener('resize', function() {
                    network.setSize('100%', '100vh');
                    fitDiagramToScreen();
                });

                zoomSlider.addEventListener("input", function() {
                    var val = parseFloat(this.value);
                    zoomValLabel.innerText = val + "%";
                    var scaleFactor = val / 100.0;
                    network.moveTo({ scale: scaleFactor });
                });

                resetBtn.addEventListener("click", function() {
                    fitDiagramToScreen();
                });

                var particles = [];
                var edgeList = edges.get();
                var nodeList = nodes.get();

                var particleCount = Math.min(edgeList.length, 35);
                for (var i = 0; i < particleCount; i++) {
                    var edge = edgeList[i % edgeList.length];
                    particles.push({
                        from: edge.from,
                        to: edge.to,
                        progress: Math.random(),
                        speed: 0.002 + Math.random() * 0.004,
                        sparklePhase: Math.random() * Math.PI * 2,
                        sparkleSpeed: 0.05 + Math.random() * 0.1
                    });
                }

                var globalPhase = 0;

                network.on("afterDrawing", function(ctx) {
                    globalPhase += 0.04;

                    nodeList.forEach(function(node) {
                        var pos = network.getPositions([node.id])[node.id];
                        var box = network.getBoundingBox(node.id);

                        if (pos && box) {
                            var actualRadius = (box.right - box.left) / 2;
                            var beamColor = 'rgba(52, 152, 219, ';
                            var localPhase = globalPhase;

                            if (node.id.startsWith('L0_')) {
                                localPhase += 0.5;
                                beamColor = 'rgba(93, 109, 126, ';
                            } else if (node.id.includes('L3_') || node.id.includes('L2_') && !node.id.startsWith('L2_N')) {
                                localPhase += 1.0;
                                beamColor = 'rgba(243, 156, 18, ';
                            }

                            var pulseIntensity = 0.5 + 0.5 * Math.sin(localPhase);
                            var strokeWidth = 1.5 + (pulseIntensity * 2.5);
                            var alpha = 0.5 + (pulseIntensity * 0.5);

                            ctx.beginPath();
                            ctx.arc(pos.x, pos.y, actualRadius, 0, 2 * Math.PI, false);
                            ctx.strokeStyle = beamColor + alpha + ')';
                            ctx.lineWidth = strokeWidth;
                            ctx.shadowColor = beamColor + '1.0)';
                            ctx.shadowBlur = 6 * pulseIntensity;
                            ctx.stroke();
                            ctx.shadowBlur = 0;
                        }
                    });

                    particles.forEach(function(p) {
                        var fromPos = network.getPositions([p.from])[p.from];
                        var toPos = network.getPositions([p.to])[p.to];

                        if (fromPos && toPos) {
                            p.progress += p.speed;
                            p.sparklePhase += p.sparkleSpeed;

                            if (p.progress >= 0.95) {
                                p.progress = 0.05;
                                var randEdge = edgeList[Math.floor(Math.random() * edgeList.length)];
                                p.from = randEdge.from;
                                p.to = randEdge.to;
                            }

                            var currX = fromPos.x + (toPos.x - fromPos.x) * p.progress;
                            var currY = fromPos.y + (toPos.y - fromPos.y) * p.progress;

                            var sparkle = 0.4 + 0.6 * Math.sin(p.sparklePhase);
                            var opacity = (0.3 + 0.7 * sparkle).toFixed(2);

                            ctx.beginPath();
                            ctx.arc(currX, currY, 3, 0, 2 * Math.PI, false);
                            ctx.fillStyle = 'rgba(255, 215, 0, ' + (opacity * 0.3) + ')';
                            ctx.fill();

                            ctx.beginPath();
                            ctx.arc(currX, currY, 1.5, 0, 2 * Math.PI, false);
                            ctx.fillStyle = 'rgba(255, 223, 0, ' + opacity + ')';
                            ctx.fill();
                        }
                    });
                });

                function animate() {
                    network.redraw();
                    requestAnimationFrame(animate);
                }
                animate();
            }
        }, 100);
    });
    </script>
    </body>
    """

    html_content = html_content.replace("</body>", controls_and_animation_script)
    components.html(html_content, height=dynamic_height)


st.subheader("Interactive Architecture Diagram")
render_pyvis_network(
    num_inputs, hidden1_size, hidden2_size, hidden3_size, num_outputs, global_activation
)


# =====================================================================
# 3. CONFIGURABLE MODEL CLASS (UP TO 3 HIDDEN LAYERS)
# =====================================================================
class ConfigurableNet(nn.Module):

    def __init__(self, in_dim, h1, h2, h3, act_fn_name, out_dim):
        super().__init__()
        act_map = {
            "Tanh (tansig)": nn.Tanh(),
            "Sigmoid (logsig)": nn.Sigmoid(),
            "ReLU": nn.ReLU(),
        }
        chosen_act = act_map[act_fn_name]

        layers = []
        prev_dim = in_dim

        for h_size in [h1, h2, h3]:
            if h_size > 0:
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

if "train_idx" not in st.session_state or "test_idx" not in st.session_state or len(st.session_state.train_idx) + len(st.session_state.test_idx) != num_samples:
    st.session_state.train_idx, st.session_state.test_idx = repartition_dataset(
        num_samples, test_ratio
    )

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
    st.session_state.train_idx, st.session_state.test_idx = repartition_dataset(
        num_samples, test_ratio
    )
    st.session_state.net = ConfigurableNet(
        num_inputs, hidden1_size, hidden2_size, hidden3_size, global_activation, num_outputs
    )
    st.session_state.loss_history = []
    st.success("New PyTorch Model initialized with freshly randomized Train/Test sets!")
    st.rerun()


# =====================================================================
# 4. WORKFLOW TABS
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

    corr = df_raw.corr()
    
    # Mask out upper triangle (keep diagonal)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.copy()
    corr_masked[mask] = np.nan

    z = corr_masked.values
    x = list(corr.columns)
    y = list(corr.index)
    text = np.round(z, 2)

    # Custom text color logic: white for deep blue (> 0.6), black otherwise
    text_colors = np.where(np.nan_to_num(z) > 0.6, "white", "black")

    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=text,
        colorscale="Blues",
        zmin=-1,
        zmax=1,
        showscale=True,
        font_colors=text_colors.tolist()
    )

    fig.update_layout(
        width=700,
        height=500,
        xaxis=dict(tickangle=-45),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    c_left, c_mid, c_right = st.columns([0.1, 0.8, 0.1])
    with c_mid:
        st.plotly_chart(fig, use_container_width=True)


# --- TAB 1: BATCH TRAINING ---
elif selected_tab == "Batch Training Phase":
    st.markdown("Train the model parameters using normalized training inputs (`X_train`, `Y_train`).")
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
                
                if torch.isnan(loss):
                    st.error("Training encountered NaN loss. Try reducing learning rate or changing activation.")
                    break

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
    st.markdown("Evaluate actual vs. predicted performance across output variables (Inverted back to engineering units).")

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
