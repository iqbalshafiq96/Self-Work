import requests
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
st.caption(
    "Developed by Iqbal SHERPA 20260824. Contact me for further information"
    " @iqbalshafiq96@gmail.com"
)

# =====================================================================
# 1. DYNAMIC GITHUB DATASET SELECTION & PARSING
# =====================================================================
GITHUB_API_DIR_URL = (
    "https://api.github.com/repos/iqbalshafiq96/Self-Work/contents/Deep%20Learning"
)


@st.cache_data(ttl=600)
def fetch_csv_file_list():
    try:
        response = requests.get(GITHUB_API_DIR_URL)
        if response.status_code == 200:
            files = response.json()
            csv_files = [
                f["name"]
                for f in files
                if f["name"].endswith(".csv") and f["type"] == "file"
            ]
            return csv_files
    except Exception:
        pass
    return ["SMR_Data.csv"]


available_csvs = fetch_csv_file_list()
selected_csv_filename = st.sidebar.selectbox(
    "Select Dataset from 'Deep Learning'", available_csvs
)

# Construct raw download link for the chosen file
GITHUB_CSV_URL = f"https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Deep%20Learning/{selected_csv_filename}"


@st.cache_data
def load_and_preprocess_smr_data(url_or_path):
    try:
        # Read header=0 for feature names, header=1 for Independent/Dependent tags
        df_full = pd.read_csv(url_or_path, header=[0, 1])
    except Exception:
        # Fallback for local files or standard structure
        df_raw_temp = pd.read_csv(url_or_path)
        # Construct mock multiindex if standard format
        return df_raw_temp, [], [], np.array([]), np.array([]), None, None

    # Flatten columns: level 0 = feature name, level 1 = label (Independent/Dependent)
    col_names = []
    col_types = []
    for col in df_full.columns:
        col_names.append(col[0])
        col_types.append(str(col[1]).strip().lower())

    # Reconstruct standard dataframe for correlation & stats (skipping timestamp column at index 0)
    data_rows = pd.read_csv(url_or_path, skiprows=[1])
    if "Unnamed" in data_rows.columns[0] or "timestamp" in data_rows.columns[0].lower():
        df_clean = data_rows.iloc[:, 1:]
    else:
        df_clean = data_rows

    # Parse inputs and outputs based on level 1 tags (ignoring first timestamp column)
    input_names = []
    output_names = []

    for name, tag in zip(col_names[1:], col_types[1:]):
        if "independent" in tag or "input" in tag:
            input_names.append(name)
        elif "dependent" in tag or "output" in tag:
            output_names.append(name)

    # Fallback if tags are missing or unstructured
    if not input_names or not output_names:
        num_in = max(1, int(len(df_clean.columns) * 0.5))
        input_names = list(df_clean.columns[:num_in])
        output_names = list(df_clean.columns[num_in:])

    X_raw = df_clean[input_names].values
    Y_raw = df_clean[output_names].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_raw)

    return (
        df_clean,
        input_names,
        output_names,
        X_scaled,
        Y_scaled,
        scaler_X,
        scaler_Y,
    )


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
    st.sidebar.success(f"Dataset '{selected_csv_filename}' Loaded & Normalized!")
except Exception as e:
    st.error(
        f"Failed to load dataset: {e}. Please check the selected file format."
    )
    st.stop()

num_inputs = len(input_names)
num_outputs = len(output_names)


# =====================================================================
# 2. SIDEBAR CONFIGURATION (UP TO 4 HIDDEN LAYERS)
# =====================================================================
st.sidebar.header("1. Network Architecture")
num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 4, 2)

hidden1_size = st.sidebar.slider("Layer 1 Neurons", 1, 50, 12)
hidden2_size = (
    st.sidebar.slider("Layer 2 Neurons", 0, 50, 6)
    if num_hidden_layers >= 2
    else 0
)
hidden3_size = (
    st.sidebar.slider("Layer 3 Neurons", 0, 50, 4)
    if num_hidden_layers >= 3
    else 0
)
hidden4_size = (
    st.sidebar.slider("Layer 4 Neurons", 0, 50, 2)
    if num_hidden_layers >= 4
    else 0
)

# Filter out zero-neuron intermediate layers automatically
active_hidden_sizes = [
    h
    for h in [hidden1_size, hidden2_size, hidden3_size, hidden4_size][:num_hidden_layers]
    if h > 0
]
if not active_hidden_sizes:
    active_hidden_sizes = [12]  # Ensure at least one hidden layer exists

global_activation = st.sidebar.selectbox(
    "Global Transfer Function (All Layers)",
    ["Tanh (tansig)", "Sigmoid (logsig)", "ReLU"],
)

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
# 3. AUTOSCALING PYVIS NETWORK DIAGRAM WITH WINDOW RESIZE LISTENERS
# =====================================================================
def render_pyvis_network(in_dim, h_sizes, out_dim, act_fn):
    max_neurons = max([in_dim] + h_sizes + [out_dim])
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

    # Compute dynamic X coordinates based on number of layers
    total_layers = len(h_sizes) + 2
    x_positions = [
        -600 + i * (1200 / (total_layers - 1)) for i in range(total_layers)
    ]

    x_input = x_positions[0]
    x_output = x_positions[-1]
    hidden_x_coords = x_positions[1:-1]

    input_nodes = [f"L0_N{i}" for i in range(in_dim)]
    hidden_layers_nodes = [
        [f"L{layer_idx+1}_N{i}" for i in range(h_size)]
        for layer_idx, h_size in enumerate(h_sizes)
    ]
    output_nodes = [f"L{len(h_sizes)+1}_N{i}" for i in range(out_dim)]

    def get_equal_y(index, total_count):
        if total_count == 1:
            return 0
        spread_height = max(350, total_count * 50)
        return -spread_height / 2 + (index / (total_count - 1)) * spread_height

    # Input Layer
    for i, nid in enumerate(input_nodes):
        label_text = (
            f"Input\n{input_names[i]}"
            if i < len(input_names)
            else f"Input\nN{i+1}"
        )
        net.add_node(
            nid,
            label=label_text,
            x=x_input,
            y=get_equal_y(i, in_dim),
            color={"background": "#2C3E50", "border": "#5D6D7E"},
            shape="circle",
        )

    # Hidden Layers
    palette = [
        ("#1B4F72", "#3498DB"),
        ("#0E6251", "#1ABC9C"),
        ("#78281F", "#E74C3C"),
        ("#512E5F", "#9B59B6"),
    ]
    for l_idx, h_nodes in enumerate(hidden_layers_nodes):
        bg, border = palette[l_idx % len(palette)]
        for i, nid in enumerate(h_nodes):
            net.add_node(
                nid,
                label=" ",
                x=hidden_x_coords[l_idx],
                y=get_equal_y(i, len(h_nodes)),
                color={"background": bg, "border": border},
                shape="circle",
            )

    # Output Layer
    for i, nid in enumerate(output_nodes):
        label_text = (
            f"Output\n{output_names[i]}"
            if i < len(output_names)
            else f"Output\nN{i+1}"
        )
        net.add_node(
            nid,
            label=label_text,
            x=x_output,
            y=get_equal_y(i, out_dim),
            color={"background": "#7E5109", "border": "#F39C12"},
            shape="circle",
        )

    # Connections between layers
    all_layer_groups = [input_nodes] + hidden_layers_nodes + [output_nodes]
    for l_idx in range(len(all_layer_groups) - 1):
        for src in all_layer_groups[l_idx]:
            for dst in all_layer_groups[l_idx + 1]:
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
                            } else if (node.id.startsWith('L_') && node.id.includes('last')) {
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
    num_inputs, active_hidden_sizes, num_outputs, global_activation
)


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
        num_inputs, active_hidden_sizes, global_activation, num_outputs
    )
    st.session_state.loss_history = []
    st.success(
        "New PyTorch Model initialized with freshly randomized Train/Test sets!"
    )
    st.rerun()


# =====================================================================
# 5. WORKFLOW TABS (CORRELATION MATRIX, BATCH TRAINING, TESTING)
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
    index=(
        tab_options.index(st.session_state.active_tab)
        if st.session_state.active_tab in tab_options
        else 0
    ),
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)


# --- TAB 0: CORRELATION MATRIX (LOWER TRIANGLE ONLY) ---
if selected_tab == "Data Correlation Matrix":
    st.write("### Feature Correlation Matrix (Lower Triangle)")

    plt.rcParams["font.sans-serif"] = [
        "Segoe UI",
        "Aptos",
        "Arial",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    corr = df_raw.corr()
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
    st.markdown(
        "Train the model parameters using normalized training inputs (`X_train`, `Y_train`)."
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


# --- TAB 2: MODEL TESTING & VERIFICATION ---
elif selected_tab == "Model Testing & Verification":
    st.markdown(
        "Evaluate actual vs. predicted performance across all Output Variables (Inverted back to engineering units)."
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
