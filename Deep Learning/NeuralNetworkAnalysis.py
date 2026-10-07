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
import plotly.express as px
import requests
import io

st.set_page_config(page_title="Neural Network Configurator", layout="wide")
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

st.write("### Dataset Selection")
selected_csv_filename = st.selectbox("Select CSV File from GitHub", available_csvs, label_visibility="collapsed")

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
    st.success(f"Loaded '{selected_csv_filename}' successfully!")
except Exception as e:
    st.error(f"Failed to load dataset: {e}. Please ensure valid format.")
    st.stop()

num_inputs = len(input_names)
num_outputs = len(output_names)

st.divider()

# =====================================================================
# 1. NETWORK ARCHITECTURE CONFIGURATION
# =====================================================================
st.subheader("Interactive Architecture Diagram")

col_arch1, col_arch2, col_arch3, col_arch4 = st.columns(4)
with col_arch1:
    hidden1_size = st.slider("Layer 1 Neurons", 0, 50, 6)
with col_arch2:
    hidden2_size = st.slider("Layer 2 Neurons", 0, 50, 3)
with col_arch3:
    hidden3_size = st.slider("Layer 3 Neurons", 0, 50, 0)
with col_arch4:
    global_activation = st.selectbox(
        "Global Transfer Function",
        ["Tanh (tansig)", "Sigmoid (logsig)", "ReLU"],
    )

activation_descriptions = {
    "Tanh (tansig)": "Outputs zero-centered values between -1 and 1. Great for continuous non-linear process dynamics.",
    "Sigmoid (logsig)": "Outputs values scaled between 0 and 1. Useful for smooth non-linear probability transitions.",
    "ReLU": "Passes positive values directly and zeroes out negative ones. Ideal for deep networks and fast convergence.",
}
st.caption(f"ℹ {activation_descriptions[global_activation]}")


# =====================================================================
# 2. AUTOSCALING PYVIS NETWORK DIAGRAM  (Synaptic impulse animation)
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

    # NOTE: edges are straight (smooth disabled) so impulses ride exactly on the synapse lines
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
        "color": { "color": "rgba(170, 190, 210, 0.14)", "highlight": "#F1C40F" },
        "smooth": { "enabled": false },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.35 } }
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
            if (typeof network !== 'undefined' && network && typeof nodes !== 'undefined' && typeof edges !== 'undefined') {
                clearInterval(checkExist);

                // ---------------- Zoom controls (unchanged) ----------------
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
                    network.moveTo({ scale: val / 100.0 });
                });

                resetBtn.addEventListener("click", function() {
                    fitDiagramToScreen();
                });

                // =========================================================
                // SYNAPTIC IMPULSE ENGINE
                // Input neurons fire -> electric impulses travel along
                // synapses -> receiving neuron flashes and fires onward
                // -> ... -> output neurons light up gold.
                // =========================================================
                var CFG = {
                    waveInterval: 2400,  // ms between new volleys from the input layer
                    inputStagger: 220,   // ms random stagger between input neurons in a volley
                    travelTime: 700,     // ms for one impulse to cross a synapse
                    synapseDelay: 90,    // ms delay before a fired neuron releases impulses
                    maxFanout: 5,        // max synapses a neuron fires along per spike
                    tailFrac: 0.30,      // length of glowing tail (fraction of synapse)
                    jitter: 3.0,         // lightning jaggedness (px)
                    refractory: 450,     // ms a neuron rests before it can fire again
                    flashDecay: 320,     // ms for the neuron flash to fade
                    shockwaveTime: 650,  // ms for the expanding ring
                    maxImpulses: 350     // safety cap for performance
                };

                var COL = {
                    input:  [140, 200, 255],
                    hidden: [ 90, 200, 255],
                    output: [255, 195,  60]
                };

                // --- Build topology ---
                var nodeList = nodes.get();
                var edgeList = edges.get();
                var nodeLayer = {};
                var layers = [];
                var maxLayer = 0;

                nodeList.forEach(function(n) {
                    var id = String(n.id);
                    var L = parseInt(id.substring(1, id.indexOf('_')), 10) || 0;
                    nodeLayer[id] = L;
                    if (!layers[L]) layers[L] = [];
                    layers[L].push(id);
                    if (L > maxLayer) maxLayer = L;
                });

                var outgoing = {};
                edgeList.forEach(function(e) {
                    var f = String(e.from), t = String(e.to);
                    if (!outgoing[f]) outgoing[f] = [];
                    outgoing[f].push(t);
                });

                function colorOf(id) {
                    var L = nodeLayer[id];
                    if (L === 0) return COL.input;
                    if (L === maxLayer) return COL.output;
                    return COL.hidden;
                }
                function rgba(c, a) {
                    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + Math.max(0, Math.min(1, a)).toFixed(3) + ')';
                }
                function shuffle(arr) {
                    for (var i = arr.length - 1; i > 0; i--) {
                        var j = Math.floor(Math.random() * (i + 1));
                        var tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
                    }
                    return arr;
                }

                // --- State ---
                var lastFire = {};     // node id -> time of last spike
                var scheduled = [];    // {id, t}: neurons queued to fire
                var impulses = [];     // travelling action potentials
                var lastWave = -1e9;
                var lastFrame = performance.now();

                function scheduleFire(id, t) {
                    scheduled.push({ id: id, t: t });
                }

                function fireNode(id, now) {
                    if (lastFire[id] !== undefined && now - lastFire[id] < CFG.refractory) return;
                    lastFire[id] = now;

                    var targets = outgoing[id];
                    if (!targets || !targets.length) return;   // output neuron: just flashes

                    var picks = shuffle(targets.slice()).slice(0, Math.min(CFG.maxFanout, targets.length));
                    picks.forEach(function(to) {
                        if (impulses.length >= CFG.maxImpulses) return;
                        impulses.push({
                            from: id,
                            to: to,
                            start: now + CFG.synapseDelay + Math.random() * 120,
                            dur: CFG.travelTime * (0.85 + Math.random() * 0.3)
                        });
                    });
                }

                function launchWave(now) {
                    var inputs = layers[0] || [];
                    inputs.forEach(function(id) {
                        // ~85% of inputs fire per volley for a natural, irregular look
                        if (Math.random() < 0.85 || inputs.length <= 2) {
                            scheduleFire(id, now + Math.random() * CFG.inputStagger);
                        }
                    });
                }

                // --- Per-frame update + render ---
                network.on("afterDrawing", function(ctx) {
                    var now = performance.now();

                    // If tab was hidden for a while, restart cleanly instead of bursting
                    if (now - lastFrame > 1000) {
                        impulses = [];
                        scheduled = [];
                        lastWave = -1e9;
                    }
                    lastFrame = now;

                    if (now - lastWave > CFG.waveInterval) {
                        lastWave = now;
                        launchWave(now);
                    }

                    // Process scheduled neuron firings
                    var stillScheduled = [];
                    scheduled.forEach(function(s) {
                        if (now >= s.t) fireNode(s.id, now);
                        else stillScheduled.push(s);
                    });
                    scheduled = stillScheduled;

                    // Positions + radii (supports user dragging nodes)
                    var pos = network.getPositions();
                    var rad = {};
                    nodeList.forEach(function(n) {
                        var box = network.getBoundingBox(n.id);
                        rad[n.id] = box ? (box.right - box.left) / 2 : 28;
                    });

                    ctx.save();
                    ctx.globalCompositeOperation = 'lighter';
                    ctx.lineCap = 'round';
                    ctx.lineJoin = 'round';

                    // ---------- 1. Impulses along synapses ----------
                    var alive = [];
                    var arrived = [];
                    impulses.forEach(function(imp) {
                        var p = (now - imp.start) / imp.dur;
                        if (p < 0) { alive.push(imp); return; }
                        if (p >= 1) {
                            // Impulse reached the next neuron -> queue it to fire.
                            // (Queued, not fired directly, so its new impulses are not
                            //  wiped out when the impulses array is rebuilt below.)
                            arrived.push(imp.to);
                            return;
                        }
                        alive.push(imp);

                        var a = pos[imp.from], b = pos[imp.to];
                        if (!a || !b) return;

                        var dx = b.x - a.x, dy = b.y - a.y;
                        var len = Math.sqrt(dx * dx + dy * dy) || 1;
                        var ux = dx / len, uy = dy / len;
                        var nx = -uy, ny = ux;
                        var rA = rad[imp.from] || 28, rB = rad[imp.to] || 28;
                        var sx = a.x + ux * rA, sy = a.y + uy * rA;
                        var span = Math.max(1, len - rA - rB);

                        // slight acceleration as the signal leaves the soma
                        var head = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
                        var tail = Math.max(0, head - CFG.tailFrac);
                        var c = colorOf(imp.to);

                        var hx = sx + ux * span * head, hy = sy + uy * span * head;
                        var tx = sx + ux * span * tail, ty = sy + uy * span * tail;

                        // charged synapse: faint glow from origin to the impulse head
                        ctx.beginPath();
                        ctx.moveTo(sx, sy);
                        ctx.lineTo(hx, hy);
                        ctx.strokeStyle = rgba(c, 0.10 * (1 - p));
                        ctx.lineWidth = 2;
                        ctx.stroke();

                        // jagged lightning trail (re-randomised each frame = electric flicker)
                        var segs = 9;
                        var pts = [];
                        for (var i = 0; i <= segs; i++) {
                            var t = tail + (head - tail) * (i / segs);
                            var px = sx + ux * span * t, py = sy + uy * span * t;
                            if (i > 0 && i < segs) {
                                var off = (Math.random() - 0.5) * 2 * CFG.jitter;
                                px += nx * off; py += ny * off;
                            }
                            pts.push([px, py]);
                        }

                        var grad = ctx.createLinearGradient(tx, ty, hx, hy);
                        grad.addColorStop(0, rgba(c, 0));
                        grad.addColorStop(1, rgba(c, 0.55));

                        // outer glow
                        ctx.beginPath();
                        ctx.moveTo(pts[0][0], pts[0][1]);
                        for (var k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0], pts[k][1]);
                        ctx.strokeStyle = grad;
                        ctx.lineWidth = 6;
                        ctx.stroke();

                        // bright core
                        var gradCore = ctx.createLinearGradient(tx, ty, hx, hy);
                        gradCore.addColorStop(0, 'rgba(255,255,255,0)');
                        gradCore.addColorStop(1, 'rgba(255,255,255,0.95)');
                        ctx.strokeStyle = gradCore;
                        ctx.lineWidth = 1.6;
                        ctx.stroke();

                        // glowing spark at the impulse head
                        var g = ctx.createRadialGradient(hx, hy, 0, hx, hy, 11);
                        g.addColorStop(0, 'rgba(255,255,255,1)');
                        g.addColorStop(0.35, rgba(c, 0.85));
                        g.addColorStop(1, rgba(c, 0));
                        ctx.beginPath();
                        ctx.arc(hx, hy, 11, 0, 2 * Math.PI);
                        ctx.fillStyle = g;
                        ctx.fill();
                    });
                    impulses = alive;

                    // Neurons hit by an impulse now fire onward to the next layer
                    // (Layer 1 -> Layer 2 -> Layer 3 -> Output, whichever exist)
                    arrived.forEach(function(id) { fireNode(id, now); });

                    // ---------- 2. Neuron firing flashes ----------
                    var tSec = now / 1000;
                    nodeList.forEach(function(n) {
                        var id = n.id;
                        var p = pos[id];
                        if (!p) return;
                        var r = rad[id] || 28;
                        var c = colorOf(id);
                        var since = (lastFire[id] !== undefined) ? now - lastFire[id] : 1e9;

                        if (since < CFG.shockwaveTime) {
                            var I = Math.exp(-since / CFG.flashDecay);

                            // soma glow
                            var g = ctx.createRadialGradient(p.x, p.y, r * 0.2, p.x, p.y, r * 1.9);
                            g.addColorStop(0, rgba(c, 0.55 * I));
                            g.addColorStop(0.55, rgba(c, 0.30 * I));
                            g.addColorStop(1, rgba(c, 0));
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, r * 1.9, 0, 2 * Math.PI);
                            ctx.fillStyle = g;
                            ctx.fill();

                            // bright membrane ring
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
                            ctx.strokeStyle = rgba([255, 255, 255], 0.9 * I);
                            ctx.lineWidth = 1.5 + 3.5 * I;
                            ctx.stroke();

                            // expanding shockwave
                            var w = since / CFG.shockwaveTime;
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, r + w * r * 1.3, 0, 2 * Math.PI);
                            ctx.strokeStyle = rgba(c, 0.6 * (1 - w));
                            ctx.lineWidth = 2;
                            ctx.stroke();
                        } else {
                            // resting potential: gentle breathing ring
                            var breath = 0.5 + 0.5 * Math.sin(tSec * 2 + nodeLayer[id] * 0.8);
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
                            ctx.strokeStyle = rgba(c, 0.15 + 0.15 * breath);
                            ctx.lineWidth = 1.5;
                            ctx.stroke();
                        }
                    });

                    ctx.restore();
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

render_pyvis_network(
    num_inputs, hidden1_size, hidden2_size, hidden3_size, num_outputs, global_activation
)

st.divider()


# =====================================================================
# 3. CONFIGURABLE MODEL CLASS & DATA PARTITIONING WITH ACTIVE RANDOMIZER
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
    indices = torch.randperm(total_samples)  # Randomly shuffles row indices
    return indices[:split_idx], indices[split_idx:]

test_ratio = st.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, step=0.05)

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


# --- TAB 0: CORRELATION MATRIX ---
if selected_tab == "Data Correlation Matrix":
    st.write("### Feature Correlation Matrix (Lower Triangle)")

    numeric_df = df_raw.select_dtypes(include=[np.number])
    if numeric_df.empty:
        numeric_df = df_raw.apply(pd.to_numeric, errors="coerce")

    stds = numeric_df.std()
    zero_var_cols = stds[stds == 0].index.tolist()

    corr = numeric_df.corr()

    for col in zero_var_cols:
        corr.loc[col, :] = np.nan
        corr.loc[:, col] = np.nan

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.copy()
    corr_masked[mask] = np.nan

    royal_blue_colorscale = [
        [0.0, "#F7FBFF"],
        [0.2, "#DEEBF7"],
        [0.4, "#C6DBEF"],
        [0.6, "#9ECAE1"],
        [0.8, "#3182BD"],
        [1.0, "#08519C"]
    ]

    fig = px.imshow(
        corr_masked,
        color_continuous_scale=royal_blue_colorscale,
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    annotations = []
    for i, row_name in enumerate(corr.index):
        for j, col_name in enumerate(corr.columns):
            val = corr_masked.iloc[i, j]

            if row_name in zero_var_cols or col_name in zero_var_cols:
                if j <= i:
                    text_label = "NaN"
                    font_color = "#555555"
                else:
                    continue
            elif np.isnan(val):
                continue
            else:
                text_label = f"{val:.2f}"
                # Follow regulation: values <= 0.7 down to -1.0 must have black font
                font_color = "white" if val > 0.7 else "black"

            annotations.append(
                dict(
                    x=col_name,
                    y=row_name,
                    text=text_label,
                    font=dict(color=font_color, size=11, family="Segoe UI, sans-serif"),
                    showarrow=False
                )
            )

    fig.update_layout(
        annotations=annotations,
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(tickangle=-45),
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    c_left, c_mid, c_right = st.columns([0.1, 0.8, 0.1])
    with c_mid:
        st.plotly_chart(fig, use_container_width=True)


# --- TAB 1: BATCH TRAINING ---
elif selected_tab == "Batch Training Phase":
    st.markdown("Train the model parameters using normalized training inputs (`X_train`, `Y_train`).")

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        lr = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.4f",
        )
    with tcol2:
        optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD"])

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
