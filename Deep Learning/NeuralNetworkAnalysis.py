import hashlib
import io
import json
import zipfile
from datetime import datetime

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

st.set_page_config(page_title="Neural Network Configurator", layout="wide")
st.title("Develop, Train & Deploy Neural Network")
st.caption("Developed by Iqbal SHERPA 20260824. Contact me for further information @iqbalshafiq96@gmail.com")
st.caption(
    "A neural network is a machine-learning algorithm that learns to predict process outputs from input variables. "
    "Its architecture consists of interconnected nodes (neurons), each holding a weight and bias, arranged in hidden layers "
    "between the input and output. During training, data flows forward through the network to generate a prediction "
    "(forward propagation), the prediction error is measured, and that error is sent backward to fine-tune the weights "
    "and biases (backpropagation). Repeating this cycle over many epochs steadily improves prediction accuracy."
)

# =====================================================================
# 0. DATASET SELECTION: GITHUB REPOSITORY OR USER UPLOAD
# =====================================================================
GITHUB_API_URL = "https://api.github.com/repos/iqbalshafiq96/Self-Work/contents/Deep%20Learning"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/iqbalshafiq96/Self-Work/main/Deep%20Learning"


@st.cache_data(ttl=600)
def fetch_csv_file_list():
    try:
        response = requests.get(GITHUB_API_URL, timeout=15)
        if response.status_code == 200:
            files = response.json()
            csv_files = [f["name"] for f in files if f["name"].lower().endswith(".csv")]
            if csv_files:
                return csv_files
    except Exception:
        pass
    return ["SMR_Data.csv"]


@st.cache_data(ttl=600)
def fetch_github_csv_bytes(filename):
    url = f"{GITHUB_RAW_BASE}/{filename.replace(' ', '%20')}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.content


@st.cache_data
def load_and_preprocess_custom_csv(csv_bytes):
    # Row 1 = tag names, Row 2 = Independent/Dependent, Column 0 = timestamp
    preview_df = pd.read_csv(io.BytesIO(csv_bytes), nrows=2, header=None)
    col_names = [str(c).strip() for c in preview_df.iloc[0].values[1:]]
    col_types = preview_df.iloc[1].values[1:]

    full_df = pd.read_csv(io.BytesIO(csv_bytes), header=None, skiprows=2)
    data_df = full_df.iloc[:, 1:].copy()
    data_df.columns = col_names
    data_df = data_df.apply(pd.to_numeric, errors="coerce").dropna()

    if data_df.empty:
        raise ValueError("No valid numeric rows found below the two header rows.")

    input_names = [col_names[i] for i, t in enumerate(col_types) if str(t).strip().lower() == "independent"]
    output_names = [col_names[i] for i, t in enumerate(col_types) if str(t).strip().lower() == "dependent"]

    if not input_names or not output_names:
        input_names = list(data_df.columns[:3])
        output_names = list(data_df.columns[3:7])

    if not input_names or not output_names:
        raise ValueError("Could not identify Independent (input) and Dependent (output) columns.")

    X_raw = np.nan_to_num(data_df[input_names].values, nan=0.0, posinf=0.0, neginf=0.0)
    Y_raw = np.nan_to_num(data_df[output_names].values, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_raw)

    return data_df, input_names, output_names, X_scaled, Y_scaled, scaler_X, scaler_Y


st.write("### Dataset Selection")
tab_github, tab_upload = st.tabs(["📂 GitHub Repository", "⬆ Upload My Own CSV"])

with tab_github:
    available_csvs = fetch_csv_file_list()
    selected_csv_filename = st.selectbox(
        "Select CSV File from GitHub", available_csvs, label_visibility="collapsed"
    )

with tab_upload:
    st.caption(
        "Required format: **Row 1** = tag names, **Row 2** = `Independent` (input) or `Dependent` (output) "
        "for each column, **Column 1** = timestamp. Data starts from Row 3. "
        "An uploaded file takes priority over the GitHub selection; remove it (✕) to switch back."
    )
    template_csv = (
        "Timestamp,Input_A,Input_B,Output_Y\n"
        ",Independent,Independent,Dependent\n"
        "2026-01-01 00:00,1.0,2.0,3.0\n"
    )
    st.download_button(
        "⬇ Download CSV Template", template_csv, file_name="NN_Template.csv", mime="text/csv"
    )
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Uploaded file (if any) takes priority; otherwise use the GitHub selection
if uploaded_file is not None:
    source_mode = "Upload"
    csv_bytes = uploaded_file.getvalue()
    dataset_label = uploaded_file.name
else:
    source_mode = "GitHub"
    try:
        csv_bytes = fetch_github_csv_bytes(selected_csv_filename)
    except Exception:
        try:
            with open("SMR_Data.csv", "rb") as f:
                csv_bytes = f.read()
            st.warning("GitHub unreachable. Loaded local fallback 'SMR_Data.csv'.")
        except FileNotFoundError:
            st.error("Could not fetch the file from GitHub and no local fallback was found.")
            st.stop()
    dataset_label = selected_csv_filename

# Reset the model and split whenever the dataset content changes
dataset_key = f"{source_mode}::{dataset_label}::{hashlib.md5(csv_bytes).hexdigest()}"
if st.session_state.get("dataset_key") != dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.net = None
    st.session_state.loss_history = []
    st.session_state.pop("train_idx", None)
    st.session_state.pop("test_idx", None)
    st.session_state.pop("eval_results", None)   # NEW: clear stale test results
    st.session_state.pop("manual_pred", None)    # NEW: clear stale manual prediction

try:
    (
        df_raw,
        input_names,
        output_names,
        X_norm,
        Y_norm,
        scaler_X,
        scaler_Y,
    ) = load_and_preprocess_custom_csv(csv_bytes)
    st.success(
        f"Loaded '{dataset_label}' ({source_mode}) successfully! "
        f"{len(df_raw)} rows | {len(input_names)} inputs | {len(output_names)} outputs"
    )
except Exception as e:
    st.error(f"Failed to load dataset: {e}. Please check the CSV format.")
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
                x=float(x_pos),
                y=float(y_pos),
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

                // ---------------- Zoom controls ----------------
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

num_samples = len(X_norm)


def repartition_dataset(total_samples, current_test_ratio):
    split_idx = int(total_samples * (1 - current_test_ratio))
    indices = torch.randperm(total_samples)  # Randomly shuffles row indices
    return indices[:split_idx], indices[split_idx:]


test_ratio = st.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, step=0.05)

if (
    "train_idx" not in st.session_state
    or "test_idx" not in st.session_state
    or len(st.session_state.train_idx) + len(st.session_state.test_idx) != num_samples
):
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

TAB_CORR = "📊 Data Correlation Matrix"
TAB_TRAIN = "🏋 Batch Training Phase"
TAB_TEST = "✅ Model Testing & Verification"
TAB_EXPORT = "💾 Export Model"

init_btn_col, init_msg_col = st.columns([0.3, 0.7])
with init_btn_col:
    init_clicked = st.button("Initialize / Reset Model Architecture")
# Fixed slot: the message appears/disappears here without shifting the tabs below,
# so Streamlit does not rebuild the tabs and jump back to the first one.
init_msg_slot = init_msg_col.empty()

if init_clicked:
    st.session_state.train_idx, st.session_state.test_idx = repartition_dataset(
        num_samples, test_ratio
    )
    st.session_state.net = ConfigurableNet(
        num_inputs, hidden1_size, hidden2_size, hidden3_size, global_activation, num_outputs
    )
    st.session_state.loss_history = []
    st.session_state.net_config = {
        "num_inputs": num_inputs,
        "hidden_layers": [h for h in [hidden1_size, hidden2_size, hidden3_size] if h > 0],
        "num_outputs": num_outputs,
        "activation": global_activation,
    }
    st.session_state.pop("eval_results", None)   # NEW: old test results no longer valid
    st.session_state.pop("manual_pred", None)    # NEW: old manual prediction no longer valid
    st.session_state.show_init_msg = True  # survive the rerun below
    # Reopen workflow on the Data Correlation Matrix tab
    st.session_state.workflow_tab = TAB_CORR
    st.session_state.tabs_version = st.session_state.get("tabs_version", 0) + 1
    st.rerun()

if st.session_state.pop("show_init_msg", False):
    init_msg_slot.success("New PyTorch Model initialized with freshly randomized Train/Test sets!")

st.caption(
    "Initialize the model before training. This builds the network from your selected architecture and randomly "
    "splits the dataset into training and test sets based on the Test Set Split Ratio, so the model learns from one "
    "portion and is verified on unseen data. Re-initialize whenever you change the hidden layers, transfer function, "
    "or split ratio."
)


# =====================================================================
# 5. MODEL EXPORT HELPERS
# =====================================================================
ACT_KEY = {"Tanh (tansig)": "tanh", "Sigmoid (logsig)": "sigmoid", "ReLU": "relu"}


def build_model_package(net, cfg):
    """Collects architecture, weights, biases and scaler parameters in plain Python types."""
    linear_layers = [m for m in net.network if isinstance(m, nn.Linear)]
    layers = []
    for i, lin in enumerate(linear_layers):
        is_output = i == len(linear_layers) - 1
        layers.append({
            "name": "Output" if is_output else f"Hidden_{i+1}",
            "in_features": lin.in_features,
            "out_features": lin.out_features,
            "activation": "linear" if is_output else ACT_KEY[cfg["activation"]],
            "weights": lin.weight.detach().cpu().numpy().tolist(),
            "bias": lin.bias.detach().cpu().numpy().tolist(),
        })
    return {
        "model_name": "Neural Network Configurator Model",
        "exported_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_label,
        "architecture": {
            "num_inputs": cfg["num_inputs"],
            "hidden_layers": cfg["hidden_layers"],
            "num_outputs": cfg["num_outputs"],
            "activation": cfg["activation"],
        },
        "input_names": list(input_names),
        "output_names": list(output_names),
        "scaler_X": {"mean": scaler_X.mean_.tolist(), "scale": scaler_X.scale_.tolist()},
        "scaler_Y": {"mean": scaler_Y.mean_.tolist(), "scale": scaler_Y.scale_.tolist()},
        "final_training_loss": st.session_state.loss_history[-1] if st.session_state.loss_history else None,
        "epochs_trained": len(st.session_state.loss_history),
        "layers": layers,
    }


def export_pytorch_checkpoint(net, pkg):
    buf = io.BytesIO()
    torch.save(
        {
            "state_dict": net.state_dict(),
            "architecture": pkg["architecture"],
            "input_names": pkg["input_names"],
            "output_names": pkg["output_names"],
            "scaler_X": pkg["scaler_X"],
            "scaler_Y": pkg["scaler_Y"],
        },
        buf,
    )
    return buf.getvalue()


def export_torchscript(net, n_in):
    net.eval()
    traced = torch.jit.trace(net, torch.zeros(1, n_in, dtype=torch.float32))
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()


def export_onnx(net, n_in, in_names, out_names):
    net.eval()
    buf = io.BytesIO()
    dummy = torch.zeros(1, n_in, dtype=torch.float32)
    kwargs = dict(
        input_names=["inputs_scaled"],
        output_names=["outputs_scaled"],
        dynamic_axes={"inputs_scaled": {0: "batch"}, "outputs_scaled": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(net, dummy, buf, dynamo=False, **kwargs)  # newer PyTorch
    except TypeError:
        torch.onnx.export(net, dummy, buf, **kwargs)  # older PyTorch
    return buf.getvalue()


def export_excel(pkg):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        arch = pkg["architecture"]
        summary = pd.DataFrame(
            {
                "Item": ["Dataset", "Exported On", "Inputs", "Hidden Layers", "Outputs",
                         "Activation", "Epochs Trained", "Final Training Loss (MSE)"],
                "Value": [pkg["dataset"], pkg["exported_on"], arch["num_inputs"],
                          str(arch["hidden_layers"]), arch["num_outputs"], arch["activation"],
                          pkg["epochs_trained"], pkg["final_training_loss"]],
            }
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)

        pd.DataFrame(
            {"Input": pkg["input_names"], "Mean": pkg["scaler_X"]["mean"], "Std": pkg["scaler_X"]["scale"]}
        ).to_excel(writer, sheet_name="Input_Scaler", index=False)
        pd.DataFrame(
            {"Output": pkg["output_names"], "Mean": pkg["scaler_Y"]["mean"], "Std": pkg["scaler_Y"]["scale"]}
        ).to_excel(writer, sheet_name="Output_Scaler", index=False)

        for li, layer in enumerate(pkg["layers"]):
            if li == 0:
                col_labels = pkg["input_names"]
            else:
                col_labels = [f"{pkg['layers'][li-1]['name']}_N{j+1}" for j in range(layer["in_features"])]
            if layer["name"] == "Output":
                row_labels = pkg["output_names"]
            else:
                row_labels = [f"{layer['name']}_N{j+1}" for j in range(layer["out_features"])]
            df_w = pd.DataFrame(layer["weights"], index=row_labels, columns=col_labels)
            df_w["Bias"] = layer["bias"]
            df_w.to_excel(writer, sheet_name=f"{layer['name']}_W"[:31])
    return buf.getvalue()


def export_numpy_script(pkg):
    header = (
        '"""\n'
        "Standalone inference script exported from the Neural Network Configurator.\n"
        "Needs only NumPy. No PyTorch required.\n\n"
        "Usage:\n"
        "    from nn_inference import predict\n"
        "    y = predict([[x1, x2, ...]])   # raw engineering units in, engineering units out\n"
        '"""\n'
        "import json\n"
        "import numpy as np\n\n"
    )
    model_line = "MODEL = json.loads(" + repr(json.dumps(pkg)) + ")\n"
    body = """
_ACT = {
    "tanh": np.tanh,
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "relu": lambda z: np.maximum(z, 0.0),
    "linear": lambda z: z,
}

INPUT_NAMES = MODEL["input_names"]
OUTPUT_NAMES = MODEL["output_names"]


def predict(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x_mean = np.array(MODEL["scaler_X"]["mean"]); x_std = np.array(MODEL["scaler_X"]["scale"])
    y_mean = np.array(MODEL["scaler_Y"]["mean"]); y_std = np.array(MODEL["scaler_Y"]["scale"])

    a = (X - x_mean) / x_std
    for layer in MODEL["layers"]:
        W = np.array(layer["weights"]); b = np.array(layer["bias"])
        a = _ACT[layer["activation"]](a @ W.T + b)
    return a * y_std + y_mean


if __name__ == "__main__":
    sample = np.array([MODEL["scaler_X"]["mean"]])
    print("Inputs :", dict(zip(INPUT_NAMES, sample[0].round(4))))
    print("Outputs:", dict(zip(OUTPUT_NAMES, predict(sample)[0].round(4))))
"""
    return header + model_line + body


def export_zip_bundle(files: dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


EXPORT_README = """NEURAL NETWORK MODEL EXPORT
===========================
All models expect STANDARDIZED inputs and return STANDARDIZED outputs, except the
NumPy script, which handles scaling for you.
    x_scaled = (x - mean_X) / std_X
    y        = y_scaled * std_Y + mean_Y
Scaler values are in model.json and in the Excel workbook.

model_checkpoint.pth : PyTorch checkpoint (state_dict + architecture + scalers).
                       Rebuild ConfigurableNet with the saved architecture, then load_state_dict.
model_torchscript.pt : torch.jit.load("model_torchscript.pt"). No class definition needed.
model.onnx           : ONNX Runtime, MATLAB, C#, C++, and other ONNX-compatible tools.
model.json           : Portable weights, biases and scalers for any language.
model_weights.xlsx   : Readable weight/bias matrices per layer plus scalers.
nn_inference.py      : NumPy-only predict() function. Raw units in and out.
"""

# =====================================================================
# 4. WORKFLOW TABS
# =====================================================================
st.divider()

if "workflow_tab" not in st.session_state:
    st.session_state.workflow_tab = TAB_CORR
if "tabs_version" not in st.session_state:
    st.session_state.tabs_version = 0


def remember_tab(tab_name):
    # Called by buttons inside a tab, before the rerun, so the same tab reopens
    st.session_state.workflow_tab = tab_name


# The container key only changes when Initialize is clicked. That forces the tabs
# to rebuild and open on the default tab (Data Correlation Matrix).
with st.container(key=f"workflow_tabs_{st.session_state.tabs_version}"):
    tab_corr, tab_train, tab_test, tab_export = st.tabs(
        [TAB_CORR, TAB_TRAIN, TAB_TEST, TAB_EXPORT],
        default=st.session_state.workflow_tab,
    )


# --- TAB 0: CORRELATION MATRIX ---
with tab_corr:
    st.write("### Feature Correlation Matrix (Lower Triangle)")
    st.caption(
        "The correlation matrix shows how strongly each feature moves in relation to another, on a scale from -1 to +1. "
        "Values near +1 indicate a strong positive correlation (both rise together), values near -1 indicate a strong "
        "negative correlation (one rises as the other falls), and values near 0 indicate little or no linear relationship. "
        "Highly correlated inputs may carry overlapping information, while inputs strongly correlated with the outputs "
        "are good predictors."
    )

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
        [1.0, "#08519C"],
    ]

    fig = px.imshow(
        corr_masked,
        color_continuous_scale=royal_blue_colorscale,
        zmin=-1,
        zmax=1,
        aspect="auto",
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
                # Values <= 0.7 down to -1.0 use black font
                font_color = "white" if val > 0.7 else "black"

            annotations.append(
                dict(
                    x=col_name,
                    y=row_name,
                    text=text_label,
                    font=dict(color=font_color, size=11, family="Segoe UI, sans-serif"),
                    showarrow=False,
                )
            )

    fig.update_layout(
        annotations=annotations,
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(tickangle=-45),
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    c_left, c_mid, c_right = st.columns([0.1, 0.8, 0.1])
    with c_mid:
        st.plotly_chart(fig, use_container_width=True)


# --- TAB 1: BATCH TRAINING ---
with tab_train:
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

    if st.button("Run Batch Training", on_click=remember_tab, args=(TAB_TRAIN,)):
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

            # NEW: weights changed, so earlier test results / manual prediction are stale
            st.session_state.pop("eval_results", None)
            st.session_state.pop("manual_pred", None)

            st.success(f"Training Complete! Final Loss: {loss.item():.6f}")


# --- TAB 2: MODEL TESTING & VERIFICATION ---
with tab_test:
    st.markdown("Evaluate actual vs. predicted performance across output variables (Inverted back to engineering units).")

    if st.button("Evaluate Model on Test Set", on_click=remember_tab, args=(TAB_TEST,)):
        if st.session_state.net is None:
            st.warning("Please initialize and train the model first!")
        else:
            net = st.session_state.net
            net.eval()

            with torch.no_grad():
                test_preds_norm = net(X_test).numpy()
                Y_test_norm = Y_test.numpy()

            # Stored in session_state so the charts stay visible when the
            # manual prediction form below triggers a rerun
            st.session_state.eval_results = {
                "actual": scaler_Y.inverse_transform(Y_test_norm),
                "pred": scaler_Y.inverse_transform(test_preds_norm),
            }

    eval_res = st.session_state.get("eval_results")
    if eval_res is not None:
        Y_test_actual = eval_res["actual"]
        Y_test_pred = eval_res["pred"]

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

    # -----------------------------------------------------------------
    # NEW: MANUAL INPUT PREDICTION
    # -----------------------------------------------------------------
    st.divider()
    st.write("### Manual Input Prediction")
    st.caption(
        "Enter your own input values in engineering units and the trained model will predict the outputs. "
        "Default values are the training-data averages. Hover over the ⓘ icon to see the range the model was "
        "trained on. Predictions outside that range are extrapolations and less reliable."
    )

    if st.session_state.net is None:
        st.info("Please initialize and train the model first to use manual prediction.")
    else:
        # Training-set range for each input (what the model has actually seen)
        train_rows = df_raw.iloc[st.session_state.train_idx.numpy()]
        in_min = train_rows[input_names].min()
        in_max = train_rows[input_names].max()
        in_mean = train_rows[input_names].mean()

        # A form means editing the boxes does not rerun the app until "Predict" is pressed
        with st.form("manual_input_form"):
            n_cols = min(3, num_inputs)
            in_cols = st.columns(n_cols)
            manual_vals = []
            for i, name in enumerate(input_names):
                with in_cols[i % n_cols]:
                    v = st.number_input(
                        name,
                        value=float(in_mean[name]),
                        format="%.4f",
                        help=f"Training range: {in_min[name]:.4g} to {in_max[name]:.4g}",
                        key=f"manual_in_{st.session_state.dataset_key}_{i}",
                    )
                    manual_vals.append(v)

            predict_clicked = st.form_submit_button(
                "🔮 Predict Output",
                type="primary",
                on_click=remember_tab,
                args=(TAB_TEST,),
            )

        if predict_clicked:
            net = st.session_state.net
            net.eval()
            x_raw = np.array([manual_vals], dtype=float)
            x_scaled = scaler_X.transform(x_raw)                      # same scaling as training
            with torch.no_grad():
                y_scaled = net(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
            y_pred = scaler_Y.inverse_transform(y_scaled)[0]          # back to engineering units

            out_of_range = [
                name for name, v in zip(input_names, manual_vals)
                if v < in_min[name] or v > in_max[name]
            ]
            st.session_state.manual_pred = {
                "inputs": manual_vals,
                "outputs": y_pred.tolist(),
                "out_of_range": out_of_range,
            }

        mp = st.session_state.get("manual_pred")
        if mp is not None:
            st.write("#### Predicted Outputs")
            if not st.session_state.loss_history:
                st.info("The model is not trained yet, so these predictions come from random starting weights.")
            if mp["out_of_range"]:
                st.warning(
                    "Outside the training range (extrapolation): " + ", ".join(mp["out_of_range"])
                )

            n_out_cols = min(4, num_outputs)
            out_cols = st.columns(n_out_cols)
            for i, (name, val) in enumerate(zip(output_names, mp["outputs"])):
                out_cols[i % n_out_cols].metric(name, f"{val:.4f}")

            with st.expander("View input/output table"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Variable": list(input_names) + list(output_names),
                            "Type": ["Input"] * num_inputs + ["Predicted Output"] * num_outputs,
                            "Value": list(mp["inputs"]) + list(mp["outputs"]),
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )


# --- TAB 3: EXPORT MODEL ---
with tab_export:
    st.write("### Export Trained Neural Network")
    st.caption(
        "Download the trained model in the format that suits where it will be used. "
        "The PyTorch, TorchScript and ONNX models work on standardized values. Their scaler parameters "
        "(mean and standard deviation) are included in the JSON, Excel and checkpoint files. "
        "The NumPy script handles scaling for you, so it takes and returns engineering units."
    )

    if st.session_state.net is None or "net_config" not in st.session_state:
        st.warning("Please initialize and train the model first!")
    else:
        if not st.session_state.loss_history:
            st.info("The model is initialized but not trained yet. Exports will contain random starting weights.")

        net_exp = st.session_state.net
        cfg = st.session_state.net_config
        pkg = build_model_package(net_exp, cfg)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        base = f"NN_{dataset_label.rsplit('.', 1)[0].replace(' ', '_')}_{stamp}"

        arch_str = " → ".join(
            [str(cfg["num_inputs"])] + [str(h) for h in cfg["hidden_layers"]] + [str(cfg["num_outputs"])]
        )
        ecol1, ecol2, ecol3 = st.columns(3)
        ecol1.metric("Architecture", arch_str)
        ecol2.metric("Activation", cfg["activation"])
        ecol3.metric(
            "Final Training Loss",
            f"{pkg['final_training_loss']:.6f}" if pkg["final_training_loss"] is not None else "N/A",
        )

        exports = {}
        errors = {}
        builders = {
            "pth": lambda: export_pytorch_checkpoint(net_exp, pkg),
            "torchscript": lambda: export_torchscript(net_exp, cfg["num_inputs"]),
            "onnx": lambda: export_onnx(net_exp, cfg["num_inputs"], input_names, output_names),
            "json": lambda: json.dumps(pkg, indent=2).encode("utf-8"),
            "xlsx": lambda: export_excel(pkg),
            "py": lambda: export_numpy_script(pkg).encode("utf-8"),
        }
        for k, fn in builders.items():
            try:
                exports[k] = fn()
            except Exception as ex:
                errors[k] = str(ex)

        formats = [
            ("pth", "🔥 PyTorch Checkpoint", f"{base}.pth", "application/octet-stream",
             "Weights + architecture + scalers. Best for continuing training in Python."),
            ("torchscript", "⚙ TorchScript", f"{base}_torchscript.pt", "application/octet-stream",
             "Self-contained model. Load with torch.jit.load(). No class code needed."),
            ("onnx", "🌐 ONNX", f"{base}.onnx", "application/octet-stream",
             "Open standard for ONNX Runtime, MATLAB, C#, C++ and other platforms."),
            ("json", "📄 JSON", f"{base}.json", "application/json",
             "Plain-text weights, biases and scalers. Readable from any language."),
            ("xlsx", "📊 Excel Weights", f"{base}_weights.xlsx",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             "One sheet per layer plus scalers and summary. Good for review or reports."),
            ("py", "🐍 NumPy Inference Script", f"{base}_inference.py", "text/x-python",
             "Standalone predict() function. Needs only NumPy. Engineering units in and out."),
        ]

        grid = st.columns(3)
        for i, (k, label, fname, mime, desc) in enumerate(formats):
            with grid[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.caption(desc)
                    if k in exports:
                        st.download_button(
                            f"⬇ Download {fname.rsplit('.', 1)[-1].upper()}",
                            data=exports[k],
                            file_name=fname,
                            mime=mime,
                            on_click="ignore",  # no rerun, so the tab stays open
                            key=f"dl_{k}",
                            use_container_width=True,
                        )
                    else:
                        st.error(f"Not available: {errors.get(k, 'unknown error')}")
                        if k == "onnx":
                            st.caption("Add `onnx` to requirements.txt to enable ONNX export.")

        st.divider()
        bundle_files = {"README.txt": EXPORT_README.encode("utf-8")}
        name_map = {
            "pth": "model_checkpoint.pth",
            "torchscript": "model_torchscript.pt",
            "onnx": "model.onnx",
            "json": "model.json",
            "xlsx": "model_weights.xlsx",
            "py": "nn_inference.py",
        }
        for k, data in exports.items():
            bundle_files[name_map[k]] = data
        st.download_button(
            "📦 Download All Formats (ZIP)",
            data=export_zip_bundle(bundle_files),
            file_name=f"{base}_bundle.zip",
            mime="application/zip",
            on_click="ignore",
            type="primary",
            key="dl_zip",
        )
