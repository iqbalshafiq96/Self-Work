import streamlit as st
import graphviz

st.title("Letdown Steam System Schematic")

# Create a Graphviz directed graph
dot = graphviz.Digraph(comment="Letdown Steam System", format="png")
dot.attr(rankdir="LR", size="8,5")

# Define node styles
dot.attr("node", shape="box", style="rounded,filled", fillcolor="#f4f4f4")

# Add components (Nodes)
dot.node("IN", "High Pressure Steam\n(Inlet)", shape="ellipse", fillcolor="#e1f5fe")
dot.node("CV", "Control Valve\n(Pressure Letdown)", shape="polygon", sides="4", distortion="0.3", fillcolor="#ffe0b2")
dot.node("FW", "Feedwater Spray\n(Inlet)", shape="ellipse", fillcolor="#e0f2f1")
dot.node("VD", "Venturi Desuperheater", shape="component", fillcolor="#fff9c4")
dot.node("OUT", "Desuperheated Steam\n(Outlet)", shape="ellipse", fillcolor="#e1f5fe")

# Connect flows (Edges)
dot.edge("IN", "CV", label=" High Temp / Press")
dot.edge("CV", "VD", label=" Low Press Steam")
dot.edge("FW", "VD", label=" Cooling Water")
dot.edge("VD", "OUT", label=" Temperature Controlled Steam")

# Render in Streamlit
st.graphviz_chart(dot)
