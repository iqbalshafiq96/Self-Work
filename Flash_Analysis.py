import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from thermo import Mixture

# Page Config
st.set_page_config(page_title="Flash Separator Calculator", layout="wide")
st.title("🧪 Quick Flash Separator Analysis")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("1. Feed Conditions & Basis")

# Stream Basis Selector
basis = st.sidebar.selectbox(
    "Input Composition Basis",
    ["Mole Fraction / Percent", "Mass Fraction / Percent", "Molar Flow (kmol/hr)", "Mass Flow (kg/hr)"]
)

total_flow = st.sidebar.number_input("Total Feed Flow Rate", value=100.0, min_value=0.1)

st.sidebar.header("2. Chemical Species Composition")
st.sidebar.caption("Enter components (must match standard names, e.g., methane, ethane, water, ethanol)")

# Component inputs with default hydrocarbon mixture
c1 = st.sidebar.text_input("Component 1", "methane")
v1 = st.sidebar.number_input("Comp 1 Value", value=20.0, min_value=0.0)

c2 = st.sidebar.text_input("Component 2", "propane")
v2 = st.sidebar.number_input("Comp 2 Value", value=50.0, min_value=0.0)

c3 = st.sidebar.text_input("Component 3", "n-butane")
v3 = st.sidebar.number_input("Comp 3 Value", value=30.0, min_value=0.0)

components = [c1, c2, c3]
raw_values = [v1, v2, v3]

st.sidebar.header("3. Separator Operating Conditions")
temp_c = st.sidebar.slider("Temperature (°C)", min_value=-50.0, max_value=200.0, value=25.0)
pressure_bar = st.sidebar.slider("Pressure (bar)", min_value=0.5, max_value=50.0, value=10.0)

# Convert T and P to SI units required by Thermo (K and Pa)
temp_k = temp_c + 273.15
pressure_pa = pressure_bar * 1e5

# --- CALCULATION ENGINE ---
try:
    # Normalize compositions to fractions
    total_val = sum(raw_values)
    fractions = [v / total_val for v in raw_values]

    # Initialize mixture based on selected basis
    if "Mole" in basis or "kmol" in basis:
        mix = Mixture(components, zs=fractions, T=temp_k, P=pressure_pa)
    else:  # Mass / Weight basis
        mix = Mixture(components, ws=fractions, T=temp_k, P=pressure_pa)

    # Calculate Vapor and Liquid Flows
    beta = mix.V_over_F  # Vapor fraction (0 = all liquid, 1 = all vapor)
    vapor_molar_flow = total_flow * beta
    liquid_molar_flow = total_flow * (1 - beta)

    # Get compositions (x = liquid mole frac, y = vapor mole frac)
    x_comp = mix.x if mix.x else [0]*len(components)
    y_comp = mix.y if mix.y else [0]*len(components)

    # --- MAIN DISPLAY LAYOUT ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Phase Equilibrium Summary")
        
        # Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Vapor Fraction (β)", f"{beta * 100:.1f} %")
        m2.metric("Vapor Flow", f"{vapor_molar_flow:.2f}")
        m3.metric("Liquid Flow", f"{liquid_molar_flow:.2f}")

        # Stream Composition Table
        df = pd.DataFrame({
            "Component": components,
            "Feed (z)": fractions,
            "Vapor Outlet (y)": y_comp,
            "Liquid Bottoms (x)": x_comp
        })
        st.dataframe(df.style.format({"Feed (z)": "{:.4f}", "Vapor Outlet (y)": "{:.4f}", "Liquid Bottoms (x)": "{:.4f}"}), use_container_width=True)

    with col2:
        st.subheader("🖼️ Separator Process Diagram")
        
        # Matplotlib P&ID Vessel Drawing
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.axis("off")

        # Draw Vessel Body
        vessel = plt.Rectangle((0.35, 0.25), 0.3, 0.5, facecolor="#e2e8f0", edgecolor="#334155", lw=3, boxstyle="round,pad=0.05")
        ax.add_patch(vessel)
        
        # Feed Arrow
        ax.annotate(f"INLET FEED\nFlow: {total_flow} ({basis})\nT: {temp_c} °C | P: {pressure_bar} bar", 
                    xy=(0.35, 0.5), xytext=(0.02, 0.5),
                    arrowprops=dict(facecolor="#2563eb", shrink=0.05, width=3, headwidth=8))

        # Vapor Outlet Arrow (Top)
        ax.annotate(f"VAPOR OUTLET\nFrac: {beta*100:.1f}%\nFlow: {vapor_molar_flow:.2f}", 
                    xy=(0.5, 0.75), xytext=(0.5, 0.92),
                    arrowprops=dict(facecolor="#dc2626", shrink=0.05, width=3, headwidth=8), ha="center")

        # Liquid Outlet Arrow (Bottom)
        ax.annotate(f"LIQUID BOTTOMS\nFrac: {(1-beta)*100:.1f}%\nFlow: {liquid_molar_flow:.2f}", 
                    xy=(0.5, 0.25), xytext=(0.5, 0.08),
                    arrowprops=dict(facecolor="#16a34a", shrink=0.05, width=3, headwidth=8), ha="center")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error during flash calculation: {str(e)}")
    st.info("Check if all component names are spelled correctly (e.g., 'water', 'methane', 'propane').")
