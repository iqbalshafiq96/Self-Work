import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

st.set_page_config(page_title="Non-Linear Evolutionary Optimizer", layout="centered")

st.title("Non-Linear Evolutionary Optimization Dashboard")
st.markdown("""
This app demonstrates how an evolutionary algorithm (**Differential Evolution**) navigates 
bounded, non-linear spaces (involving polynomials and natural logarithms) to find optimal solutions.
""")

st.sidebar.header("Optimization Settings")

# Define bounds for variables x1 and x2
st.sidebar.subheader("Variable Bounds")
x1_min = st.sidebar.number_input("Min x1", value=0.1, format="%.2f")
x1_max = st.sidebar.number_input("Max x1", value=10.0, format="%.2f")
x2_min = st.sidebar.number_input("Min x2", value=0.1, format="%.2f")
x2_max = st.sidebar.number_input("Max x2", value=10.0, format="%.2f")

bounds = [(x1_min, x1_max), (x2_min, x2_max)]

# Select objective mode
goal = st.sidebar.selectbox("Optimization Goal", ["Minimize", "Maximize"])

# Define the non-linear objective function
# Example objective: f(x1, x2) = x1^3 - 5*x1 + x2^2 - 3*ln(x2)
def objective_function(x):
    x1, x2 = x[0], x[1]
    # Non-linear equation with cubic term and natural log
    val = (x1**3 - 5.0 * x1) + (x2**2 - 3.0 * np.log(x2))
    # If maximizing, we invert the value because solvers typically minimize
    return val if goal == "Minimize" else -val

if st.button("Run Evolutionary Optimization"):
    with st.spinner("Evolutionary algorithm searching the feasible space..."):
        # Run Differential Evolution (an evolutionary metaheuristic)
        result = differential_evolution(objective_function, bounds, seed=42)
        
        opt_x1, opt_x2 = result.x[0], result.x[1]
        raw_val = result.fun if goal == "Minimize" else -result.fun

    st.success("Optimization Complete!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal x1", f"{opt_x1:.4f}")
    col2.metric("Optimal x2", f"{opt_x2:.4f}")
    col3.metric(f"Best Objective Value", f"{raw_val:.4f}")

    # Plotting the search space and optimal point
    st.subheader("Objective Contour & Optimal Point")
    
    grid_size = 100
    x1_vals = np.linspace(x1_min, x1_max, grid_size)
    x2_vals = np.linspace(x2_min, x2_max, grid_size)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    # Vectorized evaluation for contour plot
    Z = (X1**3 - 5.0 * X1) + (X2**2 - 3.0 * np.log(X2))
    if goal == "Maximize":
        Z = -Z

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X1, X2, Z, levels=50, cmap="viridis")
    fig.colorbar(contour, label="Objective Value")
    
    # Plot optimal point found by evolutionary algorithm
    ax.scatter([opt_x1], [opt_x2], color="red", s=150, marker="*", label="Optimal Point")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Non-Linear Feasible Space & Evolutionary Result")
    ax.legend()
    
    st.pyplot(fig)
