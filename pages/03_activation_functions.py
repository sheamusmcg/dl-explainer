import streamlit as st
import pandas as pd
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.viz_utils import plot_activation_functions

st.title("Activation Functions")

with st.expander("Why do neurons need activation functions?", expanded=True):
    st.markdown(
        "On the previous page you saw a neuron compute a **weighted sum** (inputs × weights + bias). "
        "That sum is just a straight line — and stacking straight lines on top of each other "
        "still gives you a straight line.\n\n"
        "- The **activation function** is applied *after* the weighted sum. "
        "It bends, clips, or squashes the output to introduce **non-linearity**.\n"
        "- This is what gives neural networks the ability to learn curves, boundaries, and complex patterns.\n"
        "- Different activations have different shapes — each with tradeoffs for speed, "
        "gradient flow, and output range.\n\n"
        "Compare them below to build intuition for when you'd pick one over another."
    )

# Controls
selected = st.multiselect(
    "Select functions to compare",
    ["relu", "sigmoid", "tanh", "leaky_relu", "linear"],
    default=["relu", "sigmoid", "tanh"],
    help=tooltips.ACTIVATION["functions"],
)

col1, col2 = st.columns([2, 1])
with col1:
    x_min, x_max = st.slider(
        "Input range (z)",
        -10.0, 10.0, (-5.0, 5.0),
        help=tooltips.ACTIVATION["x_range"],
    )
with col2:
    show_deriv = st.toggle("Show derivatives", value=False, help=tooltips.ACTIVATION["show_derivative"])

if selected:
    fig = plot_activation_functions(selected, x_range=(x_min, x_max), show_derivative=show_deriv)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Select at least one activation function to display.")

# Reading the chart
with st.expander("How to read this chart", expanded=True):
    st.markdown(
        "- The **x-axis (z)** is the weighted sum coming into a neuron — before the activation is applied.\n"
        "- The **y-axis** is what comes out after the activation transforms it.\n"
        "- **Linear** (select it above) passes z through unchanged — this is what every layer would do "
        "without an activation. Notice it's just a straight line. Stacking straight lines gives you... a straight line.\n"
        "- **ReLU** clips everything negative to zero and passes positives through. Simple, fast, the default choice.\n"
        "- **Sigmoid** squashes any input into a value between 0 and 1 — useful when you need a probability.\n"
        "- **Tanh** is like sigmoid but ranges from -1 to 1, so it's zero-centered.\n"
        "- Toggle **Show derivatives** to see the gradient at each point. "
        "Flat regions (gradient near zero) are where learning slows down — the **vanishing gradient** problem."
    )

# Summary table
st.subheader("Quick Reference")
data = {
    "Function": ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "Linear"],
    "Formula": ["max(0, z)", "1 / (1 + e^-z)", "(e^z - e^-z) / (e^z + e^-z)", "max(0.01z, z)", "z"],
    "Output Range": ["[0, +inf)", "(0, 1)", "(-1, 1)", "(-inf, +inf)", "(-inf, +inf)"],
    "Best For": [
        "Hidden layers (default choice)",
        "Output layer (binary classification)",
        "Hidden layers (alternative to ReLU)",
        "Hidden layers (if ReLU causes dead neurons)",
        "Output layer (regression)",
    ],
}
st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

with st.expander("Learn more: Why do we need non-linearity?"):
    st.markdown(explanations.ACTIVATION_INTUITION)

next_step_button("pages/04_build_a_network.py", "Next: Build a Network")
