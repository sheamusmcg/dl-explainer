import streamlit as st
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.viz_utils import draw_network_diagram

st.title("Build a Network")

with st.expander("Why does architecture matter?", expanded=True):
    st.markdown(
        "A single neuron can only draw a straight line. To learn complex patterns "
        "(curves, spirals, clusters) you need to **combine many neurons into layers**.\n\n"
        "- The **input layer** receives your raw data (here, 2 features).\n"
        "- **Hidden layers** are where the learning happens — each one transforms the data "
        "into a more useful representation.\n"
        "- The **output layer** produces the final prediction "
        "(1 neuron for yes/no, more for multiple classes).\n\n"
        "Choosing how many layers and how many neurons per layer is called **architecture design** — "
        "it's the first decision you make before training. Use the controls below to experiment."
    )

# Controls
col_layers, col_output = st.columns(2)
with col_layers:
    n_hidden = st.slider("Number of hidden layers", 1, 4, 2, help=tooltips.NETWORK["num_layers"])
with col_output:
    output_neurons = st.selectbox(
        "Output neurons",
        [1, 2, 3],
        help=tooltips.NETWORK["output_neurons"],
    )

st.subheader("Hidden Layer Configuration")
hidden_sizes = []
activations = []
cols = st.columns(n_hidden)
for i in range(n_hidden):
    with cols[i]:
        st.write(f"**Hidden Layer {i + 1}**")
        n = st.slider(f"Neurons", 1, 16, 4, key=f"neurons_{i}", help=tooltips.NETWORK["neurons"])
        act = st.selectbox(
            f"Activation",
            ["relu", "sigmoid", "tanh", "leaky_relu"],
            key=f"act_{i}",
            help=tooltips.NETWORK["activation"],
        )
        hidden_sizes.append(n)
        activations.append(act)

# Build layer sizes list
input_dim = 2
layer_sizes = [input_dim] + hidden_sizes + [output_neurons]

# Architecture diagram
fig = draw_network_diagram(layer_sizes)
st.pyplot(fig)

# Reading the diagram
with st.expander("How to read this diagram", expanded=True):
    st.markdown(
        "- Each **circle** is a neuron. Every neuron in one layer connects to every neuron in the next — "
        "that's why these are called **fully connected** (or dense) layers.\n"
        "- Each connection is a **weight** — a number the network learns during training. "
        "More connections = more parameters = more capacity to learn complex patterns.\n"
        "- **Adding neurons** (wider) lets a layer represent more features at that stage.\n"
        "- **Adding layers** (deeper) lets the network build higher-level abstractions — "
        "layer 1 might detect edges, layer 2 combines edges into shapes, etc.\n"
        "- The tradeoff: bigger networks are more powerful but need more data and are more prone to overfitting."
    )

# Parameter count
total_weights = 0
total_biases = 0
for i in range(len(layer_sizes) - 1):
    w = layer_sizes[i] * layer_sizes[i + 1]
    b = layer_sizes[i + 1]
    total_weights += w
    total_biases += b

st.subheader("Network Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Parameters", f"{total_weights + total_biases:,}")
col2.metric("Weights", f"{total_weights:,}")
col3.metric("Biases", f"{total_biases:,}")

# Layer-by-layer breakdown
st.write("**Layer-by-layer breakdown:**")
labels = ["Input"] + [f"Hidden {i+1}" for i in range(n_hidden)] + ["Output"]
for i in range(len(layer_sizes) - 1):
    w = layer_sizes[i] * layer_sizes[i + 1]
    b = layer_sizes[i + 1]
    act_label = activations[i] if i < len(activations) else ("sigmoid" if output_neurons == 1 else "softmax")
    st.write(
        f"- {labels[i]} ({layer_sizes[i]}) -> {labels[i+1]} ({layer_sizes[i+1]}): "
        f"**{w} weights + {b} biases = {w+b} params** (activation: {act_label})"
    )

with st.expander("Learn more: How deep is deep enough?"):
    st.markdown(explanations.NETWORK_DEPTH)

next_step_button("pages/05_forward_pass.py", "Next: Forward Pass")
