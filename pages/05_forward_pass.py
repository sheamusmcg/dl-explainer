import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.nn_engine import NeuralNetwork
from components.viz_utils import draw_forward_pass_diagram

st.title("Forward Pass")

with st.expander("What is a forward pass?", expanded=True):
    st.markdown(
        "A forward pass is data flowing **left to right** through the network — input in, prediction out. "
        "This is how the network makes a prediction.\n\n"
        "- Your input enters at the left (e.g. two features of a data point).\n"
        "- At each layer, every neuron multiplies the incoming values by its weights, adds its bias, "
        "and passes the result through an activation function.\n"
        "- The output of one layer becomes the input to the next.\n"
        "- The final layer produces the **prediction** — which then gets compared to the true answer "
        "by the loss function (next page).\n\n"
        "Right now the weights are random, so the prediction is meaningless. "
        "Training (coming later) adjusts these weights so the forward pass produces useful predictions."
    )

# Fixed small network for clarity
LAYER_SIZES = [2, 3, 2, 1]
ACTIVATIONS = ["relu", "relu", "sigmoid"]

st.info(f"Network: {' -> '.join(str(s) for s in LAYER_SIZES)} (ReLU hidden layers, Sigmoid output)")

# Inputs
st.subheader("Input Values")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    x1 = st.number_input("x1", value=0.5, min_value=-3.0, max_value=3.0, step=0.1,
                          help=tooltips.FORWARD_PASS["input_x1"])
with col2:
    x2 = st.number_input("x2", value=-0.3, min_value=-3.0, max_value=3.0, step=0.1,
                          help=tooltips.FORWARD_PASS["input_x2"])
with col3:
    seed = st.number_input("Weight seed", value=42, min_value=0, max_value=999,
                           help="Change to get different random weights")

# Build network with fixed seed for reproducibility
network = NeuralNetwork(LAYER_SIZES, ACTIVATIONS, seed=int(seed))

# Run forward pass
x_input = np.array([[x1, x2]])
output = network.forward(x_input)

# Step-through control
step = st.slider("Step through layers", 0, len(network.layers),
                 len(network.layers), help=tooltips.FORWARD_PASS["step"])

# Collect all layer values for visualization
layer_values = [[x1, x2]]
for i, layer in enumerate(network.layers):
    if i < step:
        vals = layer.a[0].tolist()
    else:
        vals = [0.0] * layer.n_out
    layer_values.append(vals)

weights_list = [layer.weights for layer in network.layers] if step > 0 else None

# Draw diagram with values
fig = draw_forward_pass_diagram(LAYER_SIZES, layer_values, weights_list)
st.pyplot(fig)

with st.expander("How to read this diagram", expanded=True):
    st.markdown(
        "- The **numbers inside the circles** are each neuron's output value after activation.\n"
        "- The **numbers on the lines** are the weights — how strongly one neuron's output feeds into the next.\n"
        "- Use the **step slider** to reveal the computation one layer at a time. "
        "At step 0 you see only the inputs. Each step computes one more layer.\n"
        "- At each layer, every neuron does: **multiply inputs by weights, add bias, apply activation**. "
        "The math below the diagram shows this calculation for each neuron.\n"
        "- Try changing **x1 and x2** to see how different inputs produce different outputs. "
        "Change the **weight seed** to see how different random weights lead to completely different results — "
        "that's why training matters."
    )

# Step-by-step math
st.subheader("Step-by-Step Computation")
for i, layer in enumerate(network.layers):
    if i >= step:
        break

    with st.container(border=True):
        act_name = ACTIVATIONS[i]
        layer_label = f"Hidden {i+1}" if i < len(network.layers) - 1 else "Output"
        st.write(f"**Layer {i+1} ({layer_label}) — {act_name} activation**")

        # Show inputs to this layer
        if i == 0:
            input_vals = np.array([x1, x2])
        else:
            input_vals = network.layers[i - 1].a[0]

        # Weighted sum: z = W^T * input + b
        z_vals = layer.z[0]
        a_vals = layer.a[0]

        for j in range(layer.n_out):
            terms = " + ".join(
                f"({input_vals[k]:.2f} * {layer.weights[k, j]:.2f})"
                for k in range(layer.n_in)
            )
            st.write(
                f"  Neuron {j+1}: z = {terms} + {layer.biases[0, j]:.2f} = **{z_vals[j]:.4f}**"
                f"  ->  {act_name}({z_vals[j]:.4f}) = **{a_vals[j]:.4f}**"
            )

# Final output
if step == len(network.layers):
    st.success(f"Final output: **{output[0, 0]:.4f}** (prediction: {'Class 1' if output[0, 0] >= 0.5 else 'Class 0'})")

with st.expander("Learn more: What happens inside a neural network?"):
    st.markdown(explanations.FORWARD_PASS_EXPLANATION)

next_step_button("pages/06_loss_functions.py", "Next: Loss Functions")
