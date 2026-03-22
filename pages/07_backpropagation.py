import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.nn_engine import NeuralNetwork, backprop, mse_loss, binary_cross_entropy_loss
from components.viz_utils import plot_gradient_flow

st.title("Backpropagation")

with st.expander("Where does backpropagation fit in?", expanded=True):
    st.markdown(
        "So far you've seen the forward pass (input → prediction) and the loss function "
        "(prediction vs truth = error). Now the key question: **which weights caused the error, "
        "and how should they change?**\n\n"
        "- Backpropagation works **backwards** from the output to the input, using the **chain rule** "
        "from calculus to calculate a **gradient** for every single weight.\n"
        "- A gradient tells you two things: which **direction** to adjust a weight, "
        "and by how **much**.\n"
        "- Large gradient = this weight had a big impact on the error. "
        "Small gradient = this weight barely mattered.\n"
        "- Below, toggle between the forward and backward pass to see the numbers side by side. "
        "The gradient magnitudes chart shows whether information flows well through the network — "
        "or fades out (the vanishing gradient problem)."
    )

# Small fixed network
LAYER_SIZES = [2, 2, 1]
ACTIVATIONS = ["sigmoid", "sigmoid"]

st.info("Network: 2 -> 2 -> 1 (Sigmoid activations for clear gradients)")

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    x1 = st.number_input("Input x1", value=1.0, min_value=-3.0, max_value=3.0, step=0.1,
                          help=tooltips.BACKPROP["input_x1"])
with col2:
    x2 = st.number_input("Input x2", value=0.5, min_value=-3.0, max_value=3.0, step=0.1,
                          help=tooltips.BACKPROP["input_x2"])
with col3:
    target = st.number_input("Target output", value=1.0, min_value=0.0, max_value=1.0, step=0.1,
                             help=tooltips.BACKPROP["target"])

loss_fn_name = st.selectbox("Loss Function", ["binary_cross_entropy", "mse"],
                            format_func=lambda x: "Binary Cross-Entropy" if x == "binary_cross_entropy" else "MSE")
loss_fn = binary_cross_entropy_loss if loss_fn_name == "binary_cross_entropy" else mse_loss

seed = st.number_input("Weight seed", value=42, min_value=0, max_value=999)

# Build network
network = NeuralNetwork(LAYER_SIZES, ACTIVATIONS, seed=int(seed))
x_input = np.array([[x1, x2]])
y_true = np.array([[target]])

# Forward pass
output = network.forward(x_input)

# Backprop
loss, grads = backprop(network, x_input, y_true, loss_fn)

# Step-through
step = st.radio("Show", ["Forward Pass", "Backward Pass", "Both"], horizontal=True)

# --- Forward Pass Display ---
if step in ["Forward Pass", "Both"]:
    st.subheader("Forward Pass")
    for i, layer in enumerate(network.layers):
        with st.container(border=True):
            label = f"Hidden Layer" if i == 0 else "Output Layer"
            st.write(f"**{label}**")

            input_vals = x_input[0] if i == 0 else network.layers[i - 1].a[0]
            st.write(f"Inputs: [{', '.join(f'{v:.4f}' for v in input_vals)}]")

            # Show weight matrix
            st.write("Weights:")
            for r in range(layer.weights.shape[0]):
                st.write(f"  [{', '.join(f'{layer.weights[r, c]:.4f}' for c in range(layer.weights.shape[1]))}]")
            st.write(f"Biases: [{', '.join(f'{v:.4f}' for v in layer.biases[0])}]")

            st.write(f"Pre-activation (z): [{', '.join(f'{v:.4f}' for v in layer.z[0])}]")
            st.write(f"Post-activation (a): [{', '.join(f'{v:.4f}' for v in layer.a[0])}]")

    st.metric("Prediction", f"{output[0, 0]:.4f}")
    st.metric("Loss", f"{loss:.6f}")

# --- Backward Pass Display ---
if step in ["Backward Pass", "Both"]:
    st.subheader("Backward Pass")
    st.write(f"**Loss = {loss:.6f}**")

    # Compute dL/d(output)
    _, d_out = loss_fn(output, y_true)
    st.write(f"**Step 1:** dL/d(output) = {d_out[0, 0]:.6f}")

    # Show gradients for each layer in reverse
    for i in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[i]
        dW, db = grads[i]
        with st.container(border=True):
            label = "Output Layer" if i == len(network.layers) - 1 else f"Hidden Layer {i + 1}"
            st.write(f"**{label} Gradients**")
            st.write("dL/dW (weight gradients):")
            for r in range(dW.shape[0]):
                st.write(f"  [{', '.join(f'{dW[r, c]:.6f}' for c in range(dW.shape[1]))}]")
            st.write(f"dL/db (bias gradients): [{', '.join(f'{v:.6f}' for v in db[0])}]")

            avg_grad = np.mean(np.abs(dW))
            st.write(f"Average |gradient|: **{avg_grad:.6f}**")

    # Gradient flow visualization
    st.subheader("Gradient Magnitudes by Layer")
    layer_names = [f"Hidden {i+1}" if i < len(grads) - 1 else "Output" for i in range(len(grads))]
    fig = plot_gradient_flow(grads, layer_names)
    st.plotly_chart(fig, use_container_width=True)

    # Check for vanishing gradients
    grad_mags = [np.mean(np.abs(dW)) for dW, _ in grads]
    if len(grad_mags) > 1 and grad_mags[0] < grad_mags[-1] * 0.1:
        st.warning(
            "Notice how the gradients are much smaller in the earlier layers. "
            "This is the **vanishing gradient problem** — it makes deep networks hard to train."
        )

with st.expander("Learn more: The chain rule explained simply"):
    st.markdown(explanations.BACKPROP_INTUITION)

next_step_button("pages/08_training_loop.py", "Next: Training Loop")
