import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.nn_engine import ACTIVATIONS
from components.toy_data import DATASETS
from components.viz_utils import plot_neuron_boundary

st.title("The Neuron (Perceptron)")

with st.expander("What is a neuron and why start here?", expanded=True):
    st.markdown(
        "Every neural network — no matter how large — is built from this one building block: **the neuron**.\n\n"
        "- A neuron takes in numbers (inputs), multiplies each by a **weight**, "
        "adds a **bias**, and passes the result through an **activation function**.\n"
        "- The output is a single number — a prediction.\n"
        "- Below, you're playing the role of the network: manually adjusting the weights and bias "
        "to draw a line that separates two classes of data.\n"
        "- This is exactly what training does automatically — but first, "
        "see how hard it is to find good values by hand. That builds intuition for why we need "
        "backpropagation and optimizers later."
    )

# --- Sidebar controls ---
with st.sidebar:
    st.header("Data Settings")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), help=tooltips.NEURON["dataset"])
    noise = st.slider("Noise", 0.0, 0.5, 0.15, 0.05, help=tooltips.NEURON["noise"])
    seed = st.number_input("Seed", value=42, min_value=0, max_value=999, help=tooltips.NEURON["seed"])

# Generate data
X, y = DATASETS[dataset_name]["fn"](n_samples=200, noise=noise, seed=int(seed))

# --- Neuron controls ---
col1, col2, col3 = st.columns(3)
with col1:
    w1 = st.slider("Weight w1", -5.0, 5.0, 1.0, 0.1, help=tooltips.NEURON["weight_w1"])
with col2:
    w2 = st.slider("Weight w2", -5.0, 5.0, -1.0, 0.1, help=tooltips.NEURON["weight_w2"])
with col3:
    bias = st.slider("Bias b", -5.0, 5.0, 0.0, 0.1, help=tooltips.NEURON["bias"])

activation_name = st.selectbox(
    "Activation Function",
    ["step", "sigmoid", "relu"],
    help=tooltips.NEURON["activation"],
)
act_fn, _ = ACTIVATIONS[activation_name]

# Formula
st.latex(f"y = \\text{{{activation_name}}}({w1:.1f} \\cdot x_1 + {w2:.1f} \\cdot x_2 + {bias:.1f})")

# --- Visualization ---
fig = plot_neuron_boundary(w1, w2, bias, act_fn, X, y)
st.plotly_chart(fig, use_container_width=True)

# Reading the chart
with st.expander("How to read this chart", expanded=True):
    st.markdown(
        "- Each **dot** is a data point with two features (x1, x2). "
        "**Blue** dots are class 0, **red** dots are class 1.\n"
        "- The **dashed line** is the neuron's decision boundary — where its output equals 0.5. "
        "Points on one side are predicted as class 0, the other side as class 1.\n"
        "- The **shaded regions** show the neuron's output across the whole space.\n"
        "- **w1 and w2** control the angle of the line. **Bias** shifts it.\n"
        "- Try different **activation functions**: "
        "**Step** gives a hard 0/1 cutoff. "
        "**Sigmoid** gives a smooth probability (0 to 1). "
        "**ReLU** clips everything below 0 and passes the rest through."
    )

# Accuracy
preds = act_fn(X[:, 0] * w1 + X[:, 1] * w2 + bias)
binary_preds = (preds >= 0.5).astype(float)
accuracy = np.mean(binary_preds == y.ravel())
st.metric("Classification Accuracy", f"{accuracy:.1%}")

if accuracy < 0.6:
    st.info("Try adjusting the weights and bias to find a better line.")
elif accuracy >= 0.6 and accuracy < 1.0:
    st.success("Some datasets can't be perfectly split by a single straight line — that's why we need networks.")

with st.expander("Learn more: What is a perceptron?"):
    st.markdown(explanations.WHAT_IS_A_NEURON)

next_step_button("pages/03_activation_functions.py", "Next: Activation Functions")
