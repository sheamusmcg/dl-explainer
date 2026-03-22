import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.nn_engine import (
    NeuralNetwork, backprop, binary_cross_entropy_loss,
    SGD, Adam, RMSprop,
)
from components.toy_data import DATASETS
from components.viz_utils import plot_decision_boundary_2d, plot_optimizer_loss_curves

st.title("Optimizers")

with st.expander("What is an optimizer and why are there different ones?", expanded=True):
    st.markdown(
        "Backpropagation tells you the **gradient** — which direction each weight should move "
        "to reduce the loss. The **optimizer** decides **how far** to actually move.\n\n"
        "The simplest approach is **SGD (Stochastic Gradient Descent)**: just multiply the gradient "
        "by a fixed learning rate and subtract. It works, but it can be slow and get stuck.\n\n"
        "Smarter optimizers improve on this:\n"
        "- **SGD + Momentum** — remembers previous steps and builds up speed in consistent directions, "
        "like a ball rolling downhill. Helps push through flat spots.\n"
        "- **RMSprop** — tracks how big each weight's gradients have been recently. "
        "Weights with large gradients get smaller steps, weights with small gradients get larger ones. "
        "Adapts the learning rate per weight.\n"
        "- **Adam** — combines momentum *and* adaptive learning rates. "
        "The most popular default choice in practice.\n\n"
        "Select optimizers below and hit **Train All** — they all start from the same random weights. "
        "Compare the **loss curves** (which drops fastest?) and the **decision boundaries** "
        "(do they end up in different places?)."
    )

# --- Controls ---
with st.sidebar:
    st.header("Settings")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), help=tooltips.TRAINING["dataset"])
    noise = st.slider("Noise", 0.0, 0.5, 0.15, 0.05)
    seed = st.number_input("Seed", value=42, min_value=0, max_value=999)
    neurons = st.slider("Neurons per hidden layer", 2, 16, 8)
    epochs = st.slider("Epochs", 50, 500, 200, 10)

# Optimizer selection
selected_opts = st.multiselect(
    "Select optimizers to compare",
    ["SGD (lr=0.1)", "SGD + Momentum (lr=0.1)", "Adam (lr=0.01)", "RMSprop (lr=0.01)"],
    default=["SGD (lr=0.1)", "Adam (lr=0.01)"],
    help=tooltips.OPTIMIZERS["optimizers"],
)

# Map selections to optimizer constructors
OPT_MAP = {
    "SGD (lr=0.1)": lambda: SGD(learning_rate=0.1),
    "SGD + Momentum (lr=0.1)": lambda: SGD(learning_rate=0.1, momentum=0.9),
    "Adam (lr=0.01)": lambda: Adam(learning_rate=0.01),
    "RMSprop (lr=0.01)": lambda: RMSprop(learning_rate=0.01),
}

X, y = DATASETS[dataset_name]["fn"](n_samples=300, noise=noise, seed=int(seed))

if st.button("Train All", type="primary") and selected_opts:
    layer_sizes = [2, neurons, neurons, 1]
    activations = ["relu", "relu", "sigmoid"]

    all_losses = {}
    all_networks = {}

    progress = st.progress(0)
    status = st.empty()

    for opt_idx, opt_name in enumerate(selected_opts):
        status.write(f"Training with **{opt_name}**...")
        network = NeuralNetwork(layer_sizes, activations, seed=int(seed))
        optimizer = OPT_MAP[opt_name]()

        losses = []
        for epoch in range(epochs):
            loss, grads = backprop(network, X, y, binary_cross_entropy_loss)
            optimizer.update(network, grads)
            losses.append(loss)

        all_losses[opt_name] = losses
        all_networks[opt_name] = network
        progress.progress((opt_idx + 1) / len(selected_opts))

    status.write("Training complete!")
    progress.empty()

    # Loss curves comparison
    st.subheader("Loss Curves")
    fig_losses = plot_optimizer_loss_curves(all_losses)
    st.plotly_chart(fig_losses, use_container_width=True)

    with st.expander("How to read the loss curves"):
        st.markdown(
            "- Each line is one optimizer training the **same network** from the **same starting weights**.\n"
            "- The x-axis is epochs (training iterations), the y-axis is loss (lower = better).\n"
            "- A line that drops **faster** means that optimizer converges quicker.\n"
            "- A line that drops **lower** means it found a better solution.\n"
            "- **Jagged lines** mean the optimizer is overshooting — the learning rate may be too high.\n"
            "- A line that **flattens early** and stays high means it got stuck — "
            "momentum or adaptive rates can help escape."
        )

    # Decision boundaries side by side
    st.subheader("Decision Boundaries")
    n_cols = min(len(selected_opts), 2)
    for row_start in range(0, len(selected_opts), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < len(selected_opts):
                opt_name = selected_opts[idx]
                with col:
                    fig = plot_decision_boundary_2d(
                        all_networks[opt_name], X, y,
                        title=opt_name,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read the decision boundaries"):
        st.markdown(
            "- Each chart shows what **one optimizer learned** after training.\n"
            "- The **blue region** is where the network predicts class 0, "
            "the **red region** is where it predicts class 1.\n"
            "- The **dots** are the actual data points — correctly classified dots sit in the matching color region.\n"
            "- Different optimizers can end up with **different boundaries** even from the same starting point, "
            "because they take different paths through the weight space.\n"
            "- A boundary that closely hugs the data is more accurate, but if it's too jagged "
            "it may be overfitting (next page)."
        )

    # Summary table
    st.subheader("Summary")
    for opt_name in selected_opts:
        losses = all_losses[opt_name]
        final_loss = losses[-1]
        # Find convergence epoch (first epoch where loss < 1.1 * final_loss)
        threshold = final_loss * 1.1
        converge_epoch = next((i for i, l in enumerate(losses) if l <= threshold), epochs)
        preds = all_networks[opt_name].predict(X)
        acc = np.mean(preds == y.ravel().reshape(-1, 1))
        st.write(f"**{opt_name}:** Final loss = {final_loss:.4f}, ~converged at epoch {converge_epoch}, accuracy = {acc:.1%}")

elif not selected_opts:
    st.warning("Select at least one optimizer to compare.")

with st.expander("Learn more: Why do we need different optimizers?"):
    st.markdown(explanations.OPTIMIZER_INTUITION)

next_step_button("pages/10_overfitting.py", "Next: Overfitting & Regularization")
