import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.nn_engine import (
    NeuralNetwork, Adam, binary_cross_entropy_loss, train_network,
)
from components.toy_data import DATASETS
from components.viz_utils import plot_decision_boundary_2d, plot_loss_curve

st.title("Overfitting & Regularization")

with st.expander("What is overfitting and why does it matter?", expanded=True):
    st.markdown(
        "A network that scores 99% on training data but fails on new data hasn't **learned** — "
        "it's **memorized**. That's overfitting.\n\n"
        "- To detect it, we split data into **training** (what the network learns from) "
        "and **validation** (data it never sees during training — used only to test).\n"
        "- If training loss keeps dropping but validation loss starts **rising**, "
        "the network is memorizing noise in the training data instead of learning the real pattern.\n"
        "- **Bigger networks** overfit more easily — they have enough parameters to memorize everything.\n\n"
        "**Regularization** techniques fight overfitting:\n"
        "- **L2 Regularization** — penalizes large weights, forcing the network to keep things simple.\n"
        "- **Dropout** — randomly disables neurons during training so the network can't rely on any single one.\n"
        "- **Early Stopping** — stops training when validation loss stops improving, before memorization kicks in.\n\n"
        "Try training with **Large (64 neurons)** and **no regularization** first — watch the gap appear. "
        "Then toggle techniques on in the sidebar to close it."
    )

# --- Controls ---
with st.sidebar:
    st.header("Data")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), help=tooltips.TRAINING["dataset"])
    noise = st.slider("Noise (more = harder)", 0.0, 0.8, 0.25, 0.05, help=tooltips.OVERFITTING["noise"])
    seed = st.number_input("Seed", value=42, min_value=0, max_value=999)
    val_split = st.slider("Validation split", 0.1, 0.5, 0.2, 0.05, help=tooltips.OVERFITTING["val_split"])

    st.header("Network")
    complexity = st.selectbox("Network complexity", ["Small (4 neurons)", "Medium (16 neurons)", "Large (64 neurons)"],
                              index=1, help=tooltips.OVERFITTING["complexity"])
    complexity_map = {"Small (4 neurons)": 4, "Medium (16 neurons)": 16, "Large (64 neurons)": 64}
    neurons = complexity_map[complexity]

    st.header("Regularization")
    use_l2 = st.toggle("L2 Regularization", value=False)
    l2_lambda = st.slider("L2 strength", 0.0001, 0.1, 0.01, 0.001,
                           help=tooltips.OVERFITTING["l2_lambda"]) if use_l2 else 0.0

    use_dropout = st.toggle("Dropout", value=False)
    dropout_rate = st.slider("Dropout rate", 0.1, 0.5, 0.2, 0.05,
                             help=tooltips.OVERFITTING["dropout_rate"]) if use_dropout else 0.0

    use_early_stop = st.toggle("Early Stopping", value=False)
    patience = st.slider("Patience (epochs)", 5, 50, 20, 5,
                         help=tooltips.OVERFITTING["early_stopping_patience"]) if use_early_stop else 0

    epochs = st.slider("Max Epochs", 50, 1000, 300, 50)

# Generate data
X, y = DATASETS[dataset_name]["fn"](n_samples=400, noise=noise, seed=int(seed))

# Split into train/val
n_val = int(len(X) * val_split)
indices = np.random.RandomState(int(seed)).permutation(len(X))
val_idx, train_idx = indices[:n_val], indices[n_val:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

st.write(f"Training: {len(X_train)} samples | Validation: {len(X_val)} samples")

if st.button("Train", type="primary"):
    layer_sizes = [2, neurons, neurons, 1]
    activations = ["relu", "relu", "sigmoid"]
    network = NeuralNetwork(layer_sizes, activations, seed=int(seed))
    optimizer = Adam(learning_rate=0.01)

    # Train with optional regularization
    result = train_network(
        network, X_train, y_train, optimizer, binary_cross_entropy_loss,
        epochs=epochs,
        l2_lambda=l2_lambda,
        dropout_rate=dropout_rate,
        x_val=X_val, y_val=y_val,
        early_stopping_patience=patience,
    )

    train_losses = result["train_losses"]
    val_losses = result["val_losses"]
    stopped = result["stopped_epoch"]

    if stopped < epochs and use_early_stop:
        st.info(f"Early stopping triggered at epoch {stopped} (patience={patience}).")

    # Loss curves
    st.subheader("Training vs Validation Loss")
    fig_loss = plot_loss_curve(train_losses, val_losses, title="Loss Over Epochs")
    st.plotly_chart(fig_loss, use_container_width=True)

    with st.expander("How to read this chart"):
        st.markdown(
            "- The **training loss** (blue) shows how well the network fits the data it's learning from.\n"
            "- The **validation loss** (red) shows how well it performs on data it has **never seen**.\n"
            "- Healthy training: both lines drop together and stay close.\n"
            "- **Overfitting**: training loss keeps dropping but validation loss **levels off or rises** — "
            "the growing gap means the network is memorizing, not learning.\n"
            "- Regularization works when it **closes that gap**, even if training loss is slightly higher."
        )

    # Check for overfitting
    if val_losses:
        gap = val_losses[-1] - train_losses[-1]
        if gap > 0.1:
            st.warning(
                f"Overfitting detected! Validation loss ({val_losses[-1]:.4f}) is much higher "
                f"than training loss ({train_losses[-1]:.4f}). Try enabling regularization."
            )
        elif gap > 0.03:
            st.info("Slight overfitting. The gap between train and validation loss is moderate.")
        else:
            st.success("Good generalization! Train and validation loss are close.")

    # Decision boundary
    st.subheader("Decision Boundary")
    col1, col2 = st.columns(2)
    with col1:
        fig_train = plot_decision_boundary_2d(network, X_train, y_train, title="Training Data")
        st.plotly_chart(fig_train, use_container_width=True)
    with col2:
        fig_val = plot_decision_boundary_2d(network, X_val, y_val, title="Validation Data")
        st.plotly_chart(fig_val, use_container_width=True)

    with st.expander("How to read the decision boundaries"):
        st.markdown(
            "- **Left (Training Data)** — the boundary the network learned, shown with the data it trained on. "
            "This usually looks good.\n"
            "- **Right (Validation Data)** — the *same* boundary, but tested against unseen data. "
            "This is the real test.\n"
            "- If the boundary is **smooth** and both charts look similar, the network is generalizing well.\n"
            "- If the boundary is **jagged and complex** on the left but misclassifies many dots on the right, "
            "that's overfitting — it contorted itself to fit training noise that doesn't exist in new data.\n"
            "- Regularization simplifies the boundary, which often *improves* validation accuracy even though "
            "training accuracy drops slightly."
        )

    # Metrics
    train_preds = network.predict(X_train)
    val_preds = network.predict(X_val)
    train_acc = np.mean(train_preds == y_train.ravel().reshape(-1, 1))
    val_acc = np.mean(val_preds == y_val.ravel().reshape(-1, 1))

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Training Accuracy", f"{train_acc:.1%}")
    col_m2.metric("Validation Accuracy", f"{val_acc:.1%}")
    col_m3.metric("Accuracy Gap", f"{(train_acc - val_acc):.1%}")

    # Summary of regularization used
    reg_used = []
    if use_l2:
        reg_used.append(f"L2 (lambda={l2_lambda})")
    if use_dropout:
        reg_used.append(f"Dropout (rate={dropout_rate})")
    if use_early_stop:
        reg_used.append(f"Early Stopping (patience={patience})")
    if reg_used:
        st.write(f"**Regularization:** {', '.join(reg_used)}")
    else:
        st.write("**No regularization applied.** Try toggling some on in the sidebar.")

with st.expander("Learn more: The bias-variance tradeoff"):
    st.markdown(explanations.OVERFITTING_INTUITION)

next_step_button("pages/11_digit_recognition.py", "Next: See It In Action")
