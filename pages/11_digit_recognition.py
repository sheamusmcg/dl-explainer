import streamlit as st
import numpy as np
import warnings
from sklearn.datasets import load_digits
from components.ui_helpers import next_step_button
from components.nn_engine import (
    NeuralNetwork, Adam, categorical_cross_entropy_loss, backprop,
)
from components.viz_utils import (
    plot_digit_image, plot_digit_grid, plot_prediction_bars, plot_loss_curve,
)

st.title("See It In Action: Digit Recognition")

with st.expander("From toy data to a real task", expanded=True):
    st.markdown(
        "Everything you've learned — neurons, layers, forward pass, loss, backprop, training — "
        "comes together here on a **real problem**: recognizing handwritten digits.\n\n"
        "- The input is an **8×8 grayscale image** (64 pixels). Each pixel is a number (0-16) "
        "representing brightness — that's 64 input features instead of the 2 we used before.\n"
        "- The output is **10 probabilities** (one per digit 0-9), produced by a **softmax** layer. "
        "The network picks the digit with the highest probability.\n"
        "- It's the exact same process: forward pass → loss → backprop → update weights → repeat. "
        "Just more inputs and more outputs."
    )

# --- Load data ---
digits = load_digits()
X_all = digits.data / 16.0  # Normalize to 0-1
y_all_raw = digits.target  # 0-9 integers

# One-hot encode targets
n_classes = 10
y_all_onehot = np.zeros((len(y_all_raw), n_classes))
y_all_onehot[np.arange(len(y_all_raw)), y_all_raw] = 1.0

# Train/test split (80/20)
rng = np.random.RandomState(42)
indices = rng.permutation(len(X_all))
split = int(0.8 * len(X_all))
train_idx, test_idx = indices[:split], indices[split:]
X_train, y_train = X_all[train_idx], y_all_onehot[train_idx]
X_test, y_test_onehot = X_all[test_idx], y_all_onehot[test_idx]
y_test_labels = y_all_raw[test_idx]

st.write(f"**{len(X_train)}** training images, **{len(X_test)}** test images, **10** digit classes")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Network Settings")
    arch_name = st.selectbox(
        "Architecture",
        ["Small (64→32→10)", "Medium (64→64→32→10)", "Large (64→128→64→10)"],
        index=1,
    )
    arch_map = {
        "Small (64→32→10)": ([64, 32, 10], ["relu", "softmax"]),
        "Medium (64→64→32→10)": ([64, 64, 32, 10], ["relu", "relu", "softmax"]),
        "Large (64→128→64→10)": ([64, 128, 64, 10], ["relu", "relu", "softmax"]),
    }
    layer_sizes, activations = arch_map[arch_name]

    epochs = st.slider("Epochs", 50, 500, 200, 25)
    lr = st.select_slider(
        "Learning rate",
        options=[0.001, 0.005, 0.01, 0.05],
        value=0.01,
    )

# --- Train ---
if st.button("Train Network", type="primary"):
    network = NeuralNetwork(layer_sizes, activations, seed=42)
    optimizer = Adam(learning_rate=lr)

    progress_bar = st.progress(0)
    status = st.empty()
    loss_placeholder = st.empty()

    train_losses = []
    update_interval = max(1, epochs // 20)

    for epoch in range(epochs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loss, grads = backprop(network, X_train, y_train, categorical_cross_entropy_loss)
            optimizer.update(network, grads)
        train_losses.append(loss)

        if epoch % update_interval == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs)
            status.write(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f}")
            fig_loss = plot_loss_curve(train_losses, title="Training Loss")
            loss_placeholder.plotly_chart(fig_loss, use_container_width=True, key=f"train_loss_{epoch}")

    progress_bar.progress(1.0)
    status.write(f"Training complete! Final loss: {train_losses[-1]:.4f}")

    # Compute results
    train_preds = network.predict(X_train)
    test_preds = network.predict(X_test)
    y_train_labels = y_all_raw[train_idx]
    train_acc = np.mean(train_preds.ravel() == y_train_labels)
    test_acc = np.mean(test_preds.ravel() == y_test_labels)

    # Store in session state so results survive slider changes
    st.session_state["digit_network"] = network
    st.session_state["digit_train_acc"] = train_acc
    st.session_state["digit_test_acc"] = test_acc
    st.session_state["digit_test_preds"] = test_preds
    st.session_state["digit_train_losses"] = train_losses

# --- Show results if a trained network exists ---
if "digit_network" in st.session_state:
    network = st.session_state["digit_network"]
    train_acc = st.session_state["digit_train_acc"]
    test_acc = st.session_state["digit_test_acc"]
    test_preds = st.session_state["digit_test_preds"]
    train_losses = st.session_state["digit_train_losses"]

    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{train_acc:.1%}")
    col2.metric("Test Accuracy", f"{test_acc:.1%}")

    acc_gap = train_acc - test_acc
    if acc_gap > 0.05:
        st.info(
            f"Training accuracy is {acc_gap:.0%} higher than test — the network fits the training data "
            "better than unseen data. This is the same overfitting you saw on the previous page."
        )
    elif test_acc > 0.9:
        st.success("Strong test accuracy — the network generalizes well to digits it's never seen.")
    else:
        st.info("Try a larger architecture or more epochs to improve accuracy.")

    fig_loss = plot_loss_curve(train_losses, title="Training Loss")
    st.plotly_chart(fig_loss, use_container_width=True, key="results_loss")

    st.divider()

    # --- Test the Network ---
    st.subheader("Test the Network")
    st.write("Scroll through the test images — the network trained on other handwritten samples of each digit, but **never saw these specific images**.")

    test_index = st.slider(
        f"Test image (1–{len(X_test)})",
        0, len(X_test) - 1, 0,
        help="Each position is a different handwritten digit from the test set.",
    )

    col_img, col_bars = st.columns([1, 2])

    # Get the raw probabilities
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        probs = network.forward(X_test[test_index:test_index + 1])[0]
    pred = int(np.argmax(probs))
    true = int(y_test_labels[test_index])

    with col_img:
        fig_digit = plot_digit_image(X_test[test_index] * 16, prediction=pred, true_label=true)
        st.pyplot(fig_digit)

    with col_bars:
        fig_bars = plot_prediction_bars(probs, true_label=true)
        st.plotly_chart(fig_bars, use_container_width=True)

    with st.expander("How to read this"):
        st.markdown(
            "- The **image** on the left is the 8×8 handwritten digit the network is looking at.\n"
            "- The **bar chart** on the right shows the network's confidence for each digit 0-9.\n"
            "- The **blue bar** is the true label. The **green bar** is the network's prediction "
            "(red if it's wrong).\n"
            "- A confident correct prediction has one tall green bar that matches the blue bar. "
            "A confused network spreads probability across multiple digits."
        )

    # Sample grid
    st.subheader("Sample Predictions")
    n_samples = 16
    sample_rng = np.random.RandomState(42)
    sample_idx = sample_rng.choice(len(X_test), n_samples, replace=False)
    sample_images = X_test[sample_idx] * 16
    sample_preds = test_preds.ravel()[sample_idx]
    sample_true = y_test_labels[sample_idx]

    fig_grid = plot_digit_grid(sample_images, sample_preds, sample_true, cols=8)
    st.pyplot(fig_grid)

    n_correct = sum(int(p) == int(t) for p, t in zip(sample_preds, sample_true))
    st.write(f"**{n_correct}/{n_samples}** correct in this sample. "
             f"Green ✓ = correct, Red ✗ = wrong.")

    if n_correct < n_samples:
        st.caption(
            "Mistakes usually happen on ambiguous handwriting — a messy 4 can look like a 9, "
            "a 3 can resemble an 8. Even humans struggle with some of these. "
            "Scroll up and use the test image slider to find the misclassified ones and see what confused the network."
        )

    with st.expander("What's different about real deep learning?"):
        st.markdown(
            "This demo uses the same NumPy engine from the entire course — no frameworks, no GPU. "
            "Here's what changes at scale:\n\n"
            "- **Bigger images** — real MNIST is 28×28 (784 pixels), ImageNet is 224×224 (150,000+ pixels). "
            "Dense layers can't handle that efficiently.\n"
            "- **CNNs** (Convolutional Neural Networks) — instead of connecting every pixel to every neuron, "
            "they slide small filters across the image, detecting edges, textures, and shapes. "
            "Far fewer parameters, much better at spatial patterns.\n"
            "- **Frameworks** (PyTorch, TensorFlow) — handle the backprop math, GPU acceleration, "
            "data loading, and model saving automatically.\n"
            "- **GPUs** — training millions of parameters on thousands of images requires parallel hardware. "
            "What takes minutes on a GPU could take hours on a CPU."
        )

next_step_button("pages/12_whats_next.py", "Next: What's Next")
