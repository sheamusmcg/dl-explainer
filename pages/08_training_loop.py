import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button
from components import tooltips, explanations
from components.nn_engine import (
    NeuralNetwork, backprop, binary_cross_entropy_loss, SGD, Adam, RMSprop, train_network,
)
from components.toy_data import DATASETS
from components.viz_utils import plot_decision_boundary_2d, plot_loss_curve

st.title("Training Loop")

with st.expander("This is where it all comes together", expanded=True):
    st.markdown(
        "You've now seen every piece individually — neurons, activations, forward pass, loss, backprop. "
        "The training loop is what **puts them all together in a cycle**:\n\n"
        "1. **Forward pass** — feed data through the network, get a prediction.\n"
        "2. **Loss** — measure how wrong the prediction is.\n"
        "3. **Backprop** — calculate which direction to adjust each weight.\n"
        "4. **Update weights** — nudge every weight a small step in the right direction.\n"
        "5. **Repeat** — do this hundreds or thousands of times (each pass = one **epoch**).\n\n"
        "Below, hit **Train** and watch two things happen in real time: the **loss curve** dropping "
        "(the network is getting less wrong) and the **decision boundary** shifting "
        "(the network is learning to separate the classes). "
        "Try different learning rates — too high and it overshoots, too low and it barely moves."
    )

# --- Controls ---
with st.sidebar:
    st.header("Training Settings")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), help=tooltips.TRAINING["dataset"])
    noise = st.slider("Noise", 0.0, 0.5, 0.15, 0.05)
    seed = st.number_input("Seed", value=42, min_value=0, max_value=999)

    st.subheader("Network")
    n_hidden = st.slider("Hidden layers", 1, 3, 1, help=tooltips.TRAINING["hidden_layers"])
    neurons = st.slider("Neurons per layer", 2, 16, 8, help=tooltips.TRAINING["neurons_per_layer"])
    activation = st.selectbox("Activation", ["relu", "sigmoid", "tanh"],
                              help=tooltips.TRAINING["activation"])

    st.subheader("Training")
    lr = st.select_slider(
        "Learning rate",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        value=0.1,
        help=tooltips.TRAINING["learning_rate"],
    )
    epochs = st.slider("Epochs", 10, 500, 200, 10, help=tooltips.TRAINING["epochs"])

# Generate data
X, y = DATASETS[dataset_name]["fn"](n_samples=300, noise=noise, seed=int(seed))

# Show data
st.subheader(f"Dataset: {dataset_name}")
st.write(DATASETS[dataset_name]["description"])

with st.expander("What happens when you hit Train?", expanded=True):
    st.markdown(
        "- The **left chart** shows the **decision boundary** — the network's best guess at where to "
        "draw the line (or curve) between the two classes. Blue region = predicted class 0, "
        "red region = predicted class 1.\n"
        "- At epoch 1 the boundary is random nonsense. As training progresses, it bends and shifts "
        "to wrap around the data.\n"
        "- **Why does it curve?** Each neuron in a hidden layer can only draw a straight cut. "
        "But when you combine many straight cuts through a non-linear activation, the network can "
        "approximate any shape — that's the power of depth and width.\n"
        "- The **right chart** shows the **loss curve** — it should drop quickly at first then level off. "
        "If it plateaus high, the network isn't powerful enough. If it oscillates wildly, "
        "the learning rate is too high.\n"
        "- Try the **Spiral** dataset with 1 hidden layer — it will fail. "
        "Add more layers or neurons to see what it takes."
    )

# Build network
layer_sizes = [2] + [neurons] * n_hidden + [1]
activations = [activation] * n_hidden + ["sigmoid"]

if st.button("Train Network", type="primary"):
    network = NeuralNetwork(layer_sizes, activations, seed=int(seed))
    optimizer = Adam(learning_rate=lr)

    # Placeholders for live updates
    progress_bar = st.progress(0)
    status_text = st.empty()
    col_boundary, col_loss = st.columns(2)
    boundary_placeholder = col_boundary.empty()
    loss_placeholder = col_loss.empty()

    train_losses = []
    update_interval = max(1, epochs // 20)

    for epoch in range(epochs):
        # Forward + backward
        loss, grads = backprop(network, X, y, binary_cross_entropy_loss)
        optimizer.update(network, grads)
        train_losses.append(loss)

        # Live update
        if epoch % update_interval == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs)
            status_text.write(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f}")

            fig_boundary = plot_decision_boundary_2d(network, X, y,
                                                     title=f"Epoch {epoch + 1}")
            boundary_placeholder.plotly_chart(fig_boundary, use_container_width=True)

            fig_loss = plot_loss_curve(train_losses, title="Training Loss")
            loss_placeholder.plotly_chart(fig_loss, use_container_width=True)

    progress_bar.progress(1.0)
    status_text.write(f"Training complete! Final loss: {train_losses[-1]:.4f}")

    # Final accuracy
    preds = network.predict(X)
    accuracy = np.mean(preds == y.ravel().reshape(-1, 1))
    st.metric("Final Accuracy", f"{accuracy:.1%}")

    # Show what the network learned
    w_total, b_total = network.count_parameters()
    st.write(
        f"Network: {' -> '.join(str(s) for s in layer_sizes)} | "
        f"Parameters: {w_total + b_total} | "
        f"Optimizer: Adam (lr={lr})"
    )
else:
    # Show just the data before training
    import plotly.graph_objects as go
    y_flat = y.ravel()
    colors = ["#4A90D9" if v == 0 else "#FF6B6B" for v in y_flat]
    fig = go.Figure(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(color=colors, size=6, line=dict(width=0.5, color="white")),
    ))
    fig.update_layout(
        title="Data (click Train to start)",
        xaxis_title="x1", yaxis_title="x2",
        width=500, height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Learn more: What is an epoch?"):
    st.markdown(explanations.TRAINING_LOOP_EXPLANATION)

next_step_button("pages/09_optimizers.py", "Next: Optimizers")
