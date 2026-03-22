"""Visualization helpers using Plotly and Matplotlib."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Plotly: Decision boundary
# ---------------------------------------------------------------------------


def plot_decision_boundary_2d(network, X, y, title="Decision Boundary", resolution=80):
    """Plot decision boundary as a contour with data points overlaid."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = network.forward(grid)
    if Z.shape[1] == 1:
        Z = Z.reshape(xx.shape)
    else:
        Z = np.argmax(Z, axis=1).reshape(xx.shape)

    y_flat = y.ravel()

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z,
        colorscale=[[0, "#4A90D9"], [1, "#FF6B6B"]],
        opacity=0.3,
        showscale=False,
        contours=dict(showlines=False),
    ))
    colors = ["#4A90D9" if v == 0 else "#FF6B6B" for v in y_flat]
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode="markers",
        marker=dict(color=colors, size=6, line=dict(width=0.5, color="white")),
        showlegend=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="x1", yaxis_title="x2",
        width=500, height=450,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: Perceptron decision boundary (single neuron, linear)
# ---------------------------------------------------------------------------


def plot_neuron_boundary(w1, w2, bias, activation_fn, X, y, title="Neuron Decision Boundary"):
    """Plot a single neuron's decision boundary on 2D data."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    resolution = 100
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    z = activation_fn(grid[:, 0] * w1 + grid[:, 1] * w2 + bias)
    Z = z.reshape(xx.shape)

    y_flat = y.ravel()
    colors = ["#4A90D9" if v == 0 else "#FF6B6B" for v in y_flat]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z,
        colorscale=[[0, "#4A90D9"], [1, "#FF6B6B"]],
        opacity=0.3,
        showscale=False,
        contours=dict(showlines=True, showlabels=False),
    ))

    # Decision boundary line: w1*x + w2*y + b = 0
    if abs(w2) > 1e-6:
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(w1 * x_line + bias) / w2
        mask = (y_line >= y_min) & (y_line <= y_max)
        fig.add_trace(go.Scatter(
            x=x_line[mask], y=y_line[mask],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name="boundary (z=0)",
        ))

    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode="markers",
        marker=dict(color=colors, size=7, line=dict(width=0.5, color="white")),
        showlegend=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="x1", yaxis_title="x2",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        width=550, height=450,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: Activation function plots
# ---------------------------------------------------------------------------


def plot_activation_functions(selected, x_range=(-5, 5), show_derivative=False):
    """Plot one or more activation functions (and optionally their derivatives)."""
    from components.nn_engine import ACTIVATIONS

    x = np.linspace(x_range[0], x_range[1], 300)
    colors = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, name in enumerate(selected):
        fn, deriv = ACTIVATIONS[name]
        y = fn(x)
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            name=name, line=dict(color=color, width=2.5),
        ))
        if show_derivative:
            dy = deriv(x)
            fig.add_trace(go.Scatter(
                x=x, y=dy, mode="lines",
                name=f"{name} (derivative)",
                line=dict(color=color, width=1.5, dash="dot"),
            ))

    fig.update_layout(
        title="Activation Functions",
        xaxis_title="z (input)", yaxis_title="output",
        width=700, height=450,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=0.01, y=0.99),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
    return fig


# ---------------------------------------------------------------------------
# Plotly: Loss curves
# ---------------------------------------------------------------------------


def plot_loss_curve(train_losses, val_losses=None, title="Loss Over Epochs"):
    """Line chart of training (and optionally validation) loss."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(train_losses) + 1)),
        y=train_losses,
        mode="lines", name="Training Loss",
        line=dict(color="#4A90D9", width=2),
    ))
    if val_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(val_losses) + 1)),
            y=val_losses,
            mode="lines", name="Validation Loss",
            line=dict(color="#FF6B6B", width=2),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch", yaxis_title="Loss",
        width=600, height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: Loss function visualization
# ---------------------------------------------------------------------------


def plot_loss_function_curve(loss_name, y_true_val, pred_range=(0.01, 0.99), current_pred=None):
    """Show how loss changes as predicted value varies."""
    from components.nn_engine import LOSS_FUNCTIONS

    preds = np.linspace(pred_range[0], pred_range[1], 200)
    losses = []
    grads = []
    y_true = np.array([[y_true_val]])
    for p in preds:
        y_pred = np.array([[p]])
        loss, grad = LOSS_FUNCTIONS[loss_name](y_pred, y_true)
        losses.append(loss)
        grads.append(grad[0, 0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=preds, y=losses, mode="lines",
        name="Loss", line=dict(color="#4A90D9", width=2.5),
    ))

    if current_pred is not None:
        y_pred_arr = np.array([[current_pred]])
        curr_loss, curr_grad = LOSS_FUNCTIONS[loss_name](y_pred_arr, y_true)
        fig.add_trace(go.Scatter(
            x=[current_pred], y=[curr_loss],
            mode="markers", name=f"Current (loss={curr_loss:.4f})",
            marker=dict(color="#FF6B6B", size=12, symbol="circle"),
        ))
        # Gradient arrow
        arrow_len = 0.1
        fig.add_annotation(
            x=current_pred, y=curr_loss,
            ax=current_pred - arrow_len * np.sign(curr_grad[0, 0]) if isinstance(curr_grad, np.ndarray) else current_pred - arrow_len * np.sign(curr_grad),
            ay=curr_loss,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5,
            arrowcolor="#22c55e", arrowwidth=2,
        )

    title = f"{'MSE' if loss_name == 'mse' else 'Binary Cross-Entropy'} Loss (true = {y_true_val})"
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Value", yaxis_title="Loss",
        width=600, height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: Gradient flow bar chart
# ---------------------------------------------------------------------------


def plot_gradient_flow(grads, layer_names=None):
    """Bar chart of average gradient magnitude per layer."""
    avg_grads = [np.mean(np.abs(dW)) for dW, db in grads]
    if layer_names is None:
        layer_names = [f"Layer {i+1}" for i in range(len(grads))]

    fig = go.Figure(go.Bar(
        x=layer_names, y=avg_grads,
        marker_color="#4A90D9",
    ))
    fig.update_layout(
        title="Average Gradient Magnitude per Layer",
        xaxis_title="Layer", yaxis_title="Avg |gradient|",
        width=500, height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Plotly: Optimizer comparison
# ---------------------------------------------------------------------------


def plot_optimizer_loss_curves(results_dict):
    """Overlay loss curves from multiple optimizers."""
    colors = px.colors.qualitative.Set2
    fig = go.Figure()
    for i, (name, losses) in enumerate(results_dict.items()):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(losses) + 1)),
            y=losses,
            mode="lines", name=name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.update_layout(
        title="Optimizer Comparison",
        xaxis_title="Epoch", yaxis_title="Loss",
        width=700, height=450,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib: Network architecture diagram
# ---------------------------------------------------------------------------


def draw_network_diagram(layer_sizes, title="Network Architecture", highlight_layer=None):
    """Classic neural network diagram with nodes and edges."""
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    # Use wider horizontal spacing so labels don't overlap
    x_spacing = 2.5
    total_width = (n_layers - 1) * x_spacing
    v_height = max(3, max_neurons * 0.6)
    node_radius = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 2 + n_layers * 2.2), v_height + 2))

    node_positions = {}
    layer_x_positions = [i * x_spacing for i in range(n_layers)]

    for layer_idx, n_neurons in enumerate(layer_sizes):
        x = layer_x_positions[layer_idx]
        if n_neurons > 1:
            y_positions = np.linspace(0, v_height, n_neurons)
        else:
            y_positions = [v_height / 2]

        for neuron_idx, y in enumerate(y_positions):
            node_positions[(layer_idx, neuron_idx)] = (x, y)

    # Draw edges
    for layer_idx in range(n_layers - 1):
        for src in range(layer_sizes[layer_idx]):
            for dst in range(layer_sizes[layer_idx + 1]):
                x_src, y_src = node_positions[(layer_idx, src)]
                x_dst, y_dst = node_positions[(layer_idx + 1, dst)]
                ax.plot([x_src, x_dst], [y_src, y_dst], "gray", alpha=0.3, linewidth=0.8)

    # Draw nodes
    for (layer_idx, neuron_idx), (x, y) in node_positions.items():
        if layer_idx == 0:
            color = "#4A90D9"
        elif layer_idx == n_layers - 1:
            color = "#FF6B6B"
        else:
            color = "#FFA94D" if highlight_layer == layer_idx else "#22c55e"

        circle = plt.Circle((x, y), node_radius, color=color, ec="white", linewidth=1.5, zorder=5)
        ax.add_patch(circle)

    # Labels below the diagram
    labels = ["Input"] + [f"Hidden {i+1}" for i in range(n_layers - 2)] + ["Output"]
    label_y = -0.8
    for layer_idx, label in enumerate(labels):
        x = layer_x_positions[layer_idx]
        ax.text(x, label_y, f"{label}\n({layer_sizes[layer_idx]})", ha="center", va="top",
                fontsize=9, fontweight="bold")

    ax.set_xlim(-1, total_width + 1)
    ax.set_ylim(label_y - 1.2, v_height + 0.8)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Matplotlib: Forward pass with values
# ---------------------------------------------------------------------------


def draw_forward_pass_diagram(layer_sizes, layer_values, weights_list=None, title="Forward Pass"):
    """Network diagram with actual values shown on nodes."""
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    x_spacing = 3.0
    total_width = (n_layers - 1) * x_spacing
    v_height = max(3, max_neurons * 0.7)
    node_radius = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(max(8, 3 + n_layers * 2.5), v_height + 2.5))

    node_positions = {}
    layer_x = [i * x_spacing for i in range(n_layers)]

    for li, n_neurons in enumerate(layer_sizes):
        if n_neurons > 1:
            y_positions = np.linspace(0, v_height, n_neurons)
        else:
            y_positions = [v_height / 2]
        for ni, y in enumerate(y_positions):
            node_positions[(li, ni)] = (layer_x[li], y)

    # Edges
    for li in range(n_layers - 1):
        for src in range(layer_sizes[li]):
            for dst in range(layer_sizes[li + 1]):
                xs, ys = node_positions[(li, src)]
                xd, yd = node_positions[(li + 1, dst)]
                ax.plot([xs, xd], [ys, yd], "gray", alpha=0.2, linewidth=0.7)
                if weights_list is not None and li < len(weights_list):
                    w_val = weights_list[li][src, dst]
                    mx, my = (xs + xd) / 2, (ys + yd) / 2
                    ax.text(mx, my, f"{w_val:.2f}", fontsize=6, ha="center", va="center",
                            color="#666", bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))

    # Nodes with values
    for (li, ni), (x, y) in node_positions.items():
        if li == 0:
            color = "#4A90D9"
        elif li == n_layers - 1:
            color = "#FF6B6B"
        else:
            color = "#22c55e"

        circle = plt.Circle((x, y), node_radius, color=color, ec="white", linewidth=1.5, zorder=5)
        ax.add_patch(circle)

        if li < len(layer_values) and ni < len(layer_values[li]):
            val = layer_values[li][ni]
            ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", zorder=6)

    labels = ["Input"] + [f"Hidden {i+1}" for i in range(n_layers - 2)] + ["Output"]
    label_y = -0.8
    for li, label in enumerate(labels):
        ax.text(layer_x[li], label_y, f"{label}", ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_xlim(-1, total_width + 1)
    ax.set_ylim(label_y - 1.2, v_height + 0.8)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Matplotlib: Digit image heatmap
# ---------------------------------------------------------------------------


def plot_digit_image(pixel_array, title="Digit", prediction=None, true_label=None):
    """Display an 8x8 digit image as a heatmap."""
    img = np.array(pixel_array).reshape(8, 8)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img, cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    if prediction is not None and true_label is not None:
        correct = int(prediction) == int(true_label)
        color = "#22c55e" if correct else "#FF6B6B"
        symbol = "\u2713" if correct else "\u2717"
        ax.set_title(f"{symbol} Pred: {prediction}  True: {true_label}",
                     fontsize=11, fontweight="bold", color=color)
    else:
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_digit_grid(images, predictions, true_labels, cols=4):
    """Display a grid of digit images with predictions."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    if rows == 1:
        axes = [axes]

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        if i < n:
            img = np.array(images[i]).reshape(8, 8)
            ax.imshow(img, cmap="gray_r", interpolation="nearest")
            correct = int(predictions[i]) == int(true_labels[i])
            color = "#22c55e" if correct else "#FF6B6B"
            symbol = "\u2713" if correct else "\u2717"
            ax.set_title(f"{symbol} {predictions[i]}", fontsize=10,
                         fontweight="bold", color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotly: Prediction probability bar chart
# ---------------------------------------------------------------------------


def plot_prediction_bars(probabilities, true_label=None):
    """Horizontal bar chart of class probabilities for digits 0-9."""
    digits = list(range(10))
    probs = [float(probabilities[i]) for i in digits]

    colors = []
    for i in digits:
        if true_label is not None and i == int(true_label):
            colors.append("#4A90D9")  # blue for true label
        else:
            colors.append("#d1d5db")  # gray for others

    predicted = int(np.argmax(probs))
    if true_label is not None:
        colors[predicted] = "#22c55e" if predicted == int(true_label) else "#FF6B6B"
    else:
        colors[predicted] = "#22c55e"

    fig = go.Figure(go.Bar(
        x=probs,
        y=[str(d) for d in digits],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Network Confidence",
        xaxis_title="Probability",
        yaxis_title="Digit",
        xaxis=dict(range=[0, 1.15]),
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
