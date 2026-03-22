"""Pure NumPy neural network engine.

Every function is small and mirrors the math equations students see in the app.
No PyTorch/TensorFlow — this IS the educational value.
"""

import numpy as np
import copy

# ---------------------------------------------------------------------------
# Activation functions: each returns (forward_value, derivative_value)
# ---------------------------------------------------------------------------


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh_forward(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


def linear(z):
    return z


def linear_derivative(z):
    return np.ones_like(z)


def step_function(z):
    return (z >= 0).astype(float)


def step_derivative(z):
    return np.zeros_like(z)


def softmax(z):
    """Numerically stable softmax."""
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_derivative(z):
    """Placeholder — real gradient is handled in categorical_cross_entropy_loss."""
    return np.ones_like(z)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh_forward, tanh_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "linear": (linear, linear_derivative),
    "step": (step_function, step_derivative),
    "softmax": (softmax, softmax_derivative),
}


# ---------------------------------------------------------------------------
# Loss functions: each returns (loss_value, gradient_wrt_output)
# ---------------------------------------------------------------------------


def mse_loss(y_pred, y_true):
    """Mean Squared Error."""
    n = y_true.shape[0]
    loss = np.mean((y_pred - y_true) ** 2)
    grad = 2.0 * (y_pred - y_true) / n
    return loss, grad


def binary_cross_entropy_loss(y_pred, y_true):
    """Binary Cross-Entropy. y_pred should be in (0, 1)."""
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = y_true.shape[0]
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    grad = (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / n
    return loss, grad


def categorical_cross_entropy_loss(y_pred, y_true):
    """Categorical Cross-Entropy for multi-class (softmax output).

    Expects y_true as one-hot encoded (n_samples, n_classes).
    Uses the combined softmax + cross-entropy gradient shortcut.
    """
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    n = y_true.shape[0]
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    # Combined softmax + CE gradient: (y_pred - y_true) / n
    grad = (y_pred - y_true) / n
    return loss, grad


LOSS_FUNCTIONS = {
    "mse": mse_loss,
    "binary_cross_entropy": binary_cross_entropy_loss,
    "categorical_cross_entropy": categorical_cross_entropy_loss,
}


# ---------------------------------------------------------------------------
# Dense layer
# ---------------------------------------------------------------------------


class DenseLayer:
    """A single fully-connected layer."""

    def __init__(self, n_in, n_out, activation="relu", seed=None):
        rng = np.random.RandomState(seed)
        # He initialization for ReLU, Xavier-like for others
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.weights = rng.randn(n_in, n_out) * scale
        self.biases = np.zeros((1, n_out))
        self.activation_name = activation
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]

        # Stored during forward pass for backprop
        self.input = None
        self.z = None  # pre-activation
        self.a = None  # post-activation

    def forward(self, x):
        self.input = x
        self.z = x @ self.weights + self.biases
        self.a = self.act_fn(self.z)
        return self.a

    @property
    def n_in(self):
        return self.weights.shape[0]

    @property
    def n_out(self):
        return self.weights.shape[1]


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class NeuralNetwork:
    """A sequential stack of DenseLayer objects."""

    def __init__(self, layer_sizes, activations=None, seed=42):
        """
        layer_sizes: list of ints, e.g. [2, 4, 4, 1]
                     first element is input dim, last is output dim
        activations: list of activation names, one per layer transition.
                     If None, uses 'relu' for hidden layers and 'sigmoid' for output.
        """
        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2) + ["sigmoid"]
        assert len(activations) == len(layer_sizes) - 1

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation=activations[i],
                seed=seed + i,
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        out = self.forward(x)
        if out.shape[1] == 1:
            return (out >= 0.5).astype(float)
        return np.argmax(out, axis=1)

    def get_layer_sizes(self):
        sizes = [self.layers[0].n_in]
        for layer in self.layers:
            sizes.append(layer.n_out)
        return sizes

    def count_parameters(self):
        total_w = sum(l.weights.size for l in self.layers)
        total_b = sum(l.biases.size for l in self.layers)
        return total_w, total_b

    def get_all_params(self):
        """Return flat list of (weights, biases) for each layer."""
        return [(l.weights, l.biases) for l in self.layers]

    def copy(self):
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------


def backprop(network, x, y_true, loss_fn):
    """Run forward pass then compute gradients for all layers.

    Returns:
        loss: scalar loss value
        grads: list of (dW, db) tuples, one per layer
    """
    y_pred = network.forward(x)
    loss, d_out = loss_fn(y_pred, y_true)

    grads = []
    delta = d_out
    for layer in reversed(network.layers):
        # Gradient through activation
        delta = delta * layer.act_deriv(layer.z)
        # Gradients for weights and biases
        dW = layer.input.T @ delta
        db = np.sum(delta, axis=0, keepdims=True)
        grads.append((dW, db))
        # Propagate to previous layer
        delta = delta @ layer.weights.T

    grads.reverse()
    return loss, grads


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class SGD:
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = None

    def update(self, network, grads):
        if self.velocities is None:
            self.velocities = [(np.zeros_like(dW), np.zeros_like(db)) for dW, db in grads]

        for i, (layer, (dW, db)) in enumerate(zip(network.layers, grads)):
            vW, vb = self.velocities[i]
            vW = self.momentum * vW - self.lr * dW
            vb = self.momentum * vb - self.lr * db
            self.velocities[i] = (vW, vb)
            layer.weights += vW
            layer.biases += vb


class Adam:
    """Adam optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, network, grads):
        if self.m is None:
            self.m = [(np.zeros_like(dW), np.zeros_like(db)) for dW, db in grads]
            self.v = [(np.zeros_like(dW), np.zeros_like(db)) for dW, db in grads]

        self.t += 1
        for i, (layer, (dW, db)) in enumerate(zip(network.layers, grads)):
            # Update biased first moment
            self.m[i] = (
                self.beta1 * self.m[i][0] + (1 - self.beta1) * dW,
                self.beta1 * self.m[i][1] + (1 - self.beta1) * db,
            )
            # Update biased second moment
            self.v[i] = (
                self.beta2 * self.v[i][0] + (1 - self.beta2) * dW**2,
                self.beta2 * self.v[i][1] + (1 - self.beta2) * db**2,
            )
            # Bias correction
            m_hat_W = self.m[i][0] / (1 - self.beta1**self.t)
            m_hat_b = self.m[i][1] / (1 - self.beta1**self.t)
            v_hat_W = self.v[i][0] / (1 - self.beta2**self.t)
            v_hat_b = self.v[i][1] / (1 - self.beta2**self.t)

            layer.weights -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            layer.biases -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class RMSprop:
    """RMSprop optimizer."""

    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, network, grads):
        if self.cache is None:
            self.cache = [(np.zeros_like(dW), np.zeros_like(db)) for dW, db in grads]

        for i, (layer, (dW, db)) in enumerate(zip(network.layers, grads)):
            self.cache[i] = (
                self.decay_rate * self.cache[i][0] + (1 - self.decay_rate) * dW**2,
                self.decay_rate * self.cache[i][1] + (1 - self.decay_rate) * db**2,
            )
            layer.weights -= self.lr * dW / (np.sqrt(self.cache[i][0]) + self.epsilon)
            layer.biases -= self.lr * db / (np.sqrt(self.cache[i][1]) + self.epsilon)


OPTIMIZERS = {
    "SGD": SGD,
    "SGD + Momentum": lambda lr: SGD(learning_rate=lr, momentum=0.9),
    "Adam": Adam,
    "RMSprop": RMSprop,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_step(network, x, y_true, optimizer, loss_fn, l2_lambda=0.0, dropout_rate=0.0):
    """Single training step. Returns loss value."""
    # Dropout: randomly zero out activations during forward pass
    masks = []
    if dropout_rate > 0:
        y_pred = x
        for layer in network.layers[:-1]:  # Don't dropout output layer
            y_pred = layer.forward(y_pred)
            mask = (np.random.rand(*y_pred.shape) > dropout_rate).astype(float)
            y_pred *= mask / (1 - dropout_rate)  # Inverted dropout
            layer.a = y_pred
            masks.append(mask)
        y_pred = network.layers[-1].forward(y_pred)
    else:
        y_pred = network.forward(x)

    loss, d_out = loss_fn(y_pred, y_true)

    # L2 regularization
    if l2_lambda > 0:
        for layer in network.layers:
            loss += 0.5 * l2_lambda * np.sum(layer.weights**2)

    # Backprop
    grads = []
    delta = d_out
    for idx, layer in enumerate(reversed(network.layers)):
        delta = delta * layer.act_deriv(layer.z)

        # Apply dropout mask to gradients
        if dropout_rate > 0 and idx < len(masks):
            mask_idx = len(masks) - 1 - idx
            if mask_idx >= 0 and mask_idx < len(masks):
                delta *= masks[mask_idx] / (1 - dropout_rate)

        dW = layer.input.T @ delta
        db = np.sum(delta, axis=0, keepdims=True)

        # L2 gradient
        if l2_lambda > 0:
            dW += l2_lambda * layer.weights

        grads.append((dW, db))
        delta = delta @ layer.weights.T

    grads.reverse()
    optimizer.update(network, grads)
    return loss


def train_network(
    network, x, y_true, optimizer, loss_fn,
    epochs=100, callback=None,
    l2_lambda=0.0, dropout_rate=0.0,
    x_val=None, y_val=None,
    early_stopping_patience=0,
):
    """Full training loop.

    Args:
        callback: optional function(epoch, train_loss, val_loss, network) called each epoch
        x_val, y_val: optional validation data
        early_stopping_patience: if > 0, stop when val_loss doesn't improve for this many epochs

    Returns:
        dict with 'train_losses', 'val_losses' (if validation data), 'stopped_epoch'
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    stopped_epoch = epochs

    for epoch in range(epochs):
        loss = train_step(network, x, y_true, optimizer, loss_fn,
                          l2_lambda=l2_lambda, dropout_rate=dropout_rate)
        train_losses.append(loss)

        val_loss = None
        if x_val is not None:
            y_val_pred = network.forward(x_val)
            val_loss, _ = loss_fn(y_val_pred, y_val)
            val_losses.append(val_loss)

            # Early stopping
            if early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        stopped_epoch = epoch + 1
                        if callback:
                            callback(epoch, loss, val_loss, network)
                        break

        if callback:
            callback(epoch, loss, val_loss, network)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses if val_losses else None,
        "stopped_epoch": stopped_epoch,
    }
