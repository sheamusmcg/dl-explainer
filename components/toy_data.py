"""Pure NumPy toy 2D dataset generators. No sklearn dependency."""

import numpy as np


def make_moons(n_samples=300, noise=0.1, seed=42):
    """Two interleaving half-circles."""
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    theta_outer = np.linspace(0, np.pi, n)
    theta_inner = np.linspace(0, np.pi, n)

    outer_x = np.cos(theta_outer)
    outer_y = np.sin(theta_outer)
    inner_x = 1 - np.cos(theta_inner)
    inner_y = 1 - np.sin(theta_inner) - 0.5

    X = np.vstack([
        np.column_stack([outer_x, outer_y]),
        np.column_stack([inner_x, inner_y]),
    ])
    y = np.hstack([np.zeros(n), np.ones(n)])

    X += rng.randn(*X.shape) * noise
    return X, y.reshape(-1, 1)


def make_circles(n_samples=300, noise=0.05, factor=0.5, seed=42):
    """Two concentric circles."""
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    theta_outer = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_inner = np.linspace(0, 2 * np.pi, n, endpoint=False)

    outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    inner = np.column_stack([np.cos(theta_inner), np.sin(theta_inner)]) * factor

    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(n), np.ones(n)])

    X += rng.randn(*X.shape) * noise
    return X, y.reshape(-1, 1)


def make_spiral(n_samples=300, noise=0.1, seed=42):
    """Two interleaving spirals."""
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    theta = np.linspace(0, 3 * np.pi, n)
    r = np.linspace(0.3, 1.5, n)

    x1 = r * np.cos(theta) + rng.randn(n) * noise
    y1 = r * np.sin(theta) + rng.randn(n) * noise
    x2 = r * np.cos(theta + np.pi) + rng.randn(n) * noise
    y2 = r * np.sin(theta + np.pi) + rng.randn(n) * noise

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y.reshape(-1, 1)


def make_xor(n_samples=300, noise=0.1, seed=42):
    """XOR pattern: 4 clusters in quadrants, diagonally same class."""
    rng = np.random.RandomState(seed)
    n = n_samples // 4
    centers = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    labels = [0, 1, 1, 0]

    X_list, y_list = [], []
    for center, label in zip(centers, labels):
        pts = rng.randn(n, 2) * noise + np.array(center)
        X_list.append(pts)
        y_list.append(np.full(n, label))

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y.reshape(-1, 1)


def make_blobs(n_samples=300, centers=3, seed=42):
    """Simple blob clusters."""
    rng = np.random.RandomState(seed)
    n_per = n_samples // centers
    center_pts = rng.randn(centers, 2) * 2

    X_list, y_list = [], []
    for i, c in enumerate(center_pts):
        pts = rng.randn(n_per, 2) * 0.4 + c
        X_list.append(pts)
        y_list.append(np.full(n_per, i))

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y.reshape(-1, 1)


DATASETS = {
    "Moons": {
        "fn": make_moons,
        "description": "Two interleaving half-circles — no straight line can separate them.",
    },
    "Circles": {
        "fn": make_circles,
        "description": "A ring inside a ring — the network must learn a circular boundary.",
    },
    "Spiral": {
        "fn": make_spiral,
        "description": "Two interleaving spirals — very hard, needs more layers and neurons.",
    },
    "XOR": {
        "fn": make_xor,
        "description": "Diagonal clusters — a single neuron fails here, but one hidden layer solves it.",
    },
}
