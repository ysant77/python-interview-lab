"""Independent, educational NumPy implementations for ML coding interviews.

Focus: training objectives, gradients, broadcasting, numerical stability, and
realistic edge cases. These routines are for study, not production estimators.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a nonempty N by D array")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    return x


def _target(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) != n or not np.isfinite(y).all():
        raise ValueError("target length must match samples and be finite")
    return y


def _bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1))], axis=1)


@dataclass
class LinearRegression:
    weights: np.ndarray
    bias: float
    losses: list[float]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _features(x) @ self.weights + self.bias


def linear_regression_gd(
    x: np.ndarray, y: np.ndarray, *, lr: float = 0.05, steps: int = 1000
) -> LinearRegression:
    """Batch gradient descent on mean squared error.

    T: O(steps*N*D), S: O(N*D + D) including input and temporary predictions.
    For unstable feature scales, standardize x first or choose a smaller step.
    """
    x = _features(x); y = _target(y, len(x))
    if lr <= 0 or steps < 0:
        raise ValueError("lr must be positive and steps nonnegative")
    w = np.zeros(x.shape[1]); b = 0.0; losses = []
    for _ in range(steps):
        residual = x @ w + b - y
        losses.append(float(np.mean(residual**2)))
        w -= lr * (2 / len(x)) * (x.T @ residual)
        b -= lr * 2 * float(np.mean(residual))
    return LinearRegression(w, b, losses)


def linear_regression_lstsq(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Least squares via SVD-based solver; handles rank-deficient inputs.

    Typical dense SVD cost O(N*D^2 + D^3) for N >= D, with D+1 features;
    do not explicitly invert X.T @ X because it may be singular or unstable.
    """
    x = _features(x); y = _target(y, len(x))
    theta, *_ = np.linalg.lstsq(_bias(x), y, rcond=None)
    return LinearRegression(theta[:-1], float(theta[-1]), [])


def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Sigmoid without overflowing exp for large negative logits; O(n)/O(n)."""
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    result[~positive] = ez / (1 + ez)
    return result


@dataclass
class LogisticRegression:
    weights: np.ndarray
    bias: float
    losses: list[float]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid_stable(_features(x) @ self.weights + self.bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


def logistic_regression_gd(
    x: np.ndarray, y: np.ndarray, *, lr: float = 0.1, steps: int = 1000
) -> LogisticRegression:
    """Binary cross-entropy gradients with logaddexp stability.

    T: O(steps*N*D), S: O(N+D) additional; uses mean loss.
    """
    x = _features(x); y = _target(y, len(x))
    if not np.isin(y, [0, 1]).all():
        raise ValueError("binary labels must be 0 or 1")
    if lr <= 0 or steps < 0:
        raise ValueError("lr must be positive and steps nonnegative")
    w = np.zeros(x.shape[1]); b = 0.0; losses = []
    for _ in range(steps):
        logits = x @ w + b
        losses.append(float(np.mean(np.logaddexp(0, logits) - y * logits)))
        err = sigmoid_stable(logits) - y
        w -= lr * (x.T @ err) / len(x)
        b -= lr * float(np.mean(err))
    return LogisticRegression(w, b, losses)


@dataclass
class KMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    iterations: int
    inertia: float


def kmeans(
    x: np.ndarray, n_clusters: int, *, max_iter: int = 100, seed: int = 0, tol: float = 1e-5
) -> KMeansResult:
    """Lloyd iterations with seeded random initialization and empty-cluster recovery.

    T: O(iterations*N*K*D), S: O(N*K*D) temporary plus O(N*K + K*D) additional; local optimum only.
    """
    x = _features(x)
    n, dim = x.shape
    if not (1 <= n_clusters <= n) or max_iter < 1 or tol < 0:
        raise ValueError("invalid clusters, iteration count or tolerance")
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=n_clusters, replace=False)].copy()
    for iteration in range(1, max_iter + 1):
        squared = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(squared, axis=1)
        next_centers = centers.copy()
        for k in range(n_clusters):
            if np.any(labels == k):
                next_centers[k] = x[labels == k].mean(axis=0)
            else:
                # Use the currently worst-represented sample rather than NaN.
                next_centers[k] = x[int(np.argmax(squared.min(axis=1)))]
        movement = float(np.max(np.linalg.norm(next_centers - centers, axis=1)))
        centers = next_centers
        if movement <= tol:
            break
    squared = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(squared, axis=1)
    return KMeansResult(centers, labels, iteration, float(np.min(squared, axis=1).sum()))


def knn_predict(
    train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, k: int = 3
) -> np.ndarray:
    """Euclidean k-NN classification with deterministic label tie-breaking.

    T: O(Q*N*D + Q*N) typical; S: O(Q*N), full pairwise distance matrix.
    Prefer batching for large data and a spatial index for low dimensional data.
    """
    train_x = _features(train_x); query_x = _features(query_x)
    if train_x.shape[1] != query_x.shape[1] or not 1 <= k <= len(train_x):
        raise ValueError("dimension mismatch or invalid k")
    labels = np.asarray(train_y)
    if labels.ndim != 1 or len(labels) != len(train_x):
        raise ValueError("train_y must be a 1D matching label array")
    sq = np.maximum(
        (query_x**2).sum(axis=1, keepdims=True)
        + (train_x**2).sum(axis=1)[None, :] - 2 * query_x @ train_x.T, 0
    )
    nearest = np.argpartition(sq, k - 1, axis=1)[:, :k]
    predictions = []
    for neighbors in nearest:
        choices, counts = np.unique(labels[neighbors], return_counts=True)
        predictions.append(choices[int(np.argmax(counts))])
    return np.asarray(predictions)


def layer_norm_numpy(x: np.ndarray, gamma: np.ndarray | None = None,
                     beta: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization over last axis, not batch axis; O(elements)/O(elements)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] == 0 or eps <= 0:
        raise ValueError("invalid x or eps")
    gamma = np.ones(x.shape[-1]) if gamma is None else np.asarray(gamma)
    beta = np.zeros(x.shape[-1]) if beta is None else np.asarray(beta)
    if gamma.shape != (x.shape[-1],) or beta.shape != (x.shape[-1],):
        raise ValueError("gamma and beta must match last dimension")
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean)**2).mean(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


@dataclass
class MLPParameters:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def mlp_init(n_features: int, n_hidden: int, seed: int = 0) -> MLPParameters:
    if n_features < 1 or n_hidden < 1:
        raise ValueError("invalid layer dimensions")
    rng = np.random.default_rng(seed)
    return MLPParameters(rng.normal(0, np.sqrt(2/n_features), (n_features,n_hidden)),
                         np.zeros(n_hidden), rng.normal(0, np.sqrt(1/n_hidden), (n_hidden,1)),
                         np.zeros(1))


def mlp_forward(x: np.ndarray, p: MLPParameters) -> tuple[np.ndarray, tuple]:
    """2-layer network: affine -> tanh -> affine -> sigmoid, O(N*D*H)."""
    x = _features(x)
    z1 = x @ p.w1 + p.b1
    hidden = np.tanh(z1)
    logits = hidden @ p.w2 + p.b2
    proba = sigmoid_stable(logits)
    return proba.reshape(-1), (x, hidden, logits)


def mlp_loss_gradients(x: np.ndarray, y: np.ndarray, p: MLPParameters) -> tuple[float, MLPParameters]:
    """Backpropagation through a two-layer MLP with binary cross entropy.

    T: O(N*D*H), S: O(N*H + D*H) additional; returns mean loss + gradients.
    """
    pred, (x, hidden, logits) = mlp_forward(x, p)
    y = _target(y, len(x))
    if not np.isin(y, [0,1]).all():
        raise ValueError("binary target required")
    loss = float(np.mean(np.logaddexp(0, logits[:, 0]) - y * logits[:, 0]))
    dz2 = (pred - y)[:, None] / len(x)
    dw2 = hidden.T @ dz2
    db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ p.w2.T) * (1 - hidden**2)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, MLPParameters(dw1, db1, dw2, db2)


def mlp_train(x: np.ndarray, y: np.ndarray, *, hidden: int = 12, lr: float = 0.3,
              steps: int = 1000, seed: int = 0) -> tuple[MLPParameters,list[float]]:
    x = _features(x); y = _target(y, len(x))
    if lr <= 0 or steps < 0:
        raise ValueError("invalid learning rate or steps")
    p = mlp_init(x.shape[1], hidden, seed)
    losses=[]
    for _ in range(steps):
        loss, grad = mlp_loss_gradients(x, y, p)
        losses.append(loss)
        p = MLPParameters(p.w1-lr*grad.w1, p.b1-lr*grad.b1,
                          p.w2-lr*grad.w2, p.b2-lr*grad.b2)
    return p, losses
