"""Reproducible numerical checks for introductory machine-learning derivatives.

Run after `python -m pip install -e '.[test]'`:
    python examples/gradient_walkthrough.py
"""
from __future__ import annotations

import numpy as np

from interview_patterns.ml_from_scratch import (
    mlp_init,
    mlp_loss_gradients,
    sigmoid_stable,
)


def central_difference(loss_fn, params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Numerically estimate one partial derivative per parameter.

    Complexity: O(P * C_loss) time for P scalars and loss-evaluation cost C_loss;
    O(P) returned gradient space; input array is temporarily perturbed and restored.
    """
    gradient = np.empty_like(params, dtype=np.float64)
    for index in np.ndindex(params.shape):
        old = float(params[index])
        try:
            params[index] = old + eps
            plus = float(loss_fn())
            params[index] = old - eps
            minus = float(loss_fn())
        finally:
            params[index] = old
        gradient[index] = (plus - minus) / (2 * eps)
    return gradient


def linear_example() -> None:
    x = np.array([[2.0]])
    y = np.array([5.0])
    w = np.array([1.0])
    b = np.array([0.0])

    def loss():
        return float(np.mean((x @ w + b - y) ** 2))

    residual = x @ w + b - y
    d_w = (2 / len(x)) * (x.T @ residual)
    d_b = np.array([2 * float(np.mean(residual))])
    print('LINEAR: y_hat=', x @ w + b, 'residual=', residual, 'loss=', loss())
    print('dL/dw=', d_w, 'numeric=', central_difference(loss, w))
    print('dL/db=', d_b, 'numeric=', central_difference(loss, b))
    np.testing.assert_allclose(d_w, central_difference(loss, w), rtol=1e-7)
    np.testing.assert_allclose(d_b, central_difference(loss, b), rtol=1e-7)


def logistic_example() -> None:
    x = np.array([[2.0]])
    y = np.array([1.0])
    w = np.array([0.0])
    b = np.array([0.0])

    def loss():
        z = x @ w + b
        return float(np.mean(np.logaddexp(0, z) - y * z))

    logits = x @ w + b
    p = sigmoid_stable(logits)
    d_w = x.T @ (p - y) / len(x)
    d_b = np.array([float(np.mean(p - y))])
    print('LOGISTIC: p=', p, 'loss=', loss())
    print('dL/dw=', d_w, 'numeric=', central_difference(loss, w))
    print('dL/db=', d_b, 'numeric=', central_difference(loss, b))
    np.testing.assert_allclose(d_w, central_difference(loss, w), rtol=1e-7)
    np.testing.assert_allclose(d_b, central_difference(loss, b), rtol=1e-7)


def mlp_example() -> None:
    x = np.array([[0.2, -0.7], [0.5, 1.2], [-0.3, 0.2]], dtype=float)
    y = np.array([0.0, 1.0, 0.0])
    p = mlp_init(2, 3, seed=3)
    loss, analytical = mlp_loss_gradients(x, y, p)
    print('MLP: mean binary cross entropy=', loss)
    for name in ('w1', 'b1', 'w2', 'b2'):
        value = getattr(p, name)
        expected = getattr(analytical, name)
        observed = central_difference(lambda: mlp_loss_gradients(x, y, p)[0], value)
        largest_error = float(np.max(np.abs(expected - observed)))
        print(f'  dL/d{name}: shape={expected.shape}, max abs error={largest_error:.3e}')
        np.testing.assert_allclose(expected, observed, rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    linear_example()
    logistic_example()
    mlp_example()
