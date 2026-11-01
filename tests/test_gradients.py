"""Cross-check analytical derivatives with symmetric finite differences."""
import numpy as np
import pytest

from interview_patterns.ml_from_scratch import mlp_init, mlp_loss_gradients, sigmoid_stable
from examples.gradient_walkthrough import central_difference


def test_linear_weight_and_bias_partials():
    x = np.array([[2., -1.], [-1., 3.], [0.5, 4.]])
    y = np.array([5., -2., 0.])
    w = np.array([0.6, -0.3])
    b = np.array([0.2])
    loss = lambda: np.mean((x @ w + b - y) ** 2)
    residual = x @ w + b - y
    np.testing.assert_allclose(2 * x.T @ residual / len(x), central_difference(loss, w), rtol=1e-7)
    np.testing.assert_allclose([2 * residual.mean()], central_difference(loss, b), rtol=1e-7)


def test_logistic_weight_and_bias_partials():
    x = np.array([[2., -1.], [0., 3.], [1., -2.]])
    y = np.array([1., 0., 1.])
    w = np.array([-0.7, 0.2])
    b = np.array([0.1])
    loss = lambda: np.mean(np.logaddexp(0, x @ w + b) - y * (x @ w + b))
    error = sigmoid_stable(x @ w + b) - y
    np.testing.assert_allclose(x.T @ error / len(x), central_difference(loss, w), rtol=1e-7)
    np.testing.assert_allclose([error.mean()], central_difference(loss, b), rtol=1e-7)


def test_every_mlp_parameter_gradient():
    x = np.array([[0.2,-0.7],[0.5,1.2],[-0.3,0.2]])
    y = np.array([0., 1., 0.])
    p = mlp_init(2,3,seed=3)
    _, exact = mlp_loss_gradients(x,y,p)
    for name in ('w1','b1','w2','b2'):
        estimated = central_difference(lambda: mlp_loss_gradients(x,y,p)[0], getattr(p,name))
        np.testing.assert_allclose(getattr(exact,name), estimated, rtol=1e-5, atol=1e-8)


def test_central_difference_restores_params_on_exception():
    w = np.array([4.])
    with pytest.raises(RuntimeError):
        central_difference(lambda: (_ for _ in ()).throw(RuntimeError('sample error')), w)
    np.testing.assert_equal(w,[4.])


def test_layernorm_input_and_affine_partials():
    # Cross-feature dependence makes LayerNorm a useful chain-rule challenge.
    x = np.array([0.2, -1.1, 2.0, 0.8], dtype=float)
    gamma = np.array([0.5, 1.1, 0.3, -0.7], dtype=float)
    beta = np.array([0.1, 0.2, -0.1, 0.0], dtype=float)
    upstream = np.array([0.3, -0.5, 0.7, 0.2], dtype=float)
    eps = 1e-5

    def loss():
        mean = x.mean()
        inv_std = 1 / np.sqrt(np.mean((x - mean)**2) + eps)
        normalized = (x - mean) * inv_std
        return float(np.dot(upstream, gamma * normalized + beta))

    mean = x.mean()
    inv_std = 1 / np.sqrt(np.mean((x - mean)**2) + eps)
    normalized = (x - mean) * inv_std
    u = upstream * gamma
    d_x = inv_std * (u - u.mean() - normalized * np.mean(u * normalized))
    d_gamma = upstream * normalized
    d_beta = upstream
    np.testing.assert_allclose(d_x, central_difference(loss, x), atol=1e-8)
    np.testing.assert_allclose(d_gamma, central_difference(loss, gamma), atol=1e-8)
    np.testing.assert_allclose(d_beta, central_difference(loss, beta), atol=1e-8)


def test_lora_adapter_partials():
    # Y = X @ (W0 + scale * B @ A).T, loss = 0.5 * sum((Y - target)^2).
    rng = np.random.default_rng(29)
    x = rng.normal(size=(3, 4))
    base = rng.normal(size=(2, 4))
    a = rng.normal(size=(2, 4))
    b = rng.normal(size=(2, 2))
    target = rng.normal(size=(3, 2))
    scale = 0.4

    def loss():
        prediction = x @ (base + scale * (b @ a)).T
        return float(0.5 * np.sum((prediction - target)**2))

    prediction = x @ (base + scale * (b @ a)).T
    g = prediction - target
    m = g.T @ x
    d_b = scale * (m @ a.T)
    d_a = scale * (b.T @ m)
    np.testing.assert_allclose(d_a, central_difference(loss, a), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(d_b, central_difference(loss, b), rtol=1e-6, atol=1e-8)
