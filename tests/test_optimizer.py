import numpy as np
import pytest

from jsopt import JSOptimizer, OptimizerConfig


def test_optimizer_maximizes_toy_quadratic():
    target = np.array([0.2, -0.4, 0.7])

    def objective(design):
        delta = design - target
        return -float(np.sum(delta**2)), -2.0 * delta

    optimizer = JSOptimizer(
        initial_design=np.zeros_like(target),
        config=OptimizerConfig(
            max_iters=160,
            learning_rate=0.04,
            objective_sense="maximize",
            bounds=None,
        ),
    )

    result = optimizer.optimize(objective)

    assert result.value > objective(np.zeros_like(target))[0]
    assert np.linalg.norm(result.design - target) < 0.05
    assert result.iterations == 160
    assert result.history[-1].best_value == result.best_value


def test_optimizer_minimizes_toy_quadratic():
    target = np.array([0.75, 0.25])

    def objective(design):
        delta = design - target
        return float(np.sum(delta**2)), 2.0 * delta

    optimizer = JSOptimizer(
        initial_design=np.array([0.0, 1.0]),
        config=OptimizerConfig(
            max_iters=120,
            learning_rate=0.04,
            objective_sense="minimize",
            bounds=(0.0, 1.0),
        ),
    )

    result = optimizer.optimize(objective)

    assert result.value < objective(np.array([0.0, 1.0]))[0]
    assert np.linalg.norm(result.design - target) < 0.05


def test_optimizer_clips_design_to_bounds():
    def objective(design):
        delta = design - 2.0
        return -float(np.sum(delta**2)), -2.0 * delta

    optimizer = JSOptimizer(
        initial_design=np.array([0.5, 0.5]),
        config=OptimizerConfig(
            max_iters=80,
            learning_rate=0.1,
            objective_sense="maximize",
            bounds=(0.0, 1.0),
        ),
    )

    result = optimizer.optimize(objective)

    assert np.all(result.design <= 1.0)
    assert np.all(result.design >= 0.0)
    assert np.allclose(result.design, np.ones(2))


def test_optimizer_rejects_wrong_gradient_shape():
    def objective(_design):
        return 1.0, np.ones(3)

    optimizer = JSOptimizer(
        initial_design=np.ones(2),
        config=OptimizerConfig(max_iters=1),
    )

    with pytest.raises(ValueError, match="gradient shape"):
        optimizer.optimize(objective)


def test_optimizer_rejects_nonfinite_gradient():
    def objective(design):
        return 1.0, np.full_like(design, np.nan)

    optimizer = JSOptimizer(
        initial_design=np.ones(2),
        config=OptimizerConfig(max_iters=1),
    )

    with pytest.raises(ValueError, match="non-finite"):
        optimizer.optimize(objective)
