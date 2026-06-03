"""Small backend-free jsopt optimizer example."""

import numpy as np

from jsopt import JSOptimizer, OptimizerConfig


target = np.array([0.2, 0.8, 0.5])


def objective(design: np.ndarray) -> tuple[float, np.ndarray]:
    delta = design - target
    value = -float(np.sum(delta**2))
    gradient = -2.0 * delta
    return value, gradient


if __name__ == "__main__":
    optimizer = JSOptimizer(
        initial_design=np.zeros_like(target),
        config=OptimizerConfig(
            max_iters=120,
            learning_rate=0.04,
            objective_sense="maximize",
            bounds=(0.0, 1.0),
        ),
    )
    result = optimizer.optimize(objective)
    print("best_value:", result.best_value)
    print("best_design:", np.array2string(result.best_design, precision=4))
    print("target:", np.array2string(target, precision=4))
