"""Backend-agnostic optimizer core for jsopt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from jsopt.callbacks import ObjectiveCallback, normalize_value_gradient

ObjectiveSense = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the first-pass jsopt optimizer."""

    max_iters: int = 100
    learning_rate: float = 0.05
    objective_sense: ObjectiveSense = "maximize"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    bounds: tuple[float, float] | None = (0.0, 1.0)
    gradient_clip_norm: float | None = None
    step_tolerance: float | None = None


@dataclass(frozen=True)
class OptimizerState:
    """Mutable optimization state captured as an immutable snapshot."""

    iteration: int
    design: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray


@dataclass(frozen=True)
class OptimizerHistoryEntry:
    """Per-evaluation optimizer diagnostics."""

    iteration: int
    value: float
    gradient_norm: float
    step_norm: float
    design_min: float
    design_max: float
    design_mean: float
    best_value: float


@dataclass(frozen=True)
class OptimizerResult:
    """Optimization result and history."""

    design: np.ndarray
    value: float
    gradient: np.ndarray
    best_design: np.ndarray
    best_value: float
    history: tuple[OptimizerHistoryEntry, ...]
    state: OptimizerState

    @property
    def iterations(self) -> int:
        return len(self.history)


class JSOptimizer:
    """Adam-like optimizer for callbacks returning value and design gradient."""

    def __init__(
        self,
        initial_design: np.ndarray,
        config: OptimizerConfig | None = None,
    ) -> None:
        self.config = config or OptimizerConfig()
        _validate_config(self.config)

        design = np.asarray(initial_design, dtype=float)
        if design.size == 0:
            raise ValueError("initial_design must not be empty")
        if not np.all(np.isfinite(design)):
            raise ValueError("initial_design contains non-finite values")
        design = _clip_to_bounds(design, self.config.bounds)

        zeros = np.zeros_like(design, dtype=float)
        self.initial_state = OptimizerState(
            iteration=0,
            design=design.copy(),
            first_moment=zeros.copy(),
            second_moment=zeros.copy(),
        )

    def optimize(self, objective_callback: ObjectiveCallback) -> OptimizerResult:
        """Run optimization and return the best evaluated design."""

        state = self.initial_state
        history: list[OptimizerHistoryEntry] = []
        best_design: np.ndarray | None = None
        best_gradient: np.ndarray | None = None
        best_value: float | None = None

        for iteration in range(self.config.max_iters):
            callback_result = normalize_value_gradient(objective_callback(state.design.copy()))
            gradient = np.asarray(callback_result.gradient, dtype=float)
            value = float(callback_result.value)
            _validate_callback_output(value, gradient, state.design.shape)

            if best_value is None or _is_better(value, best_value, self.config.objective_sense):
                best_value = value
                best_design = state.design.copy()
                best_gradient = gradient.copy()

            if iteration == self.config.max_iters - 1:
                step = np.zeros_like(state.design)
                next_state = state
            else:
                step, next_state = self._next_state(state, gradient, iteration + 1)

            step_norm = float(np.linalg.norm(step))
            history.append(
                OptimizerHistoryEntry(
                    iteration=iteration,
                    value=value,
                    gradient_norm=float(np.linalg.norm(gradient)),
                    step_norm=step_norm,
                    design_min=float(np.min(state.design)),
                    design_max=float(np.max(state.design)),
                    design_mean=float(np.mean(state.design)),
                    best_value=float(best_value),
                )
            )

            state = next_state
            if self.config.step_tolerance is not None and step_norm <= self.config.step_tolerance:
                break

        assert best_design is not None
        assert best_gradient is not None
        assert best_value is not None
        return OptimizerResult(
            design=best_design.copy(),
            value=float(best_value),
            gradient=best_gradient.copy(),
            best_design=best_design.copy(),
            best_value=float(best_value),
            history=tuple(history),
            state=state,
        )

    def _next_state(
        self,
        state: OptimizerState,
        gradient: np.ndarray,
        step_index: int,
    ) -> tuple[np.ndarray, OptimizerState]:
        gradient = _clip_gradient(gradient, self.config.gradient_clip_norm)
        signed_gradient = gradient if self.config.objective_sense == "maximize" else -gradient

        first_moment = self.config.beta1 * state.first_moment + (1.0 - self.config.beta1) * signed_gradient
        second_moment = self.config.beta2 * state.second_moment + (1.0 - self.config.beta2) * np.square(
            signed_gradient
        )

        first_hat = first_moment / (1.0 - self.config.beta1**step_index)
        second_hat = second_moment / (1.0 - self.config.beta2**step_index)
        raw_step = self.config.learning_rate * first_hat / (np.sqrt(second_hat) + self.config.epsilon)
        next_design = _clip_to_bounds(state.design + raw_step, self.config.bounds)
        actual_step = next_design - state.design

        return actual_step, OptimizerState(
            iteration=state.iteration + 1,
            design=next_design,
            first_moment=first_moment,
            second_moment=second_moment,
        )


def _validate_config(config: OptimizerConfig) -> None:
    if config.max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.objective_sense not in ("maximize", "minimize"):
        raise ValueError("objective_sense must be 'maximize' or 'minimize'")
    if not 0 <= config.beta1 < 1:
        raise ValueError("beta1 must be in [0, 1)")
    if not 0 <= config.beta2 < 1:
        raise ValueError("beta2 must be in [0, 1)")
    if config.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when provided")
    if config.step_tolerance is not None and config.step_tolerance < 0:
        raise ValueError("step_tolerance must be non-negative when provided")
    if config.bounds is not None:
        low, high = config.bounds
        if low >= high:
            raise ValueError("bounds must satisfy low < high")


def _validate_callback_output(value: float, gradient: np.ndarray, design_shape: tuple[int, ...]) -> None:
    if not np.isfinite(value):
        raise ValueError("objective callback returned a non-finite value")
    if gradient.shape != design_shape:
        raise ValueError(f"gradient shape {gradient.shape} does not match design shape {design_shape}")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("objective callback returned a gradient with non-finite values")


def _clip_gradient(gradient: np.ndarray, max_norm: float | None) -> np.ndarray:
    if max_norm is None:
        return gradient
    norm = float(np.linalg.norm(gradient))
    if norm <= max_norm or norm == 0:
        return gradient
    return gradient * (max_norm / norm)


def _clip_to_bounds(design: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    if bounds is None:
        return design.copy()
    low, high = bounds
    return np.clip(design, low, high)


def _is_better(value: float, incumbent: float, sense: ObjectiveSense) -> bool:
    if sense == "maximize":
        return value > incumbent
    return value < incumbent
