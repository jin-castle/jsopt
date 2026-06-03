"""Objective callback contracts used by jsopt optimizers and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np

ArrayLike: TypeAlias = np.ndarray


@dataclass(frozen=True)
class ValueGradient:
    """Scalar objective value and gradient with respect to a design array."""

    value: float
    gradient: np.ndarray


class ObjectiveCallback(Protocol):
    """Backend-independent objective callback.

    Implementations receive a design array and return a scalar objective plus
    the gradient with respect to that same design array.
    """

    def __call__(self, design: np.ndarray) -> tuple[float, np.ndarray] | ValueGradient:
        ...


def normalize_value_gradient(result: tuple[float, np.ndarray] | ValueGradient) -> ValueGradient:
    """Convert tuple-style callback results into a named dataclass."""

    if isinstance(result, ValueGradient):
        return result
    value, gradient = result
    return ValueGradient(float(value), np.asarray(gradient))
