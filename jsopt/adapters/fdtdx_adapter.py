"""FDTDX objective callback adapter.

The adapter converts an FDTDX scene into the simple jsopt callback contract:

    design -> (objective_value, objective_gradient)

It intentionally does not build a full scene-authoring layer. The caller owns
scene construction and objective extraction; this module owns the repeated
`apply_params -> run_fdtd -> value_and_grad` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Mapping

import numpy as np

from jsopt.callbacks import ValueGradient


class OptionalDependencyError(ImportError):
    """Raised when an optional simulation backend dependency is unavailable."""


@dataclass(frozen=True)
class FDTDXScene:
    """Placed FDTDX scene state.

    Attributes are intentionally typed as `Any` because they are FDTDX/JAX
    pytrees and should stay owned by FDTDX.
    """

    objects: Any
    arrays: Any
    params: Mapping[str, Any]
    config: Any
    key: Any


@dataclass(frozen=True)
class FDTDXObjectiveAdapterConfig:
    """Runtime options for FDTDX callback evaluation."""

    design_name: str = "Device"
    beta: float = 1.0
    use_jit: bool = False
    run_fdtd_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"show_progress": False})
    transform_kwargs: Mapping[str, Any] = field(default_factory=dict)


ObjectiveExtractor = Callable[[Any, Any], Any]
SceneFactory = Callable[[], FDTDXScene]


class FDTDXObjectiveAdapter:
    """Wrap a placed FDTDX scene as a jsopt objective callback."""

    def __init__(
        self,
        scene: FDTDXScene,
        objective_extractor: ObjectiveExtractor,
        config: FDTDXObjectiveAdapterConfig | None = None,
    ) -> None:
        self._fdtdx = _import_optional("fdtdx")
        self._jax = _import_optional("jax")
        self._jnp = _import_optional("jax.numpy")
        self.scene = scene
        self.objective_extractor = objective_extractor
        self.config = config or FDTDXObjectiveAdapterConfig()
        self._compiled_value_and_grad: Callable[[Any], Any] | None = None

        if self.config.design_name not in self.scene.params:
            raise KeyError(
                f"Design parameter {self.config.design_name!r} not found in scene params: "
                f"{tuple(self.scene.params.keys())!r}"
            )

    @classmethod
    def from_factory(
        cls,
        scene_factory: SceneFactory,
        objective_extractor: ObjectiveExtractor,
        config: FDTDXObjectiveAdapterConfig | None = None,
    ) -> "FDTDXObjectiveAdapter":
        """Build the adapter from a user-supplied scene factory."""

        return cls(scene_factory(), objective_extractor, config=config)

    @property
    def initial_design(self) -> np.ndarray:
        """Initial design array copied from the FDTDX parameter container."""

        return np.asarray(self._jax.device_get(self.scene.params[self.config.design_name]))

    def __call__(self, design: np.ndarray) -> tuple[float, np.ndarray]:
        """Return a tuple-compatible objective callback result."""

        result = self.value_and_gradient(design)
        return result.value, result.gradient

    def value_and_gradient(self, design: np.ndarray) -> ValueGradient:
        """Evaluate the objective and gradient for a design array."""

        design_jax = self._jnp.asarray(design)
        value_and_grad = self._value_and_grad_fn()
        value, gradient = value_and_grad(design_jax)
        return ValueGradient(
            value=float(self._jax.device_get(value)),
            gradient=np.asarray(self._jax.device_get(gradient)),
        )

    def evaluate(self, design: np.ndarray) -> float:
        """Evaluate only the scalar objective value."""

        value = self._objective(self._jnp.asarray(design))
        return float(self._jax.device_get(value))

    def _value_and_grad_fn(self) -> Callable[[Any], Any]:
        if not self.config.use_jit:
            return self._jax.value_and_grad(self._objective)
        if self._compiled_value_and_grad is None:
            self._compiled_value_and_grad = self._jax.jit(self._jax.value_and_grad(self._objective))
        return self._compiled_value_and_grad

    def _objective(self, design: Any) -> Any:
        params = dict(self.scene.params)
        params[self.config.design_name] = design

        transform_kwargs = {
            **self.config.transform_kwargs,
            "beta": self.config.beta,
        }
        arrays, objects, _info = self._fdtdx.apply_params(
            self.scene.arrays,
            self.scene.objects,
            params,
            self.scene.key,
            **transform_kwargs,
        )
        _time_step, arrays = self._fdtdx.run_fdtd(
            arrays=arrays,
            objects=objects,
            config=self.scene.config,
            key=self.scene.key,
            **self.config.run_fdtd_kwargs,
        )
        return self.objective_extractor(arrays, objects)


def _import_optional(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"Optional dependency {module_name!r} is required for the FDTDX adapter. "
            "Install jsopt with the fdtdx extra or install fdtdx separately."
        ) from exc
