"""Optional simulation backend adapters."""

from jsopt.adapters.fdtdx_adapter import (
    FDTDXObjectiveAdapter,
    FDTDXObjectiveAdapterConfig,
    FDTDXScene,
    OptionalDependencyError,
)

__all__ = [
    "FDTDXObjectiveAdapter",
    "FDTDXObjectiveAdapterConfig",
    "FDTDXScene",
    "OptionalDependencyError",
]
