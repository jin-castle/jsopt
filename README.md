# jsopt

`jsopt` is a Python optimization toolkit for photonic inverse design.

The first implementation path keeps the optimizer backend-agnostic and uses
FDTDX as an optional differentiable FDTD backend through a callback adapter.
Lumerical adjoint code is intentionally excluded until that path is separately
validated.

Current implemented surface:

- `JSOptimizer`: Adam-like backend-agnostic optimizer for callbacks returning
  `(value, gradient)`.
- `FDTDXObjectiveAdapter`: wraps a placed FDTDX scene as a jsopt objective
  callback.
- Colab smoke scripts for FDTDX GPU forward and gradient checks.
- Colab optimizer visual smoke script that saves PNG outputs for initial/final
  design slices, design delta, gradients, objective history, field energy, and
  detector flux maps.
