# FDTDX Claude skill research notes - 2026-06-03

Scope: Inspect `ymahlau/fdtdx`, especially `.claude/skills/fdtdx/SKILL.md`, for jsopt backend planning.

## Sources

- FDTDX repository: https://github.com/ymahlau/fdtdx
- FDTDX Claude skill: https://github.com/ymahlau/fdtdx/blob/main/.claude/skills/fdtdx/SKILL.md
- FDTDX skill directory: https://github.com/ymahlau/fdtdx/tree/main/.claude/skills/fdtdx
- FDTDX docs: https://fdtdx.readthedocs.io/en/latest/
- JOSS paper: https://joss.theoj.org/papers/10.21105/joss.08912.pdf
- Local inspection clone: `C:\Users\bgs43\AppData\Local\Temp\jsopt-research-fdtdx`
- Inspected commit: `c174997d2b801addc3028f1efeb6e2b5302f10dd`

## Findings

- The repository explicitly uses Claude guidance: `CLAUDE.md` says it provides guidance to Claude Code and points to `.claude/skills/fdtdx/SKILL.md` for detailed framework patterns.
- The skill file is not just generic documentation. It captures important FDTDX internal invariants: immutable TreeClass/PyTree objects, `.aset()` updates, simulation pipeline, Yee grid conventions, normalized H fields, inverse material storage, gradient strategies, device transforms, detector state shapes, and testing patterns.
- Code inspection agrees with the skill:
  - `GradientConfig` supports `reversible` and `checkpointed` gradient methods.
  - `run_fdtd` dispatches to `reversible_fdtd` or `checkpointed_fdtd`.
  - `reversible_fdtd` uses `jax.custom_vjp`.
  - `Device` owns a parameter transformation pipeline, and examples use `GaussianSmoothing2D`, `SubpixelSmoothedProjection`, beta schedules, `jax.value_and_grad`, and `optax`.
  - Tests include finite-gradient checks, reversible-vs-checkpointed cross-checks, and central finite-difference gradient comparisons.

## jsopt implication

Use FDTDX as the first practical FDTD backend candidate. Keep jsopt core backend-agnostic, but add a first-class FDTDX adapter plan:

```text
latent design v
  -> jsopt mapping/filter/projection where needed
  -> FDTDX Device params or inv_permittivity arrays
  -> fdtdx.apply_params(...)
  -> fdtdx.run_fdtd(...)
  -> detector-based objective
  -> jax.value_and_grad(...)
  -> jsopt optimizer update
```

Do not fork or rewrite FDTDX initially. Treat FDTDX as an external backend and validate the adapter through small, deterministic smoke tests before attempting production-scale inverse design.
