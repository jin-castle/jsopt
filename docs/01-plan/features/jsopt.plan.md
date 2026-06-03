# jsopt Plan

## Overview

`jsopt` is a Python optimization toolkit inspired by `msopt`, focused on reusable topology-optimization utilities rather than Lumerical adjoint internals.

The current implementation target is not JavaScript. The name `jsopt` is treated as the project/package name.

## Source References

- `msopt/Opt_MS2.py`: optimizer loop, multi-objective gradient aggregation, Adam-like momentum, backtracking, beta continuation, warm restart, binarization tracking, overlap FoMs.
- `msopt/Sub_Mapping.py`: fabrication-aware mapping, MFS/MGS enforcement, symmetry transforms, 1D/2D reference-layer mapping, vertical/slanted sidewall expansion.
- `msopt/Filters.py`: conic/cylindrical/Gaussian filters, tanh projection, morphology filters, length-scale constraints. This file is itself forked from Meep adjoint filters, so `jsopt` should treat Meep-compatible filter behavior as the stable baseline.
- `NanoComp/meep/python/adjoint`: reference for adjoint-compatible mapping/filter design and gradient backpropagation patterns.
- `NanoComp/meep/python/tests/test_adjoint_solver.py`: reference for finite-difference checks of adjoint gradients, objective-function variants, and gradient backpropagation through mappings.
- `NanoComp/meep/python/tests/test_adjoint_jax.py`: reference for wrapper-level value-and-gradient checks against finite differences.
- `NanoComp/meep/python/tests/test_adjoint_cyl.py`: reference for cylindrical/Near2Far-style geometry tests and finite-difference comparison patterns.
- `NanoComp/meep/python/tests/test_adjoint_utils.py`: reference for utility/filter tests where explicit gradient calculation is not always required.
- `ymahlau/fdtdx`: first practical FDTD backend candidate for JAX-native differentiable FDTD.
- `ymahlau/fdtdx/.claude/skills/fdtdx/SKILL.md`: reference for FDTDX internal invariants: immutable PyTree objects, `.aset()` updates, canonical simulation pipeline, Yee-grid conventions, gradient strategies, detector state shapes, and test patterns.
- `ymahlau/fdtdx/tests/simulation/fdtd`: reference for reversible-vs-checkpointed gradient validation and finite-difference checks.

## Explicit Exclusions

- Do not implement Lumerical adjoint.
- Do not depend on `lumapi`.
- Do not port `msopt/Lumerical_utill.py` adjoint source generation, imported-source logic, field-monitor handling, or custom Lumerical gradient scaling.
- Do not assume the Lumerical gradient path is correct until it is separately validated.
- Do not make `jsopt` a JavaScript package.

## Goals

1. Build a clean Python package named `jsopt`.
2. Recreate the useful non-Lumerical parts of `msopt` with clearer boundaries.
3. Keep the optimizer backend-agnostic: it should consume an objective callback returning FoM values and design gradients.
4. Support FDTDX as the first practical FDTD backend through an optional adapter.
5. Support Meep-adjoint-style workflows through callbacks or adapters without making Meep a hard dependency for the core package.
6. Provide focused tests for filters, mapping, gradient aggregation, optimizer state transitions, and finite-difference-compatible callback behavior.

## In Scope

- Filter utilities:
  - conic filter
  - cylindrical filter
  - Gaussian filter
  - tanh projection
  - optional morphology helpers
  - length-scale helper functions
- Fabrication mapping:
  - 1D reference-layer mapping
  - 2D reference-layer mapping
  - width/length/C2/quadrant/pseudo-cylindrical symmetry
  - air/waveguide masks
  - MFS/MGS enforcement through dilation and erosion
  - vertical sidewall expansion
  - slanted sidewall expansion as an advanced mapping path
- Optimizer:
  - minimax gradient aggregation
  - goal-attainment-style gradient aggregation
  - Adam-like momentum
  - Born-validity gradient pruning
  - learning-rate backtracking
  - beta continuation
  - warm restart
  - binarization and history tracking
- Objectives:
  - overlap integral FoM
  - power/flux helpers
  - phase-aware overlap variant
- Interfaces:
  - `ObjectiveCallback` protocol
  - `Mapping` protocol
  - optimizer result/history dataclasses
- Optional FDTDX adapter:
  - lazy import and optional dependency error handling
  - small scene factory hook
  - detector objective extraction
  - `jax.value_and_grad` callback wrapper
  - reversible/checkpointed gradient validation helpers for tiny scenes

## Out of Scope for First Version

- Writing a custom FDTD solver.
- Lumerical session management.
- Lumerical monitor/grid alignment.
- Lumerical imported nk grid updates.
- Lumerical adjoint dipole/source construction.
- Full Meep simulation builder.
- Full FDTDX scene authoring framework.
- GUI or web UI.
- MPI/GPU/process monitoring utilities from `msopt`.

## Proposed Package Layout

```text
jsopt/
  __init__.py
  filters.py
  projections.py
  constraints.py
  mapping.py
  optimizer.py
  objectives.py
  callbacks.py
  history.py
  diagnostics.py
  adapters/
    __init__.py
    fdtdx_adapter.py
tests/
  test_filters.py
  test_mapping.py
  test_gradient_checks.py
  test_optimizer.py
  test_objectives.py
  test_diagnostics.py
  test_fdtdx_adapter.py
examples/
  toy_quadratic.py
  meep_callback_skeleton.py
  fdtdx_callback_skeleton.py
docs/
  01-plan/features/jsopt.plan.md
  02-design/fdtdx-backend-adapter.design.md
```

## Core API Sketch

```python
def objective_callback(design: np.ndarray) -> tuple[float, np.ndarray]:
    """Return scalar FoM and gradient with respect to mapped design."""
```

```python
optimizer = JSOptimizer(
    initial_design=v0,
    mapping=mapping,
    config=OptimizerConfig(
        initial_lr=0.2,
        initial_beta=1.0,
        born_top_k=50,
        max_iters=777,
    ),
)

result = optimizer.optimize(objective_callback)
```

The optimizer owns updates in latent design space `v`. The callback evaluates the mapped physical design `x = mapping(v, beta)`. Gradients from the callback are backpropagated through the mapping before updating `v`.

## Architecture Decisions

### Keep Optimization Separate From Simulation

The optimizer should not know whether gradients came from Meep, finite differences, an analytic toy model, or a future validated Lumerical backend. This avoids importing incomplete Lumerical adjoint assumptions into `jsopt`.

FDTDX is the preferred first simulation backend, but only through an adapter. The core optimizer should still only see an objective callback.

### Use Autograd-Compatible Math Initially

`msopt` uses `autograd.numpy` and `tensor_jacobian_product`. The first `jsopt` version should preserve that style for mapping backpropagation because it closely matches the source behavior.

### Convert Stateful Arrays Into Named Dataclasses

`Opt_MS2.OPT_Ms` stores state in positional arrays such as `Array`, `Parameters`, `Best`, and `Outer_M`. `jsopt` should replace these with named dataclasses:

- `OptimizerState`
- `BestState`
- `ContinuationState`
- `OptimizerHistory`

### Treat Fabrication Mapping As First-Class

The strongest reusable part of `msopt` is not the Lumerical path; it is the mapping layer. MFS/MGS, symmetry, masks, and sidewall expansion should be implemented as explicit configurable transforms.

### Prefer FDTDX Before Writing a Solver

FDTDX already provides JAX-native differentiable FDTD, GPU execution, reversible/checkpointed gradient paths, detector APIs, and test patterns. A custom Julia mini-FDTD can remain a learning or diagnostic side project, but the practical backend path should start with FDTDX.

## Implementation Phases

### Phase 1: Foundation

- Create package metadata and test setup.
- Add `numpy`, `scipy`, `autograd`, and `pytest` dependencies.
- Add dataclasses and public type protocols.
- Add package import smoke test.

### Phase 2: Filters and Projections

- Port conic/cylindrical/Gaussian filters.
- Port tanh projection and modified tanh projection.
- Add shape validation and deterministic errors.
- Test filter output shape, value range, and basic symmetry.
- Add Meep-style utility tests for filters that should be differentiable, but do not require a full adjoint simulation.

### Phase 3: Fabrication Mapping

- Implement `ReferenceLayer2DConfig`.
- Implement MFS/MGS dilation-erosion mapping.
- Implement symmetry and mask operations.
- Implement 1D reference-layer mapping.
- Add vertical sidewall expansion.
- Keep slanted sidewall as advanced but testable.
- Add Meep-style mapping backpropagation tests using `tensor_jacobian_product`.

### Phase 4: Objective Helpers

- Implement overlap integral.
- Implement phase-aware overlap.
- Implement flux helper.
- Test complex-valued arrays and normalization behavior.
- Add finite-difference checks for objective helper gradients using analytic toy fields.

### Phase 5: Optimizer

- Implement minimax and goal-attainment gradient aggregation.
- Implement Adam-like momentum.
- Implement Born-validity pruning.
- Implement backtracking as a small explicit state machine.
- Implement beta continuation and warm restart.
- Replace `msopt` print-heavy flow with structured history records.
- Add optimizer tests that compare callback-provided gradients with finite-difference directional derivatives before running update loops.

Current status:

- A first-pass `JSOptimizer` is implemented in `jsopt/optimizer.py`.
- The optimizer consumes the backend-independent `ObjectiveCallback` contract.
- Implemented behavior:
  - maximize/minimize objective sense
  - Adam-like first/second moment updates
  - optional `[low, high]` design clipping
  - optional gradient norm clipping
  - step tolerance stop condition
  - structured `OptimizerHistoryEntry`
  - `OptimizerResult` returning the best evaluated design
- Tests currently cover toy quadratic maximization, toy quadratic minimization, bound clipping, gradient shape validation, and non-finite gradient rejection.
- Next optimizer work: add mapping backpropagation, finite-difference directional derivative checks, minimax/goal-attainment gradient aggregation, beta continuation, and backtracking.

### Phase 6: Diagnostics

- Add finite-difference directional derivative checker for toy callbacks.
- Add gradient norm and binarization diagnostics.
- Add result serialization helpers.
- Add a reusable Meep-inspired assertion helper:
  `assert_directional_derivative_matches(value_fn, grad, x, direction, step, tolerance)`.

### Phase 7: Examples

- Add analytic toy quadratic example.
- Add toy multi-objective minimax example.
- Add Meep callback skeleton showing how a Meep adjoint problem can be plugged in without coupling core code to Meep.
- Add FDTDX callback skeleton showing the adapter path:
  `place_objects -> apply_params -> run_fdtd -> detector objective -> jax.value_and_grad`.

### Phase 8: FDTDX Adapter

- Add optional dependency handling for `fdtdx`, `jax`, and `optax`.
- Add an adapter that accepts a scene factory and objective extractor.
- Keep FDTDX objects, arrays, params, config, and key inspectable.
- Add a tiny CPU-first FDTDX smoke test if the optional dependency is installed.
- Add gradient tests modelled after FDTDX:
  - finite value and finite gradients
  - nonzero gradient for driven field objectives
  - reversible-vs-checkpointed agreement on tiny scenes
  - central finite-difference check for selected variables

Current status:

- Colab GPU forward smoke passed with `jax.default_backend() == "gpu"`.
- Colab GPU gradient smoke passed for `Device` parameters with finite/nonzero gradients.
- A first-pass `FDTDXObjectiveAdapter` is implemented under `jsopt/adapters/fdtdx_adapter.py`.
- The adapter test uses a tiny CPU scene and verifies finite nonzero gradients through a `PoyntingFluxDetector` objective.
- Next adapter work: add JIT/donation benchmarking, reversible-vs-checkpointed comparison, and selected-variable central finite-difference checks.

## Test Strategy

- Unit tests for each filter and projection.
- Mapping tests for value range, shape, symmetry, mask enforcement, and binarization trend as beta increases.
- Mapping-gradient tests modelled after Meep `test_gradient_backpropagation`: compute a mapped design, backpropagate a physical-design gradient through the mapping with `tensor_jacobian_product`, and compare the directional derivative against finite differences.
- Objective-gradient tests modelled after Meep DFT/eigenmode/LDOS tests, but using analytic toy objectives instead of FDTD: compare `dot(grad, perturbation)` with `(f(x + dx) - f(x)) / step`.
- Callback-wrapper tests modelled after Meep JAX wrapper tests: verify that the public callback contract returns value and gradient with consistent shapes and finite-difference agreement.
- FDTDX adapter tests modelled after FDTDX `test_fdtd.py` and `test_time_reversal.py`: compare reversible and checkpointed gradients on tiny scenes and add central finite-difference checks for selected design variables.
- Optimizer tests using analytic functions where the expected optimum is known.
- Multi-objective tests where the minimax aggregator only uses underperforming objectives.
- Regression tests for beta continuation and warm restart state transitions.
- Smoke test for callback API.

## Meep-Inspired Test Matrix

| Meep reference | jsopt equivalent | Purpose |
| --- | --- | --- |
| `test_adjoint_solver.py` finite-difference gradient tests | `test_gradient_checks.py` | Validate `dot(gradient, perturbation)` against finite differences. |
| `test_adjoint_solver.py` mapping backpropagation test | `test_mapping.py` | Validate gradient propagation through filter/projection/mapping transforms. |
| `test_adjoint_solver.py` multi-objective and multi-frequency gradient tests | `test_optimizer.py` | Validate objective aggregation such as minimax and goal attainment. |
| `test_adjoint_jax.py` wrapper value-and-gradient checks | `test_callbacks.py` | Validate callback interface shape, dtype, finite values, and gradient consistency. |
| `test_adjoint_utils.py` filter utility tests | `test_filters.py` | Validate filters/projections without requiring a simulator. |
| `test_adjoint_cyl.py` cylindrical geometry tests | `test_mapping.py` | Validate radial/cylindrical mapping paths if included in first implementation. |
 
The Meep tests should be used as verification patterns, not copied wholesale. `jsopt` tests should avoid full electromagnetic simulation and instead use analytic toy objectives unless a future Meep adapter test is explicitly added.

## FDTDX-Inspired Test Matrix

| FDTDX reference | jsopt equivalent | Purpose |
| --- | --- | --- |
| `.claude/skills/fdtdx/SKILL.md` pipeline notes | `test_fdtdx_adapter.py` | Enforce adapter call order and `.aset()`/PyTree expectations. |
| `tests/integration/objects/test_device.py` | `test_fdtdx_adapter.py` | Validate Device parameter transforms remain differentiable. |
| `tests/simulation/fdtd/test_fdtd.py` | `test_fdtdx_adapter.py` | Validate finite/nonzero gradients and reversible-vs-checkpointed agreement. |
| `tests/simulation/fdtd/test_time_reversal.py` | `test_gradient_checks.py` | Validate selected gradients against central finite differences. |

## Success Criteria

- `pytest` passes locally.
- Package imports cleanly with `import jsopt`.
- Optimizer can improve a toy analytic objective.
- Finite-difference directional derivative tests pass for objective callbacks and mapping backpropagation.
- Mapping functions keep values in `[0, 1]`.
- Binarization metric increases as beta increases for representative mappings.
- No `lumapi` import exists in the package.
- No Lumerical adjoint source/monitor/gradient code exists in the package.

## Risks and Mitigations

- Risk: Directly copying `Opt_MS2` would preserve positional state and hidden behavior.
  - Mitigation: Rebuild as named dataclasses and small functions.
- Risk: Slanted sidewall mapping is complex and easy to mistranscribe.
  - Mitigation: Implement after 2D and vertical mapping are tested.
- Risk: Autograd may be limiting for future JAX workflows.
  - Mitigation: Keep math functions pure and isolate AD-specific calls.
- Risk: Optimizer behavior depends on many heuristics.
  - Mitigation: Start with toy objectives and explicit history inspection.

## Definition of Done for Plan Phase

- Scope excludes Lumerical adjoint explicitly.
- Reusable `msopt` components are identified.
- Module layout is defined.
- Implementation phases are ordered.
- Test strategy is defined.
