# FDTDX Backend Adapter Design

This document updates the `jsopt` direction to treat FDTDX as the first
practical FDTD backend candidate.

`jsopt` should not become a fork of FDTDX. It should expose optimization,
mapping, continuation, history, and diagnostics around FDTDX's differentiable
FDTD engine.

## 1. Why FDTDX First

FDTDX is a stronger first backend than a hand-written Julia mini-FDTD because it
already has:

- JAX-native differentiable FDTD.
- GPU support through JAX.
- `reversible` and `checkpointed` gradient methods.
- `Device` parameter transforms for smoothing, projection, discretization, and
  STE-style discrete gradients.
- detector APIs for flux, fields, phasors, diffraction, and mode overlap.
- tests that compare reversible custom VJP gradients against checkpointed AD and
  finite differences.

The FDTDX repository also contains `.claude/skills/fdtdx/SKILL.md`, which is a
useful internal development guide. For `jsopt`, treat it as framework notes, not
as authority over this repository.

## 2. Core Boundary

Keep this boundary:

```text
jsopt core
  optimizer
  mapping/filter/projection utilities
  fabrication constraints
  beta continuation
  history/diagnostics

FDTDX adapter
  scene construction
  device parameter injection
  detector objective extraction
  JAX value-and-gradient wrapper
```

The core optimizer must still consume a simple callback:

```python
def objective_callback(design):
    return value, gradient
```

The FDTDX adapter is one implementation of that callback.

## 3. FDTDX Execution Pipeline

The adapter must follow FDTDX's canonical sequence:

```python
objects, arrays, params, config, key = fdtdx.place_objects(
    object_list=object_list,
    config=config,
    constraints=constraints,
    key=key,
)

arrays, objects, info = fdtdx.apply_params(
    arrays,
    objects,
    params,
    key,
    beta=beta,
)

time_step, arrays = fdtdx.run_fdtd(
    arrays=arrays,
    objects=objects,
    config=config,
    key=key,
)
```

Important FDTDX rules:

- Objects are immutable JAX pytrees.
- Use `.aset(...)`; do not mutate object fields in place.
- Many config fields are frozen/non-differentiable metadata.
- Field arrays use component-first shape: `(3, Nx, Ny, Nz)`.
- Materials are stored internally as inverse permittivity/permeability.
- `PoyntingFluxDetector` uses key `"poynting_flux"`.
- `ModeOverlapDetector` uses `compute_overlap_to_mode()`.
- Reversible gradients require a `Recorder`.
- JIT should donate the mutable array container where possible:
  `donate_argnames=["arrays"]`.

## 4. Adapter API

Proposed package path:

```text
jsopt/
  adapters/
    __init__.py
    fdtdx_adapter.py
```

Implemented first-pass files:

```text
jsopt/
  callbacks.py
  adapters/
    fdtdx_adapter.py
tests/
  test_callbacks.py
  test_fdtdx_adapter.py
examples/
  fdtdx_callback_skeleton.py
notebooks/
  fdtdx_colab_smoke.ipynb
  fdtdx_colab_gradient_smoke.py
```

The adapter now exposes:

```python
adapter = FDTDXObjectiveAdapter(
    scene=scene,
    objective_extractor=extract_objective,
    config=FDTDXObjectiveAdapterConfig(design_name="Device", beta=1.0),
)

value, gradient = adapter(adapter.initial_design)
```

This keeps the `jsopt` optimizer-facing contract backend-agnostic while
allowing FDTDX to own the placed objects, arrays, parameter container, config,
and JAX key.

Types:

```python
@dataclass
class FDTDXAdapterConfig:
    beta: float
    gradient_method: Literal["reversible", "checkpointed"]
    use_jit: bool = True
    donate_arrays: bool = True
    objective_name: str = "transmission"
```

```python
class FDTDXObjectiveAdapter:
    def __init__(self, scene_factory, objective_factory, config):
        ...

    def initialize(self, key):
        """Run place_objects once and cache static scene containers."""

    def value_and_grad(self, design, beta):
        """Return scalar objective and gradient w.r.t. the design."""
```

The adapter should not hide FDTDX concepts completely. It should make the common
path easy while keeping `objects`, `arrays`, `params`, `config`, and `key`
inspectable.

## 5. Two Integration Modes

### Mode A: FDTDX Device params mode

Let FDTDX own the device parameter pipeline.

```text
latent params
  -> FDTDX Device param_transforms
  -> apply_params
  -> run_fdtd
  -> objective
  -> jax.value_and_grad
```

Use when:

- the design can be expressed as FDTDX `Device` parameters.
- FDTDX transforms are enough: smoothing, projection, discretization, symmetry.
- GPU/JIT compatibility matters more than msopt-specific mapping.

Pros:

- simplest JAX gradient path.
- matches FDTDX examples.
- avoids Python-side gradient bridging.

Cons:

- does not directly use msopt mapping logic.
- FDTDX parameter transform semantics may differ from Meep/msopt filters.

### Mode B: jsopt mapping before FDTDX

Let `jsopt` map latent design to FDTDX-compatible device params or material
arrays.

```text
latent v
  -> jsopt mapping/filter/fabrication constraints
  -> FDTDX params or inv_permittivity update
  -> apply_params/run_fdtd
  -> objective
  -> gradient back through mapping
```

Use when:

- MFS/MGS, reference-layer, sidewall, or msopt-specific mapping is required.
- Meep-style mapping tests are part of the acceptance criteria.

Pros:

- preserves the original `msopt` value.
- keeps fabrication mapping first-class.

Cons:

- may require JAX-compatible rewrites of mapping functions.
- mixing `autograd.numpy` and JAX arrays will break clean JIT/grad paths.

Initial recommendation: implement Mode A first. Add Mode B only after the
optimizer and callback contract are stable.

## 6. Objective Patterns

Start with simple detector objectives.

### Transmission

```python
flux = arrays.detector_states[out_name]["poynting_flux"]
value = flux[-n_avg:, 0].mean() / reference_flux
```

Use a reference run or cached reference flux for normalization.

### Field intensity

```python
fields = arrays.detector_states[field_name]["fields"]
value = jnp.mean(fields[-n_avg:] ** 2)
```

Useful for smoke tests because it avoids mode-overlap complexity.

### Mode overlap

Use FDTDX's `ModeOverlapDetector` only after transmission and field-intensity
objectives are working.

## 7. Gradient Method Choice

Default to:

```python
GradientConfig(method="reversible", recorder=Recorder(...))
```

Use `checkpointed` as a validation oracle on small problems:

```python
GradientConfig(method="checkpointed", num_checkpoints=8)
```

For `jsopt`, the test rule should be:

```text
small scene:
  reversible gradient ~= checkpointed gradient
  selected directional derivative ~= finite difference
```

Do not trust a gradient only because it is finite.

## 8. Test Plan

### Unit tests

- adapter imports without importing FDTDX when the adapter is not used.
- missing `fdtdx` raises a clear optional-dependency error.
- callback output shape matches design shape.
- beta is forwarded to `apply_params`.

### Integration tests

- create a tiny FDTDX scene.
- call `place_objects`.
- call `apply_params`.
- call `run_fdtd`.
- extract detector objective.

### Gradient tests

- `jax.value_and_grad` returns finite value and finite gradient.
- gradient is nonzero for a driven field objective.
- reversible and checkpointed gradients agree on a tiny scene.
- central finite difference agrees with `dot(grad, direction)` for one or a few
  design variables.

### Optimization smoke test

- run 2-5 optimizer steps on a tiny scene.
- objective should move in the expected direction.
- history stores objective, gradient norm, beta, and step size.

## 9. Dependency Plan

Keep FDTDX optional:

```text
pip install jsopt[fdtdx]
```

Optional dependencies:

```text
fdtdx
jax
optax
```

The base `jsopt` package should still install without JAX/FDTDX.

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| FDTDX API changes | Pin compatible versions and keep adapter thin. |
| JAX/JIT compile cost hides test failures | Keep tiny CPU smoke tests before GPU runs. |
| FDTDX Device transforms duplicate jsopt mapping | Start with Mode A, then add Mode B with explicit tests. |
| Gradients are finite but wrong | Compare reversible/checkpointed/finite-difference on small scenes. |
| PML breaks reversible assumptions | Always configure a recorder for reversible gradients. |
| Windows GPU/JAX setup is fragile | Make FDTDX tests optional/marked and CPU-first. |

## 11. Revised Implementation Order

1. Build `jsopt` core with toy analytic callbacks.
2. Add optional FDTDX adapter scaffolding.
3. Add tiny FDTDX scene smoke test.
4. Add detector objective extraction.
5. Add `jax.value_and_grad` wrapper.
6. Add reversible-vs-checkpointed gradient validation.
7. Add finite-difference directional derivative validation.
8. Connect `JSOptimizer` to FDTDX callback.
9. Add Meep comparison examples later.
10. Keep Julia mini-FDTD as a learning/diagnostic side project, not the main
    implementation path.
