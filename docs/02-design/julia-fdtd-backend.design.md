# Julia FDTD Backend Detailed Design

This document defines a concrete implementation plan for a small Julia FDTD
backend that can later plug into `jsopt`.

The target is not a Meep replacement. The target is a verified, minimal,
research-grade backend that can produce:

```text
design -> objective value
design -> objective gradient
```

The backend must remain separate from the Python `jsopt` core. `jsopt` should
consume it through an objective callback.

## 1. Scope

### First backend target

- Language: Julia
- Dimension: 2D
- Polarization: TEz
- Fields: `Ez`, `Hx`, `Hy`
- Grid: uniform Cartesian Yee grid
- Material: scalar, isotropic, nondispersive `eps_r`
- Boundary: simple lossy absorber first, CPML later
- Source: point source and line source
- Monitor: DFT field monitor
- Objective: field intensity or overlap-like scalar objective
- Gradient phase 1: finite-difference gradient checker
- Gradient phase 2: hand-written discrete adjoint
- Gradient phase 3: optional ChainRules/Enzyme integration
- GPU phase: after CPU tests pass

### Non-goals for first implementation

- 3D FDTD
- dispersive materials
- anisotropic tensor materials
- nonlinear materials
- multi-GPU
- mode decomposition
- near-to-far transform
- full Meep-compatible API
- differentiating raw timestepping with an AD tape

## 2. Proposed Location

Keep this as a backend project, not inside the optimizer core.

```text
backends/
  julia_fdtd/
    Project.toml
    src/
      JSFDTD.jl
      Constants.jl
      Types.jl
      Grid.jl
      Fields.jl
      Materials.jl
      Sources.jl
      Boundaries.jl
      Monitors.jl
      Objectives.jl
      Forward.jl
      FiniteDiff.jl
      Adjoint.jl
      CallbackAPI.jl
    test/
      runtests.jl
      test_grid.jl
      test_fields.jl
      test_sources.jl
      test_forward_free_space.jl
      test_absorber.jl
      test_dft_monitor.jl
      test_objectives.jl
      test_finite_difference_gradient.jl
      test_callback_api.jl
```

## 3. Physical Model

Use 2D TEz:

```text
Ez = out-of-plane electric field
Hx = in-plane magnetic field
Hy = in-plane magnetic field
```

Continuous equations:

```text
dHx/dt = -(1/mu0) dEz/dy
dHy/dt =  (1/mu0) dEz/dx
dEz/dt =  (1/(eps0 * eps_r)) (dHy/dx - dHx/dy)
```

The design variable controls `eps_r`.

```text
rho -> mapping/filter/projection -> eps_r
```

For the Julia backend MVP, accept `eps_r` directly. Let `jsopt` own the mapping.

## 4. Array Layout

Use same-size arrays for the first CPU implementation. This keeps code simple
and readable. Interior updates enforce Yee staggering by update regions.

```text
Ez[nx, ny]
Hx[nx, ny]
Hy[nx, ny]
```

Only these slices are updated:

```text
Hx[:, 1:ny-1]
Hy[1:nx-1, :]
Ez[2:nx-1, 2:ny-1]
```

Boundary cells can be damped or left fixed depending on the boundary mode.

Later optimization can switch to staggered-size arrays:

```text
Ez[nx, ny]
Hx[nx, ny-1]
Hy[nx-1, ny]
```

The first version should prefer correctness and tests over memory optimization.

## 5. Core Types

```julia
module JSFDTD

export Grid2D, Material2D, FieldsTEz, Simulation, PointSource
export DFTMonitor, run_forward, objective_callback

include("Constants.jl")
include("Types.jl")
include("Grid.jl")
include("Fields.jl")
include("Materials.jl")
include("Sources.jl")
include("Boundaries.jl")
include("Monitors.jl")
include("Objectives.jl")
include("Forward.jl")
include("FiniteDiff.jl")
include("Adjoint.jl")
include("CallbackAPI.jl")

end
```

### Constants

```julia
const C0 = 299792458.0
const EPS0 = 8.8541878128e-12
const MU0 = 1.25663706212e-6
```

### Grid

```julia
struct Grid2D{T}
    nx::Int
    ny::Int
    dx::T
    dy::T
    dt::T
    steps::Int
end
```

Helper:

```julia
function stable_dt(dx::T, dy::T; courant::T = T(0.5)) where {T}
    return courant / (C0 * sqrt(inv(dx^2) + inv(dy^2)))
end
```

Validation:

```text
nx >= 3
ny >= 3
dx > 0
dy > 0
dt > 0
steps > 0
dt <= 1 / (c0 * sqrt(1/dx^2 + 1/dy^2))
```

### Fields

```julia
mutable struct FieldsTEz{T,A<:AbstractMatrix{T}}
    Ez::A
    Hx::A
    Hy::A
end

function zero_fields(grid::Grid2D{T}) where {T}
    Ez = zeros(T, grid.nx, grid.ny)
    Hx = zeros(T, grid.nx, grid.ny)
    Hy = zeros(T, grid.nx, grid.ny)
    return FieldsTEz(Ez, Hx, Hy)
end
```

### Material

```julia
struct Material2D{T,A<:AbstractMatrix{T}}
    eps_r::A
    sigma_e::A
end
```

For first implementation:

```text
sigma_e = zeros(nx, ny)
```

The lossy absorber can be represented by nonzero `sigma_e` near boundaries.

### Sources

```julia
abstract type AbstractSource end

struct PointSource{T,F} <: AbstractSource
    i::Int
    j::Int
    amplitude::T
    waveform::F
end

function gaussian_pulse(t0, spread, omega0)
    return t -> exp(-((t - t0) / spread)^2) * sin(omega0 * t)
end
```

Source injection:

```julia
function apply_source!(fields::FieldsTEz, source::PointSource, t)
    fields.Ez[source.i, source.j] += source.amplitude * source.waveform(t)
    return nothing
end
```

## 6. Forward Update Equations

### H update

Discrete update:

```text
Hx[i,j] = Hx[i,j] - dt/(mu0*dy) * (Ez[i,j+1] - Ez[i,j])
Hy[i,j] = Hy[i,j] + dt/(mu0*dx) * (Ez[i+1,j] - Ez[i,j])
```

Julia implementation:

```julia
function step_h!(fields::FieldsTEz{T}, grid::Grid2D{T}) where {T}
    chx = grid.dt / (MU0 * grid.dy)
    chy = grid.dt / (MU0 * grid.dx)

    @inbounds for j in 1:(grid.ny - 1), i in 1:grid.nx
        fields.Hx[i, j] -= chx * (fields.Ez[i, j + 1] - fields.Ez[i, j])
    end

    @inbounds for j in 1:grid.ny, i in 1:(grid.nx - 1)
        fields.Hy[i, j] += chy * (fields.Ez[i + 1, j] - fields.Ez[i, j])
    end

    return nothing
end
```

### E update

Without conductivity:

```text
Ez[i,j] = Ez[i,j] + dt/(eps0*eps_r[i,j]) *
          ((Hy[i,j] - Hy[i-1,j])/dx - (Hx[i,j] - Hx[i,j-1])/dy)
```

With electric conductivity:

```text
ca = (1 - sigma_e*dt/(2*eps0*eps_r)) /
     (1 + sigma_e*dt/(2*eps0*eps_r))

cb = (dt/(eps0*eps_r)) /
     (1 + sigma_e*dt/(2*eps0*eps_r))

Ez_next = ca * Ez + cb * curlH
```

Julia implementation:

```julia
function step_e!(
    fields::FieldsTEz{T},
    grid::Grid2D{T},
    material::Material2D{T},
) where {T}
    @inbounds for j in 2:(grid.ny - 1), i in 2:(grid.nx - 1)
        eps_abs = EPS0 * material.eps_r[i, j]
        sigma = material.sigma_e[i, j]
        denom = one(T) + sigma * grid.dt / (2 * eps_abs)
        ca = (one(T) - sigma * grid.dt / (2 * eps_abs)) / denom
        cb = (grid.dt / eps_abs) / denom

        curl_h = (fields.Hy[i, j] - fields.Hy[i - 1, j]) / grid.dx -
                 (fields.Hx[i, j] - fields.Hx[i, j - 1]) / grid.dy

        fields.Ez[i, j] = ca * fields.Ez[i, j] + cb * curl_h
    end

    return nothing
end
```

## 7. Boundary Strategy

### Phase 1: lossy absorber

Create `sigma_e` ramp near the boundaries.

```julia
function make_lossy_sigma(grid::Grid2D{T}; npml::Int, sigma_max::T) where {T}
    sigma = zeros(T, grid.nx, grid.ny)

    @inbounds for j in 1:grid.ny, i in 1:grid.nx
        di = min(i - 1, grid.nx - i)
        dj = min(j - 1, grid.ny - j)
        d = min(di, dj)
        if d < npml
            x = (npml - d) / npml
            sigma[i, j] = sigma_max * x^3
        end
    end

    return sigma
end
```

This is not a true CPML. It is enough for early smoke tests and gradient
plumbing.

### Phase 2: CPML

Add auxiliary fields later:

```text
psi_Ezx
psi_Ezy
psi_Hxy
psi_Hyx
```

Do not start with CPML. It increases the number of update coefficients and
gradient paths.

## 8. Monitors

Use DFT accumulation rather than storing all timesteps.

For a scalar monitor at one point:

```text
F(omega) = sum_n Ez[n] * exp(i * omega * t_n) * dt
```

For a rectangular region:

```julia
mutable struct DFTMonitor{T,A<:AbstractArray{Complex{T},3}}
    i1::Int
    i2::Int
    j1::Int
    j2::Int
    omegas::Vector{T}
    values::A
end
```

Shape:

```text
values[nx_region, ny_region, nfreq]
```

Update:

```julia
function update_monitor!(monitor::DFTMonitor{T}, fields::FieldsTEz{T}, t, dt) where {T}
    @inbounds for k in eachindex(monitor.omegas)
        phase = cis(monitor.omegas[k] * t)
        for jj in monitor.j1:monitor.j2, ii in monitor.i1:monitor.i2
            monitor.values[ii - monitor.i1 + 1, jj - monitor.j1 + 1, k] +=
                fields.Ez[ii, jj] * phase * dt
        end
    end
    return nothing
end
```

## 9. Forward Simulation API

```julia
struct Simulation{T}
    grid::Grid2D{T}
    material::Material2D{T}
    sources::Vector{AbstractSource}
    monitors::Vector{DFTMonitor{T}}
end
```

For performance, avoid `Vector{AbstractSource}` in later versions. Use concrete
typed source containers. For the first version, readability is acceptable.

Run loop:

```julia
function run_forward(sim::Simulation{T}) where {T}
    fields = zero_fields(sim.grid)

    for n in 1:sim.grid.steps
        t_h = (n - T(0.5)) * sim.grid.dt
        t_e = n * sim.grid.dt

        step_h!(fields, sim.grid)
        step_e!(fields, sim.grid, sim.material)

        for src in sim.sources
            apply_source!(fields, src, t_e)
        end

        for mon in sim.monitors
            update_monitor!(mon, fields, t_e, sim.grid.dt)
        end
    end

    return fields, sim.monitors
end
```

Important order:

```text
H update -> E update -> source injection -> monitor update
```

Keep this order fixed and tested.

## 10. Objectives

Start with simple scalar objectives.

### Point field intensity

```text
J = abs(F(omega))^2
```

```julia
function point_intensity(monitor::DFTMonitor, local_i::Int, local_j::Int, k::Int)
    z = monitor.values[local_i, local_j, k]
    return real(conj(z) * z)
end
```

### Region intensity

```text
J = sum(abs2, F_region)
```

```julia
function region_intensity(monitor::DFTMonitor, k::Int)
    total = 0.0
    @inbounds for j in axes(monitor.values, 2), i in axes(monitor.values, 1)
        z = monitor.values[i, j, k]
        total += real(conj(z) * z)
    end
    return total
end
```

### Overlap objective

Later:

```text
J = abs(sum(conj(target[i,j]) * F[i,j]))^2
```

## 11. jsopt Callback Contract

The Julia backend should expose a minimal callable interface:

```text
eps_r::Matrix{Float64} -> objective::Float64, grad_eps::Matrix{Float64}
```

For Python interop, expose a wrapper through PythonCall or a CLI JSON path.

Phase 1 callback with finite differences:

```julia
function objective_callback_fd(eps_r::Matrix{Float64}, config)
    value = run_objective(eps_r, config)
    grad = finite_difference_gradient(eps_r, x -> run_objective(x, config))
    return value, grad
end
```

Phase 2 callback with adjoint:

```julia
function objective_callback_adjoint(eps_r::Matrix{Float64}, config)
    value, tape = run_forward_with_checkpoints(eps_r, config)
    grad = run_adjoint(tape, config)
    return value, grad
end
```

## 12. Finite-Difference Gradient Checker

Directional derivative test:

```text
dJ_fd = (J(eps + h*p) - J(eps - h*p)) / (2h)
dJ_grad = dot(grad, p)
```

Implementation:

```julia
function directional_fd(value_fn, x, direction; h=1e-4)
    xp = x .+ h .* direction
    xm = x .- h .* direction
    return (value_fn(xp) - value_fn(xm)) / (2h)
end

function directional_dot(grad, direction)
    return sum(grad .* direction)
end
```

Acceptance:

```text
relative_error = abs(dJ_fd - dJ_grad) / max(1, abs(dJ_fd), abs(dJ_grad))
relative_error < 1e-3 for simple cases
relative_error < 1e-2 for lossy boundary cases
```

## 13. Discrete Adjoint Plan

Do not start by differentiating every timestep with Zygote. It will store too
much state.

The E update contains the material dependence:

```text
Ez_next = ca(eps_r) * Ez + cb(eps_r) * curlH
```

For `sigma_e = 0`:

```text
ca = 1
cb = dt / (eps0 * eps_r)
dcb/deps_r = -dt / (eps0 * eps_r^2)
```

So the direct material derivative at each timestep is:

```text
dEz_next/deps_r = dcb/deps_r * curlH
```

The adjoint should accumulate:

```text
grad_eps[i,j] += lambda_Ez_next[i,j] * dEz_next[i,j]/deps_r[i,j]
```

The full adjoint also needs reverse propagation through:

```text
Ez -> Hx, Hy updates
Hx, Hy -> Ez updates
monitor DFT accumulators -> Ez adjoint sources
```

Implementation phases:

1. Store full forward states for tiny grids only.
2. Implement reverse timestep propagation.
3. Compare adjoint gradient with finite differences on `nx <= 40`, `ny <= 40`.
4. Add checkpointing after correctness is proven.
5. Add ChainRules `rrule` only after manual adjoint passes tests.

## 14. Tests

### Grid tests

- `stable_dt` returns positive value.
- invalid grid throws error.
- CFL violation throws error.

### Field tests

- `zero_fields` returns correct shapes.
- field arrays are finite after allocation.

### Source tests

- Gaussian pulse is deterministic.
- point source changes only one `Ez` location.

### Forward free-space tests

- run with no source keeps all fields zero.
- run with point source produces finite nonzero field.
- no NaN or Inf after all steps.

### Absorber tests

- `sigma_e` shape matches grid.
- center sigma is zero.
- boundary sigma is positive.
- stronger absorber reduces late-time boundary energy.

### Monitor tests

- DFT monitor shape is correct.
- zero fields produce zero monitor.
- sinusoidal source near monitor frequency produces larger response than an
  off-frequency monitor.

### Objective tests

- intensity objective is nonnegative.
- region intensity equals sum of point intensities for a small region.
- overlap objective is zero when target is orthogonal.

### Finite-difference tests

- random direction gradient check for tiny design region.
- repeated runs are deterministic.
- perturbing design outside design region gives zero gradient if masks are used.

### Callback tests

- callback returns `(Float64, Matrix{Float64})`.
- gradient shape equals design shape.
- gradient contains no NaN or Inf.
- directional derivative matches finite difference.

## 15. Implementation Order

### Milestone A: CPU forward solver

Files:

```text
Constants.jl
Types.jl
Grid.jl
Fields.jl
Materials.jl
Sources.jl
Forward.jl
```

Success criteria:

```text
Julia tests pass
finite nonzero fields from point source
zero-source run remains zero
```

### Milestone B: absorber and monitors

Files:

```text
Boundaries.jl
Monitors.jl
Objectives.jl
```

Success criteria:

```text
DFT monitor works
basic intensity objective works
absorber damps boundary fields
```

### Milestone C: design objective and finite differences

Files:

```text
CallbackAPI.jl
FiniteDiff.jl
```

Success criteria:

```text
eps_r -> objective works
directional finite difference tests pass
Python/jsopt can call it through a stable adapter
```

### Milestone D: manual adjoint

Files:

```text
Adjoint.jl
```

Success criteria:

```text
adjoint gradient matches finite differences on small grids
gradient sign matches optimizer direction
objective improves in a toy optimization loop
```

### Milestone E: GPU kernel experiment

Only after A-D pass.

Add:

```text
KernelAbstractions.jl
CUDA.jl optional test environment
```

Success criteria:

```text
CPU/GPU forward values match within tolerance
GPU monitor values match CPU within tolerance
no GPU gradient work until CPU adjoint is stable
```

## 16. Coding Rules

- Keep all update functions mutating and suffixed with `!`.
- Keep array shapes explicit in tests.
- Do not hide field updates behind clever abstractions early.
- Prefer simple loops with `@inbounds` before GPU kernels.
- Avoid global mutable state.
- Avoid `Vector{Any}`.
- Store simulation configuration in structs.
- Write tests before adding new physics features.
- Each physics feature must include a gradient implication note.

## 17. Main Risk Register

| Risk | Mitigation |
| --- | --- |
| Fields look plausible but equations are wrong | Start with zero-source, free-space, CFL, and monitor tests. |
| Boundary reflections pollute objective | Use large enough padding early; add absorber tests before adjoint. |
| Raw AD stores all timesteps | Do not use raw AD for production gradient. Use manual adjoint. |
| Gradient sign is wrong | Always run directional derivative tests and one-step optimizer tests. |
| Julia backend consumes jsopt design mapping incorrectly | Keep callback contract in terms of `eps_r` first. |
| GPU work distracts from correctness | GPU starts only after CPU forward and adjoint pass tests. |

## 18. First Code Slice

The first real coding task should only create:

```text
Project.toml
src/JSFDTD.jl
src/Constants.jl
src/Grid.jl
src/Fields.jl
src/Materials.jl
src/Sources.jl
src/Forward.jl
test/runtests.jl
test/test_grid.jl
test/test_forward_free_space.jl
```

Do not add monitors, objectives, or adjoint in the first slice.

The first test command should be:

```bash
julia --project=backends/julia_fdtd -e "using Pkg; Pkg.test()"
```

The first accepted behavior is:

```text
1. zero-source simulation remains exactly zero
2. point-source simulation produces finite nonzero Ez
3. CFL violation throws a clear error
```

