# Julia FDTD research notes - 2026-06-03

Scope: Julia-based FDTD / photonics simulation projects relevant to jsopt planning.

## Sources

- Khronos.jl GitHub: https://github.com/facebookresearch/Khronos.jl
- Khronos.jl roadmap: https://github.com/facebookresearch/Khronos.jl/blob/main/docs/ROADMAP.md
- Khronos.jl tests: https://github.com/facebookresearch/Khronos.jl/blob/main/test/runtests.jl
- Khronos.jl ChainRules integration: https://github.com/facebookresearch/Khronos.jl/blob/main/src/Adjoint/ChainRulesIntegration.jl
- Khronos.jl Python Meep adjoint wrapper: https://github.com/facebookresearch/Khronos.jl/blob/main/python/khronos/meep/adjoint/optimization_problem.py
- Luminescent.jl docs: https://paulxshen.github.io/Luminescent.jl/
- Luminescent.jl features: https://paulxshen.github.io/Luminescent.jl/features
- Luminescent.jl GitHub: https://github.com/paulxshen/Luminescent.jl
- FDTD.jl GitHub: https://github.com/xtalax/FDTD.jl
- FluxOptics.jl GitHub: https://github.com/anscoil/FluxOptics.jl
- FDTDX JOSS PDF: https://www.tnt.uni-hannover.de/papers/data/1813/joss.pdf

## Snapshot

- Khronos.jl is the strongest Julia FDTD reference for jsopt architecture: GPU FDTD, KernelAbstractions, MPI/NCCL, Python/Meep compatibility goals, and an adjoint module with ChainRules hooks. However, GitHub shows it archived as of 2026-05-10, and its roadmap still labels differentiability as "not yet" / Enzyme-potential, so maturity is mixed.
- Luminescent.jl is active and FDTD-oriented, but public docs state CPU-only support currently and inverse design is not available in the open-source version. Source includes Zygote/ChainRules-style adjoint/checkpointing paths, but the public optimizer modules are gated.
- FDTD.jl is an older Julia FDTD package with last activity around 2020; it is not a serious jsopt foundation.
- FluxOptics.jl is active in Julia optics but is not an FDTD adjoint simulator.
- FDTDX is Python/JAX rather than Julia, but its paper notes that Khronos.jl and FDTD-Z are GPU-capable open-source FDTD projects that are not actively maintained.

## jsopt implication

Keep jsopt simulator-agnostic. Use Julia projects as references for custom gradient interfaces, checkpointing, and test design, but do not depend on Julia FDTD as the main backend until adjoint validation, CI coverage, and public inverse-design APIs are clearly stable.

## Why Julia instead of C++ for this line of work

- The strongest rationale is not that Julia is inherently faster than C++. It is that Julia targets the "two-language problem": prototype, numerical modeling, GPU kernels, and deployment can remain closer to one codebase instead of Python/MATLAB frontends plus C++ kernels.
- For inverse design, the key bottleneck is often not only FDTD stepping speed but iteration speed: changing material mappings, filters, projections, objective functions, adjoint rules, monitors, and gradient checks. Julia's high-level syntax plus JIT specialization is attractive for this.
- Khronos.jl explicitly uses "100% Julia", KernelAbstractions, ChainRules/Zygote, and Meep-compatible Python API goals. This makes Julia a practical experiment surface for differentiable FDTD architecture.
- Khronos.jl's own roadmap is more cautious than its README: differentiability is still listed as "not yet" with Enzyme.jl potential. This means the rationale is plausible, but the implementation maturity is not yet proven.
- C++ still has the strongest case for long-term production kernels and mature low-level control, but custom adjoint APIs, C++/Python wrapping, GPU backend portability, and rapid inverse-design API iteration cost more engineering effort.
- For jsopt, Julia should inform architecture and testing, not become the required backend.

## Newer language / platform alternatives

- Mojo is newer and aims to combine Python usability with systems-level performance and MLIR-based heterogeneous hardware support. It is promising for accelerator kernels, but it is still working toward a 1.0 release in 2026 and has far less scientific-computing package maturity than Julia.
- JAX is not a new language, but it is the strongest modern differentiable-programming stack for Python. It provides NumPy-like APIs, automatic differentiation, compilation, batching, and CPU/GPU/TPU execution. For differentiable FDTD, JAX may be more mature than a new language, but raw time-domain AD still needs careful checkpointing or custom VJP design.
- Taichi is newer and is designed for high-performance differentiable physical simulation. It is attractive for GPU simulation kernels, but it is a DSL embedded in Python and has less established photonics/FDTD ecosystem support than Julia or JAX.
- Rust is newer than Julia and excellent for safe systems programming. However, its GPU and scientific-computing ecosystems are still more fragmented for this specific use case.
- Zig is useful as a cleaner C-like systems language, but it does not solve the scientific AD and inverse-design workflow problem by itself.

Conclusion: for jsopt/FDTD adjoint work, "newer" does not automatically mean better. The best candidates are Julia for a compact scientific solver, JAX for differentiable array programming, and possibly Taichi/Mojo for future GPU kernel experiments.
