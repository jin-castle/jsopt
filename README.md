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
- Colab smoke scripts for FDTDX GPU forward and gradient checks using a tiny
  1-cell PML boundary.
- Colab optimizer visual smoke script that saves FDTDX native setup/material
  plots plus PNG outputs for initial/final design slices, design delta,
  gradients, objective history, field energy, and detector flux maps.

Committed execution artifacts:

- `results/cpu/`: local CPU smoke outputs, PNGs, and run summary.
- `results/colab_gpu/`: recorded Colab GPU logs and reproduction notes.
- `results/gradient_check/`: Meep-introduction-style gradient and finite
  difference comparison artifacts, including a waveguide-bend structure diagram.
- `notebooks/fdtdx_cpu_result.ipynb`: CPU result notebook with committed PNGs.
- `notebooks/fdtdx_colab_gpu_result.ipynb`: Colab GPU result/reproduction
  notebook.
- `notebooks/adjoint_intro_gradient_check.ipynb`: gradient output and finite
  difference agreement notebook inspired by Meep's adjoint introduction.

## Colab install

```python
%pip install -U "fdtdx[cuda12]"
%pip install -U git+https://github.com/jin-castle/jsopt.git
```

After restarting the runtime, verify imports:

```python
from jsopt import JSOptimizer, OptimizerConfig
from jsopt.adapters import FDTDXObjectiveAdapter

print("jsopt import ok")
```
