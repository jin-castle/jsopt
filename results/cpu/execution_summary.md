# CPU Execution Summary

Generated locally on Windows with `JSOPT_BACKEND=cpu`.
The committed FDTDX smoke scenes use a tiny quasi-2D/2.5D slab so the examples
stay lightweight while matching the intended planar photonics workflow: x/y PML
boundaries, periodic z, and a single-z-layer design parameter.

## FDTDX gradient smoke

Command:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:JSOPT_BACKEND = 'cpu'
$env:JSOPT_OUTPUT_DIR = 'results\cpu\fdtdx_gradient_smoke'
python notebooks\fdtdx_colab_gradient_smoke.py
```

Result:

```text
devices: [CpuDevice(id=0)]
default_backend: cpu
requested_backend: cpu
steps: 8 backend: cpu
param_keys: dict_keys(['Device'])
device_param_shape: (4, 4, 1)
runtime_s: 24.84052276611328
objective: 6.0384240896382835e-06
gradient_shape: (4, 4, 1)
gradient_finite: True
gradient_norm: 2.4017538180487463e-06
gradient_nonzero: True
```

PNG outputs:

- `results/cpu/fdtdx_gradient_smoke/01_device_parameter_slices.png`
- `results/cpu/fdtdx_gradient_smoke/02_objective_gradient_slices.png`
- `results/cpu/fdtdx_gradient_smoke/03_final_field_energy_slices.png`
- `results/cpu/fdtdx_gradient_smoke/04_poynting_flux_detector_map.png`

## JSOptimizer visual smoke

Command:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:JSOPT_BACKEND = 'cpu'
$env:JSOPT_MAX_ITERS = '2'
$env:JSOPT_OUTPUT_DIR = 'results\cpu\fdtdx_optimizer_visual'
python notebooks\fdtdx_colab_optimizer_visual.py
```

Result:

```text
devices: [CpuDevice(id=0)]
default_backend: cpu
steps: 8 backend: cpu
param_shape: (4, 4, 1)
max_iters: 2 learning_rate: 0.02
gradient_scale: 1.0
runtime_s: 36.40138053894043
initial_objective: 6.0384240896382835e-06
final_evaluated_objective: 6.12574876868166e-06
best_objective: 6.12574876868166e-06
objective_improvement_from_initial: 8.732467904337682e-08
best_gradient_norm: 2.412875089123028e-06
final_design_delta_norm: 0.07043667312496735
optimization_ok: True
```

PNG outputs:

- `results/cpu/fdtdx_optimizer_visual/00_fdtdx_plot_setup.png`
- `results/cpu/fdtdx_optimizer_visual/00_fdtdx_material_slices.png`
- `results/cpu/fdtdx_optimizer_visual/01_initial_design_slices.png`
- `results/cpu/fdtdx_optimizer_visual/02_final_design_slices.png`
- `results/cpu/fdtdx_optimizer_visual/03_final_design_delta_slices.png`
- `results/cpu/fdtdx_optimizer_visual/04_best_gradient_slices.png`
- `results/cpu/fdtdx_optimizer_visual/05_objective_history.png`
- `results/cpu/fdtdx_optimizer_visual/06_gradient_norm_history.png`
- `results/cpu/fdtdx_optimizer_visual/07_best_field_energy_slices.png`
- `results/cpu/fdtdx_optimizer_visual/08_best_poynting_flux_detector_map.png`
