# CPU Execution Summary

Generated locally on Windows with `JSOPT_BACKEND=cpu`.

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
device_param_shape: (4, 4, 4)
runtime_s: 8.1306312084198
objective: 8.33450940262992e-06
gradient_shape: (4, 4, 4)
gradient_finite: True
gradient_norm: 3.1995018616726156e-06
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
param_shape: (4, 4, 4)
max_iters: 2 learning_rate: 0.02
gradient_scale: -1.0
runtime_s: 9.862805604934692
initial_objective: 8.33450940262992e-06
final_evaluated_objective: 8.516633897670545e-06
best_objective: 8.516633897670545e-06
objective_improvement_from_initial: 1.8212449504062533e-07
best_gradient_norm: 3.259879232278238e-06
final_design_delta_norm: 0.08619950947796765
optimization_ok: True
```

PNG outputs:

- `results/cpu/fdtdx_optimizer_visual/01_initial_design_slices.png`
- `results/cpu/fdtdx_optimizer_visual/02_final_design_slices.png`
- `results/cpu/fdtdx_optimizer_visual/03_final_design_delta_slices.png`
- `results/cpu/fdtdx_optimizer_visual/04_best_gradient_slices.png`
- `results/cpu/fdtdx_optimizer_visual/05_objective_history.png`
- `results/cpu/fdtdx_optimizer_visual/06_gradient_norm_history.png`
- `results/cpu/fdtdx_optimizer_visual/07_best_field_energy_slices.png`
- `results/cpu/fdtdx_optimizer_visual/08_best_poynting_flux_detector_map.png`
