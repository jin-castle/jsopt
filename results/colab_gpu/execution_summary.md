# Colab GPU Execution Summary

This file records GPU outputs observed in Google Colab. The local Windows
machine used for this repository commit only has a CPU JAX backend, so GPU PNG
generation should be reproduced in Colab with the notebook in
`notebooks/fdtdx_colab_gpu_result.ipynb`.

## Environment check

```text
numpy: 2.4.6
jax: 0.7.2
devices: [CudaDevice(id=0)]
backend: gpu
```

## GPU forward smoke

```text
result: 285921.25
devices: [CudaDevice(id=0)]
backend: gpu

SimulationConfig gpu <class 'jax.numpy.float32'>
SimulationVolume
Material

steps: 5 backend: gpu
runtime_s: 1.6956870555877686
detectors: dict_keys(['energy_last'])
energy_sum: 0.000366098596714437
smoke_ok: True
```

## GPU gradient smoke

```text
devices: [CudaDevice(id=0)]
backend: gpu
steps: 8 backend: gpu
param_keys: dict_keys(['Device'])
device_param_shape: (4, 4, 4)
runtime_s: 15.842867851257324
objective: 8.334507583640516e-06
gradient_shape: (4, 4, 4)
gradient_finite: True
gradient_norm: 3.1995018616726156e-06
gradient_nonzero: True
```

## GPU optimizer visual smoke

The optimizer visual script is committed and ready for Colab GPU execution:

```python
import os

os.environ["JSOPT_BACKEND"] = "gpu"
os.environ["JSOPT_MAX_ITERS"] = "4"
os.environ["JSOPT_OUTPUT_DIR"] = "results/colab_gpu/fdtdx_optimizer_visual"

%run notebooks/fdtdx_colab_optimizer_visual.py
```

Expected PNG outputs after running the cell above:

- `results/colab_gpu/fdtdx_optimizer_visual/01_initial_design_slices.png`
- `results/colab_gpu/fdtdx_optimizer_visual/02_final_design_slices.png`
- `results/colab_gpu/fdtdx_optimizer_visual/03_final_design_delta_slices.png`
- `results/colab_gpu/fdtdx_optimizer_visual/04_best_gradient_slices.png`
- `results/colab_gpu/fdtdx_optimizer_visual/05_objective_history.png`
- `results/colab_gpu/fdtdx_optimizer_visual/06_gradient_norm_history.png`
- `results/colab_gpu/fdtdx_optimizer_visual/07_best_field_energy_slices.png`
- `results/colab_gpu/fdtdx_optimizer_visual/08_best_poynting_flux_detector_map.png`
