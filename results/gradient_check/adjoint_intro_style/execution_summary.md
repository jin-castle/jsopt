# Adjoint Introduction Style Gradient Check

Reference pattern: NanoComp Meep's `python/examples/adjoint_optimization/01-Introduction.ipynb`
uses an adjoint gradient, samples finite-difference gradients, fits a line, and
plots `finite difference gradient` versus `adjoint gradient`.

This jsopt example mirrors that structure with a lightweight JAX
bend-transmission surrogate so it can run quickly without requiring Meep. It
demonstrates the gradient-check workflow before applying the same pattern to
heavier FDTD objectives.

## Meep waveguide bend structure checklist

Mirrored from the introduction notebook:

- Materials: Si index 3.4, SiO2 index 1.44.
- Cell: `Sx=6`, `Sy=5`.
- Boundary: `PML(1.0)`.
- Source: `EigenModeSource`, center `(-1, 0, 0)`, size `(0, 2, 0)`,
  `fcen=1/1.55`.
- Geometry: horizontal Si waveguide centered at `x=-Sx/4`, vertical Si
  waveguide centered at `y=Sy/4`.
- Design region: `1 x 1`, centered at the origin.
- Design variables: `MaterialGrid(Vector3(11, 11), SiO2, Si)`.
- Objective monitor: `EigenmodeCoefficient`, center `(0, 1, 0)`,
  size `(2, 0, 0)`.
- Objective form: `J(alpha)=abs(alpha)**2`.
- Gradient check: sample finite-difference entries and compare against the
  full gradient.

## Command

```powershell
$env:JSOPT_OUTPUT_DIR = 'results\gradient_check\adjoint_intro_style'
python examples\adjoint_intro_gradient_check.py
```

## Result

```text
seed: 240
meep_reference_structure:
 - resolution: 20
 - cell_size: (6.0, 5.0)
 - pml_thickness: 1.0
 - wavelength: 1.55
 - fcen: 0.6451612903225806
 - source_center: (-1.0, 0.0)
 - source_size: (0.0, 2.0, 0.0)
 - design_region_size: (1.0, 1.0)
 - material_grid_shape: (11, 11)
 - objective_monitor_center: (0.0, 1.0)
 - objective_monitor_size: (2.0, 0.0, 0.0)
design_shape: (11, 11)
objective: 0.25838318498002893
gradient_shape: (121,)
gradient_norm: 0.1386138675318683
fd_step: 1e-05
fd_samples: 20
fit_slope_ad_over_fd: 0.9999999999883632
fit_intercept: -2.483529989351481e-13
correlation: 0.9999999999999999
max_abs_error: 6.8232371792584234e-12
max_relative_error: 5.706610818037915e-08
mean_relative_error: 4.9529473969646876e-09
gradient_check_ok: True
```

## PNG outputs

- `results/gradient_check/adjoint_intro_style/00_meep_waveguide_bend_structure.png`
- `results/gradient_check/adjoint_intro_style/01_design_and_target.png`
- `results/gradient_check/adjoint_intro_style/02_adjoint_gradient_heatmap.png`
- `results/gradient_check/adjoint_intro_style/03_finite_difference_comparison.png`
