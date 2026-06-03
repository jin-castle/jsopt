# Adjoint Introduction Style Gradient Check

Reference pattern: NanoComp Meep's `python/examples/adjoint_optimization/01-Introduction.ipynb`
uses an adjoint gradient, samples finite-difference gradients, fits a line, and
plots `finite difference gradient` versus `adjoint gradient`.

This jsopt example mirrors that structure with a lightweight JAX
bend-transmission surrogate so it can run quickly without requiring Meep. It
demonstrates the gradient-check workflow before applying the same pattern to
heavier FDTD objectives.

## Command

```powershell
$env:JSOPT_OUTPUT_DIR = 'results\gradient_check\adjoint_intro_style'
python examples\adjoint_intro_gradient_check.py
```

## Result

```text
seed: 240
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

- `results/gradient_check/adjoint_intro_style/01_design_and_target.png`
- `results/gradient_check/adjoint_intro_style/02_adjoint_gradient_heatmap.png`
- `results/gradient_check/adjoint_intro_style/03_finite_difference_comparison.png`
