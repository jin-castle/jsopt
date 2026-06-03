# FDTDX Colab GPU gradient + PNG smoke.
#
# Paste this into a Colab cell after fdtdx[cuda12] is installed and
# jax.default_backend() prints "gpu".

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import fdtdx
import matplotlib.pyplot as plt
import numpy as np


print("devices:", jax.devices())
print("backend:", jax.default_backend())
if jax.default_backend() != "gpu":
    raise RuntimeError("JAX is not using the GPU backend.")


key = jax.random.PRNGKey(7)
config = fdtdx.SimulationConfig(
    time=3e-15,
    resolution=200e-9,
    backend="gpu",
    dtype=jnp.float32,
    gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=2),
)
print("steps:", config.time_steps_total, "backend:", config.backend)


volume = fdtdx.SimulationVolume(
    partial_grid_shape=(8, 8, 8),
    material=fdtdx.Material(permittivity=1.0),
)
bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
bound_dict, constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
object_list = [volume, *bound_dict.values()]


materials = {
    "air": fdtdx.Material(permittivity=1.0),
    "high": fdtdx.Material(permittivity=4.0),
}
device = fdtdx.Device(
    name="Device",
    partial_grid_shape=(4, 4, 4),
    materials=materials,
    param_transforms=[],
    partial_voxel_grid_shape=(1, 1, 1),
)
constraints.append(device.place_at_center(volume))
object_list.append(device)


source = fdtdx.GaussianPlaneSource(
    partial_grid_shape=(None, None, 1),
    partial_real_shape=(1.0e-6, 1.0e-6, None),
    fixed_E_polarization_vector=(1, 0, 0),
    wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
    radius=0.4e-6,
    std=1 / 3,
    direction="-",
)
constraints.append(
    source.place_relative_to(
        volume,
        axes=(0, 1, 2),
        own_positions=(0, 0, 0),
        other_positions=(0, 0, 0),
        grid_margins=(0, 0, -2),
    )
)
object_list.append(source)


flux = fdtdx.PoyntingFluxDetector(
    name="flux_z",
    partial_grid_shape=(None, None, 1),
    direction="-",
    reduce_volume=False,
    switch=fdtdx.OnOffSwitch(fixed_on_time_steps=[-1]),
)
constraints.append(
    flux.place_relative_to(
        volume,
        axes=(0, 1, 2),
        own_positions=(0, 0, 0),
        other_positions=(0, 0, 0),
        grid_margins=(0, 0, 2),
    )
)
object_list.append(flux)


key, subkey = jax.random.split(key)
objects, arrays0, params0, config, _ = fdtdx.place_objects(
    object_list=object_list,
    config=config,
    constraints=constraints,
    key=subkey,
)
print("param_keys:", params0.keys())
print("device_param_shape:", params0["Device"].shape)


def objective(device_params):
    params = {"Device": device_params}
    arrays, new_objects, _ = fdtdx.apply_params(
        arrays0,
        objects,
        params,
        key,
        beta=1.0,
    )
    _, out = fdtdx.run_fdtd(
        arrays=arrays,
        objects=new_objects,
        config=config,
        key=key,
        show_progress=False,
    )
    return jnp.sum(out.detector_states["flux_z"]["poynting_flux"])


start = time.time()
objective_value, gradient = jax.value_and_grad(objective)(params0["Device"])
runtime = time.time() - start

print("runtime_s:", runtime)
print("objective:", float(objective_value))
print("gradient_shape:", gradient.shape)
print("gradient_finite:", bool(jnp.all(jnp.isfinite(gradient))))
print("gradient_norm:", float(jnp.linalg.norm(gradient)))
print("gradient_nonzero:", bool(jnp.any(gradient != 0)))


def _to_numpy(arr):
    return np.asarray(jax.device_get(arr))


def _plot_center_slices(arr, title, path, cmap="viridis", symmetric=False):
    arr = np.squeeze(_to_numpy(arr))
    if arr.ndim != 3:
        raise ValueError(f"{title} needs a 3D array, got shape={arr.shape}")

    cx, cy, cz = (dim // 2 for dim in arr.shape)
    slices = [
        ("xy @ z-center", arr[:, :, cz]),
        ("xz @ y-center", arr[:, cy, :]),
        ("yz @ x-center", arr[cx, :, :]),
    ]

    vmin = vmax = None
    if symmetric:
        vmax = float(np.nanmax(np.abs(arr)))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    fig.suptitle(title)
    for ax, (label, data) in zip(axes, slices):
        im = ax.imshow(np.rot90(data), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_2d(arr, title, path, cmap="magma"):
    arr = np.squeeze(_to_numpy(arr))
    while arr.ndim > 2:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"{title} needs a 2D-compatible array, got shape={arr.shape}")

    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    im = ax.imshow(np.rot90(arr), origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.84)
    fig.savefig(path, dpi=180)
    plt.close(fig)


output_dir = Path("fdtdx_smoke_outputs")
output_dir.mkdir(exist_ok=True)

# Re-run one forward pass with the same initial parameter point to collect fields
# and detector maps for visualization. This keeps the gradient path above simple.
arrays_vis, objects_vis, _ = fdtdx.apply_params(
    arrays0,
    objects,
    {"Device": params0["Device"]},
    key,
    beta=1.0,
)
_, out_vis = fdtdx.run_fdtd(
    arrays=arrays_vis,
    objects=objects_vis,
    config=config,
    key=key,
    show_progress=False,
)

field_energy = (
    jnp.sum(jnp.abs(out_vis.fields.E) ** 2, axis=0)
    + jnp.sum(jnp.abs(out_vis.fields.H) ** 2, axis=0)
)
flux_map = out_vis.detector_states["flux_z"]["poynting_flux"]

png_paths = [
    output_dir / "01_device_parameter_slices.png",
    output_dir / "02_objective_gradient_slices.png",
    output_dir / "03_final_field_energy_slices.png",
    output_dir / "04_poynting_flux_detector_map.png",
]

_plot_center_slices(
    params0["Device"],
    "Initial Device Parameter (0=air, 1=high-index)",
    png_paths[0],
    cmap="viridis",
)
_plot_center_slices(
    gradient,
    "Objective Gradient wrt Device Parameter",
    png_paths[1],
    cmap="coolwarm",
    symmetric=True,
)
_plot_center_slices(
    field_energy,
    "Final Field Energy Density",
    png_paths[2],
    cmap="inferno",
)
_plot_2d(
    flux_map,
    "Poynting Flux Detector Map",
    png_paths[3],
    cmap="magma",
)

print("png_outputs:")
for path in png_paths:
    print(" -", path)

try:
    from IPython.display import Image, display

    for path in png_paths:
        display(Image(filename=str(path)))
except Exception as exc:
    print("PNG files were saved, but inline display failed:", repr(exc))
