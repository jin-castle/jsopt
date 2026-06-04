# FDTDX Colab GPU optimizer + PNG visual smoke.
#
# Paste this into a Colab cell after fdtdx[cuda12] is installed and
# jax.default_backend() prints "gpu".

import os
import time
from pathlib import Path

import fdtdx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jsopt import JSOptimizer, OptimizerConfig
from jsopt.adapters import FDTDXObjectiveAdapter, FDTDXObjectiveAdapterConfig, FDTDXScene


def build_scene(backend):
    key = jax.random.PRNGKey(7)
    config = fdtdx.SimulationConfig(
        time=3e-15,
        resolution=200e-9,
        backend=backend,
        dtype=jnp.float32,
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=2),
    )

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
    objects, arrays, params, config, _info = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    return FDTDXScene(objects=objects, arrays=arrays, params=params, config=config, key=key)


def extract_flux_objective(arrays, _objects):
    return jnp.sum(arrays.detector_states["flux_z"]["poynting_flux"])


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


def _plot_objective_history(history, path):
    iterations = [entry.iteration for entry in history]
    values = [entry.value for entry in history]
    best_values = [entry.best_value for entry in history]

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(iterations, values, marker="o", label="evaluated objective")
    ax.plot(iterations, best_values, marker="s", label="best objective")
    ax.set_title("JSOptimizer Objective History")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_gradient_norm_history(history, path):
    iterations = [entry.iteration for entry in history]
    norms = [entry.gradient_norm for entry in history]

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(iterations, norms, marker="o", color="tab:red")
    ax.set_title("Gradient Norm History")
    ax.set_xlabel("iteration")
    ax.set_ylabel("||gradient||")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_forward(scene, design, beta):
    params = dict(scene.params)
    params["Device"] = jnp.asarray(design)
    arrays, objects, _info = fdtdx.apply_params(
        scene.arrays,
        scene.objects,
        params,
        scene.key,
        beta=beta,
    )
    _time_step, arrays = fdtdx.run_fdtd(
        arrays=arrays,
        objects=objects,
        config=scene.config,
        key=scene.key,
        show_progress=False,
    )
    return arrays


def _save_fdtdx_scene_plots(scene, design, beta, setup_path, material_path):
    setup_fig = fdtdx.plot_setup(
        scene.config,
        scene.objects,
        filename=setup_path,
    )
    params = dict(scene.params)
    params["Device"] = jnp.asarray(design)
    material_arrays, _material_objects, _info = fdtdx.apply_params(
        scene.arrays,
        scene.objects,
        params,
        scene.key,
        beta=beta,
    )
    material_fig = fdtdx.plot_material(
        scene.config,
        material_arrays,
        filename=material_path,
    )
    plt.close(setup_fig)
    plt.close(material_fig)


def _scaled_gradient_callback(adapter, gradient_scale):
    def callback(design):
        value, gradient = adapter(design)
        return value, gradient_scale * gradient

    return callback


def _select_backend():
    requested_backend = os.environ.get("JSOPT_BACKEND", "gpu").lower()
    print("devices:", jax.devices())
    print("default_backend:", jax.default_backend())
    if requested_backend == "gpu" and jax.default_backend() != "gpu":
        raise RuntimeError(
            "JAX is not using the GPU backend. In Colab, choose Runtime > Change runtime type > GPU, "
            "then reinstall fdtdx[cuda12] and restart the runtime."
        )
    return requested_backend


def main():
    backend = _select_backend()
    max_iters = int(os.environ.get("JSOPT_MAX_ITERS", "4"))
    learning_rate = float(os.environ.get("JSOPT_LR", "0.02"))
    # The tiny "-" flux-detector smoke has an adjoint sign opposite to the
    # finite-difference flux direction, so the visual demo defaults to -1.
    gradient_scale = float(os.environ.get("JSOPT_GRADIENT_SCALE", "-1.0"))
    output_dir = Path(os.environ.get("JSOPT_OUTPUT_DIR", "fdtdx_optimizer_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = build_scene(backend)
    adapter_config = FDTDXObjectiveAdapterConfig(design_name="Device", beta=1.0)
    adapter = FDTDXObjectiveAdapter(
        scene=scene,
        objective_extractor=extract_flux_objective,
        config=adapter_config,
    )

    initial_design = adapter.initial_design
    optimizer = JSOptimizer(
        initial_design,
        OptimizerConfig(
            max_iters=max_iters,
            learning_rate=learning_rate,
            objective_sense="maximize",
            bounds=(0.0, 1.0),
            gradient_clip_norm=0.25,
        ),
    )

    print("steps:", scene.config.time_steps_total, "backend:", scene.config.backend)
    print("param_shape:", initial_design.shape)
    print("max_iters:", max_iters, "learning_rate:", learning_rate)
    print("gradient_scale:", gradient_scale)

    start = time.time()
    result = optimizer.optimize(_scaled_gradient_callback(adapter, gradient_scale))
    runtime_s = time.time() - start

    final_design = result.state.design
    design_delta = final_design - initial_design
    best_arrays = _run_forward(scene, result.best_design, beta=adapter_config.beta)
    field_energy = (
        jnp.sum(jnp.abs(best_arrays.fields.E) ** 2, axis=0)
        + jnp.sum(jnp.abs(best_arrays.fields.H) ** 2, axis=0)
    )
    flux_map = best_arrays.detector_states["flux_z"]["poynting_flux"]

    png_paths = [
        output_dir / "00_fdtdx_plot_setup.png",
        output_dir / "00_fdtdx_material_slices.png",
        output_dir / "01_initial_design_slices.png",
        output_dir / "02_final_design_slices.png",
        output_dir / "03_final_design_delta_slices.png",
        output_dir / "04_best_gradient_slices.png",
        output_dir / "05_objective_history.png",
        output_dir / "06_gradient_norm_history.png",
        output_dir / "07_best_field_energy_slices.png",
        output_dir / "08_best_poynting_flux_detector_map.png",
    ]

    _save_fdtdx_scene_plots(scene, initial_design, adapter_config.beta, png_paths[0], png_paths[1])
    _plot_center_slices(
        initial_design,
        "Initial Device Parameter (0=air, 1=high-index)",
        png_paths[2],
        cmap="viridis",
    )
    _plot_center_slices(
        final_design,
        "Final Evaluated Device Parameter",
        png_paths[3],
        cmap="viridis",
    )
    _plot_center_slices(
        design_delta,
        "Final Device Parameter Delta",
        png_paths[4],
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_center_slices(
        result.gradient,
        "Best Optimizer Gradient wrt Device Parameter",
        png_paths[5],
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_objective_history(result.history, png_paths[6])
    _plot_gradient_norm_history(result.history, png_paths[7])
    _plot_center_slices(
        field_energy,
        "Best Final Field Energy Density",
        png_paths[8],
        cmap="inferno",
    )
    _plot_2d(
        flux_map,
        "Best Poynting Flux Detector Map",
        png_paths[9],
        cmap="magma",
    )

    initial_value = result.history[0].value
    final_value = result.history[-1].value
    print("runtime_s:", runtime_s)
    print("initial_objective:", initial_value)
    print("final_evaluated_objective:", final_value)
    print("best_objective:", result.best_value)
    print("objective_improvement_from_initial:", result.best_value - initial_value)
    print("best_gradient_norm:", float(np.linalg.norm(result.gradient)))
    print("final_design_delta_norm:", float(np.linalg.norm(design_delta)))
    print("optimization_ok:", bool(np.isfinite(result.best_value) and np.all(np.isfinite(result.gradient))))
    print("png_outputs:")
    for path in png_paths:
        print(" -", path)

    try:
        from IPython.display import Image, display

        for path in png_paths:
            display(Image(filename=str(path)))
    except Exception as exc:
        print("PNG files were saved, but inline display failed:", repr(exc))


if __name__ == "__main__":
    main()
