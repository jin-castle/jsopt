"""Minimal FDTDX scene wrapped as a jsopt objective callback.

Run on Colab after installing:

    %pip install -U "fdtdx[cuda12]"

For local CPU smoke, install `fdtdx` and change `backend` to "cpu".
"""

import jax
import jax.numpy as jnp

import fdtdx
from jsopt.adapters import FDTDXObjectiveAdapter, FDTDXObjectiveAdapterConfig, FDTDXScene


def build_scene(backend: str = "gpu") -> FDTDXScene:
    key = jax.random.PRNGKey(7)
    config = fdtdx.SimulationConfig(
        time=3e-15,
        resolution=200e-9,
        backend=backend,
        dtype=jnp.float32,
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=2),
    )

    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(8, 8, 3),
        material=fdtdx.Material(permittivity=1.0),
    )
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=1,
        boundary_type="pml",
        override_types={"min_z": "periodic", "max_z": "periodic"},
    )
    bound_dict, constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    object_list = [volume, *bound_dict.values()]

    device = fdtdx.Device(
        name="Device",
        partial_grid_shape=(4, 4, 1),
        materials={
            "air": fdtdx.Material(permittivity=1.0),
            "high": fdtdx.Material(permittivity=4.0),
        },
        param_transforms=[],
        partial_voxel_grid_shape=(1, 1, 1),
    )
    constraints.append(device.place_at_center(volume))
    object_list.append(device)

    source = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, 1.0e-6, None),
        fixed_E_polarization_vector=(0, 0, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
        radius=0.4e-6,
        std=1 / 3,
        direction="+",
    )
    constraints.append(
        source.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, 0),
            grid_margins=(-2, 0, 0),
        )
    )
    object_list.append(source)

    flux = fdtdx.PoyntingFluxDetector(
        name="flux_y",
        partial_grid_shape=(None, 1, None),
        direction="+",
        reduce_volume=False,
        fixed_propagation_axis=1,
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=[-1]),
    )
    constraints.append(
        flux.place_relative_to(
            volume,
            axes=(0, 1, 2),
            own_positions=(0, 0, 0),
            other_positions=(0, 0, 0),
            grid_margins=(0, 2, 0),
        )
    )
    object_list.append(flux)

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    return FDTDXScene(objects=objects, arrays=arrays, params=params, config=config, key=key)


def extract_flux_objective(arrays, _objects):
    return jnp.sum(arrays.detector_states["flux_y"]["poynting_flux"])


if __name__ == "__main__":
    adapter = FDTDXObjectiveAdapter(
        scene=build_scene(),
        objective_extractor=extract_flux_objective,
        config=FDTDXObjectiveAdapterConfig(design_name="Device", beta=1.0),
    )
    result = adapter.value_and_gradient(adapter.initial_design)
    print("objective:", result.value)
    print("gradient_shape:", result.gradient.shape)
    print("gradient_norm:", float(jnp.linalg.norm(jnp.asarray(result.gradient))))
