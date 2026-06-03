"""Meep-introduction-style gradient and finite-difference check.

This mirrors the structure of NanoComp/meep's adjoint introduction notebook:

1. define a small design region,
2. evaluate an objective and its gradient,
3. visualize the gradient,
4. compare sampled gradient entries with central finite differences.

The objective here is a compact JAX surrogate for a bend-transmission objective,
so the example is lightweight enough to run locally or in Colab without Meep.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


SEED = 240
NX = 11
NY = 11
FINITE_DIFFERENCE_STEP = 1e-5
NUM_FINITE_DIFFERENCES = 20


def build_bend_masks(nx: int = NX, ny: int = NY) -> tuple[np.ndarray, np.ndarray]:
    """Return an L-bend target mask and its complement."""

    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    horizontal = np.exp(-((yy + 0.45) / 0.22) ** 2) * (xx < 0.2)
    vertical = np.exp(-((xx - 0.35) / 0.22) ** 2) * (yy > -0.6)
    corner = np.exp(-((xx - 0.2) ** 2 + (yy + 0.25) ** 2) / 0.16)

    target = np.maximum(np.maximum(horizontal, vertical), 0.8 * corner)
    target = target / np.max(target)
    complement = 1.0 - target
    return target.astype(np.float64), complement.astype(np.float64)


def smooth_density(density: jnp.ndarray) -> jnp.ndarray:
    """Small differentiable smoothing stencil for a design-grid surrogate."""

    neighbor_average = (
        jnp.roll(density, 1, axis=0)
        + jnp.roll(density, -1, axis=0)
        + jnp.roll(density, 1, axis=1)
        + jnp.roll(density, -1, axis=1)
    ) / 4.0
    return 0.65 * density + 0.35 * neighbor_average


def make_objective(target_mask: np.ndarray, complement_mask: np.ndarray):
    target = jnp.asarray(target_mask)
    complement = jnp.asarray(complement_mask)
    target_norm = jnp.sum(target)
    complement_norm = jnp.sum(complement)

    def objective(design_flat: jnp.ndarray) -> jnp.ndarray:
        density = jnp.reshape(design_flat, (NX, NY))
        smoothed = smooth_density(density)

        transmitted = jnp.sum(smoothed * target) / target_norm
        leakage = jnp.sum(smoothed * complement) / complement_norm
        grayscale_penalty = jnp.mean(smoothed * (1.0 - smoothed))
        phase_like_overlap = jnp.sum(jnp.sin(jnp.pi * smoothed) * target) / target_norm

        return transmitted**2 + 0.05 * phase_like_overlap - 0.12 * leakage**2 - 0.01 * grayscale_penalty

    return objective


def calculate_fd_gradient(objective, design: np.ndarray, indices: np.ndarray, db: float) -> np.ndarray:
    """Central finite-difference gradient for selected flattened indices."""

    fd_gradient = []
    for idx in indices:
        plus = design.copy()
        minus = design.copy()
        plus[idx] += db
        minus[idx] -= db
        fd_gradient.append((float(objective(jnp.asarray(plus))) - float(objective(jnp.asarray(minus)))) / (2.0 * db))
    return np.asarray(fd_gradient)


def plot_design_and_target(design: np.ndarray, target: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    for ax, arr, title in [
        (axes[0], design.reshape(NX, NY), "Initial Design Variables"),
        (axes[1], target, "L-Bend Target Mask"),
    ]:
        im = ax.imshow(np.rot90(arr), origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gradient(gradient: np.ndarray, path: Path) -> None:
    grad_2d = gradient.reshape(NX, NY)
    vmax = float(np.max(np.abs(grad_2d)))
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    im = ax.imshow(np.rot90(grad_2d), origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Objective Gradient wrt Design Variables")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fd_comparison(fd_gradient: np.ndarray, ad_gradient: np.ndarray, path: Path) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(fd_gradient, ad_gradient, 1)
    corr = float(np.corrcoef(fd_gradient, ad_gradient)[0, 1])

    min_g = float(min(np.min(fd_gradient), np.min(ad_gradient)))
    max_g = float(max(np.max(fd_gradient), np.max(ad_gradient)))
    rel_err = np.abs(fd_gradient - ad_gradient) / np.maximum(np.abs(fd_gradient), 1e-12)

    fig = plt.figure(figsize=(11.5, 5.0), constrained_layout=True)

    ax = fig.add_subplot(1, 2, 1)
    ax.plot([min_g, max_g], [min_g, max_g], label="y=x comparison")
    ax.plot(
        [min_g, max_g],
        [slope * min_g + intercept, slope * max_g + intercept],
        "--",
        label="best fit",
    )
    ax.plot(fd_gradient, ad_gradient, "o", label="gradient samples")
    ax.set_xlabel("Finite Difference Gradient")
    ax.set_ylabel("JAX Gradient")
    ax.grid(True, alpha=0.35)
    ax.axis("square")
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.semilogy(np.arange(len(rel_err)), rel_err, "o")
    ax.set_xlabel("sample index")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.35)

    fig.suptitle(f"Gradient check: slope={slope:.6f}, corr={corr:.6f}, max rel err={np.max(rel_err):.2e}")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return float(slope), float(intercept), corr


def main() -> None:
    output_dir = Path(os.environ.get("JSOPT_OUTPUT_DIR", "results/gradient_check/adjoint_intro_style"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    target, complement = build_bend_masks()
    objective = make_objective(target, complement)
    value_and_grad = jax.value_and_grad(objective)

    design = 0.15 + 0.70 * rng.random(NX * NY)
    value, gradient_jax = value_and_grad(jnp.asarray(design))
    gradient = np.asarray(jax.device_get(gradient_jax))

    chosen_indices = rng.choice(design.size, size=NUM_FINITE_DIFFERENCES, replace=False)
    fd_gradient = calculate_fd_gradient(objective, design, chosen_indices, FINITE_DIFFERENCE_STEP)
    sampled_gradient = gradient[chosen_indices]

    png_paths = [
        output_dir / "01_design_and_target.png",
        output_dir / "02_adjoint_gradient_heatmap.png",
        output_dir / "03_finite_difference_comparison.png",
    ]
    plot_design_and_target(design, target, png_paths[0])
    plot_gradient(gradient, png_paths[1])
    slope, intercept, corr = plot_fd_comparison(fd_gradient, sampled_gradient, png_paths[2])

    abs_err = np.abs(fd_gradient - sampled_gradient)
    rel_err = abs_err / np.maximum(np.abs(fd_gradient), 1e-12)

    print("seed:", SEED)
    print("design_shape:", (NX, NY))
    print("objective:", float(value))
    print("gradient_shape:", gradient.shape)
    print("gradient_norm:", float(np.linalg.norm(gradient)))
    print("fd_step:", FINITE_DIFFERENCE_STEP)
    print("fd_samples:", NUM_FINITE_DIFFERENCES)
    print("fit_slope_ad_over_fd:", slope)
    print("fit_intercept:", intercept)
    print("correlation:", corr)
    print("max_abs_error:", float(np.max(abs_err)))
    print("max_relative_error:", float(np.max(rel_err)))
    print("mean_relative_error:", float(np.mean(rel_err)))
    print("gradient_check_ok:", bool(abs(slope - 1.0) < 2e-3 and corr > 0.999999 and np.max(rel_err) < 1e-5))
    print("sample_table:")
    for idx, fd, ad, err in zip(chosen_indices, fd_gradient, sampled_gradient, rel_err):
        print(f" - idx={int(idx):03d} fd={fd:.9e} grad={ad:.9e} rel_err={err:.3e}")
    print("png_outputs:")
    for path in png_paths:
        print(" -", path)


if __name__ == "__main__":
    main()
