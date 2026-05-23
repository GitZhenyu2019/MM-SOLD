"""
run_2d_circular.py
==================
2D circular experiment: compare MM-SOLD moment-matched marginal vs. plain on a synthetic unit-circle dataset.

Setup
-----
  - N_TRAIN = 12 equally-spaced training points on the unit circle
  - GMM bandwidth  σ_gmm = 0.1
  - MC samples     M     = 1000  (antithetic, so 500 unique draws)
  - Smoothing grid σ ∈ {0.05, 0.10, …, 1.00}  (20 values)
"""

import os
import sys
import json
import functools

import numpy as np
import scipy.linalg
from scipy.special import logsumexp as scipy_logsumexp
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# ── Path setup: add augmentation_handwritten to sys.path ──────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
AUG_DIR = os.path.abspath(os.path.join(HERE, "..", "augmentation_handwritten"))
for _p in (AUG_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sampling_algo import gradU_LDS_mc_stationary   # reused from augmentation

# ─── Experiment config ────────────────────────────────────────────────────────
N_TRAIN     = 12
SIGMA_GMM   = 0.1
M           = 10000
SIGMA_LIST  = [round(v, 2) for v in np.arange(0.05, 0.81, 0.05).tolist()]  # 16 values
N_TEST      = 1000
GRID_N      = 300          # grid resolution per axis  (300×300 = 90 000 pts)
GRID_RANGE  = 2.5          # grid spans [-2.5, 2.5]²  (for integration)
PLOT_RANGE  = 1.7          # axes limits for density plots (circle radius = 1)
BATCH_GRID  = 2000         # grid points per GPU batch for V evaluation
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR     = os.path.join(HERE, "figures")

# ─── Training points: 12 equally-spaced points on the unit circle ─────────────
angles_train = np.linspace(0, 2 * np.pi, N_TRAIN, endpoint=False)
X_train = np.stack([np.cos(angles_train),
                    np.sin(angles_train)], axis=1).astype(np.float32)  # (12, 2)

# ─── Estimation points: 1024 equally-spaced points on the unit circle ─────────
N_EST = 1024
angles_est = np.linspace(0, 2 * np.pi, N_EST, endpoint=False)
X_est = np.stack([np.cos(angles_est),
                  np.sin(angles_est)], axis=1).astype(np.float32)     # (1024, 2)

# ─── Test points: 1000 equally-spaced points on the unit circle ───────────────
angles_test = np.linspace(0, 2 * np.pi, N_TEST, endpoint=False)
X_test = np.stack([np.cos(angles_test),
                   np.sin(angles_test)], axis=1).astype(np.float32)   # (1000, 2)

# ─── 2-D evaluation grid ─────────────────────────────────────────────────────
_xs  = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_N, dtype=np.float32)
_ys  = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_N, dtype=np.float32)
_dx  = float(_xs[1] - _xs[0])
_XX, _YY = np.meshgrid(_xs, _ys)          # each (GRID_N, GRID_N)
Z_GRID   = np.stack([_XX.ravel(),
                     _YY.ravel()], axis=1)  # (GRID_N², 2)


# ─── JAX helpers ──────────────────────────────────────────────────────────────

@jax.jit
def _U_batch(Z, X, x_norm2, sigma):
    """
    GMM potential U(z) = -logsumexp_i(-||z - xᵢ||² / (2σ²)) for each row of Z.
    Z       : (B, d)
    X       : (N, d)  training points
    x_norm2 : (N,)    precomputed ||xᵢ||²
    Returns : (B,)
    """
    z_norm2 = jnp.sum(Z * Z, axis=1)                               # (B,)
    dist2   = z_norm2[:, None] + x_norm2[None, :] - 2.0 * (Z @ X.T)  # (B, N)
    return -logsumexp(-0.5 * dist2 / (sigma * sigma), axis=1)      # (B,)


@functools.partial(jax.jit, static_argnames=("M_mc",))
def _Vc_mc_batch(key, Z, X, x_norm2, sigma, s, M_mc):
    """
    LDS-smoothed potential V(z) = E_ε[U(z + s·ε)] estimated via antithetic MC.
    Antithetic: draw M_mc//2 vectors per point, concatenate with negatives.
    Z       : (B, d)
    Returns : (B,)
    """
    B, d  = Z.shape
    half  = M_mc // 2
    eps_h = jax.random.normal(key, (B, half, d), dtype=Z.dtype)  # per-point
    eps   = jnp.concatenate([eps_h, -eps_h], axis=1)             # (B, M_mc, d)
    Zp    = Z[:, None, :] + s * eps                              # (B, M_mc, d)
    U_flat = _U_batch(Zp.reshape((-1, d)), X, x_norm2, sigma)   # (B*M_mc,)
    return U_flat.reshape((B, M_mc)).mean(axis=1)                # (B,)


def eval_V_batched(key, Z_jax, X_jax, x_norm2, sigma_gmm, s, M_mc, batch_size):
    """Evaluate V on an arbitrary set of points Z_jax in batches."""
    n = Z_jax.shape[0]
    chunks = []
    for i in range(0, n, batch_size):
        key, k = jax.random.split(key)
        v = _Vc_mc_batch(k, Z_jax[i:i + batch_size],
                         X_jax, x_norm2, sigma_gmm, s, M_mc)
        chunks.append(np.array(v))
    return np.concatenate(chunks)   # (n,)


# ─── MM-SOLD parameter estimation ─────────────────────────────────────────────

def estimate_mm_params(key, X_jax, x_norm2, sigma_gmm, sigma,
                       X_est_jax=None):
    """
    Compute MM-SOLD correction parameters for a single σ value.
    """
    # Use dense estimation points if provided, else fall back to training points
    X_ref_jax = X_est_jax if X_est_jax is not None else X_jax
    X_np = np.array(X_ref_jax)
    n, d = X_np.shape

    # μ* and Σ*
    mu_star   = X_np.mean(axis=0)                          # (d,)
    Xc        = X_np - mu_star[None, :]
    Sigma_x   = (Xc.T @ Xc) / n                            # (d,d)
    Sigma_star = (sigma_gmm ** 2) * np.eye(d) + Sigma_x   # (d,d)

    G = np.array(gradU_LDS_mc_stationary(
        key, X_ref_jax, X_jax, x_norm2,
        sigma_gmm, sigma, M,
        shared_noise=False,          # independent noise per estimation point
    ))  # (n, d)

    # λ = -mean_i g_σ(xᵢ)
    lambda_vec = -G.mean(axis=0)    # (d,)

    # C[j,k] = mean_i (xᵢ - μ*)_j · g_{σ,k}(xᵢ)
    diff = Xc                        # (n, d)
    C    = diff.T @ G / n            # (d, d)

    # S = I - ½(C + Cᵀ)
    S = np.eye(d, dtype=np.float64) - 0.5 * (C + C.T)

    # Lyapunov equation: Σ*·Λ + Λ·Σ* = 2·S
    Lambda_mat = scipy.linalg.solve_continuous_lyapunov(
        Sigma_star.astype(np.float64), 2.0 * S
    )

    return (lambda_vec.astype(np.float32),
            Lambda_mat.astype(np.float32),
            mu_star.astype(np.float32),
            Sigma_star.astype(np.float32))


# ─── Energy functions ─────────────────────────────────────────────────────────

def energy_mm(V, Z, lambda_vec, Lambda_mat, mu_star):
    """
    E*(z) = V(z) + λᵀ·z + ½(z - μ*)ᵀ·Λ·(z - μ*)
    V      : (B,)
    Z      : (B, d)  numpy array
    Returns: (B,)
    """
    linear = Z @ lambda_vec                                      # (B,)
    diff   = Z - mu_star[None, :]                               # (B, d)
    quad   = 0.5 * np.einsum('bi,ij,bj->b', diff, Lambda_mat, diff)  # (B,)
    return V + linear + quad


def log_partition_2d(E_flat, dx):
    """
    log Z = log ∫ exp(-E(z)) dz  ≈ logsumexp(-E_grid) + 2·log(dx)
    """
    return float(scipy_logsumexp(-E_flat) + 2 * np.log(dx))


def compute_nll(E_test, log_Z):
    """NLL = E_test[E(z)] + log Z"""
    return float(np.mean(E_test) + log_Z)


# ─── Plotting ──────────────────────────────────────────────────────────────────

def _plot_single_density(E_2d, X_train, sigma, fig_path):
    """
    Save one density figure: blue imshow + red ▲ training pts + dashed unit circle.
    E_2d : (GRID_N, GRID_N)  energy values on the full GRID_RANGE grid.
    """
    dens_2d = gaussian_filter(
        np.exp(-E_2d.astype(np.float64) + E_2d.min()), sigma=2.0
    )

    iy = np.abs(_ys) <= PLOT_RANGE
    ix = np.abs(_xs) <= PLOT_RANGE
    dens_display = dens_2d[np.ix_(iy, ix)]
    vmin = float(np.percentile(dens_display, 1))
    vmax = float(np.percentile(dens_display, 99))

    extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(dens_2d, origin='lower', extent=extent,
              cmap='Blues', aspect='equal', interpolation='bilinear',
              vmin=vmin, vmax=vmax)
    ax.scatter(X_train[:, 0], X_train[:, 1],
               c='red', marker='^', s=70, zorder=5)
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False,
                         linestyle='--', linewidth=1.2, zorder=4)
    ax.add_patch(circle)

    ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_densities(V_grid, lambda_vec, Lambda_mat, mu_star,
                   X_train, sigma, fig_dir):
    
    V_2d = V_grid.reshape(GRID_N, GRID_N)

    E_mm_flat = energy_mm(V_grid, Z_GRID.astype(np.float32),
                           lambda_vec, Lambda_mat, mu_star)
    E_mm_2d   = E_mm_flat.reshape(GRID_N, GRID_N)

    _plot_single_density(E_mm_2d, X_train, sigma,
                         os.path.join(fig_dir, f'mm_sigma{sigma:.2f}.png'))
    _plot_single_density(V_2d,    X_train, sigma,
                         os.path.join(fig_dir, f'base_sigma{sigma:.2f}.png'))


def _save_nll_fig(sigma_list, nll_vals, color, marker, fig_path):
    """Save a single NLL vs σ line plot (no title, no legend)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sigma_list, nll_vals, color=color, marker=marker,
            linestyle='-', linewidth=1.8, markersize=5)
    ax.set_xlabel('σ  (LDS smoothing bandwidth)', fontsize=12)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fig_path}")


def plot_nll_curves(sigma_list, nlls_mm, nlls_base, fig_dir):
    """Save three NLL figures: combined (no legend), MM-SOLD only, Baseline only."""
    # Combined (two curves, no legend)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sigma_list, nlls_mm,   'b-o',  linewidth=1.8, markersize=5)
    ax.plot(sigma_list, nlls_base, 'r--s', linewidth=1.8, markersize=5)
    ax.set_xlabel('σ  (LDS smoothing bandwidth)', fontsize=12)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    combined_path = os.path.join(fig_dir, 'nll_vs_sigma.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {combined_path}")

    # MM-SOLD only
    _save_nll_fig(sigma_list, nlls_mm, 'blue', 'o',
                  os.path.join(fig_dir, 'nll_vs_sigma_mm.png'))

    # Baseline only
    _save_nll_fig(sigma_list, nlls_base, 'red', 's',
                  os.path.join(fig_dir, 'nll_vs_sigma_base.png'))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,     exist_ok=True)

    print("=" * 60)
    print("2D Circular Experiment — MM-SOLD vs Baseline")
    print("=" * 60)
    print(f"JAX backend : {jax.default_backend()}  |  devices: {jax.devices()}")
    print(f"N_TRAIN={N_TRAIN},  N_EST={N_EST},  σ_gmm={SIGMA_GMM},  M={M}")
    print(f"Grid: {GRID_N}x{GRID_N} over [-{GRID_RANGE}, {GRID_RANGE}]²  "
          f"(Δx = {_dx:.4f})")
    print(f"N_TEST={N_TEST} test points on unit circle")
    print(f"σ values ({len(SIGMA_LIST)}): {SIGMA_LIST}")

    # Pre-convert to JAX arrays (constant throughout the loop)
    X_jax      = jnp.asarray(X_train)                        # (12, 2)
    x_norm2    = jnp.sum(X_jax * X_jax, axis=1)             # (12,)
    X_est_jax  = jnp.asarray(X_est)                          # (1024, 2)
    Z_grid_jax = jnp.asarray(Z_GRID)                         # (GRID_N², 2)
    X_test_jax = jnp.asarray(X_test)                         # (N_TEST, 2)

    key = jax.random.PRNGKey(0)

    nlls_mm   = []
    nlls_base = []
    all_results = []

    for idx, sigma in enumerate(SIGMA_LIST):
        print(f"\n[{idx+1:02d}/{len(SIGMA_LIST)}]  σ = {sigma:.2f}")

        # ── MM-SOLD parameters ────────────────────────────────────────────────
        key, k_g = jax.random.split(key)
        lambda_vec, Lambda_mat, mu_star, Sigma_star = estimate_mm_params(
            k_g, X_jax, x_norm2, SIGMA_GMM, sigma,
            X_est_jax=X_est_jax,
        )
        print(f"  λ  norm  = {np.linalg.norm(lambda_vec):.5f}")
        eigs_L = np.linalg.eigvalsh(Lambda_mat)
        print(f"  Λ  eigs  = [{eigs_L[0]:.4f},  {eigs_L[1]:.4f}]")

        # ── V on grid ─────────────────────────────────────────────────────────
        print(f"  Computing V on {GRID_N}x{GRID_N} grid ...")
        key, k_vg = jax.random.split(key)
        V_grid = eval_V_batched(
            k_vg, Z_grid_jax, X_jax, x_norm2,
            SIGMA_GMM, sigma, M, BATCH_GRID
        )  # (GRID_N², ) float32 numpy

        # ── V on test points ──────────────────────────────────────────────────
        key, k_vt = jax.random.split(key)
        V_test = eval_V_batched(
            k_vt, X_test_jax, X_jax, x_norm2,
            SIGMA_GMM, sigma, M, N_TEST
        )  # (N_TEST,)

        # ── Energies on grid and test points ──────────────────────────────────
        Z_g_np = Z_GRID.astype(np.float32)
        E_mm_grid  = energy_mm(V_grid, Z_g_np, lambda_vec, Lambda_mat, mu_star)
        E_bas_grid = V_grid

        X_test_np  = X_test.astype(np.float32)
        E_mm_test  = energy_mm(V_test, X_test_np, lambda_vec, Lambda_mat, mu_star)
        E_bas_test = V_test

        # ── Normalisation constants via 2-D grid quadrature ───────────────────
        logZ_mm  = log_partition_2d(E_mm_grid,  _dx)
        logZ_bas = log_partition_2d(E_bas_grid, _dx)
        print(f"  log Z (MM-SOLD)  = {logZ_mm:.4f}")
        print(f"  log Z (Baseline) = {logZ_bas:.4f}")

        # ── NLL ───────────────────────────────────────────────────────────────
        nll_mm  = compute_nll(E_mm_test,  logZ_mm)
        nll_bas = compute_nll(E_bas_test, logZ_bas)
        print(f"  NLL  (MM-SOLD)   = {nll_mm:.4f}")
        print(f"  NLL  (Baseline)  = {nll_bas:.4f}")

        nlls_mm.append(nll_mm)
        nlls_base.append(nll_bas)

        # ── Density figures (one per method) ─────────────────────────────────
        plot_densities(V_grid, lambda_vec, Lambda_mat, mu_star,
                       X_train, sigma, FIG_DIR)
        print(f"  Saved mm_sigma{sigma:.2f}.png  +  base_sigma{sigma:.2f}.png")

        # ── Collect numerical results ─────────────────────────────────────────
        all_results.append(dict(
            sigma        = sigma,
            nll_mm       = nll_mm,
            nll_bas      = nll_bas,
            logZ_mm      = logZ_mm,
            logZ_bas     = logZ_bas,
            lambda_norm  = float(np.linalg.norm(lambda_vec)),
            Lambda_eigs  = eigs_L.tolist(),
            mu_star      = mu_star.tolist(),
        ))

    # ── NLL vs σ plot ─────────────────────────────────────────────────────────
    plot_nll_curves(SIGMA_LIST, nlls_mm, nlls_base, FIG_DIR)

    # ── Save numerical results ────────────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved numerical results to {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"{'σ':>6}  {'NLL MM-SOLD':>12}  {'NLL Baseline':>12}")
    print("-" * 34)
    for r in all_results:
        print(f"{r['sigma']:>6.2f}  {r['nll_mm']:>12.4f}  {r['nll_bas']:>12.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
