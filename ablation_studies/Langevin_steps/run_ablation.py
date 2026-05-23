"""
run_ablation.py
===============
Ablation study: effect of Langevin step count and step size on MM-SOLD sample quality.

Distribution : 2-D checkerboard (same as low_dimension experiment)
Fixed params : M=8, σ=0.2, σ_GMM=0.1, N_TARGET=5000, N_TRAIN=500, N_GEN=5000
Grid search  : n_steps ∈ N_STEPS_LIST,  step_size ∈ STEP_SIZE_LIST
"""

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from matplotlib.colors import LinearSegmentedColormap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────

SEED        = 0
N_TARGET    = 5000      # proxy samples for target distribution
N_TRAIN     = 500        # training samples
N_GEN       = 5000      # samples to generate per configuration
M           = 8         # LDS MC noise samples (fixed)
SIGMA       = 0.2       # LDS smoothing strength (fixed)
SIGMA_GMM   = 0.1      # GMM bandwidth

# Grid-search values  (current baseline: n_steps=2000, step_size=5e-4)
N_STEPS_LIST   = [1, 5, 10, 25, 50, 100, 200, 500]
STEP_SIZE_LIST = [1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2.  Checkerboard distribution
# ─────────────────────────────────────────────

def sample_checkerboard(n: int, rng=None) -> np.ndarray:
    rng = rng or np.random
    x1  = rng.rand(n) * 4 - 2
    x2_ = rng.rand(n) - rng.randint(0, 2, n) * 2
    x2  = x2_ + (np.floor(x1) % 2)
    return (np.stack([x1, x2], axis=1) * 2).astype(np.float32)


def _fps(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Farthest Point Sampling: select n maximally spread-out points."""
    N = len(pts)
    if n >= N:
        return pts
    rng      = np.random.RandomState(seed)
    selected = np.empty(n, dtype=int)
    selected[0] = rng.randint(N)
    dists = np.sum((pts - pts[selected[0]]) ** 2, axis=1)
    for i in range(1, n):
        selected[i] = int(np.argmax(dists))
        d = np.sum((pts - pts[selected[i]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
    return pts[selected]


# ─────────────────────────────────────────────
# 3.  Manifold utilities
# ─────────────────────────────────────────────

@jax.jit
def _k_batch(Z, X, x_norm2, sigma):
    """Soft-max weighted barycenter k(Z) = Σ_i w_i(Z) X_i."""
    z_norm2 = jnp.sum(Z * Z, axis=1)
    dist2   = z_norm2[:, None] + x_norm2[None, :] - 2.0 * (Z @ X.T)
    logw    = -0.5 * dist2 / (sigma * sigma)
    logw   -= logsumexp(logw, axis=1, keepdims=True)
    return jnp.exp(logw) @ X


@jax.jit
def _gradU_batch(Z, X, x_norm2, sigma):
    """∇U(Z) = (Z - k(Z)) / σ²."""
    return (Z - _k_batch(Z, X, x_norm2, sigma)) / (sigma * sigma)


@jax.jit
def _proj_tangent(Y, G):
    """Project G onto the tangent space of the scaled Stiefel manifold at Y."""
    B   = Y.shape[0]
    sym = 0.5 * (Y.T @ G + G.T @ Y) / B
    return G - Y @ sym


@jax.jit
def _retract_qr(Y):
    """QR retraction: enforce Y^T Y = B I."""
    B    = Y.shape[0]
    Q, R = jnp.linalg.qr(Y, mode="reduced")
    s    = jnp.sign(jnp.diag(R))
    s    = jnp.where(s == 0.0, 1.0, s)
    return jnp.sqrt(float(B)) * (Q * s[None, :])


@jax.jit
def _enforce_manifold(Y):
    """Centre rows then QR-retract."""
    Y = Y - jnp.mean(Y, axis=0, keepdims=True)
    return _retract_qr(Y)


@jax.jit
def _Z_to_Y(Z, mu, L):
    return jax.scipy.linalg.solve_triangular(L, (Z - mu[None, :]).T, lower=True).T


@jax.jit
def _Y_to_Z(Y, mu, L):
    return mu[None, :] + Y @ L.T


# ─────────────────────────────────────────────
# 4.  MM-SOLD sampler
# ─────────────────────────────────────────────

_mmsold_step_cache: dict = {}


def _get_mmsold_step(M_mc: int):
    """Return (and cache) a JIT-compiled one-step Leimkuhler-Matthews function."""
    if M_mc in _mmsold_step_cache:
        return _mmsold_step_cache[M_mc]

    @jax.jit
    def _step(key, Y, xi_prev, X, x_norm2, mu_tgt, L_tgt, sigma_gmm, s, h):
        key, k_g, k_n = jax.random.split(key, 3)
        Z = _Y_to_Z(Y, mu_tgt, L_tgt)

        if M_mc == 0:
            gZ = _gradU_batch(Z, X, x_norm2, sigma_gmm)
        else:
            def _one_noise(carry, key_m):
                Zp = Z + s * jax.random.normal(key_m, Z.shape, dtype=Z.dtype)
                return carry, _gradU_batch(Zp, X, x_norm2, sigma_gmm)
            keys_m = jax.random.split(k_g, M_mc)
            _, grads = jax.lax.scan(_one_noise, None, keys_m)
            gZ = jnp.mean(grads, axis=0)

        gY_tan      = _proj_tangent(Y, gZ @ L_tgt.T)
        xi          = jax.random.normal(k_n, Y.shape, dtype=Y.dtype)
        xi_tan      = _proj_tangent(Y, xi)
        xi_prev_tan = _proj_tangent(Y, xi_prev)
        Y_new = Y - h * gY_tan + jnp.sqrt(h / 2.0) * (xi_prev_tan + xi_tan)
        Y_new = _enforce_manifold(Y_new)
        return key, Y_new, xi

    _mmsold_step_cache[M_mc] = _step
    return _step


def mmsold_sample(X_train: np.ndarray,
                  sigma: float,
                  M_mc: int,
                  n_samples: int  = N_GEN,
                  n_steps: int    = 2000,
                  sigma_gmm: float = SIGMA_GMM,
                  h: float         = 5e-4,
                  seed: int        = SEED) -> np.ndarray:
    """Generate n_samples points using MM-SOLD."""
    key   = jax.random.PRNGKey(seed)
    X     = jnp.array(X_train, dtype=jnp.float32)
    n, d  = X.shape
    x_n2  = jnp.sum(X * X, axis=1)

    mu_tgt  = jnp.mean(X, axis=0)
    Xc      = X - mu_tgt[None, :]
    Sigma_x = (Xc.T @ Xc) / n
    Sigma_t = (sigma_gmm ** 2) * jnp.eye(d) + Sigma_x
    L_tgt   = jnp.linalg.cholesky(Sigma_t)

    key, k1, k2, k3 = jax.random.split(key, 4)
    idx0   = jax.random.randint(k1, (n_samples,), 0, n)
    Z0     = X[idx0] + sigma_gmm * jax.random.normal(k2, (n_samples, d))
    Y0     = _enforce_manifold(_Z_to_Y(Z0, mu_tgt, L_tgt))
    xi_prv = jax.random.normal(k3, Y0.shape, dtype=Y0.dtype)

    step_fn = _get_mmsold_step(M_mc)
    s_f     = jnp.float32(sigma)
    sgmm_f  = jnp.float32(sigma_gmm)
    h_f     = jnp.float32(h)

    Y, xi = Y0, xi_prv
    for _ in range(n_steps):
        key, Y, xi = step_fn(key, Y, xi, X, x_n2, mu_tgt, L_tgt,
                             sgmm_f, s_f, h_f)

    return np.array(_Y_to_Z(Y, mu_tgt, L_tgt))


# ─────────────────────────────────────────────
# 5.  Sliced Wasserstein-2 distance
# ─────────────────────────────────────────────

def sw2_distance(a: np.ndarray,
                 b: np.ndarray,
                 n_projections: int = 512,
                 seed: int = 0) -> float:
    """Sliced Wasserstein-2 distance via random projections."""
    rng = np.random.RandomState(seed)
    a   = np.asarray(a, dtype=np.float64)
    b   = np.asarray(b, dtype=np.float64)
    d   = a.shape[1]
    na, nb = len(a), len(b)

    dirs = rng.randn(n_projections, d)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    pa = np.sort(a @ dirs.T, axis=0)
    pb = np.sort(b @ dirs.T, axis=0)

    if na != nb:
        n  = max(na, nb)
        ta = np.linspace(0.0, 1.0, na)
        tb = np.linspace(0.0, 1.0, nb)
        t  = np.linspace(0.0, 1.0, n)
        pa = np.stack([np.interp(t, ta, pa[:, k]) for k in range(n_projections)], axis=1)
        pb = np.stack([np.interp(t, tb, pb[:, k]) for k in range(n_projections)], axis=1)

    sw2_sq = float(np.mean((pa - pb) ** 2))
    return float(np.sqrt(max(sw2_sq, 0.0)))


# ─────────────────────────────────────────────
# 6.  Grid search
# ─────────────────────────────────────────────

def run_grid_search(X_train: np.ndarray,
                    X_target: np.ndarray,
                    n_steps_list: list = N_STEPS_LIST,
                    step_size_list: list = STEP_SIZE_LIST,
                    seed: int = SEED) -> dict:
    """
    Run MM-SOLD for all (n_steps, step_size) combinations.
    """
    nS = len(n_steps_list)
    nH = len(step_size_list)

    res = dict(
        w2_target    = np.full((nS, nH), np.nan),
        w2_train     = np.full((nS, nH), np.nan),
        samples      = {},
        n_steps_list = n_steps_list,
        step_size_list = step_size_list,
    )

    total = nS * nH
    idx   = 0
    for iS, n_steps in enumerate(n_steps_list):
        for iH, h in enumerate(step_size_list):
            idx += 1
            print(f"  [{idx}/{total}]  n_steps={n_steps}, step_size={h:.1e}",
                  flush=True)

            t0   = time.time()
            samp = mmsold_sample(X_train, SIGMA, M,
                                 n_samples=N_GEN,
                                 n_steps=n_steps,
                                 sigma_gmm=SIGMA_GMM,
                                 h=h,
                                 seed=seed)
            elapsed = time.time() - t0

            w2t  = sw2_distance(samp, X_target)
            w2tr = sw2_distance(samp, X_train)
            res["w2_target"][iS, iH]           = w2t
            res["w2_train"][iS, iH]            = w2tr
            res["samples"][(n_steps, h)]       = samp
            print(f"    {elapsed:.1f}s  SW2_target={w2t:.4f}  SW2_train={w2tr:.4f}",
                  flush=True)

    return res


# ─────────────────────────────────────────────
# 7.  Scatter plots (one figure per configuration)
# ─────────────────────────────────────────────

def plot_scatter_all(res: dict, X_train: np.ndarray):
    """
    One figure per (n_steps, step_size) combination.
    Each figure: MM-SOLD samples (blue) vs training data (red).
    """
    scatter_dir = os.path.join(FIGURES_DIR, "scatter")
    os.makedirs(scatter_dir, exist_ok=True)

    n_steps_list   = res["n_steps_list"]
    step_size_list = res["step_size_list"]

    for n_steps in n_steps_list:
        for h in step_size_list:
            samp = res["samples"].get((n_steps, h))
            if samp is None:
                continue

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(samp[:, 0], samp[:, 1],
                       s=4, alpha=0.5, linewidths=0,
                       color="#90CAF9", label="MM-SOLD")
            ax.scatter(X_train[:, 0], X_train[:, 1],
                       s=6, alpha=0.8, linewidths=0,
                       color="#EF9A9A", label="Training", zorder=5)

            ax.set_aspect("equal")
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_title(f"steps={n_steps},  size={h:.0e}", fontsize=11)
            ax.legend(fontsize=9, markerscale=3, loc="upper right")
            ax.tick_params(labelsize=8)

            fig.tight_layout()
            fname = f"scatter_steps{n_steps}_h{h:.0e}.pdf"
            path  = os.path.join(scatter_dir, fname)
            fig.savefig(path, bbox_inches="tight", dpi=200)
            plt.close(fig)

    print(f"  Saved scatter plots to {scatter_dir}/")


# ─────────────────────────────────────────────
# 8.  Line plots
# ─────────────────────────────────────────────

# Distinct, paper-quality colours for line plots
_PALETTE = ["#1565C0", "#C62828", "#2E7D32", "#E65100", "#6A1B9A", "#00838F",
            "#F9A825", "#4E342E"]


def _line_colors(n: int):
    return _PALETTE[:n]


def plot_line_vs_steps(res: dict, ref_sw2_train: float = None):
    """
    Two figures (W2-target, W2-train):
    x-axis = n_steps; one line per step_size value.
    ref_sw2_train: SW2(X_target, X_train) reference baseline drawn as dashed line
                   on the W2-to-train figure.
    """
    n_steps_list   = res["n_steps_list"]
    step_size_list = res["step_size_list"]
    colors         = _line_colors(len(step_size_list))

    for _, w2_key, ylabel, fname in [
        ("target", "w2_target", "W2 to target distribution",
         "line_target_vs_steps.pdf"),
        ("train",  "w2_train",  "W2 to training dataset",
         "line_train_vs_steps.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for iH, h in enumerate(step_size_list):
            vals = res[w2_key][:, iH]
            ax.plot(n_steps_list, vals, "o-",
                    color=colors[iH], label=f"h={h:.0e}")
        if ref_sw2_train is not None:
            ax.axhline(ref_sw2_train, color="black", linestyle="--",
                       linewidth=1.2, label="Reference (target vs train)")
        ax.set_xlabel("Number of Langevin steps")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.legend(fontsize=8, title="Step size", title_fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def plot_line_vs_stepsize(res: dict, ref_sw2_train: float = None):
    """
    Two figures (W2-target, W2-train):
    x-axis = step_size; one line per n_steps value.
    ref_sw2_train: SW2(X_target, X_train) reference baseline drawn as dashed line
                   on the W2-to-train figure.
    """
    n_steps_list   = res["n_steps_list"]
    step_size_list = res["step_size_list"]
    colors         = _line_colors(len(n_steps_list))

    for _, w2_key, ylabel, fname in [
        ("target", "w2_target", "W2 to target distribution",
         "line_target_vs_stepsize.pdf"),
        ("train",  "w2_train",  "W2 to training dataset",
         "line_train_vs_stepsize.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for iS, n_steps in enumerate(n_steps_list):
            vals = res[w2_key][iS, :]
            ax.plot(step_size_list, vals, "o-",
                    color=colors[iS], label=f"{n_steps}")
        if ref_sw2_train is not None:
            ax.axhline(ref_sw2_train, color="black", linestyle="--",
                       linewidth=1.2, label="Reference (target vs train)")
        ax.set_xlabel("Step size")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.legend(fontsize=8, title="n_steps", title_fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ─────────────────────────────────────────────
# 9.  Heatmaps
# ─────────────────────────────────────────────

def plot_heatmaps(res: dict):
    """
    Two heatmaps: W2-to-target and W2-to-train.
    """
    n_steps_list   = res["n_steps_list"]
    step_size_list = res["step_size_list"]

    for w2_key, title, fname in [
        ("w2_target", "W2 to target distribution", "heatmap_target.pdf"),
        ("w2_train",  "W2 to training dataset",    "heatmap_train.pdf"),
    ]:
        data = res[w2_key]   # (nS, nH)

        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(step_size_list) + 2),
                                        max(4, 0.8 * len(n_steps_list) + 1.5)))

        # white (low/good) → red (high/poor)
        wr_cmap = LinearSegmentedColormap.from_list(
            "white_red", ["white", "#D32F2F"])

        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if vmin == vmax:
            vmax = vmin + 1e-6

        im = ax.imshow(data, aspect="auto", cmap=wr_cmap,
                       vmin=vmin, vmax=vmax, origin="upper")

        # Axes labels
        ax.set_xticks(range(len(step_size_list)))
        ax.set_xticklabels([f"{h:.0e}" for h in step_size_list],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(n_steps_list)))
        ax.set_yticklabels([str(s) for s in n_steps_list], fontsize=8)
        ax.set_xlabel("Step size (h)", fontsize=10)
        ax.set_ylabel("Number of Langevin steps", fontsize=10)
        ax.set_title(title, fontsize=11)

        # Colorbar on the right (no label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ─────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("MM-SOLD Langevin Ablation Study")
    print(f"  M={M}, σ={SIGMA}, σ_GMM={SIGMA_GMM}")
    print(f"  N_TARGET={N_TARGET}, N_TRAIN={N_TRAIN}, N_GEN={N_GEN}")
    print(f"  n_steps grid : {N_STEPS_LIST}")
    print(f"  step_size grid: {[f'{h:.0e}' for h in STEP_SIZE_LIST]}")
    print("=" * 50)

    # ── Build datasets ──────────────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    big = sample_checkerboard(N_TARGET * 10, rng)
    X_target = _fps(big, N_TARGET, seed=SEED)

    rng2 = np.random.RandomState(SEED + 1)
    big2 = sample_checkerboard(N_TRAIN * 10, rng2)
    X_train = _fps(big2, N_TRAIN, seed=SEED + 1)

    print(f"\nTarget: {X_target.shape},  Train: {X_train.shape}")

    # ── Reference baseline: SW2(X_target, X_train) ──────────────────────────
    ref_sw2_train = sw2_distance(X_target, X_train)
    print(f"Reference SW2(target vs train) = {ref_sw2_train:.4f}")

    # ── Grid search ─────────────────────────────────────────────────────────
    print("\n=== Grid search ===")
    res = run_grid_search(X_train, X_target)

    # ── Plotting ─────────────────────────────────────────────────────────────
    print("\n=== Plotting ===")
    plot_scatter_all(res, X_train)
    plot_line_vs_steps(res, ref_sw2_train)
    plot_line_vs_stepsize(res, ref_sw2_train)
    plot_heatmaps(res)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n=== W2-to-target summary (rows=n_steps, cols=step_size) ===")
    header = "           " + "  ".join(f"{h:>8.0e}" for h in STEP_SIZE_LIST)
    print(header)
    for iS, n_steps in enumerate(N_STEPS_LIST):
        row = f"n={n_steps:>6}  " + "  ".join(
            f"{res['w2_target'][iS, iH]:8.4f}"
            for iH in range(len(STEP_SIZE_LIST))
        )
        print(row)

    print("\n=== All done. Figures saved to:", FIGURES_DIR, "===")


if __name__ == "__main__":
    main()
