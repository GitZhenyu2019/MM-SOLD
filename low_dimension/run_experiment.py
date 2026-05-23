"""
run_experiment.py
=================
MM-SOLD vs σ-CFDM comparison on low-dimensional data distributions.
"""

import os
import sys
import pickle
import time
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial

import ot                               # Python Optimal Transport (POT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import sklearn.datasets

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

N_TARGET      = 20000          # proxy samples for target distribution
N_TRAIN       = 1000           # training samples
N_GEN         = 20000          # samples to generate per method
N_STEPS       = 3000           # ODE steps (σ-CFDM) / Langevin steps (MM-SOLD)

SCFDM_BATCH   = 500           # σ-CFDM: particles processed per JIT-batch

MM_SIGMA_GMM  = 0.02          # MM-SOLD: GMM bandwidth
MM_H          = 5e-4          # MM-SOLD: Langevin step size

M_LIST        = [2, 4, 8]
SIGMA_LIST    = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2]

#M_LIST        = [2]
#SIGMA_LIST    = [0.02]

# Built-in 2-D synthetic distributions (always available without mesh files)
DATASETS_2D   = ["checkerboard", "spirals"]

def _is_3d(name: str) -> bool:
    return name not in DATASETS_2D

SEED          = 0

# ── 3-D point-cloud render camera ──────
PS_CAMERA_POS        = (-2.5, 1.8, -2.0)  # flat objects (car, airplane…): oblique top-front-left
PS_CAMERA_POS_UPRIGHT = (0.0, 0.0, 3.5)  # upright objects (guitar, lamp…): perfectly straight-on (Z-axis)
PS_CAMERA_TARGET     = (0.0, 0.0, 0.0)   # usually keep at origin

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(_SCRIPT_DIR, "results")
FIGURES_DIR   = os.path.join(_SCRIPT_DIR, "figures")

MODELNET40_DIR = os.environ.get("MODELNET40_DIR", "/path/to/datasets/ModelNet40")

# ─────────────────────────────────────────────
# 1.  Target distributions
# ─────────────────────────────────────────────

# normalize point clouds
def _normalize_pc(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0)
    pts = pts / np.max(np.linalg.norm(pts, axis=1))
    return pts.astype(np.float32)


def sample_checkerboard(n: int, rng=None) -> np.ndarray:
    rng = rng or np.random
    x1  = rng.rand(n) * 4 - 2
    x2_ = rng.rand(n) - rng.randint(0, 2, n) * 2
    x2  = x2_ + (np.floor(x1) % 2)
    return (np.stack([x1, x2], axis=1) * 2).astype(np.float32)


def sample_spirals(n: int, rng=None) -> np.ndarray:
    rng  = rng or np.random
    half = n // 2
    ang  = np.sqrt(rng.rand(half, 1)) * 540 * (2 * np.pi / 360)
    d1x  = -np.cos(ang) * ang + rng.rand(half, 1) * 0.5
    d1y  =  np.sin(ang) * ang + rng.rand(half, 1) * 0.5
    x    = np.vstack([np.hstack([d1x, d1y]), np.hstack([-d1x, -d1y])]) / 3
    x   += rng.randn(*x.shape) * 0.1
    return x.astype(np.float32)


def sample_car(n: int, rng=None) -> np.ndarray:
    rng = rng or np.random
    parts = []
    # body – elongated ellipsoid surface
    nb = int(0.65 * n)
    u, v = rng.uniform(0, 2*np.pi, nb), rng.uniform(0, np.pi, nb)
    parts.append(np.stack([
        2.0 * np.sin(v) * np.cos(u),
        0.7 * np.sin(v) * np.sin(u),
        0.5 * np.cos(v)
    ], axis=1))
    # four wheels
    nw = (n - nb) // 4
    for cx, cy, cz in [(1.3, 0.85, -0.4), (1.3, -0.85, -0.4),
                        (-1.3, 0.85, -0.4), (-1.3, -0.85, -0.4)]:
        a, b_ = rng.uniform(0, 2*np.pi, nw), rng.uniform(0, np.pi, nw)
        parts.append(np.stack([
            cx + 0.35 * np.sin(b_) * np.cos(a),
            cy + 0.15 * np.sin(b_) * np.sin(a),
            cz + 0.35 * np.cos(b_)
        ], axis=1))
    pts  = np.vstack(parts)[:n]
    pts += rng.randn(*pts.shape) * 0.02
    return _normalize_pc(pts)


def sample_chair(n: int, rng=None) -> np.ndarray:
    rng = rng or np.random
    parts = []
    ns = n // 3
    parts.append(np.stack([rng.uniform(-1, 1, ns),
                            rng.uniform(-1, 1, ns),
                            np.zeros(ns)], axis=1))
    nb = n // 4
    parts.append(np.stack([rng.uniform(-1, 1, nb),
                            np.full(nb, -1.0),
                            rng.uniform(0, 1.5, nb)], axis=1))
    nl = (n - ns - nb) // 4
    for lx, ly in [(0.85, 0.85), (0.85, -0.85), (-0.85, 0.85), (-0.85, -0.85)]:
        parts.append(np.stack([np.full(nl, lx),
                                np.full(nl, ly),
                                rng.uniform(-1.5, 0, nl)], axis=1))
    pts  = np.vstack(parts)[:n]
    pts += rng.randn(*pts.shape) * 0.02
    return _normalize_pc(pts)


def sample_fish(n: int, rng=None) -> np.ndarray:
    rng = rng or np.random
    nb  = int(0.7 * n)
    u, v = rng.uniform(0, 2*np.pi, nb), rng.uniform(0, np.pi, nb)
    body = np.stack([
        1.5 * np.sin(v) * np.cos(u),
        0.5 * np.sin(v) * np.sin(u),
        0.4 * np.cos(v)
    ], axis=1)
    nt  = n - nb
    a   = rng.uniform(0, np.pi, nt)
    tail = np.stack([
        np.full(nt, -1.5) + rng.uniform(-0.3, 0, nt),
        0.6 * np.sin(a),
        0.4 * np.cos(a)
    ], axis=1)
    pts  = np.vstack([body, tail])[:n]
    pts += rng.randn(*pts.shape) * 0.02
    return _normalize_pc(pts)


_SAMPLERS = {
    "checkerboard": sample_checkerboard,
    "spirals":      sample_spirals,
    "car":          sample_car,
    "chair":        sample_chair,
    "fish":         sample_fish,
}


def _fps(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling: select n maximally spread-out points from pts.
    """
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


def _load_off_vertices(path: str) -> np.ndarray:
    with open(path, "r") as f:
        lines = f.readlines()

    # Strip comments and blank lines
    lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

    # First line may be "OFF" or "OFF3524 2096 0"
    first = lines[0]
    if first.upper().startswith("OFF"):
        rest = first[3:].strip()
        if rest:
            # counts are on the same line as OFF
            counts_line = rest
            data_start = 1
        else:
            counts_line = lines[1]
            data_start = 2
    else:
        raise RuntimeError(f"Not a valid OFF file: {path}")

    n_verts, n_faces, _ = map(int, counts_line.split())
    verts = []
    for i in range(data_start, data_start + n_verts):
        verts.append(list(map(float, lines[i].split()[:3])))
    return np.array(verts, dtype=np.float32)


def _find_mesh_file(name: str) -> tuple:
    """
    Search for a mesh file (PLY or OFF) for the given dataset name.
    """
    search_dirs = [
        (os.path.join(_SCRIPT_DIR, "meshes"),                              "local"),
        (os.path.join(_SCRIPT_DIR, "../../σ_CFDM_Paper/results/point_clouds"), "paper"),
        (os.path.join(MODELNET40_DIR, name, "train"),                      "modelnet"),
        (os.path.join(MODELNET40_DIR, name, "test"),                       "modelnet"),
        (os.path.join(_SCRIPT_DIR, "../../Datasets/ModelNet40", name, "train"), "modelnet"),
        (os.path.join(_SCRIPT_DIR, "../../Datasets/ModelNet40", name, "test"),  "modelnet"),
    ]
    for sdir, _ in search_dirs:
        if not os.path.isdir(sdir):
            continue
        files = sorted(os.listdir(sdir))
        # Prefer files that start with name, then any file, PLY before OFF
        for ext in ("ply", "off"):
            for fn in files:
                if fn.startswith(name) and fn.endswith(f".{ext}"):
                    return os.path.join(sdir, fn), ext
            for fn in files:
                if fn.endswith(f".{ext}"):
                    return os.path.join(sdir, fn), ext
    return None, None


def get_dataset(name: str, n: int, seed: int = 0) -> np.ndarray:
    """
    Return n uniformly-spread samples from the named distribution.
    """
    if _is_3d(name):
        mesh_path, ext = _find_mesh_file(name)
        if mesh_path is not None:
            # ── Try pcu surface sampling first ───────────────────────────
            try:
                import point_cloud_utils as pcu
                print(f"  [mesh] {name}: loading {mesh_path}")
                v, f = pcu.load_mesh_vf(mesh_path)
                fid, bc = pcu.sample_mesh_poisson_disk(v, f, n)
                pts = pcu.interpolate_barycentric_coords(f, fid, bc, v)
                if len(pts) >= n:
                    print(f"  [mesh] {name}: sampled {n} pts from real mesh")
                    return _normalize_pc(pts[:n])
                fid2, bc2 = pcu.sample_mesh_random(v, f, n)
                pts2 = pcu.interpolate_barycentric_coords(f, fid2, bc2, v)
                return _normalize_pc(pts2[:n])
            except Exception as e:
                print(f"  [warn] {name}: pcu failed ({e}), "
                      f"falling back to vertex aggregation from OFF files")

            # ── Fallback: aggregate vertices from all OFF files in the split dir ─
            if ext == "off":
                mesh_dir = os.path.dirname(mesh_path)
                off_files = sorted(
                    f for f in os.listdir(mesh_dir) if f.endswith(".off")
                )
                all_verts = []
                for fn in off_files:
                    try:
                        verts = _load_off_vertices(os.path.join(mesh_dir, fn))
                        if verts.shape[1] == 3:
                            all_verts.append(verts)
                    except Exception:
                        continue
                    if sum(len(v) for v in all_verts) >= n * 3:
                        break  # enough vertices accumulated
                if not all_verts:
                    raise RuntimeError(
                        f"[{name}] Could not load any OFF files from {mesh_dir}"
                    )
                all_pts = np.vstack(all_verts)
                print(f"  [mesh] {name}: aggregated {len(all_pts)} vertices "
                      f"from {len(all_verts)} OFF files, FPS → {n}")
                return _normalize_pc(_fps(all_pts, n, seed=seed))
            else:
                raise RuntimeError(
                    f"[{name}] pcu failed and fallback only supports OFF files"
                )
        else:
            raise RuntimeError(
                f"[{name}] No mesh file found. Place a PLY or OFF file in:\n"
                f"  {os.path.join(_SCRIPT_DIR, 'meshes', name + '.ply')}\n"
                f"or set MODELNET40_DIR to your ModelNet40 path.\n"
                f"Available ModelNet40 categories include: airplane, bathtub, bed, bench,\n"
                f"  bookshelf, bottle, car, chair, cone, cup, desk, door, guitar, keyboard,\n"
                f"  lamp, laptop, monitor, person, piano, plant, sofa, table, toilet, vase"
            )

    # 2-D synthetic path
    rng = np.random.RandomState(seed)
    big = _SAMPLERS[name](n * 10, rng)
    return _fps(big, n, seed=seed)


# ─────────────────────────────────────────────
# 2.  Manifold utilities  (adapted from Experiments/augmentation_handwritten/sampling_algo.py)
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
    """∇U(Z) = (Z - k(Z)) / σ²  where U = -log GMM density."""
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
# 3.  MM-SOLD sampler
# ─────────────────────────────────────────────

# Cache: one JIT-compiled step function per M value.
_mmsold_step_cache: dict = {}


def _get_mmsold_step(M: int):
    """Return (and cache) a JIT-compiled one-step MM-SOLD function for fixed M."""
    if M in _mmsold_step_cache:
        return _mmsold_step_cache[M]

    @jax.jit
    def _step(key, Y, xi_prev, X, x_norm2, mu_tgt, L_tgt,
              sigma_gmm, s, h):
        """
        One Leimkuhler-Matthews step on the scaled Stiefel manifold.
        """
        key, k_g, k_n = jax.random.split(key, 3)
        Z = _Y_to_Z(Y, mu_tgt, L_tgt)          # (B, d)

        # ── LDS-smoothed gradient via scan over M noise samples ──────────
        if M == 0:
            # exact gradient (no smoothing)
            gZ = _gradU_batch(Z, X, x_norm2, sigma_gmm)
        else:
            def _one_noise(carry, key_m):
                Zp = Z + s * jax.random.normal(key_m, Z.shape, dtype=Z.dtype)
                return carry, _gradU_batch(Zp, X, x_norm2, sigma_gmm)

            keys_m         = jax.random.split(k_g, M)          # (M, 2)
            _, grads       = jax.lax.scan(_one_noise, None, keys_m)  # (M, B, d)
            gZ             = jnp.mean(grads, axis=0)            # (B, d)
        # ─────────────────────────────────────────────────────────────────

        gY_tan      = _proj_tangent(Y, gZ @ L_tgt.T)

        xi          = jax.random.normal(k_n, Y.shape, dtype=Y.dtype)
        xi_tan      = _proj_tangent(Y, xi)
        xi_prev_tan = _proj_tangent(Y, xi_prev)

        # Leimkuhler-Matthews update
        Y_new = (Y
                 - h * gY_tan
                 + jnp.sqrt(h / 2.0) * (xi_prev_tan + xi_tan))
        Y_new = _enforce_manifold(Y_new)
        return key, Y_new, xi          # xi becomes xi_prev next step

    _mmsold_step_cache[M] = _step
    return _step


def mmsold_sample(X_train: np.ndarray,
                  sigma: float,
                  M: int,
                  n_samples: int = N_GEN,
                  n_steps: int   = N_STEPS,
                  sigma_gmm: float = MM_SIGMA_GMM,
                  h: float         = MM_H,
                  seed: int        = SEED) -> np.ndarray:
    """
    Generate *n_samples* points using MM-SOLD.
    """
    key   = jax.random.PRNGKey(seed)
    X     = jnp.array(X_train, dtype=jnp.float32)
    n, d  = X.shape
    x_n2  = jnp.sum(X * X, axis=1)

    # Target mean and Cholesky factor
    mu_tgt   = jnp.mean(X, axis=0)
    Xc       = X - mu_tgt[None, :]
    Sigma_x  = (Xc.T @ Xc) / n
    Sigma_t  = (sigma_gmm ** 2) * jnp.eye(d) + Sigma_x
    L_tgt    = jnp.linalg.cholesky(Sigma_t)

    # Initialise particles from GMM of training data
    key, k1, k2, k3 = jax.random.split(key, 4)
    idx0   = jax.random.randint(k1, (n_samples,), 0, n)
    Z0     = X[idx0] + sigma_gmm * jax.random.normal(k2, (n_samples, d))
    Y0     = _enforce_manifold(_Z_to_Y(Z0, mu_tgt, L_tgt))
    xi_prv = jax.random.normal(k3, Y0.shape, dtype=Y0.dtype)

    step_fn = _get_mmsold_step(M)
    s_f     = jnp.float32(sigma)
    sgmm_f  = jnp.float32(sigma_gmm)
    h_f     = jnp.float32(h)

    Y, xi = Y0, xi_prv
    for _ in range(n_steps):
        key, Y, xi = step_fn(key, Y, xi, X, x_n2, mu_tgt, L_tgt,
                              sgmm_f, s_f, h_f)

    return np.array(_Y_to_Z(Y, mu_tgt, L_tgt))


# ─────────────────────────────────────────────
# 4.  σ-CFDM sampler
# ─────────────────────────────────────────────

# Cache: one JIT-compiled batch function per M value.
_scfdm_batch_cache: dict = {}


def _get_scfdm_batch_fn(M: int):
    """Return (and cache) a JIT-compiled σ-CFDM batch function for fixed M."""
    if M in _scfdm_batch_cache:
        return _scfdm_batch_cache[M]

    @partial(jax.jit, static_argnames=("n_steps",))
    def _run_batch(z, noise, X, sigma, n_steps):
        """
        Flow a batch of particles from t=0 to t=1 using the smoothed score.
        """
        h = 1.0 / n_steps

        def _outer_step(z, t_idx):
            t       = (t_idx + 1) * h           # t ∈ {h, 2h, …, (n_steps-1)·h}
            tX      = t * X                     # (n, d)
            one_mt2 = (1.0 - t) ** 2

            # Precompute ||z − tX||² for all particles (no noise perturbation yet)
            diff_z  = z[:, None, :] - tX[None, :, :]   # (B, n, d)
            dist2_z = jnp.sum(diff_z ** 2, axis=-1)     # (B, n)

            # Accumulate weighted sums over M noise samples
            def _score_one_m(carry, noise_m):
                corr    = -2.0 * sigma * t * (noise_m @ X.T)   # (n,)
                log_w   = -(dist2_z + corr[None, :]) / (2.0 * one_mt2)  # (B,n)
                log_w  -= jax.nn.logsumexp(log_w, axis=1, keepdims=True)
                w       = jnp.exp(log_w)                        # (B, n)
                return carry, w @ tX                            # (B, d)

            _, wtd_sums = jax.lax.scan(_score_one_m, None, noise)  # (M, B, d)
            avg_wtd     = jnp.mean(wtd_sums, axis=0)               # (B, d)

            # Smoothed score:  (E_ε[Σ_i w_i(z+σε) tX_i] − z) / (1−t)²
            score = (avg_wtd - z) / one_mt2                     # (B, d)
            v     = (1.0 / t) * (z + (1.0 - t) * score)        # velocity
            return z + h * v, None

        z_final, _ = jax.lax.scan(_outer_step, z, jnp.arange(n_steps - 1))
        return z_final

    _scfdm_batch_cache[M] = _run_batch
    return _run_batch


def scfdm_sample(X_train: np.ndarray,
                 sigma: float,
                 M: int,
                 n_samples: int   = N_GEN,
                 n_steps: int     = N_STEPS,
                 batch_size: int  = SCFDM_BATCH,
                 seed: int        = SEED) -> np.ndarray:
    """
    Generate *n_samples* points using σ-CFDM (plain smoothed, no NN, no DEANN).
    """
    key       = jax.random.PRNGKey(seed)
    X         = jnp.array(X_train, dtype=jnp.float32)
    n, d      = X.shape
    run_batch = _get_scfdm_batch_fn(M)

    results = []
    for b_start in range(0, n_samples, batch_size):
        b_end = min(b_start + batch_size, n_samples)
        B     = b_end - b_start

        key, k1, k2 = jax.random.split(key, 3)
        # X_0 ~ N(0, I): source distribution for rectified flow / CFDM
        z_b   = jax.random.normal(k1, (B, d))
        # Shared noise for this batch's trajectory (fixed across all 100 steps)
        noise = jax.random.normal(k2, (M, d))

        z_f = run_batch(z_b, noise, X, jnp.float32(sigma), n_steps)
        results.append(np.array(z_f))

    return np.vstack(results)


# ─────────────────────────────────────────────
# 5.  SW2 distance  (Sliced Wasserstein-2 via random projections)
# ─────────────────────────────────────────────

def sw2_distance(a: np.ndarray,
                 b: np.ndarray,
                 n_projections: int = 512,
                 seed: int = 0) -> float:
    """
    Sliced Wasserstein-2 distance.
    """
    rng = np.random.RandomState(seed)
    a   = np.asarray(a, dtype=np.float64)
    b   = np.asarray(b, dtype=np.float64)
    d   = a.shape[1]
    na, nb = len(a), len(b)

    # Random unit directions: (n_projections, d)
    dirs = rng.randn(n_projections, d)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Sorted projections: (n, n_projections)
    pa = np.sort(a @ dirs.T, axis=0)
    pb = np.sort(b @ dirs.T, axis=0)

    # Align sizes via quantile interpolation when na ≠ nb
    if na != nb:
        n  = max(na, nb)
        ta = np.linspace(0.0, 1.0, na)
        tb = np.linspace(0.0, 1.0, nb)
        t  = np.linspace(0.0, 1.0, n)
        pa = np.stack([np.interp(t, ta, pa[:, k]) for k in range(n_projections)], axis=1)
        pb = np.stack([np.interp(t, tb, pb[:, k]) for k in range(n_projections)], axis=1)

    # SW2 = sqrt( mean_over_directions( W2²_1D ) )
    sw2_sq = float(np.mean((pa - pb) ** 2))
    return float(np.sqrt(max(sw2_sq, 0.0)))


# ─────────────────────────────────────────────
# 6.  Grid-search runner
# ─────────────────────────────────────────────

def run_grid_search(dataset_name: str,
                    X_train:      np.ndarray,
                    X_target:     np.ndarray,
                    M_list:       list = M_LIST,
                    sigma_list:   list = SIGMA_LIST,
                    seed:         int  = SEED) -> dict:
    """
    Run grid search over all (M, σ) combinations for both methods.
    """
    nM, nS = len(M_list), len(sigma_list)

    res = dict(
        w2_target_scfdm  = np.full((nM, nS), np.nan),
        w2_train_scfdm   = np.full((nM, nS), np.nan),
        w2_target_mmsold = np.full((nM, nS), np.nan),
        w2_train_mmsold  = np.full((nM, nS), np.nan),
        samples_scfdm    = {},
        samples_mmsold   = {},
        M_list=M_list, sigma_list=sigma_list, dataset=dataset_name,
    )

    total = nM * nS
    idx   = 0
    for iM, M in enumerate(M_list):
        for iS, sigma in enumerate(sigma_list):
            idx += 1
            print(f"  [{dataset_name}] ({idx}/{total})  M={M}, σ={sigma}",
                  flush=True)

            # ── σ-CFDM ─────────────────────────────────────────────────
            t0 = time.time()
            samp_s = scfdm_sample(X_train, sigma, M, seed=seed)
            print(f"    σ-CFDM  {time.time()-t0:.1f}s", flush=True)

            w2t  = sw2_distance(samp_s, X_target)
            w2tr = sw2_distance(samp_s, X_train)
            res["w2_target_scfdm"][iM, iS]  = w2t
            res["w2_train_scfdm"][iM, iS]   = w2tr
            res["samples_scfdm"][(M, sigma)] = samp_s
            print(f"    σ-CFDM  SW2_target={w2t:.4f}  SW2_train={w2tr:.4f}",
                  flush=True)

            # ── MM-SOLD ────────────────────────────────────────────────
            t0 = time.time()
            samp_m = mmsold_sample(X_train, sigma, M, seed=seed)
            print(f"    MM-SOLD {time.time()-t0:.1f}s", flush=True)

            w2t  = sw2_distance(samp_m, X_target)
            w2tr = sw2_distance(samp_m, X_train)
            res["w2_target_mmsold"][iM, iS]  = w2t
            res["w2_train_mmsold"][iM, iS]   = w2tr
            res["samples_mmsold"][(M, sigma)] = samp_m
            print(f"    MM-SOLD SW2_target={w2t:.4f}  SW2_train={w2tr:.4f}",
                  flush=True)

    return res


# ─────────────────────────────────────────────
# 7.  Plotting helpers
# ─────────────────────────────────────────────

_C  = {"scfdm": "#FF4081", "mmsold": "#29B6F6"}
_LB = {"scfdm": "σ-CFDM",  "mmsold": "MM-SOLD"}


def _fig_dir(dataset_name: str) -> str:
    d = os.path.join(FIGURES_DIR, dataset_name)
    os.makedirs(d, exist_ok=True)
    return d


# ── 7a.  Line plots: fixed M, W2 vs σ  (items 1 & 2) ────────────────────────

def plot_lines_fixed_M(res: dict, dataset_name: str):
    """6 plots each for W2-to-target and W2-to-training."""
    fdir       = _fig_dir(dataset_name)
    M_list     = res["M_list"]
    sigma_list = res["sigma_list"]

    for iM, M in enumerate(M_list):
        for metric, ylabel, ks, km in [
            ("target", "W2 to target distribution",
             "w2_target_scfdm", "w2_target_mmsold"),
            ("train",  "W2 to training dataset",
             "w2_train_scfdm",  "w2_train_mmsold"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(sigma_list, res[ks][iM], "o-",
                    color=_C["scfdm"],  label=_LB["scfdm"])
            ax.plot(sigma_list, res[km][iM], "s-",
                    color=_C["mmsold"], label=_LB["mmsold"])
            ax.set_xlabel("σ")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} with M={M}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fdir,
                        f"line_{metric}_vs_sigma_M{M}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)


# ── 7b.  Line plots: fixed σ, W2 vs M  (items 3 & 4) ────────────────────────

def plot_lines_fixed_sigma(res: dict, dataset_name: str):
    """11 plots each for W2-to-target and W2-to-training."""
    fdir       = _fig_dir(dataset_name)
    M_list     = res["M_list"]
    sigma_list = res["sigma_list"]

    for iS, sigma in enumerate(sigma_list):
        for metric, ylabel, ks, km in [
            ("target", "W2 to target distribution",
             "w2_target_scfdm", "w2_target_mmsold"),
            ("train",  "W2 to training dataset",
             "w2_train_scfdm",  "w2_train_mmsold"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(M_list, res[ks][:, iS], "o-",
                    color=_C["scfdm"],  label=_LB["scfdm"])
            ax.plot(M_list, res[km][:, iS], "s-",
                    color=_C["mmsold"], label=_LB["mmsold"])
            ax.set_xlabel("M")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} with σ={sigma}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fdir,
                        f"line_{metric}_vs_M_sigma{sigma:.1f}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)


# ── 7c.  Heatmaps: % change and abs change in W2  (item 5) ──────────────────

def plot_heatmaps(res: dict, dataset_name: str):
    """
    Two heatmaps (% change and abs change in W2 to target distribution).

    change = W2(σ-CFDM) − W2(MM-SOLD)
    Positive (green) means MM-SOLD is better.
    """
    fdir  = _fig_dir(dataset_name)
    M_ls  = res["M_list"]
    S_ls  = res["sigma_list"]
    w2_s  = res["w2_target_scfdm"]   # (nM, nS)
    w2_m  = res["w2_target_mmsold"]  # (nM, nS)

    pct_change = (w2_s - w2_m) / (np.abs(w2_s) + 1e-12)
    abs_change = w2_s - w2_m

    saved = {}
    for data, title, stem in [
        (pct_change, "% change in W2",   "heatmap_pct_change"),
        (abs_change, "abs change in W2", "heatmap_abs_change"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        vmax = np.nanmax(np.abs(data))
        if vmax == 0:
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.cm.RdYlGn       # red=negative(σ-CFDM better), green=positive(MM-SOLD better)

        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                       origin="lower")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(title)

        ax.set_xticks(range(len(S_ls)))
        ax.set_xticklabels([f"{s}" for s in S_ls], rotation=45, ha="right")
        ax.set_yticks(range(len(M_ls)))
        ax.set_yticklabels([str(m) for m in M_ls])
        ax.set_xlabel("σ")
        ax.set_ylabel("M")
        ax.set_title(f"{dataset_name} — {title}\n"
                     "(positive = MM-SOLD is better)")

        # Annotate cells
        for iM in range(len(M_ls)):
            for iS in range(len(S_ls)):
                val = data[iM, iS]
                if not np.isnan(val):
                    txt = (f"{val*100:.1f}%" if "pct" in stem
                           else f"{val:.3f}")
                    ax.text(iS, iM, txt,
                            ha="center", va="center",
                            fontsize=6.5, color="black")

        fig.tight_layout()
        fig.savefig(os.path.join(fdir, f"{stem}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)
        saved[stem] = data

    return saved["heatmap_pct_change"], saved["heatmap_abs_change"]


# ── 7d.  Aggregate heatmaps across all datasets ──────────────────────────────

def plot_aggregate_heatmaps(all_pct: dict, all_abs: dict,
                             M_list: list, sigma_list: list):
    """
    Two aggregate heatmaps (mean over all datasets).
    Also return top-5 (M, σ) configurations from each.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    agg_pct = np.nanmean(np.stack(list(all_pct.values()), axis=0), axis=0)
    agg_abs = np.nanmean(np.stack(list(all_abs.values()), axis=0), axis=0)

    top5_pct = _top5_params(agg_pct, M_list, sigma_list)
    top5_abs = _top5_params(agg_abs, M_list, sigma_list)

    for data, title, stem in [
        (agg_pct, "% change in W2 (aggregate)",   "agg_heatmap_pct_change"),
        (agg_abs, "abs change in W2 (aggregate)", "agg_heatmap_abs_change"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        vmax = np.nanmax(np.abs(data))
        if vmax == 0:
            vmax = 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.cm.RdYlGn

        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                       origin="lower")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(title)

        ax.set_xticks(range(len(sigma_list)))
        ax.set_xticklabels([f"{s}" for s in sigma_list],
                           rotation=45, ha="right")
        ax.set_yticks(range(len(M_list)))
        ax.set_yticklabels([str(m) for m in M_list])
        ax.set_xlabel("σ")
        ax.set_ylabel("M")
        ax.set_title(f"All datasets (aggregate) — {title}")

        for iM in range(len(M_list)):
            for iS in range(len(sigma_list)):
                val = data[iM, iS]
                if not np.isnan(val):
                    txt = (f"{val*100:.1f}%" if "pct" in stem
                           else f"{val:.3f}")
                    ax.text(iS, iM, txt,
                            ha="center", va="center",
                            fontsize=6.5, color="black")

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"{stem}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    return top5_pct, top5_abs


def _top5_params(change_matrix: np.ndarray,
                 M_list: list,
                 sigma_list: list) -> list:
    """Return top-5 (M, σ) pairs with the largest positive change."""
    flat = change_matrix.ravel()
    flat_nan = np.where(np.isnan(flat), -np.inf, flat)
    top_idx = np.argsort(flat_nan)[::-1][:5]
    params  = []
    for fi in top_idx:
        iM, iS = np.unravel_index(fi, change_matrix.shape)
        params.append((M_list[iM], sigma_list[iS]))
    return params


# ── 7e.  Scatter plots for top-5 configurations  (item 6) ───────────────────

def _axis_limits(pts_ref, margin: float = 0.15):
    """Return (xlo, xhi, ylo, yhi) clipped to 1%-99% of pts_ref + margin."""
    xlo, xhi = np.percentile(pts_ref[:, 0], [1, 99])
    ylo, yhi = np.percentile(pts_ref[:, 1], [1, 99])
    xpad = max((xhi - xlo) * margin, 1e-3)
    ypad = max((yhi - ylo) * margin, 1e-3)
    return xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad


def _scatter_panel(ax, pts_ref, pts_gen, c_ref, c_gen, lbl_ref, lbl_gen,
                   title: str, d: int):
    """
    Draw one scatter panel (2-D or 3-D projection).
    """
    kw_gen  = dict(s=20, alpha=1.0, linewidths=0)
    kw_ref  = dict(s=30, alpha=1.0, linewidths=0)
    # Glossy highlight: small white dot offset to upper-left of each point
    kw_hi_gen = dict(s=5,  alpha=0.7, linewidths=0, zorder=kw_gen.get("zorder", 3) + 1)
    kw_hi_ref = dict(s=7,  alpha=0.7, linewidths=0, zorder=kw_ref.get("zorder", 4) + 1)

    if d == 2:
        ax.scatter(pts_gen[:, 0], pts_gen[:, 1],
                   c=c_gen, label=lbl_gen, **kw_gen, zorder=3)
        ax.scatter(pts_gen[:, 0] - 0.003, pts_gen[:, 1] + 0.003,
                   c="white", **kw_hi_gen)
        ax.scatter(pts_ref[:, 0], pts_ref[:, 1],
                   c=c_ref, label=lbl_ref, **kw_ref, zorder=4)
        ax.scatter(pts_ref[:, 0] - 0.003, pts_ref[:, 1] + 0.003,
                   c="white", **kw_hi_ref)
        xlo, xhi, ylo, yhi = _axis_limits(pts_ref)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
    else:
        # True 3-D scatter (requires ax created with projection='3d')
        ax.scatter(pts_gen[:, 0], pts_gen[:, 1], pts_gen[:, 2],
                   c=c_gen, label=lbl_gen, **kw_gen)
        ax.scatter(pts_ref[:, 0], pts_ref[:, 1], pts_ref[:, 2],
                   c=c_ref, label=lbl_ref, **kw_ref)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    ax.legend(markerscale=4, fontsize=7, loc="best")


# ── 7e-2.  Polyscope 3-D renderer  ───────────────────────────────────────────

def _pca_orient(pts_ref: np.ndarray, pts_gen: np.ndarray, upright: bool = False):
    """
    PCA-align both clouds consistently so they look natural in Polyscope y_up.
    """
    combined = np.concatenate([pts_ref, pts_gen], axis=0)
    cov = np.cov(combined.T)                          # (3,3)
    eigvals, eigvecs = np.linalg.eigh(cov)            # ascending order
    order = np.argsort(eigvals)[::-1]                 # [large, medium, small]
    if upright:
        # largest→Y (stands up), medium→X, smallest→Z
        col_order = [order[1], order[0], order[2]]
    else:
        # largest→X (lies flat), smallest→Y, medium→Z
        col_order = [order[0], order[2], order[1]]
    R = eigvecs[:, col_order].T                       # (3,3) rotation
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        R[2] *= -1
    if upright:
        pts_tmp = pts_ref @ R.T
        top_mask = pts_tmp[:, 1] > 0          # points in the +Y half
        bot_mask = ~top_mask
        spread_top = np.mean(pts_tmp[top_mask, 0] ** 2) if top_mask.any() else 0
        spread_bot = np.mean(pts_tmp[bot_mask, 0] ** 2) if bot_mask.any() else 0
        if spread_top > spread_bot:           # +Y half is wider → flip so neck is up
            R[1] *= -1
            if np.linalg.det(R) < 0:
                R[2] *= -1
        # No X-flip needed: camera is straight-on from +Z, no "front face" heuristic
    else:
        # Flat objects: ensure Y+ is "up" in the original data.
        up_axis = 1 if abs(R[1, 1]) >= abs(R[1, 2]) else 2
        if R[1, up_axis] < 0:
            R[1] *= -1
            if np.linalg.det(R) < 0:
                R[2] *= -1
        # Flip longitudinal axis so the object's front faces the camera (+Z side).
        R[0] *= -1
        if np.linalg.det(R) < 0:
            R[2] *= -1
    return pts_ref @ R.T, pts_gen @ R.T


def _render_polyscope_3d(pts_ref: np.ndarray,
                         pts_gen: np.ndarray,
                         c_ref: str,
                         c_gen: str,
                         lbl_ref: str,
                         lbl_gen: str,
                         fname: str,
                         ref_is_train: bool = False) -> bool:
    """
    Render two 3-D point clouds with Polyscope (sphere rendering + ground
    shadow) and save to *fname* (PNG).  
    """
    # Radius constants (relative to bounding-box scale ~1 after normalisation)
    _R_ALGO  = 0.003   # σ-CFDM and MM-SOLD samples: same small size
    _R_TRAIN = 0.004   # training points: slightly larger
    _ALPHA   = 1.0     # fully opaque

    try:
        import polyscope as ps
    except ImportError:
        return False

    def _hex_to_rgb01(h: str):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # Objects that naturally stand upright (longest axis = height)
    _UPRIGHT = {"guitar", "lamp", "vase", "bottle", "cone", "plant",
                "person", "cup", "flower_pot", "piano"}
    dset = os.path.basename(os.path.dirname(fname))   # e.g. "guitar"

    try:
        # PCA-align: longest axis→X (flat) or →Y (upright) depending on object
        pts_ref, pts_gen = _pca_orient(pts_ref, pts_gen, upright=(dset in _UPRIGHT))

        ps.set_allow_headless_backends(True)             # enable EGL on servers
        ps.set_up_dir("y_up")
        ps.init()
        ps.set_window_size(3200, 3200)                   # high-res screenshot
        ps.set_background_color((1.0, 1.0, 1.0, 1.0))   # white background
        ps.set_ground_plane_mode("shadow_only")           # faint ground shadow

        r_ref = _R_TRAIN if ref_is_train else _R_ALGO
        r_gen = _R_ALGO

        # Generated cloud (blue, MM-SOLD)
        gen_cloud = ps.register_point_cloud(lbl_gen, pts_gen, radius=r_gen,
                                            point_render_mode="sphere")
        gen_cloud.set_color(_hex_to_rgb01(c_gen))
        gen_cloud.set_transparency(_ALPHA)

        # Reference cloud (red – σ-CFDM or training samples)
        ref_cloud = ps.register_point_cloud(lbl_ref, pts_ref, radius=r_ref,
                                            point_render_mode="sphere")
        ref_cloud.set_color(_hex_to_rgb01(c_ref))
        ref_cloud.set_transparency(_ALPHA)

        # Camera: edit PS_CAMERA_POS / PS_CAMERA_TARGET at the top of the script
        cam_pos = PS_CAMERA_POS_UPRIGHT if (dset in _UPRIGHT) else PS_CAMERA_POS
        ps.look_at(cam_pos, PS_CAMERA_TARGET)
        ps.screenshot(fname, transparent_bg=False)
        ps.remove_all_structures()
        return True

    except Exception as e:
        print(f"  [warn] Polyscope render failed: {e}")
        try:
            ps.remove_all_structures()
        except Exception:
            pass
        return False


def plot_scatter_top5(all_results: dict,
                      X_trains: dict,
                      top5_pct: list,
                      top5_abs: list):
    n_ds   = len(all_results)
    ncols  = min(n_ds, 3)
    nrows  = (n_ds + ncols - 1) // ncols

    for hm_label, top5 in [("pct_change", top5_pct),
                             ("abs_change", top5_abs)]:
        for rank, (M, sigma) in enumerate(top5, start=1):
            for grp, c_ref, lbl_ref, key_ref in [
                ("AB", "#FF4081", "σ-CFDM",          "samples_scfdm"),
                ("CD", "#FF4081", "Training samples", "train"),
            ]:
                fig = plt.figure(figsize=(4.5*ncols, 4.0*nrows))
                # Pre-build axes list with correct projection per dataset
                ds_names = list(all_results.keys())
                axes_list = []
                for i, dn in enumerate(ds_names):
                    proj = "3d" if X_trains[dn].shape[1] == 3 else None
                    axes_list.append(
                        fig.add_subplot(nrows, ncols, i + 1,
                                        projection=proj))
                # Hide unused grid cells
                for i in range(n_ds, nrows * ncols):
                    fig.add_subplot(nrows, ncols, i + 1).set_visible(False)

                for i, (dname, res) in enumerate(all_results.items()):
                    ax  = axes_list[i]
                    d   = X_trains[dname].shape[1]
                    gen = res["samples_mmsold"].get((M, sigma))
                    ref = (res["samples_scfdm"].get((M, sigma))
                           if key_ref == "samples_scfdm"
                           else X_trains[dname])

                    if gen is None or ref is None:
                        ax.set_title(f"{dname} (missing data)", fontsize=9)
                        continue

                    _scatter_panel(ax, ref, gen,
                                   c_ref=c_ref,  c_gen="#29B6F6",
                                   lbl_ref=lbl_ref, lbl_gen="MM-SOLD",
                                   title=dname, d=d)

                fig.tight_layout()
                fname = os.path.join(
                    FIGURES_DIR,
                    f"scatter_{grp}_{hm_label}_rank{rank}.pdf")
                fig.savefig(fname, bbox_inches="tight", dpi=300)
                plt.close(fig)


# ── 7f.  Scatter plots for ALL configurations  ───────────────────────────────

def plot_scatter_all(all_results: dict, X_trains: dict):
    for dname, res in all_results.items():
        fdir  = _fig_dir(dname)
        d     = X_trains[dname].shape[1]
        M_list     = res["M_list"]
        sigma_list = res["sigma_list"]

        for M in M_list:
            for sigma in sigma_list:
                gen = res["samples_mmsold"].get((M, sigma))
                scdm = res["samples_scfdm"].get((M, sigma))
                if gen is None:
                    continue

                for grp, ref, c_ref, lbl_ref in [
                    ("vs_scfdm", scdm,              "#FF4081", "σ-CFDM"),
                    ("vs_train", X_trains[dname],   "#FF4081", "Training samples"),
                ]:
                    if ref is None:
                        continue

                    # ── Polyscope render for 3-D datasets ────────────────
                    if d == 3:
                        ps_fname = os.path.join(
                            fdir,
                            f"scatter_{grp}_M{M}_sigma{sigma}.png")
                        _render_polyscope_3d(
                            pts_ref=ref, pts_gen=gen,
                            c_ref=c_ref, c_gen="#29B6F6",
                            lbl_ref=lbl_ref, lbl_gen="MM-SOLD",
                            fname=ps_fname,
                            ref_is_train=(grp == "vs_train"))

                    # ── Matplotlib render (all datasets, also fallback) ──
                    if d == 3:
                        fig = plt.figure(figsize=(5, 5))
                        ax  = fig.add_subplot(111, projection="3d")
                    else:
                        fig, ax = plt.subplots(figsize=(5, 5))
                    _scatter_panel(ax, ref, gen,
                                   c_ref=c_ref, c_gen="#29B6F6",
                                   lbl_ref=lbl_ref, lbl_gen="MM-SOLD",
                                   title=f"M={M}, σ={sigma}", d=d)
                    fig.tight_layout()
                    fname = os.path.join(
                        fdir,
                        f"scatter_{grp}_M{M}_sigma{sigma}.pdf")
                    fig.savefig(fname, bbox_inches="tight", dpi=300)
                    plt.close(fig)


# ─────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    np.random.seed(SEED)

    RUN_DATASETS = ["guitar"]

    print("=== Preparing datasets ===")
    X_targets: dict = {}
    X_trains:  dict = {}
    for dname in RUN_DATASETS:
        X_tgt = get_dataset(dname, N_TARGET, seed=SEED)
        X_tr  = get_dataset(dname, N_TRAIN,  seed=SEED + 1)
        X_targets[dname] = X_tgt
        X_trains[dname]  = X_tr
        print(f"  {dname}: target {X_tgt.shape}, train {X_tr.shape}")

    # ── Grid search ──────────────────────────────────────────────────────────
    all_results: dict = {}
    for dname in RUN_DATASETS:
        cache_path = os.path.join(RESULTS_DIR, f"{dname}_results.pkl")
        if os.path.exists(cache_path):
            print(f"\n[{dname}] Loading cached results from {cache_path}")
            with open(cache_path, "rb") as fh:
                all_results[dname] = pickle.load(fh)
            continue

        print(f"\n=== Dataset: {dname} ===")
        res = run_grid_search(
            dname, X_trains[dname], X_targets[dname],
            M_list=M_LIST, sigma_list=SIGMA_LIST, seed=SEED,
        )
        all_results[dname] = res
        with open(cache_path, "wb") as fh:
            pickle.dump(res, fh)
        print(f"  Saved → {cache_path}")

    # ── Per-dataset plots ────────────────────────────────────────────────────
    print("\n=== Generating per-dataset plots ===")
    all_pct: dict = {}
    all_abs: dict = {}
    for dname, res in all_results.items():
        print(f"  {dname} …")
        plot_lines_fixed_M(res, dname)
        plot_lines_fixed_sigma(res, dname)
        pct, abs_ = plot_heatmaps(res, dname)
        all_pct[dname] = pct
        all_abs[dname] = abs_

    # ── Aggregate heatmaps + scatter plots (all configs) ─────────────────────
    print("\n=== Generating aggregate heatmaps & scatter plots (all configs) ===")
    top5_pct, top5_abs = plot_aggregate_heatmaps(
        all_pct, all_abs, M_LIST, SIGMA_LIST)

    print(f"  Top-5 (M,σ) by % change  : {top5_pct}")
    print(f"  Top-5 (M,σ) by abs change: {top5_abs}")

    plot_scatter_all(all_results, X_trains)

    print("\n=== All done! ===")
    print(f"  Results : {RESULTS_DIR}")
    print(f"  Figures : {FIGURES_DIR}")


if __name__ == "__main__":
    main()




