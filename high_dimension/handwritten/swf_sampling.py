"""
swf_sampling.py
===============
Sliced-Wasserstein Flow (SWF) — training-free generative baseline.
"""
import numpy as np
import jax
import jax.numpy as jnp
from whitening_utils import whiten, unwhiten

# Re-export for callers (mirrors knn_pflow_sampling.py convention)
from cfdm_sampling import compute_whitening_stats  # noqa: F401

# JIT cache keyed by (num_steps, num_projections)
_swf_cache: dict = {}


def _proj_displacement(a_l, b_l, theta_l):
    P = a_l.shape[0]
    N = b_l.shape[0]

    b_sorted = jnp.sort(b_l)                                    # (N,) sorted data
    a_ranks  = jnp.argsort(a_l)                                 # (P,) rank → orig idx
    a_sorted = a_l[a_ranks]                                     # (P,) sorted particles

    # Uniform quantile grids in [0, 1]
    p_q = jnp.arange(P, dtype=jnp.float32) / jnp.maximum(P - 1, 1)
    n_q = jnp.arange(N, dtype=jnp.float32) / jnp.maximum(N - 1, 1)

    # Interpolate data quantiles at particle rank positions
    T_sorted    = jnp.interp(p_q, n_q, b_sorted)               # (P,)
    disp_sorted = T_sorted - a_sorted                           # (P,) sorted displacements

    # Unsort: map displacements back to original particle order
    unsort_idx = jnp.argsort(a_ranks)                          # inverse permutation
    disp       = disp_sorted[unsort_idx]                        # (P,)

    return jnp.outer(disp, theta_l)                             # (P, d)


def _build_swf_fn(num_steps: int, num_projections: int):
    """Build a JIT-compiled SWF runner for fixed (num_steps, num_projections)."""

    _vmap_proj = jax.vmap(_proj_displacement, in_axes=(1, 1, 0))

    def _swf_step(Z, Xw, theta, step_size):
        a = Z  @ theta.T                            # (P, L)
        b = Xw @ theta.T                            # (N, L)
        displacements = _vmap_proj(a, b, theta)     # (L, P, d)
        delta = jnp.mean(displacements, axis=0)     # (P, d)
        return Z + step_size * delta

    @jax.jit
    def run_fn(Z_init, Xw, step_size, key):
        d = Z_init.shape[1]

        def body(Z, key_t):
            theta_raw = jax.random.normal(key_t, (num_projections, d))
            norms     = jnp.linalg.norm(theta_raw, axis=1, keepdims=True)
            theta     = theta_raw / (norms + 1e-12)   # (L, d) unit vectors
            return _swf_step(Z, Xw, theta, step_size), None

        step_keys = jax.random.split(key, num_steps)
        Z_final, _ = jax.lax.scan(body, Z_init, step_keys)
        return Z_final

    return run_fn


def _get_swf_fn(num_steps: int, num_projections: int):
    cache_key = (num_steps, num_projections)
    if cache_key not in _swf_cache:
        _swf_cache[cache_key] = _build_swf_fn(num_steps, num_projections)
    return _swf_cache[cache_key]


def sample_swf(
    Z_class:         np.ndarray,
    mean_class:      jnp.ndarray,
    S_sqrt_class:    jnp.ndarray,
    S_invsqrt_class: jnp.ndarray,
    n_particles:     int,
    num_steps:       int,
    num_projections: int,
    step_size:       float,
    init:            str,
    *,
    nn_dist:         float = None,
    seed:            int = 0,
) -> np.ndarray:
    """
    SWF sampling in NRAE latent space with partial whitening.
    """
    Xw = whiten(jnp.asarray(Z_class, dtype=jnp.float32),
                mean_class, S_invsqrt_class)               # (N, d)
    d = Xw.shape[1]

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)

    if init == "standard_normal":
        Z0 = jax.random.normal(k_init, (n_particles, d), dtype=jnp.float32)
    elif init == "train_moment_gaussian":
        # Match empirical mean and per-dimension std of whitened training data
        xw_mean = jnp.mean(Xw, axis=0)               # ≈ 0 after whitening
        xw_std  = jnp.std(Xw, axis=0) + 1e-12        # ≈ 1 after whitening
        Z0 = (xw_mean[None]
              + xw_std[None] * jax.random.normal(
                  k_init, (n_particles, d), dtype=jnp.float32))
    elif init == "gaussian_full":
        # Fitted FULL-covariance Gaussian N(mean(Xw), cov(Xw)).
        xw_mean = jnp.mean(Xw, axis=0)
        Xc = Xw - xw_mean[None, :]
        cov = (Xc.T @ Xc) / jnp.maximum(Xw.shape[0] - 1, 1)
        L = jnp.linalg.cholesky(cov + 1e-6 * jnp.eye(d, dtype=cov.dtype))
        z_std = jax.random.normal(k_init, (n_particles, d), dtype=jnp.float32)
        Z0 = xw_mean[None, :] + z_std @ L.T
    elif init == "noised_empirical":
        if nn_dist is None:
            raise ValueError("init='noised_empirical' requires nn_dist "
                             "(the eq.-77 average nearest-neighbour distance)")
        n_train = Xw.shape[0]
        k_idx, k_noise = jax.random.split(k_init)
        idx = jax.random.randint(k_idx, (n_particles,), 0, n_train)
        noise = jax.random.normal(k_noise, (n_particles, d), dtype=jnp.float32)
        Z0 = Xw[idx] + jnp.float32(nn_dist) * noise
    else:
        raise ValueError(
            f"Unknown init scheme: {init!r}  (expected one of "
            "'standard_normal', 'train_moment_gaussian', 'gaussian_full', "
            "'noised_empirical')")

    run_fn = _get_swf_fn(num_steps, num_projections)
    key, k_run = jax.random.split(key)
    step_size_jax = jnp.asarray(step_size, dtype=jnp.float32)
    Z_final_w = run_fn(Z0, Xw, step_size_jax, k_run)
    jax.block_until_ready(Z_final_w)

    return np.array(unwhiten(Z_final_w, mean_class, S_sqrt_class), dtype=np.float32)
