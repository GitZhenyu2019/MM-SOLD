"""
knn_pflow_sampling.py
=====================
KNN-score probability-flow ODE (training-free baseline).

Reference: Niedoba et al. (ICML 2024) "Nearest Neighbour Score Estimators
           for Diffusion Generative Models"
"""
import numpy as np
import jax
import jax.numpy as jnp
from whitening_utils import whiten, unwhiten

_knn_pflow_cache: dict = {}


def _make_sigma_schedule(
    sigma_max: float,
    sigma_min: float,
    num_steps: int,
    rho: float = 7.0,
) -> np.ndarray:
    """
    Karras rho-schedule, descending: σ[0]=sigma_max … σ[num_steps]=sigma_min.
    """
    t = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float64)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho
              + t * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    return sigmas.astype(np.float32)


def _build_fns(K_eff: int, solver: str):
    """
    Build JIT-compiled (ode_fn, denoise_fn) for a fixed (K_eff, solver).
    """
    use_knn = (K_eff > 0)

    # ── single-particle denoiser ──────────────────────────────────────────
    if use_knn:
        def _den_one(z_b, X, X_sqnorm, sigma):
            # Use 2z·x - ||x||² as proxy for -||z-x||² (||z||² is constant)
            inner  = 2.0 * (z_b @ X.T) - X_sqnorm         # (N,)
            _, idx = jax.lax.top_k(inner, K_eff)           # (K,)
            X_sel  = X[idx]                                  # (K, d)
            diff   = z_b[None, :] - X_sel                   # (K, d)
            logits = -jnp.sum(diff ** 2, axis=-1) / (2.0 * sigma ** 2)
            logits -= jnp.max(logits)                        # numerical stability
            w = jnp.exp(logits)
            w /= jnp.sum(w)
            return X_sel.T @ w                               # (d,)
    else:
        def _den_one(z_b, X, X_sqnorm, sigma):
            inner  = 2.0 * (z_b @ X.T) - X_sqnorm          # (N,)
            logits = inner / (2.0 * sigma ** 2)
            logits -= jnp.max(logits)                        # numerical stability
            w = jnp.exp(logits)
            w /= jnp.sum(w)
            return X.T @ w                                   # (d,)

    # batch denoiser: (B,d),(N,d),(N,),(scalar) → (B,d)
    _den_batch = jax.vmap(_den_one, in_axes=(0, None, None, None))

    # ── ODE function ──────────────────────────────────────────────────────
    if solver == "euler":
        @jax.jit
        def _ode_fn(z, X, X_sqnorm, sigmas):
            def _step(z, step_pair):
                sigma_cur  = step_pair[0]
                sigma_next = step_pair[1]
                # PF-ODE: dz/dσ = (z - D(z,σ)) / σ
                # Discrete Euler: Δz = Δσ · (z - D) / σ_cur   (Δσ < 0)
                D  = _den_batch(z, X, X_sqnorm, sigma_cur)      # (B, d)
                dz = (z - D) / sigma_cur
                z  = z + (sigma_next - sigma_cur) * dz
                return z, None

            step_pairs = jnp.stack([sigmas[:-1], sigmas[1:]], axis=1)
            z_final, _ = jax.lax.scan(_step, z, step_pairs)
            return z_final

    else:  # "heun"
        @jax.jit
        def _ode_fn(z, X, X_sqnorm, sigmas):
            def _step(z, step_pair):
                sigma_cur  = step_pair[0]
                sigma_next = step_pair[1]
                dsigma = sigma_next - sigma_cur                 # negative

                # Predictor (Euler)
                D_cur  = _den_batch(z, X, X_sqnorm, sigma_cur)  # (B, d)
                dz_cur = (z - D_cur) / sigma_cur
                z_pred = z + dsigma * dz_cur

                # Corrector (Heun)
                D_next  = _den_batch(z_pred, X, X_sqnorm, sigma_next)
                dz_next = (z_pred - D_next) / sigma_next
                z = z + dsigma * 0.5 * (dz_cur + dz_next)
                return z, None

            step_pairs = jnp.stack([sigmas[:-1], sigmas[1:]], axis=1)
            z_final, _ = jax.lax.scan(_step, z, step_pairs)
            return z_final

    # jit-wrapped for standalone use in final_denoise
    _den_batch_jit = jax.jit(_den_batch)

    return _ode_fn, _den_batch_jit


def _get_fns(K_eff: int, solver: str):
    key = (K_eff, solver)
    if key not in _knn_pflow_cache:
        _knn_pflow_cache[key] = _build_fns(K_eff, solver)
    return _knn_pflow_cache[key]


# ── Public sampler ─────────────────────────────────────────────────────────────

def sample_knn_pflow(
    Z_class:         np.ndarray,
    mean_class:      jnp.ndarray,
    S_sqrt_class:    jnp.ndarray,
    S_invsqrt_class: jnp.ndarray,
    n_particles:     int,
    sigma_max:       float,
    sigma_min:       float,
    num_steps:       int,
    K:               int,
    solver:          str   = "euler",
    *,
    rho:             float = 7.0,
    final_denoise:   bool  = True,
    batch_size:      int   = 500,
    seed:            int   = 0,
) -> np.ndarray:
    """
    KNN-score PF-ODE sampling in NRAE latent space.
    """
    Xw = whiten(jnp.asarray(Z_class, dtype=jnp.float32),
                mean_class, S_invsqrt_class)                    # (n, d)
    n, d = Xw.shape
    X_sqnorm = jnp.sum(Xw * Xw, axis=1)                        # (n,)

    K_eff = min(K, n) if K > 0 else 0
    sigmas_jax = jnp.asarray(
        _make_sigma_schedule(sigma_max, sigma_min, num_steps, rho))  # (T+1,)

    ode_fn, den_fn = _get_fns(K_eff, solver)
    key = jax.random.PRNGKey(seed)
    results = []

    for b_start in range(0, n_particles, batch_size):
        b_end = min(b_start + batch_size, n_particles)
        B     = b_end - b_start

        key, k_init = jax.random.split(key)
        # Initialise at sigma_max: z ~ N(0, sigma_max² I)
        z_b = sigma_max * jax.random.normal(k_init, (B, d), dtype=jnp.float32)

        z_f = ode_fn(z_b, Xw, X_sqnorm, sigmas_jax)

        if final_denoise:
            sigma_last = sigmas_jax[-1]                         # scalar jax array
            z_f = den_fn(z_f, Xw, X_sqnorm, sigma_last)

        jax.block_until_ready(z_f)
        results.append(np.array(z_f, dtype=np.float32))

    Z_gen_w = jnp.asarray(np.vstack(results), dtype=jnp.float32)
    return np.array(unwhiten(Z_gen_w, mean_class, S_sqrt_class), dtype=np.float32)


# ── Re-export for convenience ──────────────────────────────────────────────────
from cfdm_sampling import compute_whitening_stats  # noqa: E402
