"""
cfdm_sampling.py
================
σ-CFDM: closed form diffusion model with M-sample smoothed GMM score.
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
from whitening_utils import (
    compute_sample_mean_cov,
    symmetric_matrix_sqrt_and_invsqrt,
    whiten,
    unwhiten,
)

# ── JIT cache: one compiled function per (M, shared_noise) ────────────────────
_cfdm_cache: dict = {}


def _get_batch_fn(M: int, shared_noise: bool):
    """Return (and cache) a JIT-compiled ODE function for fixed M and noise mode."""
    cache_key = (M, shared_noise)
    if cache_key in _cfdm_cache:
        return _cfdm_cache[cache_key]

    @functools.partial(jax.jit, static_argnames=("n_steps",))
    def _run_batch_shared(z, key, X, sigma, n_steps):
        """
        z   : (B, d)   initial particles
        key : PRNGKey  — fresh noise (M, d) drawn at every ODE step
        X   : (n, d)   whitened training data
        """
        h = jnp.float32(1.0 / n_steps)

        def _step(carry, t_idx):
            z, key = carry
            key, k_noise = jax.random.split(key)
            noise = jax.random.normal(k_noise, (M, z.shape[1]), dtype=jnp.float32)

            t       = (t_idx + 1) * h
            tX      = t * X
            one_mt2 = (1.0 - t) ** 2

            diff_z  = z[:, None, :] - tX[None, :, :]     # (B, n, d)
            dist2_z = jnp.sum(diff_z ** 2, axis=-1)       # (B, n)

            def _one_m(carry, noise_m):                    # noise_m: (d,)
                corr  = -2.0 * sigma * t * (noise_m @ X.T)            # (n,)
                log_w = -(dist2_z + corr[None, :]) / (2.0 * one_mt2)  # (B, n)
                log_w -= jax.nn.logsumexp(log_w, axis=1, keepdims=True)
                return carry, jnp.exp(log_w) @ tX                      # (B, d)

            _, wtd_sums = jax.lax.scan(_one_m, None, noise)   # (M, B, d)
            score = (jnp.mean(wtd_sums, axis=0) - z) / one_mt2
            v     = (z + (1.0 - t) * score) / t
            return (z + h * v, key), None

        (z_final, _), _ = jax.lax.scan(_step, (z, key), jnp.arange(n_steps - 1))
        return z_final

    @functools.partial(jax.jit, static_argnames=("n_steps",))
    def _run_batch_unshared(z, key, X, sigma, n_steps):
        """
        z   : (B, d)   initial particles
        key : PRNGKey  — fresh noise (B, M, d) drawn at every ODE step
        X   : (n, d)   whitened training data
        """
        h = jnp.float32(1.0 / n_steps)

        def _step(carry, t_idx):
            z, key = carry
            key, k_noise = jax.random.split(key)
            noise = jax.random.normal(
                k_noise, (z.shape[0], M, z.shape[1]), dtype=jnp.float32)

            t       = (t_idx + 1) * h
            tX      = t * X
            one_mt2 = (1.0 - t) ** 2

            def _score_one(z_b, noise_b):              # z_b: (d,), noise_b: (M, d)
                dist2_b = jnp.sum((z_b[None, :] - tX) ** 2, axis=-1)  # (n,)

                def _one_m(carry, nm):                 # nm: (d,)
                    corr  = -2.0 * sigma * t * (nm @ X.T)      # (n,)
                    log_w = -(dist2_b + corr) / (2.0 * one_mt2)
                    log_w -= jax.nn.logsumexp(log_w)
                    return carry, jnp.exp(log_w) @ tX           # (d,)

                _, wtd_sums = jax.lax.scan(_one_m, None, noise_b)  # (M, d)
                avg_wtd = jnp.mean(wtd_sums, axis=0)               # (d,)
                return (avg_wtd - z_b) / one_mt2                    # score

            scores = jax.vmap(_score_one)(z, noise)    # (B, d)
            v      = (z + (1.0 - t) * scores) / t
            return (z + h * v, key), None

        (z_final, _), _ = jax.lax.scan(_step, (z, key), jnp.arange(n_steps - 1))
        return z_final

    fn = _run_batch_shared if shared_noise else _run_batch_unshared
    _cfdm_cache[cache_key] = fn
    return fn


# ── Public sampler ─────────────────────────────────────────────────────────────

def sample_cfdm(
    Z_class:         np.ndarray,
    mean_class:      jnp.ndarray,
    S_sqrt_class:    jnp.ndarray,
    S_invsqrt_class: jnp.ndarray,
    n_particles:     int,
    sigma:           float,
    M:               int,
    *,
    shared_noise: bool = False,
    nsteps:       int  = 100,
    batch_size:   int  = 500,
    seed:         int  = 0,
) -> np.ndarray:
    """
    σ-CFDM sampling in NRAE latent space.
    """
    Xw = whiten(jnp.asarray(Z_class, dtype=jnp.float32),
                mean_class, S_invsqrt_class)            # (n, d)
    d = Xw.shape[1]

    run_batch = _get_batch_fn(M, shared_noise)
    key       = jax.random.PRNGKey(seed)
    results   = []

    for b_start in range(0, n_particles, batch_size):
        b_end = min(b_start + batch_size, n_particles)
        B     = b_end - b_start

        key, k1, k2 = jax.random.split(key, 3)
        z_b = jax.random.normal(k1, (B, d), dtype=jnp.float32)
        # k2 is passed as the RNG key; fresh noise is drawn at each ODE step
        z_f = run_batch(z_b, k2, Xw, jnp.float32(sigma), nsteps)
        jax.block_until_ready(z_f)
        results.append(np.array(z_f, dtype=np.float32))

    Z_gen_w = jnp.asarray(np.vstack(results), dtype=jnp.float32)
    return np.array(unwhiten(Z_gen_w, mean_class, S_sqrt_class), dtype=np.float32)


_cfdm_knn_cache: dict = {}


def _get_cfdm_knn_fn(knn_K: int, knn_L: int, Nsamp: int, d: int):
    """Build (and cache) a JIT-compiled σ-CFDM ODE function using KNN+remainder."""
    cache_key = (knn_K, knn_L, Nsamp, d)
    if cache_key in _cfdm_knn_cache:
        return _cfdm_knn_cache[cache_key]

    m = knn_K + knn_L

    @functools.partial(jax.jit, static_argnames=("n_steps",))
    def _run_batch_knn(z, key, X, x_sqnorm, sigma, n_steps):
        h   = jnp.float32(1.0 / n_steps)
        N   = X.shape[0]
        alpha = jnp.float32((N - knn_K) / knn_L)
        log_corr_nn  = jnp.zeros((knn_K,), dtype=jnp.float32)
        log_corr_rem = jnp.full((knn_L,), jnp.log(alpha), dtype=jnp.float32)
        log_corr = jnp.concatenate([log_corr_nn, log_corr_rem], axis=0)  # (m,)

        def _step(carry, t_idx):
            z, key = carry
            key, k_sel, k_mc = jax.random.split(key, 3)

            t        = jnp.float32(t_idx + 1) * h
            tX       = t * X                                   # (n, d)
            tX_sqnorm = t * t * x_sqnorm                      # (n,)
            sigma2_t = (1.0 - t) ** 2
            c2_sigma = sigma * sigma                           # = sigma^2

            B = z.shape[0]
            # ── per-particle KNN+remainder ──────────────────────────────
            # Stage 1: KNN of z among tX
            scores = 2.0 * z @ tX.T - tX_sqnorm[None, :]     # (B, N)
            _, nn_idx_batch = jax.lax.top_k(scores, knn_K)   # (B, knn_K)

            # Split keys per particle
            keys_sel = jax.random.split(k_sel, B)             # (B, 2)
            keys_mc  = jax.random.split(k_mc,  B)             # (B, 2)

            def _score_one(z_b, nn_idx, key_sel_b, key_mc_b):
                # Stage 2: random remainder
                mask = jnp.zeros((N,), dtype=bool).at[nn_idx].set(True)
                u    = jax.random.uniform(key_sel_b, (N,), dtype=jnp.float32)
                u    = jnp.where(mask, -jnp.inf, u)
                _, rand_idx = jax.lax.top_k(u, knn_L)

                sel_idx  = jnp.concatenate([nn_idx, rand_idx], axis=0)  # (m,)
                mu_sel   = tX[sel_idx]                                   # (m, d)
                sq_sel   = tX_sqnorm[sel_idx]                            # (m,)

                inner_z   = mu_sel @ z_b                                 # (m,)
                logit_const = inner_z / sigma2_t - sq_sel / (2.0 * sigma2_t) + log_corr  # (m,)

                c_std   = jnp.sqrt(c2_sigma)
                Nhalf   = Nsamp // 2
                eps     = c_std * jax.random.normal(key_mc_b, (Nhalf, d), dtype=jnp.float32)
                eps_all = jnp.concatenate([eps, -eps], axis=0)
                eta_all = eps_all @ mu_sel.T                              # (Nsamp, m)

                logits     = logit_const[None, :] + eta_all / sigma2_t   # (Nsamp, m)
                logits_max = jnp.max(logits, axis=-1, keepdims=True)
                exp_logits = jnp.exp(logits - logits_max)
                weights    = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)
                probs      = jnp.mean(weights, axis=0)                   # (m,)

                # CFDM score = (weighted_center - z) / (1-t)^2
                weighted_center = mu_sel.T @ probs                       # (d,)
                score = (weighted_center - z_b) / sigma2_t
                return score

            scores_b = jax.vmap(_score_one)(
                z, nn_idx_batch, keys_sel, keys_mc)                      # (B, d)

            v = (z + (1.0 - t) * scores_b) / t
            return (z + h * v, key), None

        (z_final, _), _ = jax.lax.scan(
            _step, (z, key), jnp.arange(n_steps - 1))
        return z_final

    _cfdm_knn_cache[cache_key] = _run_batch_knn
    return _run_batch_knn


def sample_cfdm_knn_remainder(
    Z_class:         np.ndarray,
    mean_class:      jnp.ndarray,
    S_sqrt_class:    jnp.ndarray,
    S_invsqrt_class: jnp.ndarray,
    n_particles:     int,
    sigma:           float,
    M:               int,
    *,
    knn_K:      int  = 20,
    knn_L:      int  = 20,
    nsteps:     int  = 100,
    batch_size: int  = 500,
    seed:       int  = 0,
) -> np.ndarray:
    """
    σ-CFDM sampling using the KNN+remainder score estimator.
    """
    Xw = whiten(jnp.asarray(Z_class, dtype=jnp.float32),
                mean_class, S_invsqrt_class)       # (n, d)
    n, d = Xw.shape
    x_sqnorm = jnp.sum(Xw * Xw, axis=1)           # (n,) precomputed once

    run_batch = _get_cfdm_knn_fn(knn_K, knn_L, M, d)
    key       = jax.random.PRNGKey(seed)
    results   = []

    for b_start in range(0, n_particles, batch_size):
        b_end = min(b_start + batch_size, n_particles)
        B     = b_end - b_start

        key, k1, k2 = jax.random.split(key, 3)
        z_b  = jax.random.normal(k1, (B, d), dtype=jnp.float32)
        z_f  = run_batch(z_b, k2, Xw, x_sqnorm, jnp.float32(sigma), nsteps)
        jax.block_until_ready(z_f)
        results.append(np.array(z_f, dtype=np.float32))

    Z_gen_w = jnp.asarray(np.vstack(results), dtype=jnp.float32)
    return np.array(unwhiten(Z_gen_w, mean_class, S_sqrt_class), dtype=np.float32)


# ── Whitening statistics helper ────────────────────────────────────────────────

def compute_whitening_stats(Z_class: np.ndarray, k: int = 20):
    """Compute whitening statistics for a class."""
    Z_j = jnp.asarray(Z_class, dtype=jnp.float32)
    mean_cls, cov_cls = compute_sample_mean_cov(Z_j)
    S_sqrt_cls, S_invsqrt_cls, _ = symmetric_matrix_sqrt_and_invsqrt(
        cov_cls, eps=1e-5, k=k,
    )
    return mean_cls, S_sqrt_cls, S_invsqrt_cls
