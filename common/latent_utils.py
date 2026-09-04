"""
latent_utils.py
===============
Latent-space helpers: sample mean/covariance and nearest-center distances.
"""
import functools
import numpy as np
import jax, jax.numpy as jnp
from jax.scipy.special import logsumexp


def compute_sample_mean_cov(Z: jnp.ndarray):
    """Z: (K,d) -> mean (d,), cov (d,d) [unbiased, 1/(K-1)]."""
    K = Z.shape[0]
    m  = jnp.mean(Z, axis=0)
    Zc = Z - m[None, :]
    cov = (Zc.T @ Zc) / jnp.maximum(K - 1, 1)
    return m, cov


@functools.partial(jax.jit, static_argnames=("chunk",))
def nearest_center_distances(samples: jnp.ndarray,
                              centers: jnp.ndarray,
                              chunk: int = 256) -> jnp.ndarray:
    """
    Chunked nearest-neighbor distance (avoids O(B*N) memory).
    samples: (B,d), centers: (N,d)  ->  (B,) min Euclidean distances.
    """
    B, d    = samples.shape
    N       = centers.shape[0]
    chunk   = min(chunk, N)
    s_norm2 = jnp.sum(samples * samples, axis=1, keepdims=True)
    best    = jnp.full((B,), jnp.inf, dtype=samples.dtype)
    nblocks = (N + chunk - 1) // chunk

    def body(bi, best):
        j0      = bi * chunk
        C       = jax.lax.dynamic_slice(centers, (j0, 0), (chunk, d))
        c_norm2 = jnp.sum(C * C, axis=1)[None, :]
        dist2   = s_norm2 + c_norm2 - 2.0 * (samples @ C.T)
        valid   = ((j0 + jnp.arange(chunk)) < N)[None, :]
        dist2   = jnp.where(valid, dist2, jnp.inf)
        return jnp.minimum(best, jnp.min(jnp.maximum(dist2, 0.0), axis=1))

    return jnp.sqrt(jax.lax.fori_loop(0, nblocks, body, best))
