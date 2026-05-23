"""
whitening_utils.py
==================
Whitening / unwhitening in the VAE latent space, and nearest-center
distance computation.
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


def symmetric_matrix_sqrt_and_invsqrt(S: jnp.ndarray,
                                       eps: float = 1e-5,
                                       k: int = 30):
    """
    Partial whitening via eigendecomposition.

    Caps the top-k eigenvalues at the (k+1)-th value, then rescales so
    the (k+1)-th eigenvalue becomes 1.  
    """
    evals, evecs = jnp.linalg.eigh(S)       # ascending
    evals = jnp.maximum(evals, 0.0) + eps

    d           = evals.shape[0]
    evals_desc  = evals[::-1]
    tau         = evals_desc[jnp.minimum(k, d - 1)]

    capped_desc = jnp.where(jnp.arange(d) < k, tau, evals_desc)
    lam_new     = (capped_desc / tau)[::-1]      # back to ascending

    sqrt_e      = jnp.sqrt(lam_new)
    invsqrt_e   = 1.0 / sqrt_e

    S_sqrt      = (evecs * sqrt_e[None, :])    @ evecs.T
    S_invsqrt   = (evecs * invsqrt_e[None, :]) @ evecs.T
    return S_sqrt, S_invsqrt, capped_desc / tau


def whiten(Z: jnp.ndarray, mean: jnp.ndarray, S_invsqrt: jnp.ndarray) -> jnp.ndarray:
    """Map Z -> whitened coordinates: Zw = (Z - mean) @ S_invsqrt."""
    return (Z - mean[None, :]) @ S_invsqrt


def unwhiten(Zw: jnp.ndarray, mean: jnp.ndarray, S_sqrt: jnp.ndarray) -> jnp.ndarray:
    """Inverse: Zw -> Z = Zw @ S_sqrt + mean."""
    return Zw @ S_sqrt + mean[None, :]


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
