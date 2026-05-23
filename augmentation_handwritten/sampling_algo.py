"""
sampling_algo.py
================
Stationary overdamped Langevin dynamics on the mean/covariance manifold
in the whitened latent space.
"""
import functools
import numpy as np
import jax, jax.numpy as jnp
from jax.scipy.special import logsumexp


# ============================================================
# 1. Barycenter and gradient of the GMM energy
# ============================================================

@jax.jit
def k_batch(Z, X, x_norm2, sigma):
    """Soft-max weighted barycenter k(Z) = sum_i w_i(Z) X_i."""
    z_norm2 = jnp.sum(Z * Z, axis=1)
    dist2   = z_norm2[:, None] + x_norm2[None, :] - 2.0 * (Z @ X.T)
    logw    = -0.5 * dist2 / (sigma * sigma)
    logw   -= logsumexp(logw, axis=1, keepdims=True)
    return jnp.exp(logw) @ X


@jax.jit
def gradU_batch(Z, X, x_norm2, sigma):
    """Exact gradient of -log GMM energy."""
    return (Z - k_batch(Z, X, x_norm2, sigma)) / (sigma * sigma)


@functools.partial(jax.jit, static_argnames=("M", "shared_noise"))
def gradU_LDS_mc_stationary(key, Z, X, x_norm2, sigma, s, M,
                             shared_noise=True):
    """
    LDS-smoothed gradient: E[gradU(Z + s*eps)] via MC with M samples.
    Uses antithetic variates: draw M//2 vectors and pair with negatives,
    """
    def exact(_):
        return gradU_batch(Z, X, x_norm2, sigma)

    def mc(_):
        B, d = Z.shape
        half = M // 2

        def shared_fn(_):
            eps_h = jax.random.normal(key, (half, d), dtype=Z.dtype)
            eps   = jnp.concatenate([eps_h, -eps_h], axis=0)   # (M, d)
            return Z[:, None, :] + s * eps[None, :, :]

        def per_part_fn(_):
            eps_h = jax.random.normal(key, (B, half, d), dtype=Z.dtype)
            eps   = jnp.concatenate([eps_h, -eps_h], axis=1)   # (B, M, d)
            return Z[:, None, :] + s * eps

        Zp     = jax.lax.cond(shared_noise, shared_fn, per_part_fn, operand=None)
        g_flat = gradU_batch(Zp.reshape((-1, d)), X, x_norm2, sigma)
        return g_flat.reshape((B, M, d)).mean(axis=1)

    return jax.lax.cond(M == 0, exact, mc, operand=None)


@functools.partial(jax.jit, static_argnames=("shared_noise",))
def gradU_LDS_fixed(Z, X, x_norm2, sigma, s, eps, shared_noise=True):
    """
    LDS-smoothed gradient using pre-sampled, fixed noise eps.
    """
    B, d = Z.shape
    if shared_noise:
        Zp = Z[:, None, :] + s * eps[None, :, :]   # (B, M, d)
    else:
        Zp = Z[:, None, :] + s * eps               # (B, M, d)
    M_val  = eps.shape[0] if shared_noise else eps.shape[1]
    g_flat = gradU_batch(Zp.reshape((-1, d)), X, x_norm2, sigma)
    return g_flat.reshape((B, M_val, d)).mean(axis=1)


# ============================================================
# 2. Manifold helpers (scaled Stiefel: Y^T Y = B I, mean(Y) = 0)
# ============================================================

@jax.jit
def center_rows(Y: jnp.ndarray) -> jnp.ndarray:
    return Y - jnp.mean(Y, axis=0, keepdims=True)


@jax.jit
def proj_tangent_scaled_stiefel(Y: jnp.ndarray,
                                 G: jnp.ndarray) -> jnp.ndarray:
    """Project G onto the tangent space of the scaled Stiefel manifold at Y."""
    B   = Y.shape[0]
    sym = 0.5 * ((Y.T @ G) / B + (G.T @ Y) / B)
    return G - Y @ sym


@jax.jit
def retract_qr_scaled_stiefel(Y: jnp.ndarray) -> jnp.ndarray:
    """QR retraction: enforce Y^T Y = B I."""
    B    = Y.shape[0]
    Q, R = jnp.linalg.qr(Y, mode="reduced")
    s    = jnp.sign(jnp.diag(R))
    s    = jnp.where(s == 0.0, 1.0, s)
    return jnp.sqrt(B) * (Q * s[None, :])


@jax.jit
def enforce_mean_and_cov_manifold(Y: jnp.ndarray) -> jnp.ndarray:
    """Center + QR retraction -> mean(Y)=0, Y^T Y = B I."""
    return retract_qr_scaled_stiefel(center_rows(Y))


# ============================================================
# 3. Bijection between Z (latent) and Y (manifold coordinates)
#    Z = mu + Y L^T  <->  Y = (Z - mu) L^{-T}
# ============================================================

@jax.jit
def Z_to_Y(Z: jnp.ndarray, mu: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
    return jax.scipy.linalg.solve_triangular(L, (Z - mu[None, :]).T, lower=True).T


@jax.jit
def Y_to_Z(Y: jnp.ndarray, mu: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
    return mu[None, :] + Y @ L.T


# ============================================================
# 4. One constrained overdamped step
# ============================================================

@functools.partial(jax.jit, static_argnames=("M", "shared_noise", "discretization"))
def step_overdamped_stationary_manifold(
    key, Y, xi_prev,
    *, X, x_norm2, mu_tgt, L_tgt,
    sigma, s, h, M,
    shared_noise=True,
    discretization="LM",
):
    """
    One step of constrained overdamped Langevin:
      1. Map Y -> Z, compute LDS gradient in Z-space.
      2. Pull gradient back to Y-space via L^T.
      3. Project drift and noise onto tangent space of scaled Stiefel.
      4. Update Y and retract to manifold (center + QR).

    discretization: "EM" (Euler-Maruyama) or "LM" (Leimkuhler-Matthews).
    """
    B, d = Y.shape
    Z    = Y_to_Z(Y, mu_tgt, L_tgt)

    key, k_g, k_n = jax.random.split(key, 3)
    gZ     = gradU_LDS_mc_stationary(k_g, Z, X, x_norm2, sigma, s, M,
                                      shared_noise=shared_noise)
    gY_tan = proj_tangent_scaled_stiefel(Y, gZ @ L_tgt.T)
    xi     = jax.random.normal(k_n, (B, d), dtype=Y.dtype)
    xi_tan = proj_tangent_scaled_stiefel(Y, xi)

    if discretization == "EM":
        Y_tmp   = Y - h * gY_tan + jnp.sqrt(2.0 * h) * xi_tan
        xi_next = xi
    else:  # LM (Leimkuhler-Matthews)
        xi_prev_tan = proj_tangent_scaled_stiefel(Y, xi_prev)
        Y_tmp   = Y - h * gY_tan + jnp.sqrt(h / 2.0) * (xi_prev_tan + xi_tan)
        xi_next = xi

    return key, enforce_mean_and_cov_manifold(Y_tmp), xi_next


# ============================================================
# 5. Per-class sampling runner
# ============================================================

def sample_class_overdamped_manifold(
    *,
    Z_class: jnp.ndarray,
    mean_class: jnp.ndarray,
    S_sqrt_class: jnp.ndarray,
    S_invsqrt_class: jnp.ndarray,
    n_particles: int = 800,
    nsteps: int = 50,
    h: float = 1e-4,
    sigma_gmm: float = 0.05,
    sigma_smoothing: float = 2.0,
    M: int = 32,
    shared_noise: bool = False,
    discretization: str = "LM",
    fixed_noise: bool = False,
    seed: int = 0,
):
    """
    Run manifold-constrained overdamped Langevin for one digit class.
    """
    from whitening_utils import whiten

    Xw       = whiten(Z_class, mean_class, S_invsqrt_class)   # (n_class, d)
    d        = Xw.shape[1]
    xw_norm2 = jnp.sum(Xw * Xw, axis=1)

    # Build target mean / Cholesky factor for stationary GMM
    xbar      = jnp.mean(Xw, axis=0)
    Xc        = Xw - xbar[None, :]
    Sigma_x   = (Xc.T @ Xc) / Xw.shape[0]
    Sigma_tgt = (sigma_gmm ** 2) * jnp.eye(d, dtype=Xw.dtype) + Sigma_x
    L_tgt     = jnp.linalg.cholesky(Sigma_tgt)

    # Initialise particles from GMM
    key = jax.random.PRNGKey(seed)
    key, k_idx, k_eps, k_xi = jax.random.split(key, 4)
    idx0    = jax.random.randint(k_idx, (n_particles,), 0, Xw.shape[0])
    Zp0     = Xw[idx0] + sigma_gmm * jax.random.normal(k_eps, (n_particles, d))
    Y0      = enforce_mean_and_cov_manifold(Z_to_Y(Zp0, xbar, L_tgt))
    xi_prev = jax.random.normal(k_xi, (n_particles, d), dtype=Y0.dtype)

    # Langevin loop via fori_loop (efficient with JIT)
    if fixed_noise:
        # Sample M LDS noise vectors once; reuse across all Langevin steps
        key, k_fn = jax.random.split(key)
        eps_shape = (M, d) if shared_noise else (n_particles, M, d)
        eps_fixed = jax.random.normal(k_fn, eps_shape, dtype=Y0.dtype)

        def body(i, carry):
            key, Y, xi = carry
            Z    = Y_to_Z(Y, xbar, L_tgt)
            gZ   = gradU_LDS_fixed(Z, Xw, xw_norm2, sigma_gmm,
                                   sigma_smoothing, eps_fixed,
                                   shared_noise=shared_noise)
            gY_tan    = proj_tangent_scaled_stiefel(Y, gZ @ L_tgt.T)
            key, k_n  = jax.random.split(key)
            xi_new    = jax.random.normal(k_n, (n_particles, d), dtype=Y.dtype)
            xi_tan    = proj_tangent_scaled_stiefel(Y, xi_new)
            if discretization == "EM":
                Y_tmp = Y - h * gY_tan + jnp.sqrt(2.0 * h) * xi_tan
                xi_next = xi_new
            else:  # LM
                xi_prev_tan = proj_tangent_scaled_stiefel(Y, xi)
                Y_tmp = Y - h * gY_tan + jnp.sqrt(h / 2.0) * (xi_prev_tan + xi_tan)
                xi_next = xi_new
            return (key, enforce_mean_and_cov_manifold(Y_tmp), xi_next)
    else:
        def body(i, carry):
            key, Y, xi = carry
            key, Y, xi = step_overdamped_stationary_manifold(
                key, Y, xi,
                X=Xw, x_norm2=xw_norm2, mu_tgt=xbar, L_tgt=L_tgt,
                sigma=sigma_gmm, s=sigma_smoothing, h=h, M=M,
                shared_noise=shared_noise, discretization=discretization,
            )
            return (key, Y, xi)

    _, YT, _ = jax.lax.fori_loop(0, nsteps, body, (key, Y0, xi_prev))

    z_sampled_w = Y_to_Z(YT, xbar, L_tgt)   # (n_particles, d)
    return z_sampled_w, Xw


# ============================================================
# 6. KNN+remainder score estimator
# ============================================================

@functools.partial(jax.jit, static_argnames=("knn_K", "knn_L", "Nsamp", "antithetic"))
def _knn_rem_score_batched(
    xs,           # (B, d)  query particles
    mus,          # (n, d)  training centers (whitened)
    mus_sqnorm,   # (n,)    precomputed ||mu_i||^2
    sigma2,       # float   GMM component variance  (sigma_gmm^2)
    c2,           # float   smoothing variance       (sigma_smoothing^2)
    keys,         # (B, 2)  per-particle PRNGKeys
    knn_K,        # int (static)  deterministic nearest neighbors
    knn_L,        # int (static)  random-remainder samples
    Nsamp=256,    # int (static)  MC samples per score call
    antithetic=True,  # bool (static)
):
    """
    Batched KNN+remainder estimator of grad_z log(p * phi_{c2 I})(z).
    """
    N = mus.shape[0]
    d = xs.shape[1]
    m = knn_K + knn_L

    # Stage 1: find K nearest neighbors for all B particles at once
    scores_batch = 2.0 * xs @ mus.T - mus_sqnorm[None, :]   # (B, N)
    _, nn_idx_batch = jax.lax.top_k(scores_batch, knn_K)    # (B, knn_K)

    alpha = (N - knn_K) / knn_L
    log_corr = jnp.concatenate([
        jnp.zeros((knn_K,), dtype=mus.dtype),
        jnp.full((knn_L,), jnp.log(alpha), dtype=mus.dtype),
    ], axis=0)  # (m,)

    def _single(x, nn_idx, key):
        key_sel, key_mc = jax.random.split(key)

        # Stage 2: sample L from complement
        mask = jnp.zeros((N,), dtype=bool).at[nn_idx].set(True)
        u    = jax.random.uniform(key_sel, (N,), dtype=jnp.float32)
        u    = jnp.where(mask, -jnp.inf, u)
        _, rand_idx = jax.lax.top_k(u, knn_L)

        sel_idx    = jnp.concatenate([nn_idx, rand_idx], axis=0)   # (m,)
        mu_sel     = mus[sel_idx]                                   # (m, d)
        sq_sel     = mus_sqnorm[sel_idx]                            # (m,)

        inner_x     = mu_sel @ x                                    # (m,)
        logit_const = inner_x / sigma2 - sq_sel / (2.0 * sigma2) + log_corr  # (m,)

        # Noise sampling: m-dim Cholesky if m < d, else d-dim projection
        # Regularization: 1e-4 relative to trace(G)/m to stay safe in float32
        if m < d:
            G       = mu_sel @ mu_sel.T
            reg     = jnp.maximum(1e-6, 1e-4 * jnp.mean(jnp.diag(G)))
            cov_eta = c2 * G + reg * jnp.eye(m, dtype=mu_sel.dtype)
            L_eta   = jnp.linalg.cholesky(cov_eta)
            if antithetic:
                Nhalf   = Nsamp // 2
                z_samp  = jax.random.normal(key_mc, (Nhalf, m), dtype=mu_sel.dtype)
                eta_pos = z_samp @ L_eta.T
                eta_all = jnp.concatenate([eta_pos, -eta_pos], axis=0)
            else:
                z_samp  = jax.random.normal(key_mc, (Nsamp, m), dtype=mu_sel.dtype)
                eta_all = z_samp @ L_eta.T
        else:
            c_std = jnp.sqrt(c2)
            if antithetic:
                Nhalf   = Nsamp // 2
                eps     = c_std * jax.random.normal(key_mc, (Nhalf, d), dtype=mu_sel.dtype)
                eps_all = jnp.concatenate([eps, -eps], axis=0)
            else:
                eps_all = c_std * jax.random.normal(key_mc, (Nsamp, d), dtype=mu_sel.dtype)
            eta_all = eps_all @ mu_sel.T                            # (Nsamp, m)

        logits     = logit_const[None, :] + eta_all / sigma2       # (Nsamp, m)
        logits_max = jnp.max(logits, axis=-1, keepdims=True)
        exp_logits = jnp.exp(logits - logits_max)
        weights    = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)
        probs      = jnp.mean(weights, axis=0)                     # (m,)
        grad       = (mu_sel.T @ probs - x) / sigma2               # (d,)
        return grad, probs

    return jax.vmap(_single)(xs, nn_idx_batch, keys)


@functools.partial(jax.jit, static_argnames=("knn_K", "knn_L", "M", "discretization"))
def _step_overdamped_knn_manifold(
    key, Y, xi_prev, *,
    X, x_sqnorm, mu_tgt, L_tgt,
    sigma_gmm, sigma_smoothing, h,
    knn_K, knn_L, M,
    discretization="LM",
):
    """
    One manifold-constrained Langevin step using the KNN+remainder
    score estimator instead of the full-softmax antithetic-MC estimator.
    """
    B, d = Y.shape
    Z    = Y_to_Z(Y, mu_tgt, L_tgt)

    key, k_g, k_n = jax.random.split(key, 3)

    # Generate one PRNGKey per particle for the batched estimator
    keys_batch = jax.random.split(k_g, B)          # (B, 2)

    # Score estimate: grad = (weighted_center - Z_i) / sigma_gmm^2
    grads, _ = _knn_rem_score_batched(
        Z, X, x_sqnorm,
        sigma2=sigma_gmm ** 2,
        c2=sigma_smoothing ** 2,
        keys=keys_batch,
        knn_K=knn_K, knn_L=knn_L,
        Nsamp=M, antithetic=True,
    )
    gZ     = -grads                                 # gradU = -score

    gY_tan = proj_tangent_scaled_stiefel(Y, gZ @ L_tgt.T)
    xi     = jax.random.normal(k_n, (B, d), dtype=Y.dtype)
    xi_tan = proj_tangent_scaled_stiefel(Y, xi)

    if discretization == "EM":
        Y_tmp   = Y - h * gY_tan + jnp.sqrt(2.0 * h) * xi_tan
        xi_next = xi
    else:  # LM (Leimkuhler-Matthews)
        xi_prev_tan = proj_tangent_scaled_stiefel(Y, xi_prev)
        Y_tmp   = Y - h * gY_tan + jnp.sqrt(h / 2.0) * (xi_prev_tan + xi_tan)
        xi_next = xi

    return key, enforce_mean_and_cov_manifold(Y_tmp), xi_next


# ============================================================
# 7. Per-class sampling runner — KNN+remainder variant
# ============================================================

def sample_class_overdamped_manifold_knn(
    *,
    Z_class: jnp.ndarray,
    mean_class: jnp.ndarray,
    S_sqrt_class: jnp.ndarray,
    S_invsqrt_class: jnp.ndarray,
    n_particles: int = 800,
    nsteps: int = 100,
    h: float = 5e-4,
    sigma_gmm: float = 0.03,
    sigma_smoothing: float = 0.5,
    M: int = 32,
    knn_K: int = 20,
    knn_L: int = 20,
    discretization: str = "LM",
    seed: int = 0,
):
    """
    Manifold-constrained overdamped Langevin sampling using the
    KNN+remainder score estimator.
    """
    from whitening_utils import whiten

    Xw       = whiten(Z_class, mean_class, S_invsqrt_class)   # (n_class, d)
    d        = Xw.shape[1]
    xw_sqnorm = jnp.sum(Xw * Xw, axis=1)                     # (n_class,)

    # Build target mean / Cholesky factor for stationary GMM
    xbar      = jnp.mean(Xw, axis=0)
    Xc        = Xw - xbar[None, :]
    Sigma_x   = (Xc.T @ Xc) / Xw.shape[0]
    Sigma_tgt = (sigma_gmm ** 2) * jnp.eye(d, dtype=Xw.dtype) + Sigma_x
    L_tgt     = jnp.linalg.cholesky(Sigma_tgt)

    # Initialise particles
    key = jax.random.PRNGKey(seed)
    key, k_idx, k_eps, k_xi = jax.random.split(key, 4)
    idx0    = jax.random.randint(k_idx, (n_particles,), 0, Xw.shape[0])
    Zp0     = Xw[idx0] + sigma_gmm * jax.random.normal(k_eps, (n_particles, d))
    Y0      = enforce_mean_and_cov_manifold(Z_to_Y(Zp0, xbar, L_tgt))
    xi_prev = jax.random.normal(k_xi, (n_particles, d), dtype=Y0.dtype)

    def body(i, carry):
        key, Y, xi = carry
        key, Y, xi = _step_overdamped_knn_manifold(
            key, Y, xi,
            X=Xw, x_sqnorm=xw_sqnorm,
            mu_tgt=xbar, L_tgt=L_tgt,
            sigma_gmm=sigma_gmm,
            sigma_smoothing=sigma_smoothing,
            h=h, M=M,
            knn_K=knn_K, knn_L=knn_L,
            discretization=discretization,
        )
        return (key, Y, xi)

    _, YT, _ = jax.lax.fori_loop(0, nsteps, body, (key, Y0, xi_prev))

    z_sampled_w = Y_to_Z(YT, xbar, L_tgt)
    return z_sampled_w, Xw
