"""
tilting_lion.py
===============
Estimate the tilting parameters (lambda, Lambda) of the moment-matched target of
Proposition 1
"""
import functools
import os
import sys
import time

import numpy as np
import scipy.linalg

import jax
import jax.numpy as jnp
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "MM-SOLD", "augmentation_handwritten"))

from sampling_algo import (  # noqa: E402
    gradU_LDS_mc_stationary, _knn_rem_score_batched,
)
from latent_utils import compute_sample_mean_cov  # noqa: E402


@functools.partial(jax.jit, static_argnames=("knn_K", "knn_L", "Nsamp", "antithetic"))
def _knn_rem_score(xs, mus, mus_sqnorm, sigma2, c2, keys,
                         knn_K, knn_L, Nsamp=256, antithetic=True):
    N = mus.shape[0]
    d = xs.shape[1]
    m = knn_K + knn_L

    scores_batch = 2.0 * xs @ mus.T - mus_sqnorm[None, :]
    _, nn_idx_batch = jax.lax.top_k(scores_batch, knn_K)

    alpha = (N - knn_K) / knn_L
    log_corr = jnp.concatenate([
        jnp.zeros((knn_K,), dtype=mus.dtype),
        jnp.full((knn_L,), jnp.log(alpha), dtype=mus.dtype)], axis=0)

    def _single(x, nn_idx, key):
        key_sel, key_mc = jax.random.split(key)
        mask = jnp.zeros((N,), dtype=bool).at[nn_idx].set(True)
        u = jax.random.uniform(key_sel, (N,), dtype=jnp.float32)
        u = jnp.where(mask, -jnp.inf, u)
        _, rand_idx = jax.lax.top_k(u, knn_L)

        sel_idx = jnp.concatenate([nn_idx, rand_idx], axis=0)
        mu_sel = mus[sel_idx]
        sq_sel = mus_sqnorm[sel_idx]
        logit_const = mu_sel @ x / sigma2 - sq_sel / (2.0 * sigma2) + log_corr

        if m < d:
            # mu_sel.T = Q R gives G = R^T R (also when G is singular),
            # so eta = z @ L_eta^T has covariance exactly c2 * G.
            _, R = jnp.linalg.qr(mu_sel.T, mode="reduced")
            L_eta = jnp.sqrt(c2) * R.T
            if antithetic:
                z_samp = jax.random.normal(key_mc, (Nsamp // 2, m), dtype=mu_sel.dtype)
                eta_all = jnp.concatenate([z_samp @ L_eta.T, -(z_samp @ L_eta.T)], axis=0)
            else:
                eta_all = jax.random.normal(key_mc, (Nsamp, m), dtype=mu_sel.dtype) @ L_eta.T
        else:
            c_std = jnp.sqrt(c2)
            if antithetic:
                eps = c_std * jax.random.normal(key_mc, (Nsamp // 2, d), dtype=mu_sel.dtype)
                eps_all = jnp.concatenate([eps, -eps], axis=0)
            else:
                eps_all = c_std * jax.random.normal(key_mc, (Nsamp, d), dtype=mu_sel.dtype)
            eta_all = eps_all @ mu_sel.T

        logits = logit_const[None, :] + eta_all / sigma2
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        w = jnp.exp(logits)
        w = w / jnp.sum(w, axis=-1, keepdims=True)
        probs = jnp.mean(w, axis=0)
        return (mu_sel.T @ probs - x) / sigma2

    return jax.vmap(_single)(xs, nn_idx_batch, keys)


def score_grad(key, q, X, x_norm2, delta, sigma_s, M,
               score="full", knn_K=50, knn_L=50):
    """
    grad V(q) for a batch of states q: (B, d) -> (B, d).
    """
    if score == "full":
        return gradU_LDS_mc_stationary(key, q, X, x_norm2, delta, sigma_s, M,
                                       shared_noise=False)
    return -_knn_rem_score(
        q, X, x_norm2, sigma2=delta ** 2, c2=sigma_s ** 2,
        keys=jax.random.split(key, q.shape[0]),
        knn_K=knn_K, knn_L=knn_L, Nsamp=M, antithetic=True)


# ═══════════════════════════════════════════════════════════════════════════
# 0.  Working space: top-k principal subspace (no whitening)
# ═════════════════════════════════════════════════════════════════════════

def projection_stats(Z_class, k=400):
    """
    Top-k principal subspace of the class covariance.  No whitening.
    """
    Z_j = jnp.asarray(Z_class, dtype=jnp.float32)
    mean, cov = compute_sample_mean_cov(Z_j)
    evals, evecs = jnp.linalg.eigh(cov)            # ascending
    order = jnp.arange(evals.shape[0] - 1, -1, -1)  # -> descending
    V = evecs[:, order][:, :k]
    return mean, V, evals[order][:k]


def pooled_projection_stats(Z, labels, k=40):
    """
    ONE shared top-k basis for every class, from the pooled WITHIN-class
    covariance (each class centred on its own mean).
    """
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels)
    d = Z.shape[1]
    cov = np.zeros((d, d), dtype=np.float64)
    class_means, n_cls = {}, 0
    for c in np.unique(labels):
        Zi = Z[labels == c]
        class_means[int(c)] = Zi.mean(0)
        Zc = Zi - class_means[int(c)]
        cov += Zc.T @ Zc
        n_cls += 1
    cov /= max(len(Z) - n_cls, 1)
    ev, V = np.linalg.eigh(cov)
    return (jnp.asarray(V[:, ::-1][:, :k], jnp.float32),
            jnp.asarray(ev[::-1][:k], jnp.float32),
            {c: jnp.asarray(m, jnp.float32) for c, m in class_means.items()})


def to_subspace(Z, mean, V):
    """NRAE latents -> top-k subspace coordinates (where estimation runs)."""
    return np.asarray((jnp.asarray(Z, jnp.float32) - mean[None, :]) @ V)


def from_subspace(U, mean, V):
    """Top-k coordinates -> NRAE latents (do this before decoding).

    The d-k dropped directions come back as the class mean.
    """
    return np.asarray(jnp.asarray(U, jnp.float32) @ V.T + mean[None, :])


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Moment targets, integrability floor, plug-in warm start
# ═══════════════════════════════════════════════════════════════════════════

def moment_targets(X, delta):
    """mu* = mean(X),  Sigma* = delta^2 I + cov(X)  (moments of the GMM)."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu[None, :]
    Sigma = (Xc.T @ Xc) / n + delta ** 2 * np.eye(d)
    return mu.astype(np.float32), Sigma.astype(np.float32)


def lambda_floor(delta, eps=0.05):
    """
    Integrability bound for Lambda.
    """
    return -(1.0 - eps) / (delta ** 2)


def plugin_tilting(key, X, delta, sigma_s, M, zeta_rel=1e-6,
                   score="full", knn_K=50, knn_L=50, chunk=2048):
    """
    plug-in estimator: training-set averages of g
    plus a Lyapunov solve.  Used as the Lion warm start.
    """
    Xw_np = np.asarray(X, dtype=np.float64)
    n, d = Xw_np.shape
    mu, Sigma = moment_targets(Xw_np, delta)
    mu64, Sigma64 = mu.astype(np.float64), Sigma.astype(np.float64)

    Xw_j = jnp.asarray(X, dtype=jnp.float32)
    x_norm2 = jnp.sum(Xw_j * Xw_j, axis=1)
    Gs = []
    for a in range(0, n, chunk):
        key, k = jax.random.split(key)
        Gs.append(np.asarray(score_grad(k, Xw_j[a:a + chunk], Xw_j, x_norm2,
                                        delta, sigma_s, M, score=score,
                                        knn_K=knn_K, knn_L=knn_L),
                             dtype=np.float64))
    G = np.concatenate(Gs, axis=0)                               # (n, d)

    lam = -G.mean(axis=0)
    C = (Xw_np - mu64[None, :]).T @ G / n
    S = np.eye(d) - 0.5 * (C + C.T)

    reg = max(float(np.linalg.eigvalsh(Sigma64).max()) * zeta_rel, 1e-6)
    Lam = scipy.linalg.solve_continuous_lyapunov(Sigma64 + reg * np.eye(d), 2.0 * S)
    return lam.astype(np.float32), np.asarray(Lam, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  P parallel SG-BAOAB chains (vmap), restarted from the training GMM
# ═══════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=(
    "M", "n_steps", "n_chains", "splitting", "score", "knn_K", "knn_L"))
def run_chains_batched(key, lam, Lam, X, x_norm2, mu_star,
                       delta, sigma_s, h, gamma, M, n_steps, n_chains,
                       splitting="BAOAB", score="knn", knn_K=50, knn_L=50):
    d = X.shape[1]
    alpha = jnp.exp(-gamma * h)
    c = jnp.sqrt(1.0 - alpha * alpha)

    key, k_idx, k_eps, k_p, key_run = jax.random.split(key, 5)
    idx = jax.random.randint(k_idx, (n_chains,), 0, X.shape[0])
    q = X[idx] + delta * jax.random.normal(k_eps, (n_chains, d), dtype=X.dtype)
    p = jax.random.normal(k_p, (n_chains, d), dtype=X.dtype)

    def gradU(q, k):
        g = score_grad(k, q, X, x_norm2, delta, sigma_s, M,
                       score=score, knn_K=knn_K, knn_L=knn_L)
        return g + lam[None, :] + (q - mu_star[None, :]) @ Lam

    if splitting == "BAOAB":
        key_run, k0 = jax.random.split(key_run)
        g0 = gradU(q, k0)

        def body(carry, _):
            q, p, g, key = carry
            p = p - (h * 0.5) * g
            q = q + (h * 0.5) * p
            key, k_o = jax.random.split(key)
            p = alpha * p + c * jax.random.normal(k_o, p.shape, p.dtype)
            q = q + (h * 0.5) * p
            key, k_g = jax.random.split(key)
            g = gradU(q, k_g)
            p = p - (h * 0.5) * g
            return (q, p, g, key), None

        (q, _, _, _), _ = jax.lax.scan(body, (q, p, g0, key_run), None, length=n_steps)
    else:
        def body(carry, _):
            q, p, key = carry
            q = q + (h * 0.5) * p
            key, k_g = jax.random.split(key)
            g = gradU(q, k_g)
            p = p - (h * 0.5) * g
            key, k_o = jax.random.split(key)
            p = alpha * p + c * jax.random.normal(k_o, p.shape, p.dtype)
            p = p - (h * 0.5) * g
            q = q + (h * 0.5) * p
            return (q, p, key), None

        (q, _, _), _ = jax.lax.scan(body, (q, p, key_run), None, length=n_steps)
    return q


@functools.partial(jax.jit,
                   static_argnames=("M", "n_steps", "n_chains", "splitting"))
def run_chains(key, lam, Lam, X, x_norm2, mu_star,
               delta, sigma_s, h, gamma, M, n_steps, n_chains,
               splitting="BAOAB"):
    """
    Run `n_chains` independent kinetic-Langevin chains under jax.vmap.
    """
    d = X.shape[1]
    N = X.shape[0]
    alpha = jnp.exp(-gamma * h)
    c = jnp.sqrt(1.0 - alpha * alpha)

    def gradU(q, k):
        """Full tilted-potential gradient for ONE chain state q: (d,) -> (d,)."""
        g = gradU_LDS_mc_stationary(k, q[None, :], X, x_norm2,
                                    delta, sigma_s, M, shared_noise=False)[0]
        return g + lam + Lam @ (q - mu_star)

    def single_chain(chain_key):
        k_idx, k_eps, k_p, key_run = jax.random.split(chain_key, 4)
        i = jax.random.randint(k_idx, (), 0, N)
        q = X[i] + delta * jax.random.normal(k_eps, (d,), dtype=X.dtype)
        p = jax.random.normal(k_p, (d,), dtype=X.dtype)

        if splitting == "BAOAB":
            key_run, k0 = jax.random.split(key_run)
            g0 = gradU(q, k0)

            def body(carry, _):
                q, p, g, key = carry
                p = p - (h * 0.5) * g                                   # B
                q = q + (h * 0.5) * p                                   # A
                key, k_o = jax.random.split(key)
                p = alpha * p + c * jax.random.normal(k_o, p.shape, p.dtype)   # O
                q = q + (h * 0.5) * p                                   # A
                key, k_g = jax.random.split(key)
                g = gradU(q, k_g)                                       # force
                p = p - (h * 0.5) * g                                   # B
                return (q, p, g, key), None

            (q, _, _, _), _ = jax.lax.scan(
                body, (q, p, g0, key_run), None, length=n_steps)
        else:                                                            # ABOBA
            def body(carry, _):
                q, p, key = carry
                q = q + (h * 0.5) * p
                key, k_g = jax.random.split(key)
                g = gradU(q, k_g)
                p = p - (h * 0.5) * g
                key, k_o = jax.random.split(key)
                p = alpha * p + c * jax.random.normal(k_o, p.shape, p.dtype)
                p = p - (h * 0.5) * g
                q = q + (h * 0.5) * p
                return (q, p, key), None

            (q, _, _), _ = jax.lax.scan(body, (q, p, key_run), None, length=n_steps)
        return q

    return jax.vmap(single_chain)(jax.random.split(key, n_chains))


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Dual gradients = moment residuals
# ═══════════════════════════════════════════════════════════════════════════

def triu_to_full(u, d):
    """Upper-triangle vector (incl. diagonal) -> full symmetric matrix."""
    iu = jnp.triu_indices(d)
    L = jnp.zeros((d, d), dtype=u.dtype).at[iu].set(u)
    return L + L.T - jnp.diag(jnp.diag(L))


def full_to_triu(M):
    """Full matrix -> its upper triangle (incl. diagonal), as a flat vector."""
    return M[jnp.triu_indices(M.shape[0])]


def _sym(A):
    return 0.5 * (A + A.T)


@functools.partial(jax.jit, static_argnames=(
    "M", "n_steps", "n_chains", "splitting", "cov_estimator",
    "score", "knn_K", "knn_L"))
def moment_grads(key, params, X, x_norm2, mu_star, Sigma_star,
                 delta, sigma_s, h, gamma, M, n_steps, n_chains,
                 splitting="BAOAB", cov_estimator="second_moment",
                 score="full", knn_K=50, knn_L=50):
    """grad F together with scale-free residual diagnostics."""
    if score == "full":
        q = run_chains(key, params["lam"], params["Lam"], X, x_norm2, mu_star,
                       delta, sigma_s, h, gamma, M, n_steps, n_chains, splitting)
    else:
        q = run_chains_batched(key, params["lam"], params["Lam"], X, x_norm2,
                               mu_star, delta, sigma_s, h, gamma, M, n_steps,
                               n_chains, splitting, score, knn_K, knn_L)
    P = q.shape[0]
    m_hat = jnp.mean(q, axis=0)

    if cov_estimator == "sample_cov":            # unbiased covariance proper
        Qc = q - m_hat[None, :]
        S_hat = (Qc.T @ Qc) / (P - 1)
    else:                                        # unbiased for E[(z-mu*)(z-mu*)']
        Qc = q - mu_star[None, :]
        S_hat = (Qc.T @ Qc) / P

    g_lam = mu_star - m_hat
    g_Lam = 0.5 * _sym(Sigma_star - S_hat)

    scale = jnp.sqrt(jnp.trace(Sigma_star))      # typical |z|; mu* itself is ~0
    stats = {"res_mean": jnp.linalg.norm(g_lam) / scale,
             "res_cov": (jnp.linalg.norm(Sigma_star - S_hat)
                         / jnp.linalg.norm(Sigma_star)),
             "finite": jnp.all(jnp.isfinite(q))}
    return {"lam": g_lam, "Lam": g_Lam}, stats


@functools.partial(jax.jit, static_argnames=("eps",))
def project_Lam(Lam, delta, eps=0.05):
    """Symmetrise and clip eigenvalues to keep the tilted target integrable."""
    Lam = _sym(Lam)
    w, V = jnp.linalg.eigh(Lam)
    return (V * jnp.maximum(w, lambda_floor(delta, eps))[None, :]) @ V.T


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Outer loop: one Lion step per L BAOAB steps
# ═══════════════════════════════════════════════════════════════════════════

def estimate_tilting_lion(X, *, delta=0.03, sigma_s=0.4, M=32,
                          h=5e-4, gamma=1.0, n_steps=100, n_chains=100,
                          n_iters=300, lr_rel=0.02, b1=0.9, b2=0.99,
                          splitting="BAOAB", cov_estimator="second_moment",
                          score="full", knn_K=50, knn_L=50,
                          warm_start=True, seed=0, log_every=10, verbose=True,
                          save_every=0, save_path=None, log_path=None,
                          hp_eval_every=0, hp_eval_chains=8192):
    """
    Estimate (lambda, Lambda) with Lion + restarted parallel SG-BAOAB chains.
    """
    Xw_j = jnp.asarray(X, dtype=jnp.float32)
    x_norm2 = jnp.sum(Xw_j * Xw_j, axis=1)
    mu_np, Sigma_np = moment_targets(X, delta)
    mu_star, Sigma_star = jnp.asarray(mu_np), jnp.asarray(Sigma_np)
    d = Xw_j.shape[1]

    key = jax.random.PRNGKey(seed)
    if warm_start:
        key, k_pi = jax.random.split(key)
        lam0, Lam0 = plugin_tilting(k_pi, X, delta, sigma_s, M,
                                    score=score, knn_K=knn_K, knn_L=knn_L)
        lam0 = np.asarray(lam0)
        Lam0 = np.asarray(project_Lam(jnp.asarray(Lam0), delta))
    else:
        # cold start: no tilt at all (Lambda = 0); the Lion scale fallbacks above
        # then set sensible step sizes from delta
        lam0 = np.zeros(d, np.float32)
        Lam0 = np.zeros((d, d), np.float32)

    # Lambda is symmetric, so ONLY its upper triangle is a free parameter:
    # d(d+1)/2 instead of d^2 numbers, and symmetry holds exactly rather
    # than drifting.  The full matrix is rebuilt wherever it is needed.
    params = {"lam": jnp.asarray(lam0),
              "Lam_u": full_to_triu(jnp.asarray(Lam0))}

    rms_lam = float(np.sqrt(np.mean(lam0 ** 2)))
    rms_Lam = float(np.sqrt(np.mean(Lam0 ** 2)))
    if rms_lam < 1e-8:
        rms_lam = 1.0 / delta
        if verbose:
            print(f"  lambda block is zero -> Lion scale fallback 1/delta = {rms_lam:.3f}")
    if rms_Lam < 1e-8:
        rms_Lam = 1.0 / delta ** 2
        if verbose:
            print(f"  Lambda block is zero -> Lion scale fallback 1/delta^2 = {rms_Lam:.3f}")

    lr_lam = lr_rel * rms_lam                      # vector block: |dlam| = lr*sqrt(d)
    lr_Lam = lr_rel * rms_Lam / (2.0 * np.sqrt(d))  # matrix block: spectral-aware
    if verbose:
        print(f"  Lion step sizes: lambda {lr_lam:.4g}, Lambda {lr_Lam:.4g} "
              f"(spectral change per step ~{2*np.sqrt(d)*lr_Lam:.3g})")
    sched_lam = optax.cosine_decay_schedule(lr_lam, max(n_iters, 1))
    sched_Lam = optax.cosine_decay_schedule(lr_Lam, max(n_iters, 1))
    tx = optax.multi_transform(
        {"lam": optax.lion(sched_lam, b1=b1, b2=b2, weight_decay=0.0),
         "Lam_u": optax.lion(sched_Lam, b1=b1, b2=b2, weight_decay=0.0)},
        param_labels={"lam": "lam", "Lam_u": "Lam_u"})
    opt_state = tx.init(params)

    grad_kw = dict(M=M, n_steps=n_steps, n_chains=n_chains,
                   splitting=splitting, cov_estimator=cov_estimator,
                   score=score, knn_K=knn_K, knn_L=knn_L)
    history = []
    t0 = time.time()
    if verbose:
        print(f"Lion tilting estimation | d={d} N={Xw_j.shape[0]} "
              f"P={n_chains} chains (vmap) | L={n_steps} steps | K={n_iters} "
              f"| lr_rel={lr_rel} | {splitting} | score={score}")
        print(f"  warm start: |lam|={np.linalg.norm(lam0):.3f}  "
              f"eig(Lam) in [{np.linalg.eigvalsh(Lam0)[0]:.1f}, "
              f"{np.linalg.eigvalsh(Lam0)[-1]:.1f}]  "
              f"floor={lambda_floor(delta):.1f}")
        print(f"{'iter':>5} {'res_mean':>10} {'res_cov':>10} {'|lam|':>10} "
              f"{'eig_min':>10} {'eig_max':>10} {'sec':>7}")

    for k in range(n_iters):
        key, k_it = jax.random.split(key)
        full = {"lam": params["lam"], "Lam": triu_to_full(params["Lam_u"], d)}
        grads, stats = moment_grads(k_it, full, Xw_j, x_norm2, mu_star,
                                    Sigma_star, delta, sigma_s, h, gamma,
                                    **grad_kw)
        # g_Lam is already symmetric; its triangle is the gradient w.r.t. the
        # free parameters up to a factor 2 off-diagonal, which sign() ignores.
        grads = {"lam": grads["lam"], "Lam_u": full_to_triu(grads["Lam"])}
        if not bool(stats["finite"]) or not np.isfinite(float(stats["res_cov"])):
            raise FloatingPointError(
                f"chains diverged at iteration {k} (res_cov={float(stats['res_cov'])}); "
                f"reduce lr_rel or the step size h")

        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        Lam_full = project_Lam(triu_to_full(params["Lam_u"], d), delta)
        params["Lam_u"] = full_to_triu(Lam_full)

        hp = {}
        if hp_eval_every and ((k + 1) % hp_eval_every == 0 or k + 1 == n_iters):
            key, k_hp = jax.random.split(key)
            _, hp_stats = moment_grads(
                k_hp, {"lam": params["lam"], "Lam": Lam_full}, Xw_j, x_norm2,
                mu_star, Sigma_star, delta, sigma_s, h, gamma,
                M=M, n_steps=n_steps, n_chains=hp_eval_chains,
                splitting=splitting, cov_estimator=cov_estimator,
                score=score, knn_K=knn_K, knn_L=knn_L)
            hp = {"hp_res_mean": float(hp_stats["res_mean"]),
                  "hp_res_cov": float(hp_stats["res_cov"]),
                  "hp_chains": int(hp_eval_chains),
                  "hp_floor": res_cov_floor(Sigma_star, hp_eval_chains)}

        w = np.linalg.eigvalsh(np.asarray(Lam_full))
        rec = {"iter": k + 1,
               "res_mean": float(stats["res_mean"]),
               "res_cov": float(stats["res_cov"]),
               "lam_norm": float(jnp.linalg.norm(params["lam"])),
               "eig_min": float(w[0]), "eig_max": float(w[-1]),
               "lr_lam": float(sched_lam(k)), "lr_Lam": float(sched_Lam(k)),
               "time": time.time() - t0}
        rec.update(hp)
        history.append(rec)

        if save_every and save_path and ((k + 1) % save_every == 0 or k + 1 == n_iters):
            np.savez(save_path, lam=np.asarray(params["lam"]),
                     Lam=np.asarray(Lam_full), Lam_u=np.asarray(params["Lam_u"]),
                     iter=k + 1,
                     res_mean=rec["res_mean"], res_cov=rec["res_cov"])
            if log_path:
                import json as _json
                with open(log_path, "w") as _f:
                    _json.dump(history, _f)

        if verbose and ((k + 1) % log_every == 0 or k == 0):
            print(f"{k+1:>5} {rec['res_mean']:>10.5f} {rec['res_cov']:>10.5f} "
                  f"{rec['lam_norm']:>10.4f} {rec['eig_min']:>10.2f} "
                  f"{rec['eig_max']:>10.2f} {rec['time']:>7.1f}"
                  + (f"   | HP(n={rec['hp_chains']}): res_cov {rec['hp_res_cov']:.4f}"
                     f" (floor {rec['hp_floor']:.3f})  res_mean {rec['hp_res_mean']:.5f}"
                     if "hp_res_cov" in rec else ""), flush=True)

    return (np.asarray(params["lam"]),
            np.asarray(triu_to_full(params["Lam_u"], d)), history)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Generation with frozen (lambda, Lambda) -- identical kernel
# ═══════════════════════════════════════════════════════════════════════════

def sample_tilted(lam, Lam, X, *, delta=0.03, sigma_s=0.4, M=32, h=5e-4,
                  gamma=1.0, n_steps=100, n_samples=300, splitting="BAOAB",
                  score="full", knn_K=50, knn_L=50, seed=0):
    """
    Draw `n_samples` latents from the estimated moment-matched target, using the
    exact kernel and step count used during estimation.
    """
    Xw_j = jnp.asarray(X, dtype=jnp.float32)
    x_norm2 = jnp.sum(Xw_j * Xw_j, axis=1)
    mu_np, _ = moment_targets(X, delta)
    args = (jax.random.PRNGKey(seed), jnp.asarray(lam), jnp.asarray(Lam),
            Xw_j, x_norm2, jnp.asarray(mu_np), delta, sigma_s, h, gamma,
            M, n_steps, n_samples, splitting)
    q = (run_chains(*args) if score == "full"
         else run_chains_batched(*args, score, knn_K, knn_L))
    return np.asarray(q, dtype=np.float32)


def res_cov_floor(Sigma_star, n):
    """
    Sampling floor for the relative Frobenius covariance residual.
    """
    S = np.asarray(Sigma_star, dtype=np.float64)
    tr = float(np.trace(S))
    fro2 = float((S * S).sum())
    if fro2 <= 0 or n <= 0:
        return float("nan")
    return float(np.sqrt(tr * tr / (n * fro2) + 1.0 / n))


def moment_residuals(q, mu_star, Sigma_star):
    """Diagnostics for a batch of samples: (relative mean error, rel. cov error)."""
    q = np.asarray(q)
    Qc = q - mu_star[None, :]
    S = (Qc.T @ Qc) / q.shape[0]
    return (float(np.linalg.norm(mu_star - q.mean(0)) / np.sqrt(np.trace(Sigma_star))),
            float(np.linalg.norm(Sigma_star - S) / np.linalg.norm(Sigma_star)))
