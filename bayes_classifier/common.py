"""
common.py
============
Shared building blocks for the handwritten-digit classification experiment
(top-40 subspace, NN-scaled delta/sigma, plug-in + dual estimator).
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.optimize import minimize as sp_minimize
from sklearn.metrics import f1_score

import config as cfg
import tilting_lion as TL  # noqa: E402  (release/common, on sys.path via config)


# ============================================================
# 1. Average nearest-neighbor distance
# ============================================================

def mean_nn_distance(U_c: np.ndarray) -> float:
    U = jnp.asarray(U_c, dtype=jnp.float32)
    sq = jnp.sum(U * U, axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (U @ U.T)
    dist2 = jnp.maximum(dist2, 0.0)
    n = U.shape[0]
    # exclude self: NOT `+ eye*inf` (0*inf = nan and would corrupt every entry)
    dist2 = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist2)
    nn = jnp.sqrt(jnp.min(dist2, axis=1))
    return float(jnp.mean(nn))


# ============================================================
# 2. delta_c, sigma_c parameterization
# ============================================================

def delta_sigma(c_delta: float, c_sigma: float, nn_c: float, dsub: int = 40):
    """delta_c = c_delta * NN_c / sqrt(dsub),  sigma_c = c_sigma * NN_c."""
    delta_c = c_delta * nn_c / np.sqrt(dsub)
    sigma_c = c_sigma * nn_c
    return float(delta_c), float(sigma_c)


def h_sold(delta_c: float, c_sold: float = 0.2) -> float:
    """MM-SOLD overdamped step size"""
    return float(c_sold * delta_c ** 2)


def h_baoab(delta_c: float) -> float:
    """Dual-estimator BAOAB step size"""
    return float(delta_c / 20.0)


def gamma_baoab(U_c: np.ndarray) -> float:
    """Dual-estimator BAOAB friction"""
    U = np.asarray(U_c, dtype=np.float64)
    Uc = U - U.mean(axis=0, keepdims=True)
    Sigma_c = (Uc.T @ Uc) / max(len(U) - 1, 1)
    lam_max = float(np.linalg.eigvalsh(Sigma_c)[-1])
    return float(1.0 / np.sqrt(max(lam_max, 1e-12)))


# ============================================================
# 3. Smoothed GMM potential V_c(z) = E_eps[U_c(z + s*eps)]
# ============================================================

@jax.jit
def _U_batch(Z, X, x_norm2, sigma):
    """Exact (un-smoothed) GMM potential U(z) = -logsumexp(-||z-x_i||^2/2sigma^2)."""
    z_norm2 = jnp.sum(Z * Z, axis=1)
    dist2 = z_norm2[:, None] + x_norm2[None, :] - 2.0 * (Z @ X.T)
    return -logsumexp(-0.5 * dist2 / (sigma * sigma), axis=1)


_M_CHUNK = 64  # MC samples per GPU call, chunked to bound (M x B x N) memory


@functools.partial(jax.jit, static_argnames=("M",))
def _Vc_mc_chunk(key, Z, X, x_norm2, sigma, s, M):
    """Antithetic-MC smoothed potential for one chunk of size M."""
    B, d = Z.shape
    half = M // 2
    eps_h = jax.random.normal(key, (half, d), dtype=Z.dtype)
    eps = jnp.concatenate([eps_h, -eps_h], axis=0)
    Zp = Z[:, None, :] + s * eps[None, :, :]
    U_flat = _U_batch(Zp.reshape((-1, d)), X, x_norm2, sigma)
    return U_flat.reshape((B, M)).mean(axis=1)


def smoothed_potential(key, Z, X, x_norm2, sigma, s, M, m_chunk=_M_CHUNK):
    """V(z) = E_eps[U(z + s*eps)] via chunked antithetic MC, M samples total."""
    acc, n_chunks, remaining = None, 0, M
    while remaining > 0:
        m_use = min(m_chunk, remaining)
        key, k = jax.random.split(key)
        chunk_val = np.array(_Vc_mc_chunk(k, Z, X, x_norm2, sigma, s, m_use))
        acc = chunk_val if acc is None else acc + chunk_val
        n_chunks += 1
        remaining -= m_use
    return acc / n_chunks


def compute_class_energy(key, U_query, mu_star_c, lambda_c, Lambda_c,
                          U_train_c, u_train_norm2_c, delta_c, sigma_c, M,
                          batch_size=1000):
    """E_c(z) = V_c(z) + lambda_c'z + 0.5*(z-mu_c*)'Lambda_c(z-mu_c*) for a
    batch of query points, batched to bound GPU memory."""
    N = U_query.shape[0]
    V_chunks = []
    for i in range(0, N, batch_size):
        chunk = jnp.asarray(U_query[i:i + batch_size])
        key, k_v = jax.random.split(key)
        V_chunks.append(smoothed_potential(
            k_v, chunk, U_train_c, u_train_norm2_c, delta_c, sigma_c, M))
    V_c = np.concatenate(V_chunks)

    linear_c = np.asarray(U_query) @ np.asarray(lambda_c)
    diff_c = np.asarray(U_query) - np.asarray(mu_star_c)[None, :]
    quad_c = 0.5 * np.einsum('bi,ij,bj->b', diff_c, np.asarray(Lambda_c), diff_c)
    return V_c + linear_c + quad_c


# ============================================================
# 4. Bias calibration on the validation set (unchanged from run_classifier.py)
# ============================================================

def calibrate_biases(E_val: np.ndarray, y_val: np.ndarray, n_classes: int) -> np.ndarray:
    """Learn per-class additive biases b_c minimising cross-entropy on E_val.

    logit_c(z) = -E_c(z) - b_c ,  L(b) = -mean_i log softmax(-E(z_i)-b)[y_i].
    """
    E = np.asarray(E_val, dtype=np.float64)
    N = len(y_val)
    one_hot = np.zeros((N, n_classes), dtype=np.float64)
    one_hot[np.arange(N), y_val] = 1.0

    def loss_and_grad(b):
        logits = -E - b[None, :]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        nll = -np.mean(np.sum(one_hot * np.log(probs + 1e-12), axis=1))
        grad = (one_hot - probs).mean(axis=0)
        return float(nll), grad

    result = sp_minimize(loss_and_grad, np.zeros(n_classes), jac=True,
                          method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
    return result.x.astype(np.float32)


# ============================================================
# 5. Metrics
# ============================================================

def compute_metrics(y_true, y_pred, n_classes=10):
    acc = float(np.mean(y_pred == y_true))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    per_class_acc = []
    for c in range(n_classes):
        mask = y_true == c
        per_class_acc.append(float(np.mean(y_pred[mask] == c)) if mask.any() else 0.0)
    return {'acc': acc, 'macro_f1': f1_macro, 'per_class_acc': per_class_acc}


# ============================================================
# 6. Energy distance (for the MM-SOLD-augmentation (c_delta,c_sigma) search,
#    identical statistic to release/digits/scripts/digits_cv_grid.py)
# ============================================================

def energy_distance(X, Y, rng, cap=1500):
    """E-distance 2E|X-Y| - E|X-X'| - E|Y-Y'|; subsampled for cost."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    if len(Y) > cap:
        Y = Y[rng.choice(len(Y), cap, replace=False)]

    def md(A, B):
        d = np.sqrt(np.maximum(
            (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0))
        return float(d.mean())

    return 2 * md(X, Y) - md(X, X) - md(Y, Y)


# ============================================================
# 7. Per-class training-latent bundle
# ============================================================

def build_class_data(U_train, y_train, n_classes=10):
    """Precompute per-class top-k training coordinates and ||.||^2, plus NN_c."""
    class_data = {}
    for c in range(n_classes):
        Uc = np.asarray(U_train[y_train == c], dtype=np.float32)
        class_data[c] = {
            'U': Uc,
            'u_norm2': jnp.sum(jnp.asarray(Uc) ** 2, axis=1),
            'NN': mean_nn_distance(Uc),
        }
    return class_data
