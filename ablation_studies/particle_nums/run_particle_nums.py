"""
run_particle_nums.py
====================
Ablation study: effect of particle number and training set size on
MM-SOLD vs Kinetic Langevin Dynamics (BAOAB) sample quality.
"""

import os
import sys
import json
import time
import functools
import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# ── Path setup ────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
REPO    = os.path.abspath(os.path.join(HERE, "..", ".."))
AUG_DIR = os.path.abspath(os.path.join(REPO, "augmentation_handwritten"))
HW_DIR  = os.path.abspath(os.path.join(REPO, "high_dimension", "handwritten"))

for _p in (AUG_DIR, HW_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from whitening_utils import (
    compute_sample_mean_cov, symmetric_matrix_sqrt_and_invsqrt,
    whiten, unwhiten,
)
from sampling_algo import (
    gradU_LDS_mc_stationary,
    sample_class_overdamped_manifold,
)
from nrae_model       import make_nrae_state, decode_latents_nrae, load_params as nrae_load
from classifier_model import load_params as clf_load
from metrics          import extract_features, compute_tau, compute_duprate

# ── Paths ─────────────────────────────────────────────────────────────────────
_SERVER_ROOT = os.environ.get("DATA_ROOT", "/path/to/datasets/RP1/handwritten")
LATENT_DIR   = os.path.join(_SERVER_ROOT, "latents")
DATA_DIR     = os.path.join(_SERVER_ROOT, "data")
NRAE_CKPT    = os.path.join(AUG_DIR, "checkpoints", "nrae_best.pkl")
CLF_CKPT     = os.path.join(AUG_DIR, "checkpoints", "classifier_baseline_best.pkl")
RESULTS_DIR  = os.path.join(HERE, "results")
FIG_DIR      = os.path.join(HERE, "figures")

# ── Hyperparameters (match high_dimension/handwritten/config.py) ──────────────
SIGMA_SMOOTHING = 0.2       # LDS smoothing bandwidth (fixed)
M_MC            = 32        # MC noise samples (fixed)
SIGMA_GMM       = 0.03      # GMM kernel bandwidth
SOLD_NSTEPS     = 100       # Langevin steps
SOLD_H          = 5e-4      # step size
K_WHITEN        = 20        # top-k eigencomponents for whitening
GAMMA_KLD       = 1.0       # BAOAB friction coefficient

KID_DEGREE      = 3
N_RUNS          = 5         # independent runs per configuration

# Experiment A
N_TRAIN_A    = 1000
N_GEN_LIST_A = [101, 200, 300, 500, 800, 1000, 1500, 2000]
N_GEN_TIMING = 101          # N_GENERATE used for CPU timing

# Experiment B
N_GEN_B         = 500
N_TRAIN_LIST_B  = [50, 100, 200, 500, 700, 1000]

# Plot colours
COLOR_SOLD = "#1565C0"
COLOR_KLD  = "#C62828"


# ═════════════════════════════════════════════════════════════════════════════
# KID: unbiased polynomial-kernel MMD² (direct, no subsampling)
# ═════════════════════════════════════════════════════════════════════════════

def _poly_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> np.ndarray:
    d = X.shape[1]
    return (X @ Y.T / d + 1.0) ** degree


def compute_kid_direct(feats_real: np.ndarray,
                       feats_gen:  np.ndarray,
                       degree:     int = 3) -> float:
    """Unbiased MMD² between feats_real and feats_gen (no subsampling)."""
    n, m = feats_real.shape[0], feats_gen.shape[0]
    Kxx  = _poly_kernel(feats_real, feats_real, degree)
    Kyy  = _poly_kernel(feats_gen,  feats_gen,  degree)
    Kxy  = _poly_kernel(feats_real, feats_gen,  degree)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1))
                 + Kyy.sum() / (m * (m - 1))
                 - 2.0 * Kxy.mean())


# ═════════════════════════════════════════════════════════════════════════════
# MM-SOLD correction parameters (λ, Λ)
# ═════════════════════════════════════════════════════════════════════════════

def estimate_mm_params(key, Xw, sigma_gmm, sigma_s, M):
    """
    Compute MM-SOLD correction parameters (λ, Λ) in whitened latent space.
    """
    Xw_np = np.array(Xw, dtype=np.float64)
    n, d  = Xw_np.shape

    xbar      = Xw_np.mean(axis=0)                              # (d,)
    Xc        = Xw_np - xbar[None, :]                           # (n, d) centred
    Sigma_x   = (Xc.T @ Xc) / n                                 # (d, d) biased
    Sigma_tgt = sigma_gmm ** 2 * np.eye(d) + Sigma_x            # (d, d)

    # Smoothed scores g_σ(xᵢ) = ∇V(xᵢ)  at each training point
    xw_norm2 = jnp.sum(Xw * Xw, axis=1)
    G = np.array(gradU_LDS_mc_stationary(
        key, Xw, Xw, xw_norm2,
        sigma_gmm, sigma_s, M,
        shared_noise=False,
    ), dtype=np.float64)                                         # (n, d)

    lambda_vec = -G.mean(axis=0)                                 # (d,)
    C          = Xc.T @ G / n                                    # (d, d)
    S          = np.eye(d, dtype=np.float64) - 0.5 * (C + C.T)  # (d, d)

    # Regularise Sigma_tgt for numerical stability (critical when N_TRAIN << d)
    eig_max   = float(np.linalg.eigvalsh(Sigma_tgt).max())
    reg       = max(eig_max * 1e-6, 1e-6)
    Sigma_reg = Sigma_tgt + reg * np.eye(d)

    # Lyapunov solve: Sigma_reg Λ + Λ Sigma_reg = 2 S
    Lambda_mat = scipy.linalg.solve_continuous_lyapunov(Sigma_reg, 2.0 * S)

    return (lambda_vec.astype(np.float32),
            Lambda_mat.astype(np.float32),
            xbar.astype(np.float32),
            Sigma_tgt.astype(np.float32))


# ═════════════════════════════════════════════════════════════════════════════
# BAOAB kinetic Langevin sampler
# ═════════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("M", "n_steps"))
def _run_kld_baoab(q, p, key,
                   Xw, xw_norm2, xbar, lambda_vec, Lambda_mat,
                   sigma_gmm, sigma_s, h, gamma, M, n_steps):
    """
    BAOAB kinetic Langevin for a batch of independent particles.
    """
    alpha = jnp.exp(-gamma * h)
    c     = jnp.sqrt(1.0 - alpha * alpha)

    def body(carry, _):
        q, p, key = carry

        # B: half position step
        q = q + (h * 0.5) * p

        # Compute gradient once (reused for both A sub-steps)
        key, k_g = jax.random.split(key)
        g_V  = gradU_LDS_mc_stationary(
            k_g, q, Xw, xw_norm2,
            sigma_gmm, sigma_s, M,
            shared_noise=False,
        )                                                    # (B, d)
        g_mm = (g_V
                + lambda_vec[None, :]
                + (q - xbar[None, :]) @ Lambda_mat)         # (B, d)

        # A: first half momentum step
        p = p - (h * 0.5) * g_mm

        # O: Ornstein-Uhlenbeck thermostat
        key, k_o = jax.random.split(key)
        xi = jax.random.normal(k_o, p.shape, dtype=p.dtype)
        p  = alpha * p + c * xi

        # A: second half momentum step (same gradient)
        p = p - (h * 0.5) * g_mm

        # B: half position step
        q = q + (h * 0.5) * p

        return (q, p, key), None

    (q_final, _, _), _ = jax.lax.scan(body, (q, p, key), None, length=n_steps)
    return q_final


def sample_kld(Xw, xbar, lambda_vec, Lambda_mat,
               n_particles, n_steps, h, gamma,
               sigma_gmm, sigma_s, M, seed=0):
    """
    Sample from the moment-matched marginal distribution via BAOAB.
    """
    d        = Xw.shape[1]
    xw_norm2 = jnp.sum(Xw * Xw, axis=1)

    key = jax.random.PRNGKey(seed)
    key, k_idx, k_eps, k_p = jax.random.split(key, 4)

    idx0 = jax.random.randint(k_idx, (n_particles,), 0, Xw.shape[0])
    q0   = Xw[idx0] + sigma_gmm * jax.random.normal(k_eps, (n_particles, d))
    p0   = jax.random.normal(k_p, (n_particles, d), dtype=jnp.float32)

    q_final = _run_kld_baoab(
        q0, p0, key,
        Xw, xw_norm2,
        jnp.asarray(xbar), jnp.asarray(lambda_vec), jnp.asarray(Lambda_mat),
        sigma_gmm, sigma_s, h, gamma, M, n_steps,
    )
    jax.block_until_ready(q_final)
    return np.array(q_final, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Sampling wrappers (return raw unwhitened latents)
# ═════════════════════════════════════════════════════════════════════════════

def run_sold(Z_train, mean_cls, S_sqrt_cls, S_invsqrt_cls, n_particles, seed):
    """Run MM-SOLD. Returns (n_particles, d) raw NRAE latents."""
    z_w, _ = sample_class_overdamped_manifold(
        Z_class         = jnp.asarray(Z_train, dtype=jnp.float32),
        mean_class      = mean_cls,
        S_sqrt_class    = S_sqrt_cls,
        S_invsqrt_class = S_invsqrt_cls,
        n_particles     = n_particles,
        nsteps          = SOLD_NSTEPS,
        h               = SOLD_H,
        sigma_gmm       = SIGMA_GMM,
        sigma_smoothing = SIGMA_SMOOTHING,
        M               = M_MC,
        shared_noise    = False,
        fixed_noise     = False,
        discretization  = "LM",
        seed            = seed,
    )
    return np.array(unwhiten(z_w, mean_cls, S_sqrt_cls), dtype=np.float32)


def run_kld(Xw, xbar, lambda_vec, Lambda_mat,
            mean_cls, S_sqrt_cls, n_particles, seed):
    """Run KLD-BAOAB. Returns (n_particles, d) raw NRAE latents."""
    q_w = sample_kld(
        Xw, xbar, lambda_vec, Lambda_mat,
        n_particles = n_particles,
        n_steps     = SOLD_NSTEPS,
        h           = SOLD_H,
        gamma       = GAMMA_KLD,
        sigma_gmm   = SIGMA_GMM,
        sigma_s     = SIGMA_SMOOTHING,
        M           = M_MC,
        seed        = seed,
    )
    return np.array(
        unwhiten(jnp.asarray(q_w), mean_cls, S_sqrt_cls), dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_nrae(ckpt_path, seed=0):
    from config import LATENT_DIM, ENC1_HIDDEN, DEC_HIDDEN, N_DCT, UNET_BASE_CH
    params_np, info = nrae_load(ckpt_path)
    latent_dim   = info.get("latent_dim",   LATENT_DIM)
    enc1_hidden  = info.get("enc1_hidden",  ENC1_HIDDEN)
    dec_hidden   = info.get("dec_hidden",   DEC_HIDDEN)
    n_dct        = info.get("n_dct",        N_DCT)
    unet_base_ch = info.get("unet_base_ch", UNET_BASE_CH)
    rng = jax.random.PRNGKey(seed)
    model, _ = make_nrae_state(
        rng, latent_dim=latent_dim, enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch)
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    return model, params, n_dct


def _decode(nrae_model, nrae_params, Z, n_dct):
    imgs = decode_latents_nrae(nrae_model, nrae_params, Z, batch_size=32)
    return np.clip(np.array(imgs), 0.0, 1.0)


def _whiten_stats(Z_class):
    Z_j = jnp.asarray(Z_class, dtype=jnp.float32)
    mean_cls, cov_cls = compute_sample_mean_cov(Z_j)
    S_sqrt, S_invsqrt, _ = symmetric_matrix_sqrt_and_invsqrt(
        cov_cls, eps=1e-5, k=K_WHITEN)
    return mean_cls, S_sqrt, S_invsqrt


def _eval_metrics(Z_gen, Z_train, nrae_model, nrae_params, n_dct,
                  clf_params, feats_test, tau):
    """Decode → extract CNN features → compute KID and DupRate."""
    imgs_gen  = _decode(nrae_model, nrae_params, Z_gen, n_dct)
    feats_gen = extract_features(imgs_gen, clf_params, batch_size=64)
    kid = compute_kid_direct(feats_test, feats_gen, degree=KID_DEGREE)
    dup = compute_duprate(Z_gen, Z_train, tau)
    return kid, dup


def _measure_cpu_time(fn, n_warmup=1, n_timed=3):
    """Warm up then time fn() on current device. Returns mean wall-clock (s)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def _plot_metric(x_vals, sold_mean, sold_std, kld_mean, kld_std,
                 xlabel, ylabel, fig_path):
    """Line plot with shaded ±std bands. Style matches Langevin_steps ablation."""
    x = np.array(x_vals)
    sm, ss = np.array(sold_mean), np.array(sold_std)
    km, ks = np.array(kld_mean),  np.array(kld_std)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x, sm, color=COLOR_SOLD, marker='o', linewidth=1.8,
            markersize=5, label='MM-SOLD')
    ax.fill_between(x, sm - ss, sm + ss, color=COLOR_SOLD, alpha=0.15)

    ax.plot(x, km, color=COLOR_KLD, marker='s', linewidth=1.8,
            markersize=5, label='Kinetic Langevin')
    ax.fill_between(x, km - ks, km + ks, color=COLOR_KLD, alpha=0.15)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fig_path}")


def _plot_timing(x_vals, sold_times, kld_times, xlabel, fig_path):
    """Per-sample CPU time vs x_vals."""
    x = np.array(x_vals)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, sold_times, color=COLOR_SOLD, marker='o', linewidth=1.8,
            markersize=5, label='MM-SOLD')
    ax.plot(x, kld_times,  color=COLOR_KLD,  marker='s', linewidth=1.8,
            markersize=5, label='Kinetic Langevin')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Per-sample time (ms)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fig_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Particle-Number Ablation: MM-SOLD vs KLD-BAOAB")
    print("=" * 65)
    print(f"JAX backend : {jax.default_backend()}  devices : {jax.devices()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,     exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    Z_train_full = np.load(os.path.join(LATENT_DIR, "Z_train.npy"))
    imgs_test    = np.load(os.path.join(DATA_DIR,   "images_test.npy"))
    print(f"Loaded  Z_train={Z_train_full.shape}  imgs_test={imgs_test.shape}")

    # ── Load NRAE and CNN feature extractor ───────────────────────────────────
    print(f"\nLoading NRAE from {NRAE_CKPT}")
    nrae_model, nrae_params, n_dct = _load_nrae(NRAE_CKPT)
    print(f"  n_dct={n_dct}")

    print(f"Loading CLF from {CLF_CKPT}")
    clf_params, clf_info = clf_load(CLF_CKPT)
    print(f"  info={clf_info}")

    # ── Fixed test CNN features ────────────────────────────────────────────────
    print("\nExtracting test CNN features ...")
    feats_test = extract_features(imgs_test, clf_params, batch_size=64)
    print(f"  feats_test: {feats_test.shape}")

    # DupRate τ for the full N_TRAIN=1000 training set
    tau_full = compute_tau(Z_train_full)
    print(f"  τ (N_TRAIN=1000) = {tau_full:.5f}")

    # Global PRNGKey
    key = jax.random.PRNGKey(0)

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment A:  Fixed N_TRAIN=1000, varying N_GENERATE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("Experiment A: N_TRAIN=1000, varying N_GENERATE")
    print(f"{'='*65}")

    Z_train_A = Z_train_full
    tau_A     = tau_full

    # Whitening statistics
    mean_A, S_sqrt_A, S_invsqrt_A = _whiten_stats(Z_train_A)
    Xw_A = jnp.asarray(
        np.array(whiten(jnp.asarray(Z_train_A, dtype=jnp.float32),
                        mean_A, S_invsqrt_A)),
        dtype=jnp.float32)

    # MM correction parameters for KLD  (computed once)
    print("Computing MM params (λ, Λ) for KLD ...")
    key, k_mm = jax.random.split(key)
    lambda_A, Lambda_A, xbar_A, _ = estimate_mm_params(
        k_mm, Xw_A, SIGMA_GMM, SIGMA_SMOOTHING, M_MC)
    print(f"  |λ| = {np.linalg.norm(lambda_A):.5f}  "
          f"Λ eigs ∈ [{np.linalg.eigvalsh(Lambda_A).min():.4f}, "
          f"{np.linalg.eigvalsh(Lambda_A).max():.4f}]")

    sold_kid_A, sold_dup_A = [], []
    kld_kid_A,  kld_dup_A  = [], []
    sold_time_A = None
    kld_time_A  = None

    for n_gen in N_GEN_LIST_A:
        print(f"\n  [A] N_GENERATE={n_gen}")

        # CPU timing at N_GEN_TIMING
        if n_gen == N_GEN_TIMING:
            print(f"    Measuring CPU time (N_GENERATE={n_gen}, "
                  f"warmup=1, timed=3) ...")
            with jax.default_device(jax.devices("cpu")[0]):
                _sold_fn = lambda: run_sold(
                    Z_train_A, mean_A, S_sqrt_A, S_invsqrt_A, n_gen, seed=0)
                _kld_fn  = lambda: run_kld(
                    Xw_A, xbar_A, lambda_A, Lambda_A,
                    mean_A, S_sqrt_A, n_gen, seed=0)
                t_sold = _measure_cpu_time(_sold_fn, n_warmup=1, n_timed=3)
                t_kld  = _measure_cpu_time(_kld_fn,  n_warmup=1, n_timed=3)
            sold_time_A = t_sold / n_gen * 1000   # ms per sample
            kld_time_A  = t_kld  / n_gen * 1000
            print(f"    SOLD: {sold_time_A:.3f} ms/sample  "
                  f"KLD: {kld_time_A:.3f} ms/sample")

        # N_RUNS independent runs
        kid_s_runs, dup_s_runs = [], []
        kid_k_runs, dup_k_runs = [], []

        for run_seed in range(N_RUNS):
            # MM-SOLD
            Z_s = run_sold(Z_train_A, mean_A, S_sqrt_A, S_invsqrt_A,
                           n_gen, seed=run_seed)
            ks, ds = _eval_metrics(Z_s, Z_train_A,
                                   nrae_model, nrae_params, n_dct,
                                   clf_params, feats_test, tau_A)
            kid_s_runs.append(ks)
            dup_s_runs.append(ds)

            # KLD-BAOAB
            Z_k = run_kld(Xw_A, xbar_A, lambda_A, Lambda_A,
                          mean_A, S_sqrt_A, n_gen, seed=run_seed)
            kk, dk = _eval_metrics(Z_k, Z_train_A,
                                   nrae_model, nrae_params, n_dct,
                                   clf_params, feats_test, tau_A)
            kid_k_runs.append(kk)
            dup_k_runs.append(dk)

            print(f"    seed={run_seed}  SOLD KID={ks:.4f} Dup={ds:.3f} | "
                  f"KLD  KID={kk:.4f} Dup={dk:.3f}")

        sold_kid_A.append((float(np.mean(kid_s_runs)), float(np.std(kid_s_runs))))
        sold_dup_A.append((float(np.mean(dup_s_runs)), float(np.std(dup_s_runs))))
        kld_kid_A.append((float(np.mean(kid_k_runs)),  float(np.std(kid_k_runs))))
        kld_dup_A.append((float(np.mean(dup_k_runs)),  float(np.std(dup_k_runs))))

    results_A = {
        "n_gen_list": N_GEN_LIST_A,
        "sold":       {"kid": sold_kid_A, "dup": sold_dup_A},
        "kld":        {"kid": kld_kid_A,  "dup": kld_dup_A},
        "timing":     {"sold_ms_per_sample": sold_time_A,
                       "kld_ms_per_sample":  kld_time_A,
                       "n_gen_timing":       N_GEN_TIMING},
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment B:  Fixed N_GENERATE=1500, varying N_TRAIN
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("Experiment B: N_GENERATE=1500, varying N_TRAIN")
    print(f"{'='*65}")

    sold_kid_B, sold_dup_B = [], []
    kld_kid_B,  kld_dup_B  = [], []
    sold_time_B, kld_time_B = [], []

    rng_sub = np.random.default_rng(0)   # reproducible subsampling

    for n_train in N_TRAIN_LIST_B:
        print(f"\n  [B] N_TRAIN={n_train}")

        # Subsample training set
        idx_sub  = rng_sub.choice(len(Z_train_full), size=n_train, replace=False)
        Z_train_B = Z_train_full[idx_sub]
        tau_B     = compute_tau(Z_train_B)
        print(f"    τ = {tau_B:.5f}")

        # Whitening statistics for this N_TRAIN
        mean_B, S_sqrt_B, S_invsqrt_B = _whiten_stats(Z_train_B)
        Xw_B = jnp.asarray(
            np.array(whiten(jnp.asarray(Z_train_B, dtype=jnp.float32),
                            mean_B, S_invsqrt_B)),
            dtype=jnp.float32)

        # MM correction parameters for KLD (estimated from N_TRAIN samples)
        key, k_mm = jax.random.split(key)
        lambda_B, Lambda_B, xbar_B, _ = estimate_mm_params(
            k_mm, Xw_B, SIGMA_GMM, SIGMA_SMOOTHING, M_MC)
        print(f"    |λ| = {np.linalg.norm(lambda_B):.5f}")

        # CPU timing (always at N_GEN_TIMING=100 for comparability across N_TRAIN)
        print(f"    Measuring CPU time (N_GENERATE={N_GEN_TIMING}) ...")
        with jax.default_device(jax.devices("cpu")[0]):
            _sold_fn = lambda: run_sold(
                Z_train_B, mean_B, S_sqrt_B, S_invsqrt_B, N_GEN_TIMING, seed=0)
            _kld_fn  = lambda: run_kld(
                Xw_B, xbar_B, lambda_B, Lambda_B,
                mean_B, S_sqrt_B, N_GEN_TIMING, seed=0)
            t_sold = _measure_cpu_time(_sold_fn, n_warmup=1, n_timed=3)
            t_kld  = _measure_cpu_time(_kld_fn,  n_warmup=1, n_timed=3)
        sold_time_B.append(t_sold / N_GEN_TIMING * 1000)
        kld_time_B.append( t_kld  / N_GEN_TIMING * 1000)
        print(f"    SOLD: {sold_time_B[-1]:.3f} ms/sample  "
              f"KLD: {kld_time_B[-1]:.3f} ms/sample")

        # N_RUNS independent runs
        kid_s_runs, dup_s_runs = [], []
        kid_k_runs, dup_k_runs = [], []

        for run_seed in range(N_RUNS):
            # MM-SOLD
            Z_s = run_sold(Z_train_B, mean_B, S_sqrt_B, S_invsqrt_B,
                           N_GEN_B, seed=run_seed)
            ks, ds = _eval_metrics(Z_s, Z_train_B,
                                   nrae_model, nrae_params, n_dct,
                                   clf_params, feats_test, tau_B)
            kid_s_runs.append(ks)
            dup_s_runs.append(ds)

            # KLD-BAOAB
            Z_k = run_kld(Xw_B, xbar_B, lambda_B, Lambda_B,
                          mean_B, S_sqrt_B, N_GEN_B, seed=run_seed)
            kk, dk = _eval_metrics(Z_k, Z_train_B,
                                   nrae_model, nrae_params, n_dct,
                                   clf_params, feats_test, tau_B)
            kid_k_runs.append(kk)
            dup_k_runs.append(dk)

            print(f"    seed={run_seed}  SOLD KID={ks:.4f} Dup={ds:.3f} | "
                  f"KLD  KID={kk:.4f} Dup={dk:.3f}")

        sold_kid_B.append((float(np.mean(kid_s_runs)), float(np.std(kid_s_runs))))
        sold_dup_B.append((float(np.mean(dup_s_runs)), float(np.std(dup_s_runs))))
        kld_kid_B.append((float(np.mean(kid_k_runs)),  float(np.std(kid_k_runs))))
        kld_dup_B.append((float(np.mean(dup_k_runs)),  float(np.std(dup_k_runs))))

    results_B = {
        "n_train_list": N_TRAIN_LIST_B,
        "sold":         {"kid": sold_kid_B, "dup": sold_dup_B},
        "kld":          {"kid": kld_kid_B,  "dup": kld_dup_B},
        "timing":       {"sold_ms_per_sample": sold_time_B,
                         "kld_ms_per_sample":  kld_time_B,
                         "n_gen_timing":       N_GEN_TIMING},
    }

    # ── Save results ──────────────────────────────────────────────────────────
    all_results = {
        "config": {
            "sigma_smoothing": SIGMA_SMOOTHING,
            "M_mc":            M_MC,
            "sigma_gmm":       SIGMA_GMM,
            "n_steps":         SOLD_NSTEPS,
            "h":               SOLD_H,
            "k_whiten":        K_WHITEN,
            "gamma_kld":       GAMMA_KLD,
            "n_runs":          N_RUNS,
        },
        "experiment_A": results_A,
        "experiment_B": results_B,
    }
    results_path = os.path.join(RESULTS_DIR, "results_particle_nums.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    def _unzip(lst):
        return [x[0] for x in lst], [x[1] for x in lst]

    # Experiment A
    sm_A, ss_A = _unzip(sold_kid_A)
    km_A, ks_A = _unzip(kld_kid_A)
    _plot_metric(N_GEN_LIST_A, sm_A, ss_A, km_A, ks_A,
                 "Number of particles", "KID",
                 os.path.join(FIG_DIR, "exp_A_kid.pdf"))

    sm_A, ss_A = _unzip(sold_dup_A)
    km_A, ks_A = _unzip(kld_dup_A)
    _plot_metric(N_GEN_LIST_A, sm_A, ss_A, km_A, ks_A,
                 "Number of particles", "DupRate",
                 os.path.join(FIG_DIR, "exp_A_duprate.pdf"))

    # Experiment B
    sm_B, ss_B = _unzip(sold_kid_B)
    km_B, ks_B = _unzip(kld_kid_B)
    _plot_metric(N_TRAIN_LIST_B, sm_B, ss_B, km_B, ks_B,
                 "Number of training centers", "KID",
                 os.path.join(FIG_DIR, "exp_B_kid.pdf"))

    sm_B, ss_B = _unzip(sold_dup_B)
    km_B, ks_B = _unzip(kld_dup_B)
    _plot_metric(N_TRAIN_LIST_B, sm_B, ss_B, km_B, ks_B,
                 "Number of training centers", "DupRate",
                 os.path.join(FIG_DIR, "exp_B_duprate.pdf"))

    _plot_timing(N_TRAIN_LIST_B, sold_time_B, kld_time_B,
                 "Number of training centers",
                 os.path.join(FIG_DIR, "exp_B_timing.pdf"))

    # Print summary tables
    print("\n=== Experiment A Summary (N_TRAIN=1000) ===")
    print(f"{'N_GEN':>8}  {'SOLD KID':>10}  {'SOLD Dup':>10}  "
          f"{'KLD KID':>10}  {'KLD Dup':>10}")
    print("-" * 56)
    for i, n_gen in enumerate(N_GEN_LIST_A):
        sk, sd = sold_kid_A[i], sold_dup_A[i]
        kk, kd = kld_kid_A[i],  kld_dup_A[i]
        print(f"{n_gen:>8d}  "
              f"{sk[0]:>6.4f}±{sk[1]:.4f}  {sd[0]:>5.3f}±{sd[1]:.3f}  "
              f"{kk[0]:>6.4f}±{kk[1]:.4f}  {kd[0]:>5.3f}±{kd[1]:.3f}")
    print(f"\nCPU time at N_GEN={N_GEN_TIMING}:  "
          f"SOLD={sold_time_A:.3f} ms/sample  KLD={kld_time_A:.3f} ms/sample")

    print("\n=== Experiment B Summary (N_GENERATE=1500) ===")
    print(f"{'N_TRAIN':>8}  {'SOLD KID':>10}  {'SOLD Dup':>10}  "
          f"{'KLD KID':>10}  {'KLD Dup':>10}  {'SOLD ms':>8}  {'KLD ms':>8}")
    print("-" * 72)
    for i, n_train in enumerate(N_TRAIN_LIST_B):
        sk, sd = sold_kid_B[i], sold_dup_B[i]
        kk, kd = kld_kid_B[i],  kld_dup_B[i]
        print(f"{n_train:>8d}  "
              f"{sk[0]:>6.4f}±{sk[1]:.4f}  {sd[0]:>5.3f}±{sd[1]:.3f}  "
              f"{kk[0]:>6.4f}±{kk[1]:.4f}  {kd[0]:>5.3f}±{kd[1]:.3f}  "
              f"{sold_time_B[i]:>7.3f}  {kld_time_B[i]:>7.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
