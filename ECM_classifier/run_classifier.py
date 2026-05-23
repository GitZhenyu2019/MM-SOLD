"""
run_classifier.py
=================
minimum ECM classifier using MM-SOLD moment-matched marginal distribution
with discriminative bias calibration on a held-out validation set.
"""

import os
import sys
import argparse
import json
import time
import pickle
import functools

import numpy as np
import scipy.linalg
from scipy.optimize import minimize as sp_minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import config

# Import from augmentation_handwritten (added to sys.path by config)
from data_utils import (
    load_images_and_digits,
    select_train_val_test_split,
    full_preprocess_pipeline,
)
from nrae_model import NRAEModel, encode_dataset_nrae, load_params
from sampling_algo import gradU_LDS_mc_stationary


# ============================================================
# 1. JAX helpers for energy computation
# ============================================================

@jax.jit
def _U_batch(Z, X, x_norm2, sigma):
    """GMM potential U(z) = -logsumexp(-||z - x_i||² / (2σ²)) over i."""
    z_norm2 = jnp.sum(Z * Z, axis=1)                              # (B,)
    dist2   = z_norm2[:, None] + x_norm2[None, :] - 2.0 * (Z @ X.T)  # (B, N)
    return -logsumexp(-0.5 * dist2 / (sigma * sigma), axis=1)    # (B,)


# Max MC samples per GPU call; keeps (M_CHUNK × B × N) distance tensor in budget
_M_CHUNK = 64


@functools.partial(jax.jit, static_argnames=("M",))
def _Vc_mc_chunk(key, Z, X, x_norm2, sigma, s, M):
    """
    Smoothed potential V_c(z) = E_ε[U_c(z + s·ε)] for one M-chunk (antithetic).
    """
    B, d = Z.shape
    half = M // 2
    eps_h = jax.random.normal(key, (half, d), dtype=Z.dtype)
    eps   = jnp.concatenate([eps_h, -eps_h], axis=0)        # (M, d)
    Zp    = Z[:, None, :] + s * eps[None, :, :]             # (B, M, d)
    U_flat = _U_batch(Zp.reshape((-1, d)), X, x_norm2, sigma)  # (B*M,)
    return U_flat.reshape((B, M)).mean(axis=1)               # (B,)


def _Vc_mc_batch(key, Z, X, x_norm2, sigma, s, M, m_chunk=_M_CHUNK):
    """
    Smoothed potential V_c(z) with M-chunking to avoid GPU OOM.
    Splits M MC samples into chunks of size m_chunk, averages results.
    """
    acc = None
    n_chunks = 0
    m_remaining = M
    while m_remaining > 0:
        m_use = min(m_chunk, m_remaining)
        key, k = jax.random.split(key)
        chunk_val = np.array(_Vc_mc_chunk(k, Z, X, x_norm2, sigma, s, m_use))
        acc = chunk_val if acc is None else acc + chunk_val
        n_chunks += 1
        m_remaining -= m_use
    return acc / n_chunks   # (B,)


# ============================================================
# 2. Per-class energy estimation
# ============================================================

def estimate_class_params(Xw_c, mu_star_c, Sigma_star_c, xw_norm2_c,
                           sigma_gmm, sigma, M, key):
    """
    Estimate λ_c and Λ_c for one class from training latents in whitened space.
    """
    d   = Xw_c.shape[1]
    n_c = Xw_c.shape[0]

    # Smoothed scores at training points: g_{s,c}(z_i) for all i, shape (n_c, d)
    G_acc = None
    n_chunks = 0
    m_remaining = M
    while m_remaining > 0:
        m_use = min(_M_CHUNK, m_remaining)
        key, k_g = jax.random.split(key)
        g_chunk = np.array(gradU_LDS_mc_stationary(
            k_g, Xw_c, Xw_c, xw_norm2_c,
            sigma_gmm, sigma, m_use,
            shared_noise=False,
        ))
        G_acc = g_chunk if G_acc is None else G_acc + g_chunk
        n_chunks += 1
        m_remaining -= m_use
    G_c = G_acc / n_chunks   # (n_c, d)

    # λ_c = -mean(g_{s,c}(z_i))
    lambda_c = -G_c.mean(axis=0)   # (d,)

    # C_c[j,k] = mean_i (z_i - μ*)_j · g_{s,c,k}(z_i)
    diff = np.array(Xw_c) - mu_star_c[None, :]   # (n_c, d)
    C_c  = diff.T @ G_c / n_c                     # (d, d)

    # S_c = I - sym(C_c)
    S_c = np.eye(d, dtype=np.float64) - 0.5 * (C_c + C_c.T)

    # Lyapunov equation: Σ_c* · Λ_c + Λ_c · Σ_c* = 2·S_c
    try:
        Lambda_c = scipy.linalg.solve_continuous_lyapunov(
            Sigma_star_c.astype(np.float64),
            2.0 * S_c,
        )
    except np.linalg.LinAlgError:
        print("    [WARNING] Lyapunov solve failed; using Lambda_c = 0")
        Lambda_c = np.zeros((d, d), dtype=np.float64)

    return lambda_c.astype(np.float32), Lambda_c.astype(np.float32)


def compute_class_energy(Xw_test_c, mu_star_c, lambda_c, Lambda_c,
                          Xw_train_c, xw_norm2_c,
                          sigma_gmm, sigma, M, key, batch_size=500):
    """
    Compute total energy E_c(z) for all test samples in one class.
    """
    N_test = Xw_test_c.shape[0]

    # --- Smoothed potential V_c (batched over test samples + M-chunked internally) ---
    V_c_chunks = []
    for i in range(0, N_test, batch_size):
        chunk = jnp.asarray(Xw_test_c[i:i + batch_size])
        key, k_v = jax.random.split(key)
        vc = _Vc_mc_batch(k_v, chunk, Xw_train_c, xw_norm2_c,
                          sigma_gmm, sigma, M)
        V_c_chunks.append(vc)   # already numpy from _Vc_mc_batch
    V_c = np.concatenate(V_c_chunks)   # (N_test,)

    # --- Linear term: λ_c^T · z ---
    linear_c = Xw_test_c @ lambda_c    # (N_test,)

    # --- Quadratic term: ½ (z - μ*)^T Λ_c (z - μ*) ---
    diff_c = Xw_test_c - mu_star_c[None, :]          # (N_test, d)
    quad_c = 0.5 * np.einsum('bi,ij,bj->b',
                              diff_c, Lambda_c, diff_c)  # (N_test,)

    return V_c + linear_c + quad_c   # (N_test,)


# ============================================================
# 2.5 Bias calibration on validation set
# ============================================================

def calibrate_biases(E_val, y_val, n_classes):
    """
    Learn per-class additive biases b_c by minimising cross-entropy on val set.
    """
    E = np.asarray(E_val, dtype=np.float64)   # (N, C)
    N = len(y_val)
    one_hot = np.zeros((N, n_classes), dtype=np.float64)
    one_hot[np.arange(N), y_val] = 1.0

    def loss_and_grad(b):
        logits = -E - b[None, :]                                   # (N, C)
        logits = logits - logits.max(axis=1, keepdims=True)        # stable
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum(axis=1, keepdims=True)          # (N, C)
        nll    = -np.mean(np.sum(one_hot * np.log(probs + 1e-12), axis=1))
        grad   = (one_hot - probs).mean(axis=0)
        return float(nll), grad

    result = sp_minimize(
        loss_and_grad, np.zeros(n_classes), jac=True,
        method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12},
    )
    return result.x.astype(np.float32)


# ============================================================
# 3. Metrics and plotting
# ============================================================

def compute_metrics(y_true, y_pred, E_matrix):
    """Compute all classification metrics for one grid cell."""
    acc = float(np.mean(y_pred == y_true))

    # Macro F1 and per-class F1
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # Per-class accuracy
    acc_per = []
    for c in range(config.N_CLASSES):
        mask = y_true == c
        acc_per.append(float(np.mean(y_pred[mask] == c)) if mask.any() else 0.0)

    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred,
                                labels=list(range(config.N_CLASSES)))

    # Pseudo-NLL: mean energy under the true class
    pseudo_nll = float(np.mean(E_matrix[np.arange(len(y_true)), y_true]))

    return {
        'acc':         acc,
        'macro_f1':    f1_macro,
        'per_class_f1': f1_per,
        'per_class_acc': acc_per,
        'conf_mat':    conf_mat.tolist(),
        'pseudo_nll':  pseudo_nll,
    }


def plot_confusion_matrix(conf_mat, sigma_gmm, sigma, M, fig_dir, suffix=''):
    """Blue-toned heatmap of confusion matrix (no cell text, colorbar on side)."""
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(conf_mat, cmap='Blues', aspect='auto',
                   interpolation='nearest')
    ax.set_xlabel('Predicted class', fontsize=11)
    ax.set_ylabel('True class', fontsize=11)
    ax.set_xticks(range(config.N_CLASSES))
    ax.set_yticks(range(config.N_CLASSES))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fname = os.path.join(fig_dir,
                         f'confmat_sgmm{sigma_gmm:.3f}_sigma{sigma:.2f}_M{M:02d}{suffix}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_summary_heatmaps(all_val_metrics, fig_dir):
    """
    Val-metric heatmaps: σ (x-axis) x δ/σ_gmm (y-axis), one plot per metric.
    """
    os.makedirs(fig_dir, exist_ok=True)
    sigmas     = config.SIGMA_GRID       # x-axis
    sigma_gmms = config.SIGMA_GMM_GRID  # y-axis (δ)
    M          = config.M_GRID[0]       # fixed (only one value)

    for metric_key, cmap in [
        ('acc',      'YlGn'),
        ('macro_f1', 'YlOrRd'),
    ]:
        # grid[i, j] = metric at (δ=sigma_gmms[i], σ=sigmas[j])
        grid = np.zeros((len(sigma_gmms), len(sigmas)))
        for i, sg in enumerate(sigma_gmms):
            for j, s in enumerate(sigmas):
                cell_key = f'sgmm{sg:.3f}_s{s:.2f}_M{M}'
                grid[i, j] = all_val_metrics.get(cell_key, {}).get(metric_key, 0.0)

        fig, ax = plt.subplots(figsize=(9, 3.5))
        im = ax.imshow(grid, cmap=cmap, aspect='auto',
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f'{s:.2f}' for s in sigmas], fontsize=8)
        ax.set_yticks(range(len(sigma_gmms)))
        ax.set_yticklabels([f'{sg:.3f}' for sg in sigma_gmms], fontsize=9)
        ax.set_xlabel('σ  (LDS smoothing bandwidth)', fontsize=11)
        ax.set_ylabel('δ  (GMM bandwidth)', fontsize=11)
        for i in range(len(sigma_gmms)):
            for j in range(len(sigmas)):
                ax.text(j, i, f'{grid[i, j]:.3f}',
                        ha='center', va='center', fontsize=7,
                        color='black' if grid[i, j] < 0.7 else 'white')
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
        plt.tight_layout()
        fname = os.path.join(fig_dir, f'val_{metric_key}.png')
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ============================================================
# 4. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=config.RESULTS_DIR)
    parser.add_argument('--fig_dir',     default=config.FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir,     exist_ok=True)

    print("=" * 60)
    print("MAP Bayesian Classifier — MM-SOLD + val calibration")
    print("=" * 60)

    # ── 1. Load and preprocess data ────────────────────────────
    print("\n[1] Loading data ...")
    images_np, digits_np = load_images_and_digits(config.RAW_IMG, config.RAW_LABEL)
    print(f"  Raw: {images_np.shape}, labels: {digits_np.shape}")

    (train_imgs, train_labels,
     val_imgs,   val_labels,
     test_imgs,  test_labels) = select_train_val_test_split(
        images_np, digits_np,
        n_train_per_class=config.N_TRAIN_PER_CLASS,
        n_val_per_class=config.N_VAL_PER_CLASS,
        n_test_per_class=config.N_TEST_PER_CLASS,
        seed=config.DATA_SEED,
    )
    print(f"  Train: {train_imgs.shape}, Val: {val_imgs.shape},"
          f" Test: {test_imgs.shape}")

    print("\n[1b] Preprocessing images (CPU) ...")
    # Force CPU to avoid GPU OOM when loading large raw image arrays
    _cpu = jax.devices('cpu')[0]
    with jax.default_device(_cpu):
        train_64, target_rms = full_preprocess_pipeline(train_imgs)
        val_64,   _          = full_preprocess_pipeline(val_imgs,
                                                         target_rms=target_rms)
        test_64,  _          = full_preprocess_pipeline(test_imgs,
                                                         target_rms=target_rms)
    train_64_np = np.array(train_64)
    val_64_np   = np.array(val_64)
    test_64_np  = np.array(test_64)
    print(f"  train_64: {train_64_np.shape}, val_64: {val_64_np.shape},"
          f" test_64: {test_64_np.shape}")

    # ── 2. Load NRAE and encode data ───────────────────────────
    print("\n[2] Loading NRAE from", config.NRAE_CKPT)
    params, info = load_params(config.NRAE_CKPT)
    model = NRAEModel(
        latent_dim=config.LATENT_DIM,
        enc1_hidden=config.ENC1_HIDDEN,
        dec_hidden=config.DEC_HIDDEN,
        n_dct=config.N_DCT,
        unet_base_ch=config.UNET_BASE_CH,
    )
    print(f"  NRAE info: {info}")

    print("  Encoding train set ...")
    Z_train = encode_dataset_nrae(model, params, train_64_np,
                                  n_dct=config.N_DCT,
                                  batch_size=config.ENCODE_BATCH)
    print(f"  Z_train: {Z_train.shape}")

    print("  Encoding val set ...")
    Z_val = encode_dataset_nrae(model, params, val_64_np,
                                n_dct=config.N_DCT,
                                batch_size=config.ENCODE_BATCH)
    print(f"  Z_val: {Z_val.shape}")

    print("  Encoding test set ...")
    Z_test = encode_dataset_nrae(model, params, test_64_np,
                                 n_dct=config.N_DCT,
                                 batch_size=config.ENCODE_BATCH)
    print(f"  Z_test: {Z_test.shape}")

    y_train = train_labels.astype(np.int32)
    y_val   = val_labels.astype(np.int32)
    y_test  = test_labels.astype(np.int32)

    # ── 3. Per-class statistics in original NRAE latent space (no whitening) ──
    print("\n[3] Precomputing per-class statistics (original NRAE latent space) ...")
    d         = config.LATENT_DIM
    Z_val_np  = Z_val.astype(np.float32)
    Z_test_np = Z_test.astype(np.float32)
    class_data = {}
    for c in range(config.N_CLASSES):
        mask_c = (y_train == c)
        Z_c    = Z_train[mask_c].astype(np.float32)     # (n_c, d)

        mu_star_c   = Z_c.mean(axis=0)                  # (d,)
        Zc_centered = Z_c - mu_star_c[None, :]
        Sigma_x_c   = (Zc_centered.T @ Zc_centered) / len(Z_c)
        z_norm2_c   = jnp.sum(jnp.asarray(Z_c) ** 2, axis=1)   # (n_c,)

        class_data[c] = dict(
            Xw_c       = Z_c,
            mu_star_c  = mu_star_c,
            Sigma_x_c  = Sigma_x_c.astype(np.float64),  # σ_gmm-independent part
            xw_norm2_c = z_norm2_c,
            Xw_val_c   = Z_val_np,   # all val samples (E_c evaluated for all)
            Xw_test_c  = Z_test_np,
        )
        print(f"  Class {c}: mu norm={np.linalg.norm(mu_star_c):.3f},"
              f" cov trace={np.trace(Sigma_x_c):.2f}")

    # ── 4. Grid search ─────────────────────────────────────────
    total_cells = (len(config.SIGMA_GMM_GRID) * len(config.SIGMA_GRID)
                   * len(config.M_GRID))
    print(f"\n[4] Grid search over "
          f"{len(config.SIGMA_GMM_GRID)} x {len(config.SIGMA_GRID)} x "
          f"{len(config.M_GRID)} = {total_cells} cells ...")
    print("    Biases calibrated on val; selection by val acc; test metrics stored.")

    all_val_metrics  = {}
    all_test_metrics = {}
    all_biases       = {}
    summary_rows     = []
    key              = jax.random.PRNGKey(42)

    for sigma_gmm in config.SIGMA_GMM_GRID:
        for sigma in config.SIGMA_GRID:
            for M in config.M_GRID:
                cell_key = f'sgmm{sigma_gmm:.3f}_s{sigma:.2f}_M{M}'
                t0 = time.time()
                print(f"\n  σ_gmm={sigma_gmm:.3f}, σ={sigma:.2f}, M={M}:")

                N_val_total  = len(y_val)
                N_test_total = len(y_test)
                E_val_matrix  = np.zeros((N_val_total,  config.N_CLASSES),
                                         dtype=np.float32)
                E_test_matrix = np.zeros((N_test_total, config.N_CLASSES),
                                          dtype=np.float32)

                for c in range(config.N_CLASSES):
                    cd = class_data[c]

                    # Sigma_star_c depends on σ_gmm → recompute each cell
                    Sigma_star_c = ((sigma_gmm ** 2) * np.eye(d, dtype=np.float64)
                                    + cd['Sigma_x_c'])

                    # ── estimate λ_c and Λ_c from training data ──
                    key, k_g = jax.random.split(key)
                    lambda_c, Lambda_c = estimate_class_params(
                        jnp.asarray(cd['Xw_c']),
                        cd['mu_star_c'],
                        Sigma_star_c,
                        cd['xw_norm2_c'],
                        sigma_gmm, sigma, M, k_g,
                    )

                    # ── save λ_c and Λ_c for inspection ──
                    param_dir = os.path.join(
                        args.results_dir,
                        f'sgmm{sigma_gmm:.3f}_sigma{sigma:.2f}_M{M:02d}',
                        'params',
                    )
                    os.makedirs(param_dir, exist_ok=True)
                    np.savetxt(os.path.join(param_dir, f'lambda_c{c}.txt'),
                               lambda_c.reshape(1, -1), fmt='%.6f')
                    np.savetxt(os.path.join(param_dir, f'Lambda_c{c}.txt'),
                               Lambda_c, fmt='%.6f')

                    # ── evaluate E_c on val set ──
                    key, k_ev = jax.random.split(key)
                    E_c_val = compute_class_energy(
                        cd['Xw_val_c'], cd['mu_star_c'],
                        lambda_c, Lambda_c,
                        jnp.asarray(cd['Xw_c']), cd['xw_norm2_c'],
                        sigma_gmm, sigma, M, k_ev,
                        batch_size=config.TEST_BATCH,
                    )
                    E_val_matrix[:, c] = E_c_val

                    # ── evaluate E_c on test set ──
                    key, k_et = jax.random.split(key)
                    E_c_test = compute_class_energy(
                        cd['Xw_test_c'], cd['mu_star_c'],
                        lambda_c, Lambda_c,
                        jnp.asarray(cd['Xw_c']), cd['xw_norm2_c'],
                        sigma_gmm, sigma, M, k_et,
                        batch_size=config.TEST_BATCH,
                    )
                    E_test_matrix[:, c] = E_c_test
                    print(f"    class {c}: "
                          f"E_val [{E_c_val.min():.2f}, {E_c_val.max():.2f}]  "
                          f"E_test [{E_c_test.min():.2f}, {E_c_test.max():.2f}]")

                # ── calibrate per-class biases on validation set ──
                biases = calibrate_biases(E_val_matrix, y_val, config.N_CLASSES)
                print(f"    biases: {np.round(biases, 3).tolist()}")

                # ── val classification with calibrated biases ──
                E_val_biased  = E_val_matrix  + biases[None, :]
                y_pred_val    = E_val_biased.argmin(axis=1)
                val_metrics   = compute_metrics(y_val,  y_pred_val,  E_val_biased)

                # ── test classification with same biases (not used for selection) ──
                E_test_biased = E_test_matrix + biases[None, :]
                y_pred_test   = E_test_biased.argmin(axis=1)
                test_metrics  = compute_metrics(y_test, y_pred_test, E_test_biased)

                elapsed = time.time() - t0
                print(f"    val:  acc={val_metrics['acc']:.4f}  "
                      f"macro_f1={val_metrics['macro_f1']:.4f}")
                print(f"    test: acc={test_metrics['acc']:.4f}  "
                      f"macro_f1={test_metrics['macro_f1']:.4f}  ({elapsed:.1f}s)")

                all_val_metrics[cell_key]  = val_metrics
                all_test_metrics[cell_key] = test_metrics
                all_biases[cell_key]       = biases.tolist()

                summary_rows.append(dict(
                    sigma_gmm=sigma_gmm, sigma=sigma, M=M,
                    val_acc=val_metrics['acc'],
                    val_f1=val_metrics['macro_f1'],
                    test_acc=test_metrics['acc'],
                    test_f1=test_metrics['macro_f1'],
                ))

                # ── save per-cell results ──
                cell_dir = os.path.join(
                    args.results_dir,
                    f'sgmm{sigma_gmm:.3f}_sigma{sigma:.2f}_M{M:02d}',
                )
                os.makedirs(cell_dir, exist_ok=True)
                with open(os.path.join(cell_dir, 'val_metrics.json'), 'w') as f:
                    json.dump(val_metrics,  f, indent=2)
                with open(os.path.join(cell_dir, 'test_metrics.json'), 'w') as f:
                    json.dump(test_metrics, f, indent=2)
                np.save(os.path.join(cell_dir, 'E_val_matrix.npy'),  E_val_matrix)
                np.save(os.path.join(cell_dir, 'E_test_matrix.npy'), E_test_matrix)
                np.save(os.path.join(cell_dir, 'biases.npy'),        biases)
                np.save(os.path.join(cell_dir, 'y_pred_val.npy'),    y_pred_val)
                np.save(os.path.join(cell_dir, 'y_pred_test.npy'),   y_pred_test)

    # ── 5. Select best by val; report test ────────────────────
    print("\n[5] Selecting best config by validation macro-F1 ...")

    best_key = max(all_val_metrics, key=lambda k: all_val_metrics[k]['macro_f1'])
    best_row = next(r for r in summary_rows
                    if (f'sgmm{r["sigma_gmm"]:.3f}_s{r["sigma"]:.2f}_M{r["M"]}'
                        == best_key))

    print(f"\n  Best config (val):  σ_gmm={best_row['sigma_gmm']:.3f}, "
          f"σ={best_row['sigma']:.2f}, M={best_row['M']}")
    print(f"  Val:  acc={best_row['val_acc']:.4f}   macro_f1={best_row['val_f1']:.4f}")

    best_test = all_test_metrics[best_key]
    print(f"\n  *** TEST SET RESULTS (best config) ***")
    print(f"  Test: acc={best_test['acc']:.4f}   macro_f1={best_test['macro_f1']:.4f}")
    print(f"  Per-class acc: {[round(x, 3) for x in best_test['per_class_acc']]}")
    print(f"  Per-class F1:  {[round(x, 3) for x in best_test['per_class_f1']]}")

    # Save full summary
    summary_path = os.path.join(args.results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'grid':             summary_rows,
            'best_key':         best_key,
            'best_val_metrics': all_val_metrics[best_key],
            'best_test_metrics': best_test,
            'best_biases':      all_biases[best_key],
        }, f, indent=2)
    print(f"\n  Saved {summary_path}")

    # Val-metric heatmaps (one per σ_gmm)
    print("\n[5b] Plotting val-metric heatmaps ...")
    plot_summary_heatmaps(all_val_metrics, args.fig_dir)

    # Confusion matrix for best config (test set)
    best_cm = np.array(best_test['conf_mat'])
    plot_confusion_matrix(best_cm,
                          best_row['sigma_gmm'], best_row['sigma'], best_row['M'],
                          args.fig_dir, suffix='_best_test')
    print(f"  Saved best-test confusion matrix.")

    print("\nDone.")


if __name__ == '__main__':
    main()
