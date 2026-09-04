"""
common.py
============
Shared building blocks for the digit-8 generation experiment (top-40 pooled
subspace, NN-scaled delta/sigma, 7-method comparison).
"""
import inspect
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import config as cfg                     # noqa: E402 (sets up sys.path)
import tilting_lion as TL                    # noqa: E402 (release/common)
from nrae_model import (                     # noqa: E402 (augmentation_handwritten)
    make_nrae_state, decode_latents_nrae, load_params as nrae_load_params,
)
from classifier_model import load_params as clf_load_params   # noqa: E402
from metrics import (                        # noqa: E402 (dimension-agnostic)
    extract_features, compute_kid, compute_recall, compute_duprate,
)

import sampling_algo as _sampling_algo_mod   # noqa: E402
_sig = inspect.signature(_sampling_algo_mod.sample_class_overdamped_manifold)
if "X_class" not in _sig.parameters:
    raise ImportError(
        "`sampling_algo` resolved to augmentation_handwritten's Z_class/"
        "whitening-based module instead of release/common/sampling_algo.py's "
        "X_class-based, no-whitening one -- check config.py's sys.path "
        f"order (found sys.path={__import__('sys').path[:4]} ...). Resolved "
        f"module file: {_sampling_algo_mod.__file__}"
    )
sample_class_overdamped_manifold = _sampling_algo_mod.sample_class_overdamped_manifold


def mean_nn_distance(U: np.ndarray) -> float:
    """NN = (1/N) * sum_i min_{j != i} ||u_i - u_j||_2  ."""
    Uj = jnp.asarray(U, dtype=jnp.float32)
    sq = jnp.sum(Uj * Uj, axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (Uj @ Uj.T)
    dist2 = jnp.maximum(dist2, 0.0)
    n = Uj.shape[0]
    # exclude self: NOT `+ eye*inf` (0*inf = nan, corrupts every entry)
    dist2 = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist2)
    nn = jnp.sqrt(jnp.min(dist2, axis=1))
    return float(jnp.mean(nn))


def delta_sigma(c_delta: float, c_sigma: float, nn: float, dsub: int = 40):
    """delta = c_delta * NN / sqrt(dsub),  sigma = c_sigma * NN  ."""
    delta = c_delta * nn / np.sqrt(dsub)
    sigma = c_sigma * nn
    return float(delta), float(sigma)


def h_sold(delta: float, c_sold: float = 0.2) -> float:
    """MM-SOLD LM step size, the closed-form expression:  h = 0.2 * delta^2."""
    return float(c_sold * delta ** 2)


def h_baoab(delta: float) -> float:
    """Plug-in/SG-BAOAB step size, the closed-form expression:  h_BAOAB = delta / 20."""
    return float(delta / 20.0)


def gamma_baoab(U: np.ndarray) -> float:
    """
    BAOAB friction, the closed-form expression:  gamma = 1 / sqrt(lambda_max(Sigma_8)).
    """
    U = np.asarray(U, dtype=np.float64)
    Uc = U - U.mean(axis=0, keepdims=True)
    Sigma = (Uc.T @ Uc) / max(len(U) - 1, 1)
    lam_max = float(np.linalg.eigvalsh(Sigma)[-1])
    return float(1.0 / np.sqrt(max(lam_max, 1e-12)))


def load_nrae(ckpt_path: str, seed: int = 0):
    """Load the NRAE checkpoint with augmentation_handwritten's
    nrae_model.py code."""
    params_np, info = nrae_load_params(ckpt_path)
    latent_dim   = info.get("latent_dim",   cfg.LATENT_DIM)
    enc1_hidden  = info.get("enc1_hidden",  cfg.ENC1_HIDDEN)
    dec_hidden   = info.get("dec_hidden",   cfg.DEC_HIDDEN)
    n_dct        = info.get("n_dct",        cfg.N_DCT)
    unet_base_ch = info.get("unet_base_ch", cfg.UNET_BASE_CH)
    rng = jax.random.PRNGKey(seed)
    model, _ = make_nrae_state(
        rng, latent_dim=latent_dim, enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch)
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    print(f"  NRAE checkpoint: {ckpt_path}")
    print(f"  NRAE info: {info}")
    return model, params, n_dct


def load_clf(ckpt_path: str):
    clf_params, clf_info = clf_load_params(ckpt_path)
    print(f"  CLF checkpoint: {ckpt_path}")
    print(f"  CLF info: {clf_info}")
    return clf_params


def decode_from_subspace(U, mean_8, V, nrae_model, nrae_params, n_dct,
                         batch_size: int = 32):
    """Top-40 coords -> 100-dim NRAE latent (dropped 60 dims = digit-8 mean,
    via TL.from_subspace) -> decoded (n, 64, 64) image, clipped to [0,1]."""
    Z = TL.from_subspace(np.asarray(U), mean_8, V)
    imgs = decode_latents_nrae(nrae_model, nrae_params, Z, batch_size=batch_size)
    return np.clip(np.array(imgs), 0.0, 1.0)


def decode_and_extract(U_gen, mean_8, V, nrae_model, nrae_params, n_dct,
                       clf_params, batch_size: int = 32):
    """Top-40 samples -> decoded images -> 256-dim CNN features."""
    imgs_gen = decode_from_subspace(U_gen, mean_8, V, nrae_model, nrae_params,
                                    n_dct, batch_size=batch_size)
    return extract_features(imgs_gen, clf_params, batch_size=64)


def repeat_and_evaluate(generate_fn, feats_test, U_train8, tau, *,
                        n_repeats: int = None, seed: int = 0,
                        kid_subset: int = None, kid_n_subsets: int = None,
                        kid_degree: int = None, k_recall: int = None,
                        return_last_gen: bool = False):
    n_repeats     = cfg.N_REPEATS     if n_repeats     is None else n_repeats
    kid_subset    = cfg.KID_SUBSET    if kid_subset    is None else kid_subset
    kid_n_subsets = cfg.KID_N_SUBSETS if kid_n_subsets is None else kid_n_subsets
    kid_degree    = cfg.KID_DEGREE    if kid_degree    is None else kid_degree
    k_recall      = cfg.K_RECALL      if k_recall      is None else k_recall

    kid_vals, recall_vals, dup_vals = [], [], []
    last_U_gen = None
    for r in range(n_repeats):
        U_gen, feats_gen = generate_fn(seed + r)
        rng = np.random.default_rng(seed * 1000 + r)
        kid_subset_eff = min(kid_subset, len(feats_test), len(feats_gen))
        kid_m, _ = compute_kid(feats_test, feats_gen, subset_size=kid_subset_eff,
                               n_subsets=kid_n_subsets, degree=kid_degree, rng=rng)
        rec = compute_recall(feats_test, feats_gen, k_neighbour=k_recall)
        dup = compute_duprate(U_gen, U_train8, tau)
        kid_vals.append(kid_m)
        recall_vals.append(rec)
        dup_vals.append(dup)
        last_U_gen = U_gen

    out = {
        "kid_mean":     float(np.mean(kid_vals)),
        "kid_std":      float(np.std(kid_vals)),
        "recall_mean":  float(np.mean(recall_vals)),
        "recall_std":   float(np.std(recall_vals)),
        "duprate_mean": float(np.mean(dup_vals)),
        "duprate_std":  float(np.std(dup_vals)),
        "kid_vals":     [float(v) for v in kid_vals],
        "recall_vals":  [float(v) for v in recall_vals],
        "duprate_vals": [float(v) for v in dup_vals],
        "n_repeats":    n_repeats,
    }
    if return_last_gen:
        out["last_U_gen"] = last_U_gen
    return out


def measure_cpu_time(fn, *args, n_warmup=None, n_timed=None, **kwargs):
    """Run fn on CPU (forces JAX device placement), warm up, then average."""
    n_warmup = cfg.TIMING_N_WARMUP if n_warmup is None else n_warmup
    n_timed  = cfg.TIMING_N_TIMED  if n_timed  is None else n_timed
    with jax.default_device(jax.devices("cpu")[0]):
        for _ in range(n_warmup):
            fn(*args, **kwargs)
        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def measure_gpu_time(fn, *args, n_warmup: int = 1, n_timed: int = 1, **kwargs):
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def k_label(K):
    """Render the KNN-PFlow 'K=N' sentinel consistently in filenames/JSON."""
    return "N" if (K == "N") else str(K)


def best_cell_filtered(cells, *, duprate_key="duprate_mean", recall_key="recall_mean",
                       kid_key="kid_mean", duprate_cap=None, recall_floor=None):
    if duprate_cap is None:
        duprate_cap = cfg.REP_CELL_DUPRATE_CAP
    if recall_floor is None:
        recall_floor = cfg.REP_CELL_RECALL_FLOOR
    items = list(cells.values()) if isinstance(cells, dict) else list(cells)
    dup_ok = [c for c in items if c[duprate_key] <= duprate_cap]
    both_ok = [c for c in dup_ok if c[recall_key] >= recall_floor]
    if both_ok:
        return min(both_ok, key=lambda c: c[kid_key])
    if dup_ok:
        best = max(dup_ok, key=lambda c: c[recall_key])
        print(f"  [WARNING] best_cell_filtered: no {duprate_key}<={duprate_cap} cell "
              f"also has {recall_key}>={recall_floor}; falling back to the "
              f"highest-{recall_key} cell among the duprate-capped ones "
              f"({recall_key}={best[recall_key]:.4f}, {duprate_key}={best[duprate_key]:.4f}, "
              f"{kid_key}={best[kid_key]:.4f})")
        return best
    best = min(items, key=lambda c: c[duprate_key])
    print(f"  [WARNING] best_cell_filtered: no cell has {duprate_key} <= {duprate_cap}; "
          f"falling back to the lowest-{duprate_key} cell "
          f"({duprate_key}={best[duprate_key]:.4f}, {kid_key}={best[kid_key]:.4f})")
    return best


def load_grid_json(path: str) -> dict:
    """Return the 'grid' dict from an existing results JSON, or {} if none."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data.get("grid", {})
    return {}


def seamless_grid(images: np.ndarray, path: str, G: int = 8):
    """Tile G x G images into a seamless square PNG (white bg, black ink)."""
    H, W = images.shape[1], images.shape[2]
    canvas = np.ones((G * H, G * W), dtype=np.float32)
    for k in range(min(G * G, len(images))):
        r, c = k // G, k % G
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = images[k]
    fig = plt.figure(figsize=(G * W / 100, G * H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def find_nn(U_query: np.ndarray, U_ref: np.ndarray):
    """L2 nearest-neighbour search (e.g. in the top-40 subspace)."""
    q2 = np.sum(U_query ** 2, axis=1, keepdims=True)
    r2 = np.sum(U_ref ** 2, axis=1, keepdims=True)
    cross = U_query @ U_ref.T
    dist2 = np.maximum(q2 + r2.T - 2.0 * cross, 0.0)
    nn_idx = np.argmin(dist2, axis=1)
    nn_dist = np.sqrt(dist2[np.arange(len(U_query)), nn_idx])
    return nn_idx, nn_dist


def save_nn_vis(query_imgs: np.ndarray, nn_imgs: np.ndarray, path: str,
               n_cols: int = 4, n_rows: int = 8, gap: int = 4):
    """n_cols x n_rows grid of [generated | nearest-training] image pairs."""
    H, W = query_imgs.shape[1], query_imgs.shape[2]
    pair_w, pair_h = 2 * W, H
    cw = n_cols * pair_w + (n_cols - 1) * gap
    ch = n_rows * pair_h + (n_rows - 1) * gap
    canvas = np.ones((ch, cw), dtype=np.float32)
    for k in range(min(n_cols * n_rows, len(query_imgs))):
        row, col = k // n_cols, k % n_cols
        y0, x0 = row * (pair_h + gap), col * (pair_w + gap)
        canvas[y0:y0 + pair_h, x0:x0 + W] = query_imgs[k]
        canvas[y0:y0 + pair_h, x0 + W:x0 + pair_w] = nn_imgs[k]
    fig = plt.figure(figsize=(cw / 100, ch / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_grid_json(path: str, meta: dict, grid: dict):
    """Atomically persist {**meta, 'grid': grid} to path."""
    out = dict(meta)
    out["grid"] = grid
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)
