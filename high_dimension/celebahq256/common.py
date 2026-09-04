"""
common.py
=================
Shared building blocks for the CelebA-HQ 256x256 DiffAE-latent generation
experiment.
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

import config as cfg                   # noqa: E402 (sets up sys.path)
from metrics_celeba import (                    # noqa: E402
    compute_kid, compute_recall, compute_duprate, compute_tau,
    compute_fid_from_features, extract_inception_features,
)

import sampling_algo as _sampling_algo_mod      # noqa: E402
_sig = inspect.signature(_sampling_algo_mod.sample_class_overdamped_manifold_knn)
if "X_class" not in _sig.parameters:
    raise ImportError(
        "`sampling_algo` resolved to augmentation_handwritten's Z_class/"
        "whitening-based module instead of release/common/sampling_algo.py's "
        "X_class-based, no-whitening one -- check config.py's sys.path "
        f"order (found sys.path={__import__('sys').path[:4]} ...). Resolved "
        f"module file: {_sampling_algo_mod.__file__}"
    )
sample_class_overdamped_manifold_knn = _sampling_algo_mod.sample_class_overdamped_manifold_knn

import tilting_lion as TL                        # noqa: E402  (release/common)


# ============================================================
# 1. delta / sigma / h / gamma parameterisation
# ============================================================

def mean_nn_distance_chunked(U: np.ndarray, *, n_sample: int = None,
                             chunk: int = None, seed: int = 0) -> float:
    n_sample = cfg.NN_N_SAMPLE if n_sample is None else n_sample
    chunk    = cfg.NN_CHUNK    if chunk    is None else chunk
    U = np.asarray(U, dtype=np.float32)
    n = U.shape[0]
    n_sample = min(n_sample, n)
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, n_sample, replace=False)
    Zt = jnp.asarray(U)
    bn = jnp.sum(Zt * Zt, axis=1)
    nn_dists = []
    for i in range(0, n_sample, chunk):
        idx = sub[i:i + chunk]
        q = Zt[idx]
        d2 = jnp.sum(q * q, axis=1)[:, None] + bn[None, :] - 2.0 * (q @ Zt.T)
        d2 = d2.at[jnp.arange(len(idx)), idx].set(jnp.inf)   # exclude self
        nn_dists.append(np.sqrt(np.maximum(np.asarray(jnp.min(d2, axis=1)), 0.0)))
    return float(np.mean(np.concatenate(nn_dists)))


def delta_sigma(c_delta: float, c_sigma: float, nn: float, dsub: int = None):
    """delta = c_delta * NN / sqrt(dsub),  sigma = c_sigma * NN  ."""
    dsub = cfg.PROJ_K if dsub is None else dsub
    delta = c_delta * nn / np.sqrt(dsub)
    sigma = c_sigma * nn
    return float(delta), float(sigma)


def h_sold(delta: float, c_sold: float = None) -> float:
    """MM-SOLD LM step size, the closed-form expression:  h = 0.2 * delta^2."""
    c_sold = cfg.SOLD_H_COEF if c_sold is None else c_sold
    return float(c_sold * delta ** 2)


def h_baoab(delta: float) -> float:
    """
    Plug-in/SG-BAOAB step size for CelebA-HQ, the closed-form expression:  h = delta / 4.
    """
    return float(delta / 4.0)


def gamma_baoab(U: np.ndarray) -> float:
    """BAOAB friction, gamma = 1 / sqrt(lambda_max(Sigma_190))."""
    U = np.asarray(U, dtype=np.float64)
    Uc = U - U.mean(axis=0, keepdims=True)
    Sigma = (Uc.T @ Uc) / max(len(U) - 1, 1)
    lam_max = float(np.linalg.eigvalsh(Sigma)[-1])
    return float(1.0 / np.sqrt(max(lam_max, 1e-12)))


# ============================================================
# 2. Top-190 subspace: projection + Gaussian tail completion
# ============================================================

def to_subspace(Z: np.ndarray, mean: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """Full 512-dim DiffAE latents -> top-190 subspace coordinates."""
    Z = np.asarray(Z, dtype=np.float64)
    return ((Z - mean[None, :]) @ Vt).astype(np.float32)


def from_subspace_gaussian_tail(U: np.ndarray, mean: np.ndarray, Vt: np.ndarray,
                                Vb: np.ndarray, lambda_tail: np.ndarray, *,
                                seed: int = 0, rng: np.random.Generator = None
                                ) -> np.ndarray:
    """Top-190 coordinates -> full 512-dim DiffAE latents, with the
    discarded 322 tail directions drawn INDEPENDENTLY PER SAMPLE from
    N(0, diag(lambda_tail))
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    U = np.asarray(U, dtype=np.float64)
    n = U.shape[0]
    dtail = Vb.shape[1]
    Ub = rng.standard_normal((n, dtail)).astype(np.float64) * np.sqrt(
        np.maximum(lambda_tail, 0.0))[None, :]
    Z = mean[None, :] + U @ Vt.T + Ub @ Vb.T
    return Z.astype(np.float32)


# ============================================================
# 2b. RBF-mixture kernel latent-space KID
# ============================================================

def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T


def rbf_median_bandwidth(X: np.ndarray, *, n_sample: int = None, seed: int = None
                         ) -> float:
    """Median-heuristic RBF bandwidth, computed ONCE on real (training)
    latents and reused for every later KID computation"""
    n_sample = cfg.RBF_BW_N_SAMPLE if n_sample is None else n_sample
    seed = cfg.RBF_BW_SEED if seed is None else seed
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), n_sample)
    j = rng.integers(0, len(X), n_sample)
    d2 = _sqdist(X[i], X[j]).flatten()
    d2 = d2[d2 > 0]
    return float(np.sqrt(0.5 * np.median(d2)))


def _rbfmix_kernel(a: np.ndarray, b: np.ndarray, s: float,
                   scales=None) -> np.ndarray:
    """Characteristic mixture-of-RBF kernel: sum of RBFs at `scales` x s
    (default {0.25, 0.5, 1, 2, 4} x the median-heuristic bandwidth)."""
    scales = cfg.RBF_MIX_SCALES if scales is None else scales
    D = _sqdist(a, b)
    out = np.zeros_like(D)
    for c in scales:
        out += np.exp(-D / (2.0 * (c * s) ** 2))
    return out


def _mmd2_unbiased_kernel(X: np.ndarray, Y: np.ndarray, kern) -> float:
    """Unbiased MMD^2 for an arbitrary (already-instantiated) kernel
    function"""
    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = kern(X, X), kern(Y, Y), kern(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd = (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
          - 2.0 * Kxy.mean())
    return float(mmd)


def compute_kid_rbfmix(feats_real: np.ndarray, feats_gen: np.ndarray, bw: float, *,
                       subset_size: int = None, n_subsets: int = None,
                       scales=None, rng: np.random.Generator = None) -> tuple:
    """
    RBF-mixture-kernel KID
    """
    subset_size = cfg.RBF_KID_SUBSET if subset_size is None else subset_size
    n_subsets = cfg.RBF_KID_N_SUBSETS if n_subsets is None else n_subsets
    if rng is None:
        rng = np.random.default_rng(0)
    feats_real = np.asarray(feats_real, dtype=np.float64)
    feats_gen = np.asarray(feats_gen, dtype=np.float64)
    subset_size = min(subset_size, len(feats_real), len(feats_gen))
    kern = lambda a, b: _rbfmix_kernel(a, b, bw, scales)
    vals = []
    for _ in range(n_subsets):
        ia = rng.choice(len(feats_real), size=subset_size, replace=False)
        ib = rng.choice(len(feats_gen), size=subset_size, replace=False)
        vals.append(_mmd2_unbiased_kernel(feats_real[ia], feats_gen[ib], kern))
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


# ============================================================
# 3. Evaluation protocol: N independent repeats, LATENT-SPACE metrics
#    (used across the full 200-cell grids -- no decoding)
# ============================================================

def repeat_and_evaluate_latent(gen_fn_190, U_test190, Z_test512, U_train190, tau,
                               mean, Vt, Vb, lambda_tail, *,
                               n_repeats=None, seed=0, rbf_bw=None,
                               kid_subset=None, kid_n_subsets=None,
                               kid_degree=None, k_recall=None,
                               return_last_gen=False):
    """
    gen_fn_190(seed) -> U_gen190 (n_per_repeat, 190) top-190 subspace samples.
    """
    n_repeats     = cfg.N_REPEATS     if n_repeats     is None else n_repeats
    kid_subset    = cfg.KID_SUBSET    if kid_subset    is None else kid_subset
    kid_n_subsets = cfg.KID_N_SUBSETS if kid_n_subsets is None else kid_n_subsets
    kid_degree    = cfg.KID_DEGREE    if kid_degree    is None else kid_degree
    k_recall      = cfg.K_RECALL      if k_recall      is None else k_recall
    if rbf_bw is None:
        raise ValueError("repeat_and_evaluate_latent requires rbf_bw (from "
                         "meta.json's 'rbf_bw', computed once by step1) -- "
                         "the RBF-mixture kernel has no meaningful default "
                         "bandwidth without it.")

    kid_vals, kid_poly3_vals, recall_vals, dup_vals = [], [], [], []
    last_U_gen = None
    for r in range(n_repeats):
        U_gen190 = np.asarray(gen_fn_190(seed + r), dtype=np.float32)
        Z_gen512 = from_subspace_gaussian_tail(
            U_gen190, mean, Vt, Vb, lambda_tail, seed=seed * 1000 + r + 500000)
        rng = np.random.default_rng(seed * 1000 + r)
        kid_m, _ = compute_kid_rbfmix(Z_test512, Z_gen512, rbf_bw, rng=rng)
        kid_subset_eff = min(kid_subset, len(Z_test512), len(Z_gen512))
        kid_p3, _ = compute_kid(Z_test512, Z_gen512, subset_size=kid_subset_eff,
                                n_subsets=kid_n_subsets, degree=kid_degree, rng=rng)
        rec = compute_recall(U_test190, U_gen190, k_neighbour=k_recall)
        dup = compute_duprate(U_gen190, U_train190, tau)
        kid_vals.append(kid_m); kid_poly3_vals.append(kid_p3)
        recall_vals.append(rec); dup_vals.append(dup)
        last_U_gen = U_gen190

    out = {
        "kid_mean": float(np.mean(kid_vals)), "kid_std": float(np.std(kid_vals)),
        "kid_poly3_mean": float(np.mean(kid_poly3_vals)),
        "kid_poly3_std": float(np.std(kid_poly3_vals)),
        "recall_mean": float(np.mean(recall_vals)), "recall_std": float(np.std(recall_vals)),
        "duprate_mean": float(np.mean(dup_vals)), "duprate_std": float(np.std(dup_vals)),
        "kid_vals": [float(v) for v in kid_vals],
        "kid_poly3_vals": [float(v) for v in kid_poly3_vals],
        "recall_vals": [float(v) for v in recall_vals],
        "duprate_vals": [float(v) for v in dup_vals],
        "n_repeats": n_repeats,
    }
    if return_last_gen:
        out["last_U_gen"] = last_U_gen
    return out


# ============================================================
# 4. Image-space decode-and-score (used ONCE per method, on its selected cell)
# ============================================================

def decode_batched(U190_or_Z512, mean, Vt, Vb, lambda_tail, render_fn, *,
                   is_subspace=True, batch=None, seed=0):
    """
    Decode a batch of latents to (N, 256, 256, 3) uint8 images via the
    TensorRT DiffAE decoder's `render_fn` (from dpm_decode.make_renderer).
    """
    import torch
    batch = cfg.DECODE_BATCH if batch is None else batch
    if is_subspace:
        Z512 = from_subspace_gaussian_tail(U190_or_Z512, mean, Vt, Vb, lambda_tail, seed=seed)
    else:
        Z512 = np.asarray(U190_or_Z512, dtype=np.float32)
    N = len(Z512)
    out = np.empty((N, 256, 256, 3), dtype=np.uint8)
    torch.manual_seed(seed)
    zt = torch.from_numpy(Z512).float()
    for i in range(0, N, batch):
        b = zt[i:i + batch]
        nb = len(b)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        b = b.to(device)
        if nb < batch:
            b = torch.cat([b, b.new_zeros(batch - nb, b.shape[1])])
        x_T = torch.randn(batch, 3, 256, 256, device=device)
        with torch.no_grad():
            img = render_fn(x_T, b).clamp(0, 1)[:nb]
        out[i:i + nb] = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return out


def decode_and_score_images(gen_fn_190, mean, Vt, Vb, lambda_tail, render_fn,
                            inception_model, inception_device, f_ref, *,
                            rbf_bw_img=None, n_repeats=None, seed=0, decode_batch=None):
    n_repeats = cfg.N_REPEATS if n_repeats is None else n_repeats
    if rbf_bw_img is None:
        raise ValueError("decode_and_score_images requires rbf_bw_img (from "
                         "meta.json's 'rbf_bw_img', computed once by step1 on "
                         "f_ref) to report the RBF-mixture image-KID alongside poly3.")
    fid_vals, kid_vals, kid_rbf_vals = [], [], []
    for r in range(n_repeats):
        U_gen190 = np.asarray(gen_fn_190(seed + r), dtype=np.float32)
        imgs = decode_batched(U_gen190, mean, Vt, Vb, lambda_tail, render_fn,
                              is_subspace=True, batch=decode_batch,
                              seed=seed * 1000 + r + 700000)
        feats_gen = extract_inception_features(
            imgs.astype(np.float32) / 255.0, inception_model, inception_device)
        fid_vals.append(compute_fid_from_features(f_ref, feats_gen))
        rng = np.random.default_rng(seed * 1000 + r)
        kid_m, _ = compute_kid(f_ref, feats_gen, subset_size=min(cfg.KID_SUBSET, len(feats_gen)),
                               n_subsets=cfg.KID_N_SUBSETS, degree=cfg.KID_DEGREE, rng=rng)
        kid_vals.append(kid_m)
        kid_rbf_m, _ = compute_kid_rbfmix(f_ref, feats_gen, rbf_bw_img, rng=rng)
        kid_rbf_vals.append(kid_rbf_m)
    return {
        "fid_mean": float(np.mean(fid_vals)), "fid_std": float(np.std(fid_vals)),
        "kid_mean": float(np.mean(kid_vals)), "kid_std": float(np.std(kid_vals)),
        "kid_rbf_mean": float(np.mean(kid_rbf_vals)), "kid_rbf_std": float(np.std(kid_rbf_vals)),
        "fid_vals": [float(v) for v in fid_vals], "kid_vals": [float(v) for v in kid_vals],
        "kid_rbf_vals": [float(v) for v in kid_rbf_vals],
        "n_repeats": n_repeats,
    }


# ============================================================
# 5. Timing helpers
# ============================================================

def measure_cpu_time(fn, *args, n_warmup=None, n_timed=None, **kwargs):
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


def measure_gpu_time(fn, *args, n_warmup=1, n_timed=1, **kwargs):
    for _ in range(n_warmup):
        out = fn(*args, **kwargs); jax.block_until_ready(out)
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs); jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def best_cell_filtered(cells, *, duprate_key="duprate_mean", recall_key="recall_mean",
                       kid_key="kid_mean", duprate_cap=None, recall_floor=None):
    if duprate_cap is None: duprate_cap = cfg.REP_CELL_DUPRATE_CAP
    if recall_floor is None: recall_floor = cfg.REP_CELL_RECALL_FLOOR
    items = list(cells.values()) if isinstance(cells, dict) else list(cells)
    dup_ok = [c for c in items if c[duprate_key] <= duprate_cap]
    both_ok = [c for c in dup_ok if c[recall_key] >= recall_floor]
    if both_ok:
        return min(both_ok, key=lambda c: c[kid_key])
    if dup_ok:
        best = max(dup_ok, key=lambda c: c[recall_key])
        print(f"  [WARNING] best_cell_filtered: no {duprate_key}<={duprate_cap} cell "
              f"also has {recall_key}>={recall_floor}; falling back.")
        return best
    best = min(items, key=lambda c: c[duprate_key])
    print(f"  [WARNING] best_cell_filtered: no cell has {duprate_key} <= {duprate_cap}; falling back.")
    return best


def _pick_contrastive_cell(mmsold_grid, cfdm_grid, duprate_cap, recall_floor):
    mmsold_cells = list(mmsold_grid.values())
    ok = [c for c in mmsold_cells
          if c["duprate_mean"] <= duprate_cap and c["recall_mean"] >= recall_floor]
    if not ok:
        print(f"  [WARNING] _pick_contrastive_cell: no MM-SOLD cell has "
              f"duprate_mean<={duprate_cap} and recall_mean>={recall_floor}; "
              f"falling back to best_cell_filtered (sigma-CFDM contrast not enforced)")
        return best_cell_filtered(mmsold_cells, duprate_cap=duprate_cap,
                                  recall_floor=recall_floor)

    cfdm_by_sm = {(c["c_sigma"], c["M"]): c for c in cfdm_grid.values()}

    def recall_gap(c):
        cf = cfdm_by_sm.get((c["c_sigma"], c["M"]))
        return (c["recall_mean"] - cf["recall_mean"]) if cf is not None else -1e9

    best = max(ok, key=recall_gap)
    cf = cfdm_by_sm.get((best["c_sigma"], best["M"]))
    print(f"  contrastive pick: MM-SOLD Recall={best['recall_mean']:.4f} vs "
          f"sigma-CFDM Recall={cf['recall_mean']:.4f} at the same cell "
          f"(gap={recall_gap(best):+.4f}); sigma-CFDM KID={cf['kid_mean']:.4f}")
    return best


def _pick_table6_cell(mmsold_grid, plugin_grid, sg_grid, duprate_cap, recall_floor,
                      min_c_delta=0.3):
    def _index(grid):
        return {(c["c_delta"], c["c_sigma"]): c for c in grid.values() if c["M"] == 32}

    mm, pi, sg = _index(mmsold_grid), _index(plugin_grid), _index(sg_grid)
    shared = set(mm) & set(pi) & set(sg)
    candidates = [mm[k] for k in shared
                  if mm[k]["duprate_mean"] <= duprate_cap and mm[k]["recall_mean"] >= recall_floor
                  and mm[k]["c_delta"] >= min_c_delta]
    if not candidates:
        print(f"  [WARNING] _pick_table6_cell: no M=32 c_delta>={min_c_delta} cell has "
              f"MM-SOLD duprate_mean<={duprate_cap} and recall_mean>={recall_floor}; "
              f"falling back to best_cell_filtered restricted to c_delta>={min_c_delta} (all M)")
        restricted = [c for c in mmsold_grid.values() if c["c_delta"] >= min_c_delta]
        best = best_cell_filtered(restricted or list(mmsold_grid.values()),
                                  duprate_cap=duprate_cap, recall_floor=recall_floor)
        return best["c_delta"], best["c_sigma"], best

    def sg_improvement(c):
        key = (c["c_delta"], c["c_sigma"])
        pc, sc = pi[key], sg[key]
        return (pc["moment_res_cov"] / pc["moment_res_cov_floor"]
                - sc["moment_res_cov"] / sc["moment_res_cov_floor"])

    best_mm = max(candidates, key=sg_improvement)
    key = (best_mm["c_delta"], best_mm["c_sigma"])
    pc, sc = pi[key], sg[key]
    print(f"  CelebA-HQ posterior-moment-validation table / CelebA-HQ grid-level heatmap figure cell scan: {len(candidates)}/{len(shared)} M=32 cells pass "
          f"MM-SOLD's DupRate/Recall bar; picked c_delta={best_mm['c_delta']} "
          f"c_sigma={best_mm['c_sigma']} (Plug-in cov/floor="
          f"{pc['moment_res_cov']/pc['moment_res_cov_floor']:.2f}, SG cov/floor="
          f"{sc['moment_res_cov']/sc['moment_res_cov_floor']:.2f})")
    return best_mm["c_delta"], best_mm["c_sigma"], best_mm


# ============================================================
# 7. Atomic grid-JSON I/O
# ============================================================

def load_grid_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data.get("grid", {})
    return {}


def save_grid_json(path: str, meta: dict, grid: dict):
    out = dict(meta); out["grid"] = grid
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)


# ============================================================
# 8. Visualisation helpers
# ============================================================

def seamless_grid(images, path, G=8):
    images = np.asarray(images)
    H, W = images.shape[1], images.shape[2]
    C = images.shape[3] if images.ndim == 4 else 1
    canvas = np.ones((G * H, G * W, C), dtype=np.float32) if C > 1 else np.ones((G * H, G * W), dtype=np.float32)
    for k in range(min(G * G, len(images))):
        r, c = k // G, k % G
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = images[k]
    fig = plt.figure(figsize=(G * W / 100, G * H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, cmap=("gray" if C == 1 else None), vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=100); plt.close(fig)
    return path


def save_nn_vis(query_imgs: np.ndarray, nn_imgs: np.ndarray, path: str,
                n_cols: int = 4, n_rows: int = 8, gap: int = 4):
    query_imgs = np.asarray(query_imgs)
    nn_imgs = np.asarray(nn_imgs)
    H, W = query_imgs.shape[1], query_imgs.shape[2]
    C = query_imgs.shape[3] if query_imgs.ndim == 4 else 1
    pair_w, pair_h = 2 * W, H
    cw = n_cols * pair_w + (n_cols - 1) * gap
    ch = n_rows * pair_h + (n_rows - 1) * gap
    shape = (ch, cw, C) if C > 1 else (ch, cw)
    canvas = np.ones(shape, dtype=np.float32)
    for k in range(min(n_cols * n_rows, len(query_imgs))):
        row, col = k // n_cols, k % n_cols
        y0, x0 = row * (pair_h + gap), col * (pair_w + gap)
        canvas[y0:y0 + pair_h, x0:x0 + W] = query_imgs[k]
        canvas[y0:y0 + pair_h, x0 + W:x0 + pair_w] = nn_imgs[k]
    fig = plt.figure(figsize=(cw / 100, ch / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, cmap=("gray" if C == 1 else None), vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def find_nn(U_query, U_ref):
    q2 = np.sum(U_query**2, axis=1, keepdims=True)
    r2 = np.sum(U_ref**2, axis=1, keepdims=True)
    cross = U_query @ U_ref.T
    dist2 = np.maximum(q2 + r2.T - 2.0*cross, 0.0)
    nn_idx = np.argmin(dist2, axis=1)
    nn_dist = np.sqrt(dist2[np.arange(len(U_query)), nn_idx])
    return nn_idx, nn_dist
