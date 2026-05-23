"""
data_utils.py
=============
Data loading, geometric preprocessing (weighted centroid + PCA-align + scale
normalization), resize to 64x64, invert (ink=1), train/test split, and
mini-batch iteration utilities.
"""

import functools
import numpy as np
import os

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


# ============================================================
# 1. Raw data loading
# ============================================================

def load_images_and_digits(images_path: str, writerinfo_path: str):
    """Load handwritten digit images and digit labels from .npy files.

    Returns
    -------
    images : np.ndarray  shape (N, H, W), uint8 or float
    digits : np.ndarray  shape (N,), int32
    """
    images = np.load(images_path, allow_pickle=True)
    writerinfo = np.load(writerinfo_path, allow_pickle=True)
    images = np.asarray(images)
    writerinfo = np.asarray(writerinfo)

    if images.ndim != 3:
        raise ValueError(f"Expected images (N,H,W), got {images.shape}")
    if writerinfo.ndim != 2 or writerinfo.shape[0] != images.shape[0]:
        raise ValueError(
            f"WriterInfo shape {writerinfo.shape} incompatible with "
            f"images {images.shape}"
        )
    digits = writerinfo[:, 0].astype(np.int32)
    return images, digits


def to_float01_np(images: np.ndarray) -> np.ndarray:
    """Convert images to float32 in [0, 1]."""
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    images = images.astype(np.float32)
    mx = float(images.max())
    if mx <= 1.5:
        return np.clip(images, 0.0, 1.0)
    return np.clip(images / 255.0, 0.0, 1.0)


# ============================================================
# 2. Train / Test split
# ============================================================

def select_train_test_split(
    images_np: np.ndarray,
    digits_np: np.ndarray,
    *,
    n_train_per_class: int = 200,
    n_test_per_class: int = 50,
    seed: int = 0,
):
    """
    Randomly select n_train_per_class + n_test_per_class images per digit.
    """
    rng = np.random.default_rng(seed)
    train_imgs, train_lbls = [], []
    test_imgs, test_lbls = [], []

    for d in range(10):
        idx = np.where(digits_np == d)[0]
        total_needed = n_train_per_class + n_test_per_class
        if idx.size < total_needed:
            raise ValueError(
                f"Digit {d}: only {idx.size} samples available, "
                f"need {total_needed} ({n_train_per_class} train + "
                f"{n_test_per_class} test)"
            )
        chosen = rng.choice(idx, size=total_needed, replace=False)
        tr = chosen[:n_train_per_class]
        te = chosen[n_train_per_class:]

        train_imgs.append(images_np[tr])
        train_lbls.append(np.full(n_train_per_class, d, dtype=np.int32))
        test_imgs.append(images_np[te])
        test_lbls.append(np.full(n_test_per_class, d, dtype=np.int32))
        print(
            f"  digit {d}: {idx.size:>5} total  |  "
            f"{n_train_per_class} train  {n_test_per_class} test"
        )

    return (
        np.concatenate(train_imgs, axis=0),
        np.concatenate(train_lbls, axis=0),
        np.concatenate(test_imgs, axis=0),
        np.concatenate(test_lbls, axis=0),
    )


def select_train_val_test_split(
    images_np: np.ndarray,
    digits_np: np.ndarray,
    *,
    n_train_per_class: int = 1000,
    n_val_per_class: int = 58,
    n_test_per_class: int = 300,
    seed: int = 0,
):
    """
    Randomly select n_train + n_val + n_test images per digit class.
    """
    rng = np.random.default_rng(seed)
    train_imgs, train_lbls = [], []
    val_imgs,   val_lbls   = [], []
    test_imgs,  test_lbls  = [], []

    for d in range(10):
        idx = np.where(digits_np == d)[0]
        total_needed = n_train_per_class + n_val_per_class + n_test_per_class
        if idx.size < total_needed:
            raise ValueError(
                f"Digit {d}: only {idx.size} samples available, "
                f"need {total_needed} ({n_train_per_class} train + "
                f"{n_val_per_class} val + {n_test_per_class} test)"
            )
        chosen = rng.choice(idx, size=total_needed, replace=False)
        tr = chosen[:n_train_per_class]
        va = chosen[n_train_per_class:n_train_per_class + n_val_per_class]
        te = chosen[n_train_per_class + n_val_per_class:]

        train_imgs.append(images_np[tr])
        train_lbls.append(np.full(n_train_per_class, d, dtype=np.int32))
        val_imgs.append(images_np[va])
        val_lbls.append(np.full(n_val_per_class, d, dtype=np.int32))
        test_imgs.append(images_np[te])
        test_lbls.append(np.full(n_test_per_class, d, dtype=np.int32))
        print(
            f"  digit {d}: {idx.size:>5} total  |  "
            f"{n_train_per_class} train  {n_val_per_class} val  {n_test_per_class} test"
        )

    return (
        np.concatenate(train_imgs, axis=0),
        np.concatenate(train_lbls, axis=0),
        np.concatenate(val_imgs,   axis=0),
        np.concatenate(val_lbls,   axis=0),
        np.concatenate(test_imgs,  axis=0),
        np.concatenate(test_lbls,  axis=0),
    )


# ============================================================
# 3. Geometric preprocessing
# ============================================================

def _wrap_to_180(theta_deg):
    return (theta_deg + 90.0) % 180.0 - 90.0


def _make_xy_grid(H: int, W: int, dtype=jnp.float32):
    ys = jnp.arange(H, dtype=dtype)
    xs = jnp.arange(W, dtype=dtype)
    Y, X = jnp.meshgrid(ys, xs, indexing="ij")
    return Y, X


def _weighted_stats_single(im01, Y, X, dark_quantile, weight_power, eps=1e-12):
    thr = jnp.quantile(im01.reshape(-1), dark_quantile)
    mask = im01 <= thr
    ink = jnp.clip(1.0 - im01, 0.0, 1.0)
    w = (ink ** weight_power) * mask.astype(im01.dtype)
    sw = jnp.sum(w) + eps
    cy = jnp.sum(w * Y) / sw
    cx = jnp.sum(w * X) / sw
    dy, dx = Y - cy, X - cx
    sxx = jnp.sum(w * dx * dx) / sw
    syy = jnp.sum(w * dy * dy) / sw
    sxy = jnp.sum(w * dx * dy) / sw
    rms = jnp.sqrt(jnp.sum(w * (dx * dx + dy * dy)) / sw + eps)
    phi = 0.5 * jnp.arctan2(2.0 * sxy, sxx - syy)
    angle_deg = phi * (180.0 / jnp.pi)
    theta = _wrap_to_180(90.0 - angle_deg)
    return cy, cx, rms, theta


def _apply_affine_single(im01, cy, cx, rms, theta_deg, target_rms, Y, X):
    H, W = im01.shape
    c0y = (H - 1) / 2.0
    c0x = (W - 1) / 2.0
    scale = target_rms / jnp.maximum(rms, 1e-6)
    y0 = Y - c0y
    x0 = X - c0x
    th = theta_deg * (jnp.pi / 180.0)
    c = jnp.cos(th)
    s = jnp.sin(th)
    xr =  c * x0 + s * y0
    yr = -s * x0 + c * y0
    xs = xr / jnp.maximum(scale, 1e-6)
    ys = yr / jnp.maximum(scale, 1e-6)
    coords = jnp.stack([ys + cy, xs + cx], axis=0)
    return map_coordinates(im01, coords, order=1, mode="constant", cval=1.0)


@functools.partial(jax.jit, static_argnames=("dark_quantile", "weight_power"))
def _preprocess_batch_stats(images01, dark_quantile, weight_power):
    H, W = images01.shape[1], images01.shape[2]
    Y, X = _make_xy_grid(H, W, dtype=images01.dtype)
    stats_fn = lambda im: _weighted_stats_single(im, Y, X, dark_quantile, weight_power)
    cy, cx, rms, theta = jax.vmap(stats_fn)(images01)
    return cy, cx, rms, theta


@functools.partial(jax.jit, static_argnames=("dark_quantile", "weight_power"))
def _warp_batch(images01, cy, cx, rms, theta_deg, target_rms,
                dark_quantile, weight_power):
    H, W = images01.shape[1], images01.shape[2]
    Y, X = _make_xy_grid(H, W, dtype=images01.dtype)
    warp_fn = lambda im, _cy, _cx, _rms, _th: _apply_affine_single(
        im, _cy, _cx, _rms, _th, target_rms, Y, X
    )
    return jax.vmap(warp_fn)(images01, cy, cx, rms, theta_deg)


def preprocess_images_jax(
    images_01: jnp.ndarray,
    *,
    dark_quantile: float = 0.1,
    weight_power: float = 1.0,
    batch_size: int = 8,
    target_rms: float = None,
    rms_scale: float = 0.82,
):
    """
    Weighted centroid + PCA-align + scale normalization (JAX/JIT).
    """
    K = images_01.shape[0]
    rms_list, cy_list, cx_list, th_list = [], [], [], []

    for i in range(0, K, batch_size):
        batch = images_01[i:i + batch_size]
        cy, cx, rms, th = _preprocess_batch_stats(
            batch, dark_quantile, weight_power
        )
        rms_list.append(rms)
        cy_list.append(cy)
        cx_list.append(cx)
        th_list.append(th)

    rms_all = jnp.concatenate(rms_list)
    cy_all  = jnp.concatenate(cy_list)
    cx_all  = jnp.concatenate(cx_list)
    th_all  = jnp.concatenate(th_list)

    if target_rms is None:
        target_rms = jnp.mean(rms_all) * rms_scale  # scale < 1 leaves border margin

    proc_batches = []
    for i in range(0, K, batch_size):
        batch = images_01[i:i + batch_size]
        proc = _warp_batch(
            batch,
            cy_all[i:i + batch_size],
            cx_all[i:i + batch_size],
            rms_all[i:i + batch_size],
            th_all[i:i + batch_size],
            target_rms,
            dark_quantile,
            weight_power,
        )
        proc_batches.append(np.array(proc))   # move to CPU immediately

    return np.concatenate(proc_batches, axis=0), target_rms


@jax.jit
def resize_to_64(images: jnp.ndarray) -> jnp.ndarray:
    """Resize (K, H, W) images to (K, 64, 64) using bilinear interpolation."""
    return jax.image.resize(
        images, shape=(images.shape[0], 64, 64), method="linear"
    )


def full_preprocess_pipeline(
    images_np: np.ndarray,
    *,
    dark_quantile: float = 0.1,
    weight_power: float = 1.0,
    batch_size: int = 8,
    target_rms: float = None,
    rms_scale: float = 0.82,
):
    """Full pipeline: numpy (N,H,W) uint8/float -> (N,64,64) float32 ink=1 on device.

    Steps:
      1. to_float01
      2. geometric normalization (centroid + PCA-align + scale)
      3. resize to 64x64
      4. invert (white bg -> black bg, ink=1)

    Returns
    -------
    images_64 : jnp.ndarray  (N, 64, 64) float32, ink=1
    target_rms : float
    """
    images_f = to_float01_np(images_np)
    images_jax = jnp.asarray(images_f)
    images_proc, target_rms = preprocess_images_jax(
        images_jax,
        dark_quantile=dark_quantile,
        weight_power=weight_power,
        batch_size=batch_size,
        target_rms=target_rms,
        rms_scale=rms_scale,
    )
    # Resize in batches to avoid GPU OOM (images_proc is now a numpy array)
    resized = []
    for i in range(0, len(images_proc), batch_size * 16):
        chunk = jnp.asarray(images_proc[i:i + batch_size * 16])
        resized.append(np.array(resize_to_64(chunk)))
    images_64 = np.concatenate(resized, axis=0)
    images_64 = 1.0 - images_64  # invert: ink=1
    return jnp.asarray(images_64), target_rms


# ============================================================
# 4. Mini-batch iteration
# ============================================================

def iter_minibatches_np(X: np.ndarray, batch_size: int, seed: int):
    """Yield shuffled mini-batches of X (no labels)."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    for i in range(0, N, batch_size):
        yield X[perm[i:i + batch_size]]


def iter_minibatches_xy_np(
    X: np.ndarray, Y: np.ndarray, batch_size: int, seed: int
):
    """Yield shuffled mini-batches of (X, Y) pairs."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], Y[idx]

