"""
step1_prepare_data.py
=====================
Prepare data for the high_dimension/handwritten experiment.
"""
import argparse
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TARGET_DIGIT, N_TRAIN, N_TEST,
    LATENT_DIM, ENC1_HIDDEN, DEC_HIDDEN, N_DCT, UNET_BASE_CH,
    NRAE_CKPT,
    RAW_IMG, RAW_LABEL,
    DATA_DIR, LATENT_DIR,
)
from data_utils import full_preprocess_pipeline
from nrae_model import make_nrae_state, encode_dataset_nrae, load_params as nrae_load


def _load_digits(labels_path: str) -> np.ndarray:
    """Load digit labels from .npy — handles 1-D array or 2-D writerinfo."""
    arr = np.load(labels_path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.astype(np.int32)
    if arr.ndim == 2:
        return arr[:, 0].astype(np.int32)   # writerinfo: first column = digit
    raise ValueError(f"Unexpected label array shape: {arr.shape}")


def main(args):
    print("=" * 60)
    print(f"Step 1: Prepare digit-{TARGET_DIGIT} data + NRAE latents")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ── 1. Load raw data ─────────────────────────────────────────────────────
    print(f"\nLoading raw images : {args.raw_img}")
    print(f"Loading raw labels : {args.raw_label}")
    images_raw = np.asarray(np.load(args.raw_img, allow_pickle=True))
    digits     = _load_digits(args.raw_label)

    if images_raw.ndim != 3:
        raise ValueError(f"Expected images (N,H,W), got {images_raw.shape}")
    if images_raw.shape[0] != digits.shape[0]:
        raise ValueError(
            f"Image count {images_raw.shape[0]} != label count {digits.shape[0]}"
        )
    print(f"  Full dataset: {images_raw.shape}  unique labels: {np.unique(digits)}")

    # ── 2. Filter for target digit, split train / test ───────────────────────
    rng  = np.random.default_rng(args.seed)
    mask = digits == TARGET_DIGIT
    imgs_digit = images_raw[mask]
    n_avail    = imgs_digit.shape[0]
    n_need     = N_TRAIN + N_TEST
    print(f"\n  digit {TARGET_DIGIT}: {n_avail} available, need {n_need} "
          f"({N_TRAIN} train + {N_TEST} test)")
    if n_avail < n_need:
        raise ValueError(
            f"Not enough digit-{TARGET_DIGIT} samples: {n_avail} < {n_need}"
        )

    chosen = rng.choice(n_avail, size=n_need, replace=False)
    imgs_train_raw = imgs_digit[chosen[:N_TRAIN]]
    imgs_test_raw  = imgs_digit[chosen[N_TRAIN:]]
    lbls_train = np.full(N_TRAIN, TARGET_DIGIT, dtype=np.int32)
    lbls_test  = np.full(N_TEST,  TARGET_DIGIT, dtype=np.int32)

    # ── 3. Geometric preprocessing ───────────────────────────────────────────
    # Compute target_rms from training set; reuse for test (consistency).
    print(f"\nPreprocessing {N_TRAIN} training images ...")
    train_64, target_rms = full_preprocess_pipeline(
        imgs_train_raw,
        dark_quantile=args.dark_quantile,
        weight_power=args.weight_power,
        batch_size=args.preprocess_batch_size,
        rms_scale=args.rms_scale,
    )
    print(f"  train_64: {np.asarray(train_64).shape}  target_rms={float(target_rms):.5f}"
          f"  (rms_scale={args.rms_scale})")

    print(f"Preprocessing {N_TEST} test images (using train target_rms) ...")
    test_64, _ = full_preprocess_pipeline(
        imgs_test_raw,
        dark_quantile=args.dark_quantile,
        weight_power=args.weight_power,
        batch_size=args.preprocess_batch_size,
        target_rms=target_rms,   # fixed from training set; rms_scale already baked in
    )
    print(f"  test_64:  {np.asarray(test_64).shape}")

    imgs_train = np.array(train_64, dtype=np.float32)
    imgs_test  = np.array(test_64,  dtype=np.float32)

    # ── 4. Load NRAE checkpoint ───────────────────────────────────────────────
    print(f"\nLoading NRAE from: {args.nrae_ckpt}")
    params_np, info = nrae_load(args.nrae_ckpt)
    print(f"  Checkpoint info: {info}")

    latent_dim   = info.get("latent_dim",   LATENT_DIM)
    enc1_hidden  = info.get("enc1_hidden",  ENC1_HIDDEN)
    dec_hidden   = info.get("dec_hidden",   DEC_HIDDEN)
    n_dct        = info.get("n_dct",        N_DCT)
    unet_base_ch = info.get("unet_base_ch", UNET_BASE_CH)

    rng_jax = jax.random.PRNGKey(args.seed)
    nrae_model, _ = make_nrae_state(
        rng_jax,
        latent_dim=latent_dim,
        enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden,
        n_dct=n_dct,
        unet_base_ch=unet_base_ch,
    )
    nrae_params = jax.tree_util.tree_map(jnp.asarray, params_np)
    print(f"  NRAE loaded  latent_dim={latent_dim}  n_dct={n_dct}")

    # ── 5. Encode to NRAE latents ─────────────────────────────────────────────
    print("\nEncoding with NRAE ...")
    Z_train = encode_dataset_nrae(
        nrae_model, nrae_params, imgs_train, n_dct=n_dct, batch_size=64)
    Z_test  = encode_dataset_nrae(
        nrae_model, nrae_params, imgs_test,  n_dct=n_dct, batch_size=64)
    print(f"  Z_train: {Z_train.shape}  Z_test: {Z_test.shape}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(args.data_dir,   exist_ok=True)
    os.makedirs(args.latent_dir, exist_ok=True)

    np.save(os.path.join(args.data_dir,   "images_train.npy"),  imgs_train)
    np.save(os.path.join(args.data_dir,   "images_test.npy"),   imgs_test)
    np.save(os.path.join(args.data_dir,   "labels_train.npy"),  lbls_train)
    np.save(os.path.join(args.data_dir,   "labels_test.npy"),   lbls_test)
    np.save(os.path.join(args.data_dir,   "target_rms.npy"),    np.float32(target_rms))
    np.save(os.path.join(args.latent_dir, "Z_train.npy"),       Z_train.astype(np.float32))
    np.save(os.path.join(args.latent_dir, "Z_test.npy"),        Z_test.astype(np.float32))

    print(f"\nSaved to:")
    print(f"  {args.data_dir}/")
    print(f"    images_train.npy  {imgs_train.shape}  images_test.npy  {imgs_test.shape}")
    print(f"    labels_train.npy  labels_test.npy  target_rms.npy")
    print(f"  {args.latent_dir}/")
    print(f"    Z_train.npy  {Z_train.shape}  Z_test.npy  {Z_test.shape}")
    print(f"\nStep 1 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: preprocess raw digit data + NRAE encode.")
    parser.add_argument("--raw_img",   type=str, default=RAW_IMG,
                        help="Path to handwritten_img.npy")
    parser.add_argument("--raw_label", type=str, default=RAW_LABEL,
                        help="Path to handwritten_label.npy")
    parser.add_argument("--data_dir",    type=str, default=DATA_DIR)
    parser.add_argument("--latent_dir",  type=str, default=LATENT_DIR)
    parser.add_argument("--nrae_ckpt",   type=str, default=NRAE_CKPT)
    parser.add_argument("--dark_quantile",         type=float, default=0.1)
    parser.add_argument("--weight_power",          type=float, default=1.0)
    parser.add_argument("--preprocess_batch_size", type=int,   default=8)
    parser.add_argument("--rms_scale", type=float, default=0.82,
                        help="Scale factor for target_rms (< 1 leaves border margin, "
                             "prevents digit clipping). Default 0.82.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
