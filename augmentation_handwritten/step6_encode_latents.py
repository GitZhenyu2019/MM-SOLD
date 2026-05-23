"""
step6_encode_latents.py
=======================
Create a 3-way split (train / val / test), apply geometric preprocessing,
encode all splits with the trained VAE to get latent vectors, then run the
manifold LDS sampler on the training latents to produce augmented latents.
"""

import argparse, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils      import load_images_and_digits, to_float01_np, full_preprocess_pipeline
from vae_model       import make_vae_state, encode_dataset, load_params
from whitening_utils import compute_sample_mean_cov, symmetric_matrix_sqrt_and_invsqrt, unwhiten
from sampling_algo   import sample_class_overdamped_manifold


# ----------------------------------------------------------------
# 3-way balanced split
# ----------------------------------------------------------------

def three_way_split(images_float, digits, n_train, n_val, n_test, seed):
    """Per-class balanced 3-way split.
    Returns dict with keys 'train'/'val'/'test', each (images, labels).
    """
    rng = np.random.default_rng(seed)
    splits = {k: ([], []) for k in ("train", "val", "test")}
    for c in range(10):
        idx = np.where(digits == c)[0]
        rng.shuffle(idx)
        need = n_train + n_val + n_test
        if len(idx) < need:
            raise ValueError(
                f"Class {c}: need {need} samples but only {len(idx)} available.")
        for key, start, n in [("train", 0, n_train),
                               ("val",   n_train, n_val),
                               ("test",  n_train + n_val, n_test)]:
            splits[key][0].append(images_float[idx[start:start + n]])
            splits[key][1].append(np.full(n, c, dtype=np.int32))
    return {k: (np.concatenate(v[0]), np.concatenate(v[1]))
            for k, v in splits.items()}


# ----------------------------------------------------------------
# Per-class sampling (latent space only, no decoding)
# ----------------------------------------------------------------

def _sample_class_latents(cls, Z_train, y_train, args):
    mask  = y_train == cls
    Z_cls = jnp.asarray(Z_train[mask])
    mean_cls, cov_cls = compute_sample_mean_cov(Z_cls)
    S_sqrt, S_invsqrt, _ = symmetric_matrix_sqrt_and_invsqrt(
        cov_cls, eps=1e-5, k=args.k_whiten)
    z_w, _ = sample_class_overdamped_manifold(
        Z_class=Z_cls,
        mean_class=mean_cls,
        S_sqrt_class=S_sqrt,
        S_invsqrt_class=S_invsqrt,
        n_particles=args.n_aug_per_class,
        nsteps=args.nsteps,
        h=args.h,
        sigma_gmm=args.sigma_gmm,
        sigma_smoothing=args.sigma_smoothing,
        M=args.M,
        shared_noise=False,
        discretization=args.discretization,
        seed=args.seed + cls,
    )
    jax.block_until_ready(z_w)
    return np.array(unwhiten(z_w, mean_cls, S_sqrt))


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main(args):
    print("=" * 60)
    print("Step 6: Encode Latents (3-way split + VAE encode + LDS augment)")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ---- Load raw data ----
    print(f"\nLoading images : {args.images_path}")
    images_raw, digits = load_images_and_digits(args.images_path, args.labels_path)
    images_float = to_float01_np(images_raw)
    print(f"Dataset: {images_raw.shape[0]} images, shape {images_raw.shape[1:]}")

    # ---- 3-way split ----
    print(f"\nSplitting: train={args.n_train_per_class}/class  "
          f"val={args.n_val_per_class}/class  "
          f"test={args.n_test_per_class}/class")
    splits = three_way_split(
        images_float, digits,
        args.n_train_per_class, args.n_val_per_class, args.n_test_per_class,
        seed=args.seed)
    y_train, y_val, y_test = (splits[k][1] for k in ("train", "val", "test"))
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    # ---- Preprocess (target_rms computed from train only) ----
    print("\nPreprocessing train images ...")
    train_64, target_rms = full_preprocess_pipeline(
        splits["train"][0],
        dark_quantile=0.1, weight_power=1.0,
        batch_size=args.preprocess_batch_size)
    print(f"  target_rms = {float(target_rms):.5f}")

    print("Preprocessing val images ...")
    val_64, _ = full_preprocess_pipeline(
        splits["val"][0],
        dark_quantile=0.1, weight_power=1.0,
        batch_size=args.preprocess_batch_size,
        target_rms=target_rms)

    print("Preprocessing test images ...")
    test_64, _ = full_preprocess_pipeline(
        splits["test"][0],
        dark_quantile=0.1, weight_power=1.0,
        batch_size=args.preprocess_batch_size,
        target_rms=target_rms)

    # ---- Load VAE ----
    print(f"\nLoading VAE checkpoint: {args.vae_ckpt}")
    params_np, info = load_params(args.vae_ckpt)
    print(f"  Checkpoint info: {info}")
    rng = jax.random.PRNGKey(args.seed)
    model, _ = make_vae_state(rng, latent_dim=args.latent_dim, base_ch=args.base_ch)
    params = jax.tree_util.tree_map(jnp.asarray, params_np)

    # ---- Encode all splits ----
    print("\nEncoding splits with VAE ...")
    Z_train = encode_dataset(model, params, np.array(train_64), batch_size=64)
    Z_val   = encode_dataset(model, params, np.array(val_64),   batch_size=64)
    Z_test  = encode_dataset(model, params, np.array(test_64),  batch_size=64)
    print(f"  Z_train: {Z_train.shape}  Z_val: {Z_val.shape}  Z_test: {Z_test.shape}")

    # ---- Generate augmented latents via manifold LDS (no decoding) ----
    print(f"\nGenerating {args.n_aug_per_class} augmented latents per class ...")
    print("  JIT warmup (class 0) ...")
    _sample_class_latents(0, Z_train, y_train, args)
    print("  Warmup done. Starting timed runs ...\n")

    Z_aug_list, y_aug_list = [], []
    for cls in range(10):
        t0 = time.perf_counter()
        Z_cls_aug = _sample_class_latents(cls, Z_train, y_train, args)
        elapsed = time.perf_counter() - t0
        Z_aug_list.append(Z_cls_aug)
        y_aug_list.append(np.full(args.n_aug_per_class, cls, dtype=np.int32))
        print(f"  class {cls}: {Z_cls_aug.shape[0]} latents in {elapsed:.2f}s")

    Z_aug = np.concatenate(Z_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)

    Z_aug_train = np.concatenate([Z_train, Z_aug], axis=0).astype(np.float32)
    y_aug_train = np.concatenate([y_train, y_aug], axis=0).astype(np.int32)

    # ---- Save ----
    os.makedirs(args.latent_dir, exist_ok=True)
    saves = [
        ("Z_train.npy",     Z_train.astype(np.float32)),
        ("y_train.npy",     y_train.astype(np.int32)),
        ("Z_val.npy",       Z_val.astype(np.float32)),
        ("y_val.npy",       y_val.astype(np.int32)),
        ("Z_test.npy",      Z_test.astype(np.float32)),
        ("y_test.npy",      y_test.astype(np.int32)),
        ("Z_aug_train.npy", Z_aug_train),
        ("y_aug_train.npy", y_aug_train),
    ]
    for fname, arr in saves:
        np.save(os.path.join(args.latent_dir, fname), arr)

    print(f"\nSaved to {args.latent_dir}/")
    print(f"  Z_train:     {Z_train.shape}")
    print(f"  Z_val:       {Z_val.shape}")
    print(f"  Z_test:      {Z_test.shape}")
    print(f"  Z_aug_train: {Z_aug_train.shape}  "
          f"({args.n_train_per_class*10} orig + {args.n_aug_per_class*10} gen)")
    print("\nStep 6 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 6: 3-way split, VAE encode, and LDS latent augmentation.")
    parser.add_argument("--images_path",          type=str, required=True)
    parser.add_argument("--labels_path",          type=str, required=True)
    parser.add_argument("--vae_ckpt",             type=str, default="./checkpoints/vae_best.pkl")
    parser.add_argument("--latent_dir",           type=str, default="./latents")
    parser.add_argument("--n_train_per_class",    type=int, default=200)
    parser.add_argument("--n_val_per_class",      type=int, default=500)
    parser.add_argument("--n_test_per_class",     type=int, default=500)
    parser.add_argument("--n_aug_per_class",      type=int, default=800)
    parser.add_argument("--latent_dim",           type=int, default=100)
    parser.add_argument("--base_ch",              type=int, default=64)
    parser.add_argument("--nsteps",               type=int, default=50)
    parser.add_argument("--h",                    type=float, default=1e-4)
    parser.add_argument("--sigma_gmm",            type=float, default=0.05)
    parser.add_argument("--sigma_smoothing",      type=float, default=2.0)
    parser.add_argument("--M",                    type=int,   default=32)
    parser.add_argument("--discretization",       type=str,   default="LM",
                        choices=["LM", "EM"])
    parser.add_argument("--k_whiten",             type=int,   default=20)
    parser.add_argument("--preprocess_batch_size",type=int,   default=8)
    parser.add_argument("--seed",                 type=int,   default=0)
    args = parser.parse_args()
    main(args)
