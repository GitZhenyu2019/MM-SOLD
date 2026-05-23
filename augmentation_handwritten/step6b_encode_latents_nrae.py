"""
step6b_encode_latents_nrae.py
==============================
Encode train / val / test images with the trained NRAE, then run MM-SOLD
in latent space to produce augmented latents.
"""

import argparse, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nrae_model      import make_nrae_state, encode_dataset_nrae, load_params, print_nrae_summary
from whitening_utils import compute_sample_mean_cov, symmetric_matrix_sqrt_and_invsqrt, unwhiten
from sampling_algo   import sample_class_overdamped_manifold


def _sample_class_latents(cls, Z_train, y_train, args):
    """Run MM-SOLD in latent space for one class; returns unwhitened latents."""
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


def main(args):
    print("=" * 60)
    print("Step 6b: Encode Latents with NRAE + LDS Latent Augmentation")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ----------------------------------------------------------------
    # Load preprocessed images (output of step1)
    # ----------------------------------------------------------------
    d = args.data_dir
    train_images = np.load(os.path.join(d, "train_images.npy"))
    train_labels = np.load(os.path.join(d, "train_labels.npy"))
    val_images   = np.load(os.path.join(d, "val_images.npy"))
    val_labels   = np.load(os.path.join(d, "val_labels.npy"))
    test_images  = np.load(os.path.join(d, "test_images.npy"))
    test_labels  = np.load(os.path.join(d, "test_labels.npy"))
    print(f"\nTrain: {train_images.shape}  "
          f"Val: {val_images.shape}  Test: {test_images.shape}")

    # ----------------------------------------------------------------
    # Load NRAE checkpoint
    # ----------------------------------------------------------------
    ckpt_path = os.path.join(args.ckpt_dir, "nrae_best.pkl")
    print(f"\nLoading NRAE checkpoint: {ckpt_path}")
    params_np, info = load_params(ckpt_path)
    print(f"  Checkpoint info: {info}")

    latent_dim   = info.get("latent_dim",   100)
    enc1_hidden  = info.get("enc1_hidden",  2048)
    dec_hidden   = info.get("dec_hidden",   2048)
    n_dct        = info.get("n_dct",        20)
    unet_base_ch = info.get("unet_base_ch", 32)

    rng = jax.random.PRNGKey(args.seed)
    model, _ = make_nrae_state(
        rng,
        latent_dim=latent_dim,
        enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden,
        n_dct=n_dct,
        unet_base_ch=unet_base_ch,
    )
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    print_nrae_summary(params, latent_dim=latent_dim, enc1_hidden=enc1_hidden,
                       dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch)

    # ----------------------------------------------------------------
    # Encode all splits
    # ----------------------------------------------------------------
    print("\nEncoding splits with NRAE ...")
    Z_train = encode_dataset_nrae(model, params, train_images,
                                  n_dct=n_dct, batch_size=64)
    Z_val   = encode_dataset_nrae(model, params, val_images,
                                  n_dct=n_dct, batch_size=64)
    Z_test  = encode_dataset_nrae(model, params, test_images,
                                  n_dct=n_dct, batch_size=64)
    print(f"  Z_train: {Z_train.shape}  "
          f"Z_val: {Z_val.shape}  Z_test: {Z_test.shape}")

    # ----------------------------------------------------------------
    # Generate augmented latents via MM-SOLD (no image decoding)
    # ----------------------------------------------------------------
    print(f"\nGenerating {args.n_aug_per_class} augmented latents per class "
          f"({args.n_aug_per_class * 10} total) ...")
    print("  JIT warmup (class 0) ...")
    _sample_class_latents(0, Z_train, train_labels, args)
    print("  Warmup done. Starting timed runs ...\n")

    Z_aug_list, y_aug_list = [], []
    for cls in range(10):
        t0 = time.perf_counter()
        Z_cls_aug = _sample_class_latents(cls, Z_train, train_labels, args)
        elapsed = time.perf_counter() - t0
        Z_aug_list.append(Z_cls_aug)
        y_aug_list.append(np.full(args.n_aug_per_class, cls, dtype=np.int32))
        print(f"  class {cls}: {Z_cls_aug.shape[0]} latents in {elapsed:.2f}s")

    Z_aug = np.concatenate(Z_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)

    Z_aug_train = np.concatenate([Z_train, Z_aug], axis=0).astype(np.float32)
    y_aug_train = np.concatenate([train_labels, y_aug], axis=0).astype(np.int32)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    os.makedirs(args.latent_dir, exist_ok=True)
    saves = [
        ("Z_train.npy",     Z_train.astype(np.float32)),
        ("y_train.npy",     train_labels.astype(np.int32)),
        ("Z_val.npy",       Z_val.astype(np.float32)),
        ("y_val.npy",       val_labels.astype(np.int32)),
        ("Z_test.npy",      Z_test.astype(np.float32)),
        ("y_test.npy",      test_labels.astype(np.int32)),
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
          f"({len(train_labels)} orig + {len(y_aug)} generated)")
    print("\nStep 6b complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 6b: NRAE encode + LDS latent augmentation.")
    parser.add_argument("--data_dir",         type=str, default="./data")
    parser.add_argument("--ckpt_dir",         type=str, default="./checkpoints")
    parser.add_argument("--latent_dir",       type=str, default="./nrae_latents")
    parser.add_argument("--n_aug_per_class",  type=int, default=9000)
    parser.add_argument("--nsteps",           type=int, default=50)
    parser.add_argument("--h",                type=float, default=1e-4)
    parser.add_argument("--sigma_gmm",        type=float, default=0.05)
    parser.add_argument("--sigma_smoothing",  type=float, default=2.0)
    parser.add_argument("--M",                type=int,   default=32)
    parser.add_argument("--discretization",   type=str,   default="LM",
                        choices=["LM", "EM"])
    parser.add_argument("--k_whiten",         type=int,   default=20)
    parser.add_argument("--seed",             type=int,   default=0)
    args = parser.parse_args()
    main(args)
