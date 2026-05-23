"""
step4b_generate_augmented_nrae.py
==================================
Use the trained NRAE to generate augmented images via the MM-SOLD.
"""

import argparse, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nrae_model import (
    NRAEModel,
    make_nrae_state,
    encode_dataset_nrae,
    decode_latents_nrae,
    load_params,
    print_nrae_summary,
)
from whitening_utils import (
    compute_sample_mean_cov,
    symmetric_matrix_sqrt_and_invsqrt,
    whiten,
    unwhiten,
    nearest_center_distances,
)
from sampling_algo import (
    sample_class_overdamped_manifold,
    enforce_mean_and_cov_manifold,
    step_overdamped_stationary_manifold,
)


def run_sampling_for_class(
    cls, train_images, train_labels, model, params, args
):
    """Encode class images, build whitening, run MM-SOLD sampler, decode."""
    mask      = train_labels == cls
    imgs_cls  = train_images[mask]             # (N_cls, 64, 64) ink=1
    n_cls     = imgs_cls.shape[0]

    # Encode -> latent vectors via NRAE
    Z_cls = encode_dataset_nrae(
        model, params, imgs_cls,
        n_dct=args.n_dct, batch_size=64,
    )
    Z_jax = jnp.asarray(Z_cls)

    # Per-class partial whitening
    mean_cls, cov_cls = compute_sample_mean_cov(Z_jax)
    S_sqrt_cls, S_invsqrt_cls, eigs = symmetric_matrix_sqrt_and_invsqrt(
        cov_cls, eps=1e-5, k=args.k_whiten
    )

    # Whiten and sample (returns whitened latent vectors)
    z_sampled_w, Xw = sample_class_overdamped_manifold(
        Z_class=Z_jax,
        mean_class=mean_cls,
        S_sqrt_class=S_sqrt_cls,
        S_invsqrt_class=S_invsqrt_cls,
        n_particles=args.n_per_class,
        nsteps=args.nsteps,
        h=args.h,
        sigma_gmm=args.sigma_gmm,
        sigma_smoothing=args.sigma_smoothing,
        M=args.M,
        shared_noise=False,
        discretization=args.discretization,
        seed=args.seed + cls,
    )
    jax.block_until_ready(z_sampled_w)

    # Unwhiten -> decode via NRAE
    z_sampled = unwhiten(z_sampled_w, mean_cls, S_sqrt_cls)
    imgs_gen  = decode_latents_nrae(model, params, z_sampled, batch_size=64)

    # Diagnostics
    d_w = nearest_center_distances(z_sampled_w, Xw, chunk=256)
    print(f"  [Whitened]  NN dist: mean={float(d_w.mean()):.3f}  "
          f"std={float(d_w.std()):.3f}  "
          f"min={float(d_w.min()):.3f}  max={float(d_w.max()):.3f}")

    return imgs_gen


def main(args):
    print("=" * 60)
    print("Step 4b: Generate Augmented Data via NRAE + MM-SOLD Sampler")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ----------------------------------------------------------------
    # Load data and NRAE checkpoint
    # ----------------------------------------------------------------
    train_images = np.load(os.path.join(args.data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(args.data_dir, "train_labels.npy"))
    print(f"\nTrain images: {train_images.shape}")

    ckpt_path = os.path.join(args.ckpt_dir, "nrae_best.pkl")
    print(f"Loading NRAE checkpoint: {ckpt_path}")
    params_np, info = load_params(ckpt_path)
    print(f"  Checkpoint info: {info}")

    # Read hyperparams from checkpoint info (fall back to argparse values)
    latent_dim   = info.get("latent_dim",   args.latent_dim)
    enc1_hidden  = info.get("enc1_hidden",  args.enc1_hidden)
    dec_hidden   = info.get("dec_hidden",   args.dec_hidden)
    n_dct        = info.get("n_dct",        args.n_dct)
    unet_base_ch = info.get("unet_base_ch", args.unet_base_ch)

    # Store resolved n_dct back to args for use in run_sampling_for_class
    args.n_dct = n_dct

    # Rebuild NRAE model
    rng   = jax.random.PRNGKey(args.seed)
    model, _ = make_nrae_state(
        rng,
        latent_dim=latent_dim,
        enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden,
        n_dct=n_dct,
        unet_base_ch=unet_base_ch,
    )
    # Convert numpy params back to JAX device arrays
    params = jax.tree_util.tree_map(jnp.asarray, params_np)

    print_nrae_summary(
        params,
        latent_dim=latent_dim,
        enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden,
        n_dct=n_dct,
        unet_base_ch=unet_base_ch,
    )

    print(f"\nSampling config:")
    print(f"  n_per_class={args.n_per_class}  nsteps={args.nsteps}")
    print(f"  h={args.h}  sigma_gmm={args.sigma_gmm}  "
          f"sigma_smoothing={args.sigma_smoothing}")
    print(f"  M={args.M}  discretization={args.discretization}  "
          f"k_whiten={args.k_whiten}")
    print(f"  Total new images: {args.n_per_class * 10}")

    # ----------------------------------------------------------------
    # Warmup pass for JIT compilation (class 0, full nsteps)
    # ----------------------------------------------------------------
    print("\nRunning JIT warmup (class 0) ...")
    _ = run_sampling_for_class(0, train_images, train_labels, model, params, args)
    print("Warmup complete.  Starting timed sampling runs ...\n")

    # ----------------------------------------------------------------
    # Per-class sampling (timed, no JIT compilation overhead)
    # ----------------------------------------------------------------
    all_gen_imgs   = []
    all_gen_labels = []
    total_t = 0.0

    for cls in range(10):
        print(f"--- Digit class {cls} ---")
        t_start = time.perf_counter()
        imgs_gen = run_sampling_for_class(
            cls, train_images, train_labels, model, params, args
        )
        t_end   = time.perf_counter()
        elapsed = t_end - t_start
        total_t += elapsed

        n_gen = imgs_gen.shape[0]
        all_gen_imgs.append(imgs_gen)
        all_gen_labels.append(np.full(n_gen, cls, dtype=np.int32))

        print(f"  Generated {n_gen} images for digit {cls} "
              f"in {elapsed:.2f}s\n")

    print(f"All classes done.  Total sampling time: {total_t:.2f}s "
          f"(avg {total_t/10:.2f}s/class)")

    # ----------------------------------------------------------------
    # Save generated data
    # ----------------------------------------------------------------
    gen_images = np.concatenate(all_gen_imgs,   axis=0).astype(np.float32)
    gen_labels = np.concatenate(all_gen_labels, axis=0).astype(np.int32)

    os.makedirs(args.data_dir, exist_ok=True)
    gen_imgs_path   = os.path.join(args.data_dir, "generated_images_nrae.npy")
    gen_labels_path = os.path.join(args.data_dir, "generated_labels_nrae.npy")
    np.save(gen_imgs_path,   gen_images)
    np.save(gen_labels_path, gen_labels)

    print(f"\nGenerated data saved:")
    print(f"  {gen_imgs_path}   shape={gen_images.shape}")
    print(f"  {gen_labels_path} shape={gen_labels.shape}")

    # Build augmented training set = original + generated
    aug_images = np.concatenate([train_images, gen_images], axis=0)
    aug_labels = np.concatenate([train_labels, gen_labels], axis=0)

    aug_imgs_path   = os.path.join(args.data_dir, "augmented_nrae_train_images.npy")
    aug_labels_path = os.path.join(args.data_dir, "augmented_nrae_train_labels.npy")
    np.save(aug_imgs_path,   aug_images.astype(np.float32))
    np.save(aug_labels_path, aug_labels.astype(np.int32))

    print(f"\nAugmented training set saved:")
    print(f"  {aug_imgs_path}   shape={aug_images.shape}")
    print(f"  {aug_labels_path} shape={aug_labels.shape}")
    print(f"  Augmentation ratio: {aug_images.shape[0] / train_images.shape[0]:.1f}x")

    print("\nLabel distribution in augmented set:")
    for d in range(10):
        n = (aug_labels == d).sum()
        print(f"  digit {d}: {n}")

    print("\nStep 4b complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4b: Generate augmented data via NRAE + MM-SOLD sampler."
    )
    parser.add_argument("--data_dir",        type=str,   default="./data")
    parser.add_argument("--ckpt_dir",        type=str,   default="./checkpoints")
    parser.add_argument("--latent_dim",      type=int,   default=100,
                        help="Fallback if not stored in checkpoint info")
    parser.add_argument("--enc1_hidden",     type=int,   default=2048,
                        help="Fallback if not stored in checkpoint info")
    parser.add_argument("--dec_hidden",      type=int,   default=2048,
                        help="Fallback if not stored in checkpoint info")
    parser.add_argument("--n_dct",           type=int,   default=20,
                        help="Fallback if not stored in checkpoint info")
    parser.add_argument("--unet_base_ch",    type=int,   default=32,
                        help="Fallback if not stored in checkpoint info")
    parser.add_argument("--n_per_class",     type=int,   default=9000,
                        help="Number of new images to generate per class "
                             "(default 9000 → 90000 total + 10000 original = 100000)")
    parser.add_argument("--nsteps",          type=int,   default=50)
    parser.add_argument("--h",               type=float, default=1e-4)
    parser.add_argument("--sigma_gmm",       type=float, default=0.05)
    parser.add_argument("--sigma_smoothing", type=float, default=2.0)
    parser.add_argument("--M",               type=int,   default=32,
                        help="LDS Monte Carlo samples per step (0=exact)")
    parser.add_argument("--discretization",  type=str,   default="LM",
                        choices=["LM", "EM"])
    parser.add_argument("--k_whiten",        type=int,   default=20,
                        help="Number of top eigenvalues to cap in whitening")
    parser.add_argument("--seed",            type=int,   default=0)
    args = parser.parse_args()
    main(args)
