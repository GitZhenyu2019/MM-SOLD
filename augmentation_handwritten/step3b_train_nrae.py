"""
step3b_train_nrae.py
====================
Train a Nuclear-Norm Regularised Autoencoder (NRAE) on all 2 000 training
images, following the σ-CFDM supplementary (Scarvelis & Solomon 2024).
"""

import argparse, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nrae_model import (
    NRAEModel,
    make_nrae_state,
    nrae_train_step,
    dct2d_crop_np,
    save_params,
    count_params,
    print_nrae_summary,
)
from data_utils import iter_minibatches_np


def main(args):
    print("=" * 60)
    print("Step 3b: Train Nuclear-Norm Regularised Autoencoder (NRAE)")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ----------------------------------------------------------------
    # Load training data
    # ----------------------------------------------------------------
    train_images = np.load(os.path.join(args.data_dir, "train_images.npy"))
    N_train = train_images.shape[0]
    print(f"\nTraining images : {train_images.shape}"
          f"  (min={train_images.min():.3f}, max={train_images.max():.3f})")

    # ----------------------------------------------------------------
    # Pre-compute DCT once (CPU/numpy — avoids repeated computation in loop)
    # ----------------------------------------------------------------
    print(f"Pre-computing DCT  n_dct={args.n_dct} …", end=" ", flush=True)
    # train_images: (N, 64, 64)
    train_dct = dct2d_crop_np(train_images, n_dct=args.n_dct)   # (N, n_dct²)
    print(f"done  shape={train_dct.shape}")

    # images with channel dim for reconstruction loss: (N, 64, 64, 1)
    train_imgs_ch = train_images[..., np.newaxis].astype(np.float32)

    # ----------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------
    rng   = jax.random.PRNGKey(args.seed)
    model, state = make_nrae_state(
        rng,
        latent_dim=args.latent_dim,
        enc1_hidden=args.enc1_hidden,
        dec_hidden=args.dec_hidden,
        n_dct=args.n_dct,
        unet_base_ch=args.unet_base_ch,
        lr=args.lr,
    )
    print_nrae_summary(state.params,
                       latent_dim=args.latent_dim,
                       enc1_hidden=args.enc1_hidden,
                       dec_hidden=args.dec_hidden,
                       n_dct=args.n_dct,
                       unet_base_ch=args.unet_base_ch)
    print(f"Training config:")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  sigma={args.sigma}  (η=σ²={args.sigma**2:.1f})  alpha={args.alpha}")

    # ----------------------------------------------------------------
    # JIT warmup
    # ----------------------------------------------------------------
    dct_dim = args.n_dct ** 2
    dummy_dct  = jnp.zeros((args.batch_size, dct_dim),     dtype=jnp.float32)
    dummy_img  = jnp.zeros((args.batch_size, 64, 64, 1),   dtype=jnp.float32)
    rng, sub   = jax.random.split(rng)
    _state, _, _, _ = nrae_train_step(
        state, dummy_dct, dummy_img, sub,
        latent_dim=args.latent_dim, enc1_hidden=args.enc1_hidden,
        dec_hidden=args.dec_hidden, n_dct=args.n_dct,
        unet_base_ch=args.unet_base_ch,
        sigma=args.sigma, fixed_noise_sigma=1e-3, alpha=args.alpha,
    )
    jax.block_until_ready(_state.params)
    print("JIT compilation done.\n")

    # Re-initialise fresh state after warmup dummy step
    rng   = jax.random.PRNGKey(args.seed)
    model, state = make_nrae_state(
        rng,
        latent_dim=args.latent_dim, enc1_hidden=args.enc1_hidden,
        dec_hidden=args.dec_hidden, n_dct=args.n_dct,
        unet_base_ch=args.unet_base_ch, lr=args.lr,
    )

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "nrae_best.pkl")
    best_loss = float("inf")
    t0        = time.time()

    print(f"{'Epoch':>5}  {'Loss':>10}  {'Recon':>10}  "
          f"{'Reg':>10}  {'Best':>10}  {'Time':>7}")
    print("-" * 62)

    for ep in range(args.epochs):
        losses, recons, regs = [], [], []

        # Each mini-batch uses matching slices from train_dct and train_imgs_ch.
        # iter_minibatches_np shuffles via index permutation; we replicate it
        # for both arrays using the same seed.
        rng_np = np.random.default_rng(args.seed + ep)
        perm   = rng_np.permutation(N_train)

        for i in range(0, N_train, args.batch_size):
            idx       = perm[i:i + args.batch_size]
            x_dct_np  = train_dct[idx]          # (b, n_dct²)
            x_img_np  = train_imgs_ch[idx]       # (b, 64, 64, 1)

            rng, sub  = jax.random.split(rng)
            state, loss, recon, reg = nrae_train_step(
                state,
                jnp.asarray(x_dct_np),
                jnp.asarray(x_img_np),
                sub,
                latent_dim=args.latent_dim,
                enc1_hidden=args.enc1_hidden,
                dec_hidden=args.dec_hidden,
                n_dct=args.n_dct,
                unet_base_ch=args.unet_base_ch,
                sigma=args.sigma,
                fixed_noise_sigma=1e-3,
                alpha=args.alpha,
            )
            losses.append(float(loss))
            recons.append(float(recon))
            regs.append(float(reg))

        ep_loss = float(np.mean(losses))
        saved   = ""
        if ep_loss < best_loss:
            best_loss = ep_loss
            save_params(ckpt_path, state.params,
                        info={"epoch":        ep + 1,
                              "loss":         best_loss,
                              "latent_dim":   args.latent_dim,
                              "enc1_hidden":  args.enc1_hidden,
                              "dec_hidden":   args.dec_hidden,
                              "n_dct":        args.n_dct,
                              "unet_base_ch": args.unet_base_ch})
            saved = " *saved*"

        elapsed = time.time() - t0
        print(f"{ep+1:>5}  {ep_loss:>10.4f}  {np.mean(recons):>10.4f}  "
              f"{np.mean(regs):>10.4f}  {best_loss:>10.4f}  "
              f"{elapsed:>6.1f}s{saved}")

    print(f"\nTraining done.  Best loss: {best_loss:.4f}")
    print(f"NRAE checkpoint : {ckpt_path}")
    print("\nStep 3b complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3b: Train NRAE.")
    parser.add_argument("--data_dir",     type=str,   default="./data")
    parser.add_argument("--ckpt_dir",     type=str,   default="./checkpoints")
    parser.add_argument("--latent_dim",   type=int,   default=100)
    parser.add_argument("--enc1_hidden",  type=int,   default=2048)
    parser.add_argument("--dec_hidden",   type=int,   default=2048)
    parser.add_argument("--n_dct",        type=int,   default=20)
    parser.add_argument("--unet_base_ch", type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--sigma",        type=float, default=2.0,
                        help="Regularisation noise level; η=sigma² (default 2.0→η=4)")
    parser.add_argument("--alpha",        type=float, default=100.0,
                        help="Log-cosh sharpness factor (default 100)")
    parser.add_argument("--seed",         type=int,   default=0)
    args = parser.parse_args()
    main(args)
