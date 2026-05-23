"""
step3_train_vae.py
==================
Train a ResNet-VAE on ALL 2 000 training images.
"""

import argparse, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae_model import (
    ResVAE,
    make_vae_state,
    vae_train_step,
    save_params,
    load_params,
    count_params,
    print_vae_summary,
)
from data_utils import iter_minibatches_np


def main(args):
    print("=" * 60)
    print("Step 3: Train ResNet-VAE on 2000 training images")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ----------------------------------------------------------------
    # Load training data
    # ----------------------------------------------------------------
    train_images = np.load(os.path.join(args.data_dir, "train_images.npy"))
    N_train = train_images.shape[0]
    print(f"\nTraining images: {train_images.shape}  "
          f"(min={train_images.min():.3f}, max={train_images.max():.3f})")

    # Add channel dimension -> (N, 64, 64, 1)
    Xtrain = train_images[..., np.newaxis].astype(np.float32)

    # ----------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------
    rng = jax.random.PRNGKey(args.seed)
    model, state = make_vae_state(
        rng,
        latent_dim=args.latent_dim,
        base_ch=args.base_ch,
        lr=args.lr,
    )

    print_vae_summary(args.latent_dim, args.base_ch, state.params)
    print(f"Training config:")
    print(f"  epochs={args.epochs}  warmup_epochs={args.warmup_epochs}")
    print(f"  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  beta_final={args.beta_final}  free_bits={args.free_bits}"
          f"  pos_weight={args.pos_weight}")

    # ----------------------------------------------------------------
    # JIT warmup (excludes compilation from timing)
    # ----------------------------------------------------------------
    dummy = jnp.zeros((args.batch_size, 64, 64, 1), dtype=jnp.float32)
    rng, sub = jax.random.split(rng)
    state, loss, _, _ = vae_train_step(
        state, dummy, sub,
        latent_dim=args.latent_dim, base_ch=args.base_ch,
        beta=0.0, free_bits=args.free_bits, pos_weight=args.pos_weight,
    )
    jax.block_until_ready(loss)
    print("JIT compilation done.")

    # Re-initialise after warmup dummy step
    rng = jax.random.PRNGKey(args.seed)
    model, state = make_vae_state(
        rng, latent_dim=args.latent_dim, base_ch=args.base_ch, lr=args.lr
    )

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "vae_best.pkl")
    best_loss = float("inf")
    t0        = time.time()

    print(f"\n{'Epoch':>5}  {'beta':>6}  {'Loss':>8}  "
          f"{'BCE':>8}  {'KL':>8}  {'Best':>8}  {'Time':>7}")
    print("-" * 65)

    for ep in range(args.epochs):
        beta = min(args.beta_final,
                   args.beta_final * (ep + 1) / max(1, args.warmup_epochs))
        losses, bces, kls = [], [], []

        for batch_np in iter_minibatches_np(Xtrain, args.batch_size,
                                             seed=args.seed + ep):
            batch = jnp.asarray(batch_np)
            rng, sub = jax.random.split(rng)
            state, loss, bce, kl = vae_train_step(
                state, batch, sub,
                latent_dim=args.latent_dim, base_ch=args.base_ch,
                beta=beta, free_bits=args.free_bits,
                pos_weight=args.pos_weight,
            )
            losses.append(float(loss))
            bces.append(float(bce))
            kls.append(float(kl))

        ep_loss = float(np.mean(losses))
        saved   = ""
        if ep_loss < best_loss:
            best_loss = ep_loss
            save_params(ckpt_path, state.params,
                        info={"epoch": ep + 1, "loss": best_loss,
                              "latent_dim": args.latent_dim,
                              "base_ch": args.base_ch})
            saved = " *saved*"

        elapsed = time.time() - t0
        print(f"{ep+1:>5}  {beta:>6.3f}  {ep_loss:>8.3f}  "
              f"{np.mean(bces):>8.3f}  {np.mean(kls):>8.3f}  "
              f"{best_loss:>8.3f}  {elapsed:>6.1f}s{saved}")

    print(f"\nTraining done.  Best loss: {best_loss:.4f}")
    print(f"Best VAE checkpoint: {ckpt_path}")
    print("\nStep 3 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Train VAE.")
    parser.add_argument("--data_dir",       type=str,   default="./data")
    parser.add_argument("--ckpt_dir",       type=str,   default="./checkpoints")
    parser.add_argument("--latent_dim",     type=int,   default=100)
    parser.add_argument("--base_ch",        type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--warmup_epochs",  type=int,   default=40)
    parser.add_argument("--beta_final",     type=float, default=0.2)
    parser.add_argument("--free_bits",      type=float, default=0.0)
    parser.add_argument("--pos_weight",     type=float, default=5.0)
    parser.add_argument("--seed",           type=int,   default=0)
    args = parser.parse_args()
    main(args)
