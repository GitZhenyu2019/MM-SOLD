"""
step2_train_ddpm.py
===================
Train the Latent DDPM on digit-8 NRAE latent vectors.
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LATENT_DIR, CKPT_DIR, RESULTS_DIR,
    DDPM_HIDDEN, DDPM_N_LAYERS, DDPM_T_STEPS, DDPM_LR, DDPM_WD,
    DDPM_EPOCHS, DDPM_BATCH,
)
from ddpm_model import (
    make_ddpm_state, make_cosine_schedule, ddpm_train_step,
    save_ddpm, count_ddpm_params,
)


def main(args):
    print("=" * 60)
    print("Step 2: Train Latent DDPM on digit-8 NRAE latents")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ── Load training latents ────────────────────────────────────────────────
    Z_train = np.load(os.path.join(args.latent_dir, "Z_train.npy"))
    print(f"\nTraining latents: {Z_train.shape}  (latent_dim={Z_train.shape[1]})")
    latent_dim = Z_train.shape[1]

    print("\nNormalising latents (per-dim z-score) ...")
    mu_w   = Z_train.mean(axis=0).astype(np.float32)
    std_w  = Z_train.std(axis=0).astype(np.float32)
    # Dead dimensions (std ≈ 0) are left unchanged rather than amplified
    std_safe = np.where(std_w > 1e-3, std_w, 1.0).astype(np.float32)
    Z_train_w = ((Z_train - mu_w) / std_safe).astype(np.float32)
    print(f"  Normalised: mean={Z_train_w.mean():.4f}  std={Z_train_w.std():.4f}"
          f"  min={Z_train_w.min():.4f}  max={Z_train_w.max():.4f}"
          f"  active_dims={int((std_w > 1e-3).sum())}/{Z_train.shape[1]}")

    # ── Build noise schedule ─────────────────────────────────────────────────
    T              = args.T
    alphas_cumprod_np, betas_np = make_cosine_schedule(T)
    alphas_cumprod = jnp.array(alphas_cumprod_np)   # (T+1,)
    print(f"\nCosine schedule:  T={T}  β_min={betas_np.min():.2e}  "
          f"β_max={betas_np.max():.2e}  ᾱ_T={alphas_cumprod_np[-1]:.4f}")

    # ── Initialise model ─────────────────────────────────────────────────────
    N          = len(Z_train_w)
    steps_ep   = max(1, N // args.batch_size)
    total_steps = args.epochs * steps_ep

    rng = jax.random.PRNGKey(args.seed)
    rng, k_init = jax.random.split(rng)
    model, state = make_ddpm_state(
        k_init,
        latent_dim=latent_dim,
        hidden=args.hidden_dim,
        n_layers=args.n_layers,
        T=T,
        lr=args.lr,
        wd=args.wd,
        total_steps=total_steps,
    )
    n_params = count_ddpm_params(state.params)
    print(f"\nMLPDenoiser  hidden={args.hidden_dim}  n_layers={args.n_layers}  "
          f"params={n_params:,}")

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nTraining  epochs={args.epochs}  batch={args.batch_size}  "
          f"steps/epoch={steps_ep}  total_steps={total_steps}")
    print(f"  lr={args.lr:.1e}  wd={args.wd:.1e}\n")

    log_interval = max(1, args.epochs // 20)
    rng, k_train = jax.random.split(rng)
    best_loss      = float("inf")
    best_params    = None
    best_epoch     = 0
    losses_per_ep  = []

    t_start = time.perf_counter()

    for ep in range(args.epochs):
        ep_loss = 0.0
        n_steps = 0
        # shuffle manually
        perm = np.random.default_rng(args.seed + ep).permutation(N)
        Z_shuf = Z_train_w[perm]
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            z_batch = jnp.asarray(Z_shuf[i: i + args.batch_size])
            state, loss_val, k_train = ddpm_train_step(
                state, z_batch, k_train,
                alphas_cumprod=alphas_cumprod,
                T=T,
                latent_dim=latent_dim,
                hidden=args.hidden_dim,
                n_layers=args.n_layers,
            )
            ep_loss += float(loss_val)
            n_steps += 1

        ep_loss /= max(n_steps, 1)
        losses_per_ep.append(ep_loss)

        if ep_loss < best_loss:
            best_loss   = ep_loss
            best_params = jax.device_get(state.params)
            best_epoch  = ep + 1

        if (ep + 1) % log_interval == 0 or ep == 0:
            print(f"  epoch {ep+1:4d}/{args.epochs}  loss={ep_loss:.5f}"
                  f"  best={best_loss:.5f} @ ep{best_epoch}")

    training_time = time.perf_counter() - t_start
    print(f"\nTraining complete in {training_time:.1f}s  "
          f"({training_time/60:.1f} min)")
    print(f"Best train loss: {best_loss:.5f}  @ epoch {best_epoch}")

    # ── Save checkpoint ──────────────────────────────────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "ddpm_best.pkl")
    info = {
        "latent_dim":     latent_dim,
        "hidden":         args.hidden_dim,
        "n_layers":       args.n_layers,
        "T":              T,
        "best_epoch":     best_epoch,
        "best_loss":      best_loss,
        "training_time_s": training_time,
        "whiten_mu":      mu_w,
        "whiten_std":     std_safe,
    }
    save_ddpm(ckpt_path, best_params, info=info)
    print(f"\nCheckpoint: {ckpt_path}")

    # ── Save training time to results ────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    timing_path = os.path.join(args.results_dir, "ddpm_training_time.json")
    with open(timing_path, "w") as f:
        json.dump({
            "training_time_s":   training_time,
            "training_time_min": training_time / 60.0,
            "epochs":            args.epochs,
            "n_train":           N,
            "n_params":          n_params,
        }, f, indent=2)
    print(f"Timing: {timing_path}")

    # ── Save loss curve ──────────────────────────────────────────────────────
    np.save(os.path.join(args.results_dir, "ddpm_loss_curve.npy"),
            np.array(losses_per_ep))
    print(f"\nStep 2 complete!  Training time: {training_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Train Latent DDPM on digit-8 NRAE latents.")
    parser.add_argument("--latent_dir",  type=str, default=LATENT_DIR)
    parser.add_argument("--ckpt_dir",    type=str, default=CKPT_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--hidden_dim",  type=int, default=DDPM_HIDDEN)
    parser.add_argument("--n_layers",    type=int, default=DDPM_N_LAYERS)
    parser.add_argument("--lr",          type=float, default=DDPM_LR)
    parser.add_argument("--wd",          type=float, default=DDPM_WD)
    parser.add_argument("--epochs",      type=int, default=DDPM_EPOCHS)
    parser.add_argument("--batch_size",  type=int, default=DDPM_BATCH)
    parser.add_argument("--T",           type=int, default=DDPM_T_STEPS)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()
    main(args)
