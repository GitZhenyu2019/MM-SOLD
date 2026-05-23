"""
step1_train_nrae.py
===================
Train the NRAE on CelebA-HQ-256 and encode all images to latent vectors.
"""
import argparse
import os
import re
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_USE_DIR, VALID_USE_DIR, DATA_DIR, LATENT_DIR, CKPT_DIR, NRAE_CKPT,
    LATENT_DIM, ENC1_HIDDEN, DEC_HIDDEN, N_DCT, UNET_BASE_CH,
    N_TRAIN, N_TEST, SOLD_K_WHITEN,
)
from nrae_model_celeba import (
    dct2d_crop_rgb_np,
    make_nrae_state, nrae_train_step,
    encode_dataset_nrae, decode_latents_nrae,
    save_params, load_params, count_params, print_nrae_summary,
    NRAEModelCeleba,
)


# ─── Image loading ────────────────────────────────────────────────────────────

def _load_pngs_from_dir(folder: str, n_max: int) -> np.ndarray:
    """
    Load the first n_max PNG files from folder (sorted by filename).
    """
    from PIL import Image
    files = sorted(
        (f for f in os.listdir(folder) if f.lower().endswith(".png")),
        key=lambda f: int(re.search(r'\d+', f).group()),
    )
    files = files[:n_max]
    if len(files) < n_max:
        raise ValueError(
            f"Only {len(files)} PNGs in {folder}, need {n_max}. "
            "Run step0_prepare_subset.py first.")
    imgs = []
    for fname in files:
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.LANCZOS)
        imgs.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(imgs, axis=0)      # (N, 256, 256, 3)


# ─── Training loop ────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("Step 1: Train NRAE on CelebA-HQ-256 + encode latents")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    os.makedirs(args.data_dir,   exist_ok=True)
    os.makedirs(args.latent_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir,   exist_ok=True)

    # ── 1. Load images ────────────────────────────────────────────────────────
    print(f"\nLoading {args.n_train} training PNGs from: {args.train_use}")
    imgs_train = _load_pngs_from_dir(args.train_use, args.n_train)
    print(f"  imgs_train: {imgs_train.shape}  [{imgs_train.min():.3f}, {imgs_train.max():.3f}]")

    print(f"Loading {args.n_test} validation PNGs from: {args.valid_use}")
    imgs_test = _load_pngs_from_dir(args.valid_use, args.n_test)
    print(f"  imgs_test:  {imgs_test.shape}")

    # ── 2. DCT preprocessing ──────────────────────────────────────────────────
    print(f"\nApplying DCT-II (keep {args.n_dct}x{args.n_dct} per channel) ...")
    dct_dim = 3 * args.n_dct ** 2
    X_train_dct = dct2d_crop_rgb_np(imgs_train, n_dct=args.n_dct)
    X_test_dct  = dct2d_crop_rgb_np(imgs_test,  n_dct=args.n_dct)
    print(f"  X_train_dct: {X_train_dct.shape}  ({dct_dim}-dim vectors)")
    print(f"  X_test_dct:  {X_test_dct.shape}")

    # ── 3. Initialise NRAE model ──────────────────────────────────────────────
    rng = jax.random.PRNGKey(args.seed)
    rng, k_init = jax.random.split(rng)
    nrae_model, state = make_nrae_state(
        k_init,
        latent_dim=args.latent_dim,
        enc1_hidden=args.enc1_hidden,
        dec_hidden=args.dec_hidden,
        n_dct=args.n_dct,
        unet_base_ch=args.unet_base_ch,
        lr=args.lr,
    )
    print_nrae_summary(
        state.params,
        latent_dim=args.latent_dim,
        enc1_hidden=args.enc1_hidden,
        dec_hidden=args.dec_hidden,
        n_dct=args.n_dct,
        unet_base_ch=args.unet_base_ch,
    )

    # ── 4. Training loop ──────────────────────────────────────────────────────
    N       = len(imgs_train)
    steps_ep = max(1, N // args.batch_size)
    print(f"Training  epochs={args.epochs}  batch={args.batch_size}  "
          f"steps/epoch={steps_ep}  lr={args.lr:.1e}")
    print(f"  sigma={args.sigma}  fixed_noise_sigma={args.fixed_noise_sigma}  "
          f"alpha={args.alpha}\n")

    log_interval = max(1, args.epochs // 50)
    best_val_loss  = float("inf")
    best_params    = None
    best_epoch     = 0

    t_start = time.perf_counter()

    for ep in range(args.epochs):
        # Shuffle training set
        perm = np.random.default_rng(args.seed + ep).permutation(N)
        X_dct_shuf  = X_train_dct[perm]
        imgs_shuf   = imgs_train[perm]

        ep_loss = 0.0
        n_steps = 0
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            x_dct_b = jnp.asarray(X_dct_shuf[i: i + args.batch_size])
            x_img_b = jnp.asarray(imgs_shuf[i: i + args.batch_size])
            rng, k_step = jax.random.split(rng)
            state, loss_val, recon_val, reg_val = nrae_train_step(
                state, x_dct_b, x_img_b, k_step,
                latent_dim=args.latent_dim,
                enc1_hidden=args.enc1_hidden,
                dec_hidden=args.dec_hidden,
                n_dct=args.n_dct,
                unet_base_ch=args.unet_base_ch,
                sigma=args.sigma,
                fixed_noise_sigma=args.fixed_noise_sigma,
                alpha=args.alpha,
            )
            ep_loss += float(loss_val)
            n_steps += 1

        ep_loss /= max(n_steps, 1)

        # Validation loss (one pass over test set, no grad)
        val_loss = 0.0
        n_val = 0
        for i in range(0, len(imgs_test) - args.batch_size + 1, args.batch_size):
            xd = jnp.asarray(X_test_dct[i: i + args.batch_size])
            xi = jnp.asarray(imgs_test[i:  i + args.batch_size])
            rng, k_val = jax.random.split(rng)
            # compute only reconstruction loss for validation (no grad needed)
            from nrae_model_celeba import nrae_loss_celeba
            loss_v, _ = nrae_loss_celeba(
                state.params, nrae_model, xd, xi, k_val,
                sigma=args.sigma,
                fixed_noise_sigma=args.fixed_noise_sigma,
                alpha=args.alpha,
            )
            val_loss += float(loss_v)
            n_val += 1
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params   = jax.device_get(state.params)
            best_epoch    = ep + 1

        if (ep + 1) % log_interval == 0 or ep == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  epoch {ep+1:4d}/{args.epochs}  "
                  f"train={ep_loss:.5f}  val={val_loss:.5f}  "
                  f"best_val={best_val_loss:.5f} @ ep{best_epoch}  "
                  f"({elapsed:.0f}s)")

    training_time = time.perf_counter() - t_start
    print(f"\nTraining complete in {training_time:.1f}s  ({training_time/60:.1f} min)")
    print(f"Best val loss: {best_val_loss:.5f}  @ epoch {best_epoch}")

    # ── 5. Save NRAE checkpoint ───────────────────────────────────────────────
    info = {
        "latent_dim":       args.latent_dim,
        "enc1_hidden":      args.enc1_hidden,
        "dec_hidden":       args.dec_hidden,
        "n_dct":            args.n_dct,
        "unet_base_ch":     args.unet_base_ch,
        "best_epoch":       best_epoch,
        "best_val_loss":    best_val_loss,
        "training_time_s":  training_time,
        "n_train":          N,
        "n_params":         count_params(best_params),
    }
    nrae_ckpt_path = os.path.join(args.ckpt_dir, "nrae_best.pkl")
    save_params(nrae_ckpt_path, best_params, info=info)
    print(f"\nCheckpoint saved: {nrae_ckpt_path}")

    # ── 6. Encode to NRAE latents ─────────────────────────────────────────────
    print("\nEncoding with NRAE ...")
    best_params_j = jax.tree_util.tree_map(jnp.asarray, best_params)
    Z_train = encode_dataset_nrae(
        nrae_model, best_params_j, imgs_train,
        n_dct=args.n_dct, batch_size=args.decode_batch)
    Z_test  = encode_dataset_nrae(
        nrae_model, best_params_j, imgs_test,
        n_dct=args.n_dct, batch_size=args.decode_batch)
    print(f"  Z_train: {Z_train.shape}  Z_test: {Z_test.shape}")

    # ── 6b. Whitened-space NN distance diagnostic (for SOLD_SIGMA_GMM tuning) ─
    print(f"\n[Diagnostic] NN distances in whitened latent space (K={SOLD_K_WHITEN}) ...")
    Z_np = np.array(Z_train, dtype=np.float64)
    Z_c  = Z_np - Z_np.mean(axis=0)
    _, S, Vt = np.linalg.svd(Z_c, full_matrices=False)
    k = min(SOLD_K_WHITEN, len(S))
    Z_white = (Z_c @ Vt[:k].T) / (S[:k] + 1e-8)   # (N, k)
    # nearest-neighbour distance for a random subsample (max 1000) for speed
    rng_diag = np.random.default_rng(0)
    idx = rng_diag.choice(len(Z_white), size=min(1000, len(Z_white)), replace=False)
    Zs  = Z_white[idx]
    # pairwise L2 via broadcasting (1000×1000 is fine in memory)
    diff = Zs[:, None, :] - Zs[None, :, :]          # (n, n, k)
    dmat = np.sqrt((diff ** 2).sum(-1))              # (n, n)
    np.fill_diagonal(dmat, np.inf)
    nn_dists = dmat.min(axis=1)
    print(f"  Whitened NN dist — mean={nn_dists.mean():.4f}  "
          f"median={np.median(nn_dists):.4f}  "
          f"p5={np.percentile(nn_dists, 5):.4f}  "
          f"p95={np.percentile(nn_dists, 95):.4f}")
    print(f"  Suggested SOLD_SIGMA_GMM ≈ {nn_dists.mean()/3:.4f} ~ {nn_dists.mean()/2:.4f}  "
          f"(current config: 0.03)")

    # ── 7. Save images and latents ───────────────────────────────────────────
    np.save(os.path.join(args.data_dir,   "images_train.npy"), imgs_train)
    np.save(os.path.join(args.data_dir,   "images_test.npy"),  imgs_test)
    np.save(os.path.join(args.latent_dir, "Z_train.npy"),      Z_train.astype(np.float32))
    np.save(os.path.join(args.latent_dir, "Z_test.npy"),       Z_test.astype(np.float32))

    print(f"\nSaved to:")
    print(f"  {args.data_dir}/")
    print(f"    images_train.npy  {imgs_train.shape}")
    print(f"    images_test.npy   {imgs_test.shape}")
    print(f"  {args.latent_dir}/")
    print(f"    Z_train.npy  {Z_train.shape}  Z_test.npy  {Z_test.shape}")
    print(f"\nStep 1 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Train NRAE on CelebA-HQ-256 + encode latents.")
    parser.add_argument("--train_use",   type=str, default=TRAIN_USE_DIR)
    parser.add_argument("--valid_use",   type=str, default=VALID_USE_DIR)
    parser.add_argument("--data_dir",    type=str, default=DATA_DIR)
    parser.add_argument("--latent_dir",  type=str, default=LATENT_DIR)
    parser.add_argument("--ckpt_dir",    type=str, default=CKPT_DIR)
    parser.add_argument("--n_train",     type=int, default=N_TRAIN)
    parser.add_argument("--n_test",      type=int, default=N_TEST)
    # NRAE architecture
    parser.add_argument("--latent_dim",      type=int,   default=LATENT_DIM)
    parser.add_argument("--enc1_hidden",     type=int,   default=ENC1_HIDDEN)
    parser.add_argument("--dec_hidden",      type=int,   default=DEC_HIDDEN)
    parser.add_argument("--n_dct",           type=int,   default=N_DCT)
    parser.add_argument("--unet_base_ch",    type=int,   default=UNET_BASE_CH)
    # NRAE training
    parser.add_argument("--epochs",          type=int,   default=300)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--sigma",           type=float, default=2.0,
                        help="NRAE regularisation σ (noise level in input space)")
    parser.add_argument("--fixed_noise_sigma", type=float, default=1e-3,
                        help="σ_fixed for noise injection (encoder perturbation)")
    parser.add_argument("--alpha",           type=float, default=100.0,
                        help="log-cosh sharpness parameter")
    parser.add_argument("--decode_batch",    type=int,   default=16,
                        help="Batch size for encoding / decoding (memory)")
    parser.add_argument("--seed",            type=int,   default=0)
    args = parser.parse_args()
    main(args)
