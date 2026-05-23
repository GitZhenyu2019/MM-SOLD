"""
step1b_encode_all.py
====================
Encode all 27K training images through the trained NRAE → Z_train_full.npy.
"""
import argparse
import glob
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_DIR, LATENT_DIR, NRAE_CKPT,
    LATENT_DIM, ENC1_HIDDEN, DEC_HIDDEN, N_DCT, UNET_BASE_CH, IMG_SIZE,
)
from nrae_model_celeba import (
    NRAEModelCeleba, encode_dataset_nrae, load_params as nrae_load,
)


def _load_image(path: str, size: int) -> np.ndarray:
    """PNG/JPEG → (size, size, 3) float32 [0, 1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def main(args):
    print("=" * 60)
    print("Step 1b: Encode all training images with NRAE")
    print("=" * 60)
    print(f"JAX backend : {jax.default_backend()}  devices : {jax.devices()}")
    print(f"Train dir   : {TRAIN_DIR}")
    print(f"Output dir  : {LATENT_DIR}")

    # ── Collect image paths ──────────────────────────────────────────────────
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(glob.glob(os.path.join(TRAIN_DIR, ext)))
    paths = sorted(paths)
    N = len(paths)
    if N == 0:
        raise FileNotFoundError(f"No images found in {TRAIN_DIR}")
    print(f"\nFound {N} images")

    # ── Check if output already exists ──────────────────────────────────────
    out_path = os.path.join(LATENT_DIR, "Z_train_full.npy")
    if os.path.exists(out_path):
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape[0] == N:
            print(f"\nZ_train_full.npy already exists with {N} rows — skipping.")
            return
        print(f"\nExisting file has {existing.shape[0]} rows (expected {N}) — re-encoding.")

    # ── Load NRAE ────────────────────────────────────────────────────────────
    print(f"\nLoading NRAE: {NRAE_CKPT}")
    params_np, info = nrae_load(NRAE_CKPT)
    latent_dim   = info.get("latent_dim",   LATENT_DIM)
    enc1_hidden  = info.get("enc1_hidden",  ENC1_HIDDEN)
    dec_hidden   = info.get("dec_hidden",   DEC_HIDDEN)
    n_dct        = info.get("n_dct",        N_DCT)
    unet_base_ch = info.get("unet_base_ch", UNET_BASE_CH)

    # Inference only — create model object directly, no optimizer state
    model  = NRAEModelCeleba(
        latent_dim=latent_dim, enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch,
    )
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    print(f"  NRAE loaded  latent_dim={latent_dim}  n_dct={n_dct}")

    # ── Encode in batches ────────────────────────────────────────────────────
    os.makedirs(LATENT_DIR, exist_ok=True)
    Z_all = np.zeros((N, latent_dim), dtype=np.float32)
    B = args.batch_size
    log_every = max(1, (N // B) // 20)   # ~20 log lines total

    print(f"\nEncoding {N} images  (batch_size={B}) ...")
    for batch_start in range(0, N, B):
        batch_paths = paths[batch_start:batch_start + B]
        batch_imgs  = np.stack(
            [_load_image(p, IMG_SIZE) for p in batch_paths])  # (b, 256, 256, 3)
        z_batch = encode_dataset_nrae(
            model, params, batch_imgs, n_dct=n_dct, batch_size=len(batch_imgs))
        Z_all[batch_start:batch_start + len(batch_paths)] = z_batch

        step = batch_start // B
        if step % log_every == 0:
            done = batch_start + len(batch_paths)
            print(f"  [{done:5d}/{N}]  "
                  f"z_mean={z_batch.mean():+.4f}  z_std={z_batch.std():.4f}")

    print(f"\nFull latent array: {Z_all.shape}"
          f"  mean={Z_all.mean():+.4f}  std={Z_all.std():.4f}"
          f"  min={Z_all.min():.4f}  max={Z_all.max():.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(out_path, Z_all)
    print(f"Saved: {out_path}")

    txt_path = os.path.join(LATENT_DIR, "Z_train_full_paths.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(paths))
    print(f"Saved path list: {txt_path}")

    print(f"\nStep 1b complete!  {N} latents saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1b: Encode all training images with NRAE.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Images per encoding batch (reduce if OOM)")
    main(parser.parse_args())
