"""
step1c_encode_test_full.py
==========================
Encode all 3 000 validation images through the trained NRAE.
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
    VALID_DIR, DATA_DIR, LATENT_DIR, NRAE_CKPT,
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
    print("Step 1c: Encode all 3K validation images with NRAE")
    print("=" * 60)
    print(f"JAX backend : {jax.default_backend()}  devices : {jax.devices()}")
    print(f"Valid dir   : {VALID_DIR}")
    print(f"Latent dir  : {LATENT_DIR}")
    print(f"Data dir    : {DATA_DIR}")

    # ── Collect image paths ──────────────────────────────────────────────────
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(glob.glob(os.path.join(VALID_DIR, ext)))
    paths = sorted(paths)
    N = len(paths)
    if N == 0:
        raise FileNotFoundError(f"No images found in {VALID_DIR}")
    print(f"\nFound {N} validation images")

    # ── Check if outputs already exist ──────────────────────────────────────
    z_out   = os.path.join(LATENT_DIR, "Z_test_full.npy")
    img_out = os.path.join(DATA_DIR,   "images_test_full.npy")
    if os.path.exists(z_out) and os.path.exists(img_out):
        z_ex   = np.load(z_out,   mmap_mode="r")
        img_ex = np.load(img_out, mmap_mode="r")
        if z_ex.shape[0] == N and img_ex.shape[0] == N:
            print(f"\nOutputs already exist with {N} rows — skipping.")
            print(f"  {z_out}")
            print(f"  {img_out}")
            return
        print(f"\nExisting files have {z_ex.shape[0]} rows (expected {N}) "
              "— re-encoding.")

    # ── Load NRAE ────────────────────────────────────────────────────────────
    print(f"\nLoading NRAE: {NRAE_CKPT}")
    params_np, info = nrae_load(NRAE_CKPT)
    latent_dim   = info.get("latent_dim",   LATENT_DIM)
    enc1_hidden  = info.get("enc1_hidden",  ENC1_HIDDEN)
    dec_hidden   = info.get("dec_hidden",   DEC_HIDDEN)
    n_dct        = info.get("n_dct",        N_DCT)
    unet_base_ch = info.get("unet_base_ch", UNET_BASE_CH)

    model  = NRAEModelCeleba(
        latent_dim=latent_dim, enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch,
    )
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    print(f"  NRAE loaded  latent_dim={latent_dim}  n_dct={n_dct}")

    # ── Encode in batches, collect images ────────────────────────────────────
    os.makedirs(LATENT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)
    Z_all    = np.zeros((N, latent_dim), dtype=np.float32)
    imgs_all = np.zeros((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    B = args.batch_size
    log_every = max(1, (N // B) // 20)

    print(f"\nEncoding {N} images  (batch_size={B}) ...")
    for batch_start in range(0, N, B):
        batch_paths = paths[batch_start:batch_start + B]
        batch_imgs  = np.stack(
            [_load_image(p, IMG_SIZE) for p in batch_paths])  # (b, 256, 256, 3)
        z_batch = encode_dataset_nrae(
            model, params, batch_imgs, n_dct=n_dct, batch_size=len(batch_imgs))
        end = batch_start + len(batch_paths)
        Z_all[batch_start:end]    = z_batch
        imgs_all[batch_start:end] = batch_imgs

        step = batch_start // B
        if step % log_every == 0:
            print(f"  [{end:5d}/{N}]  "
                  f"z_mean={z_batch.mean():+.4f}  z_std={z_batch.std():.4f}")

    print(f"\nLatent array : {Z_all.shape}  "
          f"mean={Z_all.mean():+.4f}  std={Z_all.std():.4f}")
    print(f"Image array  : {imgs_all.shape}  "
          f"min={imgs_all.min():.3f}  max={imgs_all.max():.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(z_out,   Z_all)
    np.save(img_out, imgs_all)
    print(f"\nSaved:")
    print(f"  {z_out}")
    print(f"  {img_out}")
    print(f"\nStep 1c complete!  {N} test latents + images saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1c: Encode all 3K validation images with NRAE.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Images per encoding batch (reduce if OOM)")
    main(parser.parse_args())
