"""
step3b_test_nrae.py
===================
Visual reconstruction test for the trained NRAE.
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def main(args):
    print("=" * 60)
    print("Step 3b: NRAE Reconstruction Test")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    train_images = np.load(os.path.join(args.data_dir, "train_images.npy"))
    print(f"Loaded train_images: {train_images.shape}")

    # Pick evenly spaced samples (covers all digit classes if 200/class)
    rng_np = np.random.default_rng(args.seed)
    idx    = rng_np.choice(len(train_images), size=args.n_images, replace=False)
    idx    = np.sort(idx)
    sample = train_images[idx]   # (n_images, 64, 64) ink=1

    # ----------------------------------------------------------------
    # Load checkpoint
    # ----------------------------------------------------------------
    ckpt_path = os.path.join(args.ckpt_dir, "nrae_best.pkl")
    print(f"Loading checkpoint: {ckpt_path}")
    params_np, info = load_params(ckpt_path)
    print(f"  Checkpoint info: {info}")

    latent_dim   = info.get("latent_dim",   100)
    enc1_hidden  = info.get("enc1_hidden",  2048)
    dec_hidden   = info.get("dec_hidden",   2048)
    n_dct        = info.get("n_dct",        20)
    unet_base_ch = info.get("unet_base_ch", 32)

    # ----------------------------------------------------------------
    # Rebuild model
    # ----------------------------------------------------------------
    rng    = jax.random.PRNGKey(0)
    model, _ = make_nrae_state(
        rng,
        latent_dim=latent_dim, enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch,
    )
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    print_nrae_summary(params,
                       latent_dim=latent_dim, enc1_hidden=enc1_hidden,
                       dec_hidden=dec_hidden, n_dct=n_dct,
                       unet_base_ch=unet_base_ch)

    # ----------------------------------------------------------------
    # Encode → decode
    # ----------------------------------------------------------------
    print(f"Encoding {args.n_images} images …", end=" ", flush=True)
    Z    = encode_dataset_nrae(model, params, sample,
                               n_dct=n_dct, batch_size=args.n_images)
    print("done")
    print(f"Decoding …", end=" ", flush=True)
    recon = decode_latents_nrae(model, params, jnp.asarray(Z),
                                batch_size=args.n_images)
    print("done")

    # ----------------------------------------------------------------
    # Plot: two rows per image — original (top) | reconstructed (bottom)
    # ----------------------------------------------------------------
    n   = args.n_images
    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.0))

    for j in range(n):
        axes[0, j].imshow(sample[j], cmap="gray_r", vmin=0, vmax=1,
                          interpolation="nearest")
        axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_title("original", fontsize=7, pad=2)

        axes[1, j].imshow(recon[j], cmap="gray_r", vmin=0, vmax=1,
                          interpolation="nearest")
        axes[1, j].axis("off")
        if j == 0:
            axes[1, j].set_title("reconstructed", fontsize=7, pad=2)

    fig.tight_layout(pad=0.3)

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved reconstruction comparison → {out_path}")

    # ----------------------------------------------------------------
    # Print pixel-level reconstruction error
    # ----------------------------------------------------------------
    mse  = float(np.mean((sample - recon) ** 2))
    mae  = float(np.mean(np.abs(sample - recon)))
    psnr = float(-10 * np.log10(max(mse, 1e-12)))
    print(f"\nReconstruction error over {n} images:")
    print(f"  MSE  = {mse:.5f}")
    print(f"  MAE  = {mae:.5f}")
    print(f"  PSNR = {psnr:.2f} dB")
    print("\nStep 3b test complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3b: NRAE reconstruction visual test.")
    parser.add_argument("--data_dir",  type=str, default="./data")
    parser.add_argument("--ckpt_dir",  type=str, default="./checkpoints")
    parser.add_argument("--n_images",  type=int, default=16)
    parser.add_argument("--out",       type=str,
                        default="./figures/nrae_reconstruction_test.png")
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()
    main(args)
