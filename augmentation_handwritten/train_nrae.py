"""
train_nrae.py
=============
Train the NRAE used by all main digit experiments (classification and digit-8
generation): DCT 20x20 (400) -> 2048 -> 100 encoder, 100 -> 2048 -> 400 -> iDCT
-> U-Net(base 32) decoder, log-cosh reconstruction (alpha = 100) plus the
finite-difference noise-sensitivity regulariser (sigma_NRAE = 2, perturbation
1e-3); AdamW, lr 1e-4, batch 64, 100 epochs, trained on the 10,000 training
images. The checkpoint with the lowest loss on the 580 validation images is
saved as <ckpt_dir>/nrae_best.pkl; point RP1_NRAE_CKPT at it.

Model / loss / train-step code is imported unchanged from nrae_model.py.
step3b_train_nrae.py is the earlier variant that keeps the lowest training
loss; it is used only by ablation_studies/particle_nums.

Usage:
    python train_nrae.py --data_dir ./data --ckpt_dir ./checkpoints_final
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # nrae_model.py
DIGITS_DATA = "./data"                 # output of step1_prepare_data.py
DIGITS_CKPT = "./checkpoints_final"    # kept apart from step3b_train_nrae.py

import argparse
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from nrae_model import (  # noqa: E402
    NRAEModel, make_nrae_state, nrae_train_step, nrae_loss,
    dct2d_crop_np, save_params, count_params, print_nrae_summary,
)

# ── paper hyperparameters ────────────────────────────────────────────────────
LATENT_DIM = 100
ENC1_HIDDEN = 2048
DEC_HIDDEN = 2048
N_DCT = 20
UNET_BASE_CH = 32
LR = 1e-4
BATCH = 64
EPOCHS = 100
SIGMA_NRAE = 2.0          # regularisation noise level
FD_SIGMA = 1e-3           # finite-difference perturbation
ALPHA = 100.0             # log-cosh sharpness


_EVAL_CACHE = {}


def _eval_fn(model):
    """JIT-compiled per-batch validation step (cached per model instance)."""
    key = id(model)
    if key not in _EVAL_CACHE:
        @jax.jit
        def _f(params, d, im, rng):
            loss, (rec, reg) = nrae_loss(
                params, model, d, im, rng,
                sigma=SIGMA_NRAE, fixed_noise_sigma=FD_SIGMA, alpha=ALPHA)
            pred = model.apply({"params": params}, d)
            mse = jnp.mean((jnp.clip(pred, 0.0, 1.0) - im) ** 2)
            return loss, rec, reg, mse
        _EVAL_CACHE[key] = _f
    return _EVAL_CACHE[key]


def eval_epoch(model, params, x_dct, x_img, rng, batch=128):
    """Validation loss (same objective) + reconstruction MSE / PSNR."""
    losses, recons, regs, mses = [], [], [], []
    n = x_dct.shape[0]
    f = _eval_fn(model)
    for i in range(0, n, batch):
        d = jnp.asarray(x_dct[i:i + batch])
        im = jnp.asarray(x_img[i:i + batch])
        rng, sub = jax.random.split(rng)
        loss, rec, reg, mse = f(params, d, im, sub)
        w = d.shape[0]
        losses.append(float(loss) * w); recons.append(float(rec) * w)
        regs.append(float(reg) * w); mses.append(float(mse) * w)
    s = float(n)
    mse = sum(mses) / s
    return {"loss": sum(losses) / s, "recon": sum(recons) / s,
            "reg": sum(regs) / s, "mse": mse,
            "psnr": float(10.0 * np.log10(1.0 / max(mse, 1e-12)))}


def train_nrae(train_images, val_images, *, epochs=EPOCHS, batch=BATCH, lr=LR,
               latent_dim=LATENT_DIM, seed=0, ckpt_path=None, log_path=None,
               verbose=True):
    """Train the NRAE; returns (model, params, history)."""
    n_train = train_images.shape[0]
    train_dct = dct2d_crop_np(train_images, n_dct=N_DCT)
    train_ch = train_images[..., None].astype(np.float32)
    val_dct = dct2d_crop_np(val_images, n_dct=N_DCT)
    val_ch = val_images[..., None].astype(np.float32)

    rng = jax.random.PRNGKey(seed)
    model, state = make_nrae_state(
        rng, latent_dim=latent_dim, enc1_hidden=ENC1_HIDDEN,
        dec_hidden=DEC_HIDDEN, n_dct=N_DCT, unet_base_ch=UNET_BASE_CH, lr=lr)
    if verbose:
        print_nrae_summary(state.params, latent_dim=latent_dim,
                           enc1_hidden=ENC1_HIDDEN, dec_hidden=DEC_HIDDEN,
                           n_dct=N_DCT, unet_base_ch=UNET_BASE_CH)
        print(f"train {train_images.shape}  val {val_images.shape}  "
              f"epochs={epochs} batch={batch} lr={lr}")
        print(f"{'ep':>4} {'train':>10} {'recon':>10} {'reg':>10} "
              f"{'val':>10} {'val_mse':>9} {'PSNR':>6} {'sec':>7}")

    step_kw = dict(latent_dim=latent_dim, enc1_hidden=ENC1_HIDDEN,
                   dec_hidden=DEC_HIDDEN, n_dct=N_DCT,
                   unet_base_ch=UNET_BASE_CH, sigma=SIGMA_NRAE,
                   fixed_noise_sigma=FD_SIGMA, alpha=ALPHA)

    history, best = [], float("inf")
    t0 = time.time()
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(n_train)
        losses, recons, regs = [], [], []
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            rng, sub = jax.random.split(rng)
            state, loss, rec, reg = nrae_train_step(
                state, jnp.asarray(train_dct[idx]), jnp.asarray(train_ch[idx]),
                sub, **step_kw)
            losses.append(float(loss)); recons.append(float(rec))
            regs.append(float(reg))

        rng, sub = jax.random.split(rng)
        val = eval_epoch(model, state.params, val_dct, val_ch, sub)
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(losses)),
               "train_recon": float(np.mean(recons)),
               "train_reg": float(np.mean(regs)),
               "val_loss": val["loss"], "val_recon": val["recon"],
               "val_mse": val["mse"], "val_psnr": val["psnr"],
               "time": time.time() - t0}
        history.append(rec)

        saved = ""
        if ckpt_path is not None and val["loss"] < best:
            best = val["loss"]
            save_params(ckpt_path, state.params,
                        info={"epoch": ep + 1, "val_loss": best,
                              "val_psnr": val["psnr"],
                              "latent_dim": latent_dim,
                              "enc1_hidden": ENC1_HIDDEN,
                              "dec_hidden": DEC_HIDDEN, "n_dct": N_DCT,
                              "unet_base_ch": UNET_BASE_CH})
            saved = " *"
        if log_path is not None:
            with open(log_path, "w") as f:
                json.dump(history, f, indent=1)
        if verbose:
            print(f"{ep+1:>4} {rec['train_loss']:>10.4f} "
                  f"{rec['train_recon']:>10.4f} {rec['train_reg']:>10.4f} "
                  f"{rec['val_loss']:>10.4f} {rec['val_mse']:>9.5f} "
                  f"{rec['val_psnr']:>6.2f} {rec['time']:>7.1f}{saved}",
                  flush=True)

    return model, state.params, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DIGITS_DATA)
    p.add_argument("--ckpt_dir", default=DIGITS_CKPT)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    os.makedirs(a.ckpt_dir, exist_ok=True)
    tr = np.load(os.path.join(a.data_dir, "train_images.npy"))
    va = np.load(os.path.join(a.data_dir, "val_images.npy"))

    _, params, hist = train_nrae(
        tr, va, epochs=a.epochs, batch=a.batch, lr=a.lr,
        latent_dim=a.latent_dim, seed=a.seed,
        ckpt_path=os.path.join(a.ckpt_dir, "nrae_best.pkl"),
        log_path=os.path.join(a.ckpt_dir, "nrae_history.json"))
    save_params(os.path.join(a.ckpt_dir, "nrae_last.pkl"), params,
                info={"epoch": a.epochs, "latent_dim": a.latent_dim,
                      "enc1_hidden": ENC1_HIDDEN, "dec_hidden": DEC_HIDDEN,
                      "n_dct": N_DCT, "unet_base_ch": UNET_BASE_CH})
    print(f"\nbest val loss {min(h['val_loss'] for h in hist):.4f}  "
          f"best val PSNR {max(h['val_psnr'] for h in hist):.2f} dB")


if __name__ == "__main__":
    main()
