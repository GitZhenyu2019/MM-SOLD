"""
step1_prepare_latents.py
============================
Data prep for the classification experiment
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

import config as cfg  # noqa: E402  (sets up sys.path for everything below)
from data_utils import (  # noqa: E402  (augmentation_handwritten)
    load_images_and_digits,
    select_train_val_test_split,
    full_preprocess_pipeline,
)
from nrae_model import NRAEModel, make_nrae_state, encode_dataset_nrae, load_params  # noqa: E402
import tilting_lion as TL  # noqa: E402  (release/common)
from common import mean_nn_distance  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=cfg.LATENT_DIR)
    ap.add_argument("--proj_k", type=int, default=cfg.PROJ_K)
    ap.add_argument("--raw_img", default=cfg.RAW_IMG)
    ap.add_argument("--raw_label", default=cfg.RAW_LABEL)
    ap.add_argument("--nrae_ckpt", default=cfg.NRAE_CKPT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 1: prepare data + the NRAE + top-%d subspace" % a.proj_k)
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"  raw_img    = {a.raw_img}")
    print(f"  raw_label  = {a.raw_label}")
    print(f"  nrae_ckpt  = {a.nrae_ckpt}")
    for p in (a.raw_img, a.raw_label, a.nrae_ckpt):
        if not os.path.exists(p):
            print(f"  [WARNING] missing: {p}")

    # ── 1. Load + preprocess (same split and preprocessing as all digit experiments) ────
    print("\n[1] Loading + preprocessing images ...")
    images_np, digits_np = load_images_and_digits(a.raw_img, a.raw_label)
    print(f"  raw: {images_np.shape}, labels: {digits_np.shape}")

    (train_imgs, train_labels,
     val_imgs, val_labels,
     test_imgs, test_labels) = select_train_val_test_split(
        images_np, digits_np,
        n_train_per_class=cfg.N_TRAIN_PER_CLASS,
        n_val_per_class=cfg.N_VAL_PER_CLASS,
        n_test_per_class=cfg.N_TEST_PER_CLASS,
        seed=cfg.DATA_SEED,
    )
    print(f"  train {train_imgs.shape}  val {val_imgs.shape}  test {test_imgs.shape}")

    _cpu = jax.devices('cpu')[0]
    with jax.default_device(_cpu):
        train_64, target_rms = full_preprocess_pipeline(train_imgs)
        val_64, _ = full_preprocess_pipeline(val_imgs, target_rms=target_rms)
        test_64, _ = full_preprocess_pipeline(test_imgs, target_rms=target_rms)
    train_64_np = np.array(train_64)
    val_64_np = np.array(val_64)
    test_64_np = np.array(test_64)
    print(f"  preprocessed: train {train_64_np.shape}  val {val_64_np.shape}  "
          f"test {test_64_np.shape}  (target_rms={float(target_rms):.5f})")

    print(f"\n[2] Loading NRAE checkpoint: {a.nrae_ckpt}")
    params_np, info = load_params(a.nrae_ckpt)
    print(f"  checkpoint info: {info}")
    latent_dim = info.get("latent_dim", cfg.LATENT_DIM)
    enc1_hidden = info.get("enc1_hidden", cfg.ENC1_HIDDEN)
    dec_hidden = info.get("dec_hidden", cfg.DEC_HIDDEN)
    n_dct = info.get("n_dct", cfg.N_DCT)
    unet_base_ch = info.get("unet_base_ch", cfg.UNET_BASE_CH)
    if latent_dim != cfg.LATENT_DIM:
        print(f"  [WARNING] checkpoint latent_dim={latent_dim} != "
              f"config.LATENT_DIM={cfg.LATENT_DIM}")

    model, _ = make_nrae_state(jax.random.PRNGKey(0), latent_dim=latent_dim,
                               enc1_hidden=enc1_hidden, dec_hidden=dec_hidden,
                               n_dct=n_dct, unet_base_ch=unet_base_ch)
    params = jax.tree_util.tree_map(jnp.asarray, params_np)

    print("  encoding train/val/test ...")
    Z_train = encode_dataset_nrae(model, params, train_64_np, n_dct=n_dct, batch_size=64)
    Z_val = encode_dataset_nrae(model, params, val_64_np, n_dct=n_dct, batch_size=64)
    Z_test = encode_dataset_nrae(model, params, test_64_np, n_dct=n_dct, batch_size=64)
    print(f"  Z_train {Z_train.shape}  Z_val {Z_val.shape}  Z_test {Z_test.shape}")

    y_train = train_labels.astype(np.int32)
    y_val = val_labels.astype(np.int32)
    y_test = test_labels.astype(np.int32)

    # ── 3. Common top-k PCA subspace from POOLED training latents ──────────
    print(f"\n[3] Top-{a.proj_k} PCA subspace from pooled training latents "
          f"(all {cfg.N_CLASSES} classes, plain global covariance) ...")
    mean_g, V, evals = TL.projection_stats(Z_train, k=a.proj_k)
    from latent_utils import compute_sample_mean_cov  # release/common
    _, cov_full = compute_sample_mean_cov(jnp.asarray(Z_train, jnp.float32))
    var_kept = float(jnp.sum(evals) / jnp.trace(cov_full))
    print(f"  top-{a.proj_k} of {Z_train.shape[1]}: "
          f"{100 * var_kept:.4f}% of pooled latent variance retained")
    if var_kept < 0.998:
        print(f"  [NOTE] paper reports >99.8% for k={a.proj_k}; got "
              f"{100*var_kept:.4f}% -- check NRAE checkpoint / k.")

    U_train = TL.to_subspace(Z_train, mean_g, V)
    U_val = TL.to_subspace(Z_val, mean_g, V)
    U_test = TL.to_subspace(Z_test, mean_g, V)

    # ── 4. Per-class NN_c in the top-k subspace ───────────────────
    print(f"\n[4] Per-class average nearest-neighbor distance NN_c (top-{a.proj_k}):")
    NN = {}
    for c in range(cfg.N_CLASSES):
        NN[c] = mean_nn_distance(U_train[y_train == c])
        print(f"    class {c}: NN_c = {NN[c]:.4f}")

    # ── Save everything ──────────────────────────────────────────────────
    np.save(os.path.join(a.out, "U_train.npy"), U_train.astype(np.float32))
    np.save(os.path.join(a.out, "y_train.npy"), y_train)
    np.save(os.path.join(a.out, "U_val.npy"), U_val.astype(np.float32))
    np.save(os.path.join(a.out, "y_val.npy"), y_val)
    np.save(os.path.join(a.out, "U_test.npy"), U_test.astype(np.float32))
    np.save(os.path.join(a.out, "y_test.npy"), y_test)
    np.save(os.path.join(a.out, "Z_train.npy"), Z_train.astype(np.float32))
    np.save(os.path.join(a.out, "proj_mean.npy"), np.asarray(mean_g, dtype=np.float32))
    np.save(os.path.join(a.out, "proj_V.npy"), np.asarray(V, dtype=np.float32))
    with open(os.path.join(a.out, "nn_distances.json"), "w") as f:
        json.dump({str(c): NN[c] for c in NN}, f, indent=2)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({
            "proj_k": a.proj_k, "var_kept": var_kept,
            "nrae_ckpt": a.nrae_ckpt, "nrae_info": info,
            "n_train_per_class": cfg.N_TRAIN_PER_CLASS,
            "n_val_per_class": cfg.N_VAL_PER_CLASS,
            "n_test_per_class": cfg.N_TEST_PER_CLASS,
        }, f, indent=2, default=float)

    print(f"\nSaved to {a.out}/")
    print("  U_train.npy, y_train.npy, U_val.npy, y_val.npy, U_test.npy, y_test.npy")
    print("  proj_mean.npy, proj_V.npy, nn_distances.json, meta.json")
    print("\nStep 1 complete!")


if __name__ == "__main__":
    main()
