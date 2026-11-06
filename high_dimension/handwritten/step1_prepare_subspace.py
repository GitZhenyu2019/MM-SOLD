"""
step1_prepare_subspace.py
=============================
Data prep for the digit-8 GENERATION experiment
"""
import argparse
import json
import os

import numpy as np
import jax

import config as cfg                          # noqa: E402 (sets up sys.path)
from data_utils import (                         # noqa: E402 (augmentation_handwritten)
    load_images_and_digits,
    select_train_val_test_split,
    full_preprocess_pipeline,
)
import tilting_lion as TL                         # noqa: E402 (release/common)
from nrae_model import encode_dataset_nrae         # noqa: E402 (augmentation_handwritten)
from common import load_nrae, mean_nn_distance  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "subspace"))
    ap.add_argument("--proj_k", type=int, default=cfg.PROJ_K)
    ap.add_argument("--raw_img", default=cfg.RAW_IMG)
    ap.add_argument("--raw_label", default=cfg.RAW_LABEL)
    ap.add_argument("--nrae_ckpt", default=cfg.NRAE_CKPT)
    ap.add_argument("--seed", type=int, default=cfg.SPLIT_SEED)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print(f"Step 1: digit-8 generation -- pooled top-{a.proj_k} subspace")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"  raw_img    = {a.raw_img}")
    print(f"  raw_label  = {a.raw_label}")
    print(f"  nrae_ckpt  = {a.nrae_ckpt}")
    for p in (a.raw_img, a.raw_label, a.nrae_ckpt):
        if not os.path.exists(p):
            print(f"  [WARNING] missing: {p}")

    # ── 1. Load + preprocess ────
    print("\n[1] Loading + preprocessing images (all 10 classes) ...")
    images_np, digits_np = load_images_and_digits(a.raw_img, a.raw_label)
    print(f"  raw: {images_np.shape}, labels: {digits_np.shape}")

    (train_imgs, train_labels,
     val_imgs, val_labels,
     test_imgs, test_labels) = select_train_val_test_split(
        images_np, digits_np,
        n_train_per_class=cfg.N_TRAIN_PER_CLASS,
        n_val_per_class=cfg.N_VAL_PER_CLASS,
        n_test_per_class=cfg.N_TEST_PER_CLASS,
        seed=a.seed,
    )
    print(f"  train {train_imgs.shape}  val {val_imgs.shape}  test {test_imgs.shape}")

    _cpu = jax.devices("cpu")[0]
    with jax.default_device(_cpu):
        train_64, target_rms = full_preprocess_pipeline(train_imgs)
        test_64, _ = full_preprocess_pipeline(test_imgs, target_rms=target_rms)
    train_64_np = np.array(train_64, dtype=np.float32)
    test_64_np  = np.array(test_64,  dtype=np.float32)
    print(f"  preprocessed: train {train_64_np.shape}  test {test_64_np.shape}  "
          f"(target_rms={float(target_rms):.5f})")

    y_train = train_labels.astype(np.int32)
    y_test  = test_labels.astype(np.int32)

    # digit-8's 300 held-out test IMAGES (pixel space)
    imgs_test8 = test_64_np[y_test == cfg.TARGET_DIGIT]
    print(f"  digit-{cfg.TARGET_DIGIT} test images: {imgs_test8.shape}")

    # ── 2. Encode ALL classes' training images with the NRAE ──────────
    print(f"\n[2] Loading NRAE checkpoint: {a.nrae_ckpt}")
    nrae_model, nrae_params, n_dct = load_nrae(a.nrae_ckpt, seed=0)

    print("  encoding all-class training images ...")
    Z_train = encode_dataset_nrae(nrae_model, nrae_params, train_64_np,
                                  n_dct=n_dct, batch_size=64)
    print(f"  Z_train (all classes): {Z_train.shape}")

    # ── 3. Pooled top-k PCA subspace from the WITHIN-CLASS covariance ──────
    print(f"\n[3] Common top-{a.proj_k} PCA subspace from pooled WITHIN-class "
          f"covariance (all {cfg.N_CLASSES} classes) ...")
    V, evals, class_means = TL.pooled_projection_stats(Z_train, y_train, k=a.proj_k)

    within_ss = 0.0
    for c in range(cfg.N_CLASSES):
        Zc = Z_train[y_train == c]
        within_ss += float(np.sum((Zc - Zc.mean(axis=0, keepdims=True)) ** 2))
    trace_pooled = within_ss / max(Z_train.shape[0] - cfg.N_CLASSES, 1)
    var_kept = float(np.sum(np.asarray(evals)) / trace_pooled)
    print(f"  top-{a.proj_k} of {Z_train.shape[1]}: "
          f"{100 * var_kept:.4f}% of pooled within-class variance retained")
    if var_kept < cfg.MIN_VARIANCE_RETAINED:
        print(f"  [NOTE] paper reports >99.8% for dsub={a.proj_k}; got "
              f"{100 * var_kept:.4f}% -- check NRAE checkpoint / dsub.")

    mean_8 = class_means[cfg.TARGET_DIGIT]
    U_train8 = TL.to_subspace(Z_train[y_train == cfg.TARGET_DIGIT], mean_8, V)
    print(f"  U_train8 (top-{a.proj_k}): {U_train8.shape}")

    # ── 4. NN in the top-k subspace, digit-8 only ──────────────────
    NN8 = mean_nn_distance(U_train8)
    print(f"\n[4] NN (digit-8, top-{a.proj_k} subspace) = {NN8:.5f}")

    # ── Save everything ──────────────────────────────────────────────────
    np.save(os.path.join(a.out, "U_train8.npy"), U_train8.astype(np.float32))
    np.save(os.path.join(a.out, "mean_8.npy"), np.asarray(mean_8, dtype=np.float32))
    np.save(os.path.join(a.out, "proj_V.npy"), np.asarray(V, dtype=np.float32))
    np.save(os.path.join(a.out, "imgs_test8.npy"), imgs_test8.astype(np.float32))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({
            "proj_k": a.proj_k, "var_kept": var_kept, "NN8": NN8,
            "n_dct": n_dct, "target_digit": cfg.TARGET_DIGIT,
            "n_train8": int(U_train8.shape[0]), "n_test8": int(imgs_test8.shape[0]),
            "nrae_ckpt": a.nrae_ckpt, "seed": a.seed,
        }, f, indent=2, default=float)

    print(f"\nSaved to {a.out}/")
    print("  U_train8.npy, mean_8.npy, proj_V.npy, imgs_test8.npy, meta.json")
    print("\nStep 1 complete!")


if __name__ == "__main__":
    main()
