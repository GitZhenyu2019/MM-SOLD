"""
step1_prepare_data.py
=====================
Load the raw handwritten digit dataset, select a balanced subset
(n_train_per_class training + n_test_per_class test per digit),
apply geometric preprocessing (weighted centroid + PCA-align + scale
normalization) and resize to 64x64, then save as .npy files.
"""

import argparse, os, sys
import numpy as np
import jax

# Allow running from any CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    load_images_and_digits,
    to_float01_np,
    select_train_test_split,
    select_train_val_test_split,
    full_preprocess_pipeline,
)


def main(args):
    print("=" * 60)
    print("Step 1: Prepare Data")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ----------------------------------------------------------------
    # Load raw data
    # ----------------------------------------------------------------
    print(f"\nLoading images : {args.images_path}")
    print(f"Loading labels : {args.labels_path}")
    images_raw, digits = load_images_and_digits(args.images_path,
                                                 args.labels_path)
    print(f"Dataset size   : {images_raw.shape[0]} images, "
          f"shape {images_raw.shape[1:]}")

    images_float = to_float01_np(images_raw)

    os.makedirs(args.data_dir, exist_ok=True)

    if args.n_val_per_class > 0:
        # ----------------------------------------------------------------
        # Three-way split: train / val / test
        # ----------------------------------------------------------------
        print(f"\nSelecting {args.n_train_per_class} train + "
              f"{args.n_val_per_class} val + "
              f"{args.n_test_per_class} test per class (seed={args.seed}):")
        (train_raw, train_labels,
         val_raw,   val_labels,
         test_raw,  test_labels) = select_train_val_test_split(
            images_float, digits,
            n_train_per_class=args.n_train_per_class,
            n_val_per_class=args.n_val_per_class,
            n_test_per_class=args.n_test_per_class,
            seed=args.seed,
        )
        print(f"\nTrain set : {train_raw.shape[0]} images")
        print(f"Val   set : {val_raw.shape[0]} images")
        print(f"Test  set : {test_raw.shape[0]} images")

        # ----------------------------------------------------------------
        # Geometric preprocessing
        # Compute target_rms from TRAINING set, apply same to val + test.
        # ----------------------------------------------------------------
        print(f"\nPreprocessing training images "
              f"(dark_quantile={args.dark_quantile}, "
              f"weight_power={args.weight_power}) ...")
        train_64, target_rms = full_preprocess_pipeline(
            train_raw,
            dark_quantile=args.dark_quantile,
            weight_power=args.weight_power,
            batch_size=args.preprocess_batch_size,
            rms_scale=args.rms_scale,
        )
        print(f"  train_64 shape : {train_64.shape}  "
              f"target_rms = {target_rms:.5f}  (rms_scale={args.rms_scale})")

        print("Preprocessing val images (using train target_rms) ...")
        val_64, _ = full_preprocess_pipeline(
            val_raw,
            dark_quantile=args.dark_quantile,
            weight_power=args.weight_power,
            batch_size=args.preprocess_batch_size,
            target_rms=target_rms,
        )
        print(f"  val_64   shape : {val_64.shape}")

        print("Preprocessing test images (using train target_rms) ...")
        test_64, _ = full_preprocess_pipeline(
            test_raw,
            dark_quantile=args.dark_quantile,
            weight_power=args.weight_power,
            batch_size=args.preprocess_batch_size,
            target_rms=target_rms,
        )
        print(f"  test_64  shape : {test_64.shape}")

        # ----------------------------------------------------------------
        # Save to disk
        # ----------------------------------------------------------------
        train_np = np.array(train_64, dtype=np.float32)
        val_np   = np.array(val_64,   dtype=np.float32)
        test_np  = np.array(test_64,  dtype=np.float32)

        np.save(os.path.join(args.data_dir, "train_images.npy"), train_np)
        np.save(os.path.join(args.data_dir, "train_labels.npy"),
                train_labels.astype(np.int32))
        np.save(os.path.join(args.data_dir, "val_images.npy"),   val_np)
        np.save(os.path.join(args.data_dir, "val_labels.npy"),
                val_labels.astype(np.int32))
        np.save(os.path.join(args.data_dir, "test_images.npy"),  test_np)
        np.save(os.path.join(args.data_dir, "test_labels.npy"),
                test_labels.astype(np.int32))
        np.save(os.path.join(args.data_dir, "target_rms.npy"),
                np.float32(target_rms))

        print(f"\nSaved to: {args.data_dir}/")
        print(f"  train_images.npy  : {train_np.shape}  dtype={train_np.dtype}")
        print(f"  train_labels.npy  : {train_labels.shape}")
        print(f"  val_images.npy    : {val_np.shape}")
        print(f"  val_labels.npy    : {val_labels.shape}")
        print(f"  test_images.npy   : {test_np.shape}")
        print(f"  test_labels.npy   : {test_labels.shape}")
        print(f"  target_rms.npy    : {float(target_rms):.5f}")

        # Label distribution check
        print("\nLabel distribution in training set:")
        for d in range(10):
            n = (train_labels == d).sum()
            print(f"  digit {d}: {n}")

    else:
        # ----------------------------------------------------------------
        # Two-way split: train / test (original behaviour)
        # ----------------------------------------------------------------
        print(f"\nSelecting {args.n_train_per_class} train + "
              f"{args.n_test_per_class} test per class (seed={args.seed}):")
        train_raw, train_labels, test_raw, test_labels = select_train_test_split(
            images_float, digits,
            n_train_per_class=args.n_train_per_class,
            n_test_per_class=args.n_test_per_class,
            seed=args.seed,
        )
        print(f"\nTrain set : {train_raw.shape[0]} images")
        print(f"Test  set : {test_raw.shape[0]} images")

        # ----------------------------------------------------------------
        # Geometric preprocessing
        # Compute target_rms from TRAINING set, apply same to TEST set.
        # ----------------------------------------------------------------
        print(f"\nPreprocessing training images "
              f"(dark_quantile={args.dark_quantile}, "
              f"weight_power={args.weight_power}) ...")
        train_64, target_rms = full_preprocess_pipeline(
            train_raw,
            dark_quantile=args.dark_quantile,
            weight_power=args.weight_power,
            batch_size=args.preprocess_batch_size,
            rms_scale=args.rms_scale,
        )
        print(f"  train_64 shape : {train_64.shape}  "
              f"target_rms = {target_rms:.5f}  (rms_scale={args.rms_scale})")

        print("Preprocessing test images (using train target_rms) ...")
        test_64, _ = full_preprocess_pipeline(
            test_raw,
            dark_quantile=args.dark_quantile,
            weight_power=args.weight_power,
            batch_size=args.preprocess_batch_size,
            target_rms=target_rms,
        )
        print(f"  test_64  shape : {test_64.shape}")

        # ----------------------------------------------------------------
        # Save to disk
        # ----------------------------------------------------------------
        train_np = np.array(train_64, dtype=np.float32)
        test_np  = np.array(test_64,  dtype=np.float32)

        np.save(os.path.join(args.data_dir, "train_images.npy"), train_np)
        np.save(os.path.join(args.data_dir, "train_labels.npy"),
                train_labels.astype(np.int32))
        np.save(os.path.join(args.data_dir, "test_images.npy"),  test_np)
        np.save(os.path.join(args.data_dir, "test_labels.npy"),
                test_labels.astype(np.int32))
        np.save(os.path.join(args.data_dir, "target_rms.npy"),
                np.float32(target_rms))

        print(f"\nSaved to: {args.data_dir}/")
        print(f"  train_images.npy  : {train_np.shape}  dtype={train_np.dtype}")
        print(f"  train_labels.npy  : {train_labels.shape}")
        print(f"  test_images.npy   : {test_np.shape}")
        print(f"  test_labels.npy   : {test_labels.shape}")
        print(f"  target_rms.npy    : {float(target_rms):.5f}")

        # Label distribution check
        print("\nLabel distribution in training set:")
        for d in range(10):
            n = (train_labels == d).sum()
            print(f"  digit {d}: {n}")

    print("\nStep 1 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Preprocess and split the dataset."
    )
    parser.add_argument("--images_path", type=str, required=True,
                        help="Path to handwritten_img.npy")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Path to handwritten_label.npy")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Output directory for processed data")
    parser.add_argument("--n_train_per_class", type=int, default=200)
    parser.add_argument("--n_val_per_class",   type=int, default=0,
                        help="When >0, perform a 3-way train/val/test split "
                             "and save val_images.npy / val_labels.npy")
    parser.add_argument("--n_test_per_class",  type=int, default=50)
    parser.add_argument("--dark_quantile",     type=float, default=0.1)
    parser.add_argument("--weight_power",      type=float, default=1.0)
    parser.add_argument("--preprocess_batch_size", type=int, default=8)
    parser.add_argument("--rms_scale", type=float, default=0.82,
                        help="Scale factor for target_rms (< 1 leaves border margin). "
                             "Default 0.82.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
