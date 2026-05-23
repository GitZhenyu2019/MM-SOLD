"""
step5_train_classifier_augmented.py
=====================================
Train the CNN classifier on the AUGMENTED training set using grid search
over weight_decay values, with model selection based on validation accuracy.
"""

import argparse, os, sys
import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier_model import (
    CNNClassifier,
    make_classifier_state,
    clf_train_step,
    evaluate_classifier,
    eval_accuracy_np,
    run_cnn_grid_search,
    save_params,
    load_params,
    count_params,
    print_classifier_summary,
)


def main(args):
    print("=" * 60)
    print("Step 5: Train Classifier on Augmented Dataset (grid search)")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"Augmented data prefix: {args.aug_prefix}")

    # ----------------------------------------------------------------
    # Load augmented training data, val data, and test data
    # ----------------------------------------------------------------
    aug_images = np.load(
        os.path.join(args.data_dir, f"{args.aug_prefix}_train_images.npy")
    )
    aug_labels = np.load(
        os.path.join(args.data_dir, f"{args.aug_prefix}_train_labels.npy")
    )
    val_images  = np.load(os.path.join(args.data_dir, "val_images.npy"))
    val_labels  = np.load(os.path.join(args.data_dir, "val_labels.npy"))
    test_images = np.load(os.path.join(args.data_dir, "test_images.npy"))
    test_labels = np.load(os.path.join(args.data_dir, "test_labels.npy"))

    N_train = aug_images.shape[0]
    N_val   = val_images.shape[0]
    N_test  = test_images.shape[0]
    print(f"\nAugmented train: {N_train}  |  Val: {N_val}  |  Test: {N_test} images")

    orig_images = np.load(os.path.join(args.data_dir, "train_images.npy"))
    N_orig = orig_images.shape[0]
    N_gen  = N_train - N_orig
    print(f"  Original: {N_orig}  +  Generated: {N_gen}  "
          f"= {N_train}  ({N_train/N_orig:.1f}x augmentation)")

    print("\nLabel distribution in augmented set:")
    for d in range(10):
        n = (aug_labels == d).sum()
        print(f"  digit {d}: {n}")

    # ----------------------------------------------------------------
    # Parse wd_grid
    # ----------------------------------------------------------------
    wd_grid = [float(x) for x in args.wd_grid.split(",")]
    print(f"\nGrid search over weight_decay: {wd_grid}")
    print(f"max_epochs={args.max_epochs}  batch_size={args.batch_size}  lr={args.lr}")

    # ----------------------------------------------------------------
    # Run grid search
    # ----------------------------------------------------------------
    rng = jax.random.PRNGKey(args.seed)
    best_val_acc, best_wd, best_epoch, best_params = run_cnn_grid_search(
        rng,
        aug_images, aug_labels,
        val_images, val_labels,
        num_classes=10,
        base_ch=args.base_ch,
        lr=args.lr,
        wd_grid=wd_grid,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"\nGrid search complete.")
    print(f"  Best weight_decay : {best_wd:.0e}")
    print(f"  Best val accuracy : {best_val_acc:.2f}%  (epoch {best_epoch})")

    # ----------------------------------------------------------------
    # Save best checkpoint
    # ----------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"classifier_{args.aug_prefix}_best.pkl")
    save_params(
        ckpt_path, best_params,
        info={
            "best_val_acc": best_val_acc,
            "best_wd":      best_wd,
            "best_epoch":   best_epoch,
            "max_epochs":   args.max_epochs,
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "base_ch":      args.base_ch,
            "aug_prefix":   args.aug_prefix,
        },
    )
    print(f"Best checkpoint saved to: {ckpt_path}")

    # ----------------------------------------------------------------
    # Evaluate on test set using best params
    # ----------------------------------------------------------------
    print(f"\nEvaluating best model on test set ({N_test} images) ...")
    test_acc = evaluate_classifier(
        best_params, test_images, test_labels,
        num_classes=10, base_ch=args.base_ch,
        tag=f"{args.aug_prefix} classifier",
    )

    # ----------------------------------------------------------------
    # Also report baseline for comparison (if checkpoint exists)
    # ----------------------------------------------------------------
    baseline_ckpt = os.path.join(args.ckpt_dir, "classifier_baseline_best.pkl")
    if os.path.exists(baseline_ckpt):
        print("\nLoading baseline checkpoint for comparison ...")
        base_params, base_info = load_params(baseline_ckpt)
        baseline_acc = evaluate_classifier(
            base_params, test_images, test_labels,
            num_classes=10, base_ch=args.base_ch,
            tag="baseline classifier",
        )
        delta = test_acc - baseline_acc
        print(f"\n{'='*60}")
        print(f"  Baseline  val acc  : {base_info.get('best_val_acc', 'N/A')}")
        print(f"  Augmented val acc  : {best_val_acc:.2f}%  (wd={best_wd:.0e}, epoch={best_epoch})")
        print(f"  Baseline  test acc : {baseline_acc:.2f}%")
        print(f"  Augmented test acc : {test_acc:.2f}%")
        print(f"  Test improvement   : {delta:+.2f}%")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"  Best val  accuracy : {best_val_acc:.2f}%  (wd={best_wd:.0e}, epoch={best_epoch})")
        print(f"  Test accuracy      : {test_acc:.2f}%")
        print(f"{'='*60}")

    print("\nStep 5 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 5: Train classifier on augmented dataset with grid search."
    )
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--ckpt_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--aug_prefix", type=str,   default="augmented",
                        help="Prefix for augmented data files, e.g. 'augmented' or "
                             "'augmented_nrae'. Loads {prefix}_train_images.npy and "
                             "{prefix}_train_labels.npy.")
    parser.add_argument("--base_ch",    type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--wd_grid",    type=str,   default="1e-4,1e-3,1e-2,5e-2",
                        help="Comma-separated weight_decay values to search over")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--max_epochs", type=int,   default=80)
    parser.add_argument("--seed",       type=int,   default=0)
    args = parser.parse_args()
    main(args)
