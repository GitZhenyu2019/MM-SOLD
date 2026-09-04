"""
step2_train_classifier_baseline.py
===================================
Train a CNN classifier on the ORIGINAL training set using grid search
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
    print("Step 2: Train Baseline Classifier (grid search over weight_decay)")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    data_dir = args.data_dir
    train_images = np.load(os.path.join(data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    val_images   = np.load(os.path.join(data_dir, "val_images.npy"))
    val_labels   = np.load(os.path.join(data_dir, "val_labels.npy"))
    test_images  = np.load(os.path.join(data_dir, "test_images.npy"))
    test_labels  = np.load(os.path.join(data_dir, "test_labels.npy"))

    N_train = train_images.shape[0]
    N_val   = val_images.shape[0]
    N_test  = test_images.shape[0]
    print(f"\nTrain: {N_train}  |  Val: {N_val}  |  Test: {N_test} images")
    print(f"Image shape: {train_images.shape[1:]}")

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
        train_images, train_labels,
        val_images,   val_labels,
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
    ckpt_path = os.path.join(args.ckpt_dir, "classifier_baseline_best.pkl")
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
        tag="baseline classifier",
    )

    print(f"\n{'='*60}")
    print(f"  Best val  accuracy : {best_val_acc:.2f}%  (wd={best_wd:.0e}, epoch={best_epoch})")
    print(f"  Test accuracy      : {test_acc:.2f}%")
    print(f"{'='*60}")
    print("\nStep 2 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Train baseline CNN classifier with grid search."
    )
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--ckpt_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--base_ch",    type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--wd_grid",    type=str,   default="1e-4,1e-3,1e-2,5e-2",
                        help="Comma-separated weight_decay values to search over")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--max_epochs", type=int,   default=100)
    parser.add_argument("--seed",       type=int,   default=0)
    args = parser.parse_args()
    main(args)
