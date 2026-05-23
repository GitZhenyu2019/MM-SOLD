"""
step8_latent_augmented.py
=========================
Train MLP classifier on AUGMENTED VAE latent vectors
(original Z_train + manifold-LDS-generated latents).
Grid search over weight_decay; model selected by best validation accuracy.
Compares final test accuracy with the baseline from Step 7.
"""

import argparse, os, sys
import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_model        import run_grid_search, evaluate_mlp_verbose
from classifier_model import save_params, load_params


def main(args):
    print("=" * 60)
    print("Step 8: Latent-Space Augmented Classifier")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # ---- Load latents (augmented training set, shared val/test) ----
    d = args.latent_dir
    Z_train = np.load(os.path.join(d, "Z_aug_train.npy"))
    y_train = np.load(os.path.join(d, "y_aug_train.npy"))
    Z_val   = np.load(os.path.join(d, "Z_val.npy"))
    y_val   = np.load(os.path.join(d, "y_val.npy"))
    Z_test  = np.load(os.path.join(d, "Z_test.npy"))
    y_test  = np.load(os.path.join(d, "y_test.npy"))
    latent_dim = Z_train.shape[1]

    # Sizes for information
    Z_orig = np.load(os.path.join(d, "Z_train.npy"))
    n_orig = Z_orig.shape[0]
    n_gen  = Z_train.shape[0] - n_orig
    print(f"\nAug-Train: {Z_train.shape}  "
          f"({n_orig} orig + {n_gen} generated)")
    print(f"Val: {Z_val.shape}  Test: {Z_test.shape}")

    # ---- Grid search ----
    wd_grid = [float(x) for x in args.wd_grid.split(",")]
    print(f"\nGrid search  weight_decay in {wd_grid}  max_epochs={args.max_epochs}")

    best_val, best_test, best_wd, best_epoch, best_params = run_grid_search(
        Z_train, y_train, Z_val, y_val, Z_test, y_test,
        latent_dim=latent_dim,
        wd_grid=wd_grid,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
    )

    # ---- Final report ----
    print(f"\nBest: weight_decay={best_wd:.0e}  "
          f"best_val_epoch={best_epoch}  val_acc={best_val:.2f}%")
    aug_acc = evaluate_mlp_verbose(
        best_params, Z_test, y_test,
        hidden_dim=args.hidden_dim,
        tag="latent augmented")

    # Save checkpoint
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(args.ckpt_dir, "mlp_augmented_best.pkl")
    save_params(ckpt, best_params,
                info={"best_val_acc": best_val, "best_test_acc": aug_acc,
                      "best_wd": best_wd, "best_epoch": best_epoch})
    print(f"\nCheckpoint saved: {ckpt}")

    # ---- Compare with baseline (Step 7) ----
    baseline_ckpt = os.path.join(args.ckpt_dir, "mlp_baseline_best.pkl")
    if os.path.exists(baseline_ckpt):
        _, base_info = load_params(baseline_ckpt)
        base_acc = base_info.get("best_test_acc")
        if base_acc is not None:
            delta = aug_acc - base_acc
            print(f"\n{'='*60}")
            print(f"  Baseline accuracy  : {base_acc:.2f}%")
            print(f"  Augmented accuracy : {aug_acc:.2f}%")
            print(f"  Improvement        : {delta:+.2f}%")
            print(f"{'='*60}")

    print("\nStep 8 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 8: Latent-space augmented MLP with grid search.")
    parser.add_argument("--latent_dir",  type=str,   default="./latents")
    parser.add_argument("--ckpt_dir",    type=str,   default="./checkpoints")
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max_epochs",  type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--wd_grid",     type=str,   default="1e-3,1e-2,5e-2,1e-1",
                        help="Comma-separated weight_decay values")
    parser.add_argument("--seed",        type=int,   default=0)
    args = parser.parse_args()
    main(args)
