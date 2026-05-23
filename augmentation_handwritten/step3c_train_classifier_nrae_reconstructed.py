"""
step3c_train_classifier_nrae_reconstructed.py
=============================================
Trains a CNN classifier on NRAE-reconstructed training images to verify
reconstruction quality. If the NRAE is lossless, accuracy should be close
to the baseline trained on the original images.
"""

import argparse, os, sys
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nrae_model import (
    NRAEModel,
    make_nrae_state,
    encode_dataset_nrae,
    decode_latents_nrae,
    load_params as nrae_load_params,
    print_nrae_summary,
)
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
    print("Step 3c: Train Classifier on NRAE-Reconstructed Images")
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
    # Load NRAE checkpoint
    # ----------------------------------------------------------------
    ckpt_dir  = args.ckpt_dir
    nrae_ckpt = os.path.join(ckpt_dir, "nrae_best.pkl")
    print(f"\nLoading NRAE checkpoint: {nrae_ckpt}")
    params_np, info = nrae_load_params(nrae_ckpt)
    print(f"  Checkpoint info: {info}")

    # Read hyperparams from checkpoint info (fall back to defaults)
    latent_dim   = info.get("latent_dim",   100)
    enc1_hidden  = info.get("enc1_hidden",  2048)
    dec_hidden   = info.get("dec_hidden",   2048)
    n_dct        = info.get("n_dct",        20)
    unet_base_ch = info.get("unet_base_ch", 32)

    # Rebuild NRAE model
    rng = jax.random.PRNGKey(args.seed)
    model, _ = make_nrae_state(
        rng,
        latent_dim=latent_dim,
        enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden,
        n_dct=n_dct,
        unet_base_ch=unet_base_ch,
    )
    params = jax.tree_util.tree_map(jnp.asarray, params_np)

    print_nrae_summary(
        params,
        latent_dim=latent_dim,
        enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden,
        n_dct=n_dct,
        unet_base_ch=unet_base_ch,
    )

    # ----------------------------------------------------------------
    # Encode then decode all training images (reconstruction)
    # ----------------------------------------------------------------
    print(f"Encoding {N_train} training images to latent space ...")
    z_train = encode_dataset_nrae(
        model, params, train_images,
        n_dct=n_dct, batch_size=args.batch_size,
    )
    print(f"  Latent shape: {z_train.shape}")

    print("Decoding latents back to pixel space ...")
    recon_images = decode_latents_nrae(
        model, params, z_train, batch_size=args.batch_size
    )
    print(f"  Reconstructed shape: {recon_images.shape}")

    # Pixel-level reconstruction stats
    diff = np.abs(recon_images - train_images)
    print(f"  Reconstruction error: mean={diff.mean():.4f}  "
          f"max={diff.max():.4f}  std={diff.std():.4f}")

    # Optionally save reconstructed images
    if args.save_recon:
        recon_path = os.path.join(data_dir, "nrae_reconstructed_train_images.npy")
        np.save(recon_path, recon_images.astype(np.float32))
        print(f"  Reconstructed images saved to: {recon_path}")

    # ----------------------------------------------------------------
    # Grid search on reconstructed training images
    # ----------------------------------------------------------------
    wd_grid = [float(x) for x in args.wd_grid.split(",")]
    print(f"\nGrid search over weight_decay: {wd_grid}")
    print(f"max_epochs={args.max_epochs}  batch_size={args.batch_size}  lr={args.lr}")

    rng = jax.random.PRNGKey(args.seed)
    best_val_acc, best_wd, best_epoch, best_params = run_cnn_grid_search(
        rng,
        recon_images, train_labels,
        val_images,   val_labels,
        num_classes=10,
        base_ch=32,
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
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "classifier_nrae_recon_best.pkl")
    save_params(
        ckpt_path, best_params,
        info={
            "best_val_acc":  best_val_acc,
            "best_wd":       best_wd,
            "best_epoch":    best_epoch,
            "max_epochs":    args.max_epochs,
            "batch_size":    args.batch_size,
            "lr":            args.lr,
            "nrae_ckpt":     nrae_ckpt,
            "latent_dim":    latent_dim,
            "n_dct":         n_dct,
        },
    )
    print(f"Best checkpoint saved to: {ckpt_path}")

    # ----------------------------------------------------------------
    # Evaluate on test set
    # ----------------------------------------------------------------
    print(f"\nEvaluating best model on test set ({N_test} images) ...")
    test_acc = evaluate_classifier(
        best_params, test_images, test_labels,
        num_classes=10, base_ch=32,
        tag="NRAE-reconstructed classifier",
    )

    # ----------------------------------------------------------------
    # Compare with baseline if checkpoint exists
    # ----------------------------------------------------------------
    baseline_ckpt = os.path.join(ckpt_dir, "classifier_baseline_best.pkl")
    if os.path.exists(baseline_ckpt):
        print("\nLoading baseline checkpoint for comparison ...")
        base_params, base_info = load_params(baseline_ckpt)
        baseline_acc = evaluate_classifier(
            base_params, test_images, test_labels,
            num_classes=10, base_ch=32,
            tag="baseline classifier",
        )
        delta = test_acc - baseline_acc
        print(f"\n{'='*60}")
        print(f"  Baseline  val acc  : {base_info.get('best_val_acc', 'N/A')}")
        print(f"  NRAE-recon val acc : {best_val_acc:.2f}%  (wd={best_wd:.0e}, epoch={best_epoch})")
        print(f"  Baseline  test acc : {baseline_acc:.2f}%")
        print(f"  NRAE-recon test acc: {test_acc:.2f}%")
        print(f"  Difference         : {delta:+.2f}%")
        print(f"{'='*60}")
        if abs(delta) < 2.0:
            print("  => Reconstruction quality is high (accuracy within 2%).")
        elif delta < 0:
            print("  => Some information lost in NRAE encode-decode cycle.")
    else:
        print(f"\n{'='*60}")
        print(f"  Best val  accuracy : {best_val_acc:.2f}%  (wd={best_wd:.0e}, epoch={best_epoch})")
        print(f"  Test accuracy      : {test_acc:.2f}%")
        print(f"{'='*60}")

    print("\nStep 3c complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3c: Train classifier on NRAE-reconstructed training images."
    )
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--ckpt_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--wd_grid",    type=str,   default="1e-4,1e-3,1e-2,5e-2",
                        help="Comma-separated weight_decay values to search over")
    parser.add_argument("--max_epochs", type=int,   default=80)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--save_recon", action="store_true", default=False,
                        help="Save reconstructed training images to "
                             "nrae_reconstructed_train_images.npy")
    args = parser.parse_args()
    main(args)
