"""
step9_metric_floors.py
=================================
Two "floor" references for all four CelebA-HQ image-quality metrics
(FID (image), KID (image, x1000), FID (ArcFace), KID (ArcFace, x1000))
"""
import argparse
import json
import os

import numpy as np

import config as cfg                                        # noqa: E402
from common import decode_batched                            # noqa: E402
from metrics_celeba import (                                        # noqa: E402
    build_inception_extractor, extract_inception_features,
    compute_fid_from_features, compute_kid,
)
from step7_arcface_metrics import (                          # noqa: E402
    build_arcface_embedder, _dir_names, _load_dir_images,
)


def _pair_metrics(feats_real_img, feats_gen_img, E_real_arc, E_gen_arc, rng):
    """FID/KID(poly3) in both Inception-image-space and ArcFace-identity-
    space, for one generated/real-proxy set against the fixed references."""
    fid_img = compute_fid_from_features(feats_real_img, feats_gen_img)
    kid_img, _ = compute_kid(feats_real_img, feats_gen_img,
                             subset_size=min(cfg.KID_SUBSET, len(feats_gen_img)),
                             n_subsets=cfg.KID_N_SUBSETS, degree=cfg.KID_DEGREE, rng=rng)
    fid_arc = compute_fid_from_features(E_real_arc, E_gen_arc)
    kid_arc, _ = compute_kid(E_real_arc, E_gen_arc,
                             subset_size=min(cfg.KID_SUBSET, len(E_gen_arc)),
                             n_subsets=cfg.KID_N_SUBSETS, degree=cfg.KID_DEGREE, rng=rng)
    return {
        "image_fid": float(fid_img), "image_kid": float(kid_img),
        "arcface_fid": float(fid_arc), "arcface_kid": float(kid_arc),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR_DIFFAE)
    ap.add_argument("--engine_path", default=cfg.DIFFAE_TRT_ENGINE)
    ap.add_argument("--celeba_dir",
                    default="/path/to/datasets/celebahq256",
                    help="raw CelebA-HQ dataset dir with train/ (27,000 PNGs) and "
                         "valid/ (3,000 PNGs) subfolders, named img_<N>.png")
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the random 3000-of-27000 training-image draw "
                         "(default 0 matches the real_train3k convention) "
                         "and for the decoder's tail-completion/diffusion noise")
    a = ap.parse_args()

    if not os.path.isdir(os.path.join(a.celeba_dir, "valid")) or \
       not os.path.isdir(os.path.join(a.celeba_dir, "train")):
        raise FileNotFoundError(
            f"{a.celeba_dir} must contain train/ and valid/ subfolders of raw "
            "CelebA-HQ PNGs (img_<N>.png). Pass the correct path with --celeba_dir.")

    print("=" * 60)
    print("Step 9 (diffae): metric floors (real-train-vs-test, recon-train-vs-test)")
    print("=" * 60)

    out_dir = os.path.join(a.results_dir, "arcface")
    os.makedirs(out_dir, exist_ok=True)

    print("\nBuilding ArcFace recognition model (insightface buffalo_l/w600k_r50) ...")
    embed = build_arcface_embedder()

    print("\nBuilding Inception-V3 feature extractor ...")
    inception_model, inception_device = build_inception_extractor()

    print("\nBuilding image decoder (TensorRT DPM-Solver++(2M)) ...")
    from dpm_decode import make_renderer
    render_fn, _ctx = make_renderer(engine_path=a.engine_path, nsteps=cfg.DECODE_NSTEPS,
                                    batch=cfg.DECODE_BATCH)

    mean512     = np.load(os.path.join(a.subspace_dir, "mean512.npy"))
    Vt          = np.load(os.path.join(a.subspace_dir, "Vt.npy"))
    Vb          = np.load(os.path.join(a.subspace_dir, "Vb.npy"))
    lambda_tail = np.load(os.path.join(a.subspace_dir, "lambda_tail.npy"))

    # ── Fixed references (SAME ones every other FID/KID number in this
    #    pipeline is computed against) ──────────────────────────────────────
    f_ref = np.load(cfg.DIFFAE_FEATURES_NPZ)["f_ref"]
    print(f"\nf_ref (test-image Inception features): {f_ref.shape}")

    real_cache = os.path.join(out_dir, "real_valid_arcface.npy")
    if os.path.exists(real_cache):
        E_real = np.load(real_cache)
        print(f"Reusing cached REAL reference ArcFace embeddings: {E_real.shape}")
    else:
        print("No cached real reference found -- embedding 3000 REAL held-out "
              "valid faces from --celeba_dir ...")
        valid_paths = _dir_names(a.celeba_dir, "valid")
        E_real = embed(_load_dir_images(valid_paths[:3000]))
        np.save(real_cache, E_real)
    print(f"E_real (test-image ArcFace embeddings): {E_real.shape}")

    # ── Same 3,000 random training identities for BOTH floors -- isolates
    #    the DiffAE decoder's own quality loss between floor A and floor B. ──
    train_paths = _dir_names(a.celeba_dir, "train")
    if len(train_paths) < a.n_samples:
        raise ValueError(f"{a.celeba_dir}/train has only {len(train_paths)} images, "
                         f"need >= {a.n_samples}")
    Z_train_full = np.load(cfg.DIFFAE_ZSEM_TRAIN)
    if len(Z_train_full) != len(train_paths):
        raise ValueError(f"{cfg.DIFFAE_ZSEM_TRAIN} has {len(Z_train_full)} rows but "
                         f"{a.celeba_dir}/train has {len(train_paths)} PNGs -- these "
                         "must be index-aligned (row i <-> train_paths[i]) for floor B "
                         "to compare the SAME images pre/post DiffAE round-trip.")
    idx = np.random.default_rng(a.seed).choice(len(train_paths), a.n_samples, replace=False)
    print(f"\nDrew {a.n_samples} random training indices (seed={a.seed}, "
          f"matches the real_train3k convention)")

    print("Loading raw pixels for the drawn training images ...")
    raw_train = _load_dir_images([train_paths[i] for i in idx])
    Z_train_sub = Z_train_full[idx].astype(np.float32)

    rng = np.random.default_rng(a.seed)

    # ── Floor A: real train pixels, no DiffAE at all ────────────────────────
    print("\n" + "=" * 40)
    print("Floor A: real train images vs real test images (no DiffAE)")
    feats_real_train = extract_inception_features(raw_train.astype(np.float32) / 255.0,
                                                   inception_model, inception_device)
    E_arc_real_train = embed(raw_train)
    floor_a = _pair_metrics(f_ref, feats_real_train, E_real, E_arc_real_train, rng)
    print(f"  FID(image)={floor_a['image_fid']:.4f}  "
          f"KID(image,x1000)={1000 * floor_a['image_kid']:.4f}  "
          f"FID(ArcFace)={floor_a['arcface_fid']:.4f}  "
          f"KID(ArcFace,x1000)={1000 * floor_a['arcface_kid']:.4f}")

    # ── Floor B: same train images, round-tripped through the DiffAE decoder ─
    print("\n" + "=" * 40)
    print("Floor B: DiffAE-reconstructed train images vs real test images")
    imgs_recon = decode_batched(Z_train_sub, mean512, Vt, Vb, lambda_tail, render_fn,
                                is_subspace=False, batch=cfg.DECODE_BATCH, seed=a.seed)
    feats_recon_train = extract_inception_features(imgs_recon.astype(np.float32) / 255.0,
                                                    inception_model, inception_device)
    E_arc_recon_train = embed(imgs_recon)
    floor_b = _pair_metrics(f_ref, feats_recon_train, E_real, E_arc_recon_train, rng)
    print(f"  FID(image)={floor_b['image_fid']:.4f}  "
          f"KID(image,x1000)={1000 * floor_b['image_kid']:.4f}  "
          f"FID(ArcFace)={floor_b['arcface_fid']:.4f}  "
          f"KID(ArcFace,x1000)={1000 * floor_b['arcface_kid']:.4f}")

    out = {
        "n_samples": a.n_samples, "seed": a.seed,
        "real_train_vs_test": floor_a,
        "recon_train_vs_test": floor_b,
    }
    out_path = os.path.join(a.results_dir, "metric_floors.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*40}\nSummary (ready for the paper's floor rows):")
    print(f"{'Floor':<24} {'FID(img)':>10} {'KID(img,x1e3)':>14} "
          f"{'FID(ArcFace)':>13} {'KID(ArcFace,x1e3)':>18}")
    for name, d in [("Real train vs test", floor_a), ("Recon train vs test", floor_b)]:
        print(f"{name:<24} {d['image_fid']:>10.4f} {1000 * d['image_kid']:>14.4f} "
              f"{d['arcface_fid']:>13.4f} {1000 * d['arcface_kid']:>18.4f}")

    print("\nStep 9 (diffae) complete!")


if __name__ == "__main__":
    main()
