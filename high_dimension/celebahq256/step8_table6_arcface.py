"""
step8_table6_arcface.py
=================================
FID/KID in ArcFace IDENTITY-space for the CelebA-HQ posterior-moment-
validation results
"""
import argparse
import json
import os

import numpy as np

import config as cfg                                        # noqa: E402
from common import from_subspace_gaussian_tail, decode_batched  # noqa: E402
from metrics_celeba import compute_fid_from_features, compute_kid   # noqa: E402
from step7_arcface_metrics import (                          # noqa: E402
    build_arcface_embedder, _dir_names, _load_dir_images, _score,
)

C_DELTA_T6, C_SIGMA_T6 = 0.4, 0.4
T6_TAG = f"cd{C_DELTA_T6:.2f}_cs{C_SIGMA_T6:.2f}"

SETTINGS = [
    ("naive_baoab", "Naive BAOAB"),
    ("plugin_baoab", "Plug-in BAOAB"),
    ("sg_baoab", "SG-BAOAB"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR_DIFFAE)
    ap.add_argument("--engine_path", default=cfg.DIFFAE_TRT_ENGINE)
    ap.add_argument("--celeba_dir",
                    default="/path/to/datasets/celebahq256",
                    help="only used as a fallback if results/arcface/"
                         "real_valid_arcface.npy from step7 doesn't exist yet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("=" * 60)
    print("Step 8 (diffae): ArcFace identity-space FID/KID for CelebA-HQ posterior-moment-validation table")
    print("=" * 60)

    cache_dir = os.path.join(a.results_dir, "fig_cache")
    out_dir = os.path.join(a.results_dir, "arcface")
    os.makedirs(out_dir, exist_ok=True)

    missing = [name for name, _ in SETTINGS
              if not os.path.exists(os.path.join(cache_dir, f"table6_{name}_{T6_TAG}_metric.npy"))]
    if missing:
        raise FileNotFoundError(
            f"Missing cached CelebA-HQ posterior-moment-validation table metric samples for {missing} under {cache_dir} "
            f"(expected table6_<name>_{T6_TAG}_metric.npy). Run "
            "step6_additional_results.py first (without --regenerate) so "
            "CelebA-HQ posterior-moment-validation table's own image-space FID/KID numbers and these cached samples "
            "are produced together.")

    print("\nBuilding ArcFace recognition model (insightface buffalo_l/w600k_r50) ...")
    embed = build_arcface_embedder()

    print("\nBuilding image decoder (TensorRT DPM-Solver++(2M)) ...")
    from dpm_decode import make_renderer
    render_fn, _ctx = make_renderer(engine_path=a.engine_path, nsteps=cfg.DECODE_NSTEPS,
                                    batch=cfg.DECODE_BATCH)

    mean512     = np.load(os.path.join(a.subspace_dir, "mean512.npy"))
    Vt          = np.load(os.path.join(a.subspace_dir, "Vt.npy"))
    Vb          = np.load(os.path.join(a.subspace_dir, "Vb.npy"))
    lambda_tail = np.load(os.path.join(a.subspace_dir, "lambda_tail.npy"))

    # ── REAL reference embeddings: reuse step7's cache if present, else build. ──
    real_cache = os.path.join(out_dir, "real_valid_arcface.npy")
    if os.path.exists(real_cache):
        E_real = np.load(real_cache)
        print(f"\nReusing cached REAL reference ArcFace embeddings: {E_real.shape}")
    else:
        if not os.path.isdir(os.path.join(a.celeba_dir, "valid")):
            raise FileNotFoundError(
                f"No cached real reference at {real_cache} (expected step7 to have "
                f"built it already) and --celeba_dir {a.celeba_dir} has no valid/ "
                "subfolder to build it from. Run step7_arcface_metrics.py "
                "first, or pass the correct --celeba_dir.")
        print("\nNo cached real reference found -- embedding 3000 REAL held-out "
              "valid faces from --celeba_dir ...")
        valid_paths = _dir_names(a.celeba_dir, "valid")
        imgs = _load_dir_images(valid_paths[:3000])
        E_real = embed(imgs)
        np.save(real_cache, E_real)
    print(f"  E_real {E_real.shape}")

    results = {}
    for name, label in SETTINGS:
        print(f"\n{'='*40}\n{label}: decoding its cached CelebA-HQ posterior-moment-validation table metric samples ...")
        q_metric = np.load(os.path.join(cache_dir, f"table6_{name}_{T6_TAG}_metric.npy")).astype(np.float32)

        emb_cache = os.path.join(out_dir, f"table6_{name}_arcface_{T6_TAG}.npy")
        if os.path.exists(emb_cache):
            E_gen = np.load(emb_cache)
            print(f"  reusing cached embeddings {emb_cache} {E_gen.shape}")
        else:
            Z_metric512 = from_subspace_gaussian_tail(q_metric, mean512, Vt, Vb, lambda_tail,
                                                       seed=a.seed + 2)
            imgs = decode_batched(Z_metric512, mean512, Vt, Vb, lambda_tail, render_fn,
                                  is_subspace=False, batch=cfg.DECODE_BATCH, seed=a.seed + 3)
            E_gen = embed(imgs)
            np.save(emb_cache, E_gen)
            print(f"  decoded + embedded {len(imgs)} images")

        res = _score(E_real, E_gen[None, :, :], a.seed)
        results[name] = {"label": label, **res}
        print(f"  FID(ArcFace)={res['fid_mean']:.3f}  "
              f"KID(ArcFace,x1000)={1000 * res['kid_mean']:.3f}")

    out_path = os.path.join(a.results_dir, "table6_arcface.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*40}\nSummary (ready for the CelebA-HQ posterior-moment-validation table LaTeX columns):")
    print(f"{'Setting':<16} {'FID (ArcFace)':>14} {'KID (ArcFace, x1e3)':>22}")
    for name, label in SETTINGS:
        r = results[name]
        print(f"{label:<16} {r['fid_mean']:>14.3f} {1000 * r['kid_mean']:>22.3f}")

    print("\nStep 8 (diffae) complete!")


if __name__ == "__main__":
    main()
