"""
step11_table6_median_rbf_mmd.py
===============================
"MMD (image)" and "MMD (ArcFace)" rows of the CelebA-HQ posterior moment
validation table, with the same single-bandwidth median-heuristic RBF MMD^2 as
step10_median_rbf_mmd.py.

Part A (ArcFace) reads the embeddings cached by step8_table6_arcface.py and the
held-out reference embeddings cached by step7_arcface_metrics.py. Part B
(Inception) redecodes the cached top-190 samples written by
step6_additional_results.py, with the same seeds, so it needs the TensorRT
decoder engine.

Usage
-----
python step11_table6_median_rbf_mmd.py --skip_image   # ArcFace rows only
python step11_table6_median_rbf_mmd.py                # ArcFace + image rows (GPU)
Output: results/table6_median_rbf_mmd.json
"""
import argparse
import json
import os

import numpy as np

import config as cfg                                        # noqa: E402
from common import (                                         # noqa: E402
    decode_batched, from_subspace_gaussian_tail, rbf_median_bandwidth,
    compute_kid_rbfmix,
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_image", action="store_true",
                    help="ArcFace rows only (free, cache-only, no GPU/decoder needed). "
                         "Rerun WITHOUT this flag later to add the image rows -- results "
                         "are merged into the same output file, not overwritten.")
    a = ap.parse_args()

    print("=" * 60)
    print("Step 11: median-RBF MMD rows for Table 6")
    print("=" * 60)

    cache_dir = os.path.join(a.results_dir, "fig_cache")
    out_dir = os.path.join(a.results_dir, "arcface")

    out_path = os.path.join(a.results_dir, "table6_median_rbf_mmd.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    # ── Part A: ArcFace median-RBF MMD -- cache-only, no GPU needed ─────────
    print("\n" + "=" * 40 + "\nPart A: ArcFace median-RBF MMD (cache-only)\n" + "=" * 40)
    real_cache = os.path.join(out_dir, "real_valid_arcface.npy")
    if not os.path.exists(real_cache):
        raise FileNotFoundError(
            f"{real_cache} missing -- run step7_arcface_metrics.py first "
            "(it builds this cache).")
    E_real_arc = np.load(real_cache)
    bw_arc = rbf_median_bandwidth(E_real_arc)
    print(f"ArcFace median-RBF bandwidth = {bw_arc:.4f} (from {len(E_real_arc)} real refs)")

    for name, label in SETTINGS:
        emb_cache = os.path.join(out_dir, f"table6_{name}_arcface_{T6_TAG}.npy")
        if not os.path.exists(emb_cache):
            print(f"[MISSING] {label}: {emb_cache} -- run step8 first, skipping")
            continue
        E_gen = np.load(emb_cache)
        # rng=seed*1000+0, matching step7._score's convention for a single
        # (r=0) draw -- same convention step8_table6_arcface.py's
        # own poly3 KID computation uses for this exact setting.
        m, _ = compute_kid_rbfmix(E_real_arc, E_gen, bw_arc, scales=[1.0],
                                  rng=np.random.default_rng(a.seed * 1000))
        d = results.setdefault(name, {"label": label})
        d["arcface_mmd_rbf"] = float(m)
        print(f"  {label:<16} MMD(ArcFace, median-RBF, x1000) = {1000 * m:.4f}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    if a.skip_image:
        print("\n--skip_image set -- stopping here. Rerun without --skip_image "
              "later to add the image-space rows (results will be merged).")
        return

    # ── Part B: Image(Inception) median-RBF MMD -- needs one redecode pass ──
    print("\n" + "=" * 40 + "\nPart B: Image (Inception) median-RBF MMD "
         "(redecode of cached top-190 samples)\n" + "=" * 40)

    missing = [name for name, _ in SETTINGS
              if not os.path.exists(os.path.join(cache_dir, f"table6_{name}_{T6_TAG}_metric.npy"))]
    if missing:
        raise FileNotFoundError(
            f"Missing cached Table 6 metric samples for {missing} under {cache_dir} "
            f"(expected table6_<name>_{T6_TAG}_metric.npy). Run "
            "step6_additional_results.py first (without --regenerate).")

    from metrics_celeba import build_inception_extractor, extract_inception_features

    meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    bw_img = meta["rbf_bw_img"]
    print(f"Image (Inception) median-RBF bandwidth = {bw_img:.4f} (reused from "
          f"results/subspace/meta.json, same value step10/step1 use)")

    f_ref = np.load(cfg.DIFFAE_FEATURES_NPZ)["f_ref"]

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

    for name, label in SETTINGS:
        print(f"\n{'='*40}\n{label}: decoding its cached Table 6 metric samples ...")
        q_metric = np.load(os.path.join(cache_dir, f"table6_{name}_{T6_TAG}_metric.npy")).astype(np.float32)
        # SAME seed formulas as step6_additional_results.py's Table 6
        # block / step8's own reuse of them.
        Z_metric512 = from_subspace_gaussian_tail(q_metric, mean512, Vt, Vb, lambda_tail,
                                                   seed=a.seed + 2)
        imgs = decode_batched(Z_metric512, mean512, Vt, Vb, lambda_tail, render_fn,
                              is_subspace=False, batch=cfg.DECODE_BATCH, seed=a.seed + 3)
        feats_gen = extract_inception_features(imgs.astype(np.float32) / 255.0,
                                               inception_model, inception_device)
        m, _ = compute_kid_rbfmix(f_ref, feats_gen, bw_img, scales=[1.0],
                                  rng=np.random.default_rng(a.seed * 1000))
        d = results.setdefault(name, {"label": label})
        d["image_mmd_rbf"] = float(m)
        print(f"  {label:<16} MMD(image, median-RBF, x1000) = {1000 * m:.4f}")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)   # save after every setting (resumable)

    print(f"\nSaved: {out_path}")

    print(f"\n{'='*40}\nSummary (ready for the Table 6 LaTeX columns):")
    print(f"{'Setting':<16} {'MMD(image,RBF)':>16} {'MMD(ArcFace,RBF)':>18}")
    for name, label in SETTINGS:
        d = results.get(name, {})
        img = f"{1000*d['image_mmd_rbf']:.4f}" if "image_mmd_rbf" in d else "n/a"
        arc = f"{1000*d['arcface_mmd_rbf']:.4f}" if "arcface_mmd_rbf" in d else "n/a"
        print(f"{label:<16} {img:>16} {arc:>18}")

    print("\nStep 11 complete!")


if __name__ == "__main__":
    main()
