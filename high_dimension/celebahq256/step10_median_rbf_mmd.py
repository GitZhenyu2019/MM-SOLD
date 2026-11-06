"""
step10_median_rbf_mmd.py
========================
"MMD (image)" and "MMD (ArcFace)" rows of the CelebA-HQ generation results
table: the unbiased MMD^2 U-statistic (reported x10^3, no square root, no
clipping) with a single RBF kernel exp(-||x-y||^2 / (2 tau^2)), averaged over
100 random subsets of 1,000 real and 1,000 generated features per run.

The bandwidth tau is fixed once by the median heuristic on the real reference
features: for Inception features it is `rbf_bw_img` in results/subspace/meta.json
(computed by step1 on the held-out reference features); for ArcFace it is
computed here on the cached held-out ArcFace embeddings. Every method and run
uses the same kernel.

Part A (ArcFace) only reads the embeddings cached by step7_arcface_metrics.py.
Part B (Inception) regenerates and redecodes the three runs of each method at
the configuration selected in step4, with the same seeds as step4/step7, so it
needs the TensorRT decoder engine.

Usage
-----
python step10_median_rbf_mmd.py --skip_image   # ArcFace rows only (CPU, seconds)
python step10_median_rbf_mmd.py                # ArcFace + image rows (GPU)
Output: results/table3_median_rbf_mmd.json
"""
import argparse
import json
import os

import numpy as np

import config as cfg                                        # noqa: E402
from common import (                                         # noqa: E402
    decode_batched, gamma_baoab, rbf_median_bandwidth, compute_kid_rbfmix,
)
from step7_arcface_metrics import METHODS, _cell_hash        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR_DIFFAE)
    ap.add_argument("--engine_path", default=cfg.DIFFAE_TRT_ENGINE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_image", action="store_true",
                    help="ArcFace row only (free, cache-only, no GPU/decoder needed). "
                         "Rerun WITHOUT this flag later to add the image row -- results "
                         "are merged into the same output file, not overwritten.")
    a = ap.parse_args()

    print("=" * 60)
    print("Step 10: median-RBF MMD rows for Table 3")
    print("=" * 60)

    out_dir = os.path.join(a.results_dir, "arcface")
    table3 = json.load(open(os.path.join(a.results_dir, "table3.json")))["table"]

    out_path = os.path.join(a.results_dir, "table3_median_rbf_mmd.json")
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

    for method_key, label, _ in METHODS:
        cell = table3[method_key]["decoded_cell"]
        emb_cache = os.path.join(out_dir, f"{method_key}_arcface_{_cell_hash(cell)}.npy")
        if not os.path.exists(emb_cache):
            print(f"[MISSING] {label}: {emb_cache} -- run step7 first, skipping")
            continue
        E_gen_repeats = np.load(emb_cache)
        # Per-repeat rng, matching common.decode_and_score_images's own
        # convention for compute_kid_rbfmix (NOT the function's internal
        # default seed=0, which would reuse the identical subset-selection
        # draw across every repeat/method here).
        vals = [compute_kid_rbfmix(E_real_arc, E_gen_repeats[r], bw_arc, scales=[1.0],
                                   rng=np.random.default_rng(a.seed * 1000 + r))[0]
               for r in range(len(E_gen_repeats))]
        d = results.setdefault(method_key, {"label": label})
        d["arcface_mmd_rbf_mean"] = float(np.mean(vals))
        d["arcface_mmd_rbf_std"] = float(np.std(vals))
        print(f"  {label:<16} MMD(ArcFace, median-RBF, x1000) = "
              f"{1000 * np.mean(vals):.4f} +/- {1000 * np.std(vals):.4f}")

    ddpm_cache = os.path.join(out_dir, "latent_ddpm_arcface.npy")
    if os.path.exists(ddpm_cache):
        E_gen_repeats = np.load(ddpm_cache)
        vals = [compute_kid_rbfmix(E_real_arc, E_gen_repeats[r], bw_arc, scales=[1.0],
                                   rng=np.random.default_rng(a.seed * 1000 + r))[0]
               for r in range(len(E_gen_repeats))]
        d = results.setdefault("latent_ddpm", {"label": "Latent DDPM"})
        d["arcface_mmd_rbf_mean"] = float(np.mean(vals))
        d["arcface_mmd_rbf_std"] = float(np.std(vals))
        print(f"  {'Latent DDPM':<16} MMD(ArcFace, median-RBF, x1000) = "
              f"{1000 * np.mean(vals):.4f} +/- {1000 * np.std(vals):.4f}")
    else:
        print(f"[MISSING] Latent DDPM: {ddpm_cache} -- run step7 first, skipping")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    if a.skip_image:
        print("\n--skip_image set -- stopping here. Rerun without --skip_image "
              "later to add the image-space row (results will be merged).")
        return

    # ── Part B: Image(Inception) median-RBF MMD -- needs regenerate+redecode ─
    print("\n" + "=" * 40 + "\nPart B: Image (Inception) median-RBF MMD "
         "(regenerate + redecode required)\n" + "=" * 40)

    from metrics_celeba import build_inception_extractor, extract_inception_features
    from step4_select_and_decode import _make_gen_fn

    meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    bw_img = meta["rbf_bw_img"]
    print(f"Image (Inception) median-RBF bandwidth = {bw_img:.4f} (reused from "
          f"results/subspace/meta.json, computed once by step1 on f_ref)")

    f_ref = np.load(cfg.DIFFAE_FEATURES_NPZ)["f_ref"]

    print("\nBuilding Inception-V3 feature extractor ...")
    inception_model, inception_device = build_inception_extractor()

    print("\nBuilding image decoder (TensorRT DPM-Solver++(2M)) ...")
    from dpm_decode import make_renderer
    render_fn, _ctx = make_renderer(engine_path=a.engine_path, nsteps=cfg.DECODE_NSTEPS,
                                    batch=cfg.DECODE_BATCH)

    U_train190 = np.load(os.path.join(a.subspace_dir, "U_train190.npy"))
    mean512     = np.load(os.path.join(a.subspace_dir, "mean512.npy"))
    Vt          = np.load(os.path.join(a.subspace_dir, "Vt.npy"))
    Vb          = np.load(os.path.join(a.subspace_dir, "Vb.npy"))
    lambda_tail = np.load(os.path.join(a.subspace_dir, "lambda_tail.npy"))
    gamma190    = gamma_baoab(U_train190)

    for method_key, label, _ in METHODS:
        cell = table3[method_key]["decoded_cell"]
        print(f"\n{'='*40}\n{label}: regenerating at its Table 3 decoded_cell ...")
        cache_dir = (os.path.join(a.results_dir, "sgbaoab", "tilting_cache")
                    if method_key == "sg_baoab" else None)
        gen_fn = _make_gen_fn(method_key, cell, U_train190, gamma190, cache_dir=cache_dir)

        vals = []
        for r in range(cfg.N_REPEATS):
            U_gen190 = np.asarray(gen_fn(a.seed + r), dtype=np.float32)
            # SAME seed formula as common.decode_and_score_images /
            # step7_arcface_metrics.py -- bit-identical pixels to
            # every other image-space number already reported for this method.
            imgs = decode_batched(U_gen190, mean512, Vt, Vb, lambda_tail, render_fn,
                                  is_subspace=True, batch=cfg.DECODE_BATCH,
                                  seed=a.seed * 1000 + r + 700000)
            feats_gen = extract_inception_features(imgs.astype(np.float32) / 255.0,
                                                    inception_model, inception_device)
            m, _ = compute_kid_rbfmix(f_ref, feats_gen, bw_img, scales=[1.0],
                                      rng=np.random.default_rng(a.seed * 1000 + r))
            vals.append(m)
            print(f"    repeat {r + 1}/{cfg.N_REPEATS}: decoded + scored {len(imgs)} images")
        d = results.setdefault(method_key, {"label": label})
        d["image_mmd_rbf_mean"] = float(np.mean(vals))
        d["image_mmd_rbf_std"] = float(np.std(vals))
        print(f"  {label:<16} MMD(image, median-RBF, x1000) = "
              f"{1000 * np.mean(vals):.4f} +/- {1000 * np.std(vals):.4f}")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)   # save after every method (resumable)

    # ── Latent DDPM: reuse its ALREADY-SAVED full-512-dim repeat*.npy files ──
    ddpm_gen_dir = os.path.join(a.results_dir, "ddpm", "generated")
    if os.path.isdir(ddpm_gen_dir):
        print(f"\n{'='*40}\nLatent DDPM: decoding its own saved generated repeats ...")
        repeat_files = sorted(f for f in os.listdir(ddpm_gen_dir)
                              if f.startswith("repeat") and f.endswith(".npy")
                              and "_smoke" not in f)
        vals = []
        for r, fname in enumerate(repeat_files[:cfg.N_REPEATS]):
            Z_gen512 = np.load(os.path.join(ddpm_gen_dir, fname))
            # SAME seed formula as step4's own inline DDPM decode / step7's
            # Latent DDPM path (no +700000 offset -- different convention
            # from the 6 grid methods, matching each exactly is what matters).
            imgs = decode_batched(Z_gen512, mean512, Vt, Vb, lambda_tail, render_fn,
                                  is_subspace=False, batch=cfg.DECODE_BATCH,
                                  seed=a.seed * 1000 + r)
            feats_gen = extract_inception_features(imgs.astype(np.float32) / 255.0,
                                                    inception_model, inception_device)
            m, _ = compute_kid_rbfmix(f_ref, feats_gen, bw_img, scales=[1.0],
                                      rng=np.random.default_rng(a.seed * 1000 + r))
            vals.append(m)
            print(f"    repeat {r + 1}/{len(repeat_files[:cfg.N_REPEATS])}: "
                  f"decoded + scored {len(imgs)} images")
        d = results.setdefault("latent_ddpm", {"label": "Latent DDPM"})
        d["image_mmd_rbf_mean"] = float(np.mean(vals))
        d["image_mmd_rbf_std"] = float(np.std(vals))
        print(f"  {'Latent DDPM':<16} MMD(image, median-RBF, x1000) = "
              f"{1000 * np.mean(vals):.4f} +/- {1000 * np.std(vals):.4f}")
    else:
        print(f"\n[MISSING] Latent DDPM: {ddpm_gen_dir} -- run step3 first")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*40}\nSummary (ready for the Table 3 LaTeX rows):")
    print(f"{'Method':<16} {'MMD(image,RBF)':>16} {'MMD(ArcFace,RBF)':>18}")
    for method_key, label, _ in METHODS + [("latent_ddpm", "Latent DDPM", None)]:
        if method_key not in results:
            continue
        d = results[method_key]
        img = (f"{1000*d['image_mmd_rbf_mean']:.4f}+/-{1000*d['image_mmd_rbf_std']:.4f}"
              if "image_mmd_rbf_mean" in d else "n/a")
        arc = (f"{1000*d['arcface_mmd_rbf_mean']:.4f}+/-{1000*d['arcface_mmd_rbf_std']:.4f}"
              if "arcface_mmd_rbf_mean" in d else "n/a")
        print(f"{label:<16} {img:>16} {arc:>18}")

    print("\nStep 10 complete!")


if __name__ == "__main__":
    main()
