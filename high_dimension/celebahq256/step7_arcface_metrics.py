"""
step7_arcface_metrics.py
=================================
FID/KID in ArcFace IDENTITY-space
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np

import config as cfg                                     # noqa: E402
from common import decode_batched, gamma_baoab             # noqa: E402
from metrics_celeba import compute_fid_from_features, compute_kid  # noqa: E402
from step4_select_and_decode import _make_gen_fn            # noqa: E402


METHODS = [
    ("mmsold",       "MM-SOLD",       "mmsold"),
    ("cfdm",         "sigma-CFDM",    "cfdm"),
    ("knn_pflow",    "KNN-PFlow",     "knnpflow"),
    ("swf",          "SWF",           "swf"),
    ("plugin_baoab", "Plug-in BAOAB", "pluginbaoab"),
    ("sg_baoab",     "SG-BAOAB",      "sgbaoab"),
]


def _check_deps():
    try:
        import cv2                                 # noqa: F401
        from insightface.app import FaceAnalysis    # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ArcFace metrics need insightface + opencv-python + a CUDA-12 "
            "onnxruntime-gpu build (the DEFAULT pip index's onnxruntime-gpu "
            "silently falls back to CPU on a CUDA-12 box -- no error, just "
            "~50x slower). Install with:\n"
            "  pip install insightface opencv-python\n"
            "  pip install onnxruntime-gpu --index-url https://pypi.nvidia.com\n"
            f"Original error: {e}") from e


def build_arcface_embedder():
    _check_deps()
    import cv2
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(256, 256))
    rec = app.models["recognition"]
    try:
        providers = rec.session.get_providers()
    except AttributeError:
        providers = "unknown"
    print(f"  ArcFace recognition model: {rec.taskname}  input={rec.input_shape}  "
          f"onnxruntime providers={providers}")
    if providers != "unknown" and "CUDAExecutionProvider" not in providers:
        print("  [WARNING] CUDAExecutionProvider not active -- onnxruntime-gpu "
              "silently fell back to CPU (see this script's docstring for the "
              "CUDA-12 install fix). Will still work, just ~50x slower.")
    IN = rec.input_shape[2]  # 112

    def embed(imgs_uint8_rgb, bs=128):
        """imgs_uint8_rgb: (N, 256, 256, 3) uint8 RGB -> (N, 512) float32,
        L2-normalised identity embeddings."""
        out = np.empty((len(imgs_uint8_rgb), 512), dtype=np.float32)
        for i in range(0, len(imgs_uint8_rgb), bs):
            blk = imgs_uint8_rgb[i:i + bs]
            crops = [cv2.resize(im[28:228, 28:228][:, :, ::-1], (IN, IN),
                                interpolation=cv2.INTER_AREA) for im in blk]
            e = rec.get_feat(crops)
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            out[i:i + len(blk)] = e
        return out
    return embed


def _dir_names(celeba_dir, split):
    split_dir = os.path.join(celeba_dir, split)
    ns = [f for f in os.listdir(split_dir) if f.endswith(".png")]
    return [os.path.join(split_dir, f) for f in
            sorted(ns, key=lambda n: int(re.search(r"\d+", n).group()))]


def _load_dir_images(paths):
    from PIL import Image
    out = np.empty((len(paths), 256, 256, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = np.asarray(Image.open(p).convert("RGB"))
    return out


def _cell_hash(cell):
    return hashlib.md5(json.dumps(cell, sort_keys=True).encode()).hexdigest()[:8]


def _score(E_real, E_gen_repeats, seed):
    """Per-repeat FID + KID(poly3), then mean/std over repeats"""
    fid_vals, kid_vals = [], []
    for r in range(len(E_gen_repeats)):
        fid_vals.append(compute_fid_from_features(E_real, E_gen_repeats[r]))
        rng = np.random.default_rng(seed * 1000 + r)
        kid_m, _ = compute_kid(E_real, E_gen_repeats[r],
                               subset_size=min(cfg.KID_SUBSET, len(E_gen_repeats[r])),
                               n_subsets=cfg.KID_N_SUBSETS, degree=cfg.KID_DEGREE, rng=rng)
        kid_vals.append(kid_m)
    return {
        "fid_mean": float(np.mean(fid_vals)), "fid_std": float(np.std(fid_vals)),
        "kid_mean": float(np.mean(kid_vals)), "kid_std": float(np.std(kid_vals)),
        "fid_vals": [float(v) for v in fid_vals], "kid_vals": [float(v) for v in kid_vals],
        "n_repeats": len(E_gen_repeats),
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_sanity_check", action="store_true",
                    help="skip the recon_train-vs-train27k separation check "
                         "(NOT recommended -- see mmsold_minimal/arcface_readme.md)")
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(os.path.join(a.celeba_dir, "valid")) or \
       not os.path.isdir(os.path.join(a.celeba_dir, "train")):
        raise FileNotFoundError(
            f"{a.celeba_dir} must contain train/ and valid/ subfolders of raw "
            "CelebA-HQ PNGs (img_<N>.png) -- ArcFace's REAL reference set needs "
            "actual pixels (this pipeline only kept precomputed Inception "
            "features locally, no reusable ArcFace feature cache exists "
            "anywhere). Pass the correct path with --celeba_dir.")

    print("=" * 60)
    print("Step 7 (diffae): ArcFace identity-space FID/KID for CelebA-HQ generation results table")
    print("=" * 60)

    out_dir = os.path.join(a.results_dir, "arcface")
    os.makedirs(out_dir, exist_ok=True)
    n_generate = 50 if a.smoke_test else cfg.N_GENERATE
    n_repeats = 2 if a.smoke_test else cfg.N_REPEATS
    gen_kwargs = (dict(n_generate=n_generate, sold_nsteps=20, cfdm_nsteps=20,
                       knnpf_nsteps=20, baoab_nsteps=20) if a.smoke_test else {})
    suffix = "_smoke" if a.smoke_test else ""

    print("\nBuilding ArcFace recognition model (insightface buffalo_l/w600k_r50) ...")
    embed = build_arcface_embedder()

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

    valid_paths = _dir_names(a.celeba_dir, "valid")
    train_paths = _dir_names(a.celeba_dir, "train")
    if len(valid_paths) < 3000:
        raise ValueError(f"{a.celeba_dir}/valid has only {len(valid_paths)} images, need >=3000")
    print(f"\nRaw CelebA-HQ dataset: {a.celeba_dir}  "
          f"(train={len(train_paths)}, valid={len(valid_paths)})")

    # ── REAL reference embeddings (cached -- pure ArcFace embed, no DiffAE
    #    decode needed since these ARE real photos) ─────────────────────────
    real_cache = os.path.join(out_dir, "real_valid_arcface.npy")
    if os.path.exists(real_cache):
        E_real = np.load(real_cache)
        print(f"\nReusing cached REAL reference ArcFace embeddings: {E_real.shape}")
    else:
        print(f"\nEmbedding {min(3000, len(valid_paths))} REAL held-out valid faces ...")
        imgs = _load_dir_images(valid_paths[:3000])
        E_real = embed(imgs)
        np.save(real_cache, E_real)
    print(f"  E_real {E_real.shape}")

    # ── Sanity check ───────────────────
    if not a.skip_sanity_check:
        n_sanity = min(3000, len(train_paths))
        print(f"\nSanity check: decoding first {n_sanity} TRAIN latents "
              f"(decoder identity ceiling) ...")
        sanity_cache = os.path.join(out_dir, "recon_train_arcface.npy")
        if os.path.exists(sanity_cache):
            E_recon_train = np.load(sanity_cache)
        else:
            Z_train512 = np.load(cfg.DIFFAE_ZSEM_TRAIN)[:n_sanity].astype(np.float32)
            imgs_recon = decode_batched(Z_train512, mean512, Vt, Vb, lambda_tail, render_fn,
                                        is_subspace=False, batch=cfg.DECODE_BATCH, seed=a.seed)
            E_recon_train = embed(imgs_recon)
            np.save(sanity_cache, E_recon_train)
        train_cache = os.path.join(out_dir, "train3k_arcface.npy")
        if os.path.exists(train_cache):
            E_train3k = np.load(train_cache)
        else:
            imgs_train = _load_dir_images(train_paths[:n_sanity])
            E_train3k = embed(imgs_train)
            np.save(train_cache, E_train3k)
        rng = np.random.default_rng(0)
        i, j = rng.integers(0, len(E_train3k), 20000), rng.integers(0, len(E_train3k), 20000)
        m = i != j
        rand_sim = (E_train3k[i[m]] * E_train3k[j[m]]).sum(1)
        same_sim = (E_recon_train * E_train3k[:len(E_recon_train)]).sum(1)
        sep = float(np.median(same_sim) - np.median(rand_sim))
        print(f"  random train-train median cosine              = {np.median(rand_sim):.3f}")
        print(f"  same-identity (recon vs train img) median cosine = {np.median(same_sim):.3f}")
        if sep > 0.4:
            print(f"  separation = {sep:.3f}  -> GOOD, embedding is discriminative")
        else:
            print(f"  separation = {sep:.3f}  -> [WARNING] POOR: crop/alignment "
                  f"likely wrong -- every ArcFace number below may be meaningless!")

    table3_path = os.path.join(a.results_dir, "table3.json")
    table3 = json.load(open(table3_path))["table"]

    results = {}
    for method_key, label, subdir in METHODS:
        if method_key not in table3:
            print(f"\n[MISSING] {label}: not in {table3_path} -- run step4 first")
            continue
        cell = table3[method_key]["decoded_cell"]
        print(f"\n{'='*40}\n{label}: regenerating at its CelebA-HQ generation results table decoded_cell ...")
        cache_dir = (os.path.join(a.results_dir, "sgbaoab",
                                  "tilting_cache_smoke" if a.smoke_test else "tilting_cache")
                    if method_key == "sg_baoab" else None)
        gen_fn = _make_gen_fn(method_key, cell, U_train190, gamma190,
                              cache_dir=cache_dir, **gen_kwargs)

        emb_cache = os.path.join(out_dir, f"{method_key}_arcface_{_cell_hash(cell)}{suffix}.npy")
        if os.path.exists(emb_cache):
            E_gen_repeats = np.load(emb_cache)
            print(f"  reusing cached embeddings {emb_cache} {E_gen_repeats.shape}")
        else:
            E_gen_repeats = []
            for r in range(n_repeats):
                U_gen190 = np.asarray(gen_fn(a.seed + r), dtype=np.float32)
                imgs = decode_batched(U_gen190, mean512, Vt, Vb, lambda_tail, render_fn,
                                      is_subspace=True, batch=cfg.DECODE_BATCH,
                                      seed=a.seed * 1000 + r + 700000)
                E_gen_repeats.append(embed(imgs))
                print(f"    repeat {r + 1}/{n_repeats}: decoded + embedded {len(imgs)} images")
            E_gen_repeats = np.stack(E_gen_repeats)   # (n_repeats, n_generate, 512)
            np.save(emb_cache, E_gen_repeats)

        results[method_key] = {"label": label, **_score(E_real, E_gen_repeats, a.seed)}
        res = results[method_key]
        print(f"  FID(ArcFace)={res['fid_mean']:.3f}+/-{res['fid_std']:.3f}  "
              f"KID(ArcFace,x1000)={1000 * res['kid_mean']:.3f}+/-{1000 * res['kid_std']:.3f}")

    # ── Latent DDPM ─
    ddpm_gen_dir = os.path.join(a.results_dir, "ddpm", "generated")
    if os.path.isdir(ddpm_gen_dir):
        print(f"\n{'='*40}\nLatent DDPM: decoding its own saved generated repeats ...")
        repeat_files = sorted(
            f for f in os.listdir(ddpm_gen_dir)
            if f.startswith("repeat") and f.endswith(f"{suffix}.npy")
            and (suffix or "_smoke" not in f))
        emb_cache = os.path.join(out_dir, f"latent_ddpm_arcface{suffix}.npy")
        if os.path.exists(emb_cache):
            E_gen_repeats = np.load(emb_cache)
            print(f"  reusing cached embeddings {emb_cache} {E_gen_repeats.shape}")
        else:
            E_gen_repeats = []
            for r, fname in enumerate(repeat_files[:n_repeats]):
                Z_gen512 = np.load(os.path.join(ddpm_gen_dir, fname))
                imgs = decode_batched(Z_gen512, mean512, Vt, Vb, lambda_tail, render_fn,
                                      is_subspace=False, batch=cfg.DECODE_BATCH,
                                      seed=a.seed * 1000 + r)
                E_gen_repeats.append(embed(imgs))
                print(f"    repeat {r + 1}/{len(repeat_files[:n_repeats])}: "
                      f"decoded + embedded {len(imgs)} images")
            E_gen_repeats = np.stack(E_gen_repeats)
            np.save(emb_cache, E_gen_repeats)
        results["latent_ddpm"] = {"label": "Latent DDPM", **_score(E_real, E_gen_repeats, a.seed)}
        res = results["latent_ddpm"]
        print(f"  FID(ArcFace)={res['fid_mean']:.3f}+/-{res['fid_std']:.3f}  "
              f"KID(ArcFace,x1000)={1000 * res['kid_mean']:.3f}+/-{1000 * res['kid_std']:.3f}")
    else:
        print(f"\n[MISSING] Latent DDPM: {ddpm_gen_dir} -- run step3 first")

    out_path = os.path.join(a.results_dir, f"table3_arcface{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*40}\nSummary (ready for the CelebA-HQ generation results table LaTeX rows):")
    print(f"{'Method':<16} {'FID (ArcFace)':>18} {'KID (ArcFace, x1e3)':>22}")
    for method_key, label, _ in METHODS + [("latent_ddpm", "Latent DDPM", None)]:
        if method_key not in results:
            continue
        r = results[method_key]
        print(f"{label:<16} {r['fid_mean']:>10.3f}+/-{r['fid_std']:<6.3f} "
              f"{1000 * r['kid_mean']:>12.3f}+/-{1000 * r['kid_std']:<6.3f}")

    print("\nStep 7 (diffae) complete!")


if __name__ == "__main__":
    main()
