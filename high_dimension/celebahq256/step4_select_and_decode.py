"""
step4_select_and_decode.py
====================================
Assemble CelebA-HQ generation results table (CelebA-HQ generation quality + computational cost) for the
DiffAE-latent generation experiment.
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

import config as cfg                                    # noqa: E402
import tilting_lion as TL                                        # noqa: E402
from common import (                                     # noqa: E402
    decode_and_score_images, h_sold, h_baoab, to_subspace,
    gamma_baoab, sample_class_overdamped_manifold_knn,
)
from metrics_celeba import (                                     # noqa: E402
    build_inception_extractor, compute_tau, compute_recall, compute_duprate,
)
from cfdm_sampling import sample_cfdm_knn_remainder                 # noqa: E402
from knn_pflow_sampling import sample_knn_pflow                     # noqa: E402
from swf_sampling import sample_swf                                 # noqa: E402


METHODS = [
    ("mmsold",       "MM-SOLD",       "mmsold",      "results_mmsold.json",      "timing_mmsold.json"),
    ("cfdm",         "sigma-CFDM",    "cfdm",        "results_cfdm.json",        "timing_cfdm.json"),
    ("knn_pflow",    "KNN-PFlow",     "knnpflow",    "results_knnpflow.json",    "timing_knnpflow.json"),
    ("swf",          "SWF",           "swf",         "results_swf.json",         "timing_swf.json"),
    ("plugin_baoab", "Plug-in BAOAB", "pluginbaoab", "results_pluginbaoab.json", "timing_pluginbaoab.json"),
    ("sg_baoab",     "SG-BAOAB",      "sgbaoab",     "results_sgbaoab.json",     "timing_sgbaoab.json"),
]


def _make_gen_fn(method_key, cell, U_train190, gamma190, cache_dir=None, *,
                 n_generate=None, sold_nsteps=None, cfdm_nsteps=None,
                 knnpf_nsteps=None, baoab_nsteps=None):
    n_generate  = cfg.N_GENERATE   if n_generate  is None else n_generate
    sold_nsteps = cfg.SOLD_NSTEPS  if sold_nsteps is None else sold_nsteps
    cfdm_nsteps = cfg.CFDM_NSTEPS  if cfdm_nsteps is None else cfdm_nsteps
    knnpf_nsteps = cfg.KNNPF_NSTEPS if knnpf_nsteps is None else knnpf_nsteps
    baoab_nsteps = cfg.BAOAB_NSTEPS if baoab_nsteps is None else baoab_nsteps

    Xj = jnp.asarray(U_train190, dtype=jnp.float32)
    delta, sigma, M = cell.get("delta"), cell.get("sigma"), cell.get("M")

    if method_key == "mmsold":
        n_particles = max(n_generate, U_train190.shape[1] + 1)
        def gen_fn(seed):
            U_gen, _ = sample_class_overdamped_manifold_knn(
                X_class=Xj, n_particles=n_particles, nsteps=sold_nsteps,
                h=h_sold(delta), sigma_gmm=delta, sigma_smoothing=sigma, M=M,
                knn_K=cfg.KNN_K_SMOOTH, knn_L=cfg.KNN_L_SMOOTH,
                discretization=cfg.SOLD_DISCRETIZ, seed=seed)
            return np.array(U_gen, dtype=np.float32)
        return gen_fn

    if method_key == "cfdm":
        d = U_train190.shape[1]
        mean0, S_eye = jnp.zeros(d, dtype=jnp.float32), jnp.eye(d, dtype=jnp.float32)
        def gen_fn(seed):
            return sample_cfdm_knn_remainder(
                U_train190, mean0, S_eye, S_eye, n_generate, sigma, M,
                knn_K=cfg.KNN_K_SMOOTH, knn_L=cfg.KNN_L_SMOOTH,
                nsteps=cfdm_nsteps, batch_size=min(cfg.CFDM_BATCH, n_generate),
                seed=seed)
        return gen_fn

    if method_key == "knn_pflow":
        d = U_train190.shape[1]
        mean0, S_eye = jnp.zeros(d, dtype=jnp.float32), jnp.eye(d, dtype=jnp.float32)
        sigma_max, sigma_min = cell["sigma_max"], cell["sigma_min"]
        K_eff = min(cell["K"], U_train190.shape[0])
        solver = cell["solver"]
        def gen_fn(seed):
            return sample_knn_pflow(
                U_train190, mean0, S_eye, S_eye, n_particles=n_generate,
                sigma_max=sigma_max, sigma_min=sigma_min, num_steps=knnpf_nsteps,
                K=K_eff, solver=solver, batch_size=min(500, n_generate), seed=seed)
        return gen_fn

    if method_key == "swf":
        d = U_train190.shape[1]
        mean0, S_eye = jnp.zeros(d, dtype=jnp.float32), jnp.eye(d, dtype=jnp.float32)
        n_swf, R, eta, init = cell["n_SWF"], cell["R"], cell["eta"], cell["init"]
        meta = json.load(open(os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace", "meta.json")))
        NN = meta["NN"]
        def gen_fn(seed):
            return sample_swf(
                U_train190, mean0, S_eye, S_eye, n_particles=n_generate,
                num_steps=n_swf, num_projections=R, step_size=eta, init=init,
                nn_dist=NN, seed=seed)
        return gen_fn

    if method_key == "plugin_baoab":
        h = h_baoab(delta)
        lam, Lam = TL.plugin_tilting(jax.random.PRNGKey(0), Xj, delta, sigma, M,
                                     score=cfg.BAOAB_SCORE, knn_K=cfg.KNN_K_BAOAB,
                                     knn_L=cfg.KNN_L_BAOAB)
        def gen_fn(seed):
            q = TL.sample_tilted(lam, Lam, Xj, delta=delta, sigma_s=sigma, M=M,
                                 h=h, gamma=gamma190, n_steps=baoab_nsteps,
                                 n_samples=n_generate, splitting=cfg.BAOAB_SPLITTING,
                                 score=cfg.BAOAB_SCORE, knn_K=cfg.KNN_K_BAOAB,
                                 knn_L=cfg.KNN_L_BAOAB, seed=seed)
            return np.asarray(q, dtype=np.float32)
        return gen_fn

    if method_key == "sg_baoab":
        h = h_baoab(delta)
        cache_path = None
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"cell_cd{cell['c_delta']:.2f}_cs{cell['c_sigma']:.2f}_M{M}.npz")
        if cache_path and os.path.exists(cache_path):
            dd = np.load(cache_path)
            lam, Lam = dd["lam"], dd["Lam"]
            print(f"  SG-BAOAB: reusing cached (lambda, Lambda) from {cache_path}")
        else:
            print("  SG-BAOAB: no cached (lambda, Lambda) found for the selected "
                  "cell -- refitting with the grid's Lion budget (this can be slow)")
            lam, Lam, _hist = TL.estimate_tilting_lion(
                Xj, delta=delta, sigma_s=sigma, M=M, h=h, gamma=gamma190,
                n_steps=cfg.LION_N_STEPS, n_chains=cfg.LION_N_CHAINS,
                n_iters=cfg.LION_N_ITERS, lr_rel=cfg.LION_LR_REL, b1=cfg.LION_B1,
                b2=cfg.LION_B2, splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
                knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB,
                warm_start=cfg.LION_WARM_START, seed=0, verbose=True)
        def gen_fn(seed):
            q = TL.sample_tilted(lam, Lam, Xj, delta=delta, sigma_s=sigma, M=M,
                                 h=h, gamma=gamma190, n_steps=baoab_nsteps,
                                 n_samples=n_generate, splitting=cfg.BAOAB_SPLITTING,
                                 score=cfg.BAOAB_SCORE, knn_K=cfg.KNN_K_BAOAB,
                                 knn_L=cfg.KNN_L_BAOAB, seed=seed)
            return np.asarray(q, dtype=np.float32)
        return gen_fn

    raise ValueError(f"unknown method_key {method_key!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR_DIFFAE)
    ap.add_argument("--engine_path", default=cfg.DIFFAE_TRT_ENGINE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke_test", action="store_true",
                    help="use tiny sample counts (still exercises the real "
                         "decoder -- requires TensorRT engine + GPU)")
    a = ap.parse_args()
    os.makedirs(a.results_dir, exist_ok=True)

    print("=" * 60)
    print("Step 4 (diffae): select best cell per method, decode + score (CelebA-HQ generation results table)")
    print("=" * 60)

    U_train190 = np.load(os.path.join(a.subspace_dir, "U_train190.npy"))
    U_test190  = np.load(os.path.join(a.subspace_dir, "U_test190.npy"))
    Z_test512  = np.load(os.path.join(a.subspace_dir, "Z_test512.npy"))
    mean512    = np.load(os.path.join(a.subspace_dir, "mean512.npy"))
    Vt         = np.load(os.path.join(a.subspace_dir, "Vt.npy"))
    Vb         = np.load(os.path.join(a.subspace_dir, "Vb.npy"))
    lambda_tail = np.load(os.path.join(a.subspace_dir, "lambda_tail.npy"))
    gamma190 = gamma_baoab(U_train190)
    tau = compute_tau(U_train190, percentile=cfg.DUP_PERCENTILE)
    _meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    RBF_BW, RBF_BW_IMG = _meta["rbf_bw"], _meta["rbf_bw_img"]

    print("\nBuilding image decoder (TensorRT DPM-Solver++(2M), "
          f"{cfg.DECODE_NSTEPS} steps) and Inception feature extractor ...")
    from dpm_decode import make_renderer
    render_fn, _ctx = make_renderer(engine_path=a.engine_path, nsteps=cfg.DECODE_NSTEPS,
                                    batch=cfg.DECODE_BATCH)
    inception_model, inception_device = build_inception_extractor()
    f_ref = np.load(cfg.DIFFAE_FEATURES_NPZ)["f_ref"]
    print(f"  f_ref (test-image Inception features): {f_ref.shape}")

    n_repeats = 2 if a.smoke_test else cfg.N_REPEATS
    results_suffix = "_smoke" if a.smoke_test else ""
    gen_kwargs = (dict(n_generate=50, sold_nsteps=20, cfdm_nsteps=20,
                       knnpf_nsteps=20, baoab_nsteps=20)
                 if a.smoke_test else {})

    table = {}
    for method_key, label, subdir, results_fname, timing_fname in METHODS:
        rdir = os.path.join(a.results_dir, subdir)
        rpath = os.path.join(rdir, results_fname.replace(".json", f"{results_suffix}.json"))
        tpath = os.path.join(rdir, timing_fname.replace(".json", f"{results_suffix}.json"))
        if not os.path.exists(rpath):
            print(f"\n[MISSING] {label}: {rpath} -- run step2 for this method first")
            continue
        print(f"\n{'='*40}\n{label}: per-metric independent best over the grid\n")
        grid = json.load(open(rpath))["grid"]
        cells = list(grid.values())
        best_kid_cell = min(cells, key=lambda c: c["kid_mean"])
        best_recall_cell = max(cells, key=lambda c: c["recall_mean"])
        best_dup_cell = min(cells, key=lambda c: c["duprate_mean"])
        print(f"  latent-KID-best cell (this is the ONE decoded below): "
              f"c_delta={best_kid_cell.get('c_delta')} c_sigma={best_kid_cell.get('c_sigma')} "
              f"M={best_kid_cell.get('M')}  kid_mean={best_kid_cell['kid_mean']:.5f}")
        print(f"  Recall-best cell:   "
              f"c_delta={best_recall_cell.get('c_delta')} c_sigma={best_recall_cell.get('c_sigma')} "
              f"M={best_recall_cell.get('M')}  recall_mean={best_recall_cell['recall_mean']:.4f}")
        print(f"  DupRate-best cell:  "
              f"c_delta={best_dup_cell.get('c_delta')} c_sigma={best_dup_cell.get('c_sigma')} "
              f"M={best_dup_cell.get('M')}  duprate_mean={best_dup_cell['duprate_mean']:.4f}")

        cache_dir = (os.path.join(rdir, "tilting_cache_smoke" if a.smoke_test else "tilting_cache")
                    if method_key == "sg_baoab" else None)
        gen_fn = _make_gen_fn(method_key, best_kid_cell, U_train190, gamma190,
                              cache_dir=cache_dir, **gen_kwargs)

        print(f"  decoding {n_repeats} x {gen_kwargs.get('n_generate', cfg.N_GENERATE)} "
              f"samples (from the latent-KID-best cell) through the DiffAE decoder ...")
        img_res = decode_and_score_images(
            gen_fn, mean512, Vt, Vb, lambda_tail, render_fn, inception_model,
            inception_device, f_ref, rbf_bw_img=RBF_BW_IMG,
            n_repeats=n_repeats, seed=a.seed)
        print(f"  Image FID={img_res['fid_mean']:.3f}+/-{img_res['fid_std']:.3f}  "
              f"Image KID(poly3)={img_res['kid_mean']:.5f}+/-{img_res['kid_std']:.5f}  "
              f"Image KID(RBF)={img_res['kid_rbf_mean']:.5f}+/-{img_res['kid_rbf_std']:.5f}")

        timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
        table[method_key] = {
            "label": label,
            "decoded_cell": best_kid_cell, "image_metrics": img_res,
            "latent_kid": best_kid_cell["kid_mean"], "latent_kid_std": best_kid_cell["kid_std"],
            "recall": best_recall_cell["recall_mean"], "recall_std": best_recall_cell["recall_std"],
            "recall_cell": {k: best_recall_cell.get(k) for k in ("c_delta", "c_sigma", "M")},
            "duprate": best_dup_cell["duprate_mean"], "duprate_std": best_dup_cell["duprate_std"],
            "duprate_cell": {k: best_dup_cell.get(k) for k in ("c_delta", "c_sigma", "M")},
            "timing": timing,
        }

    # ── Latent DDPM: same decode-and-score treatment, no cell selection ────
    ddpm_dir = os.path.join(a.results_dir, "ddpm")
    ddpm_res_path = os.path.join(ddpm_dir, f"results_ddpm{results_suffix}.json")
    gen_dir = os.path.join(ddpm_dir, "generated")
    if os.path.exists(ddpm_res_path) and os.path.isdir(gen_dir):
        print(f"\n{'='*40}\nLatent DDPM: decoding its own generated repeats\n")
        ddpm_res = json.load(open(ddpm_res_path))
        repeat_files = sorted(
            f for f in os.listdir(gen_dir)
            if f.startswith("repeat") and f.endswith(f"{results_suffix}.npy")
            and (results_suffix or "_smoke" not in f))

        def ddpm_gen_fn(seed):
            r = (seed - a.seed) % max(len(repeat_files), 1)
            return np.load(os.path.join(gen_dir, repeat_files[r]))

        from common import decode_batched, compute_kid_rbfmix
        from metrics_celeba import (
            extract_inception_features, compute_fid_from_features,
            compute_kid as _compute_kid,
        )
        fid_vals, kid_img_vals, kid_img_rbf_vals = [], [], []
        kid_lat_vals, recall_vals, dup_vals = [], [], []
        for r in range(min(n_repeats, len(repeat_files)) or n_repeats):
            Z_gen512 = ddpm_gen_fn(a.seed + r)
            U_gen190 = to_subspace(Z_gen512, mean512, Vt)
            rng = np.random.default_rng(a.seed * 1000 + r)
            kid_lat, _ = compute_kid_rbfmix(Z_test512, Z_gen512, RBF_BW, rng=rng)
            kid_lat_vals.append(kid_lat)
            recall_vals.append(compute_recall(U_test190, U_gen190, k_neighbour=cfg.K_RECALL))
            dup_vals.append(compute_duprate(U_gen190, U_train190, tau))

            imgs = decode_batched(Z_gen512, mean512, Vt, Vb, lambda_tail, render_fn,
                                  is_subspace=False, batch=cfg.DECODE_BATCH,
                                  seed=a.seed * 1000 + r)
            feats_gen = extract_inception_features(imgs.astype(np.float32) / 255.0,
                                                    inception_model, inception_device)
            fid_vals.append(compute_fid_from_features(f_ref, feats_gen))
            kid_m, _ = _compute_kid(f_ref, feats_gen, subset_size=min(cfg.KID_SUBSET, len(feats_gen)),
                                    n_subsets=cfg.KID_N_SUBSETS, degree=cfg.KID_DEGREE, rng=rng)
            kid_img_vals.append(kid_m)
            kid_rbf_m, _ = compute_kid_rbfmix(f_ref, feats_gen, RBF_BW_IMG, rng=rng)
            kid_img_rbf_vals.append(kid_rbf_m)
        ddpm_img_res = {
            "fid_mean": float(np.mean(fid_vals)), "fid_std": float(np.std(fid_vals)),
            "kid_mean": float(np.mean(kid_img_vals)), "kid_std": float(np.std(kid_img_vals)),
            "kid_rbf_mean": float(np.mean(kid_img_rbf_vals)), "kid_rbf_std": float(np.std(kid_img_rbf_vals)),
            "n_repeats": len(fid_vals),
        }
        print(f"  Image FID={ddpm_img_res['fid_mean']:.3f}+/-{ddpm_img_res['fid_std']:.3f}  "
              f"Image KID(poly3)={ddpm_img_res['kid_mean']:.5f}+/-{ddpm_img_res['kid_std']:.5f}  "
              f"Image KID(RBF)={ddpm_img_res['kid_rbf_mean']:.5f}+/-{ddpm_img_res['kid_rbf_std']:.5f}  "
              f"Latent KID={np.mean(kid_lat_vals):.5f}+/-{np.std(kid_lat_vals):.5f}  "
              f"Recall={np.mean(recall_vals):.3f}+/-{np.std(recall_vals):.3f}  "
              f"DupRate={np.mean(dup_vals):.3f}+/-{np.std(dup_vals):.3f}")
        ddpm_timing_path = os.path.join(ddpm_dir, "timing_ddpm.json")
        table["latent_ddpm"] = {
            "label": "Latent DDPM", "image_metrics": ddpm_img_res,
            "latent_kid": float(np.mean(kid_lat_vals)), "latent_kid_std": float(np.std(kid_lat_vals)),
            "recall": float(np.mean(recall_vals)), "recall_std": float(np.std(recall_vals)),
            "duprate": float(np.mean(dup_vals)), "duprate_std": float(np.std(dup_vals)),
            "timing": json.load(open(ddpm_timing_path)) if os.path.exists(ddpm_timing_path) else {},
        }
    else:
        print(f"\n[MISSING] Latent DDPM: {ddpm_res_path} or {gen_dir} -- run step3 first")

    out_path = os.path.join(a.results_dir, "table3.json")
    with open(out_path, "w") as f:
        json.dump({"table": table}, f, indent=2, default=float)
    print(f"\nSaved CelebA-HQ generation results table: {out_path}")
    print("\nStep 4 (diffae) complete!")


if __name__ == "__main__":
    main()
