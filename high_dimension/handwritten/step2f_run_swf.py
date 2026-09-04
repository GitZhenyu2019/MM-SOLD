"""
step2f_run_swf.py
======================
Sliced-Wasserstein Flow (SWF) grid search for the revised digit-8 generation
experiment.
"""
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import config as cfg                                    # noqa: E402
from common import (                                     # noqa: E402
    load_nrae, load_clf, decode_and_extract, decode_from_subspace,
    repeat_and_evaluate, measure_cpu_time, load_grid_json, save_grid_json,
    seamless_grid, find_nn, save_nn_vis,
)
from metrics import extract_features, compute_tau           # noqa: E402
from swf_sampling import sample_swf                          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR, "subspace"))
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "swf"))
    ap.add_argument("--nrae_ckpt", default=cfg.NRAE_CKPT)
    ap.add_argument("--clf_ckpt", default=cfg.CLF_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute cells already in the results JSON")
    ap.add_argument("--make_grids", action="store_true",
                    help="also save an 8x8 sample grid + NN visualisation PNG "
                         "for every grid cell")
    ap.add_argument("--vis_grid", type=int, default=8)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 2f (v2): Sliced-Wasserstein Flow grid search")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    U_train8 = np.load(os.path.join(a.subspace_dir, "U_train8.npy"))
    mean_8   = np.load(os.path.join(a.subspace_dir, "mean_8.npy"))
    V        = np.load(os.path.join(a.subspace_dir, "proj_V.npy"))
    imgs_test8 = np.load(os.path.join(a.subspace_dir, "imgs_test8.npy"))
    meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    NN8 = meta["NN8"]
    d = U_train8.shape[1]
    print(f"  U_train8 {U_train8.shape}  imgs_test8 {imgs_test8.shape}  NN8={NN8:.5f}")

    nrae_model, nrae_params, n_dct = load_nrae(a.nrae_ckpt, seed=0)
    clf_params = load_clf(a.clf_ckpt)

    print("\nExtracting test CNN features ...")
    feats_test = extract_features(imgs_test8, clf_params, batch_size=64)
    tau = compute_tau(U_train8, percentile=1)
    print(f"  feats_test {feats_test.shape}  DupRate tau (top-40) = {tau:.5f}")

    mean0 = jnp.zeros(d, dtype=jnp.float32)
    S_eye = jnp.eye(d, dtype=jnp.float32)

    nswf_grid = [300]             if a.smoke_test else cfg.SWF_NSWF_GRID
    r_grid    = [64]              if a.smoke_test else cfg.SWF_R_GRID
    eta_grid  = [0.05]            if a.smoke_test else cfg.SWF_ETA_GRID
    init_grid = ["gaussian_full"] if a.smoke_test else cfg.SWF_INIT_GRID
    n_particles = 8 if a.smoke_test else cfg.N_PER_REPEAT
    nswf_cap    = 5 if a.smoke_test else None   # override n_SWF in smoke mode
    n_repeats   = 2  if a.smoke_test else cfg.N_REPEATS

    def _generate(n_swf, R, eta, init, n_particles_, seed):
        U_gen = sample_swf(
            U_train8, mean0, S_eye, S_eye,
            n_particles=n_particles_, num_steps=n_swf, num_projections=R,
            step_size=eta, init=init, nn_dist=NN8, seed=seed,
        )
        feats_gen = decode_and_extract(U_gen, mean_8, V, nrae_model, nrae_params,
                                       n_dct, clf_params)
        return U_gen, feats_gen

    # ── Timing (CPU) at the representative cell ─────────────────────────────
    timing_mean = timing_std = None
    if not a.smoke_test:
        print(f"\nTiming SWF on CPU (n_SWF={cfg.SWF_TIMING_NSWF}, R={cfg.SWF_TIMING_R}, "
              f"eta={cfg.SWF_TIMING_ETA}, init={cfg.SWF_TIMING_INIT}) ...")

        def _timing_fn():
            return sample_swf(
                U_train8, mean0, S_eye, S_eye,
                n_particles=cfg.N_PER_REPEAT, num_steps=cfg.SWF_TIMING_NSWF,
                num_projections=cfg.SWF_TIMING_R, step_size=cfg.SWF_TIMING_ETA,
                init=cfg.SWF_TIMING_INIT, nn_dist=NN8, seed=a.seed)

        timing_mean, timing_std = measure_cpu_time(_timing_fn)
        print(f"  SWF CPU sampling time: {timing_mean:.3f}s +/- {timing_std:.3f}s "
              f"({timing_mean / cfg.N_PER_REPEAT * 1000:.3f} ms/sample)")

    # ── Grid search (GPU / default device) ──────────────────────────────────
    print(f"\n{'='*40}")
    print("Grid search: Sliced-Wasserstein Flow\n")
    total = len(nswf_grid) * len(r_grid) * len(eta_grid) * len(init_grid)
    suffix = "_smoke" if a.smoke_test else ""
    results_path = os.path.join(a.out, f"results_swf{suffix}.json")
    grid_meta = {
        "method": "swf",
        "nswf_grid": cfg.SWF_NSWF_GRID, "r_grid": cfg.SWF_R_GRID,
        "eta_grid": cfg.SWF_ETA_GRID, "init_grid": cfg.SWF_INIT_GRID,
        "NN8": NN8, "proj_k": cfg.PROJ_K,
        "n_per_repeat": cfg.N_PER_REPEAT, "n_repeats": cfg.N_REPEATS,
        "smoke_test": a.smoke_test,
    }
    grid_results = {} if a.overwrite else load_grid_json(results_path)
    if grid_results:
        print(f"  resuming: {len(grid_results)} cell(s) already in {results_path}")
    grids_dir = os.path.join(a.out, "grids")
    if a.make_grids:
        os.makedirs(grids_dir, exist_ok=True)
    cell_num = 0
    for si, n_swf in enumerate(nswf_grid):
        n_swf_use = nswf_cap if nswf_cap is not None else n_swf
        for ri, R in enumerate(r_grid):
            for ei, eta in enumerate(eta_grid):
                for ii, init in enumerate(init_grid):
                    cell_num += 1
                    key = f"{si}_{ri}_{ei}_{ii}"
                    grid_png = os.path.join(grids_dir, f"grid_{key}.png")
                    need_metrics = (key not in grid_results) or a.overwrite
                    need_image = a.make_grids and (a.overwrite or not os.path.exists(grid_png))
                    if not need_metrics and not need_image:
                        print(f"[{cell_num:3d}/{total}]  n_SWF={n_swf}  R={R}  "
                              f"eta={eta}  init={init}  -- already done, skipping")
                        continue
                    print(f"[{cell_num:3d}/{total}]  n_SWF={n_swf}  R={R}  "
                          f"eta={eta}  init={init}")

                    base_seed = a.seed + si * 10000 + ri * 1000 + ei * 100 + ii

                    if need_metrics:
                        def gen_fn(seed, n_swf_use=n_swf_use, R=R, eta=eta, init=init):
                            return _generate(n_swf_use, R, eta, init, n_particles, seed)

                        t0 = time.perf_counter()
                        res = repeat_and_evaluate(gen_fn, feats_test, U_train8, tau,
                                                  n_repeats=n_repeats, seed=base_seed)
                        elapsed = time.perf_counter() - t0
                        print(f"  KID={res['kid_mean']:.4f}+/-{res['kid_std']:.4f}  "
                              f"Recall={res['recall_mean']:.3f}+/-{res['recall_std']:.3f}  "
                              f"DupRate={res['duprate_mean']:.3f}+/-{res['duprate_std']:.3f}  "
                              f"({elapsed:.1f}s)")

                        grid_results[key] = {
                            "n_SWF": n_swf_use, "R": R, "eta": eta, "init": init,
                            **res,
                        }
                        save_grid_json(results_path, grid_meta, grid_results)

                    if need_image:
                        n_vis = a.vis_grid * a.vis_grid
                        U_vis = sample_swf(
                            U_train8, mean0, S_eye, S_eye, n_particles=n_vis,
                            num_steps=n_swf_use, num_projections=R, step_size=eta,
                            init=init, nn_dist=NN8, seed=base_seed + 999999)
                        imgs_vis = decode_from_subspace(U_vis, mean_8, V, nrae_model,
                                                        nrae_params, n_dct)
                        seamless_grid(imgs_vis, grid_png, G=a.vis_grid)
                        n_nn = min(32, n_vis)
                        nn_idx, _ = find_nn(U_vis[:n_nn], U_train8)
                        imgs_nn_q = decode_from_subspace(U_vis[:n_nn], mean_8, V, nrae_model,
                                                         nrae_params, n_dct)
                        imgs_nn_r = decode_from_subspace(U_train8[nn_idx], mean_8, V, nrae_model,
                                                         nrae_params, n_dct)
                        save_nn_vis(imgs_nn_q, imgs_nn_r,
                                   os.path.join(grids_dir, f"nn_{key}.png"))
                        print(f"  saved grid/nn images for cell {key}")

    if not a.smoke_test:
        timing_out = {
            "method": "swf", "device": "CPU",
            "sample_time_s": timing_mean, "sample_time_std_s": timing_std,
            "sample_time_per_sample_ms": timing_mean / cfg.N_PER_REPEAT * 1000,
            "train_opt_time_h": 0.0,
            "timing_n_SWF": cfg.SWF_TIMING_NSWF, "timing_R": cfg.SWF_TIMING_R,
            "timing_eta": cfg.SWF_TIMING_ETA, "timing_init": cfg.SWF_TIMING_INIT,
            "n_warmup": cfg.TIMING_N_WARMUP, "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_swf.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 2f (v2) complete!")


if __name__ == "__main__":
    main()
