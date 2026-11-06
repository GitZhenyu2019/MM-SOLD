"""
step2e_run_knnpflow.py
============================
KNN-PFlow grid search for the digit-8 generation experiment.
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
from knn_pflow_sampling import sample_knn_pflow               # noqa: E402


def _K_eff(K_grid_val, n_train):
    return n_train if K_grid_val == "N" else int(K_grid_val)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR, "subspace"))
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "knnpflow"))
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
    print("Step 2e: KNN-PFlow grid search")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    U_train8 = np.load(os.path.join(a.subspace_dir, "U_train8.npy"))
    mean_8   = np.load(os.path.join(a.subspace_dir, "mean_8.npy"))
    V        = np.load(os.path.join(a.subspace_dir, "proj_V.npy"))
    imgs_test8 = np.load(os.path.join(a.subspace_dir, "imgs_test8.npy"))
    meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    NN8 = meta["NN8"]
    n_train = U_train8.shape[0]
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

    cmax_grid   = [1]         if a.smoke_test else cfg.KNNPF_CMAX_GRID
    cmin_grid   = [0.10]      if a.smoke_test else cfg.KNNPF_CMIN_GRID
    K_grid      = [32]        if a.smoke_test else cfg.KNNPF_K_GRID
    solver_grid = ["euler"]   if a.smoke_test else cfg.KNNPF_SOLVER_GRID
    n_particles = 8           if a.smoke_test else cfg.N_PER_REPEAT
    nsteps      = 20          if a.smoke_test else cfg.KNNPF_NSTEPS
    n_repeats   = 2            if a.smoke_test else cfg.N_REPEATS

    def _generate(sigma_max, sigma_min, K_eff, solver, nsteps_, n_particles_, seed):
        U_gen = sample_knn_pflow(
            U_train8, mean0, S_eye, S_eye,
            n_particles=n_particles_, sigma_max=sigma_max, sigma_min=sigma_min,
            num_steps=nsteps_, K=K_eff, solver=solver,
            batch_size=min(500, n_particles_), seed=seed,
        )
        feats_gen = decode_and_extract(U_gen, mean_8, V, nrae_model, nrae_params,
                                       n_dct, clf_params)
        return U_gen, feats_gen

    # ── Timing (CPU) at the representative cell ─────────────────────────────
    timing_mean = timing_std = None
    if not a.smoke_test:
        t_smax = cfg.KNNPF_TIMING_CMAX * NN8
        t_smin = cfg.KNNPF_TIMING_CMIN * NN8 / np.sqrt(cfg.PROJ_K)
        t_K = _K_eff(cfg.KNNPF_TIMING_K, n_train)
        print(f"\nTiming KNN-PFlow on CPU (c_max={cfg.KNNPF_TIMING_CMAX}, "
              f"c_min={cfg.KNNPF_TIMING_CMIN}, K={cfg.KNNPF_TIMING_K}, "
              f"solver={cfg.KNNPF_TIMING_SOLVER}) "
              f"sigma_max={t_smax:.5f} sigma_min={t_smin:.5f} ...")

        def _timing_fn():
            return sample_knn_pflow(
                U_train8, mean0, S_eye, S_eye,
                n_particles=cfg.N_PER_REPEAT, sigma_max=t_smax, sigma_min=t_smin,
                num_steps=cfg.KNNPF_TIMING_NSTEPS, K=t_K,
                solver=cfg.KNNPF_TIMING_SOLVER,
                batch_size=min(500, cfg.N_PER_REPEAT), seed=a.seed)

        timing_mean, timing_std = measure_cpu_time(_timing_fn)
        print(f"  KNN-PFlow CPU sampling time: {timing_mean:.3f}s +/- {timing_std:.3f}s "
              f"({timing_mean / cfg.N_PER_REPEAT * 1000:.3f} ms/sample)")

    # ── Grid search (GPU / default device) ──────────────────────────────────
    print(f"\n{'='*40}")
    print("Grid search: KNN-PFlow\n")
    total = len(cmax_grid) * len(cmin_grid) * len(K_grid) * len(solver_grid)
    suffix = "_smoke" if a.smoke_test else ""
    results_path = os.path.join(a.out, f"results_knnpflow{suffix}.json")
    grid_meta = {
        "method": "knn_pflow",
        "cmax_grid": cfg.KNNPF_CMAX_GRID, "cmin_grid": cfg.KNNPF_CMIN_GRID,
        "K_grid": cfg.KNNPF_K_GRID, "solver_grid": cfg.KNNPF_SOLVER_GRID,
        "NN8": NN8, "proj_k": cfg.PROJ_K, "n_train": n_train,
        "knnpf_nsteps": cfg.KNNPF_NSTEPS,
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
    for xi, c_max in enumerate(cmax_grid):
        for ni, c_min in enumerate(cmin_grid):
            for ki, K_val in enumerate(K_grid):
                for ri, solver in enumerate(solver_grid):
                    cell_num += 1
                    key = f"{xi}_{ni}_{ki}_{ri}"
                    grid_png = os.path.join(grids_dir, f"grid_{key}.png")
                    need_metrics = (key not in grid_results) or a.overwrite
                    need_image = a.make_grids and (a.overwrite or not os.path.exists(grid_png))
                    if not need_metrics and not need_image:
                        print(f"[{cell_num:3d}/{total}]  c_max={c_max}  c_min={c_min}  "
                              f"K={K_val}  solver={solver}  -- already done, skipping")
                        continue
                    sigma_max = c_max * NN8
                    sigma_min = c_min * NN8 / np.sqrt(cfg.PROJ_K)
                    K_eff = _K_eff(K_val, n_train)
                    print(f"[{cell_num:3d}/{total}]  c_max={c_max}  c_min={c_min}  "
                          f"K={K_val}  solver={solver}  "
                          f"(sigma_max={sigma_max:.5f} sigma_min={sigma_min:.5f})")

                    base_seed = a.seed + xi * 10000 + ni * 1000 + ki * 10 + ri

                    if need_metrics:
                        def gen_fn(seed, sigma_max=sigma_max, sigma_min=sigma_min,
                                  K_eff=K_eff, solver=solver):
                            return _generate(sigma_max, sigma_min, K_eff, solver,
                                             nsteps, n_particles, seed)

                        t0 = time.perf_counter()
                        res = repeat_and_evaluate(gen_fn, feats_test, U_train8, tau,
                                                  n_repeats=n_repeats, seed=base_seed)
                        elapsed = time.perf_counter() - t0
                        print(f"  KID={res['kid_mean']:.4f}+/-{res['kid_std']:.4f}  "
                              f"Recall={res['recall_mean']:.3f}+/-{res['recall_std']:.3f}  "
                              f"DupRate={res['duprate_mean']:.3f}+/-{res['duprate_std']:.3f}  "
                              f"({elapsed:.1f}s)")

                        grid_results[key] = {
                            "c_max": c_max, "c_min": c_min, "K": K_val, "solver": solver,
                            "sigma_max": sigma_max, "sigma_min": sigma_min,
                            **res,
                        }
                        save_grid_json(results_path, grid_meta, grid_results)

                    if need_image:
                        n_vis = a.vis_grid * a.vis_grid
                        U_vis = sample_knn_pflow(
                            U_train8, mean0, S_eye, S_eye, n_particles=n_vis,
                            sigma_max=sigma_max, sigma_min=sigma_min, num_steps=nsteps,
                            K=K_eff, solver=solver, batch_size=min(500, n_vis),
                            seed=base_seed + 999999)
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
            "method": "knn_pflow", "device": "CPU",
            "sample_time_s": timing_mean, "sample_time_std_s": timing_std,
            "sample_time_per_sample_ms": timing_mean / cfg.N_PER_REPEAT * 1000,
            "train_opt_time_h": 0.0,
            "timing_c_max": cfg.KNNPF_TIMING_CMAX, "timing_c_min": cfg.KNNPF_TIMING_CMIN,
            "timing_K": cfg.KNNPF_TIMING_K, "timing_solver": cfg.KNNPF_TIMING_SOLVER,
            "timing_nsteps": cfg.KNNPF_TIMING_NSTEPS,
            "n_warmup": cfg.TIMING_N_WARMUP, "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_knnpflow.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 2e complete!")


if __name__ == "__main__":
    main()
