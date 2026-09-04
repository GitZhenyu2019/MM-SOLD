"""
step2c_run_knnpflow.py
===============================
KNN-PFlow grid search for the CelebA-HQ DiffAE-latent generation experiment.
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
    repeat_and_evaluate_latent, measure_cpu_time, load_grid_json, save_grid_json,
)
from metrics_celeba import compute_tau                           # noqa: E402
from knn_pflow_sampling import sample_knn_pflow                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "knnpflow"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 2c (diffae): KNN-PFlow grid search")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    U_train190 = np.load(os.path.join(a.subspace_dir, "U_train190.npy"))
    U_test190  = np.load(os.path.join(a.subspace_dir, "U_test190.npy"))
    Z_test512  = np.load(os.path.join(a.subspace_dir, "Z_test512.npy"))
    mean512    = np.load(os.path.join(a.subspace_dir, "mean512.npy"))
    Vt         = np.load(os.path.join(a.subspace_dir, "Vt.npy"))
    Vb         = np.load(os.path.join(a.subspace_dir, "Vb.npy"))
    lambda_tail = np.load(os.path.join(a.subspace_dir, "lambda_tail.npy"))
    meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    NN = meta["NN"]
    RBF_BW = meta["rbf_bw"]
    n_train = U_train190.shape[0]
    d = U_train190.shape[1]
    print(f"  U_train190 {U_train190.shape}  U_test190 {U_test190.shape}  NN={NN:.5f}")

    tau = compute_tau(U_train190, percentile=cfg.DUP_PERCENTILE)
    print(f"  DupRate tau (top-190, percentile={cfg.DUP_PERCENTILE}) = {tau:.5f}")

    mean0 = jnp.zeros(d, dtype=jnp.float32)
    S_eye = jnp.eye(d, dtype=jnp.float32)

    cmax_grid   = [1]         if a.smoke_test else cfg.KNNPF_CMAX_GRID
    cmin_grid   = [0.10]      if a.smoke_test else cfg.KNNPF_CMIN_GRID
    K_grid      = [50]        if a.smoke_test else cfg.KNNPF_K_GRID
    solver_grid = ["euler"]   if a.smoke_test else cfg.KNNPF_SOLVER_GRID
    n_particles = 250         if a.smoke_test else cfg.N_GENERATE
    nsteps      = 20          if a.smoke_test else cfg.KNNPF_NSTEPS
    n_repeats   = 2           if a.smoke_test else cfg.N_REPEATS

    def _generate(sigma_max, sigma_min, K_eff, solver, nsteps_, n_particles_, seed):
        U_gen = sample_knn_pflow(
            U_train190, mean0, S_eye, S_eye,
            n_particles=n_particles_, sigma_max=sigma_max, sigma_min=sigma_min,
            num_steps=nsteps_, K=K_eff, solver=solver,
            batch_size=min(500, n_particles_), seed=seed,
        )
        return np.array(U_gen, dtype=np.float32)

    # ── Timing (CPU) at the representative cell ─────────────────────────────
    timing_mean = timing_std = None
    if not a.smoke_test:
        t_smax = cfg.KNNPF_TIMING_CMAX * NN
        t_smin = cfg.KNNPF_TIMING_CMIN * NN / np.sqrt(cfg.PROJ_K)
        t_K = min(cfg.KNNPF_TIMING_K, n_train)
        print(f"\nTiming KNN-PFlow on CPU (c_max={cfg.KNNPF_TIMING_CMAX}, "
              f"c_min={cfg.KNNPF_TIMING_CMIN}, K={cfg.KNNPF_TIMING_K}, "
              f"solver={cfg.KNNPF_TIMING_SOLVER}) "
              f"sigma_max={t_smax:.5f} sigma_min={t_smin:.5f} ...")

        def _timing_fn():
            return sample_knn_pflow(
                U_train190, mean0, S_eye, S_eye,
                n_particles=cfg.N_GENERATE, sigma_max=t_smax, sigma_min=t_smin,
                num_steps=cfg.KNNPF_TIMING_NSTEPS, K=t_K,
                solver=cfg.KNNPF_TIMING_SOLVER,
                batch_size=min(500, cfg.N_GENERATE), seed=a.seed)

        timing_mean, timing_std = measure_cpu_time(_timing_fn)
        print(f"  KNN-PFlow CPU sampling time: {timing_mean:.3f}s +/- {timing_std:.3f}s "
              f"({timing_mean / cfg.N_GENERATE * 1000:.3f} ms/sample)")

    # ── Grid search (GPU / default device) ──────────────────────────────────
    print(f"\n{'='*40}")
    print("Grid search: KNN-PFlow\n")
    total = len(cmax_grid) * len(cmin_grid) * len(K_grid) * len(solver_grid)
    suffix = "_smoke" if a.smoke_test else ""
    results_path = os.path.join(a.out, f"results_knnpflow{suffix}.json")
    meta_out = {
        "method": "knn_pflow",
        "cmax_grid": cfg.KNNPF_CMAX_GRID, "cmin_grid": cfg.KNNPF_CMIN_GRID,
        "K_grid": cfg.KNNPF_K_GRID, "solver_grid": cfg.KNNPF_SOLVER_GRID,
        "NN": NN, "proj_k": cfg.PROJ_K, "n_train": n_train,
        "knnpf_nsteps": cfg.KNNPF_NSTEPS,
        "n_per_repeat": cfg.N_GENERATE, "n_repeats": cfg.N_REPEATS,
        "smoke_test": a.smoke_test,
    }
    grid_results = {} if a.overwrite else load_grid_json(results_path)
    if grid_results:
        print(f"  resuming: {len(grid_results)} cell(s) already in {results_path}")
    cell_num = 0
    for xi, c_max in enumerate(cmax_grid):
        for ni, c_min in enumerate(cmin_grid):
            for ki, K_val in enumerate(K_grid):
                for ri, solver in enumerate(solver_grid):
                    cell_num += 1
                    key = f"{xi}_{ni}_{ki}_{ri}"
                    if key in grid_results and not a.overwrite:
                        print(f"[{cell_num:3d}/{total}]  c_max={c_max}  c_min={c_min}  "
                              f"K={K_val}  solver={solver}  -- already done, skipping")
                        continue
                    sigma_max = c_max * NN
                    sigma_min = c_min * NN / np.sqrt(cfg.PROJ_K)
                    K_eff = min(K_val, n_train)
                    print(f"[{cell_num:3d}/{total}]  c_max={c_max}  c_min={c_min}  "
                          f"K={K_val}  solver={solver}  "
                          f"(sigma_max={sigma_max:.5f} sigma_min={sigma_min:.5f})")

                    base_seed = a.seed + xi * 10000 + ni * 1000 + ki * 10 + ri

                    def gen_fn(seed, sigma_max=sigma_max, sigma_min=sigma_min,
                              K_eff=K_eff, solver=solver):
                        return _generate(sigma_max, sigma_min, K_eff, solver,
                                         nsteps, n_particles, seed)

                    t0 = time.perf_counter()
                    res = repeat_and_evaluate_latent(
                        gen_fn, U_test190, Z_test512, U_train190, tau,
                        mean512, Vt, Vb, lambda_tail,
                        n_repeats=n_repeats, seed=base_seed, rbf_bw=RBF_BW)
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
                    save_grid_json(results_path, meta_out, grid_results)

    if not a.smoke_test:
        timing_out = {
            "method": "knn_pflow", "device": "CPU",
            "sample_time_s": timing_mean, "sample_time_std_s": timing_std,
            "sample_time_per_sample_ms": timing_mean / cfg.N_GENERATE * 1000,
            "train_opt_time_h": 0.0,
            "timing_c_max": cfg.KNNPF_TIMING_CMAX, "timing_c_min": cfg.KNNPF_TIMING_CMIN,
            "timing_K": cfg.KNNPF_TIMING_K, "timing_solver": cfg.KNNPF_TIMING_SOLVER,
            "timing_nsteps": cfg.KNNPF_TIMING_NSTEPS,
            "n_warmup": cfg.TIMING_N_WARMUP, "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_knnpflow.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 2c (diffae) complete!")


if __name__ == "__main__":
    main()
