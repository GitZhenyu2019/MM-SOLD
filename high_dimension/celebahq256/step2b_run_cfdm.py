"""
step2b_run_cfdm.py
==========================
sigma-CFDM grid search for the CelebA-HQ DiffAE-latent generation experiment.
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
    repeat_and_evaluate_latent, measure_cpu_time, delta_sigma,
    load_grid_json, save_grid_json,
)
from metrics_celeba import compute_tau                           # noqa: E402
from cfdm_sampling import sample_cfdm_knn_remainder                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "cfdm"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 2b (diffae): sigma-CFDM grid search")
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
    d = U_train190.shape[1]
    print(f"  U_train190 {U_train190.shape}  U_test190 {U_test190.shape}  NN={NN:.5f}")

    tau = compute_tau(U_train190, percentile=cfg.DUP_PERCENTILE)
    print(f"  DupRate tau (top-190, percentile={cfg.DUP_PERCENTILE}) = {tau:.5f}")

    mean0 = jnp.zeros(d, dtype=jnp.float32)
    S_eye = jnp.eye(d, dtype=jnp.float32)

    cdelta_grid = [0.20] if a.smoke_test else cfg.CDELTA_GRID
    csigma_grid = [0.20] if a.smoke_test else cfg.CSIGMA_GRID
    m_grid      = [2]    if a.smoke_test else cfg.M_GRID
    n_particles = 250   if a.smoke_test else cfg.N_GENERATE
    nsteps      = 20    if a.smoke_test else cfg.CFDM_NSTEPS
    n_repeats   = 2     if a.smoke_test else cfg.N_REPEATS

    def _generate(sigma, M, nsteps_, n_particles_, seed):
        U_gen = sample_cfdm_knn_remainder(
            U_train190, mean0, S_eye, S_eye, n_particles_, sigma, M,
            knn_K=cfg.KNN_K_SMOOTH, knn_L=cfg.KNN_L_SMOOTH,
            nsteps=nsteps_, batch_size=min(cfg.CFDM_BATCH, n_particles_), seed=seed)
        return np.array(U_gen, dtype=np.float32)

    # ── Timing (CPU) at the representative cell ─────────────────────────────
    timing_mean = timing_std = None
    if not a.smoke_test:
        _, t_sigma = delta_sigma(cfg.TIMING_CDELTA, cfg.TIMING_CSIGMA, NN, dsub=cfg.PROJ_K)
        print(f"\nTiming sigma-CFDM on CPU (c_sigma={cfg.TIMING_CSIGMA}, M={cfg.TIMING_M}) "
              f"sigma={t_sigma:.5f} ...")

        def _timing_fn():
            return _generate(t_sigma, cfg.TIMING_M, cfg.CFDM_NSTEPS, cfg.N_GENERATE, a.seed)

        timing_mean, timing_std = measure_cpu_time(_timing_fn)
        print(f"  sigma-CFDM CPU sampling time: {timing_mean:.3f}s +/- {timing_std:.3f}s "
              f"({timing_mean / cfg.N_GENERATE * 1000:.3f} ms/sample)")

    # ── Grid search: compute once per DISTINCT (c_sigma, M), replicate across
    #    c_delta (sigma-CFDM's ODE does not depend on c_delta at all) ────────
    print(f"\n{'='*40}")
    print("Grid search: sigma-CFDM\n")
    total_distinct = len(csigma_grid) * len(m_grid)
    total = len(cdelta_grid) * total_distinct
    suffix = "_smoke" if a.smoke_test else ""
    results_path = os.path.join(a.out, f"results_cfdm{suffix}.json")
    meta_out = {
        "method": "cfdm",
        "cdelta_grid": cfg.CDELTA_GRID, "csigma_grid": cfg.CSIGMA_GRID, "m_grid": cfg.M_GRID,
        "NN": NN, "proj_k": cfg.PROJ_K,
        "cfdm_nsteps": cfg.CFDM_NSTEPS, "knn_K": cfg.KNN_K_SMOOTH, "knn_L": cfg.KNN_L_SMOOTH,
        "n_per_repeat": cfg.N_GENERATE, "n_repeats": cfg.N_REPEATS,
        "note": "sigma-CFDM's ODE does not depend on c_delta; every c_delta "
                "at a given (c_sigma, M) shares the SAME computed result.",
        "smoke_test": a.smoke_test,
    }
    grid_results = {} if a.overwrite else load_grid_json(results_path)
    if grid_results:
        print(f"  resuming: {len(grid_results)} cell(s) already in {results_path}")
    distinct_num = 0
    for si, c_sigma in enumerate(csigma_grid):
        for mi, M in enumerate(m_grid):
            distinct_num += 1
            # find (or compute) the shared result for this (c_sigma, M)
            rep_key = f"0_{si}_{mi}"
            if rep_key in grid_results and not a.overwrite:
                print(f"[{distinct_num:3d}/{total_distinct}]  c_sigma={c_sigma}  M={M}  "
                      f"-- already done, skipping (and all c_delta duplicates)")
                shared_res = None
            else:
                _, sigma = delta_sigma(0.5, c_sigma, NN, dsub=cfg.PROJ_K)  # c_delta unused
                print(f"[{distinct_num:3d}/{total_distinct}]  c_sigma={c_sigma}  M={M}  "
                      f"(sigma={sigma:.5f})")
                base_seed = a.seed + si * 100 + mi

                def gen_fn(seed, sigma=sigma, M=M):
                    return _generate(sigma, M, nsteps, n_particles, seed)

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
                shared_res = {"c_sigma": c_sigma, "M": M, "sigma": sigma, **res}

            for ci, c_delta in enumerate(cdelta_grid):
                key = f"{ci}_{si}_{mi}"
                if key in grid_results and not a.overwrite:
                    continue
                if shared_res is None:
                    shared_res = grid_results[rep_key]
                grid_results[key] = {**shared_res, "c_delta": c_delta}
            save_grid_json(results_path, meta_out, grid_results)

    if not a.smoke_test:
        timing_out = {
            "method": "cfdm", "device": "CPU",
            "sample_time_s": timing_mean, "sample_time_std_s": timing_std,
            "sample_time_per_sample_ms": timing_mean / cfg.N_GENERATE * 1000,
            "train_opt_time_h": 0.0,
            "timing_c_delta": cfg.TIMING_CDELTA, "timing_c_sigma": cfg.TIMING_CSIGMA,
            "timing_M": cfg.TIMING_M, "n_warmup": cfg.TIMING_N_WARMUP,
            "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_cfdm.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 2b (diffae) complete!")


if __name__ == "__main__":
    main()
