"""
step2d_run_swf.py
==========================
Sliced-Wasserstein Flow (SWF) grid search for the CelebA-HQ DiffAE-latent
generation experiment.
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
from swf_sampling import sample_swf                                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "swf"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 2d (diffae): Sliced-Wasserstein Flow grid search")
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

    nswf_grid = [300]             if a.smoke_test else cfg.SWF_NSWF_GRID
    r_grid    = [64]              if a.smoke_test else cfg.SWF_R_GRID
    eta_grid  = [0.05]            if a.smoke_test else cfg.SWF_ETA_GRID
    init_grid = ["gaussian_full"] if a.smoke_test else cfg.SWF_INIT_GRID
    n_particles = 250 if a.smoke_test else cfg.N_GENERATE
    nswf_cap    = 5   if a.smoke_test else None   # override n_SWF in smoke mode
    n_repeats   = 2   if a.smoke_test else cfg.N_REPEATS

    def _generate(n_swf, R, eta, init, n_particles_, seed):
        U_gen = sample_swf(
            U_train190, mean0, S_eye, S_eye,
            n_particles=n_particles_, num_steps=n_swf, num_projections=R,
            step_size=eta, init=init, nn_dist=NN, seed=seed,
        )
        return np.array(U_gen, dtype=np.float32)

    # ── Timing (CPU) at the representative cell ─────────────────────────────
    timing_mean = timing_std = None
    if not a.smoke_test:
        print(f"\nTiming SWF on CPU (n_SWF={cfg.SWF_TIMING_NSWF}, R={cfg.SWF_TIMING_R}, "
              f"eta={cfg.SWF_TIMING_ETA}, init={cfg.SWF_TIMING_INIT}) ...")

        def _timing_fn():
            return sample_swf(
                U_train190, mean0, S_eye, S_eye,
                n_particles=cfg.N_GENERATE, num_steps=cfg.SWF_TIMING_NSWF,
                num_projections=cfg.SWF_TIMING_R, step_size=cfg.SWF_TIMING_ETA,
                init=cfg.SWF_TIMING_INIT, nn_dist=NN, seed=a.seed)

        timing_mean, timing_std = measure_cpu_time(_timing_fn)
        print(f"  SWF CPU sampling time: {timing_mean:.3f}s +/- {timing_std:.3f}s "
              f"({timing_mean / cfg.N_GENERATE * 1000:.3f} ms/sample)")

    # ── Grid search (GPU / default device) ──────────────────────────────────
    print(f"\n{'='*40}")
    print("Grid search: Sliced-Wasserstein Flow\n")
    total = len(nswf_grid) * len(r_grid) * len(eta_grid) * len(init_grid)
    suffix = "_smoke" if a.smoke_test else ""
    results_path = os.path.join(a.out, f"results_swf{suffix}.json")
    meta_out = {
        "method": "swf",
        "nswf_grid": cfg.SWF_NSWF_GRID, "r_grid": cfg.SWF_R_GRID,
        "eta_grid": cfg.SWF_ETA_GRID, "init_grid": cfg.SWF_INIT_GRID,
        "NN": NN, "proj_k": cfg.PROJ_K,
        "n_per_repeat": cfg.N_GENERATE, "n_repeats": cfg.N_REPEATS,
        "smoke_test": a.smoke_test,
    }
    grid_results = {} if a.overwrite else load_grid_json(results_path)
    if grid_results:
        print(f"  resuming: {len(grid_results)} cell(s) already in {results_path}")
    cell_num = 0
    for si, n_swf in enumerate(nswf_grid):
        n_swf_use = nswf_cap if nswf_cap is not None else n_swf
        for ri, R in enumerate(r_grid):
            for ei, eta in enumerate(eta_grid):
                for ii, init in enumerate(init_grid):
                    cell_num += 1
                    key = f"{si}_{ri}_{ei}_{ii}"
                    if key in grid_results and not a.overwrite:
                        print(f"[{cell_num:3d}/{total}]  n_SWF={n_swf}  R={R}  "
                              f"eta={eta}  init={init}  -- already done, skipping")
                        continue
                    print(f"[{cell_num:3d}/{total}]  n_SWF={n_swf}  R={R}  "
                          f"eta={eta}  init={init}")

                    base_seed = a.seed + si * 10000 + ri * 1000 + ei * 100 + ii

                    def gen_fn(seed, n_swf_use=n_swf_use, R=R, eta=eta, init=init):
                        return _generate(n_swf_use, R, eta, init, n_particles, seed)

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
                        "n_SWF": n_swf_use, "R": R, "eta": eta, "init": init,
                        **res,
                    }
                    save_grid_json(results_path, meta_out, grid_results)

    if not a.smoke_test:
        timing_out = {
            "method": "swf", "device": "CPU",
            "sample_time_s": timing_mean, "sample_time_std_s": timing_std,
            "sample_time_per_sample_ms": timing_mean / cfg.N_GENERATE * 1000,
            "train_opt_time_h": 0.0,
            "timing_n_SWF": cfg.SWF_TIMING_NSWF, "timing_R": cfg.SWF_TIMING_R,
            "timing_eta": cfg.SWF_TIMING_ETA, "timing_init": cfg.SWF_TIMING_INIT,
            "n_warmup": cfg.TIMING_N_WARMUP, "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_swf.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 2d (diffae) complete!")


if __name__ == "__main__":
    main()
