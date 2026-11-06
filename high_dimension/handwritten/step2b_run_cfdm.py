"""
step2b_run_cfdm.py
=======================
sigma-CFDM grid search for the digit-8 generation experiment.
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
    repeat_and_evaluate, measure_cpu_time, delta_sigma,
    load_grid_json, save_grid_json, seamless_grid, find_nn, save_nn_vis,
)
from metrics import extract_features, compute_tau           # noqa: E402
from cfdm_sampling import sample_cfdm                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR, "subspace"))
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "cfdm"))
    ap.add_argument("--nrae_ckpt", default=cfg.NRAE_CKPT)
    ap.add_argument("--clf_ckpt", default=cfg.CLF_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute (c_sigma, M) cells already in the results JSON")
    ap.add_argument("--make_grids", action="store_true",
                    help="also save an 8x8 sample grid + NN visualisation PNG "
                         "for every DISTINCT (c_sigma, M) cell (c_delta doesn't "
                         "affect sigma-CFDM, so no duplicate images are made)")
    ap.add_argument("--vis_grid", type=int, default=8)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 2b: sigma-CFDM grid search")
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

    mean0   = jnp.zeros(d, dtype=jnp.float32)
    S_eye   = jnp.eye(d, dtype=jnp.float32)

    csigma_grid = [0.20] if a.smoke_test else cfg.CSIGMA_GRID
    m_grid      = [2]    if a.smoke_test else cfg.M_GRID
    n_particles = 8      if a.smoke_test else cfg.N_PER_REPEAT
    nsteps      = 20     if a.smoke_test else cfg.CFDM_NSTEPS
    n_repeats   = 2       if a.smoke_test else cfg.N_REPEATS

    def _generate(sigma, M, nsteps_, n_particles_, seed):
        U_gen = sample_cfdm(
            U_train8, mean0, S_eye, S_eye,
            n_particles=n_particles_, sigma=sigma, M=M,
            shared_noise=cfg.CFDM_SHARED_NOISE, nsteps=nsteps_,
            batch_size=min(cfg.CFDM_BATCH, n_particles_), seed=seed,
        )
        feats_gen = decode_and_extract(U_gen, mean_8, V, nrae_model, nrae_params,
                                       n_dct, clf_params)
        return U_gen, feats_gen

    # ── Timing (CPU) at the representative cell (only c_sigma, M matter) ────
    timing_mean = timing_std = None
    if not a.smoke_test:
        _, t_sigma = delta_sigma(cfg.TIMING_CDELTA, cfg.TIMING_CSIGMA, NN8, dsub=cfg.PROJ_K)
        print(f"\nTiming sigma-CFDM on CPU (c_sigma={cfg.TIMING_CSIGMA}, M={cfg.TIMING_M}) "
              f"sigma={t_sigma:.5f} ...")

        def _timing_fn():
            return sample_cfdm(
                U_train8, mean0, S_eye, S_eye,
                n_particles=cfg.N_PER_REPEAT, sigma=t_sigma, M=cfg.TIMING_M,
                shared_noise=cfg.CFDM_SHARED_NOISE, nsteps=cfg.CFDM_NSTEPS,
                batch_size=min(cfg.CFDM_BATCH, cfg.N_PER_REPEAT), seed=a.seed)

        timing_mean, timing_std = measure_cpu_time(_timing_fn)
        print(f"  sigma-CFDM CPU sampling time: {timing_mean:.3f}s +/- {timing_std:.3f}s "
              f"({timing_mean / cfg.N_PER_REPEAT * 1000:.3f} ms/sample)")

    # ── Grid search: only (c_sigma, M) matter, computed once each ──────────
    print(f"\n{'='*40}")
    print("Grid search: sigma-CFDM (c_delta duplicated across identical cells)\n")
    total = len(csigma_grid) * len(m_grid)
    cdelta_grid = [0.20] if a.smoke_test else cfg.CDELTA_GRID
    suffix = "_smoke" if a.smoke_test else ""
    results_path = os.path.join(a.out, f"results_cfdm{suffix}.json")
    grid_meta = {
        "method": "cfdm",
        "cdelta_grid": cfg.CDELTA_GRID, "csigma_grid": cfg.CSIGMA_GRID, "m_grid": cfg.M_GRID,
        "note": "c_delta does not affect sigma-CFDM; cells sharing (c_sigma, M) are duplicates",
        "NN8": NN8, "proj_k": cfg.PROJ_K,
        "cfdm_nsteps": cfg.CFDM_NSTEPS,
        "n_per_repeat": cfg.N_PER_REPEAT, "n_repeats": cfg.N_REPEATS,
        "smoke_test": a.smoke_test,
    }
    grid_results = {} if a.overwrite else load_grid_json(results_path)
    if grid_results:
        n_done = len({(k.split("_")[1], k.split("_")[2]) for k in grid_results})
        print(f"  resuming: {n_done} (c_sigma, M) cell(s) already in {results_path}")
    grids_dir = os.path.join(a.out, "grids")
    if a.make_grids:
        os.makedirs(grids_dir, exist_ok=True)

    cell_num = 0
    for si, c_sigma in enumerate(csigma_grid):
        for mi, M in enumerate(m_grid):
            cell_num += 1
            grid_png = os.path.join(grids_dir, f"grid_{si}_{mi}.png")
            need_metrics = (f"0_{si}_{mi}" not in grid_results) or a.overwrite
            need_image = a.make_grids and (a.overwrite or not os.path.exists(grid_png))
            if not need_metrics and not need_image:
                print(f"[{cell_num:3d}/{total}]  c_sigma={c_sigma}  M={M}  -- already done, skipping")
                continue
            _, sigma = delta_sigma(cfg.TIMING_CDELTA, c_sigma, NN8, dsub=cfg.PROJ_K)
            print(f"[{cell_num:3d}/{total}]  c_sigma={c_sigma}  M={M}  (sigma={sigma:.5f})")

            base_seed = a.seed + si * 100 + mi

            if need_metrics:
                def gen_fn(seed, sigma=sigma, M=M):
                    return _generate(sigma, M, nsteps, n_particles, seed)

                t0 = time.perf_counter()
                res = repeat_and_evaluate(gen_fn, feats_test, U_train8, tau,
                                          n_repeats=n_repeats, seed=base_seed)
                elapsed = time.perf_counter() - t0
                print(f"  KID={res['kid_mean']:.4f}+/-{res['kid_std']:.4f}  "
                      f"Recall={res['recall_mean']:.3f}+/-{res['recall_std']:.3f}  "
                      f"DupRate={res['duprate_mean']:.3f}+/-{res['duprate_std']:.3f}  "
                      f"({elapsed:.1f}s)")

                for ci, c_delta in enumerate(cdelta_grid):
                    grid_results[f"{ci}_{si}_{mi}"] = {
                        "c_delta": c_delta, "c_sigma": c_sigma, "M": M, "sigma": sigma, **res,
                    }
                save_grid_json(results_path, grid_meta, grid_results)

            if need_image:
                n_vis = a.vis_grid * a.vis_grid
                U_vis = sample_cfdm(
                    U_train8, mean0, S_eye, S_eye, n_particles=n_vis, sigma=sigma, M=M,
                    shared_noise=cfg.CFDM_SHARED_NOISE, nsteps=nsteps,
                    batch_size=min(cfg.CFDM_BATCH, n_vis), seed=base_seed + 999999)
                imgs_vis = decode_from_subspace(U_vis, mean_8, V, nrae_model, nrae_params, n_dct)
                seamless_grid(imgs_vis, grid_png, G=a.vis_grid)
                n_nn = min(32, n_vis)
                nn_idx, _ = find_nn(U_vis[:n_nn], U_train8)
                imgs_nn_q = decode_from_subspace(U_vis[:n_nn], mean_8, V, nrae_model,
                                                 nrae_params, n_dct)
                imgs_nn_r = decode_from_subspace(U_train8[nn_idx], mean_8, V, nrae_model,
                                                 nrae_params, n_dct)
                save_nn_vis(imgs_nn_q, imgs_nn_r, os.path.join(grids_dir, f"nn_{si}_{mi}.png"))
                print(f"  saved grid/nn images for (c_sigma={c_sigma}, M={M})")

    if not a.smoke_test:
        timing_out = {
            "method": "cfdm", "device": "CPU",
            "sample_time_s": timing_mean, "sample_time_std_s": timing_std,
            "sample_time_per_sample_ms": timing_mean / cfg.N_PER_REPEAT * 1000,
            "train_opt_time_h": 0.0,
            "timing_c_sigma": cfg.TIMING_CSIGMA, "timing_M": cfg.TIMING_M,
            "n_warmup": cfg.TIMING_N_WARMUP, "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_cfdm.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 2b complete!")


if __name__ == "__main__":
    main()
