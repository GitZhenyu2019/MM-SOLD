"""
step2f_run_sgbaoab.py
==============================
SG-BAOAB grid search for the CelebA-HQ DiffAE-latent generation experiment.
"""
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import config as cfg                                    # noqa: E402
import tilting_lion as TL                                        # noqa: E402
from common import (                                     # noqa: E402
    repeat_and_evaluate_latent, measure_cpu_time, measure_gpu_time, delta_sigma,
    h_baoab, gamma_baoab, load_grid_json, save_grid_json,
)
from metrics_celeba import compute_tau                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "sgbaoab"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true",
                    help="ignore any cached (lambda, Lambda) per cell")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute cells already in the results JSON (implies "
                         "re-evaluating; combine with --retrain to also refit)")
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache_dir = os.path.join(a.out, "tilting_cache_smoke" if a.smoke_test else "tilting_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("Step 2f (diffae): SG-BAOAB grid search")
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
    print(f"  U_train190 {U_train190.shape}  U_test190 {U_test190.shape}  NN={NN:.5f}")

    tau = compute_tau(U_train190, percentile=cfg.DUP_PERCENTILE)
    print(f"  DupRate tau (top-190, percentile={cfg.DUP_PERCENTILE}) = {tau:.5f}")

    gamma190 = gamma_baoab(U_train190)
    print(f"  gamma (1/sqrt(lambda_max(Sigma_190))) = {gamma190:.5f}")

    cdelta_grid = [0.20] if a.smoke_test else cfg.CDELTA_GRID
    csigma_grid = [0.20] if a.smoke_test else cfg.CSIGMA_GRID
    m_grid      = [2]    if a.smoke_test else cfg.M_GRID
    n_particles = 50     if a.smoke_test else cfg.N_GENERATE
    nsteps      = 20     if a.smoke_test else cfg.BAOAB_NSTEPS
    n_repeats   = 2       if a.smoke_test else cfg.N_REPEATS
    lion_chains = 16     if a.smoke_test else cfg.LION_N_CHAINS
    lion_steps  = 10     if a.smoke_test else cfg.LION_N_STEPS
    lion_iters  = 5      if a.smoke_test else cfg.LION_N_ITERS

    Xj = jnp.asarray(U_train190, dtype=jnp.float32)

    def _fit(delta, sigma, M, seed, cache_path, log_every=50, verbose=False):
        if cache_path and os.path.exists(cache_path) and not a.retrain:
            d = np.load(cache_path)
            return d["lam"], d["Lam"], float(d["opt_time_s"]), {
                "res_mean": float(d["res_mean"]), "res_cov": float(d["res_cov"])}
        t0 = time.perf_counter()
        lam, Lam, hist = TL.estimate_tilting_lion(
            Xj, delta=delta, sigma_s=sigma, M=M,
            h=h_baoab(delta), gamma=gamma190,
            n_steps=lion_steps, n_chains=lion_chains, n_iters=lion_iters,
            lr_rel=cfg.LION_LR_REL, b1=cfg.LION_B1, b2=cfg.LION_B2,
            splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB,
            warm_start=cfg.LION_WARM_START, seed=seed,
            log_every=log_every, verbose=verbose)
        opt_time = time.perf_counter() - t0
        last = hist[-1]
        if cache_path:
            np.savez(cache_path, lam=lam, Lam=Lam, opt_time_s=opt_time,
                     res_mean=last["res_mean"], res_cov=last["res_cov"])
        return lam, Lam, opt_time, {"res_mean": last["res_mean"], "res_cov": last["res_cov"]}

    def _sample(lam, Lam, delta, sigma, M, h, n_particles_, nsteps_, seed):
        q = TL.sample_tilted(
            lam, Lam, Xj, delta=delta, sigma_s=sigma, M=M, h=h, gamma=gamma190,
            n_steps=nsteps_, n_samples=n_particles_, splitting=cfg.BAOAB_SPLITTING,
            score=cfg.BAOAB_SCORE, knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB,
            seed=seed)
        return np.asarray(q, dtype=np.float32)

    # ── Timing at the representative cell: opt. time on GPU, sample time on CPU ──
    timing_sample_mean = timing_sample_std = None
    timing_opt_mean = timing_opt_std = None
    if not a.smoke_test:
        t_delta, t_sigma = delta_sigma(cfg.TIMING_CDELTA, cfg.TIMING_CSIGMA, NN, dsub=cfg.PROJ_K)
        t_h = h_baoab(t_delta)
        print(f"\nTiming SG-BAOAB optimisation on GPU "
              f"(c_delta={cfg.TIMING_CDELTA}, c_sigma={cfg.TIMING_CSIGMA}, M={cfg.TIMING_M}) "
              f"delta={t_delta:.5f} sigma={t_sigma:.5f} h={t_h:.5f} ...")

        def _opt_fn():
            lam, Lam, _hist = TL.estimate_tilting_lion(
                Xj, delta=t_delta, sigma_s=t_sigma, M=cfg.TIMING_M,
                h=t_h, gamma=gamma190, n_steps=cfg.LION_N_STEPS,
                n_chains=cfg.LION_N_CHAINS, n_iters=cfg.LION_N_ITERS,
                lr_rel=cfg.LION_LR_REL, b1=cfg.LION_B1, b2=cfg.LION_B2,
                splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
                knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB,
                warm_start=cfg.LION_WARM_START, seed=a.seed,
                log_every=10 ** 6, verbose=False)
            return lam

        timing_opt_mean, timing_opt_std = measure_gpu_time(_opt_fn, n_warmup=0, n_timed=1)
        print(f"  SG-BAOAB GPU optimisation time (1 run, P={cfg.LION_N_CHAINS} "
              f"L={cfg.LION_N_STEPS} K={cfg.LION_N_ITERS}): {timing_opt_mean:.1f}s")

        t_lam, t_Lam, _, _ = _fit(t_delta, t_sigma, cfg.TIMING_M, a.seed,
                                  os.path.join(cache_dir, "timing_cell.npz"))
        print("Timing SG-BAOAB sampling on CPU ...")

        def _timing_fn():
            return _sample(t_lam, t_Lam, t_delta, t_sigma, cfg.TIMING_M, t_h,
                           cfg.N_GENERATE, cfg.BAOAB_NSTEPS, a.seed)

        timing_sample_mean, timing_sample_std = measure_cpu_time(_timing_fn)
        print(f"  SG-BAOAB CPU sampling time: {timing_sample_mean:.3f}s +/- "
              f"{timing_sample_std:.3f}s "
              f"({timing_sample_mean / cfg.N_GENERATE * 1000:.3f} ms/sample)")

    # ── Grid search (GPU / default device) ──────────────────────────────────
    print(f"\n{'='*40}")
    print("Grid search: SG-BAOAB\n")
    total = len(cdelta_grid) * len(csigma_grid) * len(m_grid)
    suffix = "_smoke" if a.smoke_test else ""
    results_path = os.path.join(a.out, f"results_sgbaoab{suffix}.json")
    grid_results = {} if a.overwrite else load_grid_json(results_path)
    if grid_results:
        print(f"  resuming: {len(grid_results)} cell(s) already in {results_path}")
    cell_num = 0
    opt_times = [cell["opt_time_s"] for cell in grid_results.values()]
    for ci, c_delta in enumerate(cdelta_grid):
        for si, c_sigma in enumerate(csigma_grid):
            for mi, M in enumerate(m_grid):
                cell_num += 1
                key = f"{ci}_{si}_{mi}"
                if key in grid_results and not a.overwrite:
                    print(f"[{cell_num:3d}/{total}]  c_delta={c_delta}  c_sigma={c_sigma}  "
                          f"M={M}  -- already done, skipping")
                    continue
                delta, sigma = delta_sigma(c_delta, c_sigma, NN, dsub=cfg.PROJ_K)
                h = h_baoab(delta)
                print(f"[{cell_num:3d}/{total}]  c_delta={c_delta}  c_sigma={c_sigma}  "
                      f"M={M}  (delta={delta:.5f} sigma={sigma:.5f} h={h:.5f})")

                base_seed = a.seed + ci * 10000 + si * 100 + mi
                # Value-based (not positional ci/si/mi) cache name so a cache
                # entry can never be silently misattributed to a different
                # (c_delta, c_sigma, M) after the grid's value list changes.
                cache_path = os.path.join(cache_dir, f"cell_cd{c_delta:.2f}_cs{c_sigma:.2f}_M{M}.npz")
                lam, Lam, opt_time, fit_diag = _fit(delta, sigma, M, base_seed, cache_path)
                opt_times.append(opt_time)

                mu_star, Sigma_star = TL.moment_targets(np.asarray(U_train190), delta)

                def gen_fn(seed, lam=lam, Lam=Lam, delta=delta, sigma=sigma, M=M, h=h):
                    return _sample(lam, Lam, delta, sigma, M, h, n_particles, nsteps, seed)

                t0 = time.perf_counter()
                res = repeat_and_evaluate_latent(
                    gen_fn, U_test190, Z_test512, U_train190, tau,
                    mean512, Vt, Vb, lambda_tail,
                    n_repeats=n_repeats, seed=base_seed + 500000,
                    rbf_bw=RBF_BW, return_last_gen=True)
                elapsed = time.perf_counter() - t0

                last_q = res.pop("last_U_gen")
                res_mean, res_cov = TL.moment_residuals(last_q, mu_star, Sigma_star)
                floor = TL.res_cov_floor(Sigma_star, max(len(last_q), 1))

                print(f"  KID={res['kid_mean']:.4f}+/-{res['kid_std']:.4f}  "
                      f"Recall={res['recall_mean']:.3f}+/-{res['recall_std']:.3f}  "
                      f"DupRate={res['duprate_mean']:.3f}+/-{res['duprate_std']:.3f}  "
                      f"moment_res(mean={res_mean:.4f}, cov={res_cov:.4f}, floor={floor:.4f})  "
                      f"[Lion final res_cov={fit_diag['res_cov']:.4f}]  "
                      f"(opt {opt_time:.1f}s, eval {elapsed:.1f}s)")

                grid_results[key] = {
                    "c_delta": c_delta, "c_sigma": c_sigma, "M": M,
                    "delta": delta, "sigma": sigma, "h": h, "gamma": gamma190,
                    "opt_time_s": opt_time,
                    "lion_final_res_mean": fit_diag["res_mean"],
                    "lion_final_res_cov": fit_diag["res_cov"],
                    "moment_res_mean": res_mean, "moment_res_cov": res_cov,
                    "moment_res_cov_floor": floor,
                    **res,
                }
                grid_meta = {
                    "method": "sg_baoab",
                    "cdelta_grid": cfg.CDELTA_GRID, "csigma_grid": cfg.CSIGMA_GRID,
                    "m_grid": cfg.M_GRID,
                    "NN": NN, "proj_k": cfg.PROJ_K, "gamma": gamma190,
                    "baoab_nsteps": cfg.BAOAB_NSTEPS, "baoab_splitting": cfg.BAOAB_SPLITTING,
                    "score": cfg.BAOAB_SCORE, "knn_K": cfg.KNN_K_BAOAB, "knn_L": cfg.KNN_L_BAOAB,
                    "lion_n_chains": lion_chains, "lion_n_steps": lion_steps,
                    "lion_n_iters": lion_iters,
                    "lion_lr_rel": cfg.LION_LR_REL, "lion_b1": cfg.LION_B1,
                    "lion_b2": cfg.LION_B2, "lion_warm_start": cfg.LION_WARM_START,
                    "n_per_repeat": cfg.N_GENERATE, "n_repeats": cfg.N_REPEATS,
                    "mean_opt_time_s_over_grid": float(np.mean(opt_times)) if opt_times else None,
                    "smoke_test": a.smoke_test,
                }
                save_grid_json(results_path, grid_meta, grid_results)

    if not a.smoke_test:
        timing_out = {
            "method": "sg_baoab",
            "opt_device": "GPU", "sample_device": "CPU",
            "opt_time_s": timing_opt_mean, "opt_time_std_s": timing_opt_std,
            "sample_time_s": timing_sample_mean, "sample_time_std_s": timing_sample_std,
            "sample_time_per_sample_ms": timing_sample_mean / cfg.N_GENERATE * 1000,
            "timing_c_delta": cfg.TIMING_CDELTA, "timing_c_sigma": cfg.TIMING_CSIGMA,
            "timing_M": cfg.TIMING_M, "n_warmup": cfg.TIMING_N_WARMUP,
            "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_sgbaoab.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 2f (diffae) complete!")


if __name__ == "__main__":
    main()
