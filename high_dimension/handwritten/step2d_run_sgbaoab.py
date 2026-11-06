"""
step2d_run_sgbaoab.py
==========================
SG-BAOAB grid search for the digit-8 generation experiment.
"""
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import config as cfg                                    # noqa: E402
import tilting_lion as TL                                   # noqa: E402
from common import (                                     # noqa: E402
    load_nrae, load_clf, decode_and_extract, decode_from_subspace,
    repeat_and_evaluate, measure_cpu_time, measure_gpu_time, delta_sigma,
    h_baoab, gamma_baoab, load_grid_json, save_grid_json,
    seamless_grid, find_nn, save_nn_vis,
)
from metrics import extract_features, compute_tau           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR, "subspace"))
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "sgbaoab"))
    ap.add_argument("--nrae_ckpt", default=cfg.NRAE_CKPT)
    ap.add_argument("--clf_ckpt", default=cfg.CLF_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true",
                    help="ignore any cached (lambda, Lambda) per cell")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute cells already in the results JSON (implies "
                         "re-evaluating; combine with --retrain to also refit)")
    ap.add_argument("--make_grids", action="store_true",
                    help="also save an 8x8 sample grid + NN visualisation PNG "
                         "for every grid cell (reuses each cell's cached "
                         "(lambda, Lambda), so this does NOT repeat any "
                         "GPU-side Lion optimisation for cells already fit)")
    ap.add_argument("--vis_grid", type=int, default=8)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache_dir = os.path.join(a.out, "tilting_cache_smoke" if a.smoke_test else "tilting_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("Step 2d: SG-BAOAB grid search")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    U_train8 = np.load(os.path.join(a.subspace_dir, "U_train8.npy"))
    mean_8   = np.load(os.path.join(a.subspace_dir, "mean_8.npy"))
    V        = np.load(os.path.join(a.subspace_dir, "proj_V.npy"))
    imgs_test8 = np.load(os.path.join(a.subspace_dir, "imgs_test8.npy"))
    meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    NN8 = meta["NN8"]
    print(f"  U_train8 {U_train8.shape}  imgs_test8 {imgs_test8.shape}  NN8={NN8:.5f}")

    nrae_model, nrae_params, n_dct = load_nrae(a.nrae_ckpt, seed=0)
    clf_params = load_clf(a.clf_ckpt)

    print("\nExtracting test CNN features ...")
    feats_test = extract_features(imgs_test8, clf_params, batch_size=64)
    tau = compute_tau(U_train8, percentile=1)
    print(f"  feats_test {feats_test.shape}  DupRate tau (top-40) = {tau:.5f}")

    gamma8 = gamma_baoab(U_train8)
    print(f"  gamma (1/sqrt(lambda_max(Sigma_8))) = {gamma8:.5f}")

    cdelta_grid = [0.20] if a.smoke_test else cfg.CDELTA_GRID
    csigma_grid = [0.20] if a.smoke_test else cfg.CSIGMA_GRID
    m_grid      = [2]    if a.smoke_test else cfg.M_GRID
    n_particles = 8      if a.smoke_test else cfg.N_PER_REPEAT
    nsteps      = 20     if a.smoke_test else cfg.BAOAB_NSTEPS
    n_repeats   = 2       if a.smoke_test else cfg.N_REPEATS
    lion_chains = 8      if a.smoke_test else cfg.LION_N_CHAINS
    lion_steps  = 10     if a.smoke_test else cfg.LION_N_STEPS
    lion_iters  = 5      if a.smoke_test else cfg.LION_N_ITERS

    Xj = jnp.asarray(U_train8, dtype=jnp.float32)

    def _fit(delta, sigma, M, seed, cache_path, log_every=100, verbose=False):
        if cache_path and os.path.exists(cache_path) and not a.retrain:
            d = np.load(cache_path)
            return d["lam"], d["Lam"], float(d["opt_time_s"]), {
                "res_mean": float(d["res_mean"]), "res_cov": float(d["res_cov"])}
        t0 = time.perf_counter()
        lam, Lam, hist = TL.estimate_tilting_lion(
            Xj, delta=delta, sigma_s=sigma, M=M,
            h=h_baoab(delta), gamma=gamma8,
            n_steps=lion_steps, n_chains=lion_chains, n_iters=lion_iters,
            lr_rel=cfg.LION_LR_REL, b1=cfg.LION_B1, b2=cfg.LION_B2,
            splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
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
            lam, Lam, Xj, delta=delta, sigma_s=sigma, M=M, h=h, gamma=gamma8,
            n_steps=nsteps_, n_samples=n_particles_, splitting=cfg.BAOAB_SPLITTING,
            score=cfg.BAOAB_SCORE, seed=seed)
        return np.asarray(q, dtype=np.float32)

    # ── Timing at the representative cell: opt. time on GPU, sample time on CPU ──
    timing_sample_mean = timing_sample_std = None
    timing_opt_mean = timing_opt_std = None
    if not a.smoke_test:
        t_delta, t_sigma = delta_sigma(cfg.TIMING_CDELTA, cfg.TIMING_CSIGMA, NN8, dsub=cfg.PROJ_K)
        t_h = h_baoab(t_delta)
        print(f"\nTiming SG-BAOAB optimisation on GPU "
              f"(c_delta={cfg.TIMING_CDELTA}, c_sigma={cfg.TIMING_CSIGMA}, M={cfg.TIMING_M}) "
              f"delta={t_delta:.5f} sigma={t_sigma:.5f} h={t_h:.5f} ...")

        def _opt_fn():
            lam, Lam, _hist = TL.estimate_tilting_lion(
                Xj, delta=t_delta, sigma_s=t_sigma, M=cfg.TIMING_M,
                h=t_h, gamma=gamma8, n_steps=cfg.LION_N_STEPS,
                n_chains=cfg.LION_N_CHAINS, n_iters=cfg.LION_N_ITERS,
                lr_rel=cfg.LION_LR_REL, b1=cfg.LION_B1, b2=cfg.LION_B2,
                splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
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
                           cfg.N_PER_REPEAT, cfg.BAOAB_NSTEPS, a.seed)

        timing_sample_mean, timing_sample_std = measure_cpu_time(_timing_fn)
        print(f"  SG-BAOAB CPU sampling time: {timing_sample_mean:.3f}s +/- "
              f"{timing_sample_std:.3f}s "
              f"({timing_sample_mean / cfg.N_PER_REPEAT * 1000:.3f} ms/sample)")

    # ── Grid search (GPU / default device) ──────────────────────────────────
    print(f"\n{'='*40}")
    print("Grid search: SG-BAOAB\n")
    total = len(cdelta_grid) * len(csigma_grid) * len(m_grid)
    suffix = "_smoke" if a.smoke_test else ""
    results_path = os.path.join(a.out, f"results_sgbaoab{suffix}.json")
    grid_results = {} if a.overwrite else load_grid_json(results_path)
    if grid_results:
        print(f"  resuming: {len(grid_results)} cell(s) already in {results_path}")
    grids_dir = os.path.join(a.out, "grids")
    if a.make_grids:
        os.makedirs(grids_dir, exist_ok=True)
    cell_num = 0
    opt_times = [cell["opt_time_s"] for cell in grid_results.values()]
    for ci, c_delta in enumerate(cdelta_grid):
        for si, c_sigma in enumerate(csigma_grid):
            for mi, M in enumerate(m_grid):
                cell_num += 1
                key = f"{ci}_{si}_{mi}"
                grid_png = os.path.join(grids_dir, f"grid_{key}.png")
                need_metrics = (key not in grid_results) or a.overwrite
                need_image = a.make_grids and (a.overwrite or not os.path.exists(grid_png))
                if not need_metrics and not need_image:
                    print(f"[{cell_num:3d}/{total}]  c_delta={c_delta}  c_sigma={c_sigma}  "
                          f"M={M}  -- already done, skipping")
                    continue
                delta, sigma = delta_sigma(c_delta, c_sigma, NN8, dsub=cfg.PROJ_K)
                h = h_baoab(delta)
                print(f"[{cell_num:3d}/{total}]  c_delta={c_delta}  c_sigma={c_sigma}  "
                      f"M={M}  (delta={delta:.5f} sigma={sigma:.5f} h={h:.5f})")

                base_seed = a.seed + ci * 10000 + si * 100 + mi
                cache_path = os.path.join(cache_dir, f"cell_cd{c_delta:.2f}_cs{c_sigma:.2f}_M{M}.npz")
                lam, Lam, opt_time, fit_diag = _fit(delta, sigma, M, base_seed, cache_path)
                if need_metrics:
                    opt_times.append(opt_time)

                if need_metrics:
                    mu_star, Sigma_star = TL.moment_targets(np.asarray(U_train8), delta)

                    def gen_fn(seed, lam=lam, Lam=Lam, delta=delta, sigma=sigma, M=M, h=h):
                        U_gen = _sample(lam, Lam, delta, sigma, M, h,
                                        n_particles, nsteps, seed)
                        feats_gen = decode_and_extract(U_gen, mean_8, V, nrae_model,
                                                       nrae_params, n_dct, clf_params)
                        return U_gen, feats_gen

                    t0 = time.perf_counter()
                    res = repeat_and_evaluate(gen_fn, feats_test, U_train8, tau,
                                              n_repeats=n_repeats, seed=base_seed + 500000,
                                              return_last_gen=True)
                    elapsed = time.perf_counter() - t0

                    # moment-residual diagnostic reusing the LAST repeat's samples
                    # (no redundant regeneration -- repeat_and_evaluate already ran it)
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
                        "delta": delta, "sigma": sigma, "h": h, "gamma": gamma8,
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
                        "NN8": NN8, "proj_k": cfg.PROJ_K, "gamma": gamma8,
                        "baoab_nsteps": cfg.BAOAB_NSTEPS, "baoab_splitting": cfg.BAOAB_SPLITTING,
                        "score": cfg.BAOAB_SCORE,
                        "lion_n_chains": lion_chains, "lion_n_steps": lion_steps,
                        "lion_n_iters": lion_iters,
                        "lion_lr_rel": cfg.LION_LR_REL, "lion_b1": cfg.LION_B1,
                        "lion_b2": cfg.LION_B2, "lion_warm_start": cfg.LION_WARM_START,
                        "n_per_repeat": cfg.N_PER_REPEAT, "n_repeats": cfg.N_REPEATS,
                        "mean_opt_time_s_over_grid": float(np.mean(opt_times)) if opt_times else None,
                        "smoke_test": a.smoke_test,
                    }
                    save_grid_json(results_path, grid_meta, grid_results)

                if need_image:
                    n_vis = a.vis_grid * a.vis_grid
                    U_vis = _sample(lam, Lam, delta, sigma, M, h, n_vis, nsteps,
                                    base_seed + 999999)
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
            "method": "sg_baoab",
            "opt_device": "GPU", "sample_device": "CPU",
            "opt_time_s": timing_opt_mean, "opt_time_std_s": timing_opt_std,
            "sample_time_s": timing_sample_mean, "sample_time_std_s": timing_sample_std,
            "sample_time_per_sample_ms": timing_sample_mean / cfg.N_PER_REPEAT * 1000,
            "timing_c_delta": cfg.TIMING_CDELTA, "timing_c_sigma": cfg.TIMING_CSIGMA,
            "timing_M": cfg.TIMING_M, "n_warmup": cfg.TIMING_N_WARMUP,
            "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_sgbaoab.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 2d complete!")


if __name__ == "__main__":
    main()
