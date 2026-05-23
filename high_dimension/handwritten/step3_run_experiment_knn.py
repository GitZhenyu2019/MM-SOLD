"""
step3_run_experiment_knn.py
===========================
Grid-search experiment using the KNN+remainder score estimator for
both MM-SOLD and σ-CFDM. 
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, LATENT_DIR, CKPT_DIR,
    LATENT_DIM, ENC1_HIDDEN, DEC_HIDDEN, N_DCT, UNET_BASE_CH,
    NRAE_CKPT, CLF_CKPT,
    SIGMA_GRID,
    SOLD_NSTEPS, SOLD_H, SOLD_SIGMA_GMM, SOLD_DISCRETIZ,
    SOLD_K_WHITEN,
    CFDM_NSTEPS, CFDM_BATCH,
    DDIM_STEPS, DDPM_T_STEPS, DDPM_HIDDEN, DDPM_N_LAYERS,
    N_GENERATE, N_EVAL, N_BOOTSTRAP,
    KID_DEGREE, DUP_PERCENTILE,
    TIMING_SIGMA,
)

# M_GRID for this experiment (extended to 64)
M_GRID_KNN = [2, 4, 8, 16, 32, 64]
TIMING_M_KNN = 2   # timing measured at (σ=TIMING_SIGMA, M=TIMING_M_KNN)

from nrae_model    import make_nrae_state, decode_latents_nrae, load_params as nrae_load
from classifier_model import load_params as clf_load
from ddpm_model    import make_cosine_schedule, ddim_sample, load_ddpm  # noqa: F401
from cfdm_sampling import sample_cfdm_knn_remainder, compute_whitening_stats
from metrics       import extract_features, compute_tau, bootstrap_metrics
from whitening_utils import (
    compute_sample_mean_cov, symmetric_matrix_sqrt_and_invsqrt,
    whiten, unwhiten,
)
from sampling_algo import sample_class_overdamped_manifold_knn


# ─── helpers (identical to step3_run_experiment.py) ──────────────────────────

def _load_nrae(ckpt_path: str, seed: int = 0):
    params_np, info = nrae_load(ckpt_path)
    latent_dim   = info.get("latent_dim",   LATENT_DIM)
    enc1_hidden  = info.get("enc1_hidden",  ENC1_HIDDEN)
    dec_hidden   = info.get("dec_hidden",   DEC_HIDDEN)
    n_dct        = info.get("n_dct",        N_DCT)
    unet_base_ch = info.get("unet_base_ch", UNET_BASE_CH)
    rng = jax.random.PRNGKey(seed)
    model, _ = make_nrae_state(
        rng, latent_dim=latent_dim, enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch)
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    return model, params, n_dct


def _decode(model, params, Z: np.ndarray, n_dct: int) -> np.ndarray:
    imgs = decode_latents_nrae(model, params, Z, batch_size=32)
    return np.clip(np.array(imgs), 0.0, 1.0)


def _whiten_stats(Z_class: np.ndarray, k: int):
    Z_j = jnp.asarray(Z_class, dtype=jnp.float32)
    mean_cls, cov_cls = compute_sample_mean_cov(Z_j)
    S_sqrt, S_invsqrt, _ = symmetric_matrix_sqrt_and_invsqrt(
        cov_cls, eps=1e-5, k=k)
    return mean_cls, S_sqrt, S_invsqrt


def _measure_cpu_time(fn, *args, n_warmup=1, n_timed=3, **kwargs):
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    knn_K = args.knn_K
    knn_L = args.knn_L
    M_GRID_run = [int(x) for x in args.m_grid.split(",")]

    print("=" * 60)
    print("Step 3 [KNN+remainder]: MM-SOLD / σ-CFDM / DDPM")
    print("=" * 60)
    print(f"JAX backend  : {jax.default_backend()}  devices : {jax.devices()}")
    print(f"σ grid       : {SIGMA_GRID}")
    print(f"M grid       : {M_GRID_run}")
    print(f"KNN K={knn_K}  L={knn_L}")
    print(f"Total cells  : {len(SIGMA_GRID) * len(M_GRID_run)} per method")
    print(f"Results dir  : {args.results_dir}\n")

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    Z_train   = np.load(os.path.join(args.latent_dir, "Z_train.npy"))
    Z_test    = np.load(os.path.join(args.latent_dir, "Z_test.npy"))
    imgs_test = np.load(os.path.join(args.data_dir,   "images_test.npy"))
    print(f"Loaded  Z_train={Z_train.shape}  Z_test={Z_test.shape}  "
          f"imgs_test={imgs_test.shape}")

    # ── Load NRAE ──────────────────────────────────────────────────────────
    print(f"\nLoading NRAE: {args.nrae_ckpt}")
    nrae_model, nrae_params, n_dct = _load_nrae(args.nrae_ckpt, seed=args.seed)
    print(f"  NRAE loaded  n_dct={n_dct}")

    # ── Load CNN feature extractor ─────────────────────────────────────────
    print(f"\nLoading CNN feature extractor: {args.clf_ckpt}")
    clf_params, clf_info = clf_load(args.clf_ckpt)
    print(f"  CLF loaded  info={clf_info}")

    # ── Test features ──────────────────────────────────────────────────────
    print("\nExtracting test CNN features ...")
    feats_test = extract_features(imgs_test, clf_params, batch_size=64)
    print(f"  feats_test: {feats_test.shape}")

    tau = compute_tau(Z_train)
    print(f"  DupRate τ = {tau:.5f}  ({DUP_PERCENTILE}th pct within-train NN dist)")

    # ── Whitening stats ────────────────────────────────────────────────────
    print("\nComputing whitening stats ...")
    mean_cls, S_sqrt_cls, S_invsqrt_cls = _whiten_stats(Z_train, k=SOLD_K_WHITEN)
    print("  Done.")

    # ── Generate DDPM samples once ─────────────────────────────────────────
    print(f"\n{'='*40}")
    print("Generating Latent-DDPM samples (unchanged) ...")
    ddpm_ckpt_path = os.path.join(args.ckpt_dir, "ddpm_best.pkl")
    if not os.path.exists(ddpm_ckpt_path):
        raise FileNotFoundError(
            f"DDPM checkpoint not found: {ddpm_ckpt_path}\n"
            "Run step2_train_ddpm.py first.")
    ddpm_params, ddpm_info = load_ddpm(ddpm_ckpt_path)
    latent_dim   = ddpm_info.get("latent_dim", LATENT_DIM)
    ddpm_hidden  = ddpm_info.get("hidden",     DDPM_HIDDEN)
    ddpm_nlayers = ddpm_info.get("n_layers",   DDPM_N_LAYERS)
    T            = ddpm_info.get("T",          DDPM_T_STEPS)

    alphas_cumprod_np, _ = make_cosine_schedule(T)
    rng = jax.random.PRNGKey(args.seed)
    rng, k_ddpm = jax.random.split(rng)

    t0 = time.perf_counter()
    Z_gen_ddpm = ddim_sample(
        ddpm_params, k_ddpm,
        alphas_cumprod=alphas_cumprod_np,
        n_samples=N_GENERATE,
        latent_dim=latent_dim,
        hidden=ddpm_hidden,
        n_layers=ddpm_nlayers,
        T=T,
        ddim_steps=DDIM_STEPS,
    )
    ddpm_sample_time = time.perf_counter() - t0
    print(f"  DDPM sampling time: {ddpm_sample_time:.2f}s")

    whiten_mu  = ddpm_info.get("whiten_mu")
    whiten_std = ddpm_info.get("whiten_std")
    if whiten_mu is not None and whiten_std is not None:
        Z_gen_ddpm = (np.array(Z_gen_ddpm) * whiten_std + whiten_mu).astype(np.float32)

    imgs_gen_ddpm = _decode(nrae_model, nrae_params, Z_gen_ddpm, n_dct)
    feats_ddpm = extract_features(imgs_gen_ddpm, clf_params, batch_size=64)
    ddpm_metrics = bootstrap_metrics(
        feats_test, feats_ddpm, Z_gen_ddpm, Z_train, tau,
        n_eval=N_EVAL, n_bootstrap=N_BOOTSTRAP,
        kid_degree=KID_DEGREE, seed=args.seed,
    )
    print(f"  DDPM  KID={ddpm_metrics['kid_mean']:.4f}±{ddpm_metrics['kid_std']:.4f}"
          f"  Recall={ddpm_metrics['recall_mean']:.3f}±{ddpm_metrics['recall_std']:.3f}"
          f"  DupRate={ddpm_metrics['duprate_mean']:.3f}±{ddpm_metrics['duprate_std']:.3f}")

    if args.save_latents:
        np.save(os.path.join(args.results_dir, "Z_gen_ddpm.npy"), Z_gen_ddpm)

    # ── Grid search ────────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Grid search: MM-SOLD and σ-CFDM  [KNN K={knn_K} L={knn_L}]\n")

    timing_sold = None
    timing_cfdm = None

    grid_sold = [[None] * len(M_GRID_run) for _ in range(len(SIGMA_GRID))]
    grid_cfdm = [[None] * len(M_GRID_run) for _ in range(len(SIGMA_GRID))]

    total_cells = len(SIGMA_GRID) * len(M_GRID_run)
    cell_num    = 0

    for si, sigma in enumerate(SIGMA_GRID):
        for mi, M in enumerate(M_GRID_run):
            cell_num += 1
            print(f"[{cell_num:2d}/{total_cells}]  σ={sigma}  M={M}  K={knn_K}  L={knn_L}")

            measure_timing = (sigma == TIMING_SIGMA and M == TIMING_M_KNN)

            # ── MM-SOLD (KNN+remainder) ────────────────────────────────────
            def _run_sold(n_part):
                z_w, _ = sample_class_overdamped_manifold_knn(
                    Z_class=jnp.asarray(Z_train, dtype=jnp.float32),
                    mean_class=mean_cls,
                    S_sqrt_class=S_sqrt_cls,
                    S_invsqrt_class=S_invsqrt_cls,
                    n_particles=n_part,
                    nsteps=SOLD_NSTEPS,
                    h=SOLD_H,
                    sigma_gmm=SOLD_SIGMA_GMM,
                    sigma_smoothing=sigma,
                    M=M,
                    knn_K=knn_K,
                    knn_L=knn_L,
                    discretization=SOLD_DISCRETIZ,
                    seed=args.seed + si * 100 + mi,
                )
                return np.array(unwhiten(z_w, mean_cls, S_sqrt_cls))

            # ── σ-CFDM (KNN+remainder) ─────────────────────────────────────
            def _run_cfdm(n_part):
                return sample_cfdm_knn_remainder(
                    Z_train, mean_cls, S_sqrt_cls, S_invsqrt_cls,
                    n_particles=n_part,
                    sigma=sigma,
                    M=M,
                    knn_K=knn_K,
                    knn_L=knn_L,
                    nsteps=CFDM_NSTEPS,
                    batch_size=CFDM_BATCH,
                    seed=args.seed + si * 100 + mi + 1000,
                )

            # CPU timing (only at the designated cell)
            if measure_timing:
                print(f"  Timing at (σ={sigma}, M={M}, K={knn_K}, L={knn_L}) ...")
                with jax.default_device(jax.devices("cpu")[0]):
                    timing_sold = _measure_cpu_time(
                        _run_sold, N_GENERATE, n_warmup=1, n_timed=3)
                    timing_cfdm = _measure_cpu_time(
                        _run_cfdm, N_GENERATE, n_warmup=1, n_timed=3)
                print(f"    SOLD CPU time: {timing_sold:.2f}s  "
                      f"CFDM CPU time: {timing_cfdm:.2f}s")

            # Generate on default device (GPU if available)
            t_sold0 = time.perf_counter()
            Z_sold = _run_sold(N_GENERATE)
            t_sold1 = time.perf_counter()

            t_cfdm0 = time.perf_counter()
            Z_cfdm = _run_cfdm(N_GENERATE)
            t_cfdm1 = time.perf_counter()

            print(f"  Generated  SOLD {Z_sold.shape}  CFDM {Z_cfdm.shape}  "
                  f"({t_sold1-t_sold0:.1f}s / {t_cfdm1-t_cfdm0:.1f}s)")

            imgs_sold = _decode(nrae_model, nrae_params, Z_sold, n_dct)
            imgs_cfdm = _decode(nrae_model, nrae_params, Z_cfdm, n_dct)

            feats_sold = extract_features(imgs_sold, clf_params, batch_size=64)
            feats_cfdm = extract_features(imgs_cfdm, clf_params, batch_size=64)

            m_sold = bootstrap_metrics(
                feats_test, feats_sold, Z_sold, Z_train, tau,
                n_eval=N_EVAL, n_bootstrap=N_BOOTSTRAP,
                kid_degree=KID_DEGREE, seed=args.seed,
            )
            m_cfdm = bootstrap_metrics(
                feats_test, feats_cfdm, Z_cfdm, Z_train, tau,
                n_eval=N_EVAL, n_bootstrap=N_BOOTSTRAP,
                kid_degree=KID_DEGREE, seed=args.seed,
            )

            grid_sold[si][mi] = m_sold
            grid_cfdm[si][mi] = m_cfdm

            print(f"  SOLD  KID={m_sold['kid_mean']:.4f}  "
                  f"Rec={m_sold['recall_mean']:.3f}  "
                  f"Dup={m_sold['duprate_mean']:.3f}")
            print(f"  CFDM  KID={m_cfdm['kid_mean']:.4f}  "
                  f"Rec={m_cfdm['recall_mean']:.3f}  "
                  f"Dup={m_cfdm['duprate_mean']:.3f}")

            if args.save_latents:
                np.save(os.path.join(
                    args.results_dir, f"Z_gen_sold_{si}_{mi}.npy"), Z_sold)
                np.save(os.path.join(
                    args.results_dir, f"Z_gen_cfdm_{si}_{mi}.npy"), Z_cfdm)

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "sigma_grid":   SIGMA_GRID,
        "M_grid":       M_GRID_run,
        "knn_K":        knn_K,
        "knn_L":        knn_L,
        "score_estimator": "knn_remainder",
        "ddpm":         ddpm_metrics,
        "sold_grid":    grid_sold,
        "cfdm_grid":    grid_cfdm,
    }
    results_path = os.path.join(args.results_dir, "results_grid.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    timing = {
        "ddpm_sample_time_s":              ddpm_sample_time,
        "ddpm_sample_time_per_sample_ms":  ddpm_sample_time / N_GENERATE * 1000,
        "sold_cpu_time_s":                 timing_sold,
        "sold_cpu_time_per_sample_ms":     timing_sold / N_GENERATE * 1000
                                           if timing_sold is not None else None,
        "cfdm_cpu_time_s":                 timing_cfdm,
        "cfdm_cpu_time_per_sample_ms":     timing_cfdm / N_GENERATE * 1000
                                           if timing_cfdm is not None else None,
        "timing_sigma":    TIMING_SIGMA,
        "timing_M":        TIMING_M_KNN,
        "knn_K":           knn_K,
        "knn_L":           knn_L,
        "score_estimator": "knn_remainder",
        "n_generate":      N_GENERATE,
        "ddim_steps":      DDIM_STEPS,
        "sold_nsteps":     SOLD_NSTEPS,
        "cfdm_nsteps":     CFDM_NSTEPS,
    }
    ddpm_timing_path = os.path.join(args.results_dir, "ddpm_training_time.json")
    if os.path.exists(ddpm_timing_path):
        with open(ddpm_timing_path) as f:
            timing.update(json.load(f))

    timing_path = os.path.join(args.results_dir, "timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"Timing saved: {timing_path}")

    print(f"\nStep 3 [KNN K={knn_K} L={knn_L}] complete!")


if __name__ == "__main__":
    _m_grid_default = ",".join(str(m) for m in M_GRID_KNN)

    parser = argparse.ArgumentParser(
        description="Step 3 (KNN+remainder): grid-search MM-SOLD / σ-CFDM / DDPM.")
    parser.add_argument("--data_dir",    type=str, default=DATA_DIR)
    parser.add_argument("--latent_dir",  type=str, default=LATENT_DIR)
    parser.add_argument("--ckpt_dir",    type=str, default=CKPT_DIR)
    parser.add_argument("--nrae_ckpt",   type=str, default=NRAE_CKPT)
    parser.add_argument("--clf_ckpt",    type=str, default=CLF_CKPT)
    # KNN parameters
    parser.add_argument("--knn_K",  type=int, default=20,
                        help="Deterministic nearest neighbors (default 20)")
    parser.add_argument("--knn_L",  type=int, default=20,
                        help="Random-remainder samples (default 20)")
    # Grid / eval overrides
    parser.add_argument("--m_grid", type=str, default=_m_grid_default,
                        help="Comma-separated M values")
    # Results dir: auto-named by K/L unless overridden
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Output directory (default: results_knn_K{K}_L{L}/)")
    parser.add_argument("--save_latents", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Auto-name results dir based on K, L
    if args.results_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.results_dir = os.path.join(
            here, f"results_knn_K{args.knn_K}_L{args.knn_L}")

    main(args)
