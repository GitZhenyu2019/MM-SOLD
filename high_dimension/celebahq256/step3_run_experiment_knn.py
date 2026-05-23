"""
step3_run_experiment_knn.py
===========================
KNN+remainder variant of the CelebA-HQ-256 grid-search experiment.
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
    DATA_DIR, LATENT_DIR, CKPT_DIR, RESULTS_DIR, NRAE_CKPT,
    LATENT_DIM, ENC1_HIDDEN, DEC_HIDDEN, N_DCT, UNET_BASE_CH,
    SIGMA_GRID, M_GRID,
    SOLD_NSTEPS, SOLD_H, SOLD_SIGMA_GMM, SOLD_DISCRETIZ,
    SOLD_K_WHITEN,
    CFDM_NSTEPS, CFDM_BATCH,
    DDIM_STEPS, DDPM_T_STEPS, DDPM_HIDDEN, DDPM_N_LAYERS,
    N_GENERATE, N_EVAL, N_BOOTSTRAP,
    KID_DEGREE, DUP_PERCENTILE,
    TIMING_SIGMA, TIMING_M,
)

_M_GRID_DEFAULT = ",".join(str(m) for m in M_GRID)

from nrae_model_celeba import (
    NRAEModelCeleba, decode_latents_nrae, load_params as nrae_load,
)
from metrics_celeba import (
    build_inception_extractor, extract_inception_features,
    compute_tau, bootstrap_metrics,
)

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "handwritten")))
from ddpm_model    import make_cosine_schedule, ddim_sample, load_ddpm
from cfdm_sampling import sample_cfdm_knn_remainder
from whitening_utils import (
    compute_sample_mean_cov, symmetric_matrix_sqrt_and_invsqrt,
    unwhiten,
)
from sampling_algo import sample_class_overdamped_manifold_knn


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_nrae(ckpt_path: str):
    params_np, info = nrae_load(ckpt_path)
    model = NRAEModelCeleba(
        latent_dim=info.get("latent_dim",   LATENT_DIM),
        enc1_hidden=info.get("enc1_hidden",  ENC1_HIDDEN),
        dec_hidden=info.get("dec_hidden",    DEC_HIDDEN),
        n_dct=info.get("n_dct",             N_DCT),
        unet_base_ch=info.get("unet_base_ch", UNET_BASE_CH))
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    return model, params, info.get("n_dct", N_DCT)


def _decode(model, params, Z: np.ndarray) -> np.ndarray:
    imgs = decode_latents_nrae(model, params, Z, batch_size=16)
    return np.clip(np.array(imgs), 0.0, 1.0)


def _whiten_stats(Z: np.ndarray, k: int):
    Z_j = jnp.asarray(Z, dtype=jnp.float32)
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
    M_GRID_run      = [int(x) for x in args.m_grid.split(",")]
    N_GENERATE_run  = args.n_generate
    N_EVAL_run      = args.n_eval
    N_BOOTSTRAP_run = args.n_bootstrap
    knn_K           = args.knn_K
    knn_L           = args.knn_L

    print("=" * 60)
    print("Step 3 [KNN+remainder]: MM-SOLD / σ-CFDM / DDPM")
    print("=" * 60)
    print(f"JAX backend  : {jax.default_backend()}  devices : {jax.devices()}")
    print(f"σ grid       : {SIGMA_GRID}")
    print(f"M grid       : {M_GRID_run}")
    print(f"KNN K={knn_K}  L={knn_L}")
    print(f"Total cells  : {len(SIGMA_GRID) * len(M_GRID_run)} per method")
    print(f"N_GENERATE   : {N_GENERATE_run}  N_EVAL={N_EVAL_run}  "
          f"N_BOOTSTRAP={N_BOOTSTRAP_run}")
    print(f"Results dir  : {args.results_dir}\n")

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    Z_train   = np.load(os.path.join(args.latent_dir, args.z_train_file))
    Z_test    = np.load(os.path.join(args.latent_dir, args.z_test_file))
    imgs_test = np.load(os.path.join(args.data_dir,   args.imgs_test_file))
    print(f"Loaded  Z_train={Z_train.shape}  Z_test={Z_test.shape}  "
          f"imgs_test={imgs_test.shape}")

    # ── Load NRAE ──────────────────────────────────────────────────────────
    print(f"\nLoading NRAE: {args.nrae_ckpt}")
    nrae_model, nrae_params, n_dct = _load_nrae(args.nrae_ckpt)
    print(f"  NRAE loaded  n_dct={n_dct}")

    # ── Build Inception-V3 feature extractor ──────────────────────────────
    print("\nBuilding InceptionV3 feature extractor (clean-fid) ...")
    incept_model, incept_device = build_inception_extractor()
    print(f"  Using device: {incept_device}")

    # ── Extract test features ─────────────────────────────────────────────
    feats_test_path = os.path.join(args.results_dir, "feats_test.npy")
    if os.path.exists(feats_test_path):
        print(f"\nLoading cached test Inception features: {feats_test_path}")
        feats_test = np.load(feats_test_path)
    else:
        print(f"\nExtracting test Inception features ({imgs_test.shape[0]} images) ...")
        feats_test = extract_inception_features(
            imgs_test, incept_model, incept_device, batch_size=64)
        np.save(feats_test_path, feats_test)
        print(f"  feats_test: {feats_test.shape}  → saved")

    # ── Compute τ for DupRate ──────────────────────────────────────────────
    tau = compute_tau(Z_train, percentile=DUP_PERCENTILE)
    print(f"\n  DupRate τ = {tau:.5f}  ({DUP_PERCENTILE}th pct within-train NN dist)")

    # ── Compute whitening stats ───────────────────────────────────────────
    print(f"\nComputing whitening stats (K_WHITEN={SOLD_K_WHITEN}) ...")
    mean_cls, S_sqrt_cls, S_invsqrt_cls = _whiten_stats(Z_train, k=SOLD_K_WHITEN)
    print("  Done.")

    # ── Generate DDPM samples (once) ──────────────────────────────────────
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
        n_samples=N_GENERATE_run,
        latent_dim=latent_dim,
        hidden=ddpm_hidden,
        n_layers=ddpm_nlayers,
        T=T,
        ddim_steps=DDIM_STEPS,
    )
    ddpm_sample_time = time.perf_counter() - t0
    print(f"  DDPM sampling time: {ddpm_sample_time:.2f}s")

    # Un-normalise
    whiten_type = ddpm_info.get("whiten_type", "zscore")
    if whiten_type == "pca_zscore":
        _pca_mean  = ddpm_info.get("pca_mean")
        _pca_comps = ddpm_info.get("pca_components")
        _pca_std   = ddpm_info.get("pca_std")
        if all(x is not None for x in [_pca_mean, _pca_comps, _pca_std]):
            Z_pca_raw  = np.array(Z_gen_ddpm) * _pca_std
            Z_gen_ddpm = (Z_pca_raw @ _pca_comps + _pca_mean).astype(np.float32)
    elif whiten_type == "pca":
        _mu     = ddpm_info.get("whiten_mu")
        _S_sqrt = ddpm_info.get("whiten_S_sqrt")
        if _mu is not None and _S_sqrt is not None:
            Z_gen_ddpm = np.array(
                unwhiten(jnp.asarray(Z_gen_ddpm),
                         jnp.asarray(_mu), jnp.asarray(_S_sqrt))
            ).astype(np.float32)
    else:  # zscore (legacy)
        _mu  = ddpm_info.get("whiten_mu")
        _std = ddpm_info.get("whiten_std")
        if _mu is not None and _std is not None:
            Z_gen_ddpm = (np.array(Z_gen_ddpm) * _std + _mu).astype(np.float32)
    print(f"  Un-normalised DDPM latents: {Z_gen_ddpm.shape}")

    print(f"  Decoding {N_GENERATE_run} DDPM latents ...")
    imgs_gen_ddpm = _decode(nrae_model, nrae_params, Z_gen_ddpm)

    print("  Extracting DDPM Inception features ...")
    feats_ddpm = extract_inception_features(
        imgs_gen_ddpm, incept_model, incept_device, batch_size=64)

    print("  Computing DDPM metrics (bootstrap) ...")
    ddpm_metrics = bootstrap_metrics(
        feats_test, feats_ddpm, Z_gen_ddpm, Z_train, tau,
        n_eval=N_EVAL_run, n_bootstrap=N_BOOTSTRAP_run,
        kid_degree=KID_DEGREE, seed=args.seed,
    )
    print(f"  DDPM  FID={ddpm_metrics['fid_mean']:.2f}±{ddpm_metrics['fid_std']:.2f}"
          f"  KID={ddpm_metrics['kid_mean']:.4f}±{ddpm_metrics['kid_std']:.4f}"
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
            print(f"[{cell_num:2d}/{total_cells}]  σ={sigma}  M={M}  "
                  f"K={knn_K}  L={knn_L}")

            measure_timing = (sigma == TIMING_SIGMA and M == TIMING_M)

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
                    M=M, knn_K=knn_K, knn_L=knn_L,
                    discretization=SOLD_DISCRETIZ,
                    seed=args.seed + si * 100 + mi,
                )
                return np.array(unwhiten(z_w, mean_cls, S_sqrt_cls))

            def _run_cfdm(n_part):
                return sample_cfdm_knn_remainder(
                    Z_train, mean_cls, S_sqrt_cls, S_invsqrt_cls,
                    n_particles=n_part,
                    sigma=sigma, M=M,
                    knn_K=knn_K, knn_L=knn_L,
                    nsteps=CFDM_NSTEPS,
                    batch_size=CFDM_BATCH,
                    seed=args.seed + si * 100 + mi + 1000,
                )

            if measure_timing:
                print(f"  Timing at (σ={sigma}, M={M}, K={knn_K}, L={knn_L}) ...")
                with jax.default_device(jax.devices("cpu")[0]):
                    timing_sold = _measure_cpu_time(
                        _run_sold, N_GENERATE_run, n_warmup=1, n_timed=3)
                    timing_cfdm = _measure_cpu_time(
                        _run_cfdm, N_GENERATE_run, n_warmup=1, n_timed=3)
                print(f"    SOLD CPU time: {timing_sold:.2f}s  "
                      f"CFDM CPU time: {timing_cfdm:.2f}s")

            t_sold0 = time.perf_counter()
            Z_sold  = _run_sold(N_GENERATE_run)
            t_sold1 = time.perf_counter()

            t_cfdm0 = time.perf_counter()
            Z_cfdm  = _run_cfdm(N_GENERATE_run)
            t_cfdm1 = time.perf_counter()

            print(f"  Generated  SOLD {Z_sold.shape}  CFDM {Z_cfdm.shape}  "
                  f"({t_sold1-t_sold0:.1f}s / {t_cfdm1-t_cfdm0:.1f}s)")

            imgs_sold = _decode(nrae_model, nrae_params, Z_sold)
            imgs_cfdm = _decode(nrae_model, nrae_params, Z_cfdm)

            feats_sold = extract_inception_features(
                imgs_sold, incept_model, incept_device, batch_size=64)
            feats_cfdm = extract_inception_features(
                imgs_cfdm, incept_model, incept_device, batch_size=64)

            m_sold = bootstrap_metrics(
                feats_test, feats_sold, Z_sold, Z_train, tau,
                n_eval=N_EVAL_run, n_bootstrap=N_BOOTSTRAP_run,
                kid_degree=KID_DEGREE, seed=args.seed,
            )
            m_cfdm = bootstrap_metrics(
                feats_test, feats_cfdm, Z_cfdm, Z_train, tau,
                n_eval=N_EVAL_run, n_bootstrap=N_BOOTSTRAP_run,
                kid_degree=KID_DEGREE, seed=args.seed,
            )

            grid_sold[si][mi] = m_sold
            grid_cfdm[si][mi] = m_cfdm

            print(f"  SOLD  FID={m_sold['fid_mean']:.2f}  "
                  f"KID={m_sold['kid_mean']:.4f}  "
                  f"Rec={m_sold['recall_mean']:.3f}  "
                  f"Dup={m_sold['duprate_mean']:.3f}")
            print(f"  CFDM  FID={m_cfdm['fid_mean']:.2f}  "
                  f"KID={m_cfdm['kid_mean']:.4f}  "
                  f"Rec={m_cfdm['recall_mean']:.3f}  "
                  f"Dup={m_cfdm['duprate_mean']:.3f}")

            if args.save_latents:
                np.save(os.path.join(
                    args.results_dir, f"Z_gen_sold_{si}_{mi}.npy"), Z_sold)
                np.save(os.path.join(
                    args.results_dir, f"Z_gen_cfdm_{si}_{mi}.npy"), Z_cfdm)

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "sigma_grid":      SIGMA_GRID,
        "M_grid":          M_GRID_run,
        "knn_K":           knn_K,
        "knn_L":           knn_L,
        "score_estimator": "knn_remainder",
        "ddpm":            ddpm_metrics,
        "sold_grid":       grid_sold,
        "cfdm_grid":       grid_cfdm,
    }
    results_path = os.path.join(args.results_dir, "results_grid.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    timing = {
        "ddpm_sample_time_s":             ddpm_sample_time,
        "ddpm_sample_time_per_sample_ms": ddpm_sample_time / N_GENERATE_run * 1000,
        "sold_cpu_time_s":                timing_sold,
        "sold_cpu_time_per_sample_ms":    timing_sold / N_GENERATE_run * 1000
                                          if timing_sold is not None else None,
        "cfdm_cpu_time_s":                timing_cfdm,
        "cfdm_cpu_time_per_sample_ms":    timing_cfdm / N_GENERATE_run * 1000
                                          if timing_cfdm is not None else None,
        "timing_sigma":   TIMING_SIGMA,
        "timing_M":       TIMING_M,
        "knn_K":          knn_K,
        "knn_L":          knn_L,
        "n_generate":     N_GENERATE_run,
        "ddim_steps":     DDIM_STEPS,
        "sold_nsteps":    SOLD_NSTEPS,
        "cfdm_nsteps":    CFDM_NSTEPS,
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
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Step 3 (KNN+remainder): CelebA-HQ-256 grid-search experiment.")
    parser.add_argument("--knn_K",       type=int, default=20,
                        help="Deterministic nearest neighbours (default 20)")
    parser.add_argument("--knn_L",       type=int, default=20,
                        help="Random remainder samples (default 20)")
    parser.add_argument("--data_dir",    type=str, default=DATA_DIR)
    parser.add_argument("--latent_dir",  type=str, default=LATENT_DIR)
    parser.add_argument("--ckpt_dir",    type=str, default=CKPT_DIR)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--nrae_ckpt",   type=str, default=NRAE_CKPT)
    parser.add_argument("--z_train_file",   type=str, default="Z_train.npy")
    parser.add_argument("--z_test_file",    type=str, default="Z_test.npy")
    parser.add_argument("--imgs_test_file", type=str, default="images_test.npy")
    parser.add_argument("--n_generate",  type=int, default=N_GENERATE)
    parser.add_argument("--n_eval",      type=int, default=N_EVAL)
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--m_grid",      type=str, default=_M_GRID_DEFAULT)
    parser.add_argument("--save_latents", action="store_true")
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(
            here, f"results_knn_K{args.knn_K}_L{args.knn_L}")

    main(args)
