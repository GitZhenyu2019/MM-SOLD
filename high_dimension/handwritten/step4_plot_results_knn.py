"""
step4_plot_results_knn.py
=========================
Plot / tabulate results from step3_run_experiment_knn.py.
"""
import argparse
import json
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  
from matplotlib.colors import TwoSlopeNorm
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, LATENT_DIR, CKPT_DIR,
    LATENT_DIM, ENC1_HIDDEN, DEC_HIDDEN, N_DCT, UNET_BASE_CH,
    NRAE_CKPT, VIS_GRID, VIS_SEED,
    DDIM_STEPS, DDPM_T_STEPS, DDPM_HIDDEN, DDPM_N_LAYERS,
    SOLD_K_WHITEN, SOLD_NSTEPS, SOLD_H, SOLD_SIGMA_GMM,
    SOLD_DISCRETIZ, SOLD_SHARED_NOISE, SOLD_FIXED_NOISE,
    CFDM_NSTEPS, CFDM_BATCH, TIMING_SIGMA,
)
from nrae_model import make_nrae_state, decode_latents_nrae, load_params as nrae_load
from ddpm_model import make_cosine_schedule, ddim_sample, load_ddpm


# ─── helpers (same as step4_plot_results.py) ─────────────────────────────────

def _load_grid_metric(grid, SIGMA_GRID, M_GRID, key: str) -> np.ndarray:
    ns, nm = len(SIGMA_GRID), len(M_GRID)
    arr = np.full((ns, nm), np.nan)
    for si in range(ns):
        for mi in range(nm):
            cell = grid[si][mi]
            if cell is not None and key in cell:
                arr[si, mi] = cell[key]
    return arr


def _heatmap_relative(arr_sold, arr_cfdm, better, out_path, SIGMA_GRID, M_GRID,
                      mode="relative"):
    if mode == "absolute":
        rel = (arr_sold - arr_cfdm) * 100.0
        fmt = "{:+.1f}%"
    else:
        rel = (arr_sold - arr_cfdm) / (np.abs(arr_cfdm) + 1e-6) * 100.0
        fmt = "{:+.1f}%"

    rel_T = rel.T  # (n_M, n_sigma)
    cmap = "RdYlGn_r" if better == "lower" else "RdYlGn"
    abs_lim = max(np.nanpercentile(np.abs(rel_T[np.isfinite(rel_T)]), 95), 1.0)
    norm = TwoSlopeNorm(vmin=-abs_lim, vcenter=0, vmax=abs_lim)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(rel_T, norm=norm, cmap=cmap,
                   aspect="auto", origin="upper", interpolation="nearest")

    ax.set_xticks(range(len(SIGMA_GRID)))
    ax.set_xticklabels([str(s) for s in SIGMA_GRID], fontsize=8)
    ax.set_yticks(range(len(M_GRID)))
    ax.set_yticklabels([str(m) for m in M_GRID], fontsize=8)
    ax.set_xlabel("σ", fontsize=9)
    ax.set_ylabel("M", fontsize=9)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("")
    cb.ax.tick_params(labelsize=7)

    for i in range(rel_T.shape[0]):
        for j in range(rel_T.shape[1]):
            v = rel_T[i, j]
            if np.isfinite(v):
                normed = abs(v) / (abs_lim + 1e-9)
                color = "white" if normed > 0.6 else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=5.5, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _pick_vis(arr, n, seed=VIS_SEED):
    if len(arr) <= n:
        return arr
    idx = np.random.default_rng(seed).choice(len(arr), size=n, replace=False)
    return arr[idx]


def _save_seamless_grid(images, path):
    G = VIS_GRID
    H, W = images.shape[1], images.shape[2]
    canvas = np.ones((G * H, G * W), dtype=np.float32)
    for k in range(min(G * G, len(images))):
        r, c = k // G, k % G
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = images[k]
    fig = plt.figure(figsize=(G * W / 100, G * H / 100), dpi=100)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Saved: {path}")


def _find_nn_latent(Z_query, Z_ref):
    q2    = np.sum(Z_query ** 2, axis=1, keepdims=True)
    r2    = np.sum(Z_ref   ** 2, axis=1, keepdims=True)
    cross = Z_query @ Z_ref.T
    dist2 = np.maximum(q2 + r2.T - 2.0 * cross, 0.0)
    nn_idx  = np.argmin(dist2, axis=1)
    nn_dist = np.sqrt(dist2[np.arange(len(Z_query)), nn_idx])
    return nn_idx, nn_dist


def _save_nn_vis(query_imgs, nn_imgs, path, n_cols=4, n_rows=8, gap=4):
    H, W   = query_imgs.shape[1], query_imgs.shape[2]
    pair_w = 2 * W
    cw = n_cols * pair_w + (n_cols - 1) * gap
    ch = n_rows * H     + (n_rows - 1) * gap
    canvas = np.ones((ch, cw), dtype=np.float32)
    for k in range(n_cols * n_rows):
        row, col = k // n_cols, k % n_cols
        y0 = row * (H + gap)
        x0 = col * (pair_w + gap)
        canvas[y0:y0 + H, x0:x0 + W]          = query_imgs[k]
        canvas[y0:y0 + H, x0 + W:x0 + pair_w] = nn_imgs[k]
    fig = plt.figure(figsize=(cw / 100, ch / 100), dpi=100)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("Step 4 [KNN+remainder]: Plot results")
    print("=" * 60)

    os.makedirs(args.fig_dir,     exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load results ────────────────────────────────────────────────────────
    results_path = os.path.join(args.results_dir, "results_grid.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Results not found: {results_path}\n"
            "Run step3_run_experiment_knn.py first.")
    with open(results_path) as f:
        results = json.load(f)

    # Read grids from JSON — NOT from config, to support extended M_GRID
    SIGMA_GRID   = results["sigma_grid"]
    M_GRID       = results["M_grid"]
    knn_K        = results.get("knn_K", args.knn_K)
    knn_L        = results.get("knn_L", args.knn_L)
    grid_sold    = results["sold_grid"]
    grid_cfdm    = results["cfdm_grid"]
    ddpm_metrics = results["ddpm"]

    print(f"  σ grid : {SIGMA_GRID}")
    print(f"  M grid : {M_GRID}")
    print(f"  KNN K={knn_K}  L={knn_L}")

    # ── Load timing ─────────────────────────────────────────────────────────
    timing_path = os.path.join(args.results_dir, "timing.json")
    timing = {}
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            timing = json.load(f)

    # ── Extract metric grids ─────────────────────────────────────────────────
    metrics_cfg = [
        ("kid_mean",     "KID",     "lower",  "KID (↓ better)"),
        ("recall_mean",  "Recall",  "higher", "Recall (↑ better)"),
        ("duprate_mean", "DupRate", "lower",  "DupRate (↓ better)"),
    ]

    grids = {}
    for key, name, better, label in metrics_cfg:
        grids[key] = {
            "sold":   _load_grid_metric(grid_sold, SIGMA_GRID, M_GRID, key),
            "cfdm":   _load_grid_metric(grid_cfdm, SIGMA_GRID, M_GRID, key),
            "better": better,
            "label":  label,
            "name":   name,
        }

    # ── Heatmaps ─────────────────────────────────────────────────────────────
    print("\nGenerating heatmaps ...")
    for key, g in grids.items():
        short = key.replace("_mean", "")
        mode  = "absolute" if key == "duprate_mean" else "relative"
        _heatmap_relative(
            g["sold"], g["cfdm"],
            better=g["better"],
            out_path=os.path.join(args.fig_dir, f"heatmap_{short}.png"),
            SIGMA_GRID=SIGMA_GRID, M_GRID=M_GRID,
            mode=mode,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    def _best_cell(grid, key, better):
        arr = _load_grid_metric(grid, SIGMA_GRID, M_GRID, key)
        idx_flat = np.nanargmin(arr) if better == "lower" else np.nanargmax(arr)
        si, mi = np.unravel_index(idx_flat, arr.shape)
        return arr[si, mi], SIGMA_GRID[si], M_GRID[mi]

    rows = []
    for key, g in grids.items():
        name   = g["name"]
        better = g["better"]
        bval_s, bs_s, bm_s = _best_cell(grid_sold, key, better)
        bval_c, bs_c, bm_c = _best_cell(grid_cfdm, key, better)
        ddpm_val = ddpm_metrics.get(key, np.nan)

        sold_std = _load_grid_metric(grid_sold, SIGMA_GRID, M_GRID,
                                     key.replace("mean", "std"))
        cfdm_std = _load_grid_metric(grid_cfdm, SIGMA_GRID, M_GRID,
                                     key.replace("mean", "std"))
        si_s = SIGMA_GRID.index(bs_s); mi_s = M_GRID.index(bm_s)
        si_c = SIGMA_GRID.index(bs_c); mi_c = M_GRID.index(bm_c)
        std_s    = sold_std[si_s, mi_s]
        std_c    = cfdm_std[si_c, mi_c]
        ddpm_std = ddpm_metrics.get(key.replace("mean", "std"), np.nan)

        rows.append({
            "Metric":       name,
            "Better":       better,
            "MM-SOLD best": f"{bval_s:.4f}±{std_s:.4f}",
            "SOLD σ/M":     f"σ={bs_s}/M={bm_s}",
            "σ-CFDM best":  f"{bval_c:.4f}±{std_c:.4f}",
            "CFDM σ/M":     f"σ={bs_c}/M={bm_c}",
            "DDPM":         f"{ddpm_val:.4f}±{ddpm_std:.4f}",
        })

    print(f"\n{'='*90}")
    print(f"  KNN K={knn_K}  L={knn_L}  (score_estimator=knn_remainder)")
    print(f"{'='*90}")
    print(f"{'Metric':<12}{'MM-SOLD (best)':<22}{'config':<14}"
          f"{'σ-CFDM (best)':<22}{'config':<12}{'DDPM':<20}")
    print("-" * 90)
    for r in rows:
        print(f"{r['Metric']:<12}{r['MM-SOLD best']:<22}{r['SOLD σ/M']:<14}"
              f"{r['σ-CFDM best']:<22}{r['CFDM σ/M']:<12}{r['DDPM']:<20}")
    print("=" * 90)

    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis("off")
    headers   = ["Metric", "MM-SOLD (best)", "Config", "σ-CFDM (best)", "Config", "DDPM"]
    cell_data = [[r["Metric"], r["MM-SOLD best"], r["SOLD σ/M"],
                  r["σ-CFDM best"], r["CFDM σ/M"], r["DDPM"]] for r in rows]
    tbl = ax.table(cellText=cell_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    ax.set_title(
        f"Summary [KNN K={knn_K} L={knn_L}]  (mean±std, {results.get('score_estimator','knn_remainder')})",
        fontsize=11, pad=8)
    plt.tight_layout()
    tbl_path = os.path.join(args.fig_dir, "summary_table.png")
    plt.savefig(tbl_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {tbl_path}")

    csv_path = os.path.join(args.results_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")

    # ── Timing bar chart ──────────────────────────────────────────────────
    if timing:
        print("\nGenerating timing chart ...")
        sold_t  = timing.get("sold_cpu_time_s")
        cfdm_t  = timing.get("cfdm_cpu_time_s")
        ddpm_st = timing.get("ddpm_sample_time_s")
        ddpm_tr = timing.get("training_time_s")

        labels, vals, colors = [], [], []
        if sold_t  is not None: labels.append("MM-SOLD\n(CPU)");         vals.append(sold_t);  colors.append("#4CAF50")
        if cfdm_t  is not None: labels.append("σ-CFDM\n(CPU)");          vals.append(cfdm_t);  colors.append("#FF9800")
        if ddpm_st is not None: labels.append("DDPM sampling\n(GPU)");   vals.append(ddpm_st); colors.append("#2196F3")
        if ddpm_tr is not None: labels.append("DDPM training\n(GPU)");   vals.append(ddpm_tr); colors.append("#9C27B0")

        if labels:
            fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.8), 4))
            bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="black",
                          linewidth=0.7)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("Time (seconds)")
            ax.set_title(
                f"Sampling/Training Time  [KNN K={knn_K} L={knn_L}]  "
                f"N={timing.get('n_generate','?')} samples",
                fontsize=10)
            ax.set_ylim(0, max(vals) * 1.25)
            plt.tight_layout()
            t_path = os.path.join(args.fig_dir, "timing_bar.png")
            plt.savefig(t_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {t_path}")

    # ── Image grids ────────────────────────────────────────────────────────
    if args.make_grids:
        print("\nGenerating image grids ...")

        nrae_params_np, nrae_info = nrae_load(args.nrae_ckpt)
        n_dct = nrae_info.get("n_dct", N_DCT)
        rng   = jax.random.PRNGKey(VIS_SEED)
        nrae_model, _ = make_nrae_state(
            rng,
            latent_dim=nrae_info.get("latent_dim", LATENT_DIM),
            enc1_hidden=nrae_info.get("enc1_hidden", ENC1_HIDDEN),
            dec_hidden=nrae_info.get("dec_hidden", DEC_HIDDEN),
            n_dct=n_dct,
            unet_base_ch=nrae_info.get("unet_base_ch", UNET_BASE_CH),
        )
        nrae_p = jax.tree_util.tree_map(jnp.asarray, nrae_params_np)

        def _decode_imgs(Z):
            imgs = decode_latents_nrae(nrae_model, nrae_p, Z, batch_size=32)
            return np.clip(np.array(imgs), 0.0, 1.0)

        N_VIS = VIS_GRID * VIS_GRID
        N_NN  = N_VIS // 2
        grid_fig_dir = os.path.join(args.fig_dir, "grids")
        os.makedirs(grid_fig_dir, exist_ok=True)

        Z_train    = np.load(os.path.join(args.latent_dir, "Z_train.npy"))
        imgs_train = np.load(os.path.join(args.data_dir,   "images_train.npy"))

        imgs_real_vis = _pick_vis(imgs_train, N_VIS)
        _save_seamless_grid(imgs_real_vis,
                            os.path.join(grid_fig_dir, "grid_real.png"))

        from cfdm_sampling import sample_cfdm_knn_remainder, compute_whitening_stats
        from whitening_utils import unwhiten
        from sampling_algo   import sample_class_overdamped_manifold_knn

        mean_cls, S_sqrt_cls, S_invsqrt_cls = compute_whitening_stats(
            Z_train, k=SOLD_K_WHITEN)

        # DDPM
        ddpm_ckpt_path = os.path.join(args.ckpt_dir, "ddpm_best.pkl")
        if os.path.exists(ddpm_ckpt_path):
            ddpm_params, ddpm_info2 = load_ddpm(ddpm_ckpt_path)
            T        = ddpm_info2.get("T", DDPM_T_STEPS)
            ac_np, _ = make_cosine_schedule(T)
            ddpm_lat_path = os.path.join(args.results_dir, "Z_gen_ddpm.npy")
            if os.path.exists(ddpm_lat_path):
                Z_ddpm = np.load(ddpm_lat_path)
            else:
                rng, k_d = jax.random.split(rng)
                Z_ddpm = ddim_sample(
                    ddpm_params, k_d, alphas_cumprod=ac_np,
                    n_samples=N_VIS,
                    latent_dim=ddpm_info2.get("latent_dim", LATENT_DIM),
                    hidden=ddpm_info2.get("hidden", DDPM_HIDDEN),
                    n_layers=ddpm_info2.get("n_layers", DDPM_N_LAYERS),
                    T=T, ddim_steps=DDIM_STEPS,
                )
                # de-whiten if DDPM was trained on whitened latents
                _wmu  = ddpm_info2.get("whiten_mu")
                _wstd = ddpm_info2.get("whiten_std")
                if _wmu is not None and _wstd is not None:
                    Z_ddpm = (np.array(Z_ddpm) * _wstd + _wmu).astype(np.float32)
            Z_ddpm_pool    = _pick_vis(Z_ddpm, N_VIS)
            imgs_ddpm_pool = _decode_imgs(Z_ddpm_pool)
            _save_seamless_grid(imgs_ddpm_pool,
                                os.path.join(grid_fig_dir, "grid_ddpm.png"))
            Z_q_ddpm     = Z_ddpm_pool[:N_NN]
            imgs_q_ddpm  = imgs_ddpm_pool[:N_NN]
            nn_idx_d, nd = _find_nn_latent(Z_q_ddpm, Z_train)
            _save_nn_vis(imgs_q_ddpm, imgs_train[nn_idx_d],
                         os.path.join(grid_fig_dir, "nn_ddpm.png"))
            print(f"  DDPM  avg NN dist: {np.mean(nd):.4f}")

        # QR retraction requires n_particles >= latent_dim; use at least 256
        _N_GEN_GRID = max(N_VIS * 4, 256)

        # Grid loop
        total_cells = len(SIGMA_GRID) * len(M_GRID)
        cell_num    = 0
        for si, sigma in enumerate(SIGMA_GRID):
            for mi, M in enumerate(M_GRID):
                cell_num += 1
                tag = f"s{sigma}_M{M}".replace(".", "p")
                print(f"\n  [{cell_num:2d}/{total_cells}]  σ={sigma}  M={M}")

                # MM-SOLD
                sold_lat = os.path.join(args.results_dir, f"Z_gen_sold_{si}_{mi}.npy")
                if os.path.exists(sold_lat):
                    Z_vis_sold = np.load(sold_lat)
                else:
                    z_w, _ = sample_class_overdamped_manifold_knn(
                        Z_class=jnp.asarray(Z_train, dtype=jnp.float32),
                        mean_class=mean_cls, S_sqrt_class=S_sqrt_cls,
                        S_invsqrt_class=S_invsqrt_cls,
                        n_particles=_N_GEN_GRID,
                        nsteps=SOLD_NSTEPS, h=SOLD_H,
                        sigma_gmm=SOLD_SIGMA_GMM, sigma_smoothing=sigma,
                        M=M, knn_K=knn_K, knn_L=knn_L,
                        discretization=SOLD_DISCRETIZ, seed=VIS_SEED,
                    )
                    Z_vis_sold = np.array(unwhiten(z_w, mean_cls, S_sqrt_cls))

                Z_sold_pool    = _pick_vis(Z_vis_sold, N_VIS)
                imgs_sold_pool = _decode_imgs(Z_sold_pool)
                _save_seamless_grid(imgs_sold_pool,
                                    os.path.join(grid_fig_dir, f"grid_sold_{tag}.png"))
                nn_si, nd_s = _find_nn_latent(Z_sold_pool[:N_NN], Z_train)
                _save_nn_vis(imgs_sold_pool[:N_NN], imgs_train[nn_si],
                             os.path.join(grid_fig_dir, f"nn_sold_{tag}.png"))
                print(f"    SOLD  avg NN dist: {np.mean(nd_s):.4f}")

                # σ-CFDM
                cfdm_lat = os.path.join(args.results_dir, f"Z_gen_cfdm_{si}_{mi}.npy")
                if os.path.exists(cfdm_lat):
                    Z_vis_cfdm = np.load(cfdm_lat)
                else:
                    Z_vis_cfdm = sample_cfdm_knn_remainder(
                        Z_train, mean_cls, S_sqrt_cls, S_invsqrt_cls,
                        n_particles=N_VIS,
                        sigma=sigma, M=M,
                        knn_K=knn_K, knn_L=knn_L,
                        nsteps=CFDM_NSTEPS, batch_size=CFDM_BATCH, seed=VIS_SEED,
                    )

                Z_cfdm_pool    = _pick_vis(Z_vis_cfdm, N_VIS)
                imgs_cfdm_pool = _decode_imgs(Z_cfdm_pool)
                _save_seamless_grid(imgs_cfdm_pool,
                                    os.path.join(grid_fig_dir, f"grid_cfdm_{tag}.png"))
                nn_ci, nd_c = _find_nn_latent(Z_cfdm_pool[:N_NN], Z_train)
                _save_nn_vis(imgs_cfdm_pool[:N_NN], imgs_train[nn_ci],
                             os.path.join(grid_fig_dir, f"nn_cfdm_{tag}.png"))
                print(f"    CFDM  avg NN dist: {np.mean(nd_c):.4f}")

        print(f"\n  All grids saved to: {grid_fig_dir}/")

    print("\nStep 4 [KNN+remainder] complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4 (KNN+remainder): plot experiment results.")
    parser.add_argument("--knn_K",       type=int, default=20)
    parser.add_argument("--knn_L",       type=int, default=20)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--fig_dir",     type=str, default=None)
    parser.add_argument("--data_dir",    type=str, default=DATA_DIR)
    parser.add_argument("--latent_dir",  type=str, default=LATENT_DIR)
    parser.add_argument("--ckpt_dir",    type=str, default=CKPT_DIR)
    parser.add_argument("--nrae_ckpt",   type=str, default=NRAE_CKPT)
    parser.add_argument("--make_grids",  action="store_true")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.results_dir is None:
        args.results_dir = os.path.join(here, f"results_knn_K{args.knn_K}_L{args.knn_L}")
    if args.fig_dir is None:
        args.fig_dir = os.path.join(here, f"figures_knn_K{args.knn_K}_L{args.knn_L}")

    main(args)
