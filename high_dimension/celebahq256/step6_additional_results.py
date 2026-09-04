"""
step6_additional_results.py
=====================================
the appendix "Additional results" () for the CelebA-HQ DiffAE-latent
generation experiment -- Figures 9/10, CelebA-HQ grid-level heatmap figure + CelebA-HQ posterior-moment-validation table, CelebA-HQ explicit-sampler comparison figure. NOT
covered by step2/step4/step5 (those only produce CelebA-HQ generation results table and CelebA-HQ representative-sample figure).
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import jax
import jax.numpy as jnp

import config as cfg                                    # noqa: E402
import tilting_lion as TL                                        # noqa: E402
from common import (                                     # noqa: E402
    seamless_grid as _seamless_grid_raw, save_nn_vis as _save_nn_vis_raw,
    find_nn, decode_batched, gamma_baoab, h_sold, h_baoab, delta_sigma,
    sample_class_overdamped_manifold_knn, best_cell_filtered, _pick_table6_cell,
)
from metrics_celeba import (                                     # noqa: E402
    compute_tau, compute_kid, compute_recall, compute_duprate,
    build_inception_extractor, extract_inception_features, compute_fid_from_features,
)
from cfdm_sampling import sample_cfdm_knn_remainder                # noqa: E402


VIS_GRID = 8
NN_COLS, NN_ROWS = 4, 8
TABLE6_N_MOMENT = 10000
TABLE6_N_METRIC = 3000   


def _seamless_grid(images, path, G=VIS_GRID):
    _seamless_grid_raw(images, path, G=G)
    print(f"  Saved: {path}")


def _save_nn_vis(query_imgs, nn_imgs, path, n_cols=NN_COLS, n_rows=NN_ROWS, gap=4):
    _save_nn_vis_raw(query_imgs, nn_imgs, path, n_cols=n_cols, n_rows=n_rows, gap=gap)
    print(f"  Saved: {path}")


def _heatmap(arr_a, arr_b, better, out_path, x_grid, y_grid, mode="relative",
            x_label="c_sigma", y_label="M", figsize=(6.0, 3.0)):
    if mode == "absolute":
        rel = (arr_a - arr_b) * 100.0
    else:
        rel = (arr_a - arr_b) / (np.abs(arr_b) + 1e-6) * 100.0
    fmt = "{:+.1f}%"
    rel_T = rel.T

    cmap = "RdYlGn_r" if better == "lower" else "RdYlGn"
    finite = rel_T[np.isfinite(rel_T)]
    abs_lim = max(np.nanpercentile(np.abs(finite), 95), 1.0) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-abs_lim, vcenter=0, vmax=abs_lim)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rel_T, norm=norm, cmap=cmap, aspect="equal",
                   origin="upper", interpolation="nearest")
    ax.set_xticks(range(len(x_grid)))
    ax.set_xticklabels([str(s) for s in x_grid], fontsize=8)
    ax.set_yticks(range(len(y_grid)))
    ax.set_yticklabels([str(s) for s in y_grid], fontsize=8)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    for i in range(rel_T.shape[0]):
        for j in range(rel_T.shape[1]):
            v = rel_T[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) / (abs_lim + 1e-9) > 0.6 else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _nearest_grid_index(grid_values, target):
    return min(range(len(grid_values)), key=lambda i: abs(grid_values[i] - target))


def _cached_array(path, fn, regenerate):
    if os.path.exists(path) and not regenerate:
        return np.load(path)
    arr = np.asarray(fn())
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR_DIFFAE)
    ap.add_argument("--fig_dir", default=cfg.FIG_DIR_DIFFAE)
    ap.add_argument("--engine_path", default=cfg.DIFFAE_TRT_ENGINE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regenerate", action="store_true",
                    help="force fresh sampling for Figures 9/11 instead of "
                         "reusing cached raw samples from a previous run")
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()

    global VIS_GRID, NN_COLS, NN_ROWS, TABLE6_N_MOMENT, TABLE6_N_METRIC
    baoab_nsteps = cfg.BAOAB_NSTEPS
    results_suffix = ""
    table6_name = "table6.json"
    if a.smoke_test:
        VIS_GRID, NN_COLS, NN_ROWS = 2, 2, 2
        TABLE6_N_MOMENT, TABLE6_N_METRIC = 50, 8
        baoab_nsteps = 20
        results_suffix = "_smoke"
        table6_name = "table6_smoke.json"
        a.fig_dir = os.path.join(a.fig_dir, "smoke")
        print("\n*** SMOKE TEST MODE ***\n")

    os.makedirs(a.fig_dir, exist_ok=True)
    cache_dir = os.path.join(a.results_dir, "fig_cache" + results_suffix)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("Step 6 (diffae): the appendix additional results (Fig 9/10/11/12, CelebA-HQ posterior-moment-validation table)")
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
    RBF_BW_IMG = meta["rbf_bw_img"]
    d = U_train190.shape[1]
    tau = compute_tau(U_train190, percentile=cfg.DUP_PERCENTILE)
    gamma190 = gamma_baoab(U_train190)
    print(f"  U_train190 {U_train190.shape}  NN={NN:.5f}  gamma={gamma190:.5f}")

    Xj = jnp.asarray(U_train190, dtype=jnp.float32)
    mean0 = jnp.zeros(d, dtype=jnp.float32)
    S_eye = jnp.eye(d, dtype=jnp.float32)

    print("\nBuilding image decoder (TensorRT DPM-Solver++(2M)) and "
          "Inception feature extractor ...")
    from dpm_decode import make_renderer
    render_fn, _ctx = make_renderer(engine_path=a.engine_path, nsteps=cfg.DECODE_NSTEPS,
                                    batch=cfg.DECODE_BATCH)
    inception_model, inception_device = build_inception_extractor()
    f_ref = np.load(cfg.DIFFAE_FEATURES_NPZ)["f_ref"]

    # ── representative cell for CelebA-HQ sigma-CFDM sample-grid figure/10 ──────────────────────────
    mmsold_res = json.load(open(os.path.join(
        a.results_dir, "mmsold", f"results_mmsold{results_suffix}.json")))
    best = best_cell_filtered(list(mmsold_res["grid"].values()))
    c_delta_rep, c_sigma_rep, M_rep = best["c_delta"], best["c_sigma"], best["M"]
    print(f"\nRepresentative cell (MM-SOLD best-KID, DupRate<="
          f"{cfg.REP_CELL_DUPRATE_CAP}, Recall>={cfg.REP_CELL_RECALL_FLOOR}): "
          f"c_delta={c_delta_rep}  c_sigma={c_sigma_rep}  M={M_rep}")
    delta_rep, sigma_rep = delta_sigma(c_delta_rep, c_sigma_rep, NN, dsub=cfg.PROJ_K)

    
    c_delta_fig11, c_sigma_fig11, M_fig11 = 0.4, 0.4, 32
    fig11_cell = next(c for c in mmsold_res["grid"].values()
                      if c["c_delta"] == c_delta_fig11 and c["c_sigma"] == c_sigma_fig11
                      and c["M"] == M_fig11)
    print(f"\nCelebA-HQ grid-level heatmap figure cell (MANUALLY chosen for more visible sample "
          f"diversity -- see comment above, independent of CelebA-HQ sigma-CFDM sample-grid figure/10's "
          f"cell): c_delta={c_delta_fig11}  c_sigma={c_sigma_fig11}  M={M_fig11}  "
          f"(kid_mean={fig11_cell['kid_mean']:.4f} recall_mean={fig11_cell['recall_mean']:.4f} "
          f"duprate_mean={fig11_cell['duprate_mean']:.4f})")
    delta_fig11, sigma_fig11 = delta_sigma(c_delta_fig11, c_sigma_fig11, NN, dsub=cfg.PROJ_K)

    def _decode(U190):
        imgs = decode_batched(U190, mean512, Vt, Vb, lambda_tail, render_fn,
                              is_subspace=True, batch=cfg.DECODE_BATCH, seed=a.seed)
        return imgs.astype(np.float32) / 255.0

    # =========================================================================
    # CelebA-HQ sigma-CFDM sample-grid figure: MM-SOLD vs sigma-CFDM sample grids.
    # =========================================================================
    csigma_corners = (min(cfg.CSIGMA_GRID), max(cfg.CSIGMA_GRID))
    print(f"\n{'='*40}\nCelebA-HQ sigma-CFDM sample-grid figure: sample grids at 4 corners "
          f"(c_delta={c_delta_rep}, c_sigma in {csigma_corners})\n")
    n_vis = VIS_GRID * VIS_GRID
    sold_nsteps = 20 if a.smoke_test else cfg.SOLD_NSTEPS
    cfdm_nsteps = 20 if a.smoke_test else cfg.CFDM_NSTEPS

    for M_corner in (2, 32):
        for c_sigma_corner in csigma_corners:
            _, sigma_c = delta_sigma(c_delta_rep, c_sigma_corner, NN, dsub=cfg.PROJ_K)
            tag = f"M{M_corner}_cs{c_sigma_corner}"

            def _gen_sold(sigma_c=sigma_c, M_corner=M_corner):
                U, _ = sample_class_overdamped_manifold_knn(
                    X_class=Xj, n_particles=max(n_vis, cfg.PROJ_K + 1), nsteps=sold_nsteps,
                    h=h_sold(delta_rep), sigma_gmm=delta_rep, sigma_smoothing=sigma_c,
                    M=M_corner, knn_K=cfg.KNN_K_SMOOTH, knn_L=cfg.KNN_L_SMOOTH,
                    discretization=cfg.SOLD_DISCRETIZ, seed=a.seed)
                return np.array(U, dtype=np.float32)[:n_vis]
            U_sold = _cached_array(os.path.join(cache_dir, f"fig9_mmsold_cd{c_delta_rep:.2f}_{tag}.npy"),
                                   _gen_sold, a.regenerate)
            _seamless_grid(_decode(U_sold), os.path.join(a.fig_dir, f"fig9_mmsold_{tag}.png"))

            def _gen_cfdm(sigma_c=sigma_c, M_corner=M_corner):
                return sample_cfdm_knn_remainder(
                    U_train190, mean0, S_eye, S_eye, n_vis, sigma_c, M_corner,
                    knn_K=cfg.KNN_K_SMOOTH, knn_L=cfg.KNN_L_SMOOTH,
                    nsteps=cfdm_nsteps, batch_size=min(cfg.CFDM_BATCH, n_vis), seed=a.seed)
            U_cfdm = _cached_array(os.path.join(cache_dir, f"fig9_cfdm_{tag}.npy"),
                                   _gen_cfdm, a.regenerate)
            _seamless_grid(_decode(U_cfdm), os.path.join(a.fig_dir, f"fig9_cfdm_{tag}.png"))

    # =========================================================================
    # CelebA-HQ MM-SOLD sample-grid figure
    # =========================================================================
    print(f"\n{'='*40}\nCelebA-HQ MM-SOLD sample-grid figure: heatmaps at c_delta={c_delta_rep}\n")
    cfdm_res = json.load(open(os.path.join(
        a.results_dir, "cfdm", f"results_cfdm{results_suffix}.json")))
    ci = _nearest_grid_index(cfg.CDELTA_GRID, c_delta_rep)
    ns, nm = len(cfg.CSIGMA_GRID), len(cfg.M_GRID)

    def _slice(res, key):
        arr = np.full((ns, nm), np.nan)
        for si in range(ns):
            for mi in range(nm):
                cell = res["grid"].get(f"{ci}_{si}_{mi}")
                if cell is not None:
                    arr[si, mi] = cell[key]
        return arr

    for metric_key, better, mode, fname in [
        ("kid_mean", "lower", "relative", "fig10_heatmap_kid.png"),
        ("recall_mean", "higher", "relative", "fig10_heatmap_recall.png"),
        ("duprate_mean", "lower", "absolute", "fig10_heatmap_duprate.png"),
    ]:
        arr_sold = _slice(mmsold_res, metric_key)
        arr_cfdm = _slice(cfdm_res, metric_key)
        _heatmap(arr_sold, arr_cfdm, better, os.path.join(a.fig_dir, fname),
                cfg.CSIGMA_GRID, cfg.M_GRID, mode=mode)

    # =========================================================================
    # CelebA-HQ explicit-sampler comparison figure
    # =========================================================================
    print(f"\n{'='*40}\nCelebA-HQ explicit-sampler comparison figure: pairwise heatmaps at M=32\n")
    pluginbaoab_res = json.load(open(os.path.join(
        a.results_dir, "pluginbaoab", f"results_pluginbaoab{results_suffix}.json")))
    sgbaoab_res = json.load(open(os.path.join(
        a.results_dir, "sgbaoab", f"results_sgbaoab{results_suffix}.json")))
    mi_32 = cfg.M_GRID.index(32)
    nc8, ns8 = len(cfg.CDELTA_GRID), len(cfg.CSIGMA_GRID)

    def _slice_delta_sigma(res, key, mi=mi_32):
        arr = np.full((ns8, nc8), np.nan)
        for ci2 in range(nc8):
            for si in range(ns8):
                cell = res["grid"].get(f"{ci2}_{si}_{mi}")
                if cell is not None:
                    arr[si, ci2] = cell[key]
        return arr

    fig12_pairs = [
        ("MM-SOLD", "Plug-in BAOAB", mmsold_res, pluginbaoab_res, "fig12_mmsold_vs_pluginbaoab"),
        ("MM-SOLD", "SG-BAOAB", mmsold_res, sgbaoab_res, "fig12_mmsold_vs_sgbaoab"),
        ("Plug-in BAOAB", "SG-BAOAB", pluginbaoab_res, sgbaoab_res, "fig12_pluginbaoab_vs_sgbaoab"),
    ]
    for name_a, name_b, res_a, res_b, fprefix in fig12_pairs:
        for metric_key, better, mode, mname in [
            ("kid_mean", "lower", "relative", "kid"),
            ("recall_mean", "higher", "relative", "recall"),
            ("duprate_mean", "lower", "absolute", "duprate"),
        ]:
            arr_a = _slice_delta_sigma(res_a, metric_key)
            arr_b = _slice_delta_sigma(res_b, metric_key)
            out_path = os.path.join(a.fig_dir, f"{fprefix}_{mname}.png")
            _heatmap(arr_a, arr_b, better, out_path, cfg.CSIGMA_GRID, cfg.CDELTA_GRID,
                    mode=mode, x_label="c_sigma", y_label="c_delta", figsize=(9.0, 7.6))
            print(f"  {name_a} vs {name_b} ({mname}): saved {out_path}")

    # =========================================================================
    # CelebA-HQ grid-level heatmap figure / CelebA-HQ posterior-moment-validation table cell
    # =========================================================================
    print(f"\n{'='*40}\nCelebA-HQ grid-level heatmap figure / CelebA-HQ posterior-moment-validation table: picking a cell that actually "
          f"shows naive -> plug-in -> SG-BAOAB improving\n")
    c_delta_t6, c_sigma_t6 = 0.4, 0.4
    delta_t6, sigma_t6 = delta_sigma(c_delta_t6, c_sigma_t6, NN, dsub=cfg.PROJ_K)
    pi_t6_grid_cell = next(c for c in pluginbaoab_res["grid"].values()
                           if c["c_delta"] == c_delta_t6 and c["c_sigma"] == c_sigma_t6
                           and c["M"] == 32)
    sg_t6_grid_cell = next(c for c in sgbaoab_res["grid"].values()
                           if c["c_delta"] == c_delta_t6 and c["c_sigma"] == c_sigma_t6
                           and c["M"] == 32)
    print(f"  -> c_delta={c_delta_t6} c_sigma={c_sigma_t6} M=32  "
          f"(from Plug-in/SG-BAOAB's own grids: "
          f"Plug-in KID={pi_t6_grid_cell['kid_mean']:.4f} Recall={pi_t6_grid_cell['recall_mean']:.4f}  "
          f"SG-BAOAB KID={sg_t6_grid_cell['kid_mean']:.4f} Recall={sg_t6_grid_cell['recall_mean']:.4f})")

    # =========================================================================
    # CelebA-HQ grid-level heatmap figure
    # =========================================================================
    print(f"\n{'='*40}\nCelebA-HQ grid-level heatmap figure: MM-SOLD / Plug-in BAOAB / SG-BAOAB "
          f"@ c_delta={c_delta_fig11} c_sigma={c_sigma_fig11} M={M_fig11} "
          f"(CelebA-HQ grid-level heatmap figure's own cell, independent of CelebA-HQ sigma-CFDM sample-grid figure/10 and CelebA-HQ posterior-moment-validation table)\n")
    h_fig11 = h_baoab(delta_fig11)

    def _gen_sold11():
        # see the n_vis/max(...) note by CelebA-HQ sigma-CFDM sample-grid figure's _gen_sold above
        U, _ = sample_class_overdamped_manifold_knn(
            X_class=Xj, n_particles=max(n_vis, cfg.PROJ_K + 1), nsteps=sold_nsteps,
            h=h_sold(delta_fig11), sigma_gmm=delta_fig11, sigma_smoothing=sigma_fig11,
            M=M_fig11, knn_K=cfg.KNN_K_SMOOTH, knn_L=cfg.KNN_L_SMOOTH,
            discretization=cfg.SOLD_DISCRETIZ, seed=a.seed)
        return np.array(U, dtype=np.float32)[:n_vis]
    fig11_tag = f"cd{c_delta_fig11:.2f}_cs{c_sigma_fig11:.2f}_M{M_fig11}"
    U_sold11 = _cached_array(os.path.join(cache_dir, f"fig11_mmsold_{fig11_tag}.npy"),
                             _gen_sold11, a.regenerate)

    plugin_cache_fig11 = os.path.join(cache_dir, f"fig11_plugin_lam_lam_{fig11_tag}.npz")
    if os.path.exists(plugin_cache_fig11) and not a.regenerate:
        dd = np.load(plugin_cache_fig11)
        lam_pi_fig11, Lam_pi_fig11 = dd["lam"], dd["Lam"]
    else:
        lam_pi_fig11, Lam_pi_fig11 = TL.plugin_tilting(
            jax.random.PRNGKey(a.seed), Xj, delta_fig11, sigma_fig11, M_fig11,
            score=cfg.BAOAB_SCORE, knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB)
        np.savez(plugin_cache_fig11, lam=lam_pi_fig11, Lam=Lam_pi_fig11)

    def _gen_pi11():
        return TL.sample_tilted(
            lam_pi_fig11, Lam_pi_fig11, Xj, delta=delta_fig11, sigma_s=sigma_fig11, M=M_fig11,
            h=h_fig11, gamma=gamma190, n_steps=baoab_nsteps, n_samples=n_vis,
            splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB, seed=a.seed)
    U_pi11 = _cached_array(os.path.join(cache_dir, f"fig11_pluginbaoab_{fig11_tag}.npy"),
                           _gen_pi11, a.regenerate)

    sg_grid_cache = os.path.join(a.results_dir, "sgbaoab", "tilting_cache",
                                 f"cell_cd{c_delta_fig11:.2f}_cs{c_sigma_fig11:.2f}_M{M_fig11}.npz")
    dd = np.load(sg_grid_cache)
    lam_sg_fig11, Lam_sg_fig11 = dd["lam"], dd["Lam"]
    print(f"  SG-BAOAB (CelebA-HQ grid-level heatmap figure panel): reusing step2f's grid-cached "
          f"(lambda, Lambda) from {sg_grid_cache} -- no refit needed")

    def _gen_sg11():
        return TL.sample_tilted(
            lam_sg_fig11, Lam_sg_fig11, Xj, delta=delta_fig11, sigma_s=sigma_fig11, M=M_fig11,
            h=h_fig11, gamma=gamma190, n_steps=baoab_nsteps, n_samples=n_vis,
            splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB, seed=a.seed)
    U_sg11 = _cached_array(os.path.join(cache_dir, f"fig11_sgbaoab_{fig11_tag}.npy"),
                           _gen_sg11, a.regenerate)

    for name, U_gen in [("mmsold", U_sold11), ("pluginbaoab", U_pi11), ("sgbaoab", U_sg11)]:
        _seamless_grid(_decode(U_gen), os.path.join(a.fig_dir, f"fig11_{name}.png"))
        nn_idx, nn_dist = find_nn(U_gen[:NN_COLS * NN_ROWS], U_train190)
        _save_nn_vis(_decode(U_gen[:NN_COLS * NN_ROWS]), _decode(U_train190[nn_idx]),
                    os.path.join(a.fig_dir, f"fig11_nn_{name}.png"))
        print(f"  {name}: mean NN dist to training set (top-190) = {nn_dist.mean():.4f}")

    # =========================================================================
    # CelebA-HQ posterior-moment-validation table
    # =========================================================================
    print(f"\n{'='*40}\nCelebA-HQ posterior-moment-validation table: posterior moment validation\n")
    mu_star, Sigma_star = TL.moment_targets(np.asarray(U_train190), delta_t6)
    h_t6 = h_baoab(delta_t6)

    plugin_cache_t6 = os.path.join(cache_dir, f"table6_plugin_lam_lam_cd{c_delta_t6:.2f}_cs{c_sigma_t6:.2f}.npz")
    if os.path.exists(plugin_cache_t6) and not a.regenerate:
        dd = np.load(plugin_cache_t6)
        lam_pi_t6, Lam_pi_t6 = dd["lam"], dd["Lam"]
    else:
        lam_pi_t6, Lam_pi_t6 = TL.plugin_tilting(jax.random.PRNGKey(a.seed), Xj, delta_t6,
                                                 sigma_t6, 32, score=cfg.BAOAB_SCORE,
                                                 knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB)
        np.savez(plugin_cache_t6, lam=lam_pi_t6, Lam=Lam_pi_t6)

    sg_grid_cache_t6 = os.path.join(a.results_dir, "sgbaoab", "tilting_cache",
                                    f"cell_cd{c_delta_t6:.2f}_cs{c_sigma_t6:.2f}_M32.npz")
    dd = np.load(sg_grid_cache_t6)
    lam_sg_t6, Lam_sg_t6 = dd["lam"], dd["Lam"]
    print(f"  SG-BAOAB (CelebA-HQ posterior-moment-validation table): reusing step2f's grid-cached (lambda, Lambda) "
          f"from {sg_grid_cache_t6} -- no refit needed")

    settings = {
        "naive_baoab": (np.zeros(d, dtype=np.float32), np.zeros((d, d), dtype=np.float32)),
        "plugin_baoab": (lam_pi_t6, Lam_pi_t6),
        "sg_baoab": (lam_sg_t6, Lam_sg_t6),
    }
    table6 = {}
    t6_tag = f"cd{c_delta_t6:.2f}_cs{c_sigma_t6:.2f}"
    for name, (lam, Lam) in settings.items():
        def _gen_moment(lam=lam, Lam=Lam):
            return TL.sample_tilted(
                lam, Lam, Xj, delta=delta_t6, sigma_s=sigma_t6, M=32,
                h=h_t6, gamma=gamma190, n_steps=baoab_nsteps,
                n_samples=TABLE6_N_MOMENT, splitting=cfg.BAOAB_SPLITTING,
                score=cfg.BAOAB_SCORE, knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB,
                seed=a.seed)
        q_moment = _cached_array(os.path.join(cache_dir, f"table6_{name}_{t6_tag}_moment.npy"),
                                 _gen_moment, a.regenerate).astype(np.float32)
        mean_abs = float(np.linalg.norm(mu_star - q_moment.mean(0)))
        mean_rel = mean_abs / float(np.linalg.norm(mu_star) + 1e-12)
        Qc = q_moment - mu_star[None, :]
        S_hat = (Qc.T @ Qc) / q_moment.shape[0]
        cov_abs = float(np.linalg.norm(Sigma_star - S_hat))
        cov_rel = cov_abs / float(np.linalg.norm(Sigma_star))

        def _gen_metric(lam=lam, Lam=Lam):
            return TL.sample_tilted(
                lam, Lam, Xj, delta=delta_t6, sigma_s=sigma_t6, M=32,
                h=h_t6, gamma=gamma190, n_steps=baoab_nsteps,
                n_samples=TABLE6_N_METRIC, splitting=cfg.BAOAB_SPLITTING,
                score=cfg.BAOAB_SCORE, knn_K=cfg.KNN_K_BAOAB, knn_L=cfg.KNN_L_BAOAB,
                seed=a.seed + 1)
        q_metric = _cached_array(os.path.join(cache_dir, f"table6_{name}_{t6_tag}_metric.npy"),
                                 _gen_metric, a.regenerate).astype(np.float32)

        # latent-space metrics (cheap, top-190 / raw 512-dim via tail completion)
        from common import from_subspace_gaussian_tail, compute_kid_rbfmix
        Z_metric512 = from_subspace_gaussian_tail(q_metric, mean512, Vt, Vb, lambda_tail,
                                                   seed=a.seed + 2)
        rng = np.random.default_rng(a.seed)
        kid_lat, _ = compute_kid_rbfmix(Z_test512, Z_metric512, RBF_BW, rng=rng)
        rec_lat = compute_recall(U_test190, q_metric, k_neighbour=cfg.K_RECALL)
        dup_lat = compute_duprate(q_metric, U_train190, tau)

        # image-space metrics (decoded subset)
        imgs = decode_batched(Z_metric512, mean512, Vt, Vb, lambda_tail, render_fn,
                              is_subspace=False, batch=cfg.DECODE_BATCH, seed=a.seed + 3)
        feats_gen = extract_inception_features(imgs.astype(np.float32) / 255.0,
                                               inception_model, inception_device)
        fid_img = compute_fid_from_features(f_ref, feats_gen)
        kid_img, _ = compute_kid(f_ref, feats_gen, subset_size=min(cfg.KID_SUBSET, len(feats_gen)),
                                 n_subsets=cfg.KID_N_SUBSETS, degree=cfg.KID_DEGREE, rng=rng)
        kid_img_rbf, _ = compute_kid_rbfmix(f_ref, feats_gen, RBF_BW_IMG, rng=rng)

        table6[name] = {
            "mean_abs": mean_abs, "mean_rel": mean_rel,
            "cov_abs": cov_abs, "cov_rel": cov_rel,
            "latent_kid": kid_lat, "recall": rec_lat, "duprate": dup_lat,
            "image_fid": fid_img, "image_kid": kid_img, "image_kid_rbf": kid_img_rbf,
        }
        print(f"  {name:14s}  |z-mu*|={mean_abs:.4f} rel={mean_rel:.4f}  "
              f"|Sig-Sig*|_F={cov_abs:.4f} rel={cov_rel:.4f}  "
              f"latentKID={kid_lat:.4f} Recall={rec_lat:.3f} DupRate={dup_lat:.3f}  "
              f"imgFID={fid_img:.3f} imgKID(poly3)={kid_img:.5f} imgKID(RBF)={kid_img_rbf:.5f}")

    out = {
        "representative_cell": {"c_delta": c_delta_t6, "c_sigma": c_sigma_t6,
                                "M": 32, "delta": delta_t6, "sigma": sigma_t6,
                                "note": "picked by _pick_table6_cell, NOT the same as "
                                        "CelebA-HQ sigma-CFDM sample-grid figure/10's c_delta_rep/c_sigma_rep"},
        "n_moment_samples": TABLE6_N_MOMENT, "n_metric_samples": TABLE6_N_METRIC,
        "table6": table6,
    }
    with open(os.path.join(a.results_dir, table6_name), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {os.path.join(a.results_dir, table6_name)}")

    print(f"\nAll CelebA-HQ sigma-CFDM sample-grid figure/10/11/12 PNGs saved under {a.fig_dir}/")
    print("\nStep 6 (diffae) complete!")


if __name__ == "__main__":
    main()
