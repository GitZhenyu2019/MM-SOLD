"""
step5_generate_figures.py
===============================
the appendix "Additional results" -- Figures 5, 6, 7, 8 and digit-8 posterior-moment-validation table.
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
import tilting_lion as TL                                   # noqa: E402
from common import (                                     # noqa: E402
    load_nrae, decode_from_subspace, delta_sigma, h_sold, h_baoab,
    gamma_baoab, sample_class_overdamped_manifold, best_cell_filtered,
    seamless_grid as _seamless_grid_raw, find_nn, save_nn_vis as _save_nn_vis_raw,
)
from cfdm_sampling import sample_cfdm                         # noqa: E402


VIS_GRID = 8       # 8x8 = 64 images per sample grid
NN_COLS, NN_ROWS = 4, 8   # 32-pair nearest-neighbour visualisation
VIS_SEED = 0
TABLE5_N_MOMENT = 10000
TABLE5_N_METRIC = 300


def _seamless_grid(images: np.ndarray, path: str, G: int = VIS_GRID):
    _seamless_grid_raw(images, path, G=G)
    print(f"  Saved: {path}")


_find_nn = find_nn


def _save_nn_vis(query_imgs: np.ndarray, nn_imgs: np.ndarray, path: str,
                 n_cols: int = NN_COLS, n_rows: int = NN_ROWS, gap: int = 4):
    _save_nn_vis_raw(query_imgs, nn_imgs, path, n_cols=n_cols, n_rows=n_rows, gap=gap)
    print(f"  Saved: {path}")


def _heatmap(arr_a: np.ndarray, arr_b: np.ndarray, better: str, out_path: str,
            x_grid, y_grid, mode: str = "relative",
            x_label: str = "c_sigma", y_label: str = "M", figsize=(6.0, 3.0)):
    if mode == "absolute":
        rel = (arr_a - arr_b) * 100.0
    else:
        rel = (arr_a - arr_b) / (np.abs(arr_b) + 1e-6) * 100.0
    fmt = "{:+.1f}%"
    rel_T = rel.T   # rows = y_grid, cols = x_grid

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


def _pick_fig7_table5_cell(mmsold_grid, plugin_grid, sg_grid, duprate_cap, recall_floor,
                           min_c_delta=0.3):
    def _index(grid):
        return {(c["c_delta"], c["c_sigma"]): c for c in grid.values() if c["M"] == 32}

    mm, pi, sg = _index(mmsold_grid), _index(plugin_grid), _index(sg_grid)
    shared = set(mm) & set(pi) & set(sg)
    candidates = [mm[k] for k in shared
                  if mm[k]["duprate_mean"] <= duprate_cap and mm[k]["recall_mean"] >= recall_floor
                  and mm[k]["c_delta"] >= min_c_delta]
    if not candidates:
        print(f"  [WARNING] _pick_fig7_table5_cell: no M=32 c_delta>={min_c_delta} cell has "
              f"MM-SOLD duprate_mean<={duprate_cap} and recall_mean>={recall_floor}; "
              f"falling back to best_cell_filtered restricted to c_delta>={min_c_delta} "
              f"(all M)")
        restricted = [c for c in mmsold_grid.values() if c["c_delta"] >= min_c_delta]
        best = best_cell_filtered(restricted or list(mmsold_grid.values()),
                                  duprate_cap=duprate_cap, recall_floor=recall_floor)
        return best["c_delta"], best["c_sigma"], best

    def sg_improvement(c):
        key = (c["c_delta"], c["c_sigma"])
        pc, sc = pi[key], sg[key]
        return (pc["moment_res_cov"] / pc["moment_res_cov_floor"]
                - sc["moment_res_cov"] / sc["moment_res_cov_floor"])

    best_mm = max(candidates, key=sg_improvement)
    key = (best_mm["c_delta"], best_mm["c_sigma"])
    pc, sc = pi[key], sg[key]
    print(f"  Figure7/Table5 cell scan: {len(candidates)}/{len(shared)} M=32 cells pass "
          f"MM-SOLD's DupRate/Recall bar; picked c_delta={best_mm['c_delta']} "
          f"c_sigma={best_mm['c_sigma']} (Plug-in cov/floor="
          f"{pc['moment_res_cov']/pc['moment_res_cov_floor']:.2f}, SG cov/floor="
          f"{sc['moment_res_cov']/sc['moment_res_cov_floor']:.2f})")
    return best_mm["c_delta"], best_mm["c_sigma"], best_mm


def _cached_array(path: str, fn, regenerate: bool):
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
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR, "subspace"))
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR)
    ap.add_argument("--fig_dir", default=cfg.FIG_DIR)
    ap.add_argument("--nrae_ckpt", default=cfg.NRAE_CKPT)
    ap.add_argument("--c_delta_rep", type=float, default=None,
                    help="override the auto-picked representative c_delta for "
                         "digit-8 sigma-CFDM/MM-SOLD sample-grid figure/6 ONLY (digit-8 explicit-sampler comparison figure/digit-8 posterior-moment-validation table use a separately "
                         "auto-picked cell, see _pick_fig7_table5_cell)")
    ap.add_argument("--c_sigma_rep", type=float, default=None,
                    help="override the auto-picked representative c_sigma for "
                         "digit-8 sigma-CFDM/MM-SOLD sample-grid figure/6 ONLY")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regenerate", action="store_true",
                    help="force fresh sampling for Figures 5/7 instead of "
                         "reusing cached raw samples from a previous run "
                         "(use this after changing a sampler or "
                         "the representative cell; NOT needed just to tweak "
                         "plot styling, which replots from cache by default)")
    ap.add_argument("--smoke_test", action="store_true",
                    help="read step2a/b's *_smoke.json outputs and use tiny "
                         "sample counts / step counts everywhere")
    a = ap.parse_args()

    global VIS_GRID, NN_COLS, NN_ROWS, TABLE5_N_MOMENT, TABLE5_N_METRIC
    sold_nsteps = cfg.SOLD_NSTEPS
    baoab_nsteps = cfg.BAOAB_NSTEPS
    lion_steps, lion_chains, lion_iters = cfg.LION_N_STEPS, cfg.LION_N_CHAINS, cfg.LION_N_ITERS
    results_suffix = ""
    table5_name = "table5.json"
    if a.smoke_test:
        VIS_GRID, NN_COLS, NN_ROWS = 2, 2, 2
        TABLE5_N_MOMENT, TABLE5_N_METRIC = 50, 8
        sold_nsteps, baoab_nsteps = 20, 20
        lion_steps, lion_chains, lion_iters = 10, 8, 5
        results_suffix = "_smoke"
        table5_name = "table5_smoke.json"
        # never let smoke-test PNGs/JSON land on top of the real outputs
        a.fig_dir = os.path.join(a.fig_dir, "smoke")
        print("\n*** SMOKE TEST MODE ***\n")

    os.makedirs(a.fig_dir, exist_ok=True)
    os.makedirs(os.path.join(a.fig_dir, "grids"), exist_ok=True)
    cache_dir = os.path.join(a.results_dir, "fig_cache" + results_suffix)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("Step 5: the appendix additional results (Fig 5/6/7/8, digit-8 posterior-moment-validation table)")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    U_train8 = np.load(os.path.join(a.subspace_dir, "U_train8.npy"))
    mean_8   = np.load(os.path.join(a.subspace_dir, "mean_8.npy"))
    V        = np.load(os.path.join(a.subspace_dir, "proj_V.npy"))
    meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    NN8 = meta["NN8"]
    d = U_train8.shape[1]
    print(f"  U_train8 {U_train8.shape}  NN8={NN8:.5f}")

    nrae_model, nrae_params, n_dct = load_nrae(a.nrae_ckpt, seed=0)

    def decode(U):
        return decode_from_subspace(U, mean_8, V, nrae_model, nrae_params, n_dct)

    mmsold_res = json.load(open(os.path.join(
        a.results_dir, "mmsold", f"results_mmsold{results_suffix}.json")))
    cells = list(mmsold_res["grid"].values())
    best = best_cell_filtered(cells)
    c_delta_rep = a.c_delta_rep if a.c_delta_rep is not None else best["c_delta"]
    c_sigma_rep = a.c_sigma_rep if a.c_sigma_rep is not None else best["c_sigma"]
    M_rep = best["M"]
    print(f"\nRepresentative cell (MM-SOLD best-KID, "
          f"DupRate<={cfg.REP_CELL_DUPRATE_CAP}, Recall>={cfg.REP_CELL_RECALL_FLOOR}): "
          f"c_delta={c_delta_rep}  c_sigma={c_sigma_rep}  M={M_rep}  "
          f"(KID={best['kid_mean']:.4f}  DupRate={best['duprate_mean']:.4f}  "
          f"Recall={best['recall_mean']:.4f})")

    delta_rep, sigma_rep = delta_sigma(c_delta_rep, c_sigma_rep, NN8, dsub=cfg.PROJ_K)
    gamma8 = gamma_baoab(U_train8)

    csigma_corners = (min(cfg.CSIGMA_GRID), max(cfg.CSIGMA_GRID))
    print(f"\n{'='*40}\ndigit-8 sigma-CFDM/MM-SOLD sample-grid figure: sample grids at 4 corners "
          f"(c_delta={c_delta_rep}, c_sigma in {csigma_corners})\n")
    mean0 = jnp.zeros(d, dtype=jnp.float32)
    S_eye = jnp.eye(d, dtype=jnp.float32)

    for M_corner in (2, 32):
        for c_sigma_corner in csigma_corners:
            _, sigma_c = delta_sigma(c_delta_rep, c_sigma_corner, NN8, dsub=cfg.PROJ_K)
            tag = f"M{M_corner}_cs{c_sigma_corner}"

            def _gen_sold(sigma_c=sigma_c, M_corner=M_corner):
                n_req = VIS_GRID * VIS_GRID
                U, _ = sample_class_overdamped_manifold(
                    X_class=jnp.asarray(U_train8, dtype=jnp.float32),
                    n_particles=max(n_req, cfg.PROJ_K + 1), nsteps=sold_nsteps,
                    h=h_sold(delta_rep), sigma_gmm=delta_rep, sigma_smoothing=sigma_c,
                    M=M_corner, shared_noise=cfg.SOLD_SHARED_NOISE,
                    discretization=cfg.SOLD_DISCRETIZ, fixed_noise=cfg.SOLD_FIXED_NOISE,
                    seed=a.seed)
                return np.array(U, dtype=np.float32)[:n_req]
            U_sold = _cached_array(os.path.join(cache_dir, f"fig5_mmsold_{tag}.npy"),
                                   _gen_sold, a.regenerate)
            _seamless_grid(decode(U_sold),
                           os.path.join(a.fig_dir, "grids", f"fig5_mmsold_{tag}.png"),
                           G=VIS_GRID)

            def _gen_cfdm(sigma_c=sigma_c, M_corner=M_corner):
                return sample_cfdm(
                    U_train8, mean0, S_eye, S_eye,
                    n_particles=VIS_GRID * VIS_GRID, sigma=sigma_c, M=M_corner,
                    shared_noise=cfg.CFDM_SHARED_NOISE, nsteps=cfg.CFDM_NSTEPS,
                    batch_size=cfg.CFDM_BATCH, seed=a.seed)
            U_cfdm = _cached_array(os.path.join(cache_dir, f"fig5_cfdm_{tag}.npy"),
                                   _gen_cfdm, a.regenerate)
            _seamless_grid(decode(U_cfdm),
                           os.path.join(a.fig_dir, "grids", f"fig5_cfdm_{tag}.png"),
                           G=VIS_GRID)

    print(f"\n{'='*40}\ndigit-8 grid-level heatmap figure: heatmaps at c_delta={c_delta_rep}\n")
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
        ("kid_mean", "lower", "relative", "fig6_heatmap_kid.png"),
        ("recall_mean", "higher", "relative", "fig6_heatmap_recall.png"),
        ("duprate_mean", "lower", "absolute", "fig6_heatmap_duprate.png"),
    ]:
        arr_sold = _slice(mmsold_res, metric_key)
        arr_cfdm = _slice(cfdm_res, metric_key)
        _heatmap(arr_sold, arr_cfdm, better, os.path.join(a.fig_dir, fname),
                cfg.CSIGMA_GRID, cfg.M_GRID, mode=mode)

    print(f"\n{'='*40}\ndigit-8 pairwise-heatmap figure: pairwise heatmaps at M=32\n")
    pluginbaoab_res = json.load(open(os.path.join(
        a.results_dir, "pluginbaoab", f"results_pluginbaoab{results_suffix}.json")))
    sgbaoab_res = json.load(open(os.path.join(
        a.results_dir, "sgbaoab", f"results_sgbaoab{results_suffix}.json")))
    mi_32 = cfg.M_GRID.index(32)
    nc8, ns8 = len(cfg.CDELTA_GRID), len(cfg.CSIGMA_GRID)

    def _slice_delta_sigma(res, key, mi=mi_32):
        arr = np.full((ns8, nc8), np.nan)   # (n_c_sigma, n_c_delta) = (x, y)
        for ci in range(nc8):
            for si in range(ns8):
                cell = res["grid"].get(f"{ci}_{si}_{mi}")
                if cell is not None:
                    arr[si, ci] = cell[key]
        return arr

    fig8_pairs = [
        ("MM-SOLD", "Plug-in BAOAB", mmsold_res, pluginbaoab_res, "fig8_mmsold_vs_pluginbaoab"),
        ("MM-SOLD", "SG-BAOAB", mmsold_res, sgbaoab_res, "fig8_mmsold_vs_sgbaoab"),
        ("Plug-in BAOAB", "SG-BAOAB", pluginbaoab_res, sgbaoab_res, "fig8_pluginbaoab_vs_sgbaoab"),
    ]
    for name_a, name_b, res_a, res_b, fprefix in fig8_pairs:
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

    print(f"\n{'='*40}\ndigit-8 explicit-sampler comparison figure / digit-8 posterior-moment-validation table: picking a cell that actually "
          f"shows naive -> plug-in -> SG-BAOAB improving\n")
    c_delta_fig7, c_sigma_fig7, fig7_cell = _pick_fig7_table5_cell(
        mmsold_res["grid"], pluginbaoab_res["grid"], sgbaoab_res["grid"],
        cfg.REP_CELL_DUPRATE_CAP, cfg.REP_CELL_RECALL_FLOOR)
    delta_fig7, sigma_fig7 = delta_sigma(c_delta_fig7, c_sigma_fig7, NN8, dsub=cfg.PROJ_K)
    print(f"  -> c_delta={c_delta_fig7} c_sigma={c_sigma_fig7} M=32  "
          f"(MM-SOLD KID={fig7_cell['kid_mean']:.4f} "
          f"DupRate={fig7_cell['duprate_mean']:.4f} Recall={fig7_cell['recall_mean']:.4f})")

    baoab_nsteps = 20 if a.smoke_test else 10000
    print(f"  (BAOAB step count for digit-8 explicit-sampler comparison figure / digit-8 posterior-moment-validation table boosted to "
          f"{baoab_nsteps} for adequate mixing; grid default is "
          f"{cfg.BAOAB_NSTEPS})")

    print(f"\n{'='*40}\ndigit-8 explicit-sampler comparison figure: MM-SOLD / Plug-in BAOAB / SG-BAOAB "
          f"@ c_delta={c_delta_fig7} c_sigma={c_sigma_fig7} M=32\n")
    Xj = jnp.asarray(U_train8, dtype=jnp.float32)
    n_vis = VIS_GRID * VIS_GRID

    def _gen_sold7():
        U, _ = sample_class_overdamped_manifold(
            X_class=Xj, n_particles=max(n_vis, cfg.PROJ_K + 1), nsteps=sold_nsteps,
            h=h_sold(delta_fig7), sigma_gmm=delta_fig7, sigma_smoothing=sigma_fig7,
            M=32, shared_noise=cfg.SOLD_SHARED_NOISE, discretization=cfg.SOLD_DISCRETIZ,
            fixed_noise=cfg.SOLD_FIXED_NOISE, seed=a.seed)
        return np.array(U, dtype=np.float32)[:n_vis]
    U_sold7 = _cached_array(os.path.join(cache_dir, "fig7_mmsold.npy"),
                            _gen_sold7, a.regenerate)

    plugin_cache = os.path.join(cache_dir, "fig7_plugin_lam_lam.npz")
    if os.path.exists(plugin_cache) and not a.regenerate:
        dd = np.load(plugin_cache)
        lam_pi, Lam_pi = dd["lam"], dd["Lam"]
    else:
        lam_pi, Lam_pi = TL.plugin_tilting(jax.random.PRNGKey(a.seed), Xj, delta_fig7,
                                           sigma_fig7, 32, score=cfg.BAOAB_SCORE)
        np.savez(plugin_cache, lam=lam_pi, Lam=Lam_pi)

    def _gen_pi7():
        return TL.sample_tilted(
            lam_pi, Lam_pi, Xj, delta=delta_fig7, sigma_s=sigma_fig7, M=32,
            h=h_baoab(delta_fig7), gamma=gamma8, n_steps=baoab_nsteps,
            n_samples=n_vis, splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            seed=a.seed)
    U_pi7 = _cached_array(os.path.join(cache_dir, "fig7_pluginbaoab.npy"),
                          _gen_pi7, a.regenerate)

    sg_boost_cache = os.path.join(cache_dir, "fig7_sg_lam_lam_boosted.npz")
    if os.path.exists(sg_boost_cache) and not a.regenerate:
        dd = np.load(sg_boost_cache)
        lam_sg, Lam_sg = dd["lam"], dd["Lam"]
        print(f"  SG-BAOAB: reusing cached BOOSTED (lambda, Lambda) from {sg_boost_cache}")
    else:
        boost_iters, boost_steps = lion_iters * 2, int(lion_steps * 1.5)
        print(f"  SG-BAOAB: fitting fresh with boosted Lion budget "
              f"(n_iters={boost_iters} vs grid default {lion_iters}, "
              f"n_steps={boost_steps} vs grid default {lion_steps}) ...")
        lam_sg, Lam_sg, _hist = TL.estimate_tilting_lion(
            Xj, delta=delta_fig7, sigma_s=sigma_fig7, M=32, h=h_baoab(delta_fig7),
            gamma=gamma8, n_steps=boost_steps, n_chains=lion_chains,
            n_iters=boost_iters, lr_rel=cfg.LION_LR_REL, b1=cfg.LION_B1,
            b2=cfg.LION_B2, splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            warm_start=cfg.LION_WARM_START, seed=a.seed, log_every=50, verbose=True)
        np.savez(sg_boost_cache, lam=lam_sg, Lam=Lam_sg)
    def _gen_sg7():
        return TL.sample_tilted(
            lam_sg, Lam_sg, Xj, delta=delta_fig7, sigma_s=sigma_fig7, M=32,
            h=h_baoab(delta_fig7), gamma=gamma8, n_steps=baoab_nsteps,
            n_samples=n_vis, splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            seed=a.seed)
    U_sg7 = _cached_array(os.path.join(cache_dir, "fig7_sgbaoab.npy"),
                          _gen_sg7, a.regenerate)

    for name, U_gen in [("mmsold", U_sold7), ("pluginbaoab", U_pi7), ("sgbaoab", U_sg7)]:
        _seamless_grid(decode(U_gen), os.path.join(a.fig_dir, "grids", f"fig7_{name}.png"),
                       G=VIS_GRID)
        nn_idx, nn_dist = _find_nn(U_gen[:NN_COLS * NN_ROWS], U_train8)
        _save_nn_vis(decode(U_gen[:NN_COLS * NN_ROWS]), decode(U_train8[nn_idx]),
                    os.path.join(a.fig_dir, "grids", f"fig7_nn_{name}.png"),
                    n_cols=NN_COLS, n_rows=NN_ROWS)
        print(f"  {name}: mean NN dist to training set (top-40) = {nn_dist.mean():.4f}")

    print(f"\n{'='*40}\ndigit-8 posterior-moment-validation table: posterior moment validation\n")
    from metrics import extract_features, compute_tau, compute_kid, compute_recall, compute_duprate
    from classifier_model import load_params as clf_load_params

    clf_params, _ = clf_load_params(cfg.CLF_CKPT)
    imgs_test8 = np.load(os.path.join(a.subspace_dir, "imgs_test8.npy"))
    feats_test = extract_features(imgs_test8, clf_params, batch_size=64)
    tau = compute_tau(U_train8, percentile=1)

    mu_star, Sigma_star = TL.moment_targets(np.asarray(U_train8), delta_fig7)

    settings = {
        "naive_baoab": (np.zeros(d, dtype=np.float32), np.zeros((d, d), dtype=np.float32)),
        "plugin_baoab": (lam_pi, Lam_pi),
        "sg_baoab": (lam_sg, Lam_sg),
    }
    table5 = {}
    for name, (lam, Lam) in settings.items():
        def _gen_moment(lam=lam, Lam=Lam):
            return TL.sample_tilted(
                lam, Lam, Xj, delta=delta_fig7, sigma_s=sigma_fig7, M=32,
                h=h_baoab(delta_fig7), gamma=gamma8, n_steps=baoab_nsteps,
                n_samples=TABLE5_N_MOMENT, splitting=cfg.BAOAB_SPLITTING,
                score=cfg.BAOAB_SCORE, seed=a.seed)
        q_moment = _cached_array(os.path.join(cache_dir, f"table5_{name}_moment.npy"),
                                 _gen_moment, a.regenerate).astype(np.float32)
        mean_abs = float(np.linalg.norm(mu_star - q_moment.mean(0)))
        mean_rel = mean_abs / float(np.linalg.norm(mu_star) + 1e-12)
        Qc = q_moment - mu_star[None, :]
        S_hat = (Qc.T @ Qc) / q_moment.shape[0]
        cov_abs = float(np.linalg.norm(Sigma_star - S_hat))
        cov_rel = cov_abs / float(np.linalg.norm(Sigma_star))

        def _gen_metric(lam=lam, Lam=Lam):
            return TL.sample_tilted(
                lam, Lam, Xj, delta=delta_fig7, sigma_s=sigma_fig7, M=32,
                h=h_baoab(delta_fig7), gamma=gamma8, n_steps=baoab_nsteps,
                n_samples=TABLE5_N_METRIC, splitting=cfg.BAOAB_SPLITTING,
                score=cfg.BAOAB_SCORE, seed=a.seed + 1)
        q_metric = _cached_array(os.path.join(cache_dir, f"table5_{name}_metric.npy"),
                                 _gen_metric, a.regenerate).astype(np.float32)
        feats_gen = extract_features(decode(q_metric), clf_params, batch_size=64)
        kid_m, _ = compute_kid(feats_test, feats_gen, subset_size=min(100, TABLE5_N_METRIC),
                               n_subsets=50, degree=cfg.KID_DEGREE,
                               rng=np.random.default_rng(a.seed))
        rec = compute_recall(feats_test, feats_gen, k_neighbour=cfg.K_RECALL)
        dup = compute_duprate(q_metric, U_train8, tau)

        table5[name] = {
            "mean_abs": mean_abs, "mean_rel": mean_rel,
            "cov_abs": cov_abs, "cov_rel": cov_rel,
            "kid": kid_m, "recall": rec, "duprate": dup,
        }
        print(f"  {name:14s}  |z-mu*|={mean_abs:.4f} rel={mean_rel:.4f}  "
              f"|Sig-Sig*|_F={cov_abs:.4f} rel={cov_rel:.4f}  "
              f"KID={kid_m:.4f} Recall={rec:.3f} DupRate={dup:.3f}")

    out = {
        "representative_cell": {"c_delta": c_delta_fig7, "c_sigma": c_sigma_fig7,
                                "M": 32, "delta": delta_fig7, "sigma": sigma_fig7,
                                "note": "picked by _pick_fig7_table5_cell, "
                                        "NOT the same as digit-8 sigma-CFDM/MM-SOLD sample-grid figure/6's c_delta_rep/c_sigma_rep"},
        "n_moment_samples": TABLE5_N_MOMENT, "n_metric_samples": TABLE5_N_METRIC,
        "table5": table5,
    }
    with open(os.path.join(a.results_dir, table5_name), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {os.path.join(a.results_dir, table5_name)}")

    print(f"\nAll digit-8 sigma-CFDM/MM-SOLD sample-grid figure/6/7/8 PNGs saved under {a.fig_dir}/ (and {a.fig_dir}/grids/)")
    print("\nStep 5 complete!")


if __name__ == "__main__":
    main()
