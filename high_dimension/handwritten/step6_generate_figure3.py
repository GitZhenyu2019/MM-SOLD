"""
step6_generate_figure3.py
===============================
this section main text, digit-8 representative-sample figure: "Representative digit-8 samples" -- Real
samples plus all 7 generative methods (MM-SOLD, Plug-in BAOAB, SG-BAOAB,
Latent DDPM, sigma-CFDM, KNN-PFlow, SWF).
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import config as cfg                                    # noqa: E402
import tilting_lion as TL                                   # noqa: E402
from common import (                                     # noqa: E402
    load_nrae, decode_from_subspace, delta_sigma, h_sold, h_baoab,
    gamma_baoab, sample_class_overdamped_manifold, best_cell_filtered,
    seamless_grid as _seamless_grid_raw,
)
from cfdm_sampling import sample_cfdm                         # noqa: E402
from knn_pflow_sampling import sample_knn_pflow                # noqa: E402
from swf_sampling import sample_swf                             # noqa: E402
from ddpm_model import make_cosine_schedule, ddim_sample, load_ddpm  # noqa: E402


VIS_GRID = 8   # 8x8 = 64 images per panel (same convention as Fig 5/7)


def _seamless_grid(images: np.ndarray, path: str, G: int = VIS_GRID):
    """Thin wrapper around common.seamless_grid (shared with step2a-f's
    --make_grids and step5's Figures 5/7), adding this script's logging."""
    _seamless_grid_raw(images, path, G=G)
    print(f"  Saved: {path}")


def _cached_array(path: str, fn, regenerate: bool):
    if os.path.exists(path) and not regenerate:
        return np.load(path)
    arr = np.asarray(fn())
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)
    return arr


def _best_kid_cell(results_path):
    res = json.load(open(results_path))
    cells = list(res["grid"].values())
    return min(cells, key=lambda c: c["kid_mean"])


def _pick_contrastive_cell(mmsold_grid, cfdm_grid, duprate_cap, recall_floor):
    mmsold_cells = list(mmsold_grid.values())
    ok = [c for c in mmsold_cells
          if c["duprate_mean"] <= duprate_cap and c["recall_mean"] >= recall_floor]
    if not ok:
        print(f"  [WARNING] no MM-SOLD cell has duprate_mean<={duprate_cap} and "
              f"recall_mean>={recall_floor}; falling back to best_cell_filtered "
              f"(sigma-CFDM contrast not enforced)")
        return best_cell_filtered(mmsold_cells, duprate_cap=duprate_cap,
                                  recall_floor=recall_floor)

    cfdm_by_sm = {(c["c_sigma"], c["M"]): c for c in cfdm_grid.values()}

    def recall_gap(c):
        cf = cfdm_by_sm.get((c["c_sigma"], c["M"]))
        return (c["recall_mean"] - cf["recall_mean"]) if cf is not None else -1e9

    best = max(ok, key=recall_gap)
    cf = cfdm_by_sm.get((best["c_sigma"], best["M"]))
    print(f"  contrastive pick: MM-SOLD Recall={best['recall_mean']:.4f} vs "
          f"sigma-CFDM Recall={cf['recall_mean']:.4f} at the same cell "
          f"(gap={recall_gap(best):+.4f}); sigma-CFDM KID={cf['kid_mean']:.4f}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR, "subspace"))
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR)
    ap.add_argument("--fig_dir", default=cfg.FIG_DIR)
    ap.add_argument("--ckpt_dir", default=cfg.CKPT_DIR)
    ap.add_argument("--nrae_ckpt", default=cfg.NRAE_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c_delta_rep", type=float, default=None,
                    help="override the auto-picked shared representative c_delta "
                         "(MM-SOLD/sigma-CFDM/Plug-in BAOAB/SG-BAOAB)")
    ap.add_argument("--c_sigma_rep", type=float, default=None,
                    help="override the auto-picked shared representative c_sigma")
    ap.add_argument("--M_rep", type=int, default=None,
                    help="override the auto-picked shared representative M")
    ap.add_argument("--regenerate", action="store_true",
                    help="force fresh sampling instead of reusing results/fig_cache/")
    ap.add_argument("--smoke_test", action="store_true",
                    help="read every step2*/step3's *_smoke outputs and use a tiny grid/BAOAB config")
    a = ap.parse_args()

    global VIS_GRID
    results_suffix = ""
    sold_nsteps = cfg.SOLD_NSTEPS
    baoab_nsteps = cfg.BAOAB_NSTEPS
    lion_steps, lion_chains, lion_iters = cfg.LION_N_STEPS, cfg.LION_N_CHAINS, cfg.LION_N_ITERS
    ddpm_ckpt_name = "ddpm_digit8_top40.pkl"
    if a.smoke_test:
        VIS_GRID = 2
        sold_nsteps = 20
        baoab_nsteps = 20
        lion_steps, lion_chains, lion_iters = 10, 8, 5
        results_suffix = "_smoke"
        ddpm_ckpt_name = "ddpm_digit8_top40_smoke.pkl"
        a.fig_dir = os.path.join(a.fig_dir, "smoke")
        print("\n*** SMOKE TEST MODE ***\n")

    os.makedirs(a.fig_dir, exist_ok=True)
    os.makedirs(os.path.join(a.fig_dir, "grids"), exist_ok=True)
    cache_dir = os.path.join(a.results_dir, "fig_cache" + results_suffix)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("Step 6: digit-8 representative-sample figure -- representative samples, all methods")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    U_train8 = np.load(os.path.join(a.subspace_dir, "U_train8.npy"))
    mean_8   = np.load(os.path.join(a.subspace_dir, "mean_8.npy"))
    V        = np.load(os.path.join(a.subspace_dir, "proj_V.npy"))
    imgs_test8 = np.load(os.path.join(a.subspace_dir, "imgs_test8.npy"))
    meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
    NN8 = meta["NN8"]
    d = U_train8.shape[1]
    print(f"  U_train8 {U_train8.shape}  NN8={NN8:.5f}")

    nrae_model, nrae_params, n_dct = load_nrae(a.nrae_ckpt, seed=0)

    def decode(U):
        return decode_from_subspace(U, mean_8, V, nrae_model, nrae_params, n_dct)

    n_vis = VIS_GRID * VIS_GRID
    mean0 = jnp.zeros(d, dtype=jnp.float32)
    S_eye = jnp.eye(d, dtype=jnp.float32)
    gamma8 = gamma_baoab(U_train8)
    Xj = jnp.asarray(U_train8, dtype=jnp.float32)

    panels = {}   # name -> (H,W)-image-array canvas-ready decoded images

    # ── Real samples: actual held-out test images, no reconstruction ───────
    print("\n[Real] subset of the 300 held-out test images")
    real_imgs = imgs_test8[:n_vis]
    panels["real"] = real_imgs

    mmsold_res = json.load(open(os.path.join(
        a.results_dir, "mmsold", f"results_mmsold{results_suffix}.json")))
    cfdm_res = json.load(open(os.path.join(
        a.results_dir, "cfdm", f"results_cfdm{results_suffix}.json")))
    best = _pick_contrastive_cell(mmsold_res["grid"], cfdm_res["grid"],
                                  cfg.REP_CELL_DUPRATE_CAP, cfg.REP_CELL_RECALL_FLOOR)
    c_delta_rep = a.c_delta_rep if a.c_delta_rep is not None else best["c_delta"]
    c_sigma_rep = a.c_sigma_rep if a.c_sigma_rep is not None else best["c_sigma"]
    M_rep = a.M_rep if a.M_rep is not None else best["M"]
    delta_rep, sigma_rep = delta_sigma(c_delta_rep, c_sigma_rep, NN8, dsub=cfg.PROJ_K)
    print(f"\nShared representative cell (MM-SOLD best-KID, "
          f"DupRate<={cfg.REP_CELL_DUPRATE_CAP}, Recall>={cfg.REP_CELL_RECALL_FLOOR}, "
          f"contrasted against sigma-CFDM): "
          f"c_delta={c_delta_rep} c_sigma={c_sigma_rep} M={M_rep}  "
          f"(KID={best['kid_mean']:.4f} DupRate={best['duprate_mean']:.4f} "
          f"Recall={best['recall_mean']:.4f})")

    # ── MM-SOLD @ shared cell ────────────────────────────────────────────────
    print("\n[MM-SOLD] shared representative cell")

    def _gen_sold():
        U, _ = sample_class_overdamped_manifold(
            X_class=Xj, n_particles=max(n_vis, cfg.PROJ_K + 1), nsteps=sold_nsteps,
            h=h_sold(delta_rep), sigma_gmm=delta_rep, sigma_smoothing=sigma_rep, M=M_rep,
            shared_noise=cfg.SOLD_SHARED_NOISE, discretization=cfg.SOLD_DISCRETIZ,
            fixed_noise=cfg.SOLD_FIXED_NOISE, seed=a.seed)
        return np.array(U, dtype=np.float32)[:n_vis]
    U_sold = _cached_array(os.path.join(cache_dir, "fig3_mmsold.npy"), _gen_sold, a.regenerate)
    panels["mmsold"] = decode(U_sold)

    # ── sigma-CFDM @ shared cell ─────────────────────────────────────────────
    print("\n[sigma-CFDM] shared representative cell")

    def _gen_cfdm():
        return sample_cfdm(
            U_train8, mean0, S_eye, S_eye, n_particles=n_vis, sigma=sigma_rep,
            M=M_rep, shared_noise=cfg.CFDM_SHARED_NOISE, nsteps=cfg.CFDM_NSTEPS,
            batch_size=cfg.CFDM_BATCH, seed=a.seed)
    U_cfdm = _cached_array(os.path.join(cache_dir, "fig3_cfdm.npy"), _gen_cfdm, a.regenerate)
    panels["cfdm"] = decode(U_cfdm)

    # ── KNN-PFlow: own best-KID cell (no shared (c_delta,c_sigma,M) axis) ───
    print("\n[KNN-PFlow] own best-KID cell")
    cell = _best_kid_cell(os.path.join(a.results_dir, "knnpflow",
                                       f"results_knnpflow{results_suffix}.json"))
    n_train = U_train8.shape[0]
    K_eff = n_train if cell["K"] == "N" else int(cell["K"])
    print(f"  c_max={cell['c_max']} c_min={cell['c_min']} K={cell['K']} "
          f"solver={cell['solver']} (KID={cell['kid_mean']:.4f})")

    def _gen_knnpf(cell=cell, K_eff=K_eff):
        return sample_knn_pflow(
            U_train8, mean0, S_eye, S_eye, n_particles=n_vis,
            sigma_max=cell["sigma_max"], sigma_min=cell["sigma_min"],
            num_steps=cfg.KNNPF_NSTEPS, K=K_eff, solver=cell["solver"],
            batch_size=min(500, n_vis), seed=a.seed)
    U_knnpf = _cached_array(os.path.join(cache_dir, "fig3_knnpflow.npy"), _gen_knnpf, a.regenerate)
    panels["knnpflow"] = decode(U_knnpf)

    # ── SWF: own best-KID cell (no shared (c_delta,c_sigma,M) axis) ─────────
    print("\n[SWF] own best-KID cell")
    cell = _best_kid_cell(os.path.join(a.results_dir, "swf",
                                       f"results_swf{results_suffix}.json"))
    print(f"  n_SWF={cell['n_SWF']} R={cell['R']} eta={cell['eta']} "
          f"init={cell['init']} (KID={cell['kid_mean']:.4f})")

    def _gen_swf(cell=cell):
        return sample_swf(
            U_train8, mean0, S_eye, S_eye, n_particles=n_vis,
            num_steps=cell["n_SWF"], num_projections=cell["R"], step_size=cell["eta"],
            init=cell["init"], nn_dist=NN8, seed=a.seed)
    U_swf = _cached_array(os.path.join(cache_dir, "fig3_swf.npy"), _gen_swf, a.regenerate)
    panels["swf"] = decode(U_swf)

    # ── Plug-in BAOAB @ shared cell ──────────────────────────────────────────
    print("\n[Plug-in BAOAB] shared representative cell")
    lam_pi, Lam_pi = TL.plugin_tilting(jax.random.PRNGKey(a.seed), Xj, delta_rep,
                                       sigma_rep, M_rep, score=cfg.BAOAB_SCORE)

    def _gen_pi():
        return TL.sample_tilted(
            lam_pi, Lam_pi, Xj, delta=delta_rep, sigma_s=sigma_rep, M=M_rep,
            h=h_baoab(delta_rep), gamma=gamma8, n_steps=baoab_nsteps,
            n_samples=n_vis, splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            seed=a.seed)
    U_pi = _cached_array(os.path.join(cache_dir, "fig3_pluginbaoab.npy"), _gen_pi, a.regenerate)
    panels["pluginbaoab"] = decode(U_pi)

    # ── SG-BAOAB @ shared cell ──────────────────────────────
    print("\n[SG-BAOAB] shared representative cell")
    sg_tilting_cache_dir = "tilting_cache_smoke" if a.smoke_test else "tilting_cache"
    sg_cache_path = os.path.join(a.results_dir, "sgbaoab", sg_tilting_cache_dir,
                                 f"cell_cd{c_delta_rep:.2f}_cs{c_sigma_rep:.2f}_M{M_rep}.npz")
    if os.path.exists(sg_cache_path):
        dd = np.load(sg_cache_path)
        lam_sg, Lam_sg = dd["lam"], dd["Lam"]
        print(f"  reusing cached (lambda, Lambda) from {sg_cache_path}")
    else:
        print(f"  no cache at {sg_cache_path}, fitting now ...")
        lam_sg, Lam_sg, _hist = TL.estimate_tilting_lion(
            Xj, delta=delta_rep, sigma_s=sigma_rep, M=M_rep, h=h_baoab(delta_rep),
            gamma=gamma8, n_steps=lion_steps, n_chains=lion_chains,
            n_iters=lion_iters, lr_rel=cfg.LION_LR_REL, b1=cfg.LION_B1,
            b2=cfg.LION_B2, splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            warm_start=cfg.LION_WARM_START, seed=a.seed, log_every=50, verbose=True)

    def _gen_sg():
        return TL.sample_tilted(
            lam_sg, Lam_sg, Xj, delta=delta_rep, sigma_s=sigma_rep, M=M_rep,
            h=h_baoab(delta_rep), gamma=gamma8, n_steps=baoab_nsteps,
            n_samples=n_vis, splitting=cfg.BAOAB_SPLITTING, score=cfg.BAOAB_SCORE,
            seed=a.seed)
    U_sg = _cached_array(os.path.join(cache_dir, "fig3_sgbaoab.npy"), _gen_sg, a.regenerate)
    panels["sgbaoab"] = decode(U_sg)

    # ── Latent DDPM: trained checkpoint, no grid ────────────────────────────
    print("\n[Latent DDPM] trained checkpoint")
    ddpm_ckpt_path = os.path.join(a.ckpt_dir, ddpm_ckpt_name)
    ddpm_params, info = load_ddpm(ddpm_ckpt_path)
    T = info["T"]
    ac_np, _ = make_cosine_schedule(T)
    mu_w, std_w = info["whiten_mu"], info["whiten_std"]

    def _gen_ddpm():
        rng = jax.random.PRNGKey(a.seed)
        Z_w = ddim_sample(ddpm_params, rng, alphas_cumprod=ac_np, n_samples=n_vis,
                          latent_dim=d, hidden=info["hidden"], n_layers=info["n_layers"],
                          T=T, ddim_steps=cfg.DDIM_STEPS)
        return np.array(Z_w) * std_w + mu_w
    U_ddpm = _cached_array(os.path.join(cache_dir, "fig3_ddpm.npy"), _gen_ddpm, a.regenerate)
    panels["ddpm"] = decode(U_ddpm.astype(np.float32))

    # ── Save individual panel PNGs ──────────────────────────────────────────
    print(f"\n{'='*40}\nSaving digit-8 representative-sample figure panels\n")
    for name, imgs in panels.items():
        _seamless_grid(imgs, os.path.join(a.fig_dir, "grids", f"fig3_{name}.png"), G=VIS_GRID)

    # ── Combined 2x4 composite, matching the paper's panel layout ──────────
    layout = [
        ("real", "Real samples"), ("mmsold", "MM-SOLD"),
        ("pluginbaoab", "Plug-in BAOAB"), ("sgbaoab", "SG-BAOAB"),
        ("ddpm", "Latent DDPM"), ("cfdm", "sigma-CFDM"),
        ("knnpflow", "KNN-PFlow"), ("swf", "SWF"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(4 * 2.2, 2 * 2.4))
    for ax, (key, title) in zip(axes.flat, layout):
        H, W = panels[key].shape[1], panels[key].shape[2]
        G = VIS_GRID
        canvas = np.ones((G * H, G * W), dtype=np.float32)
        for k in range(min(G * G, len(panels[key]))):
            r, c = k // G, k % G
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = panels[key][k]
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    plt.tight_layout()
    combined_path = os.path.join(a.fig_dir, "fig3_combined.png")
    plt.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {combined_path}")

    print(f"\nAll digit-8 representative-sample figure PNGs saved under {a.fig_dir}/ (panels in {a.fig_dir}/grids/)")
    print("\nStep 6 complete!")


if __name__ == "__main__":
    main()
