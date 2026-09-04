"""
step5_generate_figure4.py
===================================
CelebA-HQ representative-sample figure: representative decoded sample grids for all 7
methods (MM-SOLD, sigma-CFDM, KNN-PFlow, SWF, Plug-in BAOAB, SG-BAOAB,
Latent DDPM).
"""
import argparse
import json
import os

import numpy as np

import config as cfg                                    # noqa: E402
from common import seamless_grid, decode_batched, gamma_baoab, _pick_contrastive_cell  # noqa: E402
from step4_select_and_decode import METHODS, _make_gen_fn   # noqa: E402

SHARED_METHODS = {"mmsold", "cfdm", "plugin_baoab", "sg_baoab"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR_DIFFAE)
    ap.add_argument("--fig_dir", default=cfg.FIG_DIR_DIFFAE)
    ap.add_argument("--engine_path", default=cfg.DIFFAE_TRT_ENGINE)
    ap.add_argument("--vis_grid", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c_delta_rep", type=float, default=None,
                    help="override the shared MM-SOLD/sigma-CFDM/Plug-in BAOAB/"
                         "SG-BAOAB cell instead of using _pick_contrastive_cell")
    ap.add_argument("--c_sigma_rep", type=float, default=None)
    ap.add_argument("--M_rep", type=int, default=None)
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    fig_dir = os.path.join(a.fig_dir, "smoke") if a.smoke_test else a.fig_dir
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Step 5 (diffae): CelebA-HQ representative-sample figure -- representative samples, all 7 methods")
    print("=" * 60)

    U_train190 = np.load(os.path.join(a.subspace_dir, "U_train190.npy"))
    mean512    = np.load(os.path.join(a.subspace_dir, "mean512.npy"))
    Vt         = np.load(os.path.join(a.subspace_dir, "Vt.npy"))
    Vb         = np.load(os.path.join(a.subspace_dir, "Vb.npy"))
    lambda_tail = np.load(os.path.join(a.subspace_dir, "lambda_tail.npy"))
    gamma190 = gamma_baoab(U_train190)

    print("\nBuilding image decoder (TensorRT DPM-Solver++(2M)) ...")
    from dpm_decode import make_renderer
    render_fn, _ctx = make_renderer(engine_path=a.engine_path, nsteps=cfg.DECODE_NSTEPS,
                                    batch=cfg.DECODE_BATCH)

    n_vis = a.vis_grid * a.vis_grid
    results_suffix = "_smoke" if a.smoke_test else ""
    gen_kwargs = (dict(n_generate=n_vis, sold_nsteps=20, cfdm_nsteps=20,
                       knnpf_nsteps=20, baoab_nsteps=20)
                 if a.smoke_test else {})

    # ── Shared cell for MM-SOLD / sigma-CFDM / Plug-in BAOAB / SG-BAOAB ─────
    mmsold_path = os.path.join(a.results_dir, "mmsold", f"results_mmsold{results_suffix}.json")
    cfdm_path = os.path.join(a.results_dir, "cfdm", f"results_cfdm{results_suffix}.json")
    if not (os.path.exists(mmsold_path) and os.path.exists(cfdm_path)):
        raise FileNotFoundError(
            f"CelebA-HQ representative-sample figure's shared MM-SOLD/sigma-CFDM/Plug-in BAOAB/SG-BAOAB cell "
            f"needs both {mmsold_path} and {cfdm_path} -- run step2a and step2b first.")
    mmsold_grid = json.load(open(mmsold_path))["grid"]
    cfdm_grid = json.load(open(cfdm_path))["grid"]
    shared_cell = _pick_contrastive_cell(mmsold_grid, cfdm_grid,
                                         cfg.REP_CELL_DUPRATE_CAP, cfg.REP_CELL_RECALL_FLOOR)
    if a.c_delta_rep is not None:
        shared_cell = dict(shared_cell)
        shared_cell["c_delta"] = a.c_delta_rep
    if a.c_sigma_rep is not None:
        shared_cell = dict(shared_cell)
        shared_cell["c_sigma"] = a.c_sigma_rep
    if a.M_rep is not None:
        shared_cell = dict(shared_cell)
        shared_cell["M"] = a.M_rep
    if a.c_delta_rep is not None or a.c_sigma_rep is not None or a.M_rep is not None:
        from common import delta_sigma
        meta = json.load(open(os.path.join(a.subspace_dir, "meta.json")))
        shared_cell["delta"], shared_cell["sigma"] = delta_sigma(
            shared_cell["c_delta"], shared_cell["c_sigma"], meta["NN"], dsub=cfg.PROJ_K)
    print(f"\nShared representative cell (MM-SOLD best-KID, "
          f"DupRate<={cfg.REP_CELL_DUPRATE_CAP}, Recall>={cfg.REP_CELL_RECALL_FLOOR}, "
          f"contrasted against sigma-CFDM): c_delta={shared_cell['c_delta']} "
          f"c_sigma={shared_cell['c_sigma']} M={shared_cell['M']}")

    for method_key, label, subdir, results_fname, _timing_fname in METHODS:
        rpath = os.path.join(a.results_dir, subdir, results_fname.replace(".json", f"{results_suffix}.json"))
        if not os.path.exists(rpath):
            print(f"[MISSING] {label}: {rpath} -- skipping")
            continue
        if method_key in SHARED_METHODS:
            cell = shared_cell
            print(f"\n{label}: sampling {n_vis} at the SHARED cell "
                  f"c_delta={cell['c_delta']} c_sigma={cell['c_sigma']} M={cell['M']} ...")
        else:
            grid = json.load(open(rpath))["grid"]
            cell = min(grid.values(), key=lambda c: c["kid_mean"])
            print(f"\n{label}: sampling {n_vis} at its OWN best-KID cell "
                  f"c_delta={cell.get('c_delta')} c_sigma={cell.get('c_sigma')} "
                  f"M={cell.get('M')} ...")
        cache_dir = (os.path.join(a.results_dir, subdir,
                                  "tilting_cache_smoke" if a.smoke_test else "tilting_cache")
                    if method_key == "sg_baoab" else None)
        gen_fn = _make_gen_fn(method_key, cell, U_train190, gamma190,
                              cache_dir=cache_dir, **gen_kwargs)
        U_vis = gen_fn(a.seed)[:n_vis]
        imgs = decode_batched(U_vis, mean512, Vt, Vb, lambda_tail, render_fn,
                              is_subspace=True, batch=cfg.DECODE_BATCH, seed=a.seed)
        out_path = os.path.join(fig_dir, f"fig4_{method_key}.png")
        seamless_grid(imgs.astype(np.float32) / 255.0, out_path, G=a.vis_grid)
        print(f"  Saved: {out_path}")

    # ── Latent DDPM: reuse step3's already-generated repeat0(.npy/_smoke.npy) ─
    ddpm_gen_path = os.path.join(a.results_dir, "ddpm", "generated",
                                 f"repeat0{results_suffix}.npy")
    if os.path.exists(ddpm_gen_path):
        print(f"\nLatent DDPM: decoding first {n_vis} of {ddpm_gen_path} ...")
        Z_ddpm = np.load(ddpm_gen_path)[:n_vis]
        imgs = decode_batched(Z_ddpm, mean512, Vt, Vb, lambda_tail, render_fn,
                              is_subspace=False, batch=cfg.DECODE_BATCH, seed=a.seed)
        out_path = os.path.join(fig_dir, "fig4_latent_ddpm.png")
        seamless_grid(imgs.astype(np.float32) / 255.0, out_path, G=a.vis_grid)
        print(f"  Saved: {out_path}")
    else:
        print(f"\n[MISSING] Latent DDPM: {ddpm_gen_path} -- run step3 first")

    print(f"\nAll CelebA-HQ representative-sample figure panels saved under {fig_dir}/")
    print("\nStep 5 (diffae) complete!")


if __name__ == "__main__":
    main()
