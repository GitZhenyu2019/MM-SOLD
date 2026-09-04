"""
plot_2d_paper_figures.py
========================
Re-plot checkerboard and spirals paper figures from cached PKL results.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from run_experiment import get_dataset, _axis_limits

# ─── Settings ────────────────────────────────────────────────────────────────
N_TRAIN        = 500       # paper setting (for "vs training" scatter)
SEED           = 0
DEFAULT_LEGEND = 12        # enlarged from original 7
C_SOLD         = "#2980b9" # original MM-SOLD blue
C_REF          = "#e74c3c" # original red (σ-CFDM / Training)

RESULTS_DIR = os.path.join(_HERE, "results")
FIG_DIR     = os.path.join(_HERE, "figures_paper_2d")


# ─── Scatter panel ────────────────────────────────────────────────────────────

def scatter_panel(ax, pts_ref, pts_gen,
                  c_ref, c_gen, lbl_ref, lbl_gen,
                  title: str, legend_fontsize: int):
    """Exact style as original figures/ (commit b306a91); only legend fontsize differs."""
    kw_gen = dict(s=5, alpha=0.35, linewidths=0, rasterized=True)
    kw_ref = dict(s=5, alpha=0.55, linewidths=0, rasterized=True)

    ax.scatter(pts_gen[:, 0], pts_gen[:, 1],
               c=c_gen, label=lbl_gen, **kw_gen)
    ax.scatter(pts_ref[:, 0], pts_ref[:, 1],
               c=c_ref, label=lbl_ref, **kw_ref)

    xlo, xhi, ylo, yhi = _axis_limits(pts_ref)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.legend(markerscale=4, fontsize=legend_fontsize, loc="best")


# ─── Save one scatter figure ──────────────────────────────────────────────────

def save_scatter(pts_ref, pts_gen,
                 c_ref, c_gen, lbl_ref, lbl_gen,
                 title: str, path: str, legend_fontsize: int):
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter_panel(ax, pts_ref, pts_gen,
                  c_ref, c_gen, lbl_ref, lbl_gen,
                  title, legend_fontsize)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Plot one (M, σ) configuration ───────────────────────────────────────────

def plot_config(dname, res, X_train, M, sigma, legend_fontsize, out_dir):
    key = (M, sigma)
    Z_sold = res["samples_mmsold"].get(key)
    Z_cfdm = res["samples_scfdm"].get(key)

    if Z_sold is None:
        print(f"  [skip] {dname} M={M} sigma={sigma}: no MM-SOLD samples in pkl")
        return
    if Z_cfdm is None:
        print(f"  [skip] {dname} M={M} sigma={sigma}: no σ-CFDM samples in pkl")
        return

    w2t_sold = res["w2_target_mmsold"][res["M_list"].index(M),
                                        res["sigma_list"].index(sigma)]
    w2t_cfdm = res["w2_target_scfdm"][res["M_list"].index(M),
                                       res["sigma_list"].index(sigma)]

    tag   = f"M{M}_s{sigma}"
    title = f"M={M}, σ={sigma}"

    print(f"  M={M} σ={sigma}  W2_target: SOLD={w2t_sold:.4f}  CFDM={w2t_cfdm:.4f}")

    # A: MM-SOLD (blue) vs σ-CFDM (red)
    save_scatter(Z_cfdm, Z_sold,
                 C_REF, C_SOLD, "σ-CFDM", "MM-SOLD",
                 title,
                 os.path.join(out_dir, f"{dname}_vs_scfdm_{tag}.pdf"),
                 legend_fontsize)

    # B: MM-SOLD (blue) vs training data (red)
    save_scatter(X_train, Z_sold,
                 C_REF, C_SOLD, "Training samples", "MM-SOLD",
                 title,
                 os.path.join(out_dir, f"{dname}_vs_train_{tag}.pdf"),
                 legend_fontsize)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M",               type=int,   default=None,
                        help="Plot only this M (must also set --sigma)")
    parser.add_argument("--sigma",           type=float, default=None,
                        help="Plot only this sigma (must also set --M)")
    parser.add_argument("--legend_fontsize", type=int,   default=DEFAULT_LEGEND)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 55)
    print("2-D Paper Figures (from cached pkl)")
    print(f"  N_TRAIN={N_TRAIN}  legend_fontsize={args.legend_fontsize}")
    if args.M is not None:
        print(f"  Mode: single config  M={args.M}, σ={args.sigma}")
    else:
        print("  Mode: ALL cached (M, σ) combinations")
    print("=" * 55)

    for dname in ["checkerboard", "spirals"]:
        print(f"\n=== {dname} ===")

        pkl_path = os.path.join(RESULTS_DIR, f"{dname}_results.pkl")
        with open(pkl_path, "rb") as f:
            res = pickle.load(f)
        print(f"  Loaded {pkl_path}")
        print(f"  M_list    : {res['M_list']}")
        print(f"  sigma_list: {res['sigma_list']}")
        print(f"  N_GEN     : {next(iter(res['samples_mmsold'].values())).shape[0]}")

        X_train = get_dataset(dname, N_TRAIN, seed=SEED + 1)
        print(f"  Training data: {X_train.shape}  (N_TRAIN={N_TRAIN})")

        out_dir = os.path.join(FIG_DIR, dname)
        os.makedirs(out_dir, exist_ok=True)

        if args.M is not None:
            plot_config(dname, res, X_train, args.M, args.sigma,
                        args.legend_fontsize, out_dir)
        else:
            for M in res["M_list"]:
                for sigma in res["sigma_list"]:
                    plot_config(dname, res, X_train, M, sigma,
                                args.legend_fontsize, out_dir)

    print(f"\n=== Done.  Figures → {FIG_DIR} ===")


if __name__ == "__main__":
    main()
