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

from run_experiment import CACHE_VERSION, _axis_limits

# ─── Settings ────────────────────────────────────────────────────────────────
DEFAULT_LEGEND = 12        # legend font size
C_SOLD         = "#2980b9" # MM-SOLD (blue)
C_REF          = "#e74c3c" # σ-CFDM / training (red)

RESULTS_DIR = os.path.join(_HERE, "results")
FIG_DIR     = os.path.join(_HERE, "figures_paper_2d")


# ─── Scatter panel ────────────────────────────────────────────────────────────

def scatter_panel(ax, pts_ref, pts_gen,
                  c_ref, c_gen, lbl_ref, lbl_gen,
                  title: str, legend_fontsize: int):
    """Scatter panel used for the paper figures."""
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
    if res.get("config", {}).get("smoke_test"):
        title += " (smoke test)"

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

def plot_results(dname, res, fig_dir=FIG_DIR, legend_fontsize=DEFAULT_LEGEND,
                 M=None, sigma=None):
    """Render panels from the actual training set and samples saved by the runner."""
    if (res.get("config", {}).get("cache_version") != CACHE_VERSION
            or "X_train" not in res):
        raise ValueError("Cached results were produced with different settings. "
                         "Regenerate with run_experiment.py before plotting.")
    X_train = res["X_train"]
    print(f"  Training data: {X_train.shape}; settings: {res['config']}")
    out_dir = os.path.join(fig_dir, dname)
    os.makedirs(out_dir, exist_ok=True)
    configs = ([(M, sigma)] if M is not None else
               [(m, s) for m in res["M_list"] for s in res["sigma_list"]])
    for m, s in configs:
        plot_config(dname, res, X_train, m, s, legend_fontsize, out_dir)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--M",               type=int,   default=None,
                        help="Plot only this M (must also set --sigma)")
    parser.add_argument("--sigma",           type=float, default=None,
                        help="Plot only this sigma (must also set --M)")
    parser.add_argument("--legend_fontsize", type=int,   default=DEFAULT_LEGEND)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=FIG_DIR)
    parser.add_argument("--datasets", nargs="+", default=["checkerboard", "spirals"])
    args = parser.parse_args(argv)
    if (args.M is None) != (args.sigma is None):
        parser.error("--M and --sigma must be supplied together")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 55)
    print("2-D Paper Figures (from cached pkl)")
    print(f"  legend_fontsize={args.legend_fontsize}")
    if args.M is not None:
        print(f"  Mode: single config  M={args.M}, σ={args.sigma}")
    else:
        print("  Mode: ALL cached (M, σ) combinations")
    print("=" * 55)

    for dname in args.datasets:
        print(f"\n=== {dname} ===")

        pkl_path = os.path.join(args.results_dir, f"{dname}_results.pkl")
        if not os.path.exists(pkl_path):
            parser.error(f"Missing {pkl_path}. Run run_experiment.py first; "
                         "it generates both caches and paper panels from scratch.")
        with open(pkl_path, "rb") as f:
            res = pickle.load(f)
        print(f"  Loaded {pkl_path}")
        print(f"  M_list    : {res['M_list']}")
        print(f"  sigma_list: {res['sigma_list']}")
        print(f"  N_GEN     : {next(iter(res['samples_mmsold'].values())).shape[0]}")

        plot_results(dname, res, args.output_dir, args.legend_fontsize,
                     M=args.M, sigma=args.sigma)

    print(f"\n=== Done.  Figures → {args.output_dir} ===")


if __name__ == "__main__":
    main()
