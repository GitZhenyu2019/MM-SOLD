"""
step2_run_ecm.py
=====================
Minimum-ECM classifier(s) for the classification experiment
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import jax

import config as cfg  # noqa: E402
import tilting_lion as TL  # noqa: E402
from common import (  # noqa: E402
    build_class_data, delta_sigma, h_baoab, gamma_baoab,
    compute_class_energy, calibrate_biases, compute_metrics,
)


def load_latents(latents_dir):
    d = latents_dir
    return dict(
        U_train=np.load(os.path.join(d, "U_train.npy")),
        y_train=np.load(os.path.join(d, "y_train.npy")),
        U_val=np.load(os.path.join(d, "U_val.npy")),
        y_val=np.load(os.path.join(d, "y_val.npy")),
        U_test=np.load(os.path.join(d, "U_test.npy")),
        y_test=np.load(os.path.join(d, "y_test.npy")),
    )


def fit_class(estimator, U_c, delta_c, sigma_c, M, args, key):
    mu_star_c, Sigma_star_c = TL.moment_targets(U_c, delta_c)
    diag = {}
    if estimator == "plugin":
        lam_c, Lam_c = TL.plugin_tilting(key, U_c, delta_c, sigma_c, M, score="full")
    else:  # dual
        hb = h_baoab(delta_c)
        gc = gamma_baoab(U_c)
        seed = int(jax.random.randint(key, (), 0, 2 ** 31 - 1))
        lam_c, Lam_c, hist = TL.estimate_tilting_lion(
            U_c, delta=delta_c, sigma_s=sigma_c, M=M, h=hb, gamma=gc,
            n_steps=args.lion_steps, n_chains=args.lion_chains,
            n_iters=args.lion_iters, lr_rel=args.lion_lr_rel,
            score="full", warm_start=True, seed=seed, verbose=False,
        )
        if hist:
            last = hist[-1]
            floor = TL.res_cov_floor(Sigma_star_c, args.lion_chains)
            diag = {"res_mean": last["res_mean"], "res_cov": last["res_cov"],
                    "res_cov_floor": floor,
                    "res_cov_ratio": (last["res_cov"] / floor
                                     if floor and np.isfinite(floor) else None),
                    "gamma": gc, "h_baoab": hb}
    return lam_c, Lam_c, mu_star_c, diag


def evaluate_cell(estimator, c_delta, c_sigma, class_data, classes,
                   U_val, y_val, U_test, y_test, args, key):
    """Fit every class at this (c_delta, c_sigma) cell; return metrics + per-class params."""
    n_classes = cfg.N_CLASSES
    E_val = np.zeros((len(y_val), n_classes), dtype=np.float64)
    E_test = np.zeros((len(y_test), n_classes), dtype=np.float64)
    per_class = {}
    diagnostics = {}

    for c in classes:
        cd = class_data[c]
        delta_c, sigma_c = delta_sigma(c_delta, c_sigma, cd["NN"], dsub=args.proj_k)
        key, k_fit = jax.random.split(key)
        lam_c, Lam_c, mu_star_c, diag = fit_class(
            estimator, cd["U"], delta_c, sigma_c, args.M, args, k_fit)

        key, k_v, k_t = jax.random.split(key, 3)
        E_val[:, c] = compute_class_energy(
            k_v, U_val, mu_star_c, lam_c, Lam_c, cd["U"], cd["u_norm2"],
            delta_c, sigma_c, args.M)
        E_test[:, c] = compute_class_energy(
            k_t, U_test, mu_star_c, lam_c, Lam_c, cd["U"], cd["u_norm2"],
            delta_c, sigma_c, args.M)
        per_class[c] = {"delta_c": delta_c, "sigma_c": sigma_c,
                        "lambda_c": np.asarray(lam_c).tolist(),
                        "Lambda_c": np.asarray(Lam_c).tolist()}
        if diag:
            diagnostics[c] = diag

    # classes not fit this run (e.g. --classes subset) get +inf energy so
    # they never win argmin -- only meaningful for --classes smoke tests.
    fit_mask = np.zeros(n_classes, dtype=bool)
    for c in classes:
        fit_mask[c] = True
    E_val[:, ~fit_mask] = np.inf
    E_test[:, ~fit_mask] = np.inf

    biases = calibrate_biases(E_val, y_val, n_classes)
    y_pred_val = (E_val + biases[None, :]).argmin(axis=1)
    y_pred_test = (E_test + biases[None, :]).argmin(axis=1)
    val_metrics = compute_metrics(y_val, y_pred_val, n_classes)
    test_metrics = compute_metrics(y_test, y_pred_test, n_classes)

    # cell-level convergence summary (dual only): worst-converged class,
    # i.e. the one whose res_cov is farthest (in multiples of its own iid
    # floor) from what P chains could in principle achieve.
    cell_diag = {}
    if diagnostics:
        ratios = {c: d["res_cov_ratio"] for c, d in diagnostics.items()
                 if d.get("res_cov_ratio") is not None}
        if ratios:
            worst_c = max(ratios, key=ratios.get)
            cell_diag = {"per_class": diagnostics,
                        "worst_class": worst_c,
                        "worst_res_cov_ratio": ratios[worst_c]}
    return val_metrics, test_metrics, biases, per_class, cell_diag


CONVERGENCE_WARN_RATIO = 3.0  # flag a class if its final res_cov exceeds
                              # this many multiples of the iid sampling floor


def _print_convergence(cell_diag, indent="    "):
    """Print a one-line convergence summary for the dual estimator (no-op
    for plug-in, where `cell_diag` is empty -- there is nothing to converge,
    it's a closed-form solve)."""
    if not cell_diag:
        return
    worst_c = cell_diag["worst_class"]
    ratio = cell_diag["worst_res_cov_ratio"]
    d = cell_diag["per_class"][worst_c]
    tag = "OK" if ratio <= CONVERGENCE_WARN_RATIO else "POOR CONVERGENCE"
    print(f"{indent}convergence: worst class={worst_c}  "
          f"res_cov={d['res_cov']:.4f} floor={d['res_cov_floor']:.4f} "
          f"ratio={ratio:.2f}x  [{tag}]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimator", choices=["plugin", "dual"], required=True)
    ap.add_argument("--latents_dir", default=cfg.LATENT_DIR)
    ap.add_argument("--out", default=None,
                    help="default: results/ecm_<estimator>")
    ap.add_argument("--classes", default="",
                    help="comma-separated subset, e.g. 0,1 for a quick test "
                         "(default: all 10)")
    ap.add_argument("--c_grid", default=",".join(str(c) for c in cfg.C_GRID),
                    help="shared grid for both c_delta and c_sigma")
    ap.add_argument("--proj_k", type=int, default=cfg.PROJ_K)
    ap.add_argument("--M", type=int, default=cfg.M_MC)
    ap.add_argument("--shared_cell", default=None,
                    help="path to a selected_cell.json to reuse instead of "
                         "grid-searching (dual estimator default: "
                         "<out_dir_for_plugin>/selected_cell.json)")
    ap.add_argument("--own_grid", action="store_true",
                    help="for --estimator dual, force an independent "
                         "(expensive) grid search instead of reusing plug-in's cell")
    # Lion hyperparameters (not specified by the paper; see config.py)
    ap.add_argument("--lion_chains", type=int, default=cfg.LION_N_CHAINS)
    ap.add_argument("--lion_steps", type=int, default=cfg.LION_N_STEPS)
    ap.add_argument("--lion_iters", type=int, default=cfg.LION_N_ITERS)
    ap.add_argument("--lion_lr_rel", type=float, default=cfg.LION_LR_REL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(cfg.RESULTS_DIR, f"ecm_{args.estimator}")
    os.makedirs(args.out, exist_ok=True)

    classes = ([int(c) for c in args.classes.split(",") if c != ""]
               if args.classes else list(range(cfg.N_CLASSES)))
    c_grid = [float(x) for x in args.c_grid.split(",") if x]
    cells = list(itertools.product(c_grid, c_grid))

    print("=" * 60)
    print(f"Step 2: MM Target Min. ECM Classifier -- estimator={args.estimator}")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"  classes    : {classes}")
    print(f"  c_delta/c_sigma grid: {c_grid}  ({len(cells)} cells)")
    print(f"  M          : {args.M}  (fixed)")

    # ── decide: grid search, or reuse a shared cell? ────────────────────────
    reuse_path = None
    if args.estimator == "dual" and not args.own_grid:
        reuse_path = args.shared_cell or os.path.join(
            cfg.RESULTS_DIR, "ecm_plugin", "selected_cell.json")
        if os.path.exists(reuse_path):
            print(f"  reuse cell : {reuse_path}")
        else:
            print(f"  [NOTE] {reuse_path} not found -- falling back to a full "
                  f"grid search for the dual estimator (slow). Run "
                  f"`--estimator plugin` first, or pass --shared_cell.")
            reuse_path = None
    elif args.shared_cell:
        reuse_path = args.shared_cell

    if args.dry_run:
        n_fits = len(cells) * len(classes) if reuse_path is None else len(classes)
        print(f"  dry run: would fit {n_fits} (class, cell) combinations")
        return

    data = load_latents(args.latents_dir)
    class_data = build_class_data(data["U_train"], data["y_train"], cfg.N_CLASSES)
    print("\n  NN_c per class:")
    for c in classes:
        print(f"    class {c}: NN_c={class_data[c]['NN']:.4f}")

    key = jax.random.PRNGKey(args.seed)
    results = []
    t0 = time.perf_counter()

    if reuse_path is not None:
        with open(reuse_path) as f:
            sel = json.load(f)
        c_delta, c_sigma = sel["c_delta"], sel["c_sigma"]
        print(f"\n[fit] reusing (c_delta={c_delta}, c_sigma={c_sigma}) from "
              f"{reuse_path}, fitting {args.estimator} once per class ...")
        val_m, test_m, biases, per_class, cell_diag = evaluate_cell(
            args.estimator, c_delta, c_sigma, class_data, classes,
            data["U_val"], data["y_val"], data["U_test"], data["y_test"],
            args, key)
        best = {"c_delta": c_delta, "c_sigma": c_sigma,
               "val_acc": val_m["acc"], "val_f1": val_m["macro_f1"],
               "test_acc": test_m["acc"], "test_f1": test_m["macro_f1"],
               "per_class_acc_test": test_m["per_class_acc"],
               "biases": biases.tolist(), "convergence": cell_diag}
        _print_convergence(cell_diag)
        results.append(best)
    else:
        best = None
        for i, (c_delta, c_sigma) in enumerate(cells, 1):
            t = time.perf_counter()
            key, k_cell = jax.random.split(key)  # independent randomness per cell
            print(f"[{i}/{len(cells)}] c_delta={c_delta:.2f} c_sigma={c_sigma:.2f} ...",
                  flush=True)
            try:
                val_m, test_m, biases, per_class, cell_diag = evaluate_cell(
                    args.estimator, c_delta, c_sigma, class_data, classes,
                    data["U_val"], data["y_val"], data["U_test"], data["y_test"],
                    args, k_cell)
            except Exception as e:  # a cell may diverge (dual estimator)
                print(f"    FAILED: {type(e).__name__}: {str(e)[:150]}")
                results.append({"c_delta": c_delta, "c_sigma": c_sigma,
                                "error": type(e).__name__})
                continue
            rec = {"c_delta": c_delta, "c_sigma": c_sigma,
                  "val_acc": val_m["acc"], "val_f1": val_m["macro_f1"],
                  "test_acc": test_m["acc"], "test_f1": test_m["macro_f1"],
                  "per_class_acc_test": test_m["per_class_acc"],
                  "biases": biases.tolist(), "convergence": cell_diag,
                  "seconds": time.perf_counter() - t}
            results.append(rec)
            print(f"    val acc={val_m['acc']:.4f} f1={val_m['macro_f1']:.4f}   "
                  f"test acc={test_m['acc']:.4f}   ({rec['seconds']:.1f}s)", flush=True)
            _print_convergence(cell_diag)
            with open(os.path.join(args.out, "grid_results.json"), "w") as f:
                json.dump(results, f, indent=2)
            if best is None or rec["val_f1"] > best["val_f1"]:
                best = rec

    print(f"\n=== done in {time.perf_counter()-t0:.0f}s ===")
    print(f"  best cell: c_delta={best['c_delta']}  c_sigma={best['c_sigma']}")
    print(f"  val:  acc={best['val_acc']:.4f}  macro_f1={best['val_f1']:.4f}")
    print(f"  test: acc={best['test_acc']:.4f}  macro_f1={best['test_f1']:.4f}")
    print(f"  per-class test acc: {[round(100*a,1) for a in best['per_class_acc_test']]}")
    _print_convergence(best.get("convergence", {}), indent="  best-cell ")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"estimator": args.estimator, "classes": classes,
                  "reused_cell_from": reuse_path, "best": best,
                  "config": vars(args)}, f, indent=2, default=float)
    with open(os.path.join(args.out, "selected_cell.json"), "w") as f:
        json.dump({"c_delta": best["c_delta"], "c_sigma": best["c_sigma"]}, f, indent=2)

    print(f"\nsaved -> {args.out}/summary.json, selected_cell.json")


if __name__ == "__main__":
    main()
