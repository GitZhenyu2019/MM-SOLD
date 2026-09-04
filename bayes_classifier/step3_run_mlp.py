"""
step3_run_mlp.py
=====================
The two MLP rows of the revised digit classification results table
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import config as cfg  # noqa: E402
import tilting_lion as TL  # noqa: E402
from mlp_model import run_grid_search, LatentMLP  # noqa: E402  (augmentation_handwritten)
from classifier_model import save_params  # noqa: E402  (generic pickle I/O, reused as-is)
from common import (  # noqa: E402
    build_class_data, delta_sigma, h_sold, energy_distance, compute_metrics,
)

sampling_algo_release = cfg.load_release_sampling_algo()


def mlp_predict(params, U, hidden_dim, num_classes=10, batch_size=512):
    """Raw predictions, dimension-agnostic (works for 40-d input)."""
    preds = []
    for i in range(0, len(U), batch_size):
        logits = LatentMLP(num_classes=num_classes, hidden_dim=hidden_dim).apply(
            {"params": params}, jnp.asarray(U[i:i + batch_size]))
        preds.append(np.array(jnp.argmax(logits, axis=-1)))
    return np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int32)


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


def train_and_report(U_train, y_train, U_val, y_val, U_test, y_test, args, tag):
    best_val, best_test, best_wd, best_epoch, best_params = run_grid_search(
        U_train, y_train, U_val, y_val, U_test, y_test,
        latent_dim=U_train.shape[1], wd_grid=args.wd_grid, max_epochs=args.max_epochs,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim, lr=args.lr, seed=args.seed,
    )
    y_pred_test = mlp_predict(best_params, U_test, args.hidden_dim, cfg.N_CLASSES)
    test_metrics = compute_metrics(y_test, y_pred_test, cfg.N_CLASSES)
    print(f"\n[{tag}] best wd={best_wd:.0e} epoch={best_epoch} "
          f"val_acc={best_val:.2f}%  test_acc={test_metrics['acc']*100:.2f}%")
    print(f"  per-class test acc: "
          f"{[round(100*a, 1) for a in test_metrics['per_class_acc']]}")
    return best_params, best_wd, best_epoch, best_val, test_metrics


def sample_class_aug(sampling_algo, U_c, n_particles, delta_c, sigma_c, nsteps, M,
                     discretization, seed):
    x_sampled, _X = sampling_algo.sample_class_overdamped_manifold(
        X_class=jnp.asarray(U_c, jnp.float32),
        n_particles=n_particles, nsteps=nsteps, h=h_sold(delta_c),
        sigma_gmm=delta_c, sigma_smoothing=sigma_c, M=M,
        shared_noise=False, discretization=discretization, seed=seed,
    )
    return np.asarray(x_sampled, dtype=np.float32)


def select_aug_cell(class_data, U_val, y_val, args):
    """Energy-distance (c_delta, c_sigma) search for MM-SOLD augmentation,
    identical statistic to release/digits/scripts/digits_cv_grid.py."""
    c_grid = [float(x) for x in args.c_grid.split(",") if x]
    cells = list(itertools.product(c_grid, c_grid))
    rng = np.random.default_rng(args.seed)
    results = []
    print(f"\n[aug cell search] {len(cells)} cells, n_probe={args.n_probe} "
          f"particles/class, nsteps={args.aug_nsteps}")
    best = None
    for i, (c_delta, c_sigma) in enumerate(cells, 1):
        t0 = time.perf_counter()
        scores = []
        for c in range(cfg.N_CLASSES):
            cd = class_data[c]
            delta_c, sigma_c = delta_sigma(c_delta, c_sigma, cd["NN"], dsub=args.proj_k)
            q = sample_class_aug(sampling_algo_release, cd["U"], args.n_probe,
                                 delta_c, sigma_c, args.aug_nsteps, args.M,
                                 args.aug_discretization, args.seed + c)
            q = q[np.isfinite(q).all(axis=1)]
            if len(q) < 20:
                scores.append(np.inf)
                continue
            scores.append(energy_distance(q, U_val[y_val == c], rng))
        mean_score = float(np.mean(scores))
        rec = {"c_delta": c_delta, "c_sigma": c_sigma, "energy": mean_score,
              "seconds": time.perf_counter() - t0}
        results.append(rec)
        print(f"  [{i}/{len(cells)}] c_delta={c_delta:.2f} c_sigma={c_sigma:.2f} "
              f"energy={mean_score:.5f}  ({rec['seconds']:.1f}s)", flush=True)
        if best is None or mean_score < best["energy"]:
            best = rec
    return best, results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["baseline", "augmented"], required=True)
    ap.add_argument("--latents_dir", default=cfg.LATENT_DIR)
    ap.add_argument("--out", default=None, help="default: results/mlp_<mode>")
    ap.add_argument("--proj_k", type=int, default=cfg.PROJ_K)
    ap.add_argument("--M", type=int, default=cfg.M_MC)
    # MLP hyperparameters
    ap.add_argument("--hidden_dim", type=int, default=cfg.MLP_HIDDEN)
    ap.add_argument("--lr", type=float, default=cfg.MLP_LR)
    ap.add_argument("--max_epochs", type=int, default=cfg.MLP_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=cfg.MLP_BATCH)
    ap.add_argument("--wd_grid", type=float, nargs="+", default=cfg.MLP_WD_GRID)
    # augmentation-only
    ap.add_argument("--n_aug_per_class", type=int, default=cfg.N_AUG_PER_CLASS)
    ap.add_argument("--aug_nsteps", type=int, default=cfg.AUG_NSTEPS)
    ap.add_argument("--aug_discretization", default=cfg.AUG_DISCRETIZATION)
    ap.add_argument("--shared_cell", default=None,
                    help="selected_cell.json to reuse (default: "
                         "results/ecm_plugin/selected_cell.json)")
    ap.add_argument("--own_grid", action="store_true",
                    help="do an independent energy-distance (c_delta,c_sigma) "
                         "search instead of reusing the plug-in ECM's cell")
    ap.add_argument("--c_grid", default=",".join(str(c) for c in cfg.C_GRID))
    ap.add_argument("--n_probe", type=int, default=2000,
                    help="particles/class used only while searching for the "
                         "best (c_delta,c_sigma) cell (cheaper than the final "
                         "n_aug_per_class draw)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(cfg.RESULTS_DIR, f"mlp_{args.mode}")
    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print(f"Step 3 (v2): MLP -- mode={args.mode}")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    data = load_latents(args.latents_dir)
    U_train, y_train = data["U_train"], data["y_train"]
    U_val, y_val = data["U_val"], data["y_val"]
    U_test, y_test = data["U_test"], data["y_test"]
    print(f"  U_train {U_train.shape}  U_val {U_val.shape}  U_test {U_test.shape}")

    if args.mode == "baseline":
        params, best_wd, best_epoch, best_val, test_metrics = train_and_report(
            U_train, y_train, U_val, y_val, U_test, y_test, args, tag="MLP (baseline)")
        save_params(os.path.join(args.out, "mlp_baseline_best.pkl"), params,
                    info={"best_val_acc": best_val, "best_wd": best_wd,
                          "best_epoch": best_epoch})
        summary = {"mode": "baseline", "best_wd": best_wd, "best_epoch": best_epoch,
                  "val_acc": best_val, "test_acc": test_metrics["acc"],
                  "test_macro_f1": test_metrics["macro_f1"],
                  "per_class_acc_test": test_metrics["per_class_acc"],
                  "config": vars(args)}
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nsaved -> {args.out}/summary.json")
        return

    # ── augmented ────────────────────────────────────────────────────────
    class_data = build_class_data(U_train, y_train, cfg.N_CLASSES)

    if args.own_grid:
        best, grid_results = select_aug_cell(class_data, U_val, y_val, args)
        with open(os.path.join(args.out, "aug_cell_grid.json"), "w") as f:
            json.dump(grid_results, f, indent=2)
        c_delta, c_sigma = best["c_delta"], best["c_sigma"]
        print(f"\n[aug cell] selected by own energy-distance grid: "
              f"c_delta={c_delta} c_sigma={c_sigma} (energy={best['energy']:.5f})")
    else:
        shared_cell = args.shared_cell or os.path.join(
            cfg.RESULTS_DIR, "ecm_plugin", "selected_cell.json")
        if not os.path.exists(shared_cell):
            raise FileNotFoundError(
                f"{shared_cell} not found -- run step2_run_ecm.py "
                f"--estimator plugin first, or pass --own_grid / --shared_cell")
        with open(shared_cell) as f:
            sel = json.load(f)
        c_delta, c_sigma = sel["c_delta"], sel["c_sigma"]
        print(f"\n[aug cell] reusing {shared_cell}: c_delta={c_delta} c_sigma={c_sigma}")

    print(f"\n[generate] {args.n_aug_per_class} samples/class, "
          f"nsteps={args.aug_nsteps}, discretization={args.aug_discretization}")
    all_U, all_y = [], []
    for c in range(cfg.N_CLASSES):
        cd = class_data[c]
        delta_c, sigma_c = delta_sigma(c_delta, c_sigma, cd["NN"], dsub=args.proj_k)
        t0 = time.perf_counter()
        q = sample_class_aug(sampling_algo_release, cd["U"], args.n_aug_per_class,
                             delta_c, sigma_c, args.aug_nsteps, args.M,
                             args.aug_discretization, args.seed + c)
        keep = np.isfinite(q).all(axis=1)
        if not keep.all():
            print(f"  class {c}: dropped {(~keep).sum()} non-finite samples")
            q = q[keep]
        mu_star_c, Sigma_star_c = TL.moment_targets(cd["U"], delta_c)
        res_mean, res_cov = TL.moment_residuals(q, mu_star_c, Sigma_star_c)
        all_U.append(q)
        all_y.append(np.full(len(q), c, dtype=np.int32))
        print(f"  class {c}: delta_c={delta_c:.4f} sigma_c={sigma_c:.4f}  "
              f"generated {len(q)}  moment residuals mean={res_mean:.4f} cov={res_cov:.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

    U_gen = np.concatenate(all_U, axis=0).astype(np.float32)
    y_gen = np.concatenate(all_y, axis=0).astype(np.int32)
    U_aug_train = np.concatenate([U_train, U_gen], axis=0)
    y_aug_train = np.concatenate([y_train, y_gen], axis=0)
    print(f"\naugmented train set: {U_aug_train.shape}  "
          f"({len(U_train)} orig + {len(U_gen)} generated, "
          f"{U_aug_train.shape[0]/len(U_train):.1f}x)")

    params, best_wd, best_epoch, best_val, test_metrics = train_and_report(
        U_aug_train, y_aug_train, U_val, y_val, U_test, y_test, args,
        tag="MLP (MM-SOLD Aug.)")
    save_params(os.path.join(args.out, "mlp_augmented_best.pkl"), params,
                info={"best_val_acc": best_val, "best_wd": best_wd,
                      "best_epoch": best_epoch,
                      "c_delta": c_delta, "c_sigma": c_sigma})
    summary = {"mode": "augmented", "c_delta": c_delta, "c_sigma": c_sigma,
              "n_aug_per_class": args.n_aug_per_class,
              "best_wd": best_wd, "best_epoch": best_epoch, "val_acc": best_val,
              "test_acc": test_metrics["acc"], "test_macro_f1": test_metrics["macro_f1"],
              "per_class_acc_test": test_metrics["per_class_acc"],
              "config": vars(args)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsaved -> {args.out}/summary.json")


if __name__ == "__main__":
    main()
