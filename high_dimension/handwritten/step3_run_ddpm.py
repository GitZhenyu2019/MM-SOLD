"""
step3_run_ddpm.py
======================
Latent DDPM baseline for the digit-8 generation experiment
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp

import config as cfg                                    # noqa: E402
from common import (                                     # noqa: E402
    load_nrae, load_clf, decode_and_extract, repeat_and_evaluate, best_cell_filtered,
)
from metrics import extract_features, compute_tau           # noqa: E402
from ddpm_model import (                                    # noqa: E402
    make_ddpm_state, make_cosine_schedule, ddpm_train_step,
    save_ddpm, load_ddpm, count_ddpm_params, ddim_sample,
)


def _save_resume_ckpt(path, state, epoch, best_loss, best_params, best_epoch, k_train):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "params": jax.device_get(state.params),
            "opt_state": jax.device_get(state.opt_state),
            "step": int(state.step),
            "epoch": epoch,
            "best_loss": best_loss,
            "best_params": jax.device_get(best_params) if best_params is not None else None,
            "best_epoch": best_epoch,
            "k_train": np.asarray(k_train),
        }, f, protocol=4)
    os.replace(tmp, path)


def _load_resume_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR, "subspace"))
    ap.add_argument("--ckpt_dir", default=cfg.CKPT_DIR)
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR, "ddpm"))
    ap.add_argument("--nrae_ckpt", default=cfg.NRAE_CKPT)
    ap.add_argument("--clf_ckpt", default=cfg.CLF_CKPT)
    ap.add_argument("--epochs", type=int, default=cfg.DDPM_EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true",
                    help="reuse an existing COMPLETE checkpoint under --ckpt_dir")
    ap.add_argument("--resume", action="store_true",
                    help="continue training from the last mid-training checkpoint "
                         "(see --checkpoint_every) instead of restarting at epoch 0")
    ap.add_argument("--checkpoint_every", type=int, default=1000,
                    help="save a resumable mid-training checkpoint every N epochs")
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 3: Latent DDPM (top-40 subspace)")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    U_train8 = np.load(os.path.join(a.subspace_dir, "U_train8.npy")).astype(np.float32)
    mean_8   = np.load(os.path.join(a.subspace_dir, "mean_8.npy"))
    V        = np.load(os.path.join(a.subspace_dir, "proj_V.npy"))
    imgs_test8 = np.load(os.path.join(a.subspace_dir, "imgs_test8.npy"))
    d = U_train8.shape[1]
    print(f"  U_train8 {U_train8.shape}  imgs_test8 {imgs_test8.shape}")

    nrae_model, nrae_params, n_dct = load_nrae(a.nrae_ckpt, seed=0)
    clf_params = load_clf(a.clf_ckpt)

    print("\nExtracting test CNN features ...")
    feats_test = extract_features(imgs_test8, clf_params, batch_size=64)
    tau = compute_tau(U_train8, percentile=1)
    print(f"  feats_test {feats_test.shape}  DupRate tau (top-40) = {tau:.5f}")

    # ── Standardise (dead dims with std<1e-3 left unscaled) ────────────────
    mu_w  = U_train8.mean(axis=0).astype(np.float32)
    std_w = U_train8.std(axis=0).astype(np.float32)
    std_safe = np.where(std_w > 1e-3, std_w, 1.0).astype(np.float32)
    U_train8_w = ((U_train8 - mu_w) / std_safe).astype(np.float32)
    print(f"  standardised: mean={U_train8_w.mean():.4f} std={U_train8_w.std():.4f}  "
          f"active_dims={int((std_w > 1e-3).sum())}/{d}")

    T = cfg.DDPM_T_STEPS
    ac_np, betas_np = make_cosine_schedule(T)
    ac = jnp.array(ac_np)

    epochs = 20 if a.smoke_test else a.epochs
    ckpt_name = "ddpm_digit8_top40_smoke.pkl" if a.smoke_test else "ddpm_digit8_top40.pkl"
    ckpt_path = os.path.join(a.ckpt_dir, ckpt_name)
    resume_path = os.path.join(a.ckpt_dir, "ddpm_digit8_top40_resume.pkl")
    checkpoint_every = 5 if a.smoke_test else a.checkpoint_every

    # Sample counts shared by BOTH the periodic quality checks during
    # training (below) and the final 5-repeat evaluation (after training).
    n_particles = 8 if a.smoke_test else cfg.N_PER_REPEAT
    ddim_steps  = 5 if a.smoke_test else cfg.DDIM_STEPS
    n_repeats   = 2 if a.smoke_test else cfg.N_REPEATS

    # ── Train (GPU) ──────────────────────────────────────────────────────
    training_time = 0.0
    if a.skip_train and os.path.exists(ckpt_path):
        print(f"\n[skip_train] loading existing COMPLETE checkpoint: {ckpt_path}")
        best_params, info = load_ddpm(ckpt_path)
        training_time = info.get("training_time_s", 0.0)
    else:
        N = len(U_train8_w)
        steps_ep = max(1, N // cfg.DDPM_BATCH)
        total_steps = epochs * steps_ep

        rng = jax.random.PRNGKey(a.seed)
        rng, k_init = jax.random.split(rng)
        model, state = make_ddpm_state(
            k_init, latent_dim=d, hidden=cfg.DDPM_HIDDEN, n_layers=cfg.DDPM_N_LAYERS,
            T=T, lr=cfg.DDPM_LR, wd=cfg.DDPM_WD, total_steps=total_steps)
        n_params = count_ddpm_params(state.params)
        print(f"\nMLPDenoiser  latent_dim={d}  hidden={cfg.DDPM_HIDDEN}  "
              f"n_layers={cfg.DDPM_N_LAYERS}  params={n_params:,}")

        rng, k_train = jax.random.split(rng)
        start_epoch = 0
        best_loss, best_params, best_epoch = float("inf"), None, 0

        if a.resume and os.path.exists(resume_path):
            try:
                ckpt = _load_resume_ckpt(resume_path)
                state = state.replace(
                    params=jax.tree_util.tree_map(jnp.asarray, ckpt["params"]),
                    opt_state=jax.tree_util.tree_map(jnp.asarray, ckpt["opt_state"]),
                    step=ckpt["step"])
                start_epoch = ckpt["epoch"]
                best_loss, best_epoch = ckpt["best_loss"], ckpt["best_epoch"]
                best_params = ckpt["best_params"]
                k_train = jnp.asarray(ckpt["k_train"])
                print(f"  --resume: continuing from epoch {start_epoch}/{epochs} "
                      f"(best_loss={best_loss:.5f} @ ep{best_epoch}), "
                      f"loaded from {resume_path}")
            except Exception as exc:
                print(f"  [WARNING] --resume failed to load/apply {resume_path} "
                      f"({exc!r}) -- falling back to training from epoch 0")
                start_epoch, best_loss, best_params, best_epoch = 0, float("inf"), None, 0

        print(f"Training on GPU  epochs {start_epoch}->{epochs}  batch={cfg.DDPM_BATCH}  "
              f"steps/epoch={steps_ep}  total_steps={total_steps}  "
              f"checkpoint_every={checkpoint_every} (also the cadence for the "
              f"periodic quality checks used to pick the final checkpoint)")

        candidates, candidate_params = [], []

        def _quick_quality(params_snapshot, seed):
            def gen_fn_q(s):
                rng_q = jax.random.PRNGKey(s)
                Z_w = ddim_sample(params_snapshot, rng_q, alphas_cumprod=ac_np,
                                  n_samples=n_particles, latent_dim=d,
                                  hidden=cfg.DDPM_HIDDEN, n_layers=cfg.DDPM_N_LAYERS,
                                  T=T, ddim_steps=ddim_steps)
                U_g = (np.array(Z_w) * std_safe + mu_w).astype(np.float32)
                feats_g = decode_and_extract(U_g, mean_8, V, nrae_model, nrae_params,
                                             n_dct, clf_params)
                return U_g, feats_g
            r = repeat_and_evaluate(gen_fn_q, feats_test, U_train8, tau,
                                    n_repeats=1, seed=seed)
            return r["kid_mean"], r["duprate_mean"], r["recall_mean"]

        log_interval = max(1, epochs // 20)
        t_start = time.perf_counter()
        for ep in range(start_epoch, epochs):
            ep_loss, n_steps = 0.0, 0
            perm = np.random.default_rng(a.seed + ep).permutation(N)
            Z_shuf = U_train8_w[perm]
            for i in range(0, N - cfg.DDPM_BATCH + 1, cfg.DDPM_BATCH):
                z_batch = jnp.asarray(Z_shuf[i:i + cfg.DDPM_BATCH])
                state, loss_val, k_train = ddpm_train_step(
                    state, z_batch, k_train, alphas_cumprod=ac, T=T,
                    latent_dim=d, hidden=cfg.DDPM_HIDDEN, n_layers=cfg.DDPM_N_LAYERS)
                ep_loss += float(loss_val)
                n_steps += 1
            ep_loss /= max(n_steps, 1)
            if ep_loss < best_loss:
                best_loss, best_params, best_epoch = ep_loss, jax.device_get(state.params), ep + 1
            if (ep + 1) % log_interval == 0 or ep == 0:
                print(f"  epoch {ep+1:5d}/{epochs}  loss={ep_loss:.5f}  "
                      f"best={best_loss:.5f} @ ep{best_epoch}")
            is_checkpoint_epoch = (ep + 1) % checkpoint_every == 0 or ep + 1 == epochs
            if not a.smoke_test and is_checkpoint_epoch:
                _save_resume_ckpt(resume_path, state, ep + 1, best_loss, best_params,
                                  best_epoch, k_train)
            if is_checkpoint_epoch:
                params_snap = jax.device_get(state.params)
                kid_q, dup_q, rec_q = _quick_quality(params_snap, a.seed + 2 * 10 ** 6 + ep)
                candidates.append({"epoch": ep + 1, "loss": ep_loss,
                                   "kid": kid_q, "duprate": dup_q, "recall": rec_q})
                candidate_params.append(params_snap)
                print(f"    [quality check @ ep{ep+1}] KID={kid_q:.4f} "
                      f"DupRate={dup_q:.4f} Recall={rec_q:.3f}")
        jax.block_until_ready(best_params)
        training_time = time.perf_counter() - t_start
        print(f"\nTraining complete in {training_time:.1f}s ({training_time/60:.1f} min)  "
              f"lowest training loss {best_loss:.5f} @ epoch {best_epoch} "
              f"(diagnostic only -- NOT used for generation, see selection below)")

        if candidates:
            selected = best_cell_filtered(candidates, duprate_key="duprate",
                                          recall_key="recall", kid_key="kid")
            selected_idx = next(i for i, c in enumerate(candidates)
                                if c["epoch"] == selected["epoch"])
            best_params = candidate_params[selected_idx]
        else:
            selected = {"epoch": best_epoch, "kid": None, "duprate": None, "recall": None}
            print("  [WARNING] no quality-check candidates gathered this run "
                  "-- keeping the previously selected checkpoint as-is")
        print(f"Selected checkpoint: epoch {selected['epoch']}/{epochs}  "
              f"KID={selected['kid']}  DupRate={selected['duprate']}  "
              f"Recall={selected['recall']}  "
              f"(duprate_cap={cfg.REP_CELL_DUPRATE_CAP}, "
              f"recall_floor={cfg.REP_CELL_RECALL_FLOOR})")

        info = {
            "latent_dim": d, "hidden": cfg.DDPM_HIDDEN, "n_layers": cfg.DDPM_N_LAYERS,
            "T": T,
            "selected_epoch": selected["epoch"], "selected_kid": selected["kid"],
            "selected_duprate": selected["duprate"], "selected_recall": selected["recall"],
            "lowest_train_loss": best_loss, "lowest_train_loss_epoch": best_epoch,
            "training_time_s": training_time, "whiten_mu": mu_w, "whiten_std": std_safe,
            "quality_curve": candidates,
        }
        save_ddpm(ckpt_path, best_params, info=info)
        print(f"Checkpoint saved: {ckpt_path}")

    # ── Generate: 5 INDEPENDENT DDIM samplings of 300 (GPU) ────────────────
    def gen_fn(seed):
        rng = jax.random.PRNGKey(seed)
        Z_w = ddim_sample(
            best_params, rng, alphas_cumprod=ac_np, n_samples=n_particles,
            latent_dim=d, hidden=cfg.DDPM_HIDDEN, n_layers=cfg.DDPM_N_LAYERS,
            T=T, ddim_steps=ddim_steps)
        U_gen = (np.array(Z_w) * std_safe + mu_w).astype(np.float32)
        feats_gen = decode_and_extract(U_gen, mean_8, V, nrae_model, nrae_params,
                                       n_dct, clf_params)
        return U_gen, feats_gen

    print(f"\nGenerating {n_repeats} independent DDIM repeats of {n_particles} samples ...")
    res = repeat_and_evaluate(gen_fn, feats_test, U_train8, tau,
                              n_repeats=n_repeats, seed=a.seed + 10 ** 6)
    print(f"  KID={res['kid_mean']:.4f}+/-{res['kid_std']:.4f}  "
          f"Recall={res['recall_mean']:.3f}+/-{res['recall_std']:.3f}  "
          f"DupRate={res['duprate_mean']:.3f}+/-{res['duprate_std']:.3f}")

    # ── Timing (GPU, per instruction): sampling time at N_PER_REPEAT ───────
    sample_mean = sample_std = None
    if not a.smoke_test:
        print(f"\nTiming DDIM sampling on GPU ({cfg.DDIM_STEPS} steps, "
              f"{cfg.N_PER_REPEAT} samples, 1 warm-up + 5 timed runs) ...")
        def _time_gen():
            rng = jax.random.PRNGKey(a.seed)
            z = ddim_sample(best_params, rng, alphas_cumprod=ac_np,
                            n_samples=cfg.N_PER_REPEAT, latent_dim=d,
                            hidden=cfg.DDPM_HIDDEN, n_layers=cfg.DDPM_N_LAYERS,
                            T=T, ddim_steps=cfg.DDIM_STEPS)
            jax.block_until_ready(z)
            return z
        _time_gen()  # warm-up
        times = []
        for _ in range(cfg.TIMING_N_TIMED):
            t0 = time.perf_counter()
            _time_gen()
            times.append(time.perf_counter() - t0)
        sample_mean, sample_std = float(np.mean(times)), float(np.std(times))
        print(f"  DDPM GPU sampling time: {sample_mean:.3f}s +/- {sample_std:.3f}s "
              f"({sample_mean / cfg.N_PER_REPEAT * 1000:.3f} ms/sample)")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "method": "latent_ddpm", "proj_k": cfg.PROJ_K,
        "ddpm_hidden": cfg.DDPM_HIDDEN, "ddpm_n_layers": cfg.DDPM_N_LAYERS,
        "T": T, "ddim_steps": cfg.DDIM_STEPS, "epochs": epochs,
        "training_time_s": training_time,
        "n_per_repeat": cfg.N_PER_REPEAT, "n_repeats": cfg.N_REPEATS,
        "smoke_test": a.smoke_test,
        **res,
    }
    suffix = "_smoke" if a.smoke_test else ""
    with open(os.path.join(a.out, f"results_ddpm{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)

    if not a.smoke_test:
        timing_out = {
            "method": "latent_ddpm", "device": "V100/GPU",
            "training_time_s": training_time, "training_time_h": training_time / 3600.0,
            "sample_time_s": sample_mean, "sample_time_std_s": sample_std,
            "sample_time_per_sample_ms": (sample_mean / cfg.N_PER_REPEAT * 1000
                                          if sample_mean is not None else None),
            "n_warmup": cfg.TIMING_N_WARMUP, "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_ddpm.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 3 complete!")


if __name__ == "__main__":
    main()
