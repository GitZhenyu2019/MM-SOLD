"""
step3_train_ddpm.py
============================
Latent DDPM baseline for the CelebA-HQ DiffAE-latent generation experiment, trained DIRECTLY in the FULL 512-dim
DiffAE latent space.
"""
import argparse
import json
import os
import time

import numpy as np

import config as cfg                                    # noqa: E402
from common import to_subspace, best_cell_filtered, compute_kid_rbfmix  # noqa: E402
from metrics_celeba import compute_recall, compute_duprate, compute_tau  # noqa: E402

import torch


def _load_diffae_net():
    """Import the pretrained-architecture MLPSkipNetConfig from a local
    `git clone https://github.com/phizaz/diffae` checkout (code only)."""
    import sys
    if cfg.DIFFAE_REPO_DIR not in sys.path:
        sys.path.insert(0, cfg.DIFFAE_REPO_DIR)
    try:
        from model.latentnet import MLPSkipNetConfig
    except ImportError as exc:
        raise ImportError(
            f"Could not import model.latentnet.MLPSkipNetConfig from "
            f"{cfg.DIFFAE_REPO_DIR!r}. Clone the official DiffAE repo there "
            f"first:  git clone https://github.com/phizaz/diffae "
            f"{cfg.DIFFAE_REPO_DIR!r}  (code only -- no pretrained weights "
            f"needed from that checkout, only the network class)."
        ) from exc
    return MLPSkipNetConfig


def _save_resume_ckpt(path, net, ema, opt, step, best_loss, best_ema, best_step,
                      candidates):
    tmp = path + ".tmp"
    torch.save({
        "net": net.state_dict(), "ema": ema, "opt": opt.state_dict(),
        "step": step, "best_loss": best_loss, "best_ema": best_ema,
        "best_step": best_step, "candidates": candidates,
    }, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def _ddim_sample(net, ema_state, alphas_cumprod, n_samples, d, T, ddim_steps, dev, seed):
    """20-step deterministic DDIM (eta=0, no clipping)"""
    orig_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(ema_state)
    net.eval()
    ts = np.arange(0, T, max(1, T // ddim_steps))[:ddim_steps]
    a_ev = alphas_cumprod[ts]
    ap_ev = torch.cat([torch.ones(1, device=dev), a_ev[:-1]])
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    out = []
    batch = min(500, n_samples)
    for i0 in range(0, n_samples, batch):
        b = min(batch, n_samples - i0)
        x = torch.randn(b, d, device=dev, generator=g)
        for i in range(len(ts) - 1, -1, -1):
            tt = torch.full((b,), float(ts[i]), device=dev)
            eps = net(x, tt).pred
            a, ap = a_ev[i], ap_ev[i]
            x0 = (x - torch.sqrt(1 - a) * eps) / torch.sqrt(a)
            x = torch.sqrt(ap) * x0 + torch.sqrt(1 - ap) * eps
        out.append(x.cpu().numpy())
    net.load_state_dict(orig_state)
    net.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subspace_dir", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--ckpt_dir", default=cfg.CKPT_DIR_DIFFAE)
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "ddpm"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true",
                    help="reuse an existing COMPLETE checkpoint under --ckpt_dir")
    ap.add_argument("--resume", action="store_true",
                    help="continue training from the last mid-training checkpoint")
    ap.add_argument("--checkpoint_every", type=int, default=cfg.DDPM_CKPT_EVERY_STEPS,
                    help="save a resumable checkpoint + run a quality check every N steps")
    ap.add_argument("--smoke_test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 3 (diffae): Latent DDPM (full 512-dim DiffAE space)")
    print("=" * 60)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device: {dev}")

    U_train190 = np.load(os.path.join(a.subspace_dir, "U_train190.npy"))
    U_test190  = np.load(os.path.join(a.subspace_dir, "U_test190.npy"))
    Z_test512  = np.load(os.path.join(a.subspace_dir, "Z_test512.npy"))
    mean512    = np.load(os.path.join(a.subspace_dir, "mean512.npy"))
    Vt         = np.load(os.path.join(a.subspace_dir, "Vt.npy"))
    tau = compute_tau(U_train190, percentile=cfg.DUP_PERCENTILE)
    RBF_BW = json.load(open(os.path.join(a.subspace_dir, "meta.json")))["rbf_bw"]

    Z = np.load(cfg.DIFFAE_ZSEM_TRAIN).astype(np.float32)
    if a.smoke_test:
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(len(Z), min(2000, len(Z)), replace=False)
        Z = Z[idx]
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
    Zn = torch.from_numpy((Z - mu) / sd).to(dev)
    n, d = Zn.shape
    assert d == cfg.DFULL
    print(f"  latents {Zn.shape}, znormalised")

    MLPSkipNetConfig = _load_diffae_net()
    conf = MLPSkipNetConfig(num_channels=d, skip_layers=cfg.DDPM_SKIP_LAYERS,
                            num_hid_channels=cfg.DDPM_NUM_HID_CHANNELS,
                            num_layers=cfg.DDPM_NUM_LAYERS)
    net = conf.make_model().to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  MLPSkipNet params: {n_params/1e6:.1f}M")

    T = cfg.DDPM_T_STEPS
    betas = torch.full((T,), cfg.DDPM_BETA_CONST, device=dev)
    abar = torch.cumprod(1 - betas, 0)

    B = 8 if a.smoke_test else cfg.DDPM_BATCH
    TOTAL = 200 if a.smoke_test else (cfg.DDPM_TOTAL_SAMPLES // B)
    checkpoint_every = 20 if a.smoke_test else a.checkpoint_every
    ddim_steps = 5 if a.smoke_test else cfg.DDIM_STEPS
    n_repeats = 2 if a.smoke_test else cfg.N_REPEATS
    n_generate = 100 if a.smoke_test else cfg.N_GENERATE

    ckpt_name = "ddpm_smoke.pt" if a.smoke_test else "ddpm.pt"
    ckpt_path = os.path.join(a.ckpt_dir, ckpt_name)
    resume_path = os.path.join(a.ckpt_dir, "ddpm_resume.pt")

    training_time = 0.0
    if a.skip_train and os.path.exists(ckpt_path):
        print(f"\n[skip_train] loading existing COMPLETE checkpoint: {ckpt_path}")
        info_ckpt = torch.load(ckpt_path, map_location=dev)
        ema = info_ckpt["ema"]
        mu, sd = info_ckpt["mu"], info_ckpt["sd"]
        training_time = info_ckpt.get("training_time_s", 0.0)
        info = info_ckpt.get("info", {})
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.DDPM_LR, weight_decay=cfg.DDPM_WD)
        ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
        start_step = 0
        best_loss, best_ema, best_step = float("inf"), None, 0
        candidates = []

        if a.resume and os.path.exists(resume_path):
            try:
                ck = torch.load(resume_path, map_location=dev)
                net.load_state_dict(ck["net"])
                ema = ck["ema"]
                opt.load_state_dict(ck["opt"])
                start_step = ck["step"]
                best_loss, best_ema, best_step = ck["best_loss"], ck["best_ema"], ck["best_step"]
                candidates = ck["candidates"]
                print(f"  --resume: continuing from step {start_step}/{TOTAL}")
            except Exception as exc:
                print(f"  [WARNING] --resume failed to load {resume_path} ({exc!r}) "
                      f"-- training from step 0")
                start_step = 0

        print(f"Training on {dev}  steps {start_step}->{TOTAL}  batch={B}  "
              f"T={T}  beta={cfg.DDPM_BETA_CONST}  lr={cfg.DDPM_LR}  wd={cfg.DDPM_WD}  "
              f"checkpoint_every={checkpoint_every}")

        t0 = time.perf_counter()
        for step in range(start_step, TOTAL):
            idx = torch.randint(0, n, (B,), device=dev)
            x0 = Zn[idx]
            t = torch.randint(0, T, (B,), device=dev)
            eps = torch.randn_like(x0)
            av = abar[t][:, None]
            xt = torch.sqrt(av) * x0 + torch.sqrt(1 - av) * eps
            loss = (net(xt, t.float()).pred - eps).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    ema[k].mul_(cfg.DDPM_EMA_DECAY).add_(v, alpha=1.0 - cfg.DDPM_EMA_DECAY)
            loss_val = float(loss.item())
            if loss_val < best_loss:
                best_loss = loss_val

            is_ckpt_step = (step + 1) % checkpoint_every == 0 or step + 1 == TOTAL
            log_every = max(1, TOTAL // 20)
            if (step + 1) % log_every == 0 or step == 0:
                el = time.perf_counter() - t0
                eta = el / max(step, 1) * (TOTAL - step) / 60
                print(f"  step {step+1}/{TOTAL}  loss={loss_val:.4f}  "
                      f"({el/60:.1f} min, ETA {eta:.0f} min)")
            if is_ckpt_step:
                q_n = 50 if a.smoke_test else 500
                Z_q = _ddim_sample(net, ema, abar, q_n, d, T, ddim_steps, dev,
                                   a.seed + 2 * 10 ** 6 + step)
                Z_q = Z_q.astype(np.float64) * sd + mu
                U_q = to_subspace(Z_q, mean512, Vt)
                rng = np.random.default_rng(a.seed + step)
                # RBF-mixture kernel, NOT poly3 -- same metric/bandwidth as
                # every grid-searched method's checkpoint/cell selection (see
                # common.repeat_and_evaluate_latent's docstring).
                kid_q, _ = compute_kid_rbfmix(Z_test512, Z_q.astype(np.float32), RBF_BW, rng=rng)
                rec_q = compute_recall(U_test190, U_q, k_neighbour=cfg.K_RECALL)
                dup_q = compute_duprate(U_q, U_train190, tau)
                candidates.append({"step": step + 1, "loss": loss_val,
                                   "kid": kid_q, "duprate": dup_q, "recall": rec_q})
                if kid_q <= min((c["kid"] for c in candidates), default=kid_q):
                    best_ema = {k: v.detach().clone() for k, v in ema.items()}
                    best_step = step + 1
                print(f"    [quality check @ step{step+1}] KID={kid_q:.4f} "
                      f"DupRate={dup_q:.4f} Recall={rec_q:.3f}")
                if not a.smoke_test:
                    _save_resume_ckpt(resume_path, net, ema, opt, step + 1,
                                      best_loss, best_ema, best_step, candidates)
        torch.cuda.synchronize() if dev == "cuda" else None
        training_time = time.perf_counter() - t0
        print(f"\nTraining complete in {training_time:.1f}s ({training_time/60:.1f} min)")

        if candidates:
            selected = best_cell_filtered(candidates, duprate_key="duprate",
                                          recall_key="recall", kid_key="kid")
            ema = best_ema if best_ema is not None else ema
        else:
            selected = {"step": best_step, "kid": None, "duprate": None, "recall": None}
            print("  [WARNING] no quality-check candidates gathered -- keeping "
                  "the raw EMA weights as-is")
        print(f"Selected checkpoint: step {selected['step']}/{TOTAL}  "
              f"KID={selected['kid']}  DupRate={selected['duprate']}  "
              f"Recall={selected['recall']}  "
              f"(duprate_cap={cfg.REP_CELL_DUPRATE_CAP}, "
              f"recall_floor={cfg.REP_CELL_RECALL_FLOOR})")

        info = {
            "dfull": d, "hidden": cfg.DDPM_NUM_HID_CHANNELS, "n_layers": cfg.DDPM_NUM_LAYERS,
            "T": T, "beta_const": cfg.DDPM_BETA_CONST,
            "selected_step": selected["step"], "selected_kid": selected["kid"],
            "selected_duprate": selected["duprate"], "selected_recall": selected["recall"],
            "training_time_s": training_time, "quality_curve": candidates,
            "n_params": n_params,
        }
        torch.save({"ema": ema, "mu": mu, "sd": sd, "step": TOTAL,
                   "training_time_s": training_time, "info": info}, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    # ── Final generation: 3 INDEPENDENT DDIM samplings of N_GENERATE (GPU),
    #    saved to disk for step4 to decode+score. ───────────────────────────
    print(f"\nGenerating {n_repeats} independent DDIM repeats of {n_generate} "
          f"512-dim latents (GPU) ...")
    gen_dir = os.path.join(a.out, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    for r in range(n_repeats):
        Z_g = _ddim_sample(net, ema, abar, n_generate, d, T, ddim_steps, dev,
                           a.seed + 10 ** 6 + r)
        Z_g = (Z_g.astype(np.float64) * sd + mu).astype(np.float32)
        suffix = "_smoke" if a.smoke_test else ""
        np.save(os.path.join(gen_dir, f"repeat{r}{suffix}.npy"), Z_g)
    print(f"  saved {n_repeats} repeat(s) to {gen_dir}/")

    # ── Timing (GPU, per instruction): DDIM sampling time at N_GENERATE ────
    sample_mean = sample_std = None
    if not a.smoke_test:
        print(f"\nTiming DDIM sampling on GPU ({cfg.DDIM_STEPS} steps, "
              f"{cfg.N_GENERATE} samples, 1 warm-up + {cfg.TIMING_N_TIMED} timed runs) ...")
        _ddim_sample(net, ema, abar, cfg.N_GENERATE, d, T, cfg.DDIM_STEPS, dev, a.seed)
        if dev == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(cfg.TIMING_N_TIMED):
            t0 = time.perf_counter()
            _ddim_sample(net, ema, abar, cfg.N_GENERATE, d, T, cfg.DDIM_STEPS, dev, a.seed)
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        sample_mean, sample_std = float(np.mean(times)), float(np.std(times))
        print(f"  DDPM GPU sampling time: {sample_mean:.3f}s +/- {sample_std:.3f}s "
              f"({sample_mean / cfg.N_GENERATE * 1000:.3f} ms/sample)")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "method": "latent_ddpm", "dfull": cfg.DFULL,
        "hidden": cfg.DDPM_NUM_HID_CHANNELS, "n_layers": cfg.DDPM_NUM_LAYERS,
        "T": T, "ddim_steps": ddim_steps, "total_steps": TOTAL,
        "training_time_s": training_time,
        "n_per_repeat": n_generate, "n_repeats": n_repeats,
        "smoke_test": a.smoke_test,
    }
    suffix = "_smoke" if a.smoke_test else ""
    with open(os.path.join(a.out, f"results_ddpm{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)

    if not a.smoke_test:
        timing_out = {
            "method": "latent_ddpm", "device": "GPU",
            "training_time_s": training_time, "training_time_h": training_time / 3600.0,
            "sample_time_s": sample_mean, "sample_time_std_s": sample_std,
            "sample_time_per_sample_ms": (sample_mean / cfg.N_GENERATE * 1000
                                          if sample_mean is not None else None),
            "n_warmup": 1, "n_timed": cfg.TIMING_N_TIMED,
        }
        with open(os.path.join(a.out, "timing_ddpm.json"), "w") as f:
            json.dump(timing_out, f, indent=2)

    print(f"\nSaved results to {a.out}/")
    print("\nStep 3 (diffae) complete!")


if __name__ == "__main__":
    main()
