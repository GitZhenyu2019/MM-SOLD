"""
step1_prepare_subspace.py
=================================
Prepare the top-190 PCA subspace of the precomputed 512-dim DiffAE
latents.
"""
import argparse
import json
import os

import numpy as np

import config as cfg                                  # noqa: E402
from common import (                                   # noqa: E402
    mean_nn_distance_chunked, to_subspace, rbf_median_bandwidth,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(cfg.RESULTS_DIR_DIFFAE, "subspace"))
    ap.add_argument("--zsem_train", default=cfg.DIFFAE_ZSEM_TRAIN)
    ap.add_argument("--zsem_valid", default=cfg.DIFFAE_ZSEM_VALID)
    ap.add_argument("--proj_k", type=int, default=cfg.PROJ_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke_test", action="store_true",
                    help="use a small random subset of train/test latents "
                         "for a quick end-to-end sanity check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 60)
    print("Step 1 (diffae): top-190 DiffAE-latent subspace preparation")
    print("=" * 60)

    print(f"\nLoading train latents: {a.zsem_train}")
    Z_train = np.load(a.zsem_train).astype(np.float64)
    print(f"Loading test  latents: {a.zsem_valid}")
    Z_test = np.load(a.zsem_valid).astype(np.float64)

    if a.smoke_test:
        rng = np.random.default_rng(a.seed)
        idx_tr = rng.choice(len(Z_train), min(2000, len(Z_train)), replace=False)
        idx_te = rng.choice(len(Z_test), min(300, len(Z_test)), replace=False)
        Z_train = Z_train[idx_tr]
        Z_test = Z_test[idx_te]
        print(f"  [smoke_test] subsampled to train={Z_train.shape} test={Z_test.shape}")

    n, dfull = Z_train.shape
    assert dfull == cfg.DFULL, f"expected {cfg.DFULL}-dim DiffAE latents, got {dfull}"
    print(f"  Z_train {Z_train.shape}  Z_test {Z_test.shape}")

    K = a.proj_k
    mean = Z_train.mean(axis=0)
    Zc = Z_train - mean[None, :]
    print(f"\nEigendecomposing the {dfull}x{dfull} empirical covariance ...")
    w, V = np.linalg.eigh(Zc.T @ Zc / max(n - 1, 1))
    w, V = w[::-1], V[:, ::-1]                          # descending eigenvalues
    Vt, Vb = V[:, :K], V[:, K:]                          # (dfull,K), (dfull,dfull-K)
    lambda_top = w[:K]
    lambda_tail = np.maximum(w[K:], 0.0)

    var_total = float(w.sum())
    var_retained = float(lambda_top.sum() / var_total)
    print(f"  top-{K} variance retained: {100*var_retained:.2f}%  "
          f"(paper: ~90.1%; MIN_VARIANCE_RETAINED={100*cfg.MIN_VARIANCE_RETAINED:.1f}%)")
    if var_retained < cfg.MIN_VARIANCE_RETAINED:
        print(f"  [WARNING] retained variance {100*var_retained:.2f}% is below "
              f"MIN_VARIANCE_RETAINED={100*cfg.MIN_VARIANCE_RETAINED:.1f}% -- check "
              f"proj_k / input data before trusting downstream results.")

    U_train190 = (Zc @ Vt).astype(np.float32)
    U_test190 = to_subspace(Z_test, mean, Vt)

    print(f"\nComputing NN (mean nearest-neighbour distance, top-{K} subspace, "
          f"{cfg.NN_N_SAMPLE}-point subsample, chunk={cfg.NN_CHUNK}) ...")
    NN = mean_nn_distance_chunked(U_train190, seed=a.seed)
    print(f"  NN = {NN:.5f}")

    print(f"\nComputing RBF-mixture-kernel latent-KID bandwidth (median heuristic, "
          f"raw {dfull}-dim space, {cfg.RBF_BW_N_SAMPLE}x{cfg.RBF_BW_N_SAMPLE} random "
          f"pairs) -- computed ONCE here, reused by every later step so every "
          f"cell/method is scored with the exact same kernel ...")
    RBF_BW = rbf_median_bandwidth(Z_train)
    print(f"  RBF_BW = {RBF_BW:.5f}")

    print(f"\nComputing RBF-mixture-kernel IMAGE-KID bandwidth (median heuristic, "
          f"on the 3,000 held-out test images' precomputed 2048-dim Inception "
          f"features, f_ref) -- both are wanted poly3 and RBF-mixture reported "
          f"for image-space KID (undecided which to ultimately use), so this "
          f"bandwidth is computed once here too and reused by step4/step6 ...")
    f_ref = np.load(cfg.DIFFAE_FEATURES_NPZ)["f_ref"]
    RBF_BW_IMG = rbf_median_bandwidth(f_ref)
    print(f"  RBF_BW_IMG = {RBF_BW_IMG:.5f}  (f_ref {f_ref.shape})")

    # ── Save artifacts ──────────────────────────────────────────────────────
    np.save(os.path.join(a.out, "mean512.npy"), mean.astype(np.float32))
    np.save(os.path.join(a.out, "Vt.npy"), Vt.astype(np.float32))
    np.save(os.path.join(a.out, "Vb.npy"), Vb.astype(np.float32))
    np.save(os.path.join(a.out, "lambda_top.npy"), lambda_top.astype(np.float32))
    np.save(os.path.join(a.out, "lambda_tail.npy"), lambda_tail.astype(np.float32))
    np.save(os.path.join(a.out, "U_train190.npy"), U_train190)
    np.save(os.path.join(a.out, "U_test190.npy"), U_test190)
    np.save(os.path.join(a.out, "Z_train512.npy"), Z_train.astype(np.float32))
    np.save(os.path.join(a.out, "Z_test512.npy"), Z_test.astype(np.float32))

    meta = {
        "proj_k": K, "dfull": dfull,
        "n_train": int(n), "n_test": int(len(Z_test)),
        "variance_retained": var_retained,
        "min_variance_retained_threshold": cfg.MIN_VARIANCE_RETAINED,
        "NN": NN,
        "nn_n_sample": cfg.NN_N_SAMPLE, "nn_chunk": cfg.NN_CHUNK,
        "rbf_bw": RBF_BW, "rbf_bw_n_sample": cfg.RBF_BW_N_SAMPLE,
        "rbf_bw_seed": cfg.RBF_BW_SEED,
        "rbf_bw_img": RBF_BW_IMG,
        "smoke_test": a.smoke_test,
        "seed": a.seed,
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved subspace artifacts + meta.json to {a.out}/")
    print("\nStep 1 (diffae) complete!")


if __name__ == "__main__":
    main()
