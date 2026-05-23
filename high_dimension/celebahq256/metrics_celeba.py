"""
metrics_celeba.py
=================
Evaluation metrics for the high_dimension/celebahq256 experiment.
"""
import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


# ─── InceptionV3 feature extractor (clean-fid) ───────────────────────────────

def build_inception_extractor(device=None):
    """
    Build the clean-fid InceptionV3 feature extractor.

    Returns (model, device) where model : PyTorch nn.Module, eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from cleanfid.features import build_feature_extractor
    model = build_feature_extractor("clean", device=device)
    return model, device


def extract_inception_features(
    images_np: np.ndarray,
    model,
    device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract 2048-dim InceptionV3 pool3 features.

    images_np : (N, H, W, 3) float32 [0, 1]  RGB images
    Returns   : (N, 2048) float32
    """
    feats = []
    N = len(images_np)
    for i in range(0, N, batch_size):
        batch = images_np[i: i + batch_size]                 # (B, H, W, 3)
        # (B, 3, H, W) float32 [0, 1]
        t = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float().to(device)
        t = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            f = model(t)                                      # (B, 2048)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


# ─── FID ─────────────────────────────────────────────────────────────────────

def compute_fid_from_features(
    feats_real: np.ndarray,
    feats_gen:  np.ndarray,
) -> float:
    """
    Fréchet Inception Distance from pre-computed feature vectors.
    """
    mu_r = feats_real.mean(0).astype(np.float64)
    mu_g = feats_gen.mean(0).astype(np.float64)
    sigma_r = np.cov(feats_real.astype(np.float64), rowvar=False)
    sigma_g = np.cov(feats_gen.astype(np.float64),  rowvar=False)

    diff = mu_r - mu_g
    product = sigma_r @ sigma_g
    covmean, _ = scipy.linalg.sqrtm(product, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = (float(np.dot(diff, diff))
           + float(np.trace(sigma_r + sigma_g - 2.0 * covmean)))
    return float(fid)


# ─── KID: polynomial-kernel MMD ───────────────────────────────────────────────

def _poly_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> np.ndarray:
    """Polynomial kernel matrix k(x,y) = (x·y/d + 1)^degree."""
    d = X.shape[1]
    return (X @ Y.T / d + 1.0) ** degree


def _mmd_polynomial(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> float:
    """Unbiased polynomial-kernel MMD²  E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]."""
    n, m = X.shape[0], Y.shape[0]
    Kxx  = _poly_kernel(X, X, degree)
    Kyy  = _poly_kernel(Y, Y, degree)
    Kxy  = _poly_kernel(X, Y, degree)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd = (Kxx.sum() / (n * (n - 1))
           + Kyy.sum() / (m * (m - 1))
           - 2.0 * Kxy.mean())
    return float(mmd)


def compute_kid(
    feats_real: np.ndarray,
    feats_gen:  np.ndarray,
    *,
    subset_size: int = 100,
    n_subsets:   int = 50,
    degree:      int = 3,
    rng:         np.random.Generator = None,
) -> tuple:
    """
    KID = mean of unbiased MMD² over random subsets.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    mmd_values = []
    for _ in range(n_subsets):
        idx_r = rng.choice(len(feats_real), size=subset_size, replace=False)
        idx_g = rng.choice(len(feats_gen),  size=subset_size, replace=False)
        mmd_values.append(
            _mmd_polynomial(feats_real[idx_r], feats_gen[idx_g], degree))
    arr = np.array(mmd_values)
    return float(arr.mean()), float(arr.std())


# ─── Recall (PRDC-based) ──────────────────────────────────────────────────────

def compute_recall(
    feats_real: np.ndarray,
    feats_gen:  np.ndarray,
    *,
    k_neighbour: int = 3,
) -> float:
    """
    Recall = fraction of real samples whose k-NN ball contains ≥1 generated sample.
    """
    nn_real = NearestNeighbors(n_neighbors=k_neighbour + 1, algorithm="auto")
    nn_real.fit(feats_real)
    dists_rr, _ = nn_real.kneighbors(feats_real)   # (N_real, k+1)
    radii = dists_rr[:, -1]                         # k-th NN distance

    nn_gen = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_gen.fit(feats_gen)
    dists_rg, _ = nn_gen.kneighbors(feats_real)    # (N_real, 1)
    covered = dists_rg[:, 0] <= radii
    return float(covered.mean())


# ─── DupRate: memorisation / duplication rate ────────────────────────────────

def compute_tau(Z_train: np.ndarray, percentile: float = 5) -> float:
    """
    τ = p-th percentile of within-training-set nearest-neighbour distances.
    """
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(Z_train)
    dists, _ = nn.kneighbors(Z_train)   # (N_train, 2)
    nn_dists  = dists[:, 1]             # skip self (index 0)
    return float(np.percentile(nn_dists, percentile))


def compute_duprate(
    Z_gen:   np.ndarray,
    Z_train: np.ndarray,
    tau:     float,
) -> float:
    """
    DupRate = fraction of generated latents closer than τ to any training latent.
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(Z_train)
    dists, _ = nn.kneighbors(Z_gen)    # (N_gen, 1)
    return float((dists[:, 0] < tau).mean())


# ─── Bootstrap evaluation ────────────────────────────────────────────────────

def bootstrap_metrics(
    feats_real:  np.ndarray,
    feats_gen:   np.ndarray,
    Z_gen:       np.ndarray,
    Z_train:     np.ndarray,
    tau:         float,
    *,
    n_eval:      int = 500,
    n_bootstrap: int = 3,
    kid_subset:  int = 100,
    kid_n_subs:  int = 50,
    kid_degree:  int = 3,
    k_recall:    int = 3,
    seed:        int = 0,
) -> dict:
    """
    Bootstrap evaluation of FID, KID, Recall, DupRate.
    """
    rng_np = np.random.default_rng(seed)

    fid_vals, kid_vals, recall_vals, dup_vals = [], [], [], []

    for _ in range(n_bootstrap):
        idx_g = rng_np.choice(len(feats_gen), size=n_eval, replace=False)
        fr = feats_real                        # full test set, fixed across replicates
        fg = feats_gen[idx_g]
        zg = Z_gen[idx_g]

        fid_val = compute_fid_from_features(fr, fg)
        kid_m, _ = compute_kid(fr, fg,
                               subset_size=kid_subset, n_subsets=kid_n_subs,
                               degree=kid_degree, rng=rng_np)
        rec = compute_recall(fr, fg, k_neighbour=k_recall)
        dup = compute_duprate(zg, Z_train, tau)

        fid_vals.append(fid_val)
        kid_vals.append(kid_m)
        recall_vals.append(rec)
        dup_vals.append(dup)

    return {
        "fid_mean":     float(np.mean(fid_vals)),
        "fid_std":      float(np.std(fid_vals)),
        "kid_mean":     float(np.mean(kid_vals)),
        "kid_std":      float(np.std(kid_vals)),
        "recall_mean":  float(np.mean(recall_vals)),
        "recall_std":   float(np.std(recall_vals)),
        "duprate_mean": float(np.mean(dup_vals)),
        "duprate_std":  float(np.std(dup_vals)),
    }
