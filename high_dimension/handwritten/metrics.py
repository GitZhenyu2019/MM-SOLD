"""
metrics.py
==========
Evaluation metrics for the high_dimension/handwritten experiment.
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from sklearn.neighbors import NearestNeighbors
from classifier_model import ClassResBlock, load_params as clf_load_params


# ─── CNN feature extractor ────────────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    num_classes: int = 10
    base_ch:     int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:   # (B,64,64,1)
        ch = self.base_ch

        # Stem
        x = nn.Conv(ch, (3, 3), padding="SAME")(x)
        x = ClassResBlock(ch, groups=8)(x)

        # Downsampling stages
        x = nn.Conv(ch * 2, (4, 4), strides=(2, 2), padding="SAME")(x)   # 32
        x = ClassResBlock(ch * 2, groups=8)(x)

        x = nn.Conv(ch * 4, (4, 4), strides=(2, 2), padding="SAME")(x)   # 16
        x = ClassResBlock(ch * 4, groups=8)(x)

        x = nn.Conv(ch * 8, (4, 4), strides=(2, 2), padding="SAME")(x)   # 8
        x = ClassResBlock(ch * 8, groups=8)(x)

        x = nn.Conv(ch * 8, (4, 4), strides=(2, 2), padding="SAME")(x)   # 4
        x = ClassResBlock(ch * 8, groups=8)(x)

        # Head
        x    = jnp.mean(x, axis=(1, 2))    # GlobalAvgPool -> (B, ch*8)
        x    = nn.LayerNorm()(x)
        feat = nn.swish(nn.Dense(256)(x))  # 256-dim features  ← we return this
        nn.Dense(self.num_classes)(feat)   # create final-layer params (discarded)
        return feat


_feat_extractor = CNNFeatureExtractor()   # stateless singleton


def extract_features(
    images_np: np.ndarray,
    clf_params,
    *,
    batch_size: int = 128,
    num_classes: int = 10,
    base_ch:     int = 32,
) -> np.ndarray:
    """
    Extract 256-dim features from a pre-trained CNNClassifier.
    """
    if images_np.ndim == 3:
        images_np = images_np[..., None]        # add channel dim

    @jax.jit
    def _batch_features(params, imgs):
        return CNNFeatureExtractor(
            num_classes=num_classes, base_ch=base_ch
        ).apply({"params": params}, imgs)

    feats = []
    for i in range(0, len(images_np), batch_size):
        batch = jnp.asarray(images_np[i: i + batch_size])
        f     = _batch_features(clf_params, batch)
        feats.append(np.array(f))
    return np.concatenate(feats, axis=0).astype(np.float32)


# ─── KID: polynomial-kernel MMD ───────────────────────────────────────────────

def _poly_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> np.ndarray:
    """Polynomial kernel matrix k(x,y) = (x·y/d + 1)^degree."""
    d = X.shape[1]
    return (X @ Y.T / d + 1.0) ** degree


def _mmd_polynomial(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> float:
    """Unbiased polynomial-kernel MMD²  E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]."""
    n, m   = X.shape[0], Y.shape[0]
    Kxx    = _poly_kernel(X, X, degree)
    Kyy    = _poly_kernel(Y, Y, degree)
    Kxy    = _poly_kernel(X, Y, degree)
    # unbiased: zero diagonal for within-set terms
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
) -> tuple[float, float]:
    """
    KID = mean of unbiased MMD² over random subsets.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    mmd_values = []
    for _ in range(n_subsets):
        idx_r = rng.choice(len(feats_real), size=subset_size, replace=False)
        idx_g = rng.choice(len(feats_gen),  size=subset_size, replace=False)
        mmd_values.append(_mmd_polynomial(feats_real[idx_r], feats_gen[idx_g], degree))
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
    # Radius of each real sample = distance to its k-th nearest real neighbour
    nn_real = NearestNeighbors(n_neighbors=k_neighbour + 1, algorithm="auto")
    nn_real.fit(feats_real)
    dists_rr, _ = nn_real.kneighbors(feats_real)    # (N_real, k+1)
    radii        = dists_rr[:, -1]                   # k-th neighbour dist

    # For each real sample, check if any generated sample falls within its radius
    nn_gen = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_gen.fit(feats_gen)
    dists_rg, _ = nn_gen.kneighbors(feats_real)     # (N_real, 1)
    covered      = (dists_rg[:, 0] <= radii)
    return float(covered.mean())


# ─── DupRate: memorisation / duplication rate ────────────────────────────────

def compute_tau(Z_train: np.ndarray) -> float:
    """
    τ = 5th percentile of within-training-set nearest-neighbour distances.
    """
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(Z_train)
    dists, _ = nn.kneighbors(Z_train)   # (N_train, 2)
    nn_dists  = dists[:, 1]             # skip self (index 0)
    return float(np.percentile(nn_dists, 5))


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
    n_eval:      int = 300,
    n_bootstrap: int = 5,
    kid_subset:  int = 100,
    kid_n_subs:  int = 50,
    kid_degree:  int = 3,
    k_recall:    int = 3,
    seed:        int = 0,
) -> dict:
    """
    Bootstrap evaluation of KID, Recall, DupRate.
    """
    rng_np = np.random.default_rng(seed)

    kid_vals, recall_vals, dup_vals = [], [], []

    for _ in range(n_bootstrap):
        idx_g = rng_np.choice(len(feats_gen), size=n_eval, replace=False)
        fg    = feats_gen[idx_g]
        zg    = Z_gen[idx_g]

        kid_m, _ = compute_kid(feats_real, fg,
                                subset_size=kid_subset, n_subsets=kid_n_subs,
                                degree=kid_degree, rng=rng_np)
        rec   = compute_recall(feats_real, fg, k_neighbour=k_recall)
        dup   = compute_duprate(zg, Z_train, tau)

        kid_vals.append(kid_m)
        recall_vals.append(rec)
        dup_vals.append(dup)

    return {
        "kid_mean":     float(np.mean(kid_vals)),
        "kid_std":      float(np.std(kid_vals)),
        "recall_mean":  float(np.mean(recall_vals)),
        "recall_std":   float(np.std(recall_vals)),
        "duprate_mean": float(np.mean(dup_vals)),
        "duprate_std":  float(np.std(dup_vals)),
    }
