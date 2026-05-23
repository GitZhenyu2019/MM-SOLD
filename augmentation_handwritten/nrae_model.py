"""
nrae_model.py
=============
Nuclear-Norm Regularised Autoencoder (NRAE) for handwritten digit data.
Follows the σ-CFDM supplementary (Scarvelis & Solomon 2024) in JAX/Flax.

Architecture
------------
  Encode :  image (64x64x1)
            → DCT-II  → keep top-left n_dctxn_dct  → flatten  (n_dct²,)
            → Enc1  [Dense(enc1_hidden) + ELU]      → (enc1_hidden,)
            → Enc2  [Dense(latent_dim)]              → latent   (latent_dim,)

  Decode :  latent (latent_dim,)
            → DecMLP [Dense(dec_hidden)+ELU → Dense(n_dct²)] → (n_dct²,)
            → iDCT (zero-pad to 64x64, inverse DCT-II)       → (64,64,1)
            → UNetRefinement                                  → (64,64,1)

Training loss  (same as σ-CFDM paper, η = σ² = 2.0² = 4):
  L = L_recon + L_reg
  L_recon = (1/α)·(1/B)·0.5·Σ log_cosh(α·(rec - x_img))   α=100
  L_reg   = (σ²/σ_fixed²)·(1/B)·0.5·(Σ||Enc1(x+ε)-Enc1(x)||²
                                      +Σ||Enc2(h+ε)-Enc2(h)||²)
            with σ_fixed=1e-3, σ=2.0
"""

import functools, math, os, pickle
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from vae_model import ResBlock, Downsample, Upsample  # reuse spatial primitives


# ============================================================
# 0.  DCT utilities  (computed once at import time)
# ============================================================

def _make_ortho_dct_matrix(N: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (NxN): D @ D.T = I."""
    k = np.arange(N)[:, None].astype(np.float64)
    n = np.arange(N)[None, :].astype(np.float64)
    D = np.cos(np.pi / N * (n + 0.5) * k)
    D[0] /= np.sqrt(N)
    D[1:] *= np.sqrt(2.0 / N)
    return D.astype(np.float32)


_DCT64: np.ndarray = _make_ortho_dct_matrix(64)   # (64,64) — module-level constant


def dct2d_crop_np(imgs: np.ndarray, n_dct: int = 20) -> np.ndarray:
    """
    2-D DCT-II and keep the top-left n_dctxn_dct low-frequency coefficients.
    """
    D = _DCT64[:n_dct, :]            # (n_dct, 64)
    X = D @ imgs @ D.T               # (N, n_dct, 64) then (N, n_dct, n_dct)
    return X.reshape(imgs.shape[0], -1).astype(np.float32)


def idct2d_from_crop_jax(x_flat: jnp.ndarray, n_dct: int = 20) -> jnp.ndarray:
    """
    Inverse 2-D DCT from truncated (zero-padded) coefficients → (B,64,64,1).
    """
    B = x_flat.shape[0]
    D_crop = jnp.asarray(_DCT64[:n_dct, :])     # (n_dct, 64), treated as constant
    X      = x_flat.reshape(B, n_dct, n_dct)    # (B, n_dct, n_dct)
    X_rec  = D_crop.T @ X @ D_crop              # (B, 64, 64)
    return X_rec[:, :, :, None]                  # (B, 64, 64, 1)


# ============================================================
# 1.  Model components
# ============================================================

class Enc1(nn.Module):
    """First encoder stage: single Dense + ELU."""
    hidden_dim: int = 2048

    @nn.compact
    def __call__(self, x):          # (B, dct_dim)
        return nn.elu(nn.Dense(self.hidden_dim)(x))


class Enc2(nn.Module):
    """Second encoder stage: linear projection to latent (no activation)."""
    latent_dim: int = 100

    @nn.compact
    def __call__(self, h):          # (B, enc1_hidden)
        return nn.Dense(self.latent_dim)(h)


class DecMLP(nn.Module):
    """2-layer MLP: latent → dec_hidden + ELU → dct_dim."""
    dec_hidden: int = 2048
    dct_dim:    int = 400           # default: 20² = 400

    @nn.compact
    def __call__(self, z):          # (B, latent_dim)
        h = nn.elu(nn.Dense(self.dec_hidden)(z))
        return nn.Dense(self.dct_dim)(h)


class UNetRefinement(nn.Module):
    """
    Small U-Net for image refinement: (B,64,64,1) → (B,64,64,1).

    Encoder : 64 → 32 → 16 → 8  (Downsample + ResBlock)
    Decoder : 8  → 16 → 32 → 64  (Upsample + concat skip + ResBlock)
    """
    base_ch: int = 32

    @nn.compact
    def __call__(self, x):                          # (B, 64, 64, 1)
        ch = self.base_ch

        # ── stem ────────────────────────────────────────────────────────
        x0 = nn.Conv(ch, (3, 3), padding="SAME")(x)  # (B, 64, 64, ch)

        # ── encoder ─────────────────────────────────────────────────────
        h1 = ResBlock(ch     )(x0)                    # (B, 64, 64, ch)
        h2 = ResBlock(ch * 2)(Downsample(ch * 2)(h1)) # (B, 32, 32, ch*2)
        h3 = ResBlock(ch * 4)(Downsample(ch * 4)(h2)) # (B, 16, 16, ch*4)
        b  = ResBlock(ch * 4)(Downsample(ch * 4)(h3)) # (B,  8,  8, ch*4)

        # ── decoder with skip connections ───────────────────────────────
        # After concat: input channels = 2 × ch*4 = ch*8; ResBlock outputs ch*4
        u3 = ResBlock(ch * 4)(
            jnp.concatenate([Upsample(ch * 4)(b),  h3], axis=-1))  # (B,16,16,ch*4)
        u2 = ResBlock(ch * 2)(
            jnp.concatenate([Upsample(ch * 2)(u3), h2], axis=-1))  # (B,32,32,ch*2)
        u1 = ResBlock(ch)(
            jnp.concatenate([Upsample(ch)     (u2), h1], axis=-1)) # (B,64,64,ch)

        return nn.Conv(1, (3, 3), padding="SAME")(u1)               # (B,64,64,1)


class NRAEModel(nn.Module):
    """
    Full Nuclear-Norm Regularised Autoencoder.
    Submodules: enc1, enc2, dec_mlp, unet.
    """
    latent_dim:   int = 100
    enc1_hidden:  int = 2048
    dec_hidden:   int = 2048
    n_dct:        int = 20
    unet_base_ch: int = 32

    def setup(self):
        dct_dim = self.n_dct ** 2
        self.enc1    = Enc1(hidden_dim=self.enc1_hidden)
        self.enc2    = Enc2(latent_dim=self.latent_dim)
        self.dec_mlp = DecMLP(dec_hidden=self.dec_hidden, dct_dim=dct_dim)
        self.unet    = UNetRefinement(base_ch=self.unet_base_ch)

    # ── public interface (called via model.apply(..., method=...)) ──────

    def encode(self, x_dct_flat):
        """(B, n_dct²) → (B, latent_dim)"""
        return self.enc2(self.enc1(x_dct_flat))

    def decode(self, z):
        """(B, latent_dim) → (B, 64, 64, 1)"""
        x_dct_rec = self.dec_mlp(z)
        x_rough   = idct2d_from_crop_jax(x_dct_rec, self.n_dct)
        return self.unet(x_rough)

    def __call__(self, x_dct_flat):
        return self.decode(self.encode(x_dct_flat))

    def forward_with_intermediates(self, x_dct_flat, rng,
                                   *, sigma, fixed_noise_sigma):
        """
        Full forward pass + noisy variants of enc1/enc2 for regularisation.
        """
        rng1, rng2 = jax.random.split(rng)
        eps_input  = jax.random.normal(rng1, x_dct_flat.shape) * fixed_noise_sigma
        h_clean    = self.enc1(x_dct_flat)
        h_noisy    = self.enc1(x_dct_flat + eps_input)
        z_clean    = self.enc2(h_clean)
        eps_hidden = jax.random.normal(rng2, h_clean.shape) * fixed_noise_sigma
        z_noisy    = self.enc2(h_clean + eps_hidden)
        rec_image  = self.decode(z_clean)
        return rec_image, h_clean, h_noisy, z_clean, z_noisy


# ============================================================
# 2.  Loss
# ============================================================

def _log_cosh(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable log-cosh: x + softplus(-2x) - log 2."""
    return x + jax.nn.softplus(-2.0 * x) - math.log(2.0)


def nrae_loss(params, model: NRAEModel,
              x_dct_flat: jnp.ndarray,
              x_img:      jnp.ndarray,
              rng:        jnp.ndarray,
              *,
              sigma:             float = 2.0,
              fixed_noise_sigma: float = 1e-3,
              alpha:             float = 100.0):
    """
    Compute NRAE training loss.
    """
    B = x_dct_flat.shape[0]

    rec_image, h_clean, h_noisy, z_clean, z_noisy = model.apply(
        {"params": params},
        x_dct_flat, rng,
        sigma=sigma, fixed_noise_sigma=fixed_noise_sigma,
        method=NRAEModel.forward_with_intermediates,
    )

    # ── reconstruction loss ──────────────────────────────────────────────
    recon = jnp.sum(_log_cosh(alpha * (rec_image - x_img)))
    recon = (1.0 / alpha) * (1.0 / B) * 0.5 * recon

    # ── noise-sensitivity regularisation ────────────────────────────────
    reg_enc1 = jnp.sum((h_noisy - h_clean) ** 2)
    reg_enc2 = jnp.sum((z_noisy - z_clean) ** 2)
    reg = ((sigma ** 2) / (fixed_noise_sigma ** 2)
           * (1.0 / B) * 0.5 * (reg_enc1 + reg_enc2))

    return recon + reg, (recon, reg)


# ============================================================
# 3.  Training state & JIT step
# ============================================================

class NRAETrainState(train_state.TrainState):
    pass


def make_nrae_state(rng, *,
                    latent_dim:   int   = 100,
                    enc1_hidden:  int   = 2048,
                    dec_hidden:   int   = 2048,
                    n_dct:        int   = 20,
                    unet_base_ch: int   = 32,
                    lr:           float = 1e-4):
    model     = NRAEModel(latent_dim=latent_dim, enc1_hidden=enc1_hidden,
                          dec_hidden=dec_hidden, n_dct=n_dct,
                          unet_base_ch=unet_base_ch)
    dummy_dct = jnp.zeros((1, n_dct ** 2), dtype=jnp.float32)
    params    = model.init(rng, dummy_dct)["params"]
    return model, NRAETrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(lr, weight_decay=0.0),
    )


@functools.partial(
    jax.jit,
    static_argnames=("latent_dim", "enc1_hidden", "dec_hidden", "n_dct",
                     "unet_base_ch", "sigma", "fixed_noise_sigma", "alpha"),
)
def nrae_train_step(state, x_dct_flat, x_img, rng, *,
                    latent_dim, enc1_hidden, dec_hidden, n_dct, unet_base_ch,
                    sigma, fixed_noise_sigma, alpha):
    model = NRAEModel(latent_dim=latent_dim, enc1_hidden=enc1_hidden,
                      dec_hidden=dec_hidden, n_dct=n_dct,
                      unet_base_ch=unet_base_ch)

    def _loss(params):
        return nrae_loss(params, model, x_dct_flat, x_img, rng,
                         sigma=sigma, fixed_noise_sigma=fixed_noise_sigma,
                         alpha=alpha)

    (loss, (recon, reg)), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, recon, reg


# ============================================================
# 4.  Encode / Decode helpers  (same interface as vae_model.py)
# ============================================================

def encode_dataset_nrae(model, params, images_np: np.ndarray,
                        *, n_dct: int = 20, batch_size: int = 64) -> np.ndarray:
    """
    (N, 64, 64) ink=1 float32  →  latent vectors (N, latent_dim) numpy.

    Applies DCT compression on the CPU (numpy) then runs Enc1+Enc2 via JAX.
    """
    Z = []
    for i in range(0, images_np.shape[0], batch_size):
        batch   = images_np[i:i + batch_size]             # (b, 64, 64)
        x_dct   = dct2d_crop_np(batch, n_dct=n_dct)      # (b, n_dct²) numpy
        z       = model.apply({"params": params},
                              jnp.asarray(x_dct),
                              method=NRAEModel.encode)
        Z.append(np.array(z))
    return np.concatenate(Z, axis=0)


def decode_latents_nrae(model, params, z,
                        *, batch_size: int = 64) -> np.ndarray:
    """
    (N, latent_dim) jnp/numpy  →  (N, 64, 64) ink=1 float32 numpy.

    Output is clipped to [0, 1] (UNet is unconstrained).
    """
    imgs = []
    for i in range(0, z.shape[0], batch_size):
        rec = model.apply({"params": params},
                          jnp.asarray(z[i:i + batch_size]),
                          method=NRAEModel.decode)
        imgs.append(np.clip(np.array(rec[..., 0]), 0.0, 1.0))
    return np.concatenate(imgs, axis=0)


# ============================================================
# 5.  Checkpoint utilities  (same interface as vae_model.py)
# ============================================================

def save_params(path, params, info=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {"params": jax.device_get(params)}
    if info:
        data["info"] = info
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_params(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["params"], data.get("info", {})


def count_params(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def print_nrae_summary(params, *, latent_dim, enc1_hidden, dec_hidden,
                       n_dct, unet_base_ch):
    dct_dim = n_dct ** 2
    print("\n--- NRAE Architecture ---")
    print(f"  Image        : (64, 64, 1)")
    print(f"  DCT crop     : {n_dct}x{n_dct} = {dct_dim}-dim vector")
    print(f"  Enc1         : {dct_dim} → {enc1_hidden}  [Dense + ELU]")
    print(f"  Enc2         : {enc1_hidden} → {latent_dim}  [Dense, linear]")
    print(f"  Latent dim   : {latent_dim}")
    print(f"  DecMLP       : {latent_dim} → {dec_hidden} → {dct_dim}  [Dense+ELU → Dense]")
    print(f"  iDCT         : {dct_dim} → {n_dct}x{n_dct} → pad → 64x64 → iDCT → (64,64,1)")
    print(f"  UNet         : (64,64,1) → (64,64,1)  [base_ch={unet_base_ch}]")
    print(f"  Parameters   : {count_params(params):,}")
    print()
