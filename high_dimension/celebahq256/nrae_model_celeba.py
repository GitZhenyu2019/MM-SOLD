"""
nrae_model_celeba.py
====================
Nuclear-Norm Regularised Autoencoder (NRAE) for CelebA-HQ 256x256 RGB images.
Follows σ-CFDM paper (Scarvelis & Solomon 2024) CelebA supplementary, in JAX/Flax.
"""
import functools
import math
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

# Reuse spatial ResBlock / Downsample / Upsample from augmentation_handwritten
from vae_model import ResBlock, Downsample, Upsample


# ============================================================
# 0.  DCT utilities for 256×256 RGB
# ============================================================

def _make_ortho_dct_matrix(N: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (NxN): D @ D.T = I."""
    k = np.arange(N)[:, None].astype(np.float64)
    n = np.arange(N)[None, :].astype(np.float64)
    D = np.cos(np.pi / N * (n + 0.5) * k)
    D[0] /= np.sqrt(N)
    D[1:] *= np.sqrt(2.0 / N)
    return D.astype(np.float32)


_DCT256: np.ndarray = _make_ortho_dct_matrix(256)   # (256,256) module-level constant


def dct2d_crop_rgb_np(imgs: np.ndarray, n_dct: int = 80) -> np.ndarray:
    """
    Per-channel 2-D DCT-II, keep top-left n_dctxn_dct low-frequency coefficients.
    """
    D = _DCT256[:n_dct, :]                          # (n_dct, 256)
    x = imgs.transpose(0, 3, 1, 2)                  # (N, 3, 256, 256)
    # Apply DCT along rows: sum_h D[i,h]*x[n,c,h,w] → (N, 3, n_dct, 256)
    tmp = np.einsum('ih,nchw->nciw', D, x)
    # Apply DCT along cols: sum_w tmp[n,c,i,w]*D[j,w] → (N, 3, n_dct, n_dct)
    dct_coeff = np.einsum('nciw,jw->ncij', tmp, D)
    return dct_coeff.reshape(imgs.shape[0], -1).astype(np.float32)


def idct2d_from_crop_rgb_jax(x_flat: jnp.ndarray,
                              n_dct: int = 80,
                              img_size: int = 256) -> jnp.ndarray:
    """
    Inverse 2-D DCT from truncated coefficients → (B, 256, 256, 3).
    """
    B = x_flat.shape[0]
    C = 3
    D_crop = jnp.asarray(_DCT256[:n_dct, :])        # (n_dct, img_size)

    # Reshape: (B, C, n_dct, n_dct)
    Y = x_flat.reshape(B, C, n_dct, n_dct)

    # Step 1: D_crop.T @ Y  →  einsum('ih,bcij->bchj', D_crop, Y)
    #   sum_i: D_crop[i,h] * Y[b,c,i,j] → tmp[b,c,h,j]   (h indexes img_size)
    tmp = jnp.einsum('ih,bcij->bchj', D_crop, Y)    # (B, C, img_size, n_dct)

    # Step 2: tmp @ D_crop  →  einsum('bchj,jw->bchw', tmp, D_crop)
    #   sum_j: tmp[b,c,h,j] * D_crop[j,w] → result[b,c,h,w]
    result = jnp.einsum('bchj,jw->bchw', tmp, D_crop)  # (B, C, img_size, img_size)

    return result.transpose(0, 2, 3, 1)              # (B, img_size, img_size, C)


# ============================================================
# 1.  Model components
# ============================================================

class Enc1(nn.Module):
    """First encoder stage: Dense(10000) + ELU."""
    hidden_dim: int = 10000

    @nn.compact
    def __call__(self, x):              # (B, 19200)
        return nn.elu(nn.Dense(self.hidden_dim)(x))


class Enc2(nn.Module):
    """Second encoder stage: linear projection to latent."""
    latent_dim: int = 700

    @nn.compact
    def __call__(self, h):              # (B, 10000)
        return nn.Dense(self.latent_dim)(h)


class DecMLP(nn.Module):
    """2-layer MLP: latent → dec_hidden + ELU → dct_dim."""
    dec_hidden: int = 10000
    dct_dim:    int = 19200             # 3 * 80² = 19200

    @nn.compact
    def __call__(self, z):              # (B, 700)
        h = nn.elu(nn.Dense(self.dec_hidden)(z))
        return nn.Dense(self.dct_dim)(h)


class UNetRefinement256RGB(nn.Module):
    """
    6-level U-Net for RGB image refinement: (B,256,256,3) → (B,256,256,3).

    Encoder : 256 → 128 → 64 → 32 → 16 → 8  (Downsample + ResBlock)
    Decoder : 8 → 16 → 32 → 64 → 128 → 256  (Upsample + concat skip + ResBlock)
    """
    base_ch: int = 32

    @nn.compact
    def __call__(self, x):                              # (B, 256, 256, 3)
        ch = self.base_ch

        # ── stem ─────────────────────────────────────────────────────────
        x0 = nn.Conv(ch, (3, 3), padding="SAME")(x)    # (B, 256, 256, ch)

        # ── encoder ──────────────────────────────────────────────────────
        h1 = ResBlock(ch      )(x0)                               # (256, 256, ch)
        h2 = ResBlock(ch *  2)(Downsample(ch *  2)(h1))          # (128, 128, ch*2)
        h3 = ResBlock(ch *  4)(Downsample(ch *  4)(h2))          # ( 64,  64, ch*4)
        h4 = ResBlock(ch *  4)(Downsample(ch *  4)(h3))          # ( 32,  32, ch*4)
        h5 = ResBlock(ch *  4)(Downsample(ch *  4)(h4))          # ( 16,  16, ch*4)
        b  = ResBlock(ch *  4)(Downsample(ch *  4)(h5))          # (  8,   8, ch*4)

        # ── decoder with skip connections ─────────────────────────────────
        u5 = ResBlock(ch * 4)(
            jnp.concatenate([Upsample(ch * 4)(b),  h5], axis=-1))   # (16, 16, ch*4)
        u4 = ResBlock(ch * 4)(
            jnp.concatenate([Upsample(ch * 4)(u5), h4], axis=-1))   # (32, 32, ch*4)
        u3 = ResBlock(ch * 4)(
            jnp.concatenate([Upsample(ch * 4)(u4), h3], axis=-1))   # (64, 64, ch*4)
        u2 = ResBlock(ch * 2)(
            jnp.concatenate([Upsample(ch * 2)(u3), h2], axis=-1))   # (128,128, ch*2)
        u1 = ResBlock(ch)(
            jnp.concatenate([Upsample(ch)     (u2), h1], axis=-1))  # (256,256, ch)

        return nn.Conv(3, (3, 3), padding="SAME")(u1)               # (B,256,256,3)


class NRAEModelCeleba(nn.Module):
    """
    Full NRAE for CelebA-HQ 256x256 RGB.
    Submodules: enc1, enc2, dec_mlp, unet.
    """
    latent_dim:   int = 700
    enc1_hidden:  int = 10000
    dec_hidden:   int = 10000
    n_dct:        int = 80
    unet_base_ch: int = 32

    def setup(self):
        dct_dim = 3 * self.n_dct ** 2          # 3 * 6400 = 19200
        self.enc1    = Enc1(hidden_dim=self.enc1_hidden)
        self.enc2    = Enc2(latent_dim=self.latent_dim)
        self.dec_mlp = DecMLP(dec_hidden=self.dec_hidden, dct_dim=dct_dim)
        self.unet    = UNetRefinement256RGB(base_ch=self.unet_base_ch)

    def encode(self, x_dct_flat):
        """(B, 19200) → (B, 700)"""
        return self.enc2(self.enc1(x_dct_flat))

    def decode(self, z):
        """(B, 700) → (B, 256, 256, 3)"""
        x_dct_rec = self.dec_mlp(z)
        x_rough   = idct2d_from_crop_rgb_jax(x_dct_rec, self.n_dct)
        return self.unet(x_rough)

    def __call__(self, x_dct_flat):
        return self.decode(self.encode(x_dct_flat))

    def forward_with_intermediates(self, x_dct_flat, rng,
                                   *, sigma, fixed_noise_sigma):
        """Full forward pass + noisy enc1/enc2 for regularisation."""
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
    return x + jax.nn.softplus(-2.0 * x) - math.log(2.0)


def nrae_loss_celeba(params, model: NRAEModelCeleba,
                     x_dct_flat: jnp.ndarray,
                     x_img:      jnp.ndarray,
                     rng:        jnp.ndarray,
                     *,
                     sigma:             float = 2.0,
                     fixed_noise_sigma: float = 1e-3,
                     alpha:             float = 100.0):
    """
    NRAE loss for RGB images.

    x_dct_flat : (B, 19200)
    x_img      : (B, 256, 256, 3) float32 in [0, 1]
    """
    B = x_dct_flat.shape[0]

    rec_image, h_clean, h_noisy, z_clean, z_noisy = model.apply(
        {"params": params},
        x_dct_flat, rng,
        sigma=sigma, fixed_noise_sigma=fixed_noise_sigma,
        method=NRAEModelCeleba.forward_with_intermediates,
    )

    recon = jnp.sum(_log_cosh(alpha * (rec_image - x_img)))
    recon = (1.0 / alpha) * (1.0 / B) * 0.5 * recon

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
                    latent_dim:   int   = 700,
                    enc1_hidden:  int   = 10000,
                    dec_hidden:   int   = 10000,
                    n_dct:        int   = 80,
                    unet_base_ch: int   = 32,
                    lr:           float = 1e-4):
    model     = NRAEModelCeleba(
        latent_dim=latent_dim, enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch)
    dct_dim   = 3 * n_dct ** 2
    dummy_dct = jnp.zeros((1, dct_dim), dtype=jnp.float32)
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
    model = NRAEModelCeleba(
        latent_dim=latent_dim, enc1_hidden=enc1_hidden,
        dec_hidden=dec_hidden, n_dct=n_dct, unet_base_ch=unet_base_ch)

    def _loss(params):
        return nrae_loss_celeba(params, model, x_dct_flat, x_img, rng,
                                sigma=sigma, fixed_noise_sigma=fixed_noise_sigma,
                                alpha=alpha)

    (loss, (recon, reg)), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, recon, reg


# ============================================================
# 4.  Encode / Decode helpers
# ============================================================

def encode_dataset_nrae(model, params, images_np: np.ndarray,
                        *, n_dct: int = 80, batch_size: int = 16) -> np.ndarray:
    """
    (N, 256, 256, 3) float32 in [0,1]  →  latent vectors (N, 700) numpy.

    Applies DCT compression on CPU (numpy) then runs Enc1+Enc2 via JAX.
    """
    Z = []
    for i in range(0, images_np.shape[0], batch_size):
        batch  = images_np[i:i + batch_size]                # (b, 256, 256, 3)
        x_dct  = dct2d_crop_rgb_np(batch, n_dct=n_dct)     # (b, 19200) numpy
        z      = model.apply({"params": params},
                             jnp.asarray(x_dct),
                             method=NRAEModelCeleba.encode)
        Z.append(np.array(z))
    return np.concatenate(Z, axis=0)


def decode_latents_nrae(model, params, z,
                        *, batch_size: int = 16) -> np.ndarray:
    """
    (N, 700) jnp/numpy  →  (N, 256, 256, 3) float32 numpy in [0, 1].
    """
    imgs = []
    for i in range(0, z.shape[0], batch_size):
        rec = model.apply({"params": params},
                          jnp.asarray(z[i:i + batch_size]),
                          method=NRAEModelCeleba.decode)
        imgs.append(np.clip(np.array(rec), 0.0, 1.0))
    return np.concatenate(imgs, axis=0)


# ============================================================
# 5.  Checkpoint utilities
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
    dct_dim = 3 * n_dct ** 2
    print("\n--- NRAE Architecture (CelebA-HQ 256) ---")
    print(f"  Image        : (256, 256, 3)  RGB")
    print(f"  DCT crop     : {n_dct}x{n_dct}x3 = {dct_dim}-dim vector")
    print(f"  Enc1         : {dct_dim} → {enc1_hidden}  [Dense + ELU]")
    print(f"  Enc2         : {enc1_hidden} → {latent_dim}  [Dense, linear]")
    print(f"  Latent dim   : {latent_dim}")
    print(f"  DecMLP       : {latent_dim} → {dec_hidden} → {dct_dim}  [Dense+ELU → Dense]")
    print(f"  iDCT         : {dct_dim} → 3x{n_dct}x{n_dct} → pad → 3x256x256 → iDCT")
    print(f"  UNet (6-lvl) : (256,256,3) → (256,256,3)  [base_ch={unet_base_ch}]")
    print(f"  Parameters   : {count_params(params):,}")
    print()
