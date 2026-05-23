"""
vae_model.py
============
ResNet-VAE architecture, loss, training step, encode/decode utilities,
and checkpoint save/load.
"""
import functools, os, pickle
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax


# ============================================================
# 1. Model components
# ============================================================

class ResBlock(nn.Module):
    ch: int
    groups: int = 16

    @nn.compact
    def __call__(self, x):
        h = nn.GroupNorm(num_groups=self.groups)(x)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3, 3), padding="SAME")(h)
        h = nn.GroupNorm(num_groups=self.groups)(h)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3, 3), padding="SAME")(h)
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1), padding="SAME")(x)
        return x + h


class Downsample(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.ch, (4, 4), strides=(2, 2), padding="SAME")(x)


class Upsample(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x):
        x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
        return nn.Conv(self.ch, (3, 3), padding="SAME")(x)


class ResEncoder(nn.Module):
    latent_dim: int
    base_ch: int = 64

    @nn.compact
    def __call__(self, x):          # (B, 64, 64, 1)
        ch = self.base_ch
        x = nn.Conv(ch, (3, 3), padding="SAME")(x)
        x = ResBlock(ch)(x)
        x = Downsample(ch * 2)(x);  x = ResBlock(ch * 2)(x)   # 32
        x = Downsample(ch * 4)(x);  x = ResBlock(ch * 4)(x)   # 16
        x = Downsample(ch * 4)(x);  x = ResBlock(ch * 4)(x)   # 8
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x);  x = nn.swish(x)
        mu     = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar


class ResDecoder(nn.Module):
    latent_dim: int
    base_ch: int = 64

    @nn.compact
    def __call__(self, z):          # (B, latent_dim)
        ch = self.base_ch
        x = nn.Dense(8 * 8 * ch * 4)(z);  x = nn.swish(x)
        x = x.reshape((z.shape[0], 8, 8, ch * 4))
        x = ResBlock(ch * 4)(x);  x = Upsample(ch * 4)(x)    # 16
        x = ResBlock(ch * 4)(x);  x = Upsample(ch * 2)(x)    # 32
        x = ResBlock(ch * 2)(x);  x = Upsample(ch)(x)        # 64
        x = ResBlock(ch)(x)
        return nn.Conv(1, (3, 3), padding="SAME")(x)


class ResVAE(nn.Module):
    latent_dim: int
    base_ch: int = 64

    def setup(self):
        self.enc = ResEncoder(self.latent_dim, base_ch=self.base_ch)
        self.dec = ResDecoder(self.latent_dim, base_ch=self.base_ch)

    def __call__(self, x, key):
        mu, logvar = self.enc(x)
        eps = jax.random.normal(key, mu.shape)
        z   = mu + jnp.exp(0.5 * logvar) * eps
        return self.dec(z), mu, logvar, z

    def encode(self, x):
        return self.enc(x)

    def decode_logits(self, z):
        return self.dec(z)


# ============================================================
# 2. Loss
# ============================================================

def kl_per_example(mu, logvar):
    return -0.5 * jnp.sum(1.0 + logvar - mu ** 2 - jnp.exp(logvar), axis=1)


def vae_loss_bce_logits_sum(x, logits, mu, logvar, *,
                             beta, free_bits, pos_weight):
    per_pixel = optax.sigmoid_binary_cross_entropy(logits, x)
    w = 1.0 + (pos_weight - 1.0) * x
    bce      = (per_pixel * w).sum(axis=(1, 2, 3)).mean()
    kl       = kl_per_example(mu, logvar)
    kl_used  = jnp.maximum(kl, free_bits * mu.shape[-1]).mean()
    return bce + beta * kl_used, (bce, kl_used)


# ============================================================
# 3. Training state & step
# ============================================================

class VAETrainState(train_state.TrainState):
    pass


def make_vae_state(rng, *, latent_dim=100, base_ch=64, lr=1e-4):
    model  = ResVAE(latent_dim=latent_dim, base_ch=base_ch)
    dummy  = jnp.zeros((1, 64, 64, 1), dtype=jnp.float32)
    params = model.init(rng, dummy, rng)["params"]
    return model, VAETrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(lr)
    )


@functools.partial(jax.jit, static_argnames=("latent_dim", "base_ch"))
def vae_train_step(state, batch, rng, *,
                   latent_dim, base_ch, beta, free_bits, pos_weight):
    def loss_fn(params):
        logits, mu, logvar, _z = ResVAE(
            latent_dim=latent_dim, base_ch=base_ch
        ).apply({"params": params}, batch, rng)
        return vae_loss_bce_logits_sum(
            batch, logits, mu, logvar,
            beta=beta, free_bits=free_bits, pos_weight=pos_weight)

    (loss, (bce, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, bce, kl


# ============================================================
# 4. Encode / Decode helpers
# ============================================================

def encode_dataset(model, params, images_np, batch_size=64):
    """(N,64,64) ink=1 -> latent means (N,latent_dim) numpy."""
    Z = []
    for i in range(0, images_np.shape[0], batch_size):
        b  = jnp.asarray(images_np[i:i+batch_size, ..., None])
        mu, _ = model.apply({"params": params}, b, method=model.encode)
        Z.append(np.array(mu))
    return np.concatenate(Z, axis=0)


def decode_latents(model, params, z, batch_size=64):
    """(N,latent_dim) jnp -> (N,64,64) ink=1 numpy."""
    imgs = []
    for i in range(0, z.shape[0], batch_size):
        logits = model.apply({"params": params}, z[i:i+batch_size],
                             method=model.decode_logits)
        imgs.append(np.array(jax.nn.sigmoid(logits)[..., 0]))
    return np.concatenate(imgs, axis=0)


# ============================================================
# 5. Checkpoint utilities
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


# ============================================================
# 6. Utilities
# ============================================================

def count_params(params):
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def print_vae_summary(latent_dim, base_ch, params):
    ch = base_ch
    print("\n--- ResVAE Architecture ---")
    print(f"  ResVAE(latent_dim={latent_dim}, base_ch={base_ch})")
    print(f"  Encoder: (B,64,64,1) -> 32->16->8 -> Dense(512) -> ({latent_dim},)")
    print(f"  Decoder: ({latent_dim},) -> 8x8x{ch*4} -> 16->32->64 -> (B,64,64,1)")
    print(f"  Total parameters: {count_params(params):,}")
    print()
