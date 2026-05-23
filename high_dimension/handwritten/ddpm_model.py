"""
ddpm_model.py
=============
MLP-based DDPM operating in the 100-dim NRAE latent space.
"""
import os
import pickle
import functools
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax


# ─── Sinusoidal time embedding ─────────────────────────────────────────────────

def sinusoidal_embedding(t: jnp.ndarray, dim: int = 128) -> jnp.ndarray:
    half  = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
    args  = t[:, None].astype(jnp.float32) * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


# ─── Residual block with AdaLN time conditioning (DiT-style) ──────────────────

class ResBlock(nn.Module):
    """
    MLP residual block with Adaptive LayerNorm (AdaLN).
    """
    hidden: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        # Predict scale1, shift1, scale2, shift2 from t_emb (zero-init for stability)
        ada = nn.Dense(
            4 * self.hidden,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(nn.swish(t_emb))                                   # (B, 4*hidden)
        s1, b1, s2, b2 = jnp.split(ada, 4, axis=-1)         # each (B, hidden)

        # First sub-block: AdaLN → Swish → Dense
        h = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        h = nn.swish((1.0 + s1) * h + b1)
        h = nn.Dense(self.hidden)(h)

        # Second sub-block: AdaLN → Swish → Dense
        h = nn.LayerNorm(use_scale=False, use_bias=False)(h)
        h = nn.swish((1.0 + s2) * h + b2)
        h = nn.Dense(self.hidden)(h)

        # Skip connection
        if x.shape[-1] != self.hidden:
            x = nn.Dense(self.hidden)(x)
        return x + h


# ─── MLP denoiser ─────────────────────────────────────────────────────────────

class MLPDenoiser(nn.Module):
    """
    Noise predictor ε_θ(z_t, t) with DiT-style AdaLN conditioning.
    """
    latent_dim : int = 100
    hidden     : int = 512
    n_layers   : int = 6
    t_emb_dim  : int = 128

    @nn.compact
    def __call__(self, z_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # Time embedding: sinusoidal → 2-layer MLP
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)     # (B, t_emb_dim)
        t_emb = nn.swish(nn.Dense(self.hidden)(t_emb))
        t_emb = nn.Dense(self.hidden)(t_emb)                 # (B, hidden)

        # Input projection: z_t only (t_emb conditions via AdaLN in each block)
        x = nn.Dense(self.hidden)(z_t)                       # (B, hidden)

        # Residual blocks with AdaLN
        for _ in range(self.n_layers):
            x = ResBlock(self.hidden)(x, t_emb)

        # Output head: final AdaLN → Dense(latent_dim)
        ada_out = nn.Dense(
            2 * self.hidden,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(nn.swish(t_emb))
        s_out, b_out = jnp.split(ada_out, 2, axis=-1)
        x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        x = nn.swish((1.0 + s_out) * x + b_out)
        return nn.Dense(self.latent_dim)(x)


# ─── Cosine beta schedule ──────────────────────────────────────────────────────

def make_cosine_schedule(T: int, s: float = 8e-3):
    """
    Cosine schedule from Nichol & Dhariwal (2021).
    """
    t    = np.arange(T + 1, dtype=np.float64)
    frac = (t / T + s) / (1.0 + s)
    ac   = np.cos(frac * np.pi / 2.0) ** 2
    ac   = ac / ac[0]
    betas = 1.0 - ac[1:] / ac[:-1]
    betas = np.clip(betas, 0.0, 0.999)
    ac    = np.concatenate([[1.0], np.cumprod(1.0 - betas)])
    return ac.astype(np.float32), betas.astype(np.float32)


# ─── Training state ───────────────────────────────────────────────────────────

class DDPMTrainState(train_state.TrainState):
    pass


def make_ddpm_state(
    rng,
    *,
    latent_dim: int = 100,
    hidden: int = 512,
    n_layers: int = 6,
    t_emb_dim: int = 128,
    T: int = 1000,
    lr: float = 1e-4,
    wd: float = 1e-4,
    total_steps: int = 50_000,
):
    """Initialise DDPM model + AdamW training state."""
    model   = MLPDenoiser(latent_dim=latent_dim, hidden=hidden,
                          n_layers=n_layers, t_emb_dim=t_emb_dim)
    dummy_z = jnp.zeros((1, latent_dim))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    params  = model.init(rng, dummy_z, dummy_t)["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=min(2000, max(1, total_steps // 20)),
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )
    tx    = optax.adamw(learning_rate=schedule, weight_decay=wd)
    state = DDPMTrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


# ─── Training step ────────────────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=("T", "latent_dim", "hidden", "n_layers"))
def ddpm_train_step(
    state,
    z0: jnp.ndarray,
    key: jnp.ndarray,
    *,
    alphas_cumprod: jnp.ndarray,
    T: int,
    latent_dim: int,
    hidden: int,
    n_layers: int,
):
    """
    One DDPM training step (simple noise prediction loss).
    """
    B = z0.shape[0]
    key, k_t, k_eps = jax.random.split(key, 3)
    t     = jax.random.randint(k_t, (B,), 0, T, dtype=jnp.int32)
    eps   = jax.random.normal(k_eps, z0.shape, dtype=z0.dtype)
    ac_t  = alphas_cumprod[t + 1][:, None]          # (B,1)
    z_t   = jnp.sqrt(ac_t) * z0 + jnp.sqrt(1.0 - ac_t) * eps

    def loss_fn(params):
        eps_hat = MLPDenoiser(
            latent_dim=latent_dim, hidden=hidden, n_layers=n_layers
        ).apply({"params": params}, z_t, t)
        return jnp.mean((eps_hat - eps) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss, key


# ─── DDIM sampling ────────────────────────────────────────────────────────────

def ddim_sample(
    params,
    rng,
    *,
    alphas_cumprod: np.ndarray,
    n_samples: int = 1500,
    latent_dim: int = 100,
    hidden: int = 512,
    n_layers: int = 6,
    T: int = 1000,
    ddim_steps: int = 100,
    clip_x0: float = 15.0,
) -> np.ndarray:
    """
    DDIM deterministic sampling (eta=0).
    """
    model = MLPDenoiser(latent_dim=latent_dim, hidden=hidden, n_layers=n_layers)
    ac    = np.array(alphas_cumprod, dtype=np.float64)   # (T+1,)

    # Uniformly spaced DDIM timesteps T→0
    step_idx = np.linspace(0, T - 1, ddim_steps, dtype=int)
    step_idx = step_idx[::-1]                             # descending

    rng, k_init = jax.random.split(rng)
    z = jax.random.normal(k_init, (n_samples, latent_dim), dtype=jnp.float32)

    for i, t_idx in enumerate(step_idx):
        t_batch = jnp.full((n_samples,), int(t_idx), dtype=jnp.int32)
        eps_hat = model.apply({"params": params}, z, t_batch)

        ac_t    = float(ac[t_idx + 1])
        x0_pred = (z - jnp.sqrt(1.0 - ac_t) * eps_hat) / jnp.sqrt(ac_t)
        x0_pred = jnp.clip(x0_pred, -clip_x0, clip_x0)

        if i < ddim_steps - 1:
            t_next  = int(step_idx[i + 1])
            ac_next = float(ac[t_next + 1])
            z = jnp.sqrt(ac_next) * x0_pred + jnp.sqrt(1.0 - ac_next) * eps_hat
        else:
            z = x0_pred

    jax.block_until_ready(z)
    return np.array(z, dtype=np.float32)


# ─── Checkpoint utilities ─────────────────────────────────────────────────────

def save_ddpm(path: str, params, info: dict = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {"params": jax.device_get(params)}
    if info:
        data["info"] = info
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_ddpm(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["params"], data.get("info", {})


def count_ddpm_params(params) -> int:
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params)))
