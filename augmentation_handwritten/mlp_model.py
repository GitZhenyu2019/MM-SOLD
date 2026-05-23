"""
mlp_model.py
============
MLP classifier for latent-space digit recognition.
Includes grid search over weight_decay with val-accuracy-based model selection.
"""
import functools
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from data_utils import iter_minibatches_xy_np


# ============================================================
# 1. Model
# ============================================================

class LatentMLP(nn.Module):
    """3-layer MLP: latent_dim -> hidden_dim -> hidden_dim//2 -> num_classes."""
    num_classes: int = 10
    hidden_dim:  int = 256

    @nn.compact
    def __call__(self, x):   # (B, latent_dim)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x=nn.LayerNorm(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.gelu(x)
        x=nn.LayerNorm(x)
        return nn.Dense(self.num_classes)(x)


# ============================================================
# 2. Training state
# ============================================================

class MLPTrainState(train_state.TrainState):
    pass


def make_mlp_state(rng, *, latent_dim, num_classes=10, hidden_dim=256,
                   lr=1e-3, weight_decay=1e-2, total_steps=10000,
                   warmup_steps=200):
    model  = LatentMLP(num_classes=num_classes, hidden_dim=hidden_dim)
    dummy  = jnp.zeros((1, latent_dim))
    params = model.init(rng, dummy)["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr,
        warmup_steps=warmup_steps, decay_steps=total_steps,
        end_value=lr * 1e-2,
    )
    tx    = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    state = MLPTrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


# ============================================================
# 3. Train / eval steps (JIT)
# ============================================================

@functools.partial(jax.jit, static_argnames=("num_classes", "hidden_dim"))
def mlp_train_step(state, Z, labels, *, num_classes, hidden_dim):
    def loss_fn(params):
        logits = LatentMLP(num_classes=num_classes, hidden_dim=hidden_dim).apply(
            {"params": params}, Z)
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        return loss, acc
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, acc


@functools.partial(jax.jit, static_argnames=("num_classes", "hidden_dim"))
def _mlp_eval_batch(params, Z, labels, *, num_classes, hidden_dim):
    logits  = LatentMLP(num_classes=num_classes, hidden_dim=hidden_dim).apply(
        {"params": params}, Z)
    preds   = jnp.argmax(logits, axis=-1)
    correct = (preds == labels).astype(jnp.int32)
    return correct, preds


def evaluate_mlp(params, Z_np, labels_np, *,
                 num_classes=10, hidden_dim=256, batch_size=512):
    """Returns overall accuracy (%) without printing."""
    all_correct = []
    for i in range(0, Z_np.shape[0], batch_size):
        correct, _ = _mlp_eval_batch(
            params,
            jnp.asarray(Z_np[i:i+batch_size]),
            jnp.asarray(labels_np[i:i+batch_size]),
            num_classes=num_classes, hidden_dim=hidden_dim)
        all_correct.append(np.array(correct))
    return np.concatenate(all_correct).mean() * 100.0


def evaluate_mlp_verbose(params, Z_np, labels_np, *,
                         num_classes=10, hidden_dim=256,
                         batch_size=512, tag=""):
    """Per-class + overall accuracy with printing. Returns overall %."""
    all_correct = []
    for i in range(0, Z_np.shape[0], batch_size):
        correct, _ = _mlp_eval_batch(
            params,
            jnp.asarray(Z_np[i:i+batch_size]),
            jnp.asarray(labels_np[i:i+batch_size]),
            num_classes=num_classes, hidden_dim=hidden_dim)
        all_correct.append(np.array(correct))
    all_correct = np.concatenate(all_correct)
    header = f"--- MLP accuracy{' (' + tag + ')' if tag else ''} ---"
    print(f"\n{header}")
    for c in range(num_classes):
        mask = labels_np == c
        if mask.sum() == 0:
            continue
        cacc = all_correct[mask].mean() * 100
        print(f"  digit {c}: {cacc:5.1f}%  "
              f"({int(all_correct[mask].sum()):>3} / {int(mask.sum())})")
    overall = all_correct.mean() * 100
    print(f"\n  Overall: {overall:.2f}%  ({int(all_correct.sum())} / {Z_np.shape[0]})")
    return overall


# ============================================================
# 4. Grid search
# ============================================================

def run_grid_search(
    Z_train, y_train, Z_val, y_val, Z_test, y_test,
    *,
    latent_dim,
    wd_grid=(1e-3, 1e-2, 5e-2, 1e-1),
    max_epochs=100,
    batch_size=256,
    hidden_dim=256,
    lr=1e-3,
    num_classes=10,
    seed=0,
):
    """
    Grid search over weight_decay.
    For each wd: train for max_epochs, pick the epoch with best val accuracy.
    Returns (best_val_acc, best_test_acc, best_wd, best_epoch, best_params).
    """
    steps_per_epoch = max(1, len(Z_train) // batch_size)
    total_steps     = max_epochs * steps_per_epoch
    warmup_steps    = max(1, total_steps // 10)

    # JIT warmup once before the grid search loop
    dummy_Z = jnp.zeros((batch_size, latent_dim))
    dummy_l = jnp.zeros((batch_size,), dtype=jnp.int32)
    _, _tmp_state = make_mlp_state(
        jax.random.PRNGKey(0), latent_dim=latent_dim,
        num_classes=num_classes, hidden_dim=hidden_dim,
        lr=lr, weight_decay=float(wd_grid[0]),
        total_steps=max(1, total_steps), warmup_steps=max(1, warmup_steps))
    _tmp_state, _, _ = mlp_train_step(
        _tmp_state, dummy_Z, dummy_l,
        num_classes=num_classes, hidden_dim=hidden_dim)
    jax.block_until_ready(_tmp_state.params)

    print(f"\n{'wd':>8}  {'best_val':>9}  {'best_ep':>8}")
    print("-" * 32)

    grid_results = {}
    for wd in wd_grid:
        rng = jax.random.PRNGKey(seed)
        _, state = make_mlp_state(
            rng, latent_dim=latent_dim,
            num_classes=num_classes, hidden_dim=hidden_dim,
            lr=lr, weight_decay=float(wd),
            total_steps=total_steps, warmup_steps=warmup_steps)

        best_val  = 0.0
        best_ep   = 0
        best_prms = None

        for ep in range(max_epochs):
            for Z_b, l_b in iter_minibatches_xy_np(
                Z_train, y_train, batch_size, seed=seed + ep
            ):
                state, _, _ = mlp_train_step(
                    state,
                    jnp.asarray(Z_b),
                    jnp.asarray(l_b),
                    num_classes=num_classes, hidden_dim=hidden_dim)

            val_acc = evaluate_mlp(
                state.params, Z_val, y_val,
                num_classes=num_classes, hidden_dim=hidden_dim)
            if val_acc > best_val:
                best_val  = val_acc
                best_ep   = ep + 1
                best_prms = jax.device_get(state.params)

        grid_results[wd] = (best_val, best_ep, best_prms)
        print(f"  {wd:>6.0e}  {best_val:>8.2f}%  {best_ep:>8}")

    # Pick best wd by val accuracy
    best_wd = max(grid_results, key=lambda w: grid_results[w][0])
    best_val_acc, best_epoch, best_params = grid_results[best_wd]
    best_test_acc = evaluate_mlp(
        best_params, Z_test, y_test,
        num_classes=num_classes, hidden_dim=hidden_dim)

    return best_val_acc, best_test_acc, best_wd, best_epoch, best_params
