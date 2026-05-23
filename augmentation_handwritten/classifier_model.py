"""
classifier_model.py
===================
CNN classifier (ResNet-style with GroupNorm+Swish) for 10-class digit recognition.
Uses Flax/JAX with AdamW + warmup-cosine LR schedule.
"""
import functools, os, pickle
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from data_utils import iter_minibatches_xy_np


# ============================================================
# 1. Model
# ============================================================

class ClassResBlock(nn.Module):
    """Residual block with GroupNorm + Swish activation."""
    ch: int
    groups: int = 8

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


class CNNClassifier(nn.Module):
    """
    5-stage ResNet classifier for 64x64 greyscale images.

    Stage sizes with base_ch=32:
      Input   : (B, 64, 64, 1)
      64x64   : ch=32   (stem Conv + ResBlock)
      32x32   : ch=64   (stride-2 Conv + ResBlock)
      16x16   : ch=128  (stride-2 Conv + ResBlock)
       8x8    : ch=256  (stride-2 Conv + ResBlock)
       4x4    : ch=256  (stride-2 Conv + ResBlock)
      Head    : GlobalAvgPool -> LayerNorm -> Dense(256) -> Dense(num_classes)
    """
    num_classes: int = 10
    base_ch: int = 32

    @nn.compact
    def __call__(self, x):   # (B, 64, 64, 1)
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
        x = jnp.mean(x, axis=(1, 2))   # GlobalAvgPool -> (B, ch*8)
        x = nn.LayerNorm()(x)
        x = nn.Dense(256)(x)
        x = nn.swish(x)
        return nn.Dense(self.num_classes)(x)   # logits


# ============================================================
# 2. Training state
# ============================================================

class ClfTrainState(train_state.TrainState):
    pass


def make_classifier_state(
    rng,
    *,
    num_classes: int = 10,
    base_ch: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    total_steps: int = 10000,
    warmup_steps: int = 500,
):
    """Create classifier TrainState with AdamW + warmup-cosine schedule."""
    model  = CNNClassifier(num_classes=num_classes, base_ch=base_ch)
    dummy  = jnp.zeros((1, 64, 64, 1), dtype=jnp.float32)
    params = model.init(rng, dummy)["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 1e-3,
    )
    tx    = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    state = ClfTrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


# ============================================================
# 3. Train step
# ============================================================

@functools.partial(jax.jit, static_argnames=("num_classes", "base_ch"))
def clf_train_step(state, images, labels, *, num_classes, base_ch):
    """One gradient step; returns updated state, loss, accuracy."""
    def loss_fn(params):
        logits = CNNClassifier(
            num_classes=num_classes, base_ch=base_ch
        ).apply({"params": params}, images)
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        )
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        return loss, acc

    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, acc


# ============================================================
# 4. Evaluation
# ============================================================

@functools.partial(jax.jit, static_argnames=("num_classes", "base_ch"))
def _clf_eval_batch(params, images, labels, *, num_classes, base_ch):
    logits  = CNNClassifier(
        num_classes=num_classes, base_ch=base_ch
    ).apply({"params": params}, images)
    preds   = jnp.argmax(logits, axis=-1)
    correct = (preds == labels).astype(jnp.int32)
    return correct, preds


def evaluate_classifier(
    params,
    images_np: np.ndarray,
    labels_np: np.ndarray,
    *,
    num_classes: int = 10,
    base_ch: int = 32,
    batch_size: int = 128,
    tag: str = "",
):
    """
    Evaluate classifier; print per-class and overall accuracy.
    images_np : (N, 64, 64) float32, ink=1
    labels_np : (N,) int32
    Returns overall accuracy (float, %).
    """
    N = images_np.shape[0]
    all_correct, all_preds = [], []

    for i in range(0, N, batch_size):
        imgs = jnp.asarray(images_np[i:i+batch_size, ..., None])
        lbls = jnp.asarray(labels_np[i:i+batch_size])
        correct, preds = _clf_eval_batch(
            params, imgs, lbls, num_classes=num_classes, base_ch=base_ch
        )
        all_correct.append(np.array(correct))
        all_preds.append(np.array(preds))

    all_correct = np.concatenate(all_correct)
    all_preds   = np.concatenate(all_preds)

    header = f"--- Classifier accuracy{' (' + tag + ')' if tag else ''} ---"
    print(f"\n{header}")
    for c in range(num_classes):
        mask = labels_np == c
        if mask.sum() == 0:
            continue
        cacc = all_correct[mask].mean() * 100
        print(f"  digit {c}: {cacc:5.1f}%  "
              f"({int(all_correct[mask].sum()):>3} / {int(mask.sum())})")

    overall = all_correct.mean() * 100
    print(f"\n  Overall: {overall:.2f}%  "
          f"({int(all_correct.sum())} / {N})")
    return overall


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

def count_params(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def print_classifier_summary(num_classes, base_ch, params):
    ch = base_ch
    print("\n--- CNNClassifier Architecture ---")
    print(f"  Input:   (B, 64, 64, 1)")
    print(f"  64x64  : Conv({ch})    + ClassResBlock({ch},  groups=8)")
    print(f"  32x32  : Conv({ch*2}, s=2) + ClassResBlock({ch*2}, groups=8)")
    print(f"  16x16  : Conv({ch*4}, s=2) + ClassResBlock({ch*4}, groups=8)")
    print(f"   8x8   : Conv({ch*8}, s=2) + ClassResBlock({ch*8}, groups=8)")
    print(f"   4x4   : Conv({ch*8}, s=2) + ClassResBlock({ch*8}, groups=8)")
    print(f"  Head   : GlobalAvgPool({ch*8}) -> LayerNorm -> Dense(256) -> Dense({num_classes})")
    print(f"  Optimizer : AdamW + warmup-cosine LR schedule")
    print(f"  Total parameters: {count_params(params):,}")
    print()


def eval_accuracy_np(
    params,
    images_np: np.ndarray,
    labels_np: np.ndarray,
    *,
    num_classes: int = 10,
    base_ch: int = 32,
    batch_size: int = 128,
) -> float:
    """Return overall accuracy (%) without printing. Used for grid search."""
    N = images_np.shape[0]
    correct_count = 0
    for i in range(0, N, batch_size):
        imgs = jnp.asarray(images_np[i:i+batch_size, ..., None])
        lbls = jnp.asarray(labels_np[i:i+batch_size])
        correct, _ = _clf_eval_batch(
            params, imgs, lbls, num_classes=num_classes, base_ch=base_ch
        )
        correct_count += int(np.array(correct).sum())
    return correct_count / N * 100.0


def run_cnn_grid_search(
    rng,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    *,
    num_classes: int = 10,
    base_ch: int = 32,
    lr: float = 3e-4,
    wd_grid=None,
    max_epochs: int = 80,
    batch_size: int = 64,
    seed: int = 0,
):
    """
    Grid search over weight_decay for the CNN classifier.
    Model selected by best validation accuracy.

    Returns
    -------
    best_val_acc  : float  (%)
    best_test_acc : None   (caller must evaluate on test set)
    best_wd       : float
    best_epoch    : int
    best_params   : pytree
    """
    if wd_grid is None:
        wd_grid = [1e-4, 1e-3, 1e-2, 5e-2]

    N_train = train_images.shape[0]
    steps_per_epoch = max(1, N_train // batch_size)
    total_steps     = max_epochs * steps_per_epoch
    warmup_steps    = max(1, total_steps // 10)

    overall_best_val  = -1.0
    overall_best_wd   = wd_grid[0]
    overall_best_ep   = 1
    overall_best_params = None

    for wd in wd_grid:
        print(f"  [grid] weight_decay={wd:.0e} ...", flush=True)
        rng, sub = jax.random.split(rng)
        _, state = make_classifier_state(
            sub,
            num_classes=num_classes,
            base_ch=base_ch,
            lr=lr,
            weight_decay=wd,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        best_val_this_wd  = -1.0
        best_params_this_wd = None
        best_ep_this_wd   = 1

        for ep in range(max_epochs):
            for imgs_np, lbls_np in iter_minibatches_xy_np(
                train_images, train_labels, batch_size, seed=seed + ep
            ):
                imgs = jnp.asarray(imgs_np[..., None])
                lbls = jnp.asarray(lbls_np)
                state, _, _ = clf_train_step(
                    state, imgs, lbls,
                    num_classes=num_classes, base_ch=base_ch,
                )

            val_acc = eval_accuracy_np(
                state.params, val_images, val_labels,
                num_classes=num_classes, base_ch=base_ch,
            )
            if val_acc > best_val_this_wd:
                best_val_this_wd    = val_acc
                best_params_this_wd = jax.device_get(state.params)
                best_ep_this_wd     = ep + 1

        print(f"    best val_acc={best_val_this_wd:.2f}%  (epoch {best_ep_this_wd})")

        if best_val_this_wd > overall_best_val:
            overall_best_val    = best_val_this_wd
            overall_best_wd     = wd
            overall_best_ep     = best_ep_this_wd
            overall_best_params = best_params_this_wd

    return overall_best_val, overall_best_wd, overall_best_ep, overall_best_params
