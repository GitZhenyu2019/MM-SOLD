"""
config.py
============
Configuration for the handwritten-digit classification experiment
"""
import importlib.util
import os
import sys


def _env(name, default):
    return os.environ.get(name, default)


# ─── Directory layout ─────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
AUG_DIR   = os.path.abspath(os.path.join(HERE, "..", "augmentation_handwritten"))

RELEASE_ROOT   = _env("RP1_RELEASE_ROOT",
                      os.path.abspath(os.path.join(HERE, "..", "..", "release")))
RELEASE_COMMON = os.path.join(RELEASE_ROOT, "common")

NRAE_CKPT = _env("RP1_NRAE_CKPT",
                 os.path.join(RELEASE_ROOT, "digits", "checkpoint", "nrae_best.pkl"))

_HANDWRITTEN_ROOT = _env("RP1_HANDWRITTEN_ROOT",
                         "/path/to/datasets/handwritten")
RAW_IMG   = _env("RP1_RAW_IMG",   os.path.join(_HANDWRITTEN_ROOT, "handwritten_img.npy"))
RAW_LABEL = _env("RP1_RAW_LABEL", os.path.join(_HANDWRITTEN_ROOT, "handwritten_label.npy"))

RESULTS_DIR  = os.path.join(HERE, "results")
FIG_DIR      = os.path.join(HERE, "figures")
LATENT_DIR   = os.path.join(RESULTS_DIR, "latents")

# ─── NRAE architecture ─────────────────
LATENT_DIM   = 100
ENC1_HIDDEN  = 2048
DEC_HIDDEN   = 2048
N_DCT        = 20
UNET_BASE_CH = 32

# ─── Data split ──────────────────────
N_TRAIN_PER_CLASS = 1000
N_VAL_PER_CLASS   = 58
N_TEST_PER_CLASS  = 300
N_CLASSES         = 10
DATA_SEED         = 0

# ─── Top-40 subspace ─────────────────
PROJ_K = 40

# ─── (c_delta, c_sigma) grid ────────────────────────────────────
# {0.05, 0.10, ..., 0.50} in both axes -> 10 x 10 = 100 cells.
C_GRID = [round(0.05 * i, 2) for i in range(1, 11)]
M_MC   = 32  # Monte Carlo budget, fixed everywhere

# ─── MM-SOLD augmentation  ──────
N_AUG_PER_CLASS    = 9000
AUG_NSTEPS         = 500   
AUG_DISCRETIZATION = "LM"

# ─── Consistent dual estimator ───────────────────────────
LION_N_CHAINS = 2000   # P: parallel BAOAB chains (vmap)
LION_N_STEPS  = 500    # L: BAOAB steps per outer Lion iteration
LION_N_ITERS  = 300    # K: number of Lion steps
LION_LR_REL   = 0.01

# ─── MLP baselines ────────────────────────────
MLP_HIDDEN   = 256
MLP_LR       = 1e-3
MLP_EPOCHS   = 100
MLP_BATCH    = 256
MLP_WD_GRID  = [1e-3, 1e-2, 5e-2, 1e-1]

for _p in (AUG_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if RELEASE_COMMON not in sys.path:
    sys.path.insert(0, RELEASE_COMMON)


def _load_module_from_path(name, path):
    """
    Load a module from an explicit file path under a private name.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_release_sampling_algo():
    path = os.path.join(RELEASE_COMMON, "sampling_algo.py")
    return _load_module_from_path("sampling_algo_release", path)


def describe():
    rows = [
        ("AUG_DIR", AUG_DIR), ("RELEASE_ROOT", RELEASE_ROOT),
        ("RELEASE_COMMON", RELEASE_COMMON), ("NRAE_CKPT", NRAE_CKPT),
        ("RAW_IMG", RAW_IMG), ("RAW_LABEL", RAW_LABEL),
        ("RESULTS_DIR", RESULTS_DIR), ("LATENT_DIR", LATENT_DIR),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        mark = "" if os.path.exists(v) else "  [missing]"
        print(f"  {k:<{width}}  {v}{mark}")


if __name__ == "__main__":
    print("Resolved configuration (config.py):\n")
    describe()
