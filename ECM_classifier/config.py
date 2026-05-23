"""
config.py
=========
Configuration for the ECM_classifier experiment.

Experiment: minimum ECM classification via MM-SOLD moment-matched marginal.
For each class c, the energy function is:
  E_c(z) = V_c(z) + λ_c^T·z + ½(z - μ_c*)^T·Λ_c·(z - μ_c*)
"""
import os
import sys

# ─── Directory layout ─────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
AUG_DIR = os.path.abspath(os.path.join(HERE, "..", "augmentation_handwritten"))

# Large data files live on the server
_SERVER_ROOT = os.environ.get("DATA_ROOT", "/path/to/datasets/RP1/handwritten")
RAW_IMG      = os.path.join(_SERVER_ROOT, "handwritten_img.npy")
RAW_LABEL    = os.path.join(_SERVER_ROOT, "handwritten_label.npy")

# Small outputs stay in the repo
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR     = os.path.join(HERE, "figures")

# ─── Pre-trained NRAE checkpoint ──────────────────────────────────────────────
NRAE_CKPT = os.path.join(AUG_DIR, "checkpoints", "nrae_best.pkl")

# ─── NRAE architecture (must match nrae_best.pkl) ─────────────────────────────
LATENT_DIM   = 100
ENC1_HIDDEN  = 2048
DEC_HIDDEN   = 2048
N_DCT        = 20
UNET_BASE_CH = 32

# ─── Data split ───────────────────────────────────────────────────────────────
N_TRAIN_PER_CLASS = 1000
N_VAL_PER_CLASS   = 58
N_TEST_PER_CLASS  = 300
N_CLASSES         = 10
DATA_SEED         = 0

# ─── Grid search ──────────────────────────────────────────────────────────────
# σ_gmm = GMM kernel bandwidth
# σ     = LDS smoothing bandwidth (s in the theory)
# M     = number of MC noise samples for estimating g_{s,c} and V_c
SIGMA_GMM_GRID = [0.02, 0.03, 0.05, 0.08]                       # 4 values
SIGMA_GRID     = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]  # 8 values
M_GRID         = [512]                                           # 1 value → 32 cells total

# ─── MM-SOLD / energy estimation hyperparameters ──────────────────────────────
SOLD_SIGMA_GMM = 0.03   # default GMM bandwidth (kept for reference; grid uses SIGMA_GMM_GRID)

# ─── Batch sizes (tune to avoid GPU OOM) ──────────────────────────────────────
ENCODE_BATCH  = 64    # images per batch for NRAE encoding
TEST_BATCH    = 500   # test samples per batch when evaluating V_c

# ─── sys.path registration ────────────────────────────────────────────────────
for _p in (AUG_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
