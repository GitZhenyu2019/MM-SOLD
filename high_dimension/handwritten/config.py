"""
config.py
=========
Central configuration for the high_dimension/handwritten experiment.
"""
import os
import sys

# ─── Directory layout ─────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
AUG_DIR     = os.path.abspath(os.path.join(HERE, "..", "..", "augmentation_handwritten"))

_SERVER_ROOT = os.environ.get("DATA_ROOT", "/path/to/datasets/RP1/handwritten")
RAW_IMG     = os.path.join(_SERVER_ROOT, "handwritten_img.npy")    # raw images (N,H,W)
RAW_LABEL   = os.path.join(_SERVER_ROOT, "handwritten_label.npy")  # labels (N,) or (N,k)
DATA_DIR    = os.path.join(_SERVER_ROOT, "data")     # preprocessed images + labels
LATENT_DIR  = os.path.join(_SERVER_ROOT, "latents")  # NRAE-encoded latents

CKPT_DIR    = os.path.join(HERE, "checkpoints")  # DDPM checkpoint
RESULTS_DIR = os.path.join(HERE, "results")      # metric grids (JSON / npy)
FIG_DIR     = os.path.join(HERE, "figures")      # plots

# ─── Pre-trained model checkpoints ────────────────────────────────────────────
# NRAE trained on all 10 digit classes (augmentation_handwritten step3b)
NRAE_CKPT   = os.path.join(AUG_DIR, "checkpoints", "nrae_best.pkl")
# CNN classifier feature extractor (256-dim penultimate layer)
CLF_CKPT    = os.path.join(AUG_DIR, "checkpoints", "classifier_baseline_best.pkl")

# ─── Data subset ──────────────────────────────────────────────────────────────
TARGET_DIGIT = 8       # only digit "8" used in this experiment
N_TRAIN      = 1000   # training samples (digit 8 only)
N_TEST       = 300    # test samples     (digit 8 only)
IMG_SIZE     = 64      # images are 64×64 greyscale

# ─── NRAE architecture (must match nrae_best.pkl) ─────────────────────────────
LATENT_DIM   = 100
ENC1_HIDDEN  = 2048
DEC_HIDDEN   = 2048
N_DCT        = 20
UNET_BASE_CH = 32

# ─── Latent DDPM hyperparameters ─────────────────────────────────────────────
DDPM_HIDDEN   = 256      # MLP hidden dim  (AdaLN: ~1.3M params, suitable for N=1000)
DDPM_N_LAYERS = 4        # residual blocks
DDPM_T_STEPS  = 1000     # diffusion timesteps
DDIM_STEPS    = 100      # deterministic inference steps
DDPM_LR       = 1e-4
DDPM_WD       = 1e-3     # stronger regularisation for small dataset
DDPM_EPOCHS   = 50000
DDPM_BATCH    = 128

# ─── Grid search ──────────────────────────────────────────────────────────────
SIGMA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]   # 8 values
M_GRID     = [2, 4, 6, 8, 16, 32]                          # 6 values  → 48 cells

# ─── MM-SOLD fixed hyperparameters ────────────────────────────────────────────
SOLD_NSTEPS        = 100       # Langevin steps
SOLD_H             = 5e-4      # step size
SOLD_SIGMA_GMM     = 0.03      # GMM kernel bandwidth (fixed; σ_smoothing is grid-searched)
SOLD_DISCRETIZ     = "LM"      # Leimkuhler-Matthews discretization
SOLD_K_WHITEN      = 20        # top-k eigencomponents for whitening
SOLD_SHARED_NOISE  = False     # different groups of M LDS noise between particles
SOLD_FIXED_NOISE   = False   # resample M LDS noise independently at each Langevin step

# ─── σ-CFDM fixed hyperparameters ────────────────────────────────────────────
CFDM_NSTEPS       = 100    # Euler ODE steps (t: h → 0.99, step = 1/nsteps)
CFDM_BATCH        = 300    # particles per JIT call
CFDM_SHARED_NOISE = False  # per-particle M noise vectors (consistent with MM-SOLD)

# ─── Evaluation ───────────────────────────────────────────────────────────────
N_GENERATE   = 300  
N_EVAL       = 300   
N_BOOTSTRAP  = 5      

# KID: polynomial-kernel MMD  k(x,y) = (x·y/d + 1)^3
KID_DEGREE   = 3
KID_GAMMA    = None   # set to 1/d at runtime

# DupRate threshold: 5th percentile of within-training-set NN distances
DUP_PERCENTILE = 5

# Sampling-time measurement: only at this (σ, M) to keep runtime manageable
TIMING_SIGMA = 0.1
TIMING_M     = 2

# ─── Visualisation ────────────────────────────────────────────────────────────
VIS_GRID    = 8    # 8×8 image grids
VIS_SEED    = 0
VIS_NN      = 5    # nearest-neighbour examples per query

# ─── sys.path registration (runs once on first import of config) ──────────────
for _p in (AUG_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
