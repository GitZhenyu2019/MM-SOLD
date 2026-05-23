"""
config.py
=========
Central configuration for the high_dimension/celebahq256 experiment.

Experiment: Compare MM-SOLD vs σ-CFDM vs Latent-DDPM
            on CelebA-HQ-256 faces encoded into 700-dim NRAE latent space.
"""
import os
import sys

# ─── Directory layout ─────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
HW_DIR      = os.path.abspath(os.path.join(HERE, "..", "handwritten"))
AUG_DIR     = os.path.abspath(os.path.join(HERE, "..", "..", "augmentation_handwritten"))

# Large data files live on the server
_SERVER_ROOT   = os.environ.get("DATA_ROOT", "/path/to/datasets/RP1/celebahq256")
TRAIN_DIR      = os.path.join(_SERVER_ROOT, "train")       # all 27000 train PNGs
VALID_DIR      = os.path.join(_SERVER_ROOT, "valid")       # all 3000 valid PNGs
TRAIN_USE_DIR  = os.path.join(_SERVER_ROOT, "train_use")   # first 5000 (subset)
VALID_USE_DIR  = os.path.join(_SERVER_ROOT, "valid_use")   # first 500  (subset)
DATA_DIR       = os.path.join(_SERVER_ROOT, "data")        # images_train/test .npy
LATENT_DIR     = os.path.join(_SERVER_ROOT, "latents")     # Z_train/test .npy

# Small outputs stay in the repo
CKPT_DIR    = os.path.join(HERE, "checkpoints")
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR     = os.path.join(HERE, "figures")

# ─── Pre-trained NRAE checkpoint (trained in step1) ───────────────────────────
NRAE_CKPT   = os.path.join(CKPT_DIR, "nrae_best.pkl")

# ─── Data subset ──────────────────────────────────────────────────────────────
N_TRAIN      = 27000    # training samples
N_TEST       = 3000     # test samples
IMG_SIZE     = 256     # 256×256 RGB
IMG_CHANNELS = 3

# ─── NRAE architecture (RGB 256×256, following σ-CFDM paper for CelebA) ───────
LATENT_DIM   = 700
ENC1_HIDDEN  = 10000
DEC_HIDDEN   = 10000
N_DCT        = 80      # keep 80×80 DCT coefficients per channel → 19200 dims
UNET_BASE_CH = 32      # UNet base channels (6-level, handles 256×256)

# ─── Latent DDPM hyperparameters ─────────────────────────────────────────────
DDPM_HIDDEN   = 512
DDPM_N_LAYERS = 8
DDPM_T_STEPS  = 1000
DDIM_STEPS    = 100
DDPM_LR       = 1e-4
DDPM_WD       = 1e-3
DDPM_EPOCHS   = 100000
DDPM_BATCH    = 256    

# ─── Grid search ──────────────────────────────────────────────────────────────
SIGMA_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]          # 8 values: crossover → high-σ SOLD wins
M_GRID     = [2, 4, 6, 8, 16, 32]                                # 6 values → 48 cells

# ─── MM-SOLD fixed hyperparameters ────────────────────────────────────────────
SOLD_NSTEPS        = 100
SOLD_H             = 5e-4      # same as handwritten (whitened space geometry is similar)
SOLD_SIGMA_GMM     = 0.05      # GMM bandwidth in whitened space
SOLD_DISCRETIZ     = "LM"
SOLD_K_WHITEN      = 500       # cover most variance in 700-dim latent space
SOLD_SHARED_NOISE  = False
SOLD_FIXED_NOISE   = False

# ─── σ-CFDM fixed hyperparameters ────────────────────────────────────────────
CFDM_NSTEPS       = 100     
CFDM_BATCH        = 500
CFDM_SHARED_NOISE = False

# ─── Evaluation ───────────────────────────────────────────────────────────────
N_GENERATE   = 3000   # samples generated per config (= N_TEST)
N_EVAL       = 3000    # subsample per bootstrap replicate (= N_TEST)
N_BOOTSTRAP  = 5      # bootstrap replicates

KID_DEGREE   = 3
DUP_PERCENTILE = 1        # 1st-percentile within-train NN dist

# Timing: only at this (σ, M)
TIMING_SIGMA = 1.0  
TIMING_M     = 2

# ─── Visualisation ────────────────────────────────────────────────────────────
VIS_GRID    = 8    # 8×8 image grids
VIS_SEED    = 0

# ─── sys.path registration ────────────────────────────────────────────────────
# augmentation_handwritten: sampling_algo, whitening_utils, vae_model
# handwritten: cfdm_sampling, ddpm_model
for _p in (AUG_DIR, HW_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
