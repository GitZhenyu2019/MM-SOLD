"""
config.py
============
Central configuration for the handwritten digit-8 generation experiment.
"""
import os
import sys

# ─── Directory layout ─────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
AUG_DIR  = os.path.abspath(os.path.join(HERE, "..", "..", "augmentation_handwritten"))


def _env(name, default):
    return os.environ.get(name, default)


_SERVER_ROOT = _env("RP1_HANDWRITTEN_ROOT",
                    "/path/to/datasets/handwritten")
RAW_IMG   = _env("RP1_RAW_IMG",   os.path.join(_SERVER_ROOT, "handwritten_img.npy"))
RAW_LABEL = _env("RP1_RAW_LABEL", os.path.join(_SERVER_ROOT, "handwritten_label.npy"))

DATA_DIR    = os.path.join(_SERVER_ROOT, "data_top40")
CKPT_DIR    = os.path.join(HERE, "checkpoints")
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR     = os.path.join(HERE, "figures")

# ─── release/ (the checkpoints + common estimator library) ──────────────
RELEASE_ROOT = _env("RP1_RELEASE_ROOT",
                    os.path.abspath(os.path.join(HERE, "..", "..", "..", "release")))
RELEASE_COMMON = os.path.join(RELEASE_ROOT, "common")

# the trained NRAE checkpoint
NRAE_CKPT = _env("RP1_NRAE_CKPT",
                 os.path.join(RELEASE_ROOT, "digits", "checkpoint", "nrae_best.pkl"))

# CNN classifier checkpoint for KID/Recall features
CLF_CKPT = _env("RP1_CLF_CKPT",
                os.path.join(AUG_DIR, "checkpoints", "classifier_baseline_best.pkl"))

# ─── NRAE architecture ──────────────────
LATENT_DIM   = 100
ENC1_HIDDEN  = 2048
DEC_HIDDEN   = 2048
N_DCT        = 20
UNET_BASE_CH = 32

# ─── Data split ──
N_CLASSES        = 10
TARGET_DIGIT     = 8
N_TRAIN_PER_CLASS = 1000
N_VAL_PER_CLASS   = 58     # unused by this experiment; kept so the split
                           # call matches this section's convention exactly
N_TEST_PER_CLASS  = 300
IMG_SIZE          = 64
SPLIT_SEED        = 0      # rng seed for select_train_val_test_split

# ─── Top-40 pooled PCA subspace (dsub) ────────────────────────────────────────
PROJ_K = 40   # dsub
MIN_VARIANCE_RETAINED = 0.998   # paper claims >99.8%; step1 warns if not met

# ─── Common (c_delta, c_sigma, M) grid -- shared by MM-SOLD, sigma-CFDM,
#     Plug-in BAOAB, SG-BAOAB ─────────────────────────────────────────
CDELTA_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
CSIGMA_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
M_GRID      = [2, 32]

TIMING_CDELTA = 0.05
TIMING_CSIGMA = 0.05
TIMING_M      = 2
TIMING_N_WARMUP = 1
TIMING_N_TIMED  = 5     # paper: "averaged over five timed runs"

# ─── MM-SOLD fixed hyperparameters ───────────────────────────────────
SOLD_NSTEPS      = 1000     # LM steps
SOLD_H_COEF      = 0.2      # h = SOLD_H_COEF * delta^2
SOLD_DISCRETIZ   = "LM"
SOLD_SHARED_NOISE = False
SOLD_FIXED_NOISE  = False

# ─── sigma-CFDM fixed hyperparameters ─────────────────────────────────────────
CFDM_NSTEPS       = 1000    # Euler steps; t: 0.001 -> 0.999 automatically
CFDM_BATCH        = 500
CFDM_SHARED_NOISE = False

# ─── Plug-in BAOAB / SG-BAOAB shared BAOAB sampler settings ───────────────────
BAOAB_NSTEPS   = 1000        # sampling steps (independent of Lion's L)
BAOAB_SPLITTING = "BAOAB"
BAOAB_SCORE     = "full"     # full-softmax score (small dataset, per E.2)

# SG-BAOAB Lion optimisation settings (E.2 "SG-BAOAB sampler")
LION_N_CHAINS = 100   # P
LION_N_STEPS  = 200   # L (BAOAB steps per outer Lion iteration)
LION_N_ITERS  = 300   # K (Lion steps)
LION_LR_REL   = 0.02
LION_B1       = 0.9
LION_B2       = 0.99
LION_WARM_START = True

# ─── KNN-PFlow grid ────────────────────────────────────────────────
KNNPF_CMAX_GRID   = [1, 2, 4]
KNNPF_CMIN_GRID   = [0.025, 0.05, 0.10]
KNNPF_K_GRID      = [32, 128, "N"]     # "N" = exact / full training set
KNNPF_SOLVER_GRID = ["euler", "heun"]
KNNPF_NSTEPS      = 1000

# KNN-PFlow timing configuration (E.2 "Timing")
KNNPF_TIMING_CMAX   = 1
KNNPF_TIMING_CMIN   = 0.10
KNNPF_TIMING_K      = 32
KNNPF_TIMING_SOLVER = "euler"
KNNPF_TIMING_NSTEPS = 1000

# ─── SWF grid ──────────────────────────────────────────────────────
SWF_NSWF_GRID = [300, 1000]
SWF_R_GRID    = [64, 128, 256]
SWF_ETA_GRID  = [0.05, 0.10]
SWF_INIT_GRID = ["gaussian_full", "noised_empirical"]

# SWF timing configuration (E.2 "Timing")
SWF_TIMING_NSWF = 300
SWF_TIMING_R    = 64
SWF_TIMING_ETA  = 0.05
SWF_TIMING_INIT = "gaussian_full"

# ─── Latent DDPM (trained directly in the 40-dim subspace) ────────────────────
DDPM_HIDDEN   = 256
DDPM_N_LAYERS = 4
DDPM_T_STEPS  = 1000
DDIM_STEPS    = 100
DDPM_LR       = 1e-4
DDPM_WD       = 1e-3
DDPM_EPOCHS   = 50000
DDPM_BATCH    = 128

# ─── Evaluation protocol ───────────────────────────────────────────────────────
N_PER_REPEAT = 300   # samples generated per hyperparameter cell per repeat
N_REPEATS    = 5     # independent repeats (fresh RNG), NOT bootstrap resampling

KID_DEGREE     = 3
KID_SUBSET     = 100
KID_N_SUBSETS  = 50
K_RECALL       = 3
DUP_PERCENTILE = 5

# column.
REP_CELL_DUPRATE_CAP  = 0.1
REP_CELL_RECALL_FLOOR = 0.7

for _p in (AUG_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if RELEASE_COMMON not in sys.path:
    sys.path.insert(0, RELEASE_COMMON)


def describe():
    rows = [
        ("AUG_DIR", AUG_DIR), ("RELEASE_ROOT", RELEASE_ROOT),
        ("RELEASE_COMMON", RELEASE_COMMON),
        ("RAW_IMG", RAW_IMG), ("RAW_LABEL", RAW_LABEL),
        ("DATA_DIR", DATA_DIR), ("CKPT_DIR", CKPT_DIR),
        ("RESULTS_DIR", RESULTS_DIR),
        ("NRAE_CKPT", NRAE_CKPT), ("CLF_CKPT", CLF_CKPT),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        flag = "" if os.path.exists(v) else "  [missing]"
        print(f"  {k:<{width}}  {v}{flag}")


if __name__ == "__main__":
    print("Resolved config paths:\n")
    describe()
