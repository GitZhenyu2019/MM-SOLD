"""
config.py
=================
Central configuration for the CelebA-HQ 256x256 generation experiment, implemented directly in the precomputed
512-dim DiffAE latent space (a fixed, externally pretrained encoder -- no
autoencoder training happens in this experiment).
"""
import os
import sys

# ─── Directory layout ─────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
HW_DIR  = os.path.abspath(os.path.join(HERE, "..", "handwritten"))
AUG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "augmentation_handwritten"))


def _env(name, default):
    return os.environ.get(name, default)


DATA_DIFFAE_DIR = _env("RP1_DIFFAE_DATA_ROOT", os.path.join(HERE, "data"))

DIFFAE_ZSEM_TRAIN     = _env("RP1_DIFFAE_ZSEM_TRAIN",
                             os.path.join(DATA_DIFFAE_DIR, "diffae_zsem27k.npy"))
DIFFAE_ZSEM_VALID     = _env("RP1_DIFFAE_ZSEM_VALID",
                             os.path.join(DATA_DIFFAE_DIR, "zsem_valid3k.npy"))
DIFFAE_FEATURES_NPZ   = _env("RP1_DIFFAE_FEATURES_NPZ",
                             os.path.join(DATA_DIFFAE_DIR, "zsem_features_s025nn_g2_h20.npz"))
DIFFAE_UNET_ONNX      = _env("RP1_DIFFAE_UNET_ONNX",
                             os.path.join(DATA_DIFFAE_DIR, "unet_step_amp16.onnx"))
DIFFAE_UNET_ONNX_DATA = _env("RP1_DIFFAE_UNET_ONNX_DATA",
                             os.path.join(DATA_DIFFAE_DIR, "unet_step_amp16.onnx.data"))
DIFFAE_TRT_ENGINE     = _env("RP1_DIFFAE_TRT_ENGINE",
                             os.path.join(DATA_DIFFAE_DIR, "unet_step_fp16.trt"))

DIFFAE_REPO_DIR = _env("RP1_DIFFAE_REPO_DIR", os.path.join(HERE, "diffae_repo"))

# ─── Outputs (kept in the repo, small) ─────────────────────────────────────────
CKPT_DIR_DIFFAE    = os.path.join(HERE, "checkpoints")
RESULTS_DIR_DIFFAE = os.path.join(HERE, "results")
FIG_DIR_DIFFAE     = os.path.join(HERE, "figures")
DDPM_CKPT_DIFFAE   = os.path.join(CKPT_DIR_DIFFAE, "latent_ddpm.pt")

# ─── Subspace geometry ─────────────────────────────────────────────────────────
DFULL = 512    # full DiffAE latent dimension
PROJ_K = 190   # dsub -- top-190 PCA subspace (paper: ~90.1% variance)
MIN_VARIANCE_RETAINED = 0.90   # paper claims 90.1%; step1 warns if not met
NN_N_SAMPLE = 4000   # random subsample size for the mean-NN-distance estimate
NN_CHUNK    = 500    # chunk size for the pairwise-distance computation
                     # (memory-safe for 27,000 points; see mmsold_top190.py)

# ─── Common (c_delta, c_sigma, M) grid -- MM-SOLD, sigma-CFDM, Plug-in BAOAB,
#     SG-BAOAB ───────────────────────────────────────────────────
CDELTA_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
CSIGMA_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
M_GRID      = [2, 32]

# ─── Score estimator: KNN+remainder everywhere (N=27,000 makes full-softmax
#     infeasible), K=L differs by method family ─────────────────
KNN_K_SMOOTH = 50    # MM-SOLD, sigma-CFDM
KNN_L_SMOOTH = 50
KNN_K_BAOAB  = 200   # Plug-in BAOAB, SG-BAOAB
KNN_L_BAOAB  = 200

# ─── Timing representative cell (mirrors handwritten/common.py's convention: one cheap,
#     small-(c_delta,c_sigma,M) cell, NOT necessarily a literal grid member) ──
TIMING_CDELTA   = 0.10
TIMING_CSIGMA   = 0.10
TIMING_M        = 2
TIMING_N_WARMUP = 1
TIMING_N_TIMED  = 5

# ─── MM-SOLD fixed hyperparameters ─────────────────────────────
SOLD_NSTEPS    = 1000     # LM steps
SOLD_H_COEF    = 0.2      # h = SOLD_H_COEF * delta^2
SOLD_DISCRETIZ = "LM"

# ─── sigma-CFDM fixed hyperparameters ─────────────────────────────────────────
CFDM_NSTEPS = 1000    # Euler steps; t: 0.001 -> 0.999 automatically
CFDM_BATCH  = 500

# ─── Plug-in BAOAB / SG-BAOAB shared BAOAB sampler settings ─
BAOAB_NSTEPS    = 1000        # sampling steps (independent of Lion's L)
BAOAB_SPLITTING = "BAOAB"
BAOAB_SCORE     = "knn"       # KNN+remainder score estimator throughout

# SG-BAOAB Lion optimisation settings.
LION_N_CHAINS   = 1000   # P (BAOAB chains per Lion iteration)
LION_N_STEPS    = 100    # L (BAOAB steps per outer Lion iteration)
LION_N_ITERS    = 300    # K (Lion steps)
LION_LR_REL     = 1.0
LION_B1         = 0.9
LION_B2         = 0.99
LION_WARM_START = True

# ─── KNN-PFlow grid ────────────────────────────────────────
KNNPF_CMAX_GRID   = [1, 2, 4]
KNNPF_CMIN_GRID   = [0.025, 0.05, 0.10]
KNNPF_K_GRID      = [50, 100, 250]
KNNPF_SOLVER_GRID = ["euler", "heun"]
KNNPF_NSTEPS      = 1000

# KNN-PFlow timing configuration
KNNPF_TIMING_CMAX   = 1
KNNPF_TIMING_CMIN   = 0.10
KNNPF_TIMING_K      = 50
KNNPF_TIMING_SOLVER = "euler"
KNNPF_TIMING_NSTEPS = 1000

# ─── SWF grid ──────────────────────────────────────────────
SWF_NSWF_GRID = [300, 1000]
SWF_R_GRID    = [64, 128, 256]
SWF_ETA_GRID  = [0.05, 0.10]
SWF_INIT_GRID = ["gaussian_full", "noised_empirical"]

# SWF timing configuration
SWF_TIMING_NSWF = 300
SWF_TIMING_R    = 64
SWF_TIMING_ETA  = 0.05
SWF_TIMING_INIT = "gaussian_full"

# ─── Latent DDPM ──────────────────
DDPM_NUM_HID_CHANNELS = 2048
DDPM_NUM_LAYERS       = 10
DDPM_SKIP_LAYERS      = list(range(1, 10))
DDPM_T_STEPS          = 1000
DDPM_BETA_CONST       = 0.008
DDPM_LR               = 1e-4
DDPM_WD               = 0.01
DDPM_BATCH            = 256
DDPM_EMA_DECAY        = 0.9999
DDPM_TOTAL_SAMPLES    = 101_000_000   # training budget (samples, not epochs)
DDPM_CKPT_EVERY_STEPS = 50_000
DDIM_STEPS            = 20

# ─── Decoding (all methods) -- fixed pretrained DiffAE decoder, 20-step
#     DPM-Solver++(2M) on a uniform log-SNR grid, TensorRT fp16 engine ─────────
DECODE_NSTEPS = 20
DECODE_BATCH  = 16

# ─── Evaluation protocol ───────────────────────────────────────────────────────
N_GENERATE = 3000    # samples generated per hyperparameter cell per repeat
N_REPEATS  = 3       # independent repeats (fresh RNG), NOT bootstrap resampling

KID_DEGREE    = 3
KID_SUBSET    = 100
KID_N_SUBSETS = 50
K_RECALL      = 3
DUP_PERCENTILE = 1   # 1st-percentile within-train NN distance (top-190 space)

# ─── Latent-space KID kernel ────────────────────────────────────────────────
RBF_BW_N_SAMPLE = 2000
RBF_BW_SEED     = 123   # matches latent_kid_raw.py's fixed seed
RBF_MIX_SCALES  = (0.25, 0.5, 1.0, 2.0, 4.0)
RBF_KID_SUBSET    = 1000   # matches latent_kid_raw.py's convention (bigger
RBF_KID_N_SUBSETS = 100    # subsets than the poly3/image-space default, since
                          # this is the metric grid search now actually relies on

REP_CELL_DUPRATE_CAP  = 0.1
REP_CELL_RECALL_FLOOR = 0.7

# ─── sys.path registration ─────────────────────────────────────────────────────
RELEASE_ROOT   = _env("RP1_RELEASE_ROOT",
                      os.path.abspath(os.path.join(HERE, "..", "..", "..", "release")))
RELEASE_COMMON = os.path.join(RELEASE_ROOT, "common")

for _p in (AUG_DIR, HW_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if RELEASE_COMMON not in sys.path:
    sys.path.insert(0, RELEASE_COMMON)


def describe():
    rows = [
        ("HERE", HERE), ("HW_DIR", HW_DIR), ("AUG_DIR", AUG_DIR),
        ("RELEASE_ROOT", RELEASE_ROOT), ("RELEASE_COMMON", RELEASE_COMMON),
        ("DATA_DIFFAE_DIR", DATA_DIFFAE_DIR),
        ("DIFFAE_ZSEM_TRAIN", DIFFAE_ZSEM_TRAIN),
        ("DIFFAE_ZSEM_VALID", DIFFAE_ZSEM_VALID),
        ("DIFFAE_FEATURES_NPZ", DIFFAE_FEATURES_NPZ),
        ("DIFFAE_UNET_ONNX", DIFFAE_UNET_ONNX),
        ("DIFFAE_UNET_ONNX_DATA", DIFFAE_UNET_ONNX_DATA),
        ("DIFFAE_TRT_ENGINE", DIFFAE_TRT_ENGINE),
        ("DIFFAE_REPO_DIR", DIFFAE_REPO_DIR),
        ("CKPT_DIR_DIFFAE", CKPT_DIR_DIFFAE),
        ("RESULTS_DIR_DIFFAE", RESULTS_DIR_DIFFAE),
        ("FIG_DIR_DIFFAE", FIG_DIR_DIFFAE),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        flag = "" if os.path.exists(v) else "  [missing]"
        print(f"  {k:<{width}}  {v}{flag}")


if __name__ == "__main__":
    print("Resolved config paths:\n")
    describe()
