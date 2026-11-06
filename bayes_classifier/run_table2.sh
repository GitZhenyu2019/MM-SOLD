#!/usr/bin/env bash
# Reproduces the handwritten-digit classification results table.
#  - all classifiers use the full 100-dimensional NRAE latents (--proj_k 100:
#    the PCA step keeps all 100 directions, i.e. an orthogonal change of basis);
#  - plug-in classifier: (c_delta, c_sigma) selected on the validation set
#    from {0.05,...,0.50}^2, shared by all classes;
#  - dual classifier: fitted once at the cell (0.35, 0.25) with 2,000 chains x
#    500 BAOAB steps x 300 Lion iterations (config.py), seed 7;
#  - MM-SOLD augmentation reuses the dual classifier's cell.
# Requires RP1_NRAE_CKPT (see README) and the raw data paths in config.py.
set -euo pipefail
cd "$(dirname "$0")"
export RP1_RELEASE_ROOT="${RP1_RELEASE_ROOT:-$(cd .. && pwd)}"

OUT="${OUT:-results_table2}"
DUAL_SEED=7

python step1_prepare_latents.py --proj_k 100 --out "$OUT/latents"

python step2_run_ecm.py --estimator plugin --proj_k 100 \
    --latents_dir "$OUT/latents" --out "$OUT/ecm_plugin"

mkdir -p "$OUT/ecm_dual_cell"
echo '{"c_delta": 0.35, "c_sigma": 0.25}' > "$OUT/ecm_dual_cell/selected_cell.json"
# autotune level 0 avoids a very slow XLA GEMM autotuning pass for this shape
XLA_FLAGS="--xla_gpu_autotune_level=0" \
python step2_run_ecm.py --estimator dual --proj_k 100 \
    --shared_cell "$OUT/ecm_dual_cell/selected_cell.json" --seed "$DUAL_SEED" \
    --latents_dir "$OUT/latents" --out "$OUT/ecm_dual"

python step3_run_mlp.py --mode baseline --proj_k 100 \
    --latents_dir "$OUT/latents" --out "$OUT/mlp_baseline"

python step3_run_mlp.py --mode augmented --proj_k 100 \
    --shared_cell "$OUT/ecm_dual/selected_cell.json" \
    --latents_dir "$OUT/latents" --out "$OUT/mlp_augmented"

python step4_summarize_table2.py --results_dir "$OUT"
