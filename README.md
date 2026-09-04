# Training-Free Generative Sampling via Moment-Matched Score Smoothing

Official code for the paper **"Training-Free Generative Sampling via Moment-Matched Score Smoothing"**.

## Overview

We propose **MM-SOLD** (Moment-Matched Score-Smoothed Overdamped Langevin Dynamics), a training-free interacting particle sampler for generative modeling. MM-SOLD smooths the empirical score of the training data and enforces the particle mean and covariance to match those of the training set at every Langevin iteration. The method requires no neural network training and runs on CPUs, while producing samples with fidelity and diversity competitive with neural diffusion baselines.

This release contains the code needed to reproduce every result in the paper. 

## Directory Structure

```
MM-SOLD_code_release/
├── common/                        # Core MM-SOLD + BAOAB/tilting library, shared by
│   │                               # the handwritten, CelebA-HQ, and classifier experiments
│   ├── sampling_algo.py           # MM-SOLD manifold sampler
│   ├── tilting_lion.py            # Plug-in / SG-BAOAB tilting-parameter estimation
│   └── latent_utils.py            # Small sample mean/covariance helper
├── 2d_circular/                    # naive vs moment-matched density on the 2D circle
│   └── run_2d_circular.py
├── low_dimension/                  # MM-SOLD vs σ-CFDM on checkerboard/spirals distributions
│   ├── run_experiment.py
│   └── plot_2d_paper_figures.py
├── high_dimension/
│   ├── handwritten/               # digit-8 generation: results table, representative samples,
│   │   │                          # and additional sample grids / heatmaps / moment validation
│   │   ├── config.py, common.py
│   │   ├── step1_prepare_subspace.py
│   │   ├── step2a_run_mmsold.py ... step2f_run_swf.py     # 6 methods' grid search
│   │   ├── step3_run_ddpm.py                              # latent DDPM baseline
│   │   ├── step4_summarize_table1.py                      # assembles the generation results table
│   │   ├── step5_generate_figures.py                      # additional sample grids, heatmaps,
│   │   │                                                  # and posterior-moment-validation table
│   │   ├── step6_generate_figure3.py                      # main-text representative-sample figure
│   │   └── ddpm_model.py, cfdm_sampling.py, knn_pflow_sampling.py, swf_sampling.py, metrics.py
│   └── celebahq256/                # CelebA-HQ generation: results table, representative samples,
│       │                            # and additional sample grids / heatmaps / moment validation
│       ├── config.py, common.py
│       ├── step1_prepare_subspace.py
│       ├── step2a_run_mmsold.py ... step2f_run_sgbaoab.py
│       ├── step3_train_ddpm.py                            # latent DDPM baseline
│       ├── step4_select_and_decode.py                     # assembles the generation results table
│       ├── step5_generate_figure4.py                      # main-text representative-sample figure
│       ├── step6_additional_results.py                    # additional sample grids, heatmaps,
│       │                                                  # and posterior-moment-validation table
│       ├── step7_arcface_metrics.py                       # generation table's ArcFace FID/KID rows
│       ├── step8_table6_arcface.py                        # moment-validation table's ArcFace rows
│       ├── step9_metric_floors.py                         # the two reference floors
│       └── metrics_celeba.py, dpm_decode.py, build_fp16_engine.py
├── bayes_classifier/               # minimum-ECM classifier for handwritten-digit classification
│   ├── config.py, common.py
│   ├── step1_prepare_latents.py
│   ├── step2_run_ecm.py                       # run twice: --estimator plugin, --estimator dual
│   ├── step3_run_mlp.py                       # run twice: --mode baseline, --mode augmented
│   └── step4_summarize_table2.py              # assembles the classification results table
├── augmentation_handwritten/       # Shared preprocessing/model code + the NRAE encoder recipe
│   ├── data_utils.py, nrae_model.py, mlp_model.py, classifier_model.py
│   ├── whitening_utils.py, sampling_algo.py
│   ├── vae_model.py                # NOT a separate experiment here -- nrae_model.py imports its
│   │                                # ResBlock/Downsample/Upsample CNN primitives directly
│   └── step1_prepare_data.py, step2_train_classifier_baseline.py,
│       step3b_train_nrae.py, step3b_test_nrae.py, step3c_train_classifier_nrae_reconstructed.py,
│       step4b_generate_augmented_nrae.py, step5_train_classifier_augmented.py,
│       step6b_encode_latents_nrae.py, step7b_latent_baseline_nrae.py, step8b_latent_augmented_nrae.py
└── ablation_studies/               # sensitivity studies
    ├── Langevin_steps/run_ablation.py         # step-size / iteration-count sensitivity
    └── particle_nums/run_particle_nums.py     # particle count / training-set size sensitivity
```

**Note on `augmentation_handwritten/sampling_algo.py` vs `common/sampling_algo.py`:** these are two
*different* files with the same name. `common/sampling_algo.py` is the newer, no-whitening sampler
used by `bayes_classifier/`, `high_dimension/handwritten/`, and `high_dimension/celebahq256/`.
`augmentation_handwritten/sampling_algo.py` is an older, whitening-based variant used only within
`augmentation_handwritten/`'s own NRAE-augmentation pipeline. Keep both, in their own folders — do
not merge them. Similarly, `high_dimension/handwritten/config.py` and `high_dimension/celebahq256/config.py`
(and their `common.py` counterparts) are two different files that happen to share a filename — always
run each experiment from inside its own folder so Python resolves `import config` to the right one.

**Cross-folder imports are load-bearing, not accidental.** Several scripts add sibling folders to
`sys.path` at import time (see each `config.py`'s "sys.path registration" section):
`high_dimension/celebahq256/*.py` imports `cfdm_sampling`/`knn_pflow_sampling`/`swf_sampling` from
`high_dimension/handwritten/`; `high_dimension/handwritten/*.py` and `bayes_classifier/*.py` import
`data_utils`/`nrae_model`/`mlp_model`/`classifier_model`/`whitening_utils` from
`augmentation_handwritten/`; `2d_circular/run_2d_circular.py` imports `sampling_algo` from
`augmentation_handwritten/`; `ablation_studies/particle_nums/run_particle_nums.py` imports from both
`augmentation_handwritten/` and `high_dimension/handwritten/`. **Keep the directory layout above
intact** (don't flatten or relocate individual folders) — these relative-path lookups depend on it.

## Requirements

```
numpy, scipy, jax, jaxlib, flax, optax, matplotlib, Pillow, scikit-learn
torch, torchvision                     # CelebA-HQ: Inception-V3 feature extraction, DiffAE decode
cleanfid                                # InceptionV3 (pool3) feature extractor
tensorrt, onnx                          # CelebA-HQ: DiffAE decoder engine (build_fp16_engine.py)
insightface, opencv-python, onnxruntime-gpu   # CelebA-HQ: ArcFace FID/KID
```
For GPU JAX, replace `jax`/`jaxlib` with the CUDA build, e.g. `pip install -U "jax[cuda12]"`.
`onnxruntime-gpu` from the default PyPI index frequently resolves to a CUDA-13 build that silently
falls back to CPU on CUDA-12 machines — install a CUDA-12-matched build explicitly if that happens
(the ArcFace scripts print their active onnxruntime execution providers on startup so this is easy
to catch).

`low_dimension/run_experiment.py` additionally imports `ot` (POT), `polyscope`, and
`point_cloud_utils` for its optional 3D point-cloud demos (car/chair/fish) — these are **not** part
of the paper's reported checkerboard/spirals comparison; install
them only if you want to run the extra 3D demonstrations.

`high_dimension/celebahq256/step3_train_ddpm.py` needs the DiffAE network-architecture code (not
weights) from https://github.com/phizaz/diffae — clone it and point `RP1_DIFFAE_REPO_DIR` (or the
`config.py` default `DIFFAE_REPO_DIR`) at the checkout; the script raises a clear error naming the
expected path if it's missing.

## Path configuration (read this before running anything)

Several modules resolve the `common/` library's location using a relative path from their own file
location (e.g. `high_dimension/handwritten/config.py` looks 3 directories up for a `common/`
sibling). That default only happens to work if this release sits in the exact spot it was assembled
in; once you move or re-clone this folder anywhere else, set this explicitly instead of relying on
the relative default:

- `RP1_RELEASE_ROOT` — the folder that **contains** `common/` (i.e. `MM-SOLD_code_release/` itself)
- `RP1_NRAE_CKPT` — path to a trained NRAE checkpoint (see "Handwritten digits" below; this is a
  ~100-dim autoencoder trained on the handwritten-digits dataset, used by `high_dimension/handwritten/`
  and `bayes_classifier/`)
- `RP1_DIFFAE_DATA_ROOT` / `RP1_DIFFAE_ZSEM_TRAIN` / `RP1_DIFFAE_ZSEM_VALID` / `RP1_DIFFAE_FEATURES_NPZ`
  / `RP1_DIFFAE_UNET_ONNX` / `RP1_DIFFAE_UNET_ONNX_DATA` / `RP1_DIFFAE_TRT_ENGINE` — CelebA-HQ DiffAE
  latents, decoder ONNX export, and TensorRT engine path (see "CelebA-HQ" below)
- `RP1_DIFFAE_REPO_DIR` — local checkout of https://github.com/phizaz/diffae (code only)
- `RP1_RAW_IMG` / `RP1_RAW_LABEL` / `RP1_HANDWRITTEN_ROOT` — raw handwritten-digit images/labels
- `RP1_CLF_CKPT` — pretrained digit-classifier checkpoint used as a KID/Recall feature extractor

Every `config.py` file lists its own environment-variable names and defaults near the top — check
there first if a script complains about a missing path.

## Dataset and checkpoint prerequisites

### Handwritten digits
- **Source**: Beaulac & Rosenthal (2022), *Introducing a new high-resolution handwritten digits data
  set with writer characteristics*, SN Computer Science.
  https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or
- After downloading, run `augmentation_handwritten/step1_prepare_data.py` with `--raw_img`/`--raw_label`
  pointing at the raw files, then `augmentation_handwritten/step3b_train_nrae.py` to train the NRAE
  encoder used throughout (`high_dimension/handwritten/`, `bayes_classifier/`, and this folder's own
  classification pipeline all load this checkpoint via `RP1_NRAE_CKPT`).

### CelebA-HQ (256×256)
- **Source**: Karras et al. (2017), *Progressive Growing of GANs*.
  https://github.com/tkarras/progressive_growing_of_gans (or the Kaggle mirror
  `badasstechie/celebahq-resized-256x256`).
- Images are encoded into 512-dim latents by a **fixed, pretrained Diffusion Autoencoder (DiffAE)**
  — this encoder is not part of this release; obtain the precomputed 512-dim train/valid latent
  arrays (`RP1_DIFFAE_ZSEM_TRAIN` / `RP1_DIFFAE_ZSEM_VALID`), the reference Inception features
  (`RP1_DIFFAE_FEATURES_NPZ`), and the DiffAE decoder's ONNX export (`RP1_DIFFAE_UNET_ONNX(+_DATA)`)
  from wherever your lab stores the fixed DiffAE artifacts.
- Before any decode step, build the TensorRT engine once per GPU architecture:
  `python high_dimension/celebahq256/build_fp16_engine.py`.
- ArcFace metrics ("FID (ArcFace)"/"KID (ArcFace, ×10³)" rows in both results tables, and the
  reference floors in `step9_metric_floors.py`) need the raw CelebA-HQ pixels for the 3,000 held-out
  test images (and, for the floors, 3,000 random training images) — point `--celeba_dir` at a
  directory with `train/`/`valid/` subfolders of `img_<N>.png` files.

## Running the experiments

### 2D circular density plots
```bash
cd 2d_circular && python run_2d_circular.py
```

### Checkerboard & spirals: MM-SOLD vs σ-CFDM
```bash
cd low_dimension
python run_experiment.py
python plot_2d_paper_figures.py   # paper-formatted re-plot of the cached results
```

### Handwritten digit-8 generation
```bash
cd high_dimension/handwritten
python step1_prepare_subspace.py
python step2a_run_mmsold.py --overwrite
python step2b_run_cfdm.py --overwrite
python step2c_run_pluginbaoab.py --overwrite
python step2d_run_sgbaoab.py --overwrite
python step2e_run_knnpflow.py --overwrite
python step2f_run_swf.py --overwrite
python step3_run_ddpm.py
python step4_summarize_table1.py    # -> the generation results table
python step5_generate_figures.py    # -> additional sample grids, heatmaps, moment validation
python step6_generate_figure3.py    # -> main-text representative-sample figure
```

### Handwritten digit classification (ECM classifier)
```bash
cd bayes_classifier
python step1_prepare_latents.py --proj_k 40
python step2_run_ecm.py --estimator plugin --proj_k 40 --latents_dir <out>/latents --out <out>/ecm_plugin
python step2_run_ecm.py --estimator dual   --proj_k 40 --latents_dir <out>/latents --out <out>/ecm_dual
python step3_run_mlp.py --mode baseline  --proj_k 40 --latents_dir <out>/latents --out <out>/mlp_baseline
python step3_run_mlp.py --mode augmented --proj_k 40 --shared_cell <out>/ecm_dual/selected_cell.json \
                         --latents_dir <out>/latents --out <out>/mlp_augmented
python step4_summarize_table2.py --results_dir <out>    # -> the classification results table
```
Pass `--proj_k 40` explicitly as above (a top-40 PCA subspace).

### CelebA-HQ generation
```bash
cd high_dimension/celebahq256
python build_fp16_engine.py                    # once per GPU architecture
python step1_prepare_subspace.py
python step2a_run_mmsold.py --overwrite
python step2b_run_cfdm.py --overwrite
python step2c_run_knnpflow.py --overwrite
python step2d_run_swf.py --overwrite
python step2e_run_pluginbaoab.py --overwrite
python step2f_run_sgbaoab.py --overwrite
python step3_train_ddpm.py
python step4_select_and_decode.py               # -> the generation results table (image-space FID/KID)
python step5_generate_figure4.py                # -> main-text representative-sample figure
python step6_additional_results.py              # -> additional sample grids, heatmaps, moment validation
python step7_arcface_metrics.py --celeba_dir <path>   # -> generation table's ArcFace rows
python step8_table6_arcface.py                        # -> moment-validation table's ArcFace rows
python step9_metric_floors.py --celeba_dir <path>     # -> the two reference floors
```

### Ablation studies
```bash
cd ablation_studies/Langevin_steps && python run_ablation.py          # step-size / iteration-count sensitivity
cd ../particle_nums && python run_particle_nums.py                    # particle count / training-set size sensitivity
```

## Key algorithm

The core MM-SOLD sampler is implemented in [`common/sampling_algo.py`](common/sampling_algo.py).
Plug-in and SG-BAOAB tilting-parameter estimation is implemented in
[`common/tilting_lion.py`](common/tilting_lion.py). The nearest-neighbor-plus-remainder score
estimator is implemented independently inside each experiment's own `common.py`.
