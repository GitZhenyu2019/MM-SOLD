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
│       ├── step9_metric_floors.py                         # optional reference floors (not in the paper)
│       ├── step10_median_rbf_mmd.py                       # generation table's MMD (image/ArcFace) rows
│       ├── step11_table6_median_rbf_mmd.py                # moment-validation table's MMD rows
│       └── metrics_celeba.py, dpm_decode.py, build_fp16_engine.py
├── bayes_classifier/               # minimum-ECM classifier for handwritten-digit classification
│   ├── config.py, common.py
│   ├── step1_prepare_latents.py
│   ├── step2_run_ecm.py                       # run twice: --estimator plugin, --estimator dual
│   ├── step3_run_mlp.py                       # run twice: --mode baseline, --mode augmented
│   ├── step4_summarize_table2.py              # assembles the classification results table
│   └── run_table2.sh                          # runs steps 1-4 with the paper's settings
├── augmentation_handwritten/       # Shared preprocessing/model code + the NRAE encoder recipe
│   ├── data_utils.py, nrae_model.py, mlp_model.py, classifier_model.py
│   ├── whitening_utils.py, sampling_algo.py
│   ├── vae_model.py                # NOT a separate experiment here -- nrae_model.py imports its
│   │                                # ResBlock/Downsample/Upsample CNN primitives directly
│   ├── train_nrae.py               # NRAE of the main digit experiments (validation-loss checkpoint)
│   └── step1_prepare_data.py, step2_train_classifier_baseline.py,
│       step3b_train_nrae.py, step3b_test_nrae.py, step3c_train_classifier_nrae_reconstructed.py,
│       step4b_generate_augmented_nrae.py, step5_train_classifier_augmented.py,
│       step6b_encode_latents_nrae.py, step7b_latent_baseline_nrae.py, step8b_latent_augmented_nrae.py
└── ablation_studies/               # sensitivity studies
    ├── Langevin_steps/run_ablation.py         # step-size / iteration-count sensitivity
    └── particle_nums/run_particle_nums.py     # particle count / training-set size sensitivity
```

**Note on `augmentation_handwritten/sampling_algo.py` vs `common/sampling_algo.py`:** these are two
*different* files with the same name. `common/sampling_algo.py` is the no-whitening sampler
used by `bayes_classifier/`, `high_dimension/handwritten/`, and `high_dimension/celebahq256/`.
`augmentation_handwritten/sampling_algo.py` is a whitening-based variant used by
`augmentation_handwritten/` and `ablation_studies/particle_nums/`. Keep both, in their own folders — do
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

`low_dimension/run_experiment.py` only needs NumPy, JAX and Matplotlib for the checkerboard/spirals
experiments. Its optional 3D point-cloud demos require external meshes and, optionally, `polyscope`
and `point_cloud_utils`; they are not part of the paper.

`high_dimension/celebahq256/step3_train_ddpm.py` needs the DiffAE network-architecture code (not
weights) from https://github.com/phizaz/diffae — clone it and point `RP1_DIFFAE_REPO_DIR` (or the
`config.py` default `DIFFAE_REPO_DIR`) at the checkout; the script raises a clear error naming the
expected path if it's missing.

## Path configuration (read this before running anything)

Several modules resolve the `common/` library's location using a relative path from their own file
location (e.g. `high_dimension/handwritten/config.py` looks 3 directories up for a `common/`
sibling). If you place the folders differently, set the locations explicitly:

- `RP1_RELEASE_ROOT` — the folder that **contains** `common/` (the repository root)
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
- After downloading, build the class-balanced 1,000/58/300 split (seed 0) used by every digit
  experiment, then train the NRAE and the CNN feature extractor on it:
  ```bash
  cd augmentation_handwritten
  python step1_prepare_data.py --images_path <raw>/handwritten_img.npy \
      --labels_path <raw>/handwritten_label.npy \
      --n_train_per_class 1000 --n_val_per_class 58 --n_test_per_class 300
  python train_nrae.py --data_dir ./data --ckpt_dir ./checkpoints_final   # NRAE of the main experiments
  python step2_train_classifier_baseline.py --data_dir ./data              # CNN feature extractor (KID/Recall)
  python step3b_train_nrae.py --data_dir ./data                            # earlier NRAE, particle-number ablation only
  ```
  Set `RP1_NRAE_CKPT=<...>/augmentation_handwritten/checkpoints_final/nrae_best.pkl` for
  `high_dimension/handwritten/` and `bayes_classifier/`. `train_nrae.py` keeps the checkpoint with the
  lowest validation loss; `step3b_train_nrae.py` keeps the lowest training loss and writes
  `checkpoints/nrae_best.pkl`, which `ablation_studies/particle_nums/` loads. The CNN feature extractor is
  written to `checkpoints/classifier_baseline_best.pkl` (the default of `RP1_CLF_CKPT`).

### CelebA-HQ (256×256)
- **Source**: Karras et al. (2017), *Progressive Growing of GANs*.
  https://github.com/tkarras/progressive_growing_of_gans (or the Kaggle mirror
  `badasstechie/celebahq-resized-256x256`).
- Images are encoded into 512-dim latents by a **fixed, pretrained Diffusion Autoencoder (DiffAE)**
  — this encoder is not part of this release; obtain the precomputed 512-dim train/valid latent
  arrays (`RP1_DIFFAE_ZSEM_TRAIN` / `RP1_DIFFAE_ZSEM_VALID`), the reference Inception features
  (`RP1_DIFFAE_FEATURES_NPZ`), and the DiffAE decoder's ONNX export (`RP1_DIFFAE_UNET_ONNX(+_DATA)`)
  computed with the official DiffAE release (https://github.com/phizaz/diffae).
- Before any decode step, build the TensorRT engine once per GPU architecture:
  `python high_dimension/celebahq256/build_fp16_engine.py`.
- ArcFace metrics ("FID (ArcFace)"/"KID (ArcFace, ×10³)"/"MMD (ArcFace)" rows in both results tables, and the
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
python run_experiment.py                 # samplers + paper panels (figures_paper_2d/)
python plot_2d_paper_figures.py          # optional: re-plot the cached results
```
Defaults: 500 training points and a 5,000-point reference set (farthest-point selections from
10x larger candidate pools, seeds 1 and 0), P = 5,000 particles, 3,000 steps of size 5e-4,
GMM component standard deviation delta = 0.1, M in {2, 32} and smoothing bandwidths
sigma in {0.05, 0.1, 0.15, 0.4}. Use `--help` for all options.

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
bash run_table2.sh        # -> the classification results table (results_table2/)
```
The script runs every step with the settings of the paper: the full 100-dimensional NRAE latents
(`--proj_k 100`), the plug-in classifier's validation grid search, the dual classifier at
(c_delta, c_sigma) = (0.35, 0.25) with seed 7, the two MLP baselines (the augmented one reuses the dual
cell), and the summary table.

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
python step9_metric_floors.py --celeba_dir <path>     # -> optional reference floors (not reported in the paper)
python step10_median_rbf_mmd.py                       # -> generation table's MMD (image) / MMD (ArcFace) rows
python step11_table6_median_rbf_mmd.py                # -> moment-validation table's MMD rows
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
