# Training-Free Generative Sampling via Moment-Matched Score Smoothing

Official code for the paper **"Training-Free Generative Sampling via Moment-Matched Score Smoothing"**.

## Overview

We propose **MM-SOLD** (Moment-Matched Score-Smoothed Overdamped Langevin Dynamics), a training-free interacting particle sampler for generative modeling. MM-SOLD smooths the empirical score of the training data and enforces the particle mean and covariance to match those of the training set at every Langevin iteration. The method requires no neural network training and runs on CPUs, while producing samples with fidelity and diversity competitive with neural diffusion baselines.

## Directory Structure

```
MM-SOLD Code/
├── 2d_circular/                   # Fig. 1: density plots on the 2D unit circle
│   └── run_2d_circular.py
├── low_dimension/                 # Fig. 2: MM-SOLD vs σ-CFDM on 2D distributions
│   └── run_experiment.py
├── high_dimension/
│   ├── handwritten/               # Table 2 & Figs. 3, 5, 6: digit-8 generation
│   │   ├── step1_prepare_data.py
│   │   ├── step2_train_ddpm.py
│   │   ├── step3_run_experiment.py
│   │   ├── step3_run_experiment_knn.py
│   │   └── step4_plot_results.py
│   └── celebahq256/               # Table 2 & Figs. 4, 7, 8: CelebA-HQ generation
│       ├── step0_prepare_subset.py
│       ├── step1_train_nrae.py
│       ├── step1b_encode_all.py
│       ├── step2_train_ddpm.py
│       ├── step3_run_experiment_knn.py
│       └── step4_plot_results_knn.py
├── ECM_classifier/              # Table 1: minimum-ECM classifier
│   ├── run_classifier.py
│   └── config.py
├── augmentation_handwritten/      # Shared utilities: NRAE, sampling, whitening
│   ├── sampling_algo.py           # Core MM-SOLD and LDS sampling algorithm
│   ├── whitening_utils.py
│   ├── nrae_model.py
│   ├── data_utils.py
│   └── ...
├── ablation_studies/              # Fig. 9, 10: ablation studies
    ├── Langevin_steps/
    └── particle_nums/
```

## Requirements

All experiments use **Python 3.10+** and the following packages. All code is implemented in **JAX**.

```
numpy>=1.24
scipy>=1.11
jax>=0.4.30
jaxlib>=0.4.30
flax>=0.8.0
optax>=0.2.0
matplotlib>=3.7
Pillow>=10.0
scikit-learn>=1.3
```

Install all requirements:

```bash
pip install -r requirements.txt
```

For GPU support, replace `jax` and `jaxlib` with the appropriate CUDA build, e.g.:

```bash
pip install -U "jax[cuda12]"
```

## Dataset Download

Please download the original datasets from the links below.

### Handwritten Digits Dataset

Used in Sections 4.1 (classification), 4.2 (generation), and Appendix E.1–E.2.

- **Source**: Beaulac & Rosenthal (2022), *Introducing a new high-resolution handwritten digits data set with writer characteristics*, SN Computer Science.
- **Download**: https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or

After downloading, place the raw images (500×500 grayscale PNG files organised by digit class) in a directory, then run `step1_prepare_data.py` (in `high_dimension/handwritten/` or `augmentation_handwritten/`) with the path to that directory.

### CelebA-HQ (256×256)

Used in Section 4.3 and Appendix E.3.

- **Source**: Karras et al. (2017), *Progressive Growing of GANs for Improved Quality, Stability, and Variation*.
- **Download**: https://github.com/tkarras/progressive_growing_of_gans (see the *Preparing datasets for training* section) or via the Kaggle mirror: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256

After downloading the 256×256 PNG images, place training images in one folder and test images in another, then run `step0_prepare_subset.py` in `high_dimension/celebahq256/`.

## Running the Experiments

### 1. Figure 1: 2D Circular Density Plots

```bash
cd 2d_circular
python run_2d_circular.py
# Outputs: figures/mm_sigma*.png, figures/base_sigma*.png
```

### 2. Figure 2: Low-Dimensional 2D Experiments (MM-SOLD vs σ-CFDM)

```bash
cd low_dimension
python run_experiment.py
# Outputs: figures/{checkerboard,spirals}/ (scatter plots and heatmaps)
```

### 3. Table 1: Handwritten Digit Classification (Minimum-ECM Classifier)

First prepare data and train the NRAE (steps 1–3 in `augmentation_handwritten/`), then:

```bash
cd ECM_classifier
python run_classifier.py
```

### 4. Table 2 & Figures 3, 5, 6: Handwritten Digit-8 Generation

```bash
cd high_dimension/handwritten

# Step 1: prepare and preprocess data
python step1_prepare_data.py --data_dir /path/to/handwritten_digits

# Step 2: train latent DDPM baseline
python step2_train_ddpm.py

# Step 3: run grid search (MM-SOLD, σ-CFDM, DDPM)
python step3_run_experiment.py
# or with KNN score estimator:
python step3_run_experiment_knn.py

# Step 4: generate result figures and heatmaps
python step4_plot_results.py
```

### 5. Table 2 & Figures 4, 7, 8: CelebA-HQ Generation

```bash
cd high_dimension/celebahq256

# Step 0: prepare image subset
python step0_prepare_subset.py --train_dir /path/to/celebahq/train \
                               --test_dir  /path/to/celebahq/test

# Step 1: train NRAE and encode all images
python step1_train_nrae.py
python step1b_encode_all.py

# Step 2: train latent DDPM baseline
python step2_train_ddpm.py

# Step 3: run grid search with KNN score estimator
python step3_run_experiment_knn.py

# Step 4: generate result figures and heatmaps
python step4_plot_results_knn.py
```

### 6. Figures 9 & 10: Ablation Studies

```bash
# Figure 9: effect of Langevin step size and number of steps
cd ablation_studies/Langevin_steps
python run_ablation.py

# Figure 10: effect of particle count and training-set size
cd ablation_studies/particle_nums
python run_particle_nums.py
```

## Key Algorithm

The core MM-SOLD sampler is implemented in [`augmentation_handwritten/sampling_algo.py`](augmentation_handwritten/sampling_algo.py). It implements:

- `gradU_LDS_mc_stationary`: LDS-smoothed gradient via antithetic Monte Carlo estimation (Eq. 8 in the paper).
- `sample_class_overdamped_manifold`: full MM-SOLD loop using the Leimkuhler–Matthews discretization on the centered Stiefel manifold (Algorithm 1 in the paper).

The nearest-neighbor score estimator for high-dimensional settings (Algorithm 2 in the paper) is implemented in [`high_dimension/handwritten/cfdm_sampling.py`](high_dimension/handwritten/cfdm_sampling.py) and [`high_dimension/celebahq256/step3_run_experiment_knn.py`](high_dimension/celebahq256/step3_run_experiment_knn.py).

