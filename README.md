# TrimerNet-WBC

Official implementation of "TrimerNet-WBC: A Hybrid Vision Transformer for White Blood Cell Classification"

TrimerNet-WBC is a validation-weighted ensemble of three transformer architectures; Vision Transformer (ViT-Base), Data-efficient Image Transformer (DeiT-Base), and Swin Transformer (Swin-Base) for the classification of five white blood cell (WBC) types: Basophil, Eosinophil, Lymphocyte, Monocyte, and Neutrophil. The high capacity ensemble (teacher, 260.7M parameters) is compressed via knowledge distillation into a compact DeiT-Small student (21.9M parameters), reducing the parameter count by 11.9× and inference time by 36.9× while maintaining comparable or higher accuracy, making it suitable for clinical deployment.

## Key Features

- **Transformer ensemble** (ViT + DeiT + Swin) with validation-accuracy-based weighting.
- **Knowledge distillation** into a lightweight DeiT-Small student (T = 4.0, α = 0.4).
- **Diffusion-based augmentation** (DDPM training + DDIM sampling) for minority-class balancing, applied to the **training set only**.
- **Quantitative interpretability** via multi-layer Grad-CAM and Intersection-over-Union (IoU) against ground-truth masks.
- **Statistical significance analysis** via 10-fold cross-validation (Friedman, Wilcoxon, and paired *t*-tests).

## Repository Structure

```
.
├── src/                                   # Core pipeline: training, evaluation, CV
│   ├── train_ensemble.py                  # Train the ViT+DeiT+Swin teacher ensemble
│   ├── evaluate_ensemble.py               # Validation-weighted ensemble evaluation
│   ├── knowledge_distillation_trimernet.py# Distill teacher -> DeiT-Small student
│   ├── cross_validation_trimernet.py      # 5-fold CV (student stability)
│   ├── cv10_significance.py               # 10-fold CV (per-fold accuracies for significance)
│   ├── resnet50_baseline.py               # ResNet50 CNN baseline
│   └── vanilla_deit_small.py              # Non-distilled DeiT-Small baseline (ablation)
│
├── utils/                                 # Measurement & analysis tools
│   ├── complexity_analysis.py             # Params / FLOPs / inference latency
│   ├── ensemble_kappa_mcc.py              # Teacher ensemble: Cohen's Kappa & MCC
│   ├── student_kappa_mcc_auc.py           # Student: Kappa, MCC & AUC
│   └── resnet50_roc_auc.py                # ResNet50 baseline: ROC curves & AUC
│
├── diffusion/                             # Diffusion-based synthetic data generation
│   ├── train_ddpm_wbc.py                  # Train a DDPM per minority class
│   └── generate_ddpm_wbc.py               # Generate synthetic images via DDIM sampling
│
├── notebooks/                             # Per-dataset end-to-end experiment notebooks
│   ├── deit_vit_swin_Raabin-WBC.ipynb     # Raabin-WBC (original, imbalanced)
│   ├── deit_vit_swin_Raabin_Diffusion.ipynb # Raabin-WBC (balanced with synthetics)
│   ├── deit_vit_swin_PBC.ipynb            # PBC
│   └── deit_vit_swin_LISC.ipynb           # LISC
│
├── results/                               # Curated result figures & metrics
│   ├── TrimerNet-WBC_results/             # Ensemble: AUC, Kappa/MCC, ROC, confusion
│   ├── student_results/                   # Student: metrics, ROC, confusion
│   ├── ResNet50_results/                  # ResNet50 baseline outputs
│   ├── vanilla_results/                   # Non-distilled DeiT-Small outputs
│   ├── 5-fold_CV_results/                 # 5-fold CV charts, ROC, confusion
│   └── 10-fold_CV_statistical_analyses/   # cv10_state.json + JASP significance reports
│
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note on large assets.** Real datasets, trained model weights (`.pth`, `.safetensors`), and DDPM-generated synthetic images are **not** stored in this repository. They are archived on Zenodo (see [Pretrained Models & Data](#pretrained-models--data)). Only source code and small result figures are version-controlled here.

## Installation

```bash
git clone https://github.com/ankc-lab/TrimerNet-WBC.git
cd TrimerNet-WBC
pip install -r requirements.txt
```

Developed and tested with **Python 3.10** and **PyTorch 2.5.1+** (CUDA 11.8/12.4). A CUDA-capable GPU is strongly recommended.

## Datasets

This study uses three publicly available WBC datasets. They are **not** redistributed here; please download them from their original sources and use manifest files:

| Dataset | Description | Source |
|---------|-------------|--------|
| **Raabin-WBC** | Double-labelled cropped WBC images | (https://raabindata.com/free-data/) Accessed on Nov 6,2025. Other alternative download link available here: https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset |
| **PBC** | Peripheral Blood Cells (Barcelona) | https://www.kaggle.com/datasets/kylewang1999/pbc-dataset|
| **LISC** | Leukocyte Images for Segmentation and Classification | (https://github.com/nimaadmed/WBC_Feature) |

Expected directory layout per dataset (one subfolder per class):

```
<DatasetRoot>/
├── train/
│   ├── Basophil/
│   ├── Eosinophil/
│   ├── Lymphocyte/
│   ├── Monocyte/
│   └── Neutrophil/
├── validation/   (same class subfolders)
└── test/         (same class subfolders)
```

For the balanced **Raabin_Diffusion** split, synthetic minority-class images are added to the `train/` set only (validation and test sets contain real images exclusively). Final per-class split: **700 train / 100 validation / 50 test**. Minority classes are augmented to reach the training target (Basophil: 151 real + 549 synthetic; Monocyte: 645 real + 55 synthetic).

**Preprocessing.** Images are converted to HSV; CLAHE is applied to the V channel (clip limit 3.0, tile grid 8×8); converted back to RGB; resized (384×384 for the teacher ensemble, 224×224 for the student and ResNet50); and normalized to [0, 1].

## Pretrained Models & Data

Trained weights (ViT, DeiT, Swin teachers; distilled DeiT-Small student; ResNet50 baseline), the trained DDPM checkpoints, and the generated synthetic images are archived on Zenodo:

**Zenodo DOI:** [`10.5281/zenodo.20938518`](https://doi.org/10.5281/zenodo.20938518) *(to be updated upon release)*

To ensure reproducibility and respect the licensing of the original datasets (Raabin-WBC, PBC, LISC), we provide the list of filenames used for training, validation, and testing in this study. The complete file manifests (filenames, labels, and subsets) can be found in the data/manifests directory. All datasets are available for download from their original sources given in the Datasets section).

## Usage

The end-to-end workflow proceeds: **(1)** generate synthetic minority-class data → **(2)** train the teacher ensemble → **(3)** distill into the student → **(4)** evaluate → **(5)** run baselines, complexity, and cross-validation. Per-dataset notebooks in `notebooks/` reproduce the full pipeline for each dataset; the scripts below expose the same steps from the command line.

### 1. Diffusion-based synthetic data (minority classes)

```bash
# Train a DDPM for a minority class (trained on real train/ images only)
python diffusion/train_ddpm_wbc.py --class_name Basophil --image_size 256 --epochs 250
python diffusion/train_ddpm_wbc.py --class_name Monocyte --image_size 256 --epochs 100

# Generate synthetic images via fast DDIM sampling
# (50 DDIM steps with mild saturation boost gave the best-composed cells)
python diffusion/generate_ddpm_wbc.py --class_name Basophil --epoch 250 \
    --n_images 549 --num_steps 30 --saturation 1.0 --contrast 1.5
python diffusion/generate_ddpm_wbc.py --class_name Monocyte --epoch 100 \
    --n_images 55  --num_steps 30 --saturation 1.0 --contrast 1.2
```

### 2. Train the teacher ensemble

```bash
python src/train_ensemble.py \
    --data_root path/to/<DatasetRoot> \
    --save_dir  path/to/weights \
    --epochs 50
```
This trains the three backbones, saves their best checkpoints (DeiT_best_model.pth, ViT_best_model.pth, Swin_best_model.pth) into --save_dir, and prints the validation-derived ensemble weights. The weights are printed in DeiT ViT Swin order — pass them to --weights in exactly that order.

### 3. Evaluate the ensemble

python src/evaluate_ensemble.py \
    --data_root path/to/<DatasetRoot>/test \
    --model_dir path/to/weights \
    --weights DeiT's weight ViT's weight Swin's weight

Ensemble weights are derived from each model's validation accuracy.

### NOTES: If you want to use jupyter notebook instead of python, you can also run the codes in the notebooks files for each datasets.

### 4. Knowledge distillation

```bash
USAGE (Raabin example):
    python src/knowledge_distillation_trimernet.py \
        --dataset Raabin \
        --train_path your_path/train \
        --val_path   your_path/validation \
        --test_path  your_path/test \
        --deit_ckpt  DeiT_best_model.pth \
        --vit_ckpt   ViT_best_model.pth \
        --swin_ckpt  Swin_best_model.pth \
        --temperature 4.0 --alpha 0.4 \
        --output_dir ./distilled_models
```

### 5. Baselines, complexity, and cross-validation

```bash
# ResNet50 CNN baseline
USAGE (Raabin example):
    python src/resnet50_baseline.py \
        --dataset Raabin \
        --train_path path/to/Raabin/train \
        --val_path   path/to/Raabin/validation \
        --test_path  path/to/Raabin/test \
        --output_dir ./resnet50_results

# Non-distilled DeiT-Small (ablation)
USAGE (Raabin example):
    python src/vanilla_deit_small.py \
        --dataset Raabin \
        --train_path path/to/Raabin/train \
        --val_path   path/to/Raabin/validation \
        --test_path  path/to/Raabin/test \
        --output_dir ./vanilla_deit

# Model complexity (no checkpoint required)
python utils/complexity_analysis.py

# 5-fold CV (student stability)
USAGE (Raabin example):
    python src/cross_validation.py \
        --dataset     Raabin \
        --train_path  your_path/Raabin/train \
        --val_path    your_path/Raabin/validation \
        --test_path   your_path/Raabin/test \
        --deit_ckpt   your_path/Raabin/DeiT_best_model.pth \
        --vit_ckpt    your_path/Raabin/ViT_best_model.pth \
        --swin_ckpt   your_path/Raabin/Swin_best_model.pth \
        --output_dir  your_path/Raabin/cv_results \
        --n_folds     5

# 10-fold CV (produces per-fold accuracies for significance testing)
python src/cv10_significance.py \
        --train_path /path/to/Raabin/train \
        --val_path   /path/to/Raabin/validation \
        --output_dir /path/to/cv10_results \
        --n_folds 10 --epochs 50 --patience 5 --batch_size 32 --lr 5e-5
```

### 6. Additional metrics (Kappa, MCC, AUC, ROC)

```bash
# Teacher ensemble: Cohen's Kappa & MCC
USAGE (Raabin example):
    python utils/ensemble_kappa_mcc.py \
        --dataset Raabin_Diffusion \
        --test_path  path/to/Raabin_Diffusion/test \
        --val_path   path/to/Raabin_Diffusion/validation \
        --deit_ckpt  path/to/DeiT_best_model.pth \
        --vit_ckpt   path/to/ViT_best_model.pth \
        --swin_ckpt  path/to/Swin_best_model.pth \
        --output_dir ./ensemble_metrics

# Student: Kappa, MCC & AUC
USAGE (Raabin example):
    python utils/student_kappa_mcc_auc.py \
        --dataset Raabin \
        --test_path path/to/Raabin/test \
        --ckpt path/to/student_deit_small_Raabin_best.pth \
        --output_dir ./student_metrics

# ResNet50 baseline: ROC & AUC
USAGE (Raabin example):
    python utils/resnet50_roc_auc.py \
        --dataset Raabin \
        --test_path  path/to/Raabin/test \
        --ckpt       path/to/resnet50_Raabin_best.pth \
        --output_dir ./resnet50_roc
```

## Statistical Significance Testing

`cv10_significance.py` runs a 10-fold cross-validation and writes per-fold accuracies for the three backbones and the ensemble to `cv10_state.json`. The statistical significance tests reported in the paper — the **Friedman test**, **Wilcoxon signed-rank test**, and **paired *t*-tests** — were performed in **JASP 0.97.1** using these per-fold accuracies. The corresponding JASP reports are included under `results/10-fold_CV_statistical_analyses/`.

## Interpretability

Model decisions are explained using **multi-layer Grad-CAM**. Class-discriminative attention maps are compared against ground-truth cell masks using Intersection-over-Union (IoU), achieving a mean IoU of **0.98 ± 0.04**, confirming that predictions are driven by the target leukocyte rather than background artifacts.

## Notes: 

You can update your dataset paths in the code to match your local setup. If you use the datasets in your study, you can cite the following studies. 

* Tavakoli, S., Ghaffari, A., Kouzehkanan, Z.M. et al. New segmentation and feature extraction algorithm for classification of white blood cells in peripheral smear images. Sci Rep 11, 19428 (2021). https://doi.org/10.1038/s41598-021-98599-0
* Kouzehkanan, Z.M., Saghari, S., Tavakoli, S. et al. A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm. Sci Rep 12, 1123 (2022). https://doi.org/10.1038/s41598-021-04426-x
* Acevedo, A., Alf´erez, S., Merino, A., Puigv´ı, L., Rodellar, J.: Recognition of peripheral blood cell images using convolutional neural networks. Computer Methods and Programs in Biomedicine 180, 105020 (2019) https://doi.org/10.1016/j.cmpb.2019.105020
* Rezatofighi, S.H., Soltanian-Zadeh, H.: Automatic recognition of five types of white blood cells in peripheral blood. Computerized Medical Imaging and Graphics 35(4), 333–343 (2011) https://doi.org/10.1016/j.compmedimag.2011.01.003

## Citation

If you use this code, please cite:NA


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
