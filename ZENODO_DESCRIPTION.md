# TrimerNet-WBC — Trained Models, Synthetic Images, and Dataset Manifests

This record contains the trained model weights, diffusion (DDPM) checkpoints,
DDPM-generated synthetic images, and dataset split manifests accompanying the paper:

**"TrimerNet-WBC: A Hybrid Vision Transformer for White Blood Cell Classification."**

The source code is available on GitHub: `https://github.com/<username>/TrimerNet-WBC`

> **Note on real datasets.** The original white blood cell images (Raabin-WBC, PBC, LISC)
> are **not** redistributed here. They must be obtained from their original sources
> (see the GitHub README and the dataset citations below). The `manifests/` provided in
> this record let you reconstruct the exact train/validation/test splits used in the study
> from those original images, ensuring full reproducibility without redistributing the data.

---

## Contents

This record is organized into the following items. Models are grouped by experiment and
by dataset, where `<dataset>` is one of: **Raabin**, **Raabin_Diffusion** (class-balanced
with synthetic minority samples), **PBC**, or **LISC**.

### 1. `manifests/`
Per-dataset CSV files specifying, for every image, its class and split assignment
(`train` / `validation` / `test`). Use these together with the original datasets to
reproduce the exact partitions used in the paper. Synthetic images (see item 2) are
assigned to the **training** split of the `Raabin_Diffusion` dataset only.

### 2. `generated_image_data/`
DDPM-generated synthetic images used to balance minority classes in the
`Raabin_Diffusion` dataset. Generated via DDIM sampling (50 steps, saturation 1.4):
- `Basophil/` — 549 synthetic images (from the epoch-250 diffusion checkpoint)
- `Monocyte/` — 55 synthetic images (from the epoch-100 diffusion checkpoint)

These images were added to the **training set only**; validation and test sets contain
real images exclusively.

### 3. Diffusion (DDPM) model checkpoints
Trained denoising diffusion models used to generate the synthetic images above. Stored in
Hugging Face `diffusers` format (`diffusion_pytorch_model.safetensors` + `config.json`):
- `Basophil_diffusion_model/model_epoch_0250/`
- `Monocyte_diffusion_model/model_epoch_0100/`

### 4. Teacher-ensemble weights — `ensemble_<dataset>/`
Each ensemble folder contains **four** files:
- `DeiT_best_model.pth`, `ViT_best_model.pth`, `Swin_best_model.pth`
  — the three individual fine-tuned backbones.
- `blood_cell_classification_ensemble.pth`
  — a single bundled checkpoint containing all three backbones together with the
    ensemble weights and class labels (as produced by the experiment notebooks).

The three separate files are used by `evaluate_ensemble.py`; the bundle is a convenience
file for loading the complete ensemble from a single checkpoint.

### 5. Student-model weights — `student_<dataset>/`
The distilled DeiT-Small student model (knowledge distillation, T = 4.0, α = 0.4):
- `student_deit_small_<dataset>_best.pth`

### 6. ResNet50 baseline weights — `resnet50_<dataset>/`
The ResNet50 CNN baseline:
- `resnet50_<dataset>_best.pth`

### 7. Non-distilled ablation weights — `vanilla_deit_small_<dataset>/`
A DeiT-Small trained **without** knowledge distillation (ablation baseline):
- `vanilla_deit_small_<dataset>_best.pth`

> **Cross-validation models are not included.** The 5-fold and 10-fold cross-validation
> experiments produce evaluation metrics (per-fold accuracies) rather than deployable
> models, and the fold splits are generated deterministically by the scripts. These
> experiments are fully reproducible via `cv10_significance.py` and
> `cross_validation_trimernet.py` (fixed random seed); per-fold accuracies are provided
> in the GitHub `results/` directory.

---

## Usage

All checkpoints are PyTorch `state_dict` files unless noted otherwise. The model class
definitions are in the GitHub repository. Below, `NUM_CLASSES = 5`
(Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil).

### Reconstructing a dataset split (manifests)
1. Download the original dataset (Raabin-WBC / PBC / LISC) from its source.
2. Use the corresponding manifest CSV to place each image into `train/`, `validation/`,
   or `test/` under its class subfolder.
3. For `Raabin_Diffusion`, add the images from `generated_image_data/` to the
   `train/` split of the Basophil and Monocyte classes.

### Loading the teacher ensemble (three separate files)
This is the format expected by `evaluate_ensemble.py`. Point `--model_dir` at the
extracted `ensemble_<dataset>/` folder:

```bash
python src/evaluate_ensemble.py \
    --data_root path/to/<dataset>/test \
    --model_dir path/to/ensemble_<dataset> \
    --weights <DeiT_w> <ViT_w> <Swin_w>
```
The three `--weights` values (in **DeiT ViT Swin** order) are the validation-derived
ensemble weights reported for each dataset; they are also printed by `train_ensemble.py`.

### Loading the bundled ensemble checkpoint
The single-file bundle contains everything needed to run the ensemble:

```python
import torch

bundle = torch.load("blood_cell_classification_ensemble.pth", map_location="cpu")
# bundle is a dict with:
#   bundle["models"]        -> list of three state_dicts (DeiT, ViT, Swin)
#   bundle["model_names"]   -> ["DeiT", "ViT", "Swin"]
#   bundle["class_labels"]  -> class-index mapping
#   bundle["model_weights"] -> the three ensemble weights
```
Instantiate the three model classes from the GitHub repo and load each
`state_dict` from `bundle["models"]`, then combine their softmax outputs using
`bundle["model_weights"]`.

### Loading the student / ResNet50 / vanilla models
These are single-model `state_dict` files. For example, the distilled student:

```python
import torch
from src.knowledge_distillation_trimernet import EnhancedDeiTSmallModel  # see repo

model = EnhancedDeiTSmallModel(num_classes=5)
model.load_state_dict(
    torch.load("student_deit_small_<dataset>_best.pth", map_location="cpu"))
model.eval()
```
The ResNet50 (`resnet50_<dataset>_best.pth`) and vanilla DeiT-Small
(`vanilla_deit_small_<dataset>_best.pth`) checkpoints load the same way using their
respective model definitions from the repository.

### Loading a diffusion (DDPM) checkpoint and generating images
The diffusion checkpoints are in `diffusers` format and load with `from_pretrained`:

```python
from diffusers import UNet2DModel
model = UNet2DModel.from_pretrained("Basophil_diffusion_model/model_epoch_0250")
```
Or simply use the provided generation script:

```bash
python diffusion/generate_ddpm_wbc.py --class_name Basophil --epoch 250 \
    --n_images 549 --num_steps 50 --saturation 1.4
```

---

## Dataset Citations

If you use the datasets, please cite the original sources:

- **Raabin-WBC** — Kouzehkanan, Z.M., Saghari, S., Tavakoli, S. et al. *A large dataset of
  white blood cells containing cell locations and types, along with segmented nuclei and
  cytoplasm.* Scientific Reports 12, 1123 (2022).
  https://doi.org/10.1038/s41598-021-04426-x
  (See also Tavakoli, S., Ghaffari, A., Kouzehkanan, Z.M. et al., Sci Rep 11, 19428 (2021),
  https://doi.org/10.1038/s41598-021-98599-0). Available from https://raabindata.com/free-data/

- **PBC** — Acevedo, A., Merino, A., Alférez, S., Molina, Á., Boldú, L., Rodellar, J. (2020).
  *A dataset for microscopic peripheral blood cell images for development of automatic
  recognition systems.* Mendeley Data, V1. https://doi.org/10.17632/snkd93bnjr.1

- **LISC** — Rezatofighi, S.H., Soltanian-Zadeh, H. (2011). *Automatic recognition of five
  types of white blood cells in peripheral blood.* Computerized Medical Imaging and
  Graphics, 35(4), 333–343. https://doi.org/10.1016/j.compmedimag.2011.01.003

## License

The trained models, synthetic images, and manifests in this record are released under
[CC BY 4.0]. The accompanying source code is released under the MIT License (see the
GitHub repository).
