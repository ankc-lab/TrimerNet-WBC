"""
TrimerNet-WBC Knowledge Distillation
======================================
Teacher : EnhancedDeiT + EnhancedViT + EnhancedSwin (Weighted Average Ensemble)
          -- Trained HuggingFace based models --
Student : EnhancedDeiT-Small (224x224, ImageNet pre-trained, ~4x smaller)

Datasets: Raabin / Raabin_Diffusion / PBC / LISC
GPU     : RTX 4090 16GB

USAGE (Raabin example):
    python knowledge_distillation_trimernet.py \
        --dataset Raabin \
        --train_path your_path/train \
        --val_path   your_path/validation \
        --test_path  your_path/test \
        --deit_ckpt  DeiT_best_model.pth \
        --vit_ckpt   ViT_best_model.pth \
        --swin_ckpt  Swin_best_model.pth \
        --output_dir ./distilled_models
"""

import os
import argparse
import time
import gc

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DeiTForImageClassification, DeiTConfig,
    ViTForImageClassification,  ViTConfig,
    SwinForImageClassification, SwinConfig,
)
import matplotlib
matplotlib.use('Agg')  # No GUI required, saves to file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                              confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────
# 1. ARGUMENTS
# ─────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",    required=True,
                        choices=["Raabin", "Raabin_Diffusion", "PBC", "LISC"])
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path",   required=True)
    parser.add_argument("--test_path",  required=True)

    # Teacher checkpoint files
    parser.add_argument("--deit_ckpt",  required=True)
    parser.add_argument("--vit_ckpt",   required=True)
    parser.add_argument("--swin_ckpt",  required=True)

    # Distillation hyperparameters
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha",       type=float, default=0.4,
                        help="CE loss weight. KL = (1 - alpha)")

    # Training hyperparameters
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  type=str,   default="./distilled_models")

    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. CLASS DEFINITIONS
# ─────────────────────────────────────────────
CLASS_LABELS = {
    "Basophil":   0,
    "Eosinophil": 1,
    "Lymphocyte": 2,
    "Monocyte":   3,
    "Neutrophil": 4,
}
NUM_CLASSES  = 5
CLASS_NAMES  = list(CLASS_LABELS.keys())


# ─────────────────────────────────────────────
# 3. DATASET
#    384x384 for Teacher, 224x224 for Student
#    Generates both in a single pass — efficient
# ─────────────────────────────────────────────
def clahe_process(image, target_size):
    """Same logic as your original process_image function."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl      = clahe.apply(v)
    himg    = cv2.merge((h, s, cl))
    processed = cv2.cvtColor(himg, cv2.COLOR_HSV2RGB)
    resized   = cv2.resize(processed, (target_size, target_size),
                            interpolation=cv2.INTER_AREA)
    return resized / 255.0


class WBCDataset(Dataset):
    def __init__(self, data_path, teacher_size=384, student_size=224):
        self.file_paths   = []
        self.labels       = []
        self.teacher_size = teacher_size
        self.student_size = student_size

        for folder_name in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = CLASS_LABELS.get(folder_name)
            if label is None:
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.file_paths.append(os.path.join(folder_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label    = self.labels[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            t_img = torch.tensor(
                clahe_process(image, self.teacher_size), dtype=torch.float32
            ).permute(2, 0, 1)
            s_img = torch.tensor(
                clahe_process(image, self.student_size), dtype=torch.float32
            ).permute(2, 0, 1)
            return t_img, s_img, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error ({img_path}): {e}")
            return (
                torch.zeros(3, self.teacher_size, self.teacher_size),
                torch.zeros(3, self.student_size, self.student_size),
                torch.tensor(label, dtype=torch.long),
            )


def get_dataloaders(args):
    print("\n[1] Loading data...")
    train_ds = WBCDataset(args.train_path)
    val_ds   = WBCDataset(args.val_path)
    test_ds  = WBCDataset(args.test_path)
    print(f"  Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)}")

    kw = dict(batch_size=args.batch_size,
              num_workers=args.num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(test_ds,  shuffle=False, **kw))


# ─────────────────────────────────────────────
# 4. MODEL CLASSES (Teacher & Student)
# ─────────────────────────────────────────────
class EnhancedDeiTModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super().__init__()
        self.backbone   = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0]
        return self.classifier(self.batch_norm(cls))


class EnhancedViTModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super().__init__()
        self.backbone   = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0]
        return self.classifier(self.batch_norm(cls))


class EnhancedSwinModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super().__init__()
        self.backbone   = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 768),
            nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(768, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out    = self.backbone(x, output_hidden_states=True)
        pooled = torch.mean(out.hidden_states[-1], dim=1)
        return self.classifier(self.batch_norm(pooled))


# ─────────────────────────────────────────────
# 5. LOADING TEACHER
# ─────────────────────────────────────────────
def build_teacher(args, device):
    print("\n[2] Loading Teacher models...")

    deit_base = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    deit = EnhancedDeiTModel(deit_base).to(device)
    deit.load_state_dict(torch.load(args.deit_ckpt, map_location=device))
    deit.eval()
    for p in deit.parameters(): p.requires_grad = False
    print(f"  DeiT loaded: {args.deit_ckpt}")

    vit_base = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    vit = EnhancedViTModel(vit_base).to(device)
    vit.load_state_dict(torch.load(args.vit_ckpt, map_location=device))
    vit.eval()
    for p in vit.parameters(): p.requires_grad = False
    print(f"  ViT loaded: {args.vit_ckpt}")

    swin_base = SwinForImageClassification.from_pretrained(
        'microsoft/swin-base-patch4-window12-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    swin = EnhancedSwinModel(swin_base).to(device)
    swin.load_state_dict(torch.load(args.swin_ckpt, map_location=device))
    swin.eval()
    for p in swin.parameters(): p.requires_grad = False
    print(f"  Swin loaded: {args.swin_ckpt}")

    return deit, vit, swin


@torch.no_grad()
def teacher_predict(deit, vit, swin, teacher_imgs, weights, device):
    teacher_imgs = teacher_imgs.to(device)
    p_deit = F.softmax(deit(teacher_imgs), dim=1)
    p_vit  = F.softmax(vit(teacher_imgs),  dim=1)
    p_swin = F.softmax(swin(teacher_imgs), dim=1)
    return weights[0]*p_deit + weights[1]*p_vit + weights[2]*p_swin


# ─────────────────────────────────────────────
# 6. STUDENT — DeiT-Small 224x224
# ─────────────────────────────────────────────
class EnhancedDeiTSmallModel(nn.Module):
    """
    Exact copy of your EnhancedDeiTModel,
    only the backbone is deit-small-distilled-patch16-224.
    Params: ~22M  vs  Teacher total ~300M+
    """
    def __init__(self, num_classes=5):
        super().__init__()
        config   = DeiTConfig.from_pretrained(
            'facebook/deit-small-distilled-patch16-224', num_labels=num_classes)
        backbone = DeiTForImageClassification.from_pretrained(
            'facebook/deit-small-distilled-patch16-224',
            config=config, ignore_mismatched_sizes=True)
        self.backbone   = backbone
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0]
        return self.classifier(self.batch_norm(cls))


# ─────────────────────────────────────────────
# 7. DISTILLATION LOSS
# ─────────────────────────────────────────────
class DistillationLoss(nn.Module):
    """
    Loss = alpha * CE(student_logits, true_labels)
         + (1-alpha) * T^2 * KL(student_soft || teacher_soft)
    """
    def __init__(self, temperature=4.0, alpha=0.4):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss()
        self.kl    = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_probs, true_labels):
        ce_loss      = self.ce(student_logits, true_labels)
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(
            torch.log(teacher_probs.clamp(min=1e-8)) / self.T, dim=1)
        kl_loss = self.kl(student_soft, teacher_soft) * (self.T ** 2)
        total   = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        return total, ce_loss, kl_loss


# ─────────────────────────────────────────────
# 8. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_one_epoch(student, deit, vit, swin, ens_weights,
                    loader, optimizer, criterion, device, scaler):
    student.train()
    total_loss = correct = total = 0

    for teacher_imgs, student_imgs, labels in loader:
        labels       = labels.to(device)
        student_imgs = student_imgs.to(device)

        with torch.no_grad():
            teacher_probs = teacher_predict(
                deit, vit, swin, teacher_imgs, ens_weights, device)

        with torch.cuda.amp.autocast():
            logits        = student(student_imgs)
            loss, _, _    = criterion(logits, teacher_probs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    correct = total = 0
    for _, student_imgs, labels in loader:
        student_imgs, labels = student_imgs.to(device), labels.to(device)
        correct += (student(student_imgs).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


@torch.no_grad()
def full_metrics(student, loader, device, output_dir, dataset_name):
    student.eval()
    all_preds, all_labels, all_probs = [], [], []

    for _, student_imgs, labels in loader:
        student_imgs = student_imgs.to(device)
        logits = student(student_imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    rep = classification_report(all_labels, all_preds,
                                 target_names=CLASS_NAMES, digits=4)

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Student DeiT-Small — Confusion Matrix ({dataset_name})")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"confusion_matrix_{dataset_name}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  Confusion matrix saved: {cm_path}")

    # ── ROC Curve ──
    labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(9, 7))
    for i, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.4f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Student DeiT-Small — ROC Curve ({dataset_name})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"roc_curve_{dataset_name}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  ROC curve saved: {roc_path}")

    return acc, f1, rep


# ─────────────────────────────────────────────
# 9. ENSEMBLE WEIGHTS CALCULATION
#    Same strategy as calculate_ensemble_weights
# ─────────────────────────────────────────────
@torch.no_grad()
def calculate_ensemble_weights(deit, vit, swin, val_loader, device):
    print("\n[3] Calculating ensemble weights (based on val accuracy)...")
    accs = []
    for model in [deit, vit, swin]:
        correct = total = 0
        for teacher_imgs, _, labels in val_loader:
            teacher_imgs, labels = teacher_imgs.to(device), labels.to(device)
            correct += (model(teacher_imgs).argmax(1) == labels).sum().item()
            total   += labels.size(0)
        accs.append(correct / total)

    total_acc = sum(accs)
    weights   = [a / total_acc for a in accs]
    for n, a, w in zip(["DeiT", "ViT", "Swin"], accs, weights):
        print(f"  {n}: acc={a*100:.2f}%  weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────
# 10. PARAMETER & INFERENCE MEASUREMENT
# ─────────────────────────────────────────────
def model_complexity(model, input_size, device, batch_size=32):
    params = sum(p.numel() for p in model.parameters()) / 1e6
    dummy  = torch.randn(batch_size, 3, input_size, input_size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(dummy)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): model(dummy)
        torch.cuda.synchronize()
    ms = (time.time() - t0) / 100 * 1000
    return params, ms


# ─────────────────────────────────────────────
# 11. MAIN FUNCTION
# ─────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" TrimerNet-WBC Knowledge Distillation")
    print(f" Dataset : {args.dataset}")
    print(f" Device  : {device}")
    print(f" T={args.temperature}  alpha={args.alpha}")
    print(f"{'='*60}")

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args)

    # Teacher
    deit, vit, swin = build_teacher(args, device)

    # Ensemble weights
    ens_weights = calculate_ensemble_weights(deit, vit, swin, val_loader, device)

    # Complexity
    print("\n[4] Complexity analysis...")
    teacher_params = sum(
        sum(p.numel() for p in m.parameters()) / 1e6
        for m in [deit, vit, swin]
    )
    _, t_inf = model_complexity(deit, 384, device)
    print(f"  Teacher total : {teacher_params:.1f}M param  (~{t_inf*3:.1f}ms x3)")

    print("\n[5] Creating Student (DeiT-Small 224x224)...")
    student = EnhancedDeiTSmallModel(num_classes=NUM_CLASSES).to(device)
    s_params, s_inf = model_complexity(student, 224, device)
    print(f"  Student        : {s_params:.1f}M param  ({s_inf:.1f}ms)")
    print(f"  Reduction      : {teacher_params/s_params:.1f}x param  |  "
          f"~{(t_inf*3)/s_inf:.1f}x speed")

    # Loss & Optimizer
    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    optimizer = torch.optim.AdamW(student.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # Training
    print(f"\n[6] Distillation ({args.epochs} epochs)...")
    best_val_acc = 0.0
    patience_cnt = 0
    best_ckpt    = os.path.join(
        args.output_dir, f"student_deit_small_{args.dataset}_best.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            student, deit, vit, swin, ens_weights,
            train_loader, optimizer, criterion, device, scaler)
        val_acc = evaluate(student, val_loader, device)
        scheduler.step()

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save(student.state_dict(), best_ckpt)
            tag = " <- best"
        else:
            patience_cnt += 1

        print(f"Epoch [{epoch:03d}/{args.epochs}]  "
              f"Loss:{tr_loss:.4f}  Train:{tr_acc*100:.2f}%  "
              f"Val:{val_acc*100:.2f}%  ({time.time()-t0:.1f}s){tag}")

        if patience_cnt >= args.patience:
            print(f"\nEarly stopping.")
            break

        gc.collect(); torch.cuda.empty_cache()

    # Test
    print(f"\n[7] Test evaluation...")
    student.load_state_dict(torch.load(best_ckpt))
    test_acc, test_f1, report = full_metrics(student, test_loader, device, args.output_dir, args.dataset)

    print(f"\n{'='*60}")
    print(f" RESULTS — {args.dataset}  |  Student DeiT-Small")
    print(f"{'='*60}")
    print(f" Test Accuracy : {test_acc*100:.2f}%")
    print(f" Weighted F1   : {test_f1*100:.2f}%")
    print(f"\n{report}")
    print(f" Teacher (3 model) : {teacher_params:.1f}M param")
    print(f" Student           : {s_params:.1f}M param  ({teacher_params/s_params:.1f}x smaller)")
    print(f" Inference gain    : ~{(t_inf*3)/s_inf:.1f}x faster")
    print(f"{'='*60}")

    # Save
    out_txt = os.path.join(args.output_dir, f"results_{args.dataset}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Dataset       : {args.dataset}\n")
        f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
        f.write(f"Weighted F1   : {test_f1*100:.2f}%\n")
        f.write(f"Teacher Params: {teacher_params:.1f}M (3 model)\n")
        f.write(f"Student Params: {s_params:.1f}M\n")
        f.write(f"Param Reduction: {teacher_params/s_params:.1f}x\n")
        f.write(f"Speed Gain    : ~{(t_inf*3)/s_inf:.1f}x\n\n")
        f.write(report)

    print(f"\nResults: {out_txt}")
    print(f"Student: {best_ckpt}")


if __name__ == "__main__":
    main()