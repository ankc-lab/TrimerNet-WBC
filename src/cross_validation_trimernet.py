"""
TrimerNet-WBC — 5-Fold Cross Validation (Student Distillation)
================================================================
Teacher : EnhancedDeiT + EnhancedViT + EnhancedSwin (Weighted Average Ensemble)
Student : EnhancedDeiT-Small (224x224)

Per fold process:
  - Train and Validation sets are combined, then split into 5 equal parts.
  - Teacher remains fixed (pre-trained .pth files).
  - Student is trained from scratch.
  - Test set remains untouched throughout the cross-validation.

USAGE (Raabin example):
    python scripts/cross_validation.py \
        --dataset     Raabin \
        --train_path  your_path/Raabin/train \
        --val_path    your_path/Raabin/validation \
        --test_path   your_path/Raabin/test \
        --deit_ckpt   your_path/Raabin/DeiT_best_model.pth \
        --vit_ckpt    your_path/Raabin/ViT_best_model.pth \
        --swin_ckpt   your_path/Raabin/Swin_best_model.pth \
        --output_dir  your_path/Raabin/cv_results \
        --n_folds     5
"""

import os
import argparse
import time
import gc
import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import (
    DeiTForImageClassification, DeiTConfig,
    ViTForImageClassification,
    SwinForImageClassification,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix,
                              cohen_kappa_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────
# 1. ARGUMENTS
# ─────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True,
                        choices=["Raabin", "Raabin_DCGAN", "PBC", "LISC"])
    parser.add_argument("--train_path",  required=True)
    parser.add_argument("--val_path",    required=True)
    parser.add_argument("--test_path",   required=True)
    parser.add_argument("--deit_ckpt",   required=True)
    parser.add_argument("--vit_ckpt",    required=True)
    parser.add_argument("--swin_ckpt",   required=True)
    parser.add_argument("--n_folds",     type=int,   default=5)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha",       type=float, default=0.4)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--output_dir",  type=str,   default="./cv_results")
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
CLASS_LABELS = {
    "Basophil":   0,
    "Eosinophil": 1,
    "Lymphocyte": 2,
    "Monocyte":   3,
    "Neutrophil": 4,
}
NUM_CLASSES = 5
CLASS_NAMES = list(CLASS_LABELS.keys())


# ─────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────
def clahe_process(image, target_size):
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
            print(f"Hata ({img_path}): {e}")
            return (
                torch.zeros(3, self.teacher_size, self.teacher_size),
                torch.zeros(3, self.student_size, self.student_size),
                torch.tensor(label, dtype=torch.long),
            )


# ─────────────────────────────────────────────
# 4. MODEL CLASSES
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


class EnhancedDeiTSmallModel(nn.Module):
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
# 5. LOADING TEACHER MODELS
# ─────────────────────────────────────────────
def build_teacher(args, device):
    print("\n[1] Loading teacher models...")

    deit_base = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    deit = EnhancedDeiTModel(deit_base).to(device)
    deit.load_state_dict(torch.load(args.deit_ckpt, map_location=device))
    deit.eval()
    for p in deit.parameters(): p.requires_grad = False

    vit_base = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    vit = EnhancedViTModel(vit_base).to(device)
    vit.load_state_dict(torch.load(args.vit_ckpt, map_location=device))
    vit.eval()
    for p in vit.parameters(): p.requires_grad = False

    swin_base = SwinForImageClassification.from_pretrained(
        'microsoft/swin-base-patch4-window12-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    swin = EnhancedSwinModel(swin_base).to(device)
    swin.load_state_dict(torch.load(args.swin_ckpt, map_location=device))
    swin.eval()
    for p in swin.parameters(): p.requires_grad = False

    print("  DeiT, ViT, Swin loaded.")
    return deit, vit, swin


@torch.no_grad()
def teacher_predict(deit, vit, swin, teacher_imgs, weights, device):
    teacher_imgs = teacher_imgs.to(device)
    p_deit = F.softmax(deit(teacher_imgs), dim=1)
    p_vit  = F.softmax(vit(teacher_imgs),  dim=1)
    p_swin = F.softmax(swin(teacher_imgs), dim=1)
    return weights[0]*p_deit + weights[1]*p_vit + weights[2]*p_swin


# ─────────────────────────────────────────────
# 6. ENSEMBLE WEIGHTS
# ─────────────────────────────────────────────
@torch.no_grad()
def calculate_ensemble_weights(deit, vit, swin, loader, device):
    accs = []
    for model in [deit, vit, swin]:
        correct = total = 0
        for teacher_imgs, _, labels in loader:
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
# 7. DISTILLATION LOSS
# ─────────────────────────────────────────────
class DistillationLoss(nn.Module):
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
        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss


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
        with torch.amp.autocast('cuda'):
            logits = student(student_imgs)
            loss   = criterion(logits, teacher_probs, labels)
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
def full_metrics(student, loader, device):
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
    acc   = accuracy_score(all_labels, all_preds)
    f1m   = f1_score(all_labels, all_preds, average="macro")
    f1w   = f1_score(all_labels, all_preds, average="weighted")
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc   = matthews_corrcoef(all_labels, all_preds)
    rep   = classification_report(all_labels, all_preds,
                                   target_names=CLASS_NAMES, digits=4)
    return acc, f1m, f1w, kappa, mcc, rep, all_labels, all_preds, all_probs


# ─────────────────────────────────────────────
# 9. SINGLE FOLD TRAINING
# ─────────────────────────────────────────────
def train_fold(fold_idx, train_subset, val_subset, deit, vit, swin,
               ens_weights, args, device):
    print(f"\n  --- Fold {fold_idx+1}/{args.n_folds} ---")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # Initialize a new student — from scratch for each fold
    student   = EnhancedDeiTSmallModel(num_classes=NUM_CLASSES).to(device)
    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    optimizer = torch.optim.AdamW(student.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_acc  = 0.0
    patience_cnt  = 0
    best_ckpt     = os.path.join(
        args.output_dir, f"fold{fold_idx+1}_{args.dataset}_best.pth")

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

        print(f"  Epoch [{epoch:02d}/{args.epochs}]  "
              f"Loss:{tr_loss:.4f}  Train:{tr_acc*100:.2f}%  "
              f"Val:{val_acc*100:.2f}%  ({time.time()-t0:.1f}s){tag}")

        if patience_cnt >= args.patience:
            print(f"  Early stopping.")
            break

        gc.collect()
        torch.cuda.empty_cache()

    student.load_state_dict(torch.load(best_ckpt))
    val_acc, val_f1m, val_f1w, _, _, _, _, _, _ = full_metrics(student, val_loader, device)
    print(f"  Fold {fold_idx+1} Val Accuracy: {val_acc*100:.2f}%  "
          f"Macro F1: {val_f1m*100:.2f}%")

    # Save checkpoint — required for best fold test evaluation
    del student
    gc.collect()
    torch.cuda.empty_cache()

    return val_acc, val_f1m, val_f1w, best_ckpt


# ─────────────────────────────────────────────
# 10. MAIN EXECUTION
# ─────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" TrimerNet-WBC  5-Fold Cross Validation")
    print(f" Dataset : {args.dataset}")
    print(f" Folds   : {args.n_folds}")
    print(f" Device  : {device}")
    print(f"{'='*60}")

    # ── Dataset: Combine Train and Val ──
    print("\n[2] Loading data (Train and Val will be combined)...")
    train_ds = WBCDataset(args.train_path)
    val_ds   = WBCDataset(args.val_path)
    test_ds  = WBCDataset(args.test_path)
    combined = ConcatDataset([train_ds, val_ds])

    all_labels = (
        train_ds.labels + val_ds.labels
    )
    all_labels = np.array(all_labels)

    print(f"  Train+Val: {len(combined)} | Test: {len(test_ds)}")

    # ── Teacher ──
    deit, vit, swin = build_teacher(args, device)

    print("\n[3] Ensemble weights being calculated...")
    full_loader = DataLoader(combined, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    ens_weights = calculate_ensemble_weights(deit, vit, swin, full_loader, device)

    # ── 5-Fold CV ──
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    fold_accs, fold_f1m, fold_f1w, fold_ckpts = [], [], [], []

    print(f"\n[4] {args.n_folds}-Fold CV starting...")
    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(combined)), all_labels)):

        train_subset = Subset(combined, train_idx)
        val_subset   = Subset(combined, val_idx)

        acc, f1m, f1w, ckpt = train_fold(
            fold_idx, train_subset, val_subset,
            deit, vit, swin, ens_weights, args, device
        )
        fold_accs.append(acc)
        fold_f1m.append(f1m)
        fold_f1w.append(f1w)
        fold_ckpts.append(ckpt)

    # ── CV Results ──
    acc_mean  = np.mean(fold_accs)  * 100
    acc_std   = np.std(fold_accs)   * 100
    f1m_mean  = np.mean(fold_f1m)   * 100
    f1m_std   = np.std(fold_f1m)    * 100
    f1w_mean  = np.mean(fold_f1w)   * 100
    f1w_std   = np.std(fold_f1w)    * 100

    print(f"\n{'='*60}")
    print(f" {args.n_folds}-FOLD CV RESULTS — {args.dataset}")
    print(f"{'='*60}")
    for i, (a, f) in enumerate(zip(fold_accs, fold_f1m)):
        print(f"  Fold {i+1}: Acc={a*100:.2f}%  Macro F1={f*100:.2f}%")
    print(f"{'─'*60}")
    print(f"  Accuracy   : {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f"  Macro F1   : {f1m_mean:.2f}% ± {f1m_std:.2f}%")
    print(f"  Weighted F1: {f1w_mean:.2f}% ± {f1w_std:.2f}%")
    print(f"{'='*60}")

    # ── Best fold → Test set evaluation ──
    best_fold_idx  = int(np.argmax(fold_accs))
    best_fold_ckpt = fold_ckpts[best_fold_idx]
    print(f"\n[5] Best fold: Fold {best_fold_idx+1} "
          f"(Val Acc={fold_accs[best_fold_idx]*100:.2f}%)")
    print(f"    Evaluating on test set...")

    best_student = EnhancedDeiTSmallModel(num_classes=NUM_CLASSES).to(device)
    best_student.load_state_dict(torch.load(best_fold_ckpt))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    test_acc, test_f1m, test_f1w, test_kappa, test_mcc, test_rep, \
        test_labels, test_preds, test_probs = \
        full_metrics(best_student, test_loader, device)

    print(f"\n{'='*60}")
    print(f" TEST RESULTS — {args.dataset} (Best Fold {best_fold_idx+1})")
    print(f"{'='*60}")
    print(f"  Test Accuracy  : {test_acc*100:.2f}%")
    print(f"  Macro F1       : {test_f1m*100:.2f}%")
    print(f"  Weighted F1    : {test_f1w*100:.2f}%")
    print(f"  Cohen's Kappa  : {test_kappa:.4f}")
    print(f"  MCC            : {test_mcc:.4f}")
    print(f"\n{test_rep}")

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Best Fold {best_fold_idx+1} — Test Confusion Matrix ({args.dataset})")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, f"cv_confusion_matrix_{args.dataset}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  Confusion matrix: {cm_path}")

    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    labels_bin = label_binarize(test_labels, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(9, 7))
    for i, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], test_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.4f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Best Fold {best_fold_idx+1} — ROC Curve ({args.dataset})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(args.output_dir, f"cv_roc_curve_{args.dataset}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  ROC curve: {roc_path}")

    del best_student
    gc.collect()
    torch.cuda.empty_cache()

    # ── Save Results ──
    out_txt = os.path.join(args.output_dir, f"cv_results_{args.dataset}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Dataset   : {args.dataset}\n")
        f.write(f"Folds     : {args.n_folds}\n\n")
        f.write("--- CV (Val) Results ---\n")
        for i, (a, fm) in enumerate(zip(fold_accs, fold_f1m)):
            f.write(f"Fold {i+1}: Acc={a*100:.2f}%  Macro F1={fm*100:.2f}%\n")
        f.write(f"\nAccuracy   : {acc_mean:.2f}% +/- {acc_std:.2f}%\n")
        f.write(f"Macro F1   : {f1m_mean:.2f}% +/- {f1m_std:.2f}%\n")
        f.write(f"Weighted F1: {f1w_mean:.2f}% +/- {f1w_std:.2f}%\n")
        f.write(f"\n--- Test Results (Best Fold {best_fold_idx+1}) ---\n")
        f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
        f.write(f"Macro F1      : {test_f1m*100:.2f}%\n")
        f.write(f"Weighted F1   : {test_f1w*100:.2f}%\n")
        f.write(f"Cohen's Kappa : {test_kappa:.4f}\n")
        f.write(f"MCC           : {test_mcc:.4f}\n\n")
        f.write(test_rep)

    print(f"\nResults saved: {out_txt}")

    # ── CV Results Graph ──
    plt.figure(figsize=(8, 5))
    folds = [f"Fold {i+1}" for i in range(args.n_folds)]
    plt.bar(folds, [a*100 for a in fold_accs], color='steelblue', alpha=0.8)
    plt.axhline(y=acc_mean, color='red', linestyle='--',
                label=f'Mean: {acc_mean:.2f}%')
    plt.ylabel("Accuracy (%)")
    plt.title(f"5-Fold CV — Student DeiT-Small ({args.dataset})")
    plt.legend()
    plt.ylim([min([a*100 for a in fold_accs])-1, 100])
    plt.tight_layout()
    chart_path = os.path.join(args.output_dir, f"cv_chart_{args.dataset}.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Graph saved: {chart_path}")

if __name__ == '__main__':
    main()
