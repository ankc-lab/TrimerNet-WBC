"""
TrimerNet-WBC — ResNet50 Baseline
==================================

SAME pipeline, SAME preprocessing, SAME split as the student DeiT-Small.

NO knowledge distillation — standard supervised training.
This enables a fair comparison: ResNet50 vs Student DeiT-Small.

USAGE (Raabin example):
    python resnet50_baseline.py \
        --dataset Raabin \
        --train_path path/to/Raabin/train \
        --val_path   path/to/Raabin/validation \
        --test_path  path/to/Raabin/test \
        --output_dir ./resnet50_results
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
import torchvision.models as tv_models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, cohen_kappa_score,
                              matthews_corrcoef, roc_curve, auc)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────
# 1. ARGUMENTS
# ─────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True,
                        choices=["Raabin", "Raabin_Diffusion", "PBC", "LISC"])
    parser.add_argument("--train_path",  required=True)
    parser.add_argument("--val_path",    required=True)
    parser.add_argument("--test_path",   required=True)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--output_dir",  type=str,   default="./resnet50_results")
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
CLASS_LABELS = {
    "Basophil":   0, "Eosinophil": 1, "Lymphocyte": 2,
    "Monocyte":   3, "Neutrophil": 4,
}
NUM_CLASSES = 5
CLASS_NAMES = list(CLASS_LABELS.keys())


# ─────────────────────────────────────────────
# 3. DATASET — same preprocessing as the student (CLAHE)
#    ResNet50 uses standard 224x224 input
# ─────────────────────────────────────────────
def clahe_process(image, target_size=224):
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
    def __init__(self, data_path, img_size=224):
        self.file_paths = []
        self.labels     = []
        self.img_size   = img_size
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
            img   = torch.tensor(
                clahe_process(image, self.img_size), dtype=torch.float32
            ).permute(2, 0, 1)
            return img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error ({img_path}): {e}")
            return (torch.zeros(3, self.img_size, self.img_size),
                    torch.tensor(label, dtype=torch.long))


# ─────────────────────────────────────────────
# 4. RESNET50 MODEL
# ─────────────────────────────────────────────
def build_resnet50(num_classes=5, device="cuda"):
    """ImageNet pre-trained ResNet50 with the final layer adapted to 5 classes."""
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    # Classifier head similar to the student (fair comparison)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model.to(device)


# ─────────────────────────────────────────────
# 5. TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


@torch.no_grad()
def full_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
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
# 6. MAIN
# ─────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" ResNet50 Baseline — {args.dataset}")
    print(f" Device: {device}")
    print(f"{'='*60}")

    # -- Data --
    print("\n[1] Loading data...")
    train_ds = WBCDataset(args.train_path)
    val_ds   = WBCDataset(args.val_path)
    test_ds  = WBCDataset(args.test_path)
    print(f"  Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)}")

    kw = dict(batch_size=args.batch_size,
              num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # ── Model ──
    print("\n[2] Building ResNet50 (ImageNet pre-trained)...")
    model  = build_resnet50(NUM_CLASSES, device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parametre: {params:.1f}M")

    # -- Class-weighted loss (same logic as in the paper) --
    class_counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES)
    class_weights = class_counts.sum() / (class_counts + 1e-8)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    # -- Training --
    print(f"\n[3] Training ({args.epochs} epochs)...")
    best_val_acc = 0.0
    patience_cnt = 0
    best_ckpt    = os.path.join(args.output_dir,
                                 f"resnet50_{args.dataset}_best.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(tr_loss)

        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt)
            tag = " <- best"
        else:
            patience_cnt += 1

        print(f"Epoch [{epoch:02d}/{args.epochs}]  "
              f"Loss:{tr_loss:.4f}  Train:{tr_acc*100:.2f}%  "
              f"Val:{val_acc*100:.2f}%  ({time.time()-t0:.1f}s){tag}")

        if patience_cnt >= args.patience:
            print(f"\nEarly stopping.")
            break
        gc.collect(); torch.cuda.empty_cache()

    # ── Test ──
    print(f"\n[4] Test evaluation...")
    model.load_state_dict(torch.load(best_ckpt))
    test_acc, test_f1m, test_f1w, kappa, mcc, rep, labels, preds, probs = \
        full_metrics(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f" RESNET50 RESULTS — {args.dataset}")
    print(f"{'='*60}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Macro F1      : {test_f1m*100:.2f}%")
    print(f"  Weighted F1   : {test_f1w*100:.2f}%")
    print(f"  Cohen's Kappa : {kappa:.4f}")
    print(f"  MCC           : {mcc:.4f}")
    print(f"  Parametre     : {params:.1f}M")
    print(f"\n{rep}")
    print(f"{'='*60}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"ResNet50 Baseline — Confusion Matrix ({args.dataset})")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, f"resnet50_cm_{args.dataset}.png")
    plt.savefig(cm_path, dpi=300); plt.close()

    # -- Save --
    out_txt = os.path.join(args.output_dir, f"resnet50_results_{args.dataset}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Dataset       : {args.dataset}\n")
        f.write(f"Model         : ResNet50 (CNN baseline)\n")
        f.write(f"Params        : {params:.1f}M\n")
        f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
        f.write(f"Macro F1      : {test_f1m*100:.2f}%\n")
        f.write(f"Weighted F1   : {test_f1w*100:.2f}%\n")
        f.write(f"Cohen's Kappa : {kappa:.4f}\n")
        f.write(f"MCC           : {mcc:.4f}\n\n")
        f.write(rep)

    print(f"\nResults: {out_txt}")
    print(f"Confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
