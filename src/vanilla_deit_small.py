"""
TrimerNet-WBC — Vanilla DeiT-Small (Distillation Ablation Study)
=======================================================
Goal: train DeiT-Small WITHOUT knowledge distillation, using standard supervised
      training. For the ablation study:
        DeiT-Small (vanilla)  vs  DeiT-Small (distilled/student)
      This demonstrates the contribution of knowledge distillation.

SAME architecture, SAME preprocessing, SAME split as the student.
ONLY DIFFERENCE: no distillation loss, just standard CrossEntropy.

USAGE (Raabin example):
    python vanilla_deit_small.py \
        --dataset Raabin \
        --train_path path/to/Raabin/train \
        --val_path   path/to/Raabin/validation \
        --test_path  path/to/Raabin/test \
        --output_dir ./vanilla_deit
"""

import os
import argparse
import time
import gc
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DeiTForImageClassification, DeiTConfig
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, cohen_kappa_score,
                              matthews_corrcoef)


CLASS_LABELS = {
    "Basophil": 0, "Eosinophil": 1, "Lymphocyte": 2,
    "Monocyte": 3, "Neutrophil": 4,
}
NUM_CLASSES = 5
CLASS_NAMES = list(CLASS_LABELS.keys())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     required=True,
                   choices=["Raabin", "Raabin_Diffusion", "PBC", "LISC"])
    p.add_argument("--train_path",  required=True)
    p.add_argument("--val_path",    required=True)
    p.add_argument("--test_path",   required=True)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--output_dir",  type=str,   default="./vanilla_deit")
    return p.parse_args()


# -- Preprocessing (same as student: 224x224, CLAHE) --
def clahe_process(image, target_size=224):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(v)
    himg = cv2.merge((h, s, cl))
    processed = cv2.cvtColor(himg, cv2.COLOR_HSV2RGB)
    resized = cv2.resize(processed, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)
    return resized / 255.0


class WBCDataset(Dataset):
    def __init__(self, data_path, img_size=224):
        self.file_paths, self.labels = [], []
        self.img_size = img_size
        for folder in os.listdir(data_path):
            fp = os.path.join(data_path, folder)
            if not os.path.isdir(fp):
                continue
            label = CLASS_LABELS.get(folder)
            if label is None:
                continue
            for f in os.listdir(fp):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.file_paths.append(os.path.join(fp, f))
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path, label = self.file_paths[idx], self.labels[idx]
        try:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = torch.tensor(clahe_process(image, self.img_size),
                               dtype=torch.float32).permute(2, 0, 1)
            return img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error ({path}): {e}")
            return (torch.zeros(3, self.img_size, self.img_size),
                    torch.tensor(label, dtype=torch.long))


# -- Model: SAME architecture as the student (EnhancedDeiTSmallModel) --
class EnhancedDeiTSmallModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        config = DeiTConfig.from_pretrained(
            'facebook/deit-small-distilled-patch16-224', num_labels=num_classes)
        backbone = DeiTForImageClassification.from_pretrained(
            'facebook/deit-small-distilled-patch16-224',
            config=config, ignore_mismatched_sizes=True)
        self.backbone = backbone
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        return self.classifier(self.batch_norm(out.hidden_states[-1][:, 0]))


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def full_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro")
    f1w = f1_score(all_labels, all_preds, average="weighted")
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    rep = classification_report(all_labels, all_preds,
                                 target_names=CLASS_NAMES, digits=4)
    return acc, f1m, f1w, kappa, mcc, rep, all_labels, all_preds


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Vanilla DeiT-Small (NO distillation) — {args.dataset}")
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
    print("\n[2] Building Vanilla DeiT-Small...")
    model = EnhancedDeiTSmallModel(NUM_CLASSES).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.1f}M")

    # -- Loss: SAME as the distilled student — NO class weight --
    # (The distilled student's CE component also uses no class weight,
    #  kept identical for a fair ablation)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # -- Training --
    print(f"\n[3] Training ({args.epochs} epochs, NO distillation)...")
    best_val_acc = 0.0
    patience_cnt = 0
    best_ckpt = os.path.join(args.output_dir,
                             f"vanilla_deit_small_{args.dataset}_best.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

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
    acc, f1m, f1w, kappa, mcc, rep, labels, preds = \
        full_metrics(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f" VANILLA DeiT-Small RESULTS — {args.dataset}")
    print(f"{'='*60}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Macro F1      : {f1m*100:.2f}%")
    print(f"  Weighted F1   : {f1w*100:.2f}%")
    print(f"  Cohen's Kappa : {kappa:.4f}")
    print(f"  MCC           : {mcc:.4f}")
    print(f"  Parameters    : {params:.1f}M")
    print(f"\n{rep}")
    print(f"{'='*60}")

    # -- Save --
    out_txt = os.path.join(args.output_dir, f"vanilla_deit_results_{args.dataset}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Dataset       : {args.dataset}\n")
        f.write(f"Model         : Vanilla DeiT-Small (NO distillation)\n")
        f.write(f"Params        : {params:.1f}M\n")
        f.write(f"Test Accuracy : {acc*100:.2f}%\n")
        f.write(f"Macro F1      : {f1m*100:.2f}%\n")
        f.write(f"Weighted F1   : {f1w*100:.2f}%\n")
        f.write(f"Cohen's Kappa : {kappa:.4f}\n")
        f.write(f"MCC           : {mcc:.4f}\n\n")
        f.write(rep)

    print(f"\nResults: {out_txt}")


if __name__ == "__main__":
    main()
