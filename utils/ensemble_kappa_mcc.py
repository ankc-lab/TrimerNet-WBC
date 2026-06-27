"""
TrimerNet-WBC — Teacher Ensemble Kappa & MCC Computation
========================================================
Goal: compute Cohen's Kappa and MCC for the teacher ensemble
      on the test set.
      For metrics consistent with the student and ResNet50.

USAGE (Raabin example):
    python ensemble_kappa_mcc.py \
        --dataset Raabin_Diffusion \
        --test_path  path/to/Raabin_Diffusion/test \
        --val_path   path/to/Raabin_Diffusion/validation \
        --deit_ckpt  path/to/DeiT_best_model.pth \
        --vit_ckpt   path/to/ViT_best_model.pth \
        --swin_ckpt  path/to/Swin_best_model.pth \
        --output_dir ./ensemble_metrics
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DeiTForImageClassification,
    ViTForImageClassification,
    SwinForImageClassification,
)
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              cohen_kappa_score, matthews_corrcoef)


CLASS_LABELS = {
    "Basophil": 0, "Eosinophil": 1, "Lymphocyte": 2,
    "Monocyte": 3, "Neutrophil": 4,
}
NUM_CLASSES = 5
CLASS_NAMES = list(CLASS_LABELS.keys())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True)
    p.add_argument("--test_path",  required=True)
    p.add_argument("--val_path",   required=True)
    p.add_argument("--deit_ckpt",  required=True)
    p.add_argument("--vit_ckpt",   required=True)
    p.add_argument("--swin_ckpt",  required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers",type=int, default=0)
    p.add_argument("--output_dir", default="./ensemble_metrics")
    return p.parse_args()


def clahe_process(image, target_size=384):
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
    def __init__(self, data_path, img_size=384):
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


class EnhancedDeiTModel(nn.Module):
    def __init__(self, m, num_classes=5):
        super().__init__()
        self.backbone = m
        self.batch_norm = nn.BatchNorm1d(m.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(m.config.hidden_size, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        return self.classifier(self.batch_norm(out.hidden_states[-1][:, 0]))


class EnhancedViTModel(nn.Module):
    def __init__(self, m, num_classes=5):
        super().__init__()
        self.backbone = m
        self.batch_norm = nn.BatchNorm1d(m.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(m.config.hidden_size, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        return self.classifier(self.batch_norm(out.hidden_states[-1][:, 0]))


class EnhancedSwinModel(nn.Module):
    def __init__(self, m, num_classes=5):
        super().__init__()
        self.backbone = m
        self.batch_norm = nn.BatchNorm1d(m.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(m.config.hidden_size, 768),
            nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(768, num_classes))

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        pooled = torch.mean(out.hidden_states[-1], dim=1)
        return self.classifier(self.batch_norm(pooled))


def load_teachers(args, device):
    deit_b = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    deit = EnhancedDeiTModel(deit_b).to(device)
    deit.load_state_dict(torch.load(args.deit_ckpt, map_location=device))
    deit.eval()

    vit_b = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    vit = EnhancedViTModel(vit_b).to(device)
    vit.load_state_dict(torch.load(args.vit_ckpt, map_location=device))
    vit.eval()

    swin_b = SwinForImageClassification.from_pretrained(
        'microsoft/swin-base-patch4-window12-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    swin = EnhancedSwinModel(swin_b).to(device)
    swin.load_state_dict(torch.load(args.swin_ckpt, map_location=device))
    swin.eval()
    return deit, vit, swin


@torch.no_grad()
def get_weights(deit, vit, swin, loader, device):
    accs = []
    for model in [deit, vit, swin]:
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
        accs.append(correct / total)
    s = sum(accs)
    return [a / s for a in accs]


@torch.no_grad()
def ensemble_predict(deit, vit, swin, weights, loader, device):
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        p = (weights[0] * F.softmax(deit(imgs), dim=1) +
             weights[1] * F.softmax(vit(imgs), dim=1) +
             weights[2] * F.softmax(swin(imgs), dim=1))
        all_preds.extend(p.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*55}\n Teacher Ensemble Kappa & MCC — {args.dataset}\n{'='*55}")

    val_ds  = WBCDataset(args.val_path)
    test_ds = WBCDataset(args.test_path)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, shuffle=False)
    val_loader  = DataLoader(val_ds, **kw)
    test_loader = DataLoader(test_ds, **kw)

    print("\nLoading teacher models...")
    deit, vit, swin = load_teachers(args, device)

    print("Computing ensemble weights...")
    weights = get_weights(deit, vit, swin, val_loader, device)
    print(f"  DeiT={weights[0]:.4f}  ViT={weights[1]:.4f}  Swin={weights[2]:.4f}")

    print("Running predictions on the test set...")
    preds, labels = ensemble_predict(deit, vit, swin, weights, test_loader, device)

    acc   = accuracy_score(labels, preds)
    f1m   = f1_score(labels, preds, average="macro")
    f1w   = f1_score(labels, preds, average="weighted")
    kappa = cohen_kappa_score(labels, preds)
    mcc   = matthews_corrcoef(labels, preds)

    print(f"\n{'='*55}\n RESULTS — {args.dataset} (Teacher Ensemble)\n{'='*55}")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  Macro F1      : {f1m*100:.2f}%")
    print(f"  Weighted F1   : {f1w*100:.2f}%")
    print(f"  Cohen's Kappa : {kappa:.4f}")
    print(f"  MCC           : {mcc:.4f}")
    print(f"{'='*55}")

    out_txt = os.path.join(args.output_dir, f"ensemble_kappa_mcc_{args.dataset}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Dataset: {args.dataset} (Teacher Ensemble)\n\n")
        f.write(f"Accuracy    : {acc*100:.2f}%\n")
        f.write(f"Macro F1    : {f1m*100:.2f}%\n")
        f.write(f"Weighted F1 : {f1w*100:.2f}%\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"MCC         : {mcc:.4f}\n")
    print(f"\nResults: {out_txt}")


if __name__ == "__main__":
    main()
