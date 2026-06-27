"""
TrimerNet-WBC — ResNet50 ROC/AUC Computation
============================================
Goal: compute ROC curves and AUC for the saved ResNet50
      models on the test set.
      NO retraining — evaluation only.

USAGE (Raabin example):
    python resnet50_roc_auc.py \
        --dataset Raabin \
        --test_path  path/to/Raabin/test \
        --ckpt       path/to/resnet50_Raabin_best.pth \
        --output_dir ./resnet50_roc
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


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
    p.add_argument("--ckpt",       required=True,
                   help="Path to the saved ResNet50 .pth file")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers",type=int, default=0)
    p.add_argument("--output_dir", default="./resnet50_roc")
    return p.parse_args()


# ── Preprocessing (ResNet50 224x224, student ile ayni) ──
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


def build_resnet50(num_classes=5):
    """resnet50_baseline.py ile AYNI mimari."""
    model = tv_models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


@torch.no_grad()
def get_probs(model, loader, device):
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        probs = F.softmax(model(imgs), dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*55}\n ResNet50 ROC/AUC — {args.dataset}\n{'='*55}")

    test_ds = WBCDataset(args.test_path)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True, shuffle=False)

    print("\nLoading ResNet50 model...")
    model = build_resnet50(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"  Loaded: {args.ckpt}")

    print("Running predictions on the test set...")
    probs, labels = get_probs(model, test_loader, device)

    # ── AUC ──
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))

    print(f"\n{'='*55}\n AUC RESULTS — {args.dataset}\n{'='*55}")
    auc_values = {}
    for i, cname in enumerate(CLASS_NAMES):
        a = roc_auc_score(labels_bin[:, i], probs[:, i])
        auc_values[cname] = a
        print(f"  {cname:<12}: AUC = {a:.4f}")

    macro_auc = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
    weighted_auc = roc_auc_score(labels_bin, probs, average="weighted", multi_class="ovr")
    print(f"  {'-'*30}")
    print(f"  Macro AUC   : {macro_auc:.4f}")
    print(f"  Weighted AUC: {weighted_auc:.4f}")
    print(f"{'='*55}")

    # -- ROC curve --
    plt.figure(figsize=(9, 7))
    for i, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{cname} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ResNet50 Baseline — ROC Curve ({args.dataset})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(args.output_dir, f"resnet50_roc_{args.dataset}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"\nROC curve: {roc_path}")

    # -- Save --
    out_txt = os.path.join(args.output_dir, f"resnet50_auc_{args.dataset}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Dataset: {args.dataset}\n\n")
        for cname, a in auc_values.items():
            f.write(f"{cname}: {a:.4f}\n")
        f.write(f"\nMacro AUC: {macro_auc:.4f}\n")
        f.write(f"Weighted AUC: {weighted_auc:.4f}\n")
    print(f"Results: {out_txt}")


if __name__ == "__main__":
    main()
