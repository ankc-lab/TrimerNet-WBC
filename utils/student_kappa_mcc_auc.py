"""
TrimerNet-WBC — Student DeiT-Small Kappa, MCC & AUC Computation
===============================================================
Goal: compute Cohen's Kappa, MCC and AUC for the saved student
      models on the test set.
      For a metric table consistent with the teacher.

NO retraining — evaluation only.

USAGE (Raabin example):
    python student_kappa_mcc_auc.py \
        --dataset Raabin \
        --test_path path/to/Raabin/test \
        --ckpt path/to/student_deit_small_Raabin_best.pth \
        --output_dir ./student_metrics
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DeiTForImageClassification, DeiTConfig
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                              matthews_corrcoef, roc_auc_score)
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
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers",type=int, default=0)
    p.add_argument("--output_dir", default="./student_metrics")
    return p.parse_args()


# ── Preprocessing (student 224x224) ──
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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
    return (np.array(all_labels), np.array(all_preds), np.array(all_probs))


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*55}\n Student Kappa/MCC/AUC — {args.dataset}\n{'='*55}")

    test_ds = WBCDataset(args.test_path)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True, shuffle=False)

    print(f"\nLoading student model: {args.ckpt}")
    model = EnhancedDeiTSmallModel(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    print("Running predictions on the test set...")
    labels, preds, probs = evaluate(model, test_loader, device)

    # -- Metrics --
    acc   = accuracy_score(labels, preds)
    f1m   = f1_score(labels, preds, average="macro")
    f1w   = f1_score(labels, preds, average="weighted")
    kappa = cohen_kappa_score(labels, preds)
    mcc   = matthews_corrcoef(labels, preds)

    # ── AUC ──
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    per_class_auc = {}
    for i, cname in enumerate(CLASS_NAMES):
        per_class_auc[cname] = roc_auc_score(labels_bin[:, i], probs[:, i])
    macro_auc = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
    weighted_auc = roc_auc_score(labels_bin, probs, average="weighted", multi_class="ovr")

    print(f"\n{'='*55}\n RESULTS — {args.dataset} (Student DeiT-Small)\n{'='*55}")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  Macro F1      : {f1m*100:.2f}%")
    print(f"  Weighted F1   : {f1w*100:.2f}%")
    print(f"  Cohen's Kappa : {kappa:.4f}")
    print(f"  MCC           : {mcc:.4f}")
    print(f"  ---")
    for cname, a in per_class_auc.items():
        print(f"  AUC {cname:<11}: {a:.4f}")
    print(f"  Macro AUC     : {macro_auc:.4f}")
    print(f"  Weighted AUC  : {weighted_auc:.4f}")
    print(f"{'='*55}")

    # -- Save --
    out_txt = os.path.join(args.output_dir, f"student_metrics_{args.dataset}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Dataset: {args.dataset} (Student DeiT-Small)\n\n")
        f.write(f"Accuracy     : {acc*100:.2f}%\n")
        f.write(f"Macro F1     : {f1m*100:.2f}%\n")
        f.write(f"Weighted F1  : {f1w*100:.2f}%\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"MCC          : {mcc:.4f}\n\n")
        for cname, a in per_class_auc.items():
            f.write(f"AUC {cname}: {a:.4f}\n")
        f.write(f"\nMacro AUC   : {macro_auc:.4f}\n")
        f.write(f"Weighted AUC: {weighted_auc:.4f}\n")
    print(f"\nResults: {out_txt}")


if __name__ == "__main__":
    main()
