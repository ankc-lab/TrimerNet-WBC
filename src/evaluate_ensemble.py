"""
TrimerNet-WBC: Weighted Ensemble Evaluation Script
==================================================
Loads trained weights for DeiT, ViT, and Swin models, performs
VALIDATION-WEIGHTED ensemble inference on a test dataset.
Includes CLAHE (V-channel) preprocessing and 384x384 resolution.

Usage:
    python src/evaluate_ensemble.py --data_root ../data/Raabin/test --weights 0.35 0.35 0.30
"""

import os
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DeiTForImageClassification, DeiTConfig, ViTForImageClassification, ViTConfig, SwinForImageClassification, SwinConfig

# Global class definitions
CLASS_LABELS = {
    "Basophil": 0,
    "Eosinophil": 1,
    "Lymphocyte": 2,
    "Monocyte": 3,
    "Neutrophil": 4,
}

# ----------------------------- PREPROCESSING (MATCHING TRAIN) -----------------------------
def process_image(image):
    """Applies CLAHE histogram equalization to the V channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(v)
    himg = cv2.merge((h, s, cl))
    processed_img = cv2.cvtColor(himg, cv2.COLOR_HSV2RGB)

    resized = cv2.resize(processed_img, (384, 384), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

class FilePathDataset(Dataset):
    def __init__(self, data_path):
        self.file_paths = []
        self.labels = []
        
        for folder_name in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder_name)
            if os.path.isdir(folder_path):
                label = CLASS_LABELS.get(folder_name)
                if label is not None:
                    for file_name in os.listdir(folder_path):
                        if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
                            self.file_paths.append(os.path.join(folder_path, file_name))
                            self.labels.append(label)
                            
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_img = process_image(image)
            processed_img = torch.tensor(processed_img, dtype=torch.float32).permute(2, 0, 1)
            return processed_img, torch.tensor(label, dtype=torch.long)
        except Exception:
            return torch.zeros((3, 384, 384), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ----------------------------- ARCHITECTURES -----------------------------
class EnhancedDeiTModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super(EnhancedDeiTModel, self).__init__()
        self.backbone = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        outputs = self.backbone(x, output_hidden_states=True)
        cls_token = outputs.hidden_states[-1][:, 0]
        normalized = self.batch_norm(cls_token)
        return self.classifier(normalized)

class EnhancedViTModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super(EnhancedViTModel, self).__init__()
        self.backbone = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        outputs = self.backbone(x, output_hidden_states=True)
        cls_token = outputs.hidden_states[-1][:, 0]
        normalized = self.batch_norm(cls_token)
        return self.classifier(normalized)

class EnhancedSwinModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super(EnhancedSwinModel, self).__init__()
        self.backbone = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)
        )
    def forward(self, x):
        outputs = self.backbone(x, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = torch.mean(last_hidden_state, dim=1)
        normalized = self.batch_norm(pooled_output)
        return self.classifier(normalized)

# ----------------------------- EVALUATION LOGIC -----------------------------
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")

    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Test data directory not found: {args.data_root}")

    # Load dataset using the custom FilePathDataset (includes CLAHE + 384px)
    test_dataset = FilePathDataset(args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    class_names = list(CLASS_LABELS.keys())
    
    print(f"Loaded {len(test_dataset)} test images (CLAHE + 384px Applied).")

    # Verify and normalize weights
    weights_array = np.array(args.weights)
    weights_array = weights_array / np.sum(weights_array)
    print(f"Ensemble Weights (DeiT, ViT, Swin): {weights_array}")

    # Model Initialization
    print("Initializing architectures...")
    deit_config = DeiTConfig.from_pretrained('facebook/deit-base-distilled-patch16-384', num_labels=5)
    deit_base = DeiTForImageClassification(config=deit_config)
    model_deit = EnhancedDeiTModel(deit_base).to(device)
    
    vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-384', num_labels=5)
    vit_base = ViTForImageClassification(config=vit_config)
    model_vit = EnhancedViTModel(vit_base).to(device)

    swin_config = SwinConfig.from_pretrained('microsoft/swin-base-patch4-window12-384', num_labels=5)
    swin_base = SwinForImageClassification(config=swin_config)
    model_swin = EnhancedSwinModel(swin_base).to(device)

    model_paths = {
        "DeiT": os.path.join(args.model_dir, "DeiT_best_model.pth"),
        "ViT": os.path.join(args.model_dir, "ViT_best_model.pth"),
        "Swin": os.path.join(args.model_dir, "Swin_best_model.pth")
    }

    print("Loading pre-trained weights...")
    for name, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing weights for {name}: {path}")
            
    model_deit.load_state_dict(torch.load(model_paths["DeiT"], map_location=device))
    model_vit.load_state_dict(torch.load(model_paths["ViT"], map_location=device))
    model_swin.load_state_dict(torch.load(model_paths["Swin"], map_location=device))

    model_deit.eval()
    model_vit.eval()
    model_swin.eval()

    # Inference
    all_preds = []
    all_labels = []

    print("Starting validation-weighted ensemble inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits_deit = model_deit(images)
            logits_vit = model_vit(images)
            logits_swin = model_swin(images)
            
            probs_deit = torch.softmax(logits_deit, dim=1)
            probs_vit = torch.softmax(logits_vit, dim=1)
            probs_swin = torch.softmax(logits_swin, dim=1)
            
            # Validation-Weighted Ensemble
            ensemble_probs = (weights_array[0] * probs_deit) + \
                             (weights_array[1] * probs_vit) + \
                             (weights_array[2] * probs_swin)
                             
            _, preds = torch.max(ensemble_probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics and Visualization
    print("\n" + "="*60)
    print("         Weighted Ensemble Classification Report          ")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})

    plt.title('Validation-Weighted Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "ensemble_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TrimerNet-WBC Weighted Ensemble")
    parser.add_argument("--data_root", type=str, required=True, help="Path to test dataset directory")
    parser.add_argument("--model_dir", type=str, default="../checkpoints", help="Directory containing .pth files")
    parser.add_argument("--output_dir", type=str, default="../results", help="Directory to save output plots")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--weights", nargs=3, type=float, required=True, 
                        help="MANDATORY: Validation-calculated weights for DeiT, ViT, and Swin respectively (e.g., 0.35 0.35 0.30)")
    
    args = parser.parse_args()
    evaluate(args)