"""
TrimerNet-WBC: Ensemble Training Script
=======================================
Trains an ensemble of DeiT, ViT, and Swin Transformers with:
- V-channel CLAHE preprocessing
- Progressive Unfreezing
- Dynamically calculated class-weighted CrossEntropyLoss

Usage:
    python src/train_ensemble.py --data_root ./data/Raabin --epochs 30 --batch_size 32
"""

import os
import gc
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DeiTForImageClassification, DeiTConfig, ViTForImageClassification, ViTConfig, SwinForImageClassification, SwinConfig

# ----------------------------- SETUP -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")

# Global class definitions
CLASS_LABELS = {
    "Basophil": 0,
    "Eosinophil": 1,
    "Lymphocyte": 2,
    "Monocyte": 3,
    "Neutrophil": 4,
}

# ----------------------------- DATA PROCESSING -----------------------------
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
        self.class_counts = {k: 0 for k in CLASS_LABELS.keys()}
        
        for folder_name in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder_name)
            if os.path.isdir(folder_path):
                label = CLASS_LABELS.get(folder_name)
                if label is not None:
                    for file_name in os.listdir(folder_path):
                        if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
                            self.file_paths.append(os.path.join(folder_path, file_name))
                            self.labels.append(label)
                            self.class_counts[folder_name] += 1
                            
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

def get_dynamic_weighted_loss(dataset):
    """Calculates inverse frequency weights based on current dataset distribution."""
    counts = np.array([dataset.class_counts[k] for k in CLASS_LABELS.keys()])
    weights = 1.0 / (counts / np.sum(counts))
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Dynamic Class Weights: {class_weights.cpu().numpy()}")
    return nn.CrossEntropyLoss(weight=class_weights)

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
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
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
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
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
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        outputs = self.backbone(x, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = torch.mean(last_hidden_state, dim=1)
        normalized = self.batch_norm(pooled_output)
        return self.classifier(normalized)

def create_ensemble_models():
    deit_config = DeiTConfig.from_pretrained('facebook/deit-base-distilled-patch16-384', num_labels=5)
    deit_base = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-384', config=deit_config, ignore_mismatched_sizes=True)
    deit_model = EnhancedDeiTModel(deit_base)
    
    vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-384', num_labels=5)
    vit_base = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384', config=vit_config, ignore_mismatched_sizes=True)
    vit_model = EnhancedViTModel(vit_base)

    swin_config = SwinConfig.from_pretrained('microsoft/swin-base-patch4-window12-384', num_labels=5)
    swin_base = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window12-384', config=swin_config, ignore_mismatched_sizes=True)
    swin_model = EnhancedSwinModel(swin_base)
    
    return [deit_model.to(device), vit_model.to(device), swin_model.to(device)]

# ----------------------------- TRAINING LOGIC -----------------------------
def unfreeze_layers_progressively(model, epoch, total_epochs):
    progress = epoch / total_epochs
    
    if epoch == 0:
        for param in model.parameters(): param.requires_grad = False
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters(): param.requires_grad = True
        return
    
    backbone_params = list(model.backbone.parameters()) if hasattr(model, 'backbone') else list(model.parameters())
    total_params = len(backbone_params)
    
    unfreeze_idx = 0
    if 0.25 <= progress < 0.5: unfreeze_idx = int(total_params * 0.9)
    elif 0.5 <= progress < 0.75: unfreeze_idx = int(total_params * 0.75)
    elif progress >= 0.75: unfreeze_idx = int(total_params * 0.5)
        
    for i, param in enumerate(reversed(backbone_params)):
        if i < total_params - unfreeze_idx:
            param.requires_grad = True

def train_single_model(model, model_name, train_loader, val_loader, criterion, args, save_dir):
    print(f"\n--- Training {model_name} ---")
    save_path = os.path.join(save_dir, f"{model_name}_best_model.pth")
    
    # Initial state: backbone frozen, classifier trainable
    for param in model.parameters(): param.requires_grad = False
    for param in model.classifier.parameters(): param.requires_grad = True
        
    # FIX: pass all parameters to the optimizer at once.
    # PyTorch automatically skips those with requires_grad=False (since their grad is None).
    # This way the Adam state and the scheduler's patience counter are NEVER reset!
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=args.lr/2000)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        # Layers are unfrozen progressively; the optimizer detects the gradients automatically
        unfreeze_layers_progressively(model, epoch, args.epochs)

        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # The scheduler works normally and reduces the LR
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved improved model to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("--> Early stopping triggered.")
                break
                
    return model
# ----------------------------- MAIN ENTRY -----------------------------
def main(args):
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "validation")
    
    print("Loading datasets...")
    train_dataset = FilePathDataset(train_dir)
    val_dataset = FilePathDataset(val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # Calculate dynamic loss function
    criterion = get_dynamic_weighted_loss(train_dataset)
    
    models = create_ensemble_models()
    model_names = ["DeiT", "ViT", "Swin"]
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for i, model in enumerate(models):
        train_single_model(model, model_names[i], train_loader, val_loader, criterion, args, args.save_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ----------------------------- WEIGHT CALCULATION -----------------------------
    print("\n" + "="*60)
    print("Calculating Optimal Validation-Weighted Ensemble Ratios...")
    print("="*60)
    
    val_accuracies = []
    
    # Load the best models from disk and evaluate on the validation set
    with torch.no_grad():
        for i, model in enumerate(models):
            model_name = model_names[i]
            best_path = os.path.join(args.save_dir, f"{model_name}_best_model.pth")
            
            # Load the saved best weights
            model.load_state_dict(torch.load(best_path, map_location=device))
            model.eval()
            
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            acc = correct / total
            val_accuracies.append(acc)
            print(f"{model_name} Best Validation Accuracy: {acc:.4f}")

    # Convert accuracy values into ensemble weights
    val_accuracies = np.array(val_accuracies)
    optimal_weights = val_accuracies / np.sum(val_accuracies)
    
    print("\nOptimal Ensemble Weights (DeiT, ViT, Swin):")
    weights_str = " ".join([f"{w:.4f}" for w in optimal_weights])
    print(f"[{weights_str}]")
    
    print("\n" + "="*60)
    print("Training completely finished! Use the following command for evaluation:")
    print(f"python src/evaluate_ensemble.py --data_root {os.path.join(args.data_root, 'test')} --weights {weights_str}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ensemble Models for TrimerNet-WBC")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root (must contain train/ and validation/ folders)")
    parser.add_argument("--save_dir", type=str, default="../checkpoints", help="Directory to save trained model weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()
    
    main(args)