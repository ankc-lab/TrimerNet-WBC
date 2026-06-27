"""
================================================================================
TrimerNet-WBC — 10-Fold Cross-Validation for Statistical Significance
================================================================================
Self-contained script: trains ViT, DeiT, and Swin from scratch in each fold and
evaluates the validation-weighted ensemble, to test whether the ensemble's
improvement over the individual backbones is statistically significant.


Method:
  - Raabin-WBC only. Train + Validation are combined; the independent TEST set
    is excluded entirely.
  - StratifiedKFold(n_splits=10) preserves the class ratio in each fold.
  - In each fold, DeiT, ViT, and Swin are trained FROM SCRATCH on that fold's
    training portion; ensemble weights are computed from the held-out fold; all models are evaluated on the held-out fold.

Crash protection (RESUME):
  - After each model/fold, the state is written atomically to a JSON file.
  - On restart, completed folds/models are skipped.

Usage:
    python cv10_significance.py \
        --train_path /path/to/Raabin/train \
        --val_path   /path/to/Raabin/validation \
        --output_dir /path/to/cv10_results \
        --n_folds 10 --epochs 50 --patience 5 --batch_size 32 --lr 5e-5
================================================================================
"""

import os
import gc
import json
import time
import random
import copy
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from transformers import (DeiTForImageClassification, DeiTConfig,
                          ViTForImageClassification, ViTConfig,
                          SwinForImageClassification, SwinConfig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
#  PART 1 — CORE MODEL / TRAINING FUNCTIONS
# ==============================================================================

# device is already defined above
print(f"Device: {device}")

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Class labels
class_labels = {
    "Basophil": 0,
    "Eosinophil": 1,
    "Lymphocyte": 2,
    "Monocyte": 3,
    "Neutrophil": 4,
}

# Image processing function
def process_image(image):
    """
    Process an image with CLAHE histogram equalization
    
    Parameters:
    - image: Input RGB image
    
    Returns:
    - Processed image tensor
    """
    # CLAHE histogram equalization (applied only to V channel)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(v)
    himg = cv2.merge((h, s, cl))
    processed_img = cv2.cvtColor(himg, cv2.COLOR_HSV2RGB)

    # Resize and normalize
    resized = cv2.resize(processed_img, (384, 384), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    
    return normalized

class FilePathDataset(Dataset):
    def __init__(self, data_path):
        self.file_paths = []
        self.labels = []
        
        # Walk through data directories and collect file paths and labels
        for folder_name in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder_name)
            if os.path.isdir(folder_path):
                # Check class label
                label = class_labels.get(folder_name)
                if label is not None:
                    # Collect all jpg files in this folder
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith(".jpg"):
                            file_path = os.path.join(folder_path, file_name)
                            self.file_paths.append(file_path)
                            self.labels.append(label)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Image could not be loaded: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use process_image function
            processed_img = process_image(image)
            
            # Convert to tensor
            processed_img = torch.tensor(processed_img, dtype=torch.float32).permute(2, 0, 1)  # (3, 384, 384)
            return processed_img, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image in case of error
            dummy_img = torch.zeros((3, 384, 384), dtype=torch.float32)
            return dummy_img, torch.tensor(label, dtype=torch.long)

# Custom model class with Batch Normalization for DeiT
class EnhancedDeiTModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super(EnhancedDeiTModel, self).__init__()
        self.backbone = pretrained_model
        
        # Add batch normalization to stabilize training
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        
        # Replace the classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the classifier weights
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Extract features from the backbone
        outputs = self.backbone(x, output_hidden_states=True)
        
        # Get the [CLS] token representation
        cls_token = outputs.hidden_states[-1][:, 0]
        
        # Apply batch normalization
        normalized = self.batch_norm(cls_token)
        
        # Forward through classifier
        logits = self.classifier(normalized)
        
        return logits

# Custom model class with Batch Normalization for ViT
class EnhancedViTModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super(EnhancedViTModel, self).__init__()
        self.backbone = pretrained_model
        
        # Add batch normalization to stabilize training
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        
        # Replace the classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the classifier weights
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Extract features from the backbone
        outputs = self.backbone(x, output_hidden_states=True)
        
        # Get the [CLS] token representation
        cls_token = outputs.hidden_states[-1][:, 0]
        
        # Apply batch normalization
        normalized = self.batch_norm(cls_token)
        
        # Forward through classifier
        logits = self.classifier(normalized)
        
        return logits

# Custom model class with Batch Normalization for Swin Transformer
class EnhancedSwinModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super(EnhancedSwinModel, self).__init__()
        self.backbone = pretrained_model
        
        # Add batch normalization to stabilize training
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        
        # Replace the classification head with a more complex one
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)

        )
        
        # Initialize the classifier weights
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
            # Extract features from the backbone
            outputs = self.backbone(x, output_hidden_states=True)
            
            # Get the correct output for Swin
            last_hidden_state = outputs.hidden_states[-1]
            # Apply global average pooling
            pooled_output = torch.mean(last_hidden_state, dim=1)
            
            # Apply batch normalization
            normalized = self.batch_norm(pooled_output)
            
            # Forward through classifier
            logits = self.classifier(normalized)
            
            return logits

# Create class-weighted loss function
def get_weighted_loss():
    # Class counts from training set
    counts = np.array([211, 490, 2212, 364, 4900])
    weights = 1.0 / (counts / np.sum(counts))
    class_weights = torch.FloatTensor(weights).to(device)
    return nn.CrossEntropyLoss(weight=class_weights)

# Create ensemble models
def create_ensemble_models():
    # DeiT model
    deit_config = DeiTConfig.from_pretrained('facebook/deit-base-distilled-patch16-384', num_labels=5)
    deit_base = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-384', 
        config=deit_config,
        ignore_mismatched_sizes=True
    )
    deit_model = EnhancedDeiTModel(deit_base)
    
    # ViT model
    vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-384', num_labels=5)
    vit_base = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-384', 
        config=vit_config,
        ignore_mismatched_sizes=True
    )
    vit_model = EnhancedViTModel(vit_base)

    # Swin Transformer model
    swin_config = SwinConfig.from_pretrained('microsoft/swin-base-patch4-window12-384', num_labels=5)
    swin_base = SwinForImageClassification.from_pretrained(
        'microsoft/swin-base-patch4-window12-384', 
        config=swin_config,
        ignore_mismatched_sizes=True
    )
    swin_model = EnhancedSwinModel(swin_base)
    
    return [deit_model.to(device), vit_model.to(device), swin_model.to(device)]

# Model evaluation function
def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            # Handle Hugging Face model outputs
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            running_loss += loss.item()

            _, preds = torch.max(logits, 1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    accuracy = correct_preds / total_preds
    return val_loss, accuracy, all_preds, all_labels

# Print function for trainable parameters
def print_trainable_params(model):
    """Print statistics about trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

# Improved progressive unfreezing function
def unfreeze_layers_progressively(model, epoch, total_epochs):
    progress = epoch / total_epochs
    
    # Initial epoch - Only train classifier layers
    if epoch == 0:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze only the classifier
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        # For Hugging Face models
        elif hasattr(model, 'classifier') and hasattr(model.classifier, 'parameters'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        # For alternative architectures
        elif hasattr(model, 'head') and hasattr(model.head, 'parameters'):
            for param in model.head.parameters():
                param.requires_grad = True
        
        print_trainable_params(model)
        return
    
    # More gradual unfreezing (4 stages)
    elif progress >= 0.25 and progress < 0.5:
        # Unfreeze last 10% of layers
        backbone_params = []
        if hasattr(model, 'backbone'):
            backbone_params = list(model.backbone.parameters())
        else:
            # If model is directly a backbone
            backbone_params = list(model.parameters())
        
        # Calculate total parameter count
        total_params = len(backbone_params)
        
        # Unfreeze the last 10%
        unfreeze_idx = int(total_params * 0.9)
        for i, param in enumerate(reversed(backbone_params)):
            if i < total_params - unfreeze_idx:
                param.requires_grad = True
    
    elif progress >= 0.5 and progress < 0.75:
        # Unfreeze last 25% of layers
        backbone_params = []
        if hasattr(model, 'backbone'):
            backbone_params = list(model.backbone.parameters())
        else:
            backbone_params = list(model.parameters())
        
        total_params = len(backbone_params)
        
        # Unfreeze the last 25%
        unfreeze_idx = int(total_params * 0.75)
        for i, param in enumerate(reversed(backbone_params)):
            if i < total_params - unfreeze_idx:
                param.requires_grad = True
    
    elif progress >= 0.75:
        # Unfreeze last 50% of layers
        backbone_params = []
        if hasattr(model, 'backbone'):
            backbone_params = list(model.backbone.parameters())
        else:
            backbone_params = list(model.parameters())
        
        total_params = len(backbone_params)
        
        # Unfreeze the last 50%
        unfreeze_idx = int(total_params * 0.5)
        for i, param in enumerate(reversed(backbone_params)):
            if i < total_params - unfreeze_idx:
                param.requires_grad = True
                
    print_trainable_params(model)

# Training function with progressive unfreezing
def train_model_with_progressive_unfreezing(model, train_loader, val_loader, epochs, lr, patience, save_path):
    """Training with progressive unfreezing and ReduceLROnPlateau learning rate scheduling"""
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=0.001  # Reduced from 0.01
    )
    
    # Use ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # Try to minimize validation loss
        factor=0.5,          # Multiply learning rate by 0.2 when reducing
        patience=2,          # Wait 3 epochs before reducing
        min_lr=lr/2000       # Minimum learning rate
    )
    
    criterion = get_weighted_loss()

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    for epoch in range(epochs):
        start_time = time.time()
        
        # Apply progressive unfreezing
        unfreeze_layers_progressively(model, epoch, epochs)
        
        # Update optimizer to include newly unfrozen parameters
        if epoch > 0:
            # Get current learning rate
            last_lr = optimizer.param_groups[0]['lr']
            
            # Update optimizer
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=last_lr,  # Use current learning rate
                weight_decay=0.001
            )
            
            # Create new scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                min_lr=lr/2000
            )
        
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle Hugging Face model outputs
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation
        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Step scheduler based on validation loss
        scheduler.step(val_loss)

        # Save current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        end_time = time.time()
        epoch_duration = end_time - start_time
        estimated_remaining_time = epoch_duration * (epochs - epoch - 1)
        estimated_remaining_minutes = estimated_remaining_time / 60

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {epoch_loss:.4f}, Train accuracy: {epoch_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")
        print(f"Estimated time remaining: {estimated_remaining_minutes:.2f} minutes")
        print(f"Epoch {epoch+1}, Learning Rate: {current_lr:.8f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model improved, saved to {save_path}!")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
                
        print("Performing memory cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Memory cleaned after epoch {epoch+1}")

    return train_losses, val_losses, train_accuracies, val_accuracies, learning_rates

# Calculate weights for ensemble models based on validation performance
def calculate_ensemble_weights(models, val_loader):
    individual_accuracies = []
    criterion = get_weighted_loss()
    
    for model in models:
        _, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion)
        individual_accuracies.append(val_accuracy)
    
    # Normalize accuracies to get weights
    total_accuracy = sum(individual_accuracies)
    weights = [acc/total_accuracy for acc in individual_accuracies]
    
    print("Ensemble model weights based on validation accuracy:")
    for i, weight in enumerate(weights):
        print(f"Model {i+1}: {weight:.4f}")
        
    return weights

# Ensemble prediction function
def ensemble_predict(models, test_loader, weights=None):
    # Normalize weights
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    else:
        weights = np.array(weights) / np.sum(weights)
    
    all_probabilities = []
    all_labels = []
    
    # For each example, get predictions from each model
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            batch_probs = np.zeros((inputs.size(0), len(class_labels)))
            
            # Add weighted contribution from each model
            for i, model in enumerate(models):
                model.eval()
                outputs = model(inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                batch_probs += weights[i] * probs
            
            all_probabilities.append(batch_probs)
            all_labels.extend(labels.cpu().numpy())
    
    # Combine and get predictions
    all_probabilities = np.vstack(all_probabilities)
    predictions = np.argmax(all_probabilities, axis=1)
    
    return predictions, all_probabilities, np.array(all_labels)




# ==============================================================================
#  PART 2 — 10-FOLD CROSS-VALIDATION DRIVER
# ==============================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", required=True)
    p.add_argument("--val_path",   required=True)
    p.add_argument("--output_dir", default="./cv10_results")
    p.add_argument("--n_folds",    type=int, default=10)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--patience",   type=int, default=7)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 3) RESUME HELPERS (crash protection)
# ──────────────────────────────────────────────────────────────────────────────
def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"folds": {}}   # {"folds": {"1": {"DeiT":acc,"ViT":acc,"Swin":acc,"Ensemble":acc}, ...}}

def save_state(state, state_path):
    tmp = state_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)   # atomic write (protects against mid-write crash)


# ──────────────────────────────────────────────────────────────────────────────
# 4) Subset -> DataLoader (using StratifiedKFold indices)
# ──────────────────────────────────────────────────────────────────────────────
def make_loader(dataset, indices, batch_size, num_workers, shuffle):
    sub = Subset(dataset, indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
# 5) Train a single model and evaluate on the held-out fold
# ──────────────────────────────────────────────────────────────────────────────
def train_and_eval_one(model_name, model, train_loader, val_loader,
                       epochs, lr, patience, ckpt_path):
    """Uses the train_model_with_progressive_unfreezing function."""
    print(f"\n    >>> Training {model_name} (will be evaluated on the held-out fold)...")
    train_model_with_progressive_unfreezing(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, patience=patience, save_path=ckpt_path
    )
    # Load the best (min val loss) weights — the training function saves the checkpoint
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    # Held-out fold'da accuracy
    criterion = get_weighted_loss()
    _, acc, _, _ = evaluate_model(model, val_loader, criterion)
    print(f"    >>> {model_name} held-out accuracy: {acc*100:.2f}%")
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# 6) MAIN 10-FOLD LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, "cv10_state.json")
    state = load_state(state_path)

    print("="*70)
    print(" TrimerNet-WBC  10-Fold CV — Statistical Significance (R1-8)")
    print(f" Folds: {args.n_folds} | Epochs: {args.epochs} | Device: {device}")
    print("="*70)

    # -- Combine Train + Validation (TEST EXCLUDED) --
    print("\n[1] Loading Train + Validation (TEST IS NOT USED)...")
    train_ds = FilePathDataset(args.train_path)
    val_ds   = FilePathDataset(args.val_path)
    combined = ConcatDataset([train_ds, val_ds])
    all_labels = np.array(train_ds.labels + val_ds.labels)
    print(f"    Combined (train+val): {len(combined)} images")
    print(f"    Class distribution: {np.bincount(all_labels)}")

    # ── StratifiedKFold ──
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.zeros(len(combined)), all_labels))

    # -- Fold loop --
    for fold_idx, (train_index, val_index) in enumerate(splits):
        fold_key = str(fold_idx + 1)

        # RESUME: skip this fold if it is fully completed
        if fold_key in state["folds"] and \
           all(m in state["folds"][fold_key] for m in ["DeiT", "ViT", "Swin", "Ensemble"]):
            print(f"\n[Fold {fold_key}] already completed, skipping.")
            continue

        print(f"\n{'='*70}\n[Fold {fold_key}/{args.n_folds}]")
        print(f"  Train: {len(train_index)} | Held-out (val): {len(val_index)}")
        # Held-out class distribution
        ho_labels = all_labels[val_index]
        print(f"  Held-out class distribution: {np.bincount(ho_labels)}")

        state["folds"].setdefault(fold_key, {})

        train_loader = make_loader(combined, train_index, args.batch_size, args.num_workers, shuffle=True)
        val_loader   = make_loader(combined, val_index,   args.batch_size, args.num_workers, shuffle=False)

        # Build the 3 models from scratch for this fold (create_ensemble_models -> [DeiT, ViT, Swin])
        # Note: create_ensemble_models builds a NEW model from pretrained weights on each call (from scratch).
        models = create_ensemble_models()   # [deit, vit, swin]
        model_names = ["DeiT", "ViT", "Swin"]

        trained_models = {}

        # -- Train each model + evaluate on held-out (per-model resume) --
        for name, model in zip(model_names, models):
            ckpt_path = os.path.join(args.output_dir, f"fold{fold_key}_{name}_best.pth")

            if name in state["folds"][fold_key]:
                # This model was already evaluated in this fold -> load from ckpt (needed for the ensemble)
                print(f"  [{name}] already completed (acc={state['folds'][fold_key][name]:.4f}), loading ckpt.")
                if os.path.exists(ckpt_path):
                    model.load_state_dict(torch.load(ckpt_path, map_location=device))
                trained_models[name] = model
                continue

            acc = train_and_eval_one(name, model, train_loader, val_loader,
                                     args.epochs, args.lr, args.patience, ckpt_path)
            state["folds"][fold_key][name] = acc
            save_state(state, state_path)   # save after each model completes (crash protection)
            trained_models[name] = model

        # -- Ensemble: weights from held-out (Eq.4), evaluated on held-out --
        if "Ensemble" not in state["folds"][fold_key]:
            print("  >>> Computing ensemble weights (held-out validation, Eq.4)...")
            # calculate_ensemble_weights model ORDER: [DeiT, ViT, Swin] (same as notebook)
            ordered = [trained_models["DeiT"], trained_models["ViT"], trained_models["Swin"]]
            weights = calculate_ensemble_weights(ordered, val_loader)

            # Ensemble accuracy (held-out) — using the ensemble_predict function.
            # ensemble_predict(models, loader, weights) -> (predictions, probs, labels)
            # NOTE: labels are collected from the loader (compatible with Subset), NOT test_dataset.labels.
            preds, _, labels_arr = ensemble_predict(ordered, val_loader, weights=weights)
            ens_acc = float(np.mean(preds == labels_arr))
            print(f"  >>> Ensemble held-out accuracy: {ens_acc*100:.2f}%")
            state["folds"][fold_key]["Ensemble"] = ens_acc
            state["folds"][fold_key]["weights"]  = weights
            save_state(state, state_path)

        # -- Memory cleanup --
        del models, trained_models
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[Fold {fold_key}] TAMAMLANDI: {state['folds'][fold_key]}")

    # -- Summary --
    print(f"\n{'='*70}\n 10-FOLD CV TAMAMLANDI\n{'='*70}")
    print_summary(state)
    print(f"\nResults: {state_path}")
    print("Pass this JSON to the significance test script for statistical testing.")

# ──────────────────────────────────────────────────────────────────────────────
# 7) Print summary
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(state):
    rows = {"DeiT": [], "ViT": [], "Swin": [], "Ensemble": []}
    for fk in sorted(state["folds"], key=lambda x: int(x)):
        f = state["folds"][fk]
        for m in rows:
            if m in f:
                rows[m].append(f[m] * 100)
    print(f"\n{'Model':<10} | " + " | ".join(f"F{i+1}" for i in range(len(rows['DeiT']))) + " | Mean±Std")
    for m, vals in rows.items():
        if vals:
            mean, std = np.mean(vals), np.std(vals)
            vstr = " ".join(f"{v:.2f}" for v in vals)
            print(f"{m:<10} | {vstr} | {mean:.2f}±{std:.2f}")


if __name__ == "__main__":
    main()
