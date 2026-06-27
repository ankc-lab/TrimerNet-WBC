"""
TrimerNet-WBC — Computational Complexity Analysis
==================================================
Evaluates the model complexity in terms of Parameters, FLOPs, and Inference time.

Evaluated architectures:
  - ResNet50 (CNN baseline)
  - ViT-Base, DeiT-Base, Swin-Base (Single transformer backbones)
  - Teacher Ensemble (Combination of 3 models)
  - Student DeiT-Small (Knowledge distillation target)

USAGE:
    python utils/complexity_analysis.py
    (No checkpoint required — purely measures architectural complexity)

REQUIREMENTS:
    pip install fvcore
"""

import time
import torch
import torch.nn as nn
from transformers import (
    DeiTForImageClassification, DeiTConfig,
    ViTForImageClassification,
    SwinForImageClassification,
)
import torchvision.models as tv_models

# -------------------------------------------------------------
# Dependency Check for FLOPs
# -------------------------------------------------------------
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("WARNING: 'fvcore' library not found. FLOPs measurement will be skipped.")
    print("To install, run: pip install fvcore")


NUM_CLASSES = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------
# Model Architectures (Matching the paper implementation)
# -------------------------------------------------------------
class EnhancedDeiTModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super().__init__()
        self.backbone = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        cls_token = out.hidden_states[-1][:, 0]
        return self.classifier(self.batch_norm(cls_token))


class EnhancedViTModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super().__init__()
        self.backbone = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        cls_token = out.hidden_states[-1][:, 0]
        return self.classifier(self.batch_norm(cls_token))


class EnhancedSwinModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=5):
        super().__init__()
        self.backbone = pretrained_model
        self.batch_norm = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 768),
            nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        pooled = torch.mean(out.hidden_states[-1], dim=1)
        return self.classifier(self.batch_norm(pooled))


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
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        cls_token = out.hidden_states[-1][:, 0]
        return self.classifier(self.batch_norm(cls_token))


# -------------------------------------------------------------
# Measurement Functions
# -------------------------------------------------------------
def count_params(model):
    """Returns total trainable parameters in Millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_flops(model, img_size):
    """Returns GFLOPs for a single image."""
    if not HAS_FVCORE:
        return None
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    try:
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except Exception as e:
        print(f"  [Error] Failed to measure FLOPs: {e}")
        return None


def measure_inference(model, img_size, batch_size=32, n_runs=100):
    """Returns average inference latency per batch in milliseconds."""
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size).to(device)
    with torch.no_grad():
        # Warm-up phase
        for _ in range(10):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        # Measurement phase
        t0 = time.time()
        for _ in range(n_runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        
    return (t1 - t0) / n_runs * 1000


def analyze(name, model, img_size):
    """Executes the full measurement pipeline for a given model."""
    model = model.to(device)
    params = count_params(model)
    flops = measure_flops(model, img_size)
    inf = measure_inference(model, img_size)
    
    flops_str = f"{flops:.2f}" if flops is not None else "N/A"
    print(f"{name:<28} {params:>8.1f}  {flops_str:>8}  {inf:>8.1f}  {img_size:>6}")
    
    del model
    torch.cuda.empty_cache()
    return name, params, flops, inf


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
def main():
    print(f"\n{'='*75}")
    print(f" Computational Complexity Analysis  (Device: {device})")
    print(f"{'='*75}")
    print(f"{'Model':<28} {'Params(M)':>8}  {'GFLOPs':>8}  {'Inf(ms)':>8}  {'Input':>6}")
    print(f"{'-'*75}")

    results = []

    # -- 1. CNN baseline: ResNet50 --
    resnet = tv_models.resnet50(weights=None)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES),
    )
    results.append(analyze("ResNet50 (CNN baseline)", resnet, 224))

    # -- 2. Single Transformers (384px) --
    vit_base = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    results.append(analyze("ViT-Base", EnhancedViTModel(vit_base), 384))

    deit_base = DeiTForImageClassification.from_pretrained(
        'facebook/deit-base-distilled-patch16-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    results.append(analyze("DeiT-Base", EnhancedDeiTModel(deit_base), 384))

    swin_base = SwinForImageClassification.from_pretrained(
        'microsoft/swin-base-patch4-window12-384',
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    results.append(analyze("Swin-Base", EnhancedSwinModel(swin_base), 384))

    # -- 3. Student DeiT-Small (224px) --
    results.append(analyze("Student DeiT-Small (Ours)", EnhancedDeiTSmallModel(), 224))

    print(f"{'-'*75}")

    # -- 4. Teacher Ensemble (Sum of 3 models) --
    vit_p, deit_p, swin_p = results[1][1], results[2][1], results[3][1]
    vit_f, deit_f, swin_f = results[1][2] or 0, results[2][2] or 0, results[3][2] or 0
    vit_i, deit_i, swin_i = results[1][3], results[2][3], results[3][3]

    ens_params = vit_p + deit_p + swin_p
    ens_flops = vit_f + deit_f + swin_f
    ens_inf = vit_i + deit_i + swin_i
    
    flops_str = f"{ens_flops:.2f}" if ens_flops else "N/A"
    print(f"{'Teacher Ensemble (3 models)':<28} {ens_params:>8.1f}  "
          f"{flops_str:>8}  {ens_inf:>8.1f}  {'384':>6}")

    print(f"{'='*75}")

    # -- 5. Executive Summary --
    student_p = results[4][1]
    student_i = results[4][3]
    
    print(f"\n SUMMARY (Distillation Efficiency):")
    print(f"  Teacher Ensemble   : {ens_params:.1f}M params, {ens_inf:.1f}ms latency")
    print(f"  Student DeiT-Small : {student_p:.1f}M params, {student_i:.1f}ms latency")
    print(f"  Parameter Reduction: {ens_params/student_p:.1f}x smaller")
    print(f"  Inference Speedup  : {ens_inf/student_i:.1f}x faster")
    print(f"{'='*75}")

    # -- 6. Export Results --
    output_file = "complexity_results.csv"
    with open(output_file, "w") as f:
        f.write("Model,Params(M),GFLOPs,Inference(ms),Input_Size\n")
        for name, p, fl, inf in results:
            fl_s = f"{fl:.2f}" if fl is not None else "N/A"
            f.write(f"{name},{p:.1f},{fl_s},{inf:.1f},384\n")
        f.write(f"Teacher Ensemble,{ens_params:.1f},{flops_str},{ens_inf:.1f},384\n")
        
    print(f"\n[Success] Results successfully exported to {output_file}")


if __name__ == "__main__":
    main()