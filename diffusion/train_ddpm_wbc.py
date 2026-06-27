"""
DDPM Training for WBC Synthetic Image Generation
=================================================
Trains separate DDPMs for each class (e.g., Basophil, Monocyte).
Trained ONLY on real images from the train directory.

Usage:
    python train_ddpm_wbc.py --class_name Basophil --batch_size 8 --epochs 400
    python train_ddpm_wbc.py --class_name Monocyte --batch_size 8 --epochs 400

Requirements: diffusers, accelerate, torch, torchvision
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm


# ----------------------------- CONFIG -----------------------------
class Config:
    def __init__(self, args):
        self.image_size = args.image_size
        self.train_batch_size = args.batch_size
        self.num_epochs = args.epochs
        self.learning_rate = args.lr
        self.gradient_accumulation_steps = 1
        self.lr_warmup_steps = 500
        self.num_train_timesteps = 1000
        self.save_image_epochs = 25        # Save sample images every 25 epochs
        self.save_model_epochs = 25        # Save model checkpoint every 25 epochs
        self.mixed_precision = "fp16"      # Memory efficiency and speed
        self.seed = 42
        self.sample_during_train = False   # Set to True only for debugging
        self.keep_epoch_checkpoints = True # Keep separate epoch checkpoints for FID comparison


# ----------------------------- DATASET -----------------------------
class WBCDataset(Dataset):
    """Loads training images for a single class with mild augmentation."""
    def __init__(self, image_dir, image_size):
        self.paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        if len(self.paths) == 0:
            raise ValueError(f"No images found in: {image_dir}")
        print(f"  Loaded {len(self.paths)} images from: {image_dir}")

        # Augmentation: Kept mild to preserve cell morphology and composition
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ----------------------------- MODEL -----------------------------
def build_unet(image_size):
    """UNet2DModel architecture for WBC generation."""
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )


# ----------------------------- SAMPLE -----------------------------
@torch.no_grad()
def sample_images(model, n_images, image_size, device, batch=16, num_steps=30):
    """Fast sampling using DDIM for validation checks during training."""
    model.eval()
    ddim = DDIMScheduler(num_train_timesteps=1000)
    ddim.set_timesteps(num_steps)
    
    all_imgs = []
    remaining = n_images
    while remaining > 0:
        b = min(batch, remaining)
        x = torch.randn(b, 3, image_size, image_size, device=device)
        for t in ddim.timesteps:
            noise_pred = model(x, t).sample
            x = ddim.step(noise_pred, t, x).prev_sample
        x = (x / 2 + 0.5).clamp(0, 1)
        all_imgs.append(x.cpu())
        remaining -= b
        
    model.train()
    return torch.cat(all_imgs, dim=0)


def save_grid(images, path, nrow=4):
    from torchvision.utils import make_grid, save_image
    grid = make_grid(images, nrow=nrow)
    save_image(grid, path)


# ----------------------------- TRAIN -----------------------------
def train(cfg, class_name, data_root, output_root, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    train_dir = os.path.join(data_root, "train", class_name)
    out_dir = os.path.join(output_root, class_name)
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DDPM Training: {class_name}")
    print(f"{'='*60}")
    print(f"Data Root: {train_dir}")
    print(f"Output Root: {out_dir}")
    print(f"Device: {device}")

    dataset = WBCDataset(train_dir, cfg.image_size)
    loader = DataLoader(
        dataset, batch_size=cfg.train_batch_size,
        shuffle=True, num_workers=4, drop_last=True
    )

    scheduler = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)

    # RESUME: Load checkpoint if exists
    ckpt_path = os.path.join(out_dir, "model")
    start_epoch = 0
    if resume and os.path.isdir(ckpt_path):
        print(f"  -> RESUME: Loading model from: {ckpt_path}")
        model = UNet2DModel.from_pretrained(ckpt_path).to(device)
        epoch_file = os.path.join(out_dir, "last_epoch.txt")
        if os.path.isfile(epoch_file):
            with open(epoch_file) as f:
                start_epoch = int(f.read().strip())
            print(f"  -> Resuming from epoch {start_epoch}")
    else:
        model = build_unet(cfg.image_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=len(loader) * cfg.num_epochs,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.mixed_precision == "fp16"))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model Parameters: {n_params:.1f}M")
    print(f"Batches per epoch: {len(loader)}, Target epochs: {cfg.num_epochs}")
    print(f"Starting epoch: {start_epoch}\n")

    # Loss logging
    loss_log_path = os.path.join(out_dir, "loss_log.csv")
    if start_epoch == 0:
        with open(loss_log_path, "w") as f:
            f.write("epoch,avg_loss\n")

    global_step = start_epoch * len(loader)
    for epoch in range(start_epoch, cfg.num_epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        epoch_loss = 0.0
        
        for clean_images in progress:
            clean_images = clean_images.to(device)
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()
            noisy = scheduler.add_noise(clean_images, noise, timesteps)

            with torch.amp.autocast('cuda', enabled=(cfg.mixed_precision == "fp16")):
                noise_pred = model(noisy, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            progress.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save loss to log
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f}\n")

        # Periodic sampling
        if cfg.sample_during_train and ((epoch + 1) % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1):
            try:
                imgs = sample_images(model, 8, cfg.image_size, device, batch=8)
                save_grid(imgs, os.path.join(sample_dir, f"epoch_{epoch+1:04d}.png"))
                print(f"  -> Sample saved: epoch_{epoch+1:04d}.png")
            except RuntimeError as e:
                print(f"  -> Sampling skipped (Memory Error): {e}")
                torch.cuda.empty_cache()

        # Save checkpoints
        if (epoch + 1) % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            ckpt = os.path.join(out_dir, "model")
            model.save_pretrained(ckpt)
            with open(os.path.join(out_dir, "last_epoch.txt"), "w") as f:
                f.write(str(epoch + 1))
            print(f"  -> Main checkpoint saved: {ckpt} (epoch {epoch+1})")

            if cfg.keep_epoch_checkpoints:
                epoch_ckpt = os.path.join(out_dir, "checkpoints", f"model_epoch_{epoch+1:04d}")
                model.save_pretrained(epoch_ckpt)
                print(f"  -> Epoch checkpoint saved: {epoch_ckpt}")

    print(f"\nTraining completed for {class_name}.")
    print(f"Final model path: {os.path.join(out_dir, 'model')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM for WBC Synthetic Image Generation")
    parser.add_argument("--class_name", type=str, required=True,
                        help="Target class for training (e.g., Basophil, Monocyte)")
    parser.add_argument("--data_root", type=str,
                        default="./data/Raabin",
                        help="Root directory containing train/ folder")
    parser.add_argument("--output_root", type=str,
                        default="./checkpoints/diffusion_models",
                        help="Root directory for saving models")
    parser.add_argument("--epochs", type=int, default=400,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()

    cfg = Config(args)
    train(cfg, args.class_name, args.data_root, args.output_root, resume=args.resume)