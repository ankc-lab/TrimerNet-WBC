"""
DDPM Synthetic Image Generation (Fast DDIM Sampling)
=======================================================
Loads a trained DDPM model and generates synthetic WBC images
using DDIM for fast sampling.

Usage:
    # Quick preview (16-image grid) to check quality:
    python diffusion/generate_ddpm_wbc.py --class_name Basophil --preview

    # Full generation for dataset augmentation:
    python diffusion/generate_ddpm_wbc.py --class_name Basophil --n_images 549
    python diffusion/generate_ddpm_wbc.py --class_name Monocyte --n_images 55

Output: <output_root>/<class>/generated/ (Individual PNG files)
"""

import os
import argparse
import torch
from diffusers import UNet2DModel, DDIMScheduler
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import ImageEnhance


def enhance_color(tensor_img, saturation=1.0, contrast=1.0):
    """
    Applies post-processing (saturation and contrast) to a single [0,1] tensor image.
    This helps mitigate the common color fading issue in DDPM generated images,
    bringing them closer to real microscopic images.
    saturation=1.0, contrast=1.0 -> no change.
    """
    pil = to_pil_image(tensor_img.cpu().clamp(0, 1))
    if saturation != 1.0:
        pil = ImageEnhance.Color(pil).enhance(saturation)
    if contrast != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(contrast)
    return pil


@torch.no_grad()
def generate(class_name, model_root, output_root, n_images,
             image_size=256, num_steps=30, batch=16, preview=False, seed=42,
             epoch=None, saturation=1.0, contrast=1.0):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # Use a specific epoch checkpoint if provided, otherwise use the main model
    if epoch is not None:
        model_path = os.path.join(model_root, class_name, "checkpoints", f"model_epoch_{epoch:04d}")
    else:
        model_path = os.path.join(model_root, class_name, "model")
        
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Please run training first: python diffusion/train_ddpm_wbc.py --class_name {class_name}"
        )

    print(f"\n{'='*60}")
    print(f"Generation: {class_name}" + (f" (epoch {epoch})" if epoch else ""))
    print(f"{'='*60}")
    print(f"Model Path: {model_path}")
    print(f"Device: {device}")

    model = UNet2DModel.from_pretrained(model_path).to(device)
    model.eval()

    ddim = DDIMScheduler(num_train_timesteps=1000)
    ddim.set_timesteps(num_steps)

    if preview:
        # Generate a 16-image grid for visual inspection
        out_dir = os.path.join(output_root, class_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"PREVIEW Mode: Generating 16-sample grid (saturation={saturation}, contrast={contrast})...")
        
        x = torch.randn(16, 3, image_size, image_size, device=device)
        for t in ddim.timesteps:
            noise_pred = model(x, t).sample
            x = ddim.step(noise_pred, t, x).prev_sample
        x = (x / 2 + 0.5).clamp(0, 1)
        
        # Apply post-processing to each image and convert back to tensor
        import torchvision.transforms.functional as TF
        enhanced = []
        for i in range(x.shape[0]):
            pil = enhance_color(x[i], saturation, contrast)
            enhanced.append(TF.to_tensor(pil))
            
        x = torch.stack(enhanced)
        grid = make_grid(x, nrow=4)
        
        # Add epoch, steps, and saturation to filename for easy comparison
        tag = (f"_epoch{epoch}" if epoch else "") + f"_steps{num_steps}_sat{saturation}"
        preview_path = os.path.join(out_dir, f"preview_grid{tag}.png")
        save_image(grid, preview_path)
        print(f"-> Preview saved to: {preview_path}")
        return

    # Full generation: save individual files
    gen_dir = os.path.join(output_root, class_name, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    print(f"Target: {n_images} images ({num_steps} DDIM steps, sat={saturation}, contrast={contrast})")
    count = 0
    while count < n_images:
        b = min(batch, n_images - count)
        x = torch.randn(b, 3, image_size, image_size, device=device)
        
        for t in ddim.timesteps:
            noise_pred = model(x, t).sample
            x = ddim.step(noise_pred, t, x).prev_sample
            
        x = (x / 2 + 0.5).clamp(0, 1)
        
        for i in range(b):
            fname = os.path.join(gen_dir, f"{class_name}_synth_{count+i+1:04d}.png")
            # Save the post-processed PIL image directly
            pil = enhance_color(x[i], saturation, contrast)
            pil.save(fname)
            
        count += b
        print(f"  {count}/{n_images} generated...")

    print(f"\nCompleted: {count} images saved to -> {gen_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic WBC images using trained DDPM/DDIM")
    parser.add_argument("--class_name", type=str, required=True,
                        choices=["Basophil", "Monocyte"],
                        help="Target class to generate")
    parser.add_argument("--model_root", type=str,
                        default="./checkpoints/diffusion_models",
                        help="Root directory of trained models")
    parser.add_argument("--output_root", type=str,
                        default="./checkpoints/diffusion_models",
                        help="Root directory for outputs (creates generated/ subfolder)")
    parser.add_argument("--n_images", type=int, default=100,
                        help="Number of images to generate")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--num_steps", type=int, default=30,
                        help="Number of DDIM steps (lower=faster, higher=better quality)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--preview", action="store_true",
                        help="Generate a 16-image grid for quick inspection")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Use a specific epoch checkpoint (e.g., 225). Leave blank for main model")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="Color saturation multiplier (1.0=no change, 1.3=30% more vibrant)")
    parser.add_argument("--contrast", type=float, default=1.0,
                        help="Contrast multiplier (1.0=no change)")
    args = parser.parse_args()

    generate(
        class_name=args.class_name,
        model_root=args.model_root,
        output_root=args.output_root,
        n_images=args.n_images,
        image_size=args.image_size,
        num_steps=args.num_steps,
        batch=args.batch,
        preview=args.preview,
        epoch=args.epoch,
        saturation=args.saturation,
        contrast=args.contrast,
    )