"""
Test a single image with the base GLF-CR (RDN) model.
Usage: python test_image_base.py --image_path <cloudy_optical> --model_checkpoint <base_glf_cr.pth> --output_dir ./images_pred_base
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Add codes directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net_CR_RDN import RDN_residual_CR

# -----------------
# Utility functions
# -----------------

def load_tiff_image(image_path):
    image = tifffile.imread(image_path)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        h, w, c = image.shape
        if c <= 20 and h > c and w > c:
            image = np.transpose(image, (2, 0, 1))
    image[np.isnan(image)] = np.nanmean(image)
    return image.astype("float32")


def normalize_optical_image(image, scale=10000):
    return image / scale


def normalize_sar_image(image):
    clip_min = [-25.0, -32.5]
    clip_max = [0.0, 0.0]
    normalized = np.zeros_like(image)
    for channel in range(len(image)):
        data = image[channel]
        data = np.clip(data, clip_min[channel], clip_max[channel])
        data -= clip_min[channel]
        normalized[channel] = data / (clip_max[channel] - clip_min[channel])
    return normalized


def calculate_psnr(pred, ref):
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    mse = np.mean((pred - ref) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim(pred, ref):
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    if len(pred.shape) == 3:
        ssim_values = [ssim(ref[c], pred[c], data_range=1.0) for c in range(pred.shape[0])]
        return np.mean(ssim_values)
    return ssim(ref, pred, data_range=1.0)


def calculate_sam(pred, ref):
    pred_flat = pred.reshape(pred.shape[0], -1)
    ref_flat = ref.reshape(ref.shape[0], -1)
    dots = np.sum(pred_flat * ref_flat, axis=0)
    norms_pred = np.linalg.norm(pred_flat, axis=0)
    norms_ref = np.linalg.norm(ref_flat, axis=0)
    valid = (norms_pred > 1e-8) & (norms_ref > 1e-8)
    cos_angles = np.clip(dots[valid] / (norms_pred[valid] * norms_ref[valid]), -1, 1)
    angles = np.arccos(cos_angles)
    return np.degrees(np.mean(angles))


def calculate_rmse(pred, ref):
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    return np.sqrt(np.mean((pred - ref) ** 2))


def find_reference_image(optical_path):
    image_dir = os.path.dirname(optical_path)
    filename_base = os.path.basename(optical_path)
    filename_base = filename_base.replace("_B1_B12", "").replace(".tif", "").replace(".TIF", "")
    scene_id_parts = filename_base.split("_")
    scene_id = "_".join(scene_id_parts[-2:]) if len(scene_id_parts) >= 2 else filename_base
    parent_dir = os.path.dirname(image_dir)
    ref_candidates = [
        os.path.join(image_dir, f"{filename_base}_cloudfree_B1_B12.tif"),
        os.path.join(image_dir, f"{filename_base}_cloudfree_B1_B12.TIF"),
        os.path.join(image_dir, f"ROIs2017_winter_s2_{scene_id}_B1_B12.tif"),
        os.path.join(image_dir, f"ROIs2017_winter_s2_{scene_id}_B1_B12.TIF"),
    ]
    if parent_dir != image_dir:
        cloudfree_dirs = glob.glob(os.path.join(parent_dir, "*cloudfree*"), recursive=False)
        for cf_dir in cloudfree_dirs:
            ref_candidates.extend([
                os.path.join(cf_dir, f"{scene_id}_B1_B12.tif"),
                os.path.join(cf_dir, f"{scene_id}_B1_B12.TIF"),
                os.path.join(cf_dir, f"ROIs2017_winter_s2_{scene_id}_B1_B12.tif"),
                os.path.join(cf_dir, f"ROIs2017_winter_s2_{scene_id}_B1_B12.TIF"),
            ])
    for candidate in ref_candidates:
        if os.path.exists(candidate):
            return candidate
    return None


# -----------------
# Model helpers
# -----------------

def load_base_model(checkpoint_path, device, crop_size=256):
    model = RDN_residual_CR(crop_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("network", checkpoint)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Drop incompatible buffer keys (e.g., attn_mask shapes differ between training/inference crops)
    state_dict = {k: v for k, v in state_dict.items() if "attn_mask" not in k}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# -----------------
# Main inference
# -----------------

def test_single_image(image_path, model_checkpoint, output_dir, sar_path=None, cloudfree_path=None, device="cuda", crop_size=256):
    os.makedirs(output_dir, exist_ok=True)

    if image_path.endswith((".tif", ".TIF", ".tiff", ".TIFF")):
        optical_path = image_path
        base_path = image_path.rsplit(".", 1)[0]
    else:
        base_path = image_path
        optical_path = None

    if optical_path is None or not os.path.exists(optical_path):
        for candidate in [base_path + suffix for suffix in ["_B1_B12.tif", "_B1_B12.TIF", ".tif", ".TIF"]]:
            if os.path.exists(candidate):
                optical_path = candidate
                break
    if not optical_path or not os.path.exists(optical_path):
        raise FileNotFoundError(f"Could not find optical image for {base_path}")

    if sar_path is None:
        image_dir = os.path.dirname(optical_path)
        filename_base = os.path.basename(optical_path)
        filename_base_clean = filename_base.replace("_B1_B12", "").replace(".tif", "").replace(".TIF", "")
        scene_id_parts = filename_base_clean.split("_")
        scene_id = "_".join(scene_id_parts[-2:]) if len(scene_id_parts) >= 2 else filename_base_clean
        sar_candidates = [
            os.path.join(image_dir, filename_base_clean + "_sar.tif"),
            os.path.join(image_dir, filename_base_clean + "_VV_VH.tif"),
            os.path.join(image_dir, filename_base_clean + "_sar.TIF"),
            os.path.join(image_dir, filename_base_clean + "_VV_VH.TIF"),
        ]
        parent_dir = os.path.dirname(image_dir)
        if parent_dir != image_dir:
            sar_candidates.extend([
                os.path.join(parent_dir, f"ROIs2017_winter_s1_{scene_id}.tif"),
                os.path.join(parent_dir, f"ROIs2017_winter_s1_{scene_id}.TIF"),
            ])
            for pattern in [os.path.join(parent_dir, "*s1*", f"*{scene_id}.tif"), os.path.join(parent_dir, "*s1*", f"*{scene_id}.TIF")]:
                sar_candidates.extend(glob.glob(pattern, recursive=True))
        sar_found = next((c for c in sar_candidates if os.path.exists(c)), None)
        if sar_found:
            sar_path = sar_found
        else:
            raise FileNotFoundError(f"Could not auto-detect SAR image for scene {scene_id}. Checked: {sar_candidates}")
    elif not os.path.exists(sar_path):
        raise FileNotFoundError(f"SAR image not found: {sar_path}")

    print(f"Loading SAR image: {sar_path}")
    print(f"Loading Optical image: {optical_path}")

    sar_data = load_tiff_image(sar_path)
    optical_data = load_tiff_image(optical_path)

    sar_normalized = normalize_sar_image(sar_data)
    optical_normalized = normalize_optical_image(optical_data)

    sar_tensor = torch.from_numpy(sar_normalized).unsqueeze(0).to(device)
    optical_tensor = torch.from_numpy(optical_normalized).unsqueeze(0).to(device)

    print("\nInput shapes:")
    print(f"  SAR: {sar_tensor.shape}")
    print(f"  Optical: {optical_tensor.shape}")

    print(f"\nLoading base model from: {model_checkpoint}")
    model = load_base_model(model_checkpoint, device, crop_size=crop_size)

    print("\nRunning inference with base model...")
    with torch.no_grad():
        output = model(optical_tensor, sar_tensor)

    output_np = output.cpu().squeeze(0).numpy()  # (13, H, W)
    output_np = np.clip(output_np * 10000.0, 0, 10000).astype("float32")

    print(f"\n✓ Output shape: {output_np.shape}")
    print(f"✓ Output range: [{output_np.min():.2f}, {output_np.max():.2f}]")

    if output_np.shape != optical_data.shape:
        raise ValueError(f"DIMENSION MISMATCH! Output {output_np.shape} vs Input {optical_data.shape}")

    print("\n" + "=" * 60)
    print("Computing Quality Metrics...")
    print("=" * 60)

    ref_path = cloudfree_path or find_reference_image(optical_path)
    if ref_path and os.path.exists(ref_path):
        try:
            print(f"Found reference image: {ref_path}")
            ref_image = load_tiff_image(ref_path)
            ref_normalized = normalize_optical_image(ref_image)
            output_normalized = output_np / 10000.0
            psnr = calculate_psnr(output_normalized, ref_normalized)
            ssim_val = calculate_ssim(output_normalized, ref_normalized)
            sam = calculate_sam(output_normalized, ref_normalized)
            rmse = calculate_rmse(output_normalized, ref_normalized)
            print(f"\n✓ PSNR:  {psnr:.4f} dB")
            print(f"✓ SSIM:  {ssim_val:.4f}")
            print(f"✓ SAM:   {sam:.4f}°")
            print(f"✓ RMSE:  {rmse:.6f}")
            metrics_txt = os.path.join(output_dir, "metrics_base.txt")
            with open(metrics_txt, "w") as f:
                f.write(f"Image: {os.path.basename(optical_path)}\n")
                f.write("=" * 50 + "\n")
                f.write(f"PSNR (dB):         {psnr:.4f}\n")
                f.write(f"SSIM:              {ssim_val:.4f}\n")
                f.write(f"SAM (degrees):     {sam:.4f}\n")
                f.write(f"RMSE:              {rmse:.6f}\n")
            print(f"✓ Metrics saved to: {metrics_txt}")
        except Exception as e:
            print(f"Warning: Could not calculate metrics: {e}")
    else:
        print("Reference image not found. Skipping metric calculation.")
        print("(Provide --cloudfree_path to specify reference image)")

    print("=" * 60)

    output_tiff_path = os.path.join(output_dir, "output_base_13bands.tif")
    tifffile.imwrite(output_tiff_path, output_np.astype("float32"))
    print(f"✓ Saved 13-band TIFF: {output_tiff_path}")

    if output_np.shape[0] >= 3:
        rgb = np.stack([output_np[2], output_np[1], output_np[0]], axis=0)
        rgb_min = rgb.min()
        rgb_max = rgb.max()
        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
        else:
            rgb = np.zeros_like(rgb)
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        rgb_pil = np.transpose(rgb, (1, 2, 0))
        plt.figure(figsize=(12, 10))
        plt.imshow(rgb_pil)
        plt.title("Cloud-Removed Image (RGB) - Base Model")
        plt.axis("off")
        output_png_path = os.path.join(output_dir, "output_base_cloudremoved_rgb.png")
        plt.savefig(output_png_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"✓ Saved PNG visualization: {output_png_path}")
    else:
        print("Warning: Not enough bands for RGB visualization")

    print("\n" + "=" * 60)
    print("Base Model Testing Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    return output_np


def main():
    parser = argparse.ArgumentParser(description="Test base GLF-CR model on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the cloudy optical image (S2)")
    parser.add_argument("--sar_path", type=str, default=None, help="Path to the SAR image (S1). If not provided, will auto-detect")
    parser.add_argument("--cloudfree_path", type=str, default=None, help="Path to cloud-free reference image for metrics")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to base model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./images_pred_base", help="Directory to save output images")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--crop_size", type=int, default=256, help="Crop size used for the base model (RDN)")
    args = parser.parse_args()

    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")

    device = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    test_single_image(
        image_path=args.image_path,
        sar_path=args.sar_path,
        cloudfree_path=args.cloudfree_path,
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        device=device,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
