"""
Test a single image with BASELINE RDN model (original GLF-CR)
Usage: python test_baseline.py --image_path <path_to_image> --model_checkpoint <path_to_checkpoint>
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from skimage.metrics import structural_similarity as ssim

# Add codes directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# IMPORTING ORIGINAL RDN ARCHITECTURE
from net_CR_RDN import RDN_residual_CR


def load_tiff_image(image_path):
    """Load TIFF image and ensure proper shape (C, H, W)"""
    image = tifffile.imread(image_path)
    
    # Ensure proper shape: (channels, height, width)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        h, w, c = image.shape
        if c <= 20 and h > c and w > c:  # Channel-last format
            image = np.transpose(image, (2, 0, 1))
    
    # Handle NaN values
    image[np.isnan(image)] = np.nanmean(image)
    
    return image.astype('float32')


def normalize_optical_image(image, scale=10000):
    """Normalize optical image by scale factor"""
    return image / scale


def normalize_sar_image(image):
    """Normalize SAR image"""
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
        return float('inf')
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr


def calculate_ssim(pred, ref):
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    if len(pred.shape) == 3:
        ssim_values = []
        for c in range(pred.shape[0]):
            ssim_val = ssim(ref[c], pred[c], data_range=1.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        return ssim(ref, pred, data_range=1.0)


def calculate_sam(pred, ref):
    pred_flat = pred.reshape(pred.shape[0], -1) 
    ref_flat = ref.reshape(ref.shape[0], -1)   
    dots = np.sum(pred_flat * ref_flat, axis=0) 
    norms_pred = np.linalg.norm(pred_flat, axis=0) 
    norms_ref = np.linalg.norm(ref_flat, axis=0)   
    valid = (norms_pred > 1e-8) & (norms_ref > 1e-8)
    norms_prod = norms_pred[valid] * norms_ref[valid]
    dots_valid = dots[valid]
    cos_angles = np.clip(dots_valid / norms_prod, -1, 1)
    angles = np.arccos(cos_angles)
    sam_degrees = np.degrees(np.mean(angles))
    return sam_degrees


def calculate_rmse(pred, ref):
    mse = np.mean((pred - ref) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def find_reference_image(image_path):
    # (Same as test_image.py)
    base_path = image_path.replace('.tif', '').replace('.TIF', '')
    filename_base = os.path.basename(base_path)
    scene_id_parts = filename_base.split('_')
    if len(scene_id_parts) >= 2:
        scene_id = '_'.join(scene_id_parts[-2:])
    else:
        scene_id = filename_base
    image_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(image_dir)
    ref_candidates = [
        os.path.join(image_dir, f'ROIs2017_winter_s2_{scene_id}_B1_B12.tif'),
        os.path.join(image_dir, f'ROIs2017_winter_s2_{scene_id}_B1_B12.TIF'),
        os.path.join(parent_dir, 'ROIs2017_winter_s2_cloudfree', filename_base.replace('cloudy', 'cloudfree') + '_B1_B12.tif'),
        os.path.join(parent_dir, 'ROIs2017_winter_s2_cloudfree', filename_base.replace('cloudy', 'cloudfree') + '_B1_B12.TIF'),
    ]
    if parent_dir != image_dir:
        cloudfree_dirs = glob.glob(os.path.join(parent_dir, '*cloudfree*'), recursive=False)
        for cf_dir in cloudfree_dirs:
            ref_candidates.extend([
                os.path.join(cf_dir, f'{scene_id}_B1_B12.tif'),
                os.path.join(cf_dir, f'{scene_id}_B1_B12.TIF'),
                os.path.join(cf_dir, f'ROIs2017_winter_s2_{scene_id}_B1_B12.tif'),
                os.path.join(cf_dir, f'ROIs2017_winter_s2_{scene_id}_B1_B12.TIF'),
            ])
    for candidate in ref_candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def test_single_image(image_path, model_checkpoint, output_dir, sar_path=None, cloudfree_path=None, device='cuda'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle paths (Same as test_image.py)
    if image_path.endswith(('.tif', '.TIF', '.tiff', '.TIFF')):
        optical_path = image_path
        base_path = image_path.replace('.tif', '').replace('.TIF', '').replace('.tiff', '').replace('.TIFF', '')
    else:
        base_path = image_path
        optical_path = None
    
    if optical_path is None or not os.path.exists(optical_path):
        optical_candidates = [base_path + '_B1_B12.tif', base_path + '_B1_B12.TIF']
        for candidate in optical_candidates:
            if os.path.exists(candidate):
                optical_path = candidate
                break
    
    if not optical_path or not os.path.exists(optical_path):
        raise FileNotFoundError(f"Could not find optical image for {base_path}")
    
    # Find SAR image
    if sar_path is None:
        image_dir = os.path.dirname(optical_path)
        filename_base = os.path.basename(optical_path)
        filename_base_clean = filename_base.replace('_B1_B12', '').replace('.tif', '').replace('.TIF', '')
        scene_id_parts = filename_base_clean.split('_')
        if len(scene_id_parts) >= 2:
            scene_id = '_'.join(scene_id_parts[-2:])
        else:
            scene_id = filename_base_clean
        
        sar_candidates = [
            os.path.join(image_dir, filename_base_clean + '_sar.tif'),
            os.path.join(image_dir, filename_base_clean + '_VV_VH.tif'),
        ]
        parent_dir = os.path.dirname(image_dir)
        if parent_dir != image_dir:
            sar_candidates.extend([
                os.path.join(parent_dir, f'ROIs2017_winter_s1_{scene_id}.tif'),
            ])
            s1_dir = os.path.join(parent_dir, 'ROIs2017_winter_s1')
            if os.path.exists(s1_dir):
                sar_files = glob.glob(os.path.join(s1_dir, '**', f'*{scene_id}*.tif'), recursive=True)
                if sar_files:
                    sar_candidates.append(sar_files[0])

        sar_found = None
        for candidate in sar_candidates:
            if os.path.exists(candidate):
                sar_found = candidate
                break
        
        if sar_found:
            sar_path = sar_found
        else:
            raise FileNotFoundError(f"Could not auto-detect SAR image for scene {scene_id}.")
    elif not os.path.exists(sar_path):
        raise FileNotFoundError(f"SAR image not found: {sar_path}")
    
    print(f"Loading SAR image: {sar_path}")
    print(f"Loading Optical image: {optical_path}")
    
    # Load images
    sar_data = load_tiff_image(sar_path)
    optical_data = load_tiff_image(optical_path)
    
    sar_normalized = normalize_sar_image(sar_data)
    optical_normalized = normalize_optical_image(optical_data)
    
    sar_tensor = torch.from_numpy(sar_normalized).unsqueeze(0).to(device)
    optical_tensor = torch.from_numpy(optical_normalized).unsqueeze(0).to(device)
    
    print(f"\nInput shapes:")
    print(f"  SAR: {sar_tensor.shape}")
    print(f"  Optical: {optical_tensor.shape}")
    
    # Load BASELINE RDN MODEL
    print(f"\nLoading BASELINE model from: {model_checkpoint}")
    # Using crop_size=256 as standard, cross_attn=False for baseline
    model = RDN_residual_CR(256, use_cross_attn=False).to(device)
    
    checkpoint = torch.load(model_checkpoint, map_location=device)
    
    # Handle checkpoint structure
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Sometimes the checkpoint IS the state dict
        state_dict = checkpoint
    
    # Handle DataParallel wrapping
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Try loading
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Loaded weights successfully (Strict mode)")
    except Exception as e:
        print(f"Warning: Strict loading failed ({e}). Trying strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded weights (Relaxed mode)")
        
    model.eval()
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(optical_tensor, sar_tensor)
    
    output_np = output.cpu().squeeze(0).numpy()
    output_np = np.clip(output_np * 10000.0, 0, 10000).astype('float32')
    
    print(f"\n✓ Output shape: {output_np.shape}")
    
    # Metrics
    ref_path = cloudfree_path
    if ref_path is None:
        ref_path = find_reference_image(optical_path)
    
    if ref_path and os.path.exists(ref_path):
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
        
        metrics_txt = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_txt, 'w') as f:
            f.write(f"BASELINE RESULT\n")
            f.write(f"PSNR: {psnr:.4f}\n")
            f.write(f"SSIM: {ssim_val:.4f}\n")
    else:
        print("Reference image not found. Skipping metric calculation.")
    
    # Save Output
    output_tiff_path = os.path.join(output_dir, 'output_13bands.tif')
    tifffile.imwrite(output_tiff_path, output_np.astype('float32'))
    print(f"✓ Saved result: {output_tiff_path}")
    
    # RGB Preview
    if output_np.shape[0] >= 4:
        rgb = np.stack([output_np[3], output_np[2], output_np[1]], axis=0)
        rgb = rgb / 10000.0
        rgb = np.clip(rgb, 0.0, 0.35) / 0.35
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = np.power(rgb, 1/1.4)
        rgb = (rgb * 255).astype(np.uint8)
        rgb_pil = np.transpose(rgb, (1, 2, 0))
        plt.figure(figsize=(12, 10))
        plt.imshow(rgb_pil)
        plt.title('Baseline Result (RGB)')
        plt.axis('off')
        output_png_path = os.path.join(output_dir, 'output_rgb.png')
        plt.savefig(output_png_path, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved PNG preview: {output_png_path}")

    return output_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--sar_path', type=str, default=None)
    parser.add_argument('--cloudfree_path', type=str, default=None)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/images_pred')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        
    test_single_image(args.image_path, args.model_checkpoint, args.output_dir, args.sar_path, args.cloudfree_path, args.device)
