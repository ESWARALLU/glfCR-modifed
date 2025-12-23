"""
Test a single image with CloudRemovalCrossAttention model
Usage: python test_image.py --image_path <path_to_image> --model_checkpoint <path_to_checkpoint> --output_dir /kaggle/working/images_pred
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

from net_CR_CrossAttention import CloudRemovalCrossAttention


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
    """
    Calculate PSNR between predicted and reference images
    
    Args:
        pred: Predicted image (C, H, W) or (H, W) in [0, 1]
        ref: Reference image (C, H, W) or (H, W) in [0, 1]
    
    Returns:
        PSNR value in dB
    """
    # Ensure same shape
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    
    # Calculate MSE
    mse = np.mean((pred - ref) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Max value is 1.0 for normalized images
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    return psnr


def calculate_ssim(pred, ref):
    """
    Calculate SSIM between predicted and reference images
    
    Args:
        pred: Predicted image (C, H, W) or (H, W) in [0, 1]
        ref: Reference image (C, H, W) or (H, W) in [0, 1]
    
    Returns:
        SSIM value (mean across channels if multi-channel)
    """
    # Ensure same shape
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    
    if len(pred.shape) == 3:
        # Multi-channel: calculate SSIM for each channel and average
        ssim_values = []
        for c in range(pred.shape[0]):
            ssim_val = ssim(ref[c], pred[c], data_range=1.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        # Single channel
        return ssim(ref, pred, data_range=1.0)


def calculate_sam(pred, ref):
    """
    Calculate Spectral Angle Mapper (SAM) between predicted and reference images
    Common metric in remote sensing
    
    Args:
        pred: Predicted image (C, H, W) normalized to [0, 1]
        ref: Reference image (C, H, W) normalized to [0, 1]
    
    Returns:
        SAM value in degrees (mean across all pixels)
    """
    # Flatten spatial dimensions: (C, H, W) -> (C, H*W)
    pred_flat = pred.reshape(pred.shape[0], -1)  # (C, N)
    ref_flat = ref.reshape(ref.shape[0], -1)    # (C, N)
    
    # Calculate spectral angle for each pixel
    dots = np.sum(pred_flat * ref_flat, axis=0)  # (N,)
    norms_pred = np.linalg.norm(pred_flat, axis=0)  # (N,)
    norms_ref = np.linalg.norm(ref_flat, axis=0)   # (N,)
    
    # Avoid division by zero
    valid = (norms_pred > 1e-8) & (norms_ref > 1e-8)
    
    norms_prod = norms_pred[valid] * norms_ref[valid]
    dots_valid = dots[valid]
    
    # Clip to avoid numerical errors in arccos
    cos_angles = np.clip(dots_valid / norms_prod, -1, 1)
    angles = np.arccos(cos_angles)
    
    # Convert to degrees and return mean
    sam_degrees = np.degrees(np.mean(angles))
    return sam_degrees


def calculate_rmse(pred, ref):
    """
    Calculate Root Mean Square Error (RMSE)
    
    Args:
        pred: Predicted image (C, H, W) in [0, 1]
        ref: Reference image (C, H, W) in [0, 1]
    
    Returns:
        RMSE value (mean across all channels and pixels)
    """
    mse = np.mean((pred - ref) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def find_reference_image(image_path):
    """
    Find the corresponding cloud-free reference image for a cloudy image
    
    Args:
        image_path: Path to cloudy optical image
    
    Returns:
        Path to reference image if found, None otherwise
    """
    # Extract base name and clean it
    base_path = image_path.replace('.tif', '').replace('.TIF', '')
    filename_base = os.path.basename(base_path)
    
    # Extract scene ID (e.g., 102_p100 from ROIs2017_winter_s2_cloudy_102_p100)
    scene_id_parts = filename_base.split('_')
    if len(scene_id_parts) >= 2:
        scene_id = '_'.join(scene_id_parts[-2:])
    else:
        scene_id = filename_base
    
    # Try different reference image naming conventions
    image_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(image_dir)
    
    ref_candidates = [
        # Same directory with different prefix
        os.path.join(image_dir, f'ROIs2017_winter_s2_{scene_id}_B1_B12.tif'),
        os.path.join(image_dir, f'ROIs2017_winter_s2_{scene_id}_B1_B12.TIF'),
        # Cloud-free subdirectory
        os.path.join(parent_dir, 'ROIs2017_winter_s2_cloudfree', filename_base.replace('cloudy', 'cloudfree') + '_B1_B12.tif'),
        os.path.join(parent_dir, 'ROIs2017_winter_s2_cloudfree', filename_base.replace('cloudy', 'cloudfree') + '_B1_B12.TIF'),
    ]
    
    # Also search in parent directories for cloudfree folder
    if parent_dir != image_dir:
        cloudfree_dirs = glob.glob(os.path.join(parent_dir, '*cloudfree*'), recursive=False)
        for cf_dir in cloudfree_dirs:
            ref_candidates.extend([
                os.path.join(cf_dir, f'{scene_id}_B1_B12.tif'),
                os.path.join(cf_dir, f'{scene_id}_B1_B12.TIF'),
                os.path.join(cf_dir, f'ROIs2017_winter_s2_{scene_id}_B1_B12.tif'),
                os.path.join(cf_dir, f'ROIs2017_winter_s2_{scene_id}_B1_B12.TIF'),
            ])
    
    # Find first existing reference
    for candidate in ref_candidates:
        if os.path.exists(candidate):
            return candidate
    
    return None


def test_single_image(image_path, model_checkpoint, output_dir, sar_path=None, cloudfree_path=None, device='cuda'):
    """
    Test a single image with the CloudRemovalCrossAttention model
    
    Args:
        image_path: Path to the cloudy optical image (S2) (with or without extension)
                    Can be full path like: /path/to/ROIs2017_winter_s2_cloudy_102_p100.tif
                    Or base path like: /path/to/image_base_name
        model_checkpoint: Path to the model checkpoint
        output_dir: Directory to save output images
        sar_path: Path to SAR image (S1). If None, will try to auto-detect
        cloudfree_path: Path to cloud-free reference image. If None, will try to auto-detect
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        output_image: Predicted cloud-free optical image
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle both full paths with extension and base paths
    if image_path.endswith(('.tif', '.TIF', '.tiff', '.TIFF')):
        # Full path with extension - extract base and directory
        optical_path = image_path
        base_path = image_path.replace('.tif', '').replace('.TIF', '').replace('.tiff', '').replace('.TIFF', '')
    else:
        # Base path without extension
        base_path = image_path
        optical_path = None
    
    # If optical_path not found, look for it
    if optical_path is None or not os.path.exists(optical_path):
        optical_candidates = [
            base_path + '_B1_B12.tif',
            base_path + '_B1_B12.TIF',
            base_path + '.tif',
            base_path + '.TIF',
        ]
        
        optical_path = None
        for candidate in optical_candidates:
            if os.path.exists(candidate):
                optical_path = candidate
                break
    
    if not optical_path or not os.path.exists(optical_path):
        raise FileNotFoundError(f"Could not find optical image for {base_path}")
    
    # Find SAR image
    if sar_path is None:
        # Auto-detect SAR image
        image_dir = os.path.dirname(optical_path)
        filename_base = os.path.basename(optical_path)
        
        # Extract the base name and scene ID (e.g., 102_p100 from ROIs2017_winter_s2_cloudy_102_p100)
        filename_base_clean = filename_base.replace('_B1_B12', '').replace('.tif', '').replace('.TIF', '')
        # Extract scene ID (last part after last underscore)
        scene_id_parts = filename_base_clean.split('_')
        if len(scene_id_parts) >= 2:
            scene_id = '_'.join(scene_id_parts[-2:])  # e.g., "102_p100"
        else:
            scene_id = filename_base_clean
        
        # Try same directory first
        sar_candidates = [
            os.path.join(image_dir, filename_base_clean + '_sar.tif'),
            os.path.join(image_dir, filename_base_clean + '_VV_VH.tif'),
            os.path.join(image_dir, filename_base_clean + '_sar.TIF'),
            os.path.join(image_dir, filename_base_clean + '_VV_VH.TIF'),
        ]
        
        # Also try parent directory (for cases where S1 and S2 are in different subdirs)
        parent_dir = os.path.dirname(image_dir)
        if parent_dir != image_dir:  # Check we're not at root
            sar_candidates.extend([
                os.path.join(parent_dir, f'ROIs2017_winter_s1_{scene_id}.tif'),
                os.path.join(parent_dir, f'ROIs2017_winter_s1_{scene_id}.TIF'),
            ])
            
            # Also check s1 subdirectories
            s1_search_patterns = [
                os.path.join(parent_dir, '*s1*', f'*{scene_id}.tif'),
                os.path.join(parent_dir, '*s1*', f'*{scene_id}.TIF'),
            ]
            for pattern in s1_search_patterns:
                matches = glob.glob(pattern, recursive=True)
                sar_candidates.extend(matches)
        
        # Search for SAR file
        sar_found = None
        for candidate in sar_candidates:
            if os.path.exists(candidate):
                sar_found = candidate
                break
        
        if not sar_found and parent_dir != image_dir:
            # Fallback: search more broadly in parent directories
            s1_dir = os.path.join(parent_dir, 'ROIs2017_winter_s1')
            if os.path.exists(s1_dir):
                sar_files = glob.glob(os.path.join(s1_dir, '**', f'*{scene_id}*.tif'), recursive=True)
                if sar_files:
                    sar_found = sar_files[0]
        
        if sar_found:
            sar_path = sar_found
        else:
            raise FileNotFoundError(
                f"Could not auto-detect SAR image for scene {scene_id}. "
                f"Please provide --sar_path directly. Checked: {sar_candidates}"
            )
    elif not os.path.exists(sar_path):
        raise FileNotFoundError(f"SAR image not found: {sar_path}")
    
    print(f"Loading SAR image: {sar_path}")
    print(f"Loading Optical image: {optical_path}")
    
    # Load images
    sar_data = load_tiff_image(sar_path)
    optical_data = load_tiff_image(optical_path)
    
    # Normalize
    sar_normalized = normalize_sar_image(sar_data)
    optical_normalized = normalize_optical_image(optical_data)
    
    # Convert to tensors and add batch dimension
    sar_tensor = torch.from_numpy(sar_normalized).unsqueeze(0).to(device)  # (1, 2, H, W)
    optical_tensor = torch.from_numpy(optical_normalized).unsqueeze(0).to(device)  # (1, 13, H, W)
    
    print(f"\nInput shapes:")
    print(f"  SAR: {sar_tensor.shape}")
    print(f"  Optical: {optical_tensor.shape}")
    
    # Load model
    print(f"\nLoading model from: {model_checkpoint}")
    model = CloudRemovalCrossAttention().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
    
    # Handle DataParallel wrapping
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(optical_tensor, sar_tensor)
    
    output_np = output.cpu().squeeze(0).numpy()  # (13, H, W)
    
    # Scale output back to match input optical image range [0, 10000]
    output_np = np.clip(output_np * 10000.0, 0, 10000).astype('float32')
    
    print(f"\n✓ Output shape: {output_np.shape}")
    print(f"✓ Expected shape: {optical_normalized.shape}")
    print(f"✓ Shape match: {output_np.shape == optical_normalized.shape * 10000}")
    # Verify spatial dimensions match input
    if output_np.shape != optical_data.shape:
        raise ValueError(f"DIMENSION MISMATCH! Output {output_np.shape} vs Input {optical_data.shape}")
    
    # Try to find and load reference image for metrics
    print("\n" + "="*60)
    print("Computing Quality Metrics...")
    print("="*60)
    
    ref_path = cloudfree_path
    if ref_path is None:
        # Try to auto-detect if not provided
        ref_path = find_reference_image(optical_path)
    
    if ref_path and os.path.exists(ref_path):
        try:
            print(f"Found reference image: {ref_path}")
            ref_image = load_tiff_image(ref_path)
            ref_normalized = normalize_optical_image(ref_image)
            
            # Normalize output for metrics
            output_normalized = output_np / 10000.0
            
            # Calculate metrics
            psnr = calculate_psnr(output_normalized, ref_normalized)
            ssim_val = calculate_ssim(output_normalized, ref_normalized)
            sam = calculate_sam(output_normalized, ref_normalized)
            rmse = calculate_rmse(output_normalized, ref_normalized)
            
            print(f"\n✓ PSNR:  {psnr:.4f} dB")
            print(f"✓ SSIM:  {ssim_val:.4f}")
            print(f"✓ SAM:   {sam:.4f}°")
            print(f"✓ RMSE:  {rmse:.6f}")
            
            # Also save metrics to a text file
            metrics_txt = os.path.join(output_dir, 'metrics.txt')
            with open(metrics_txt, 'w') as f:
                f.write(f"Image: {os.path.basename(optical_path)}\n")
                f.write("="*50 + "\n")
                f.write(f"PSNR (dB):         {psnr:.4f}\n")
                f.write(f"SSIM:              {ssim_val:.4f}\n")
                f.write(f"SAM (degrees):     {sam:.4f}\n")
                f.write(f"RMSE:              {rmse:.6f}\n")
            print(f"✓ Metrics saved to: {metrics_txt}")
            
        except Exception as e:
            print(f"Warning: Could not calculate metrics: {e}")
            print("Continuing with visualization...")
    else:
        print("Reference image not found. Skipping metric calculation.")
        print("(Provide --cloudfree_path to specify reference image)")
    
    print("="*60)
    
    # Save outputs in multiple formats for clarity
    # Format 1: Full 13-band TIFF (this is the actual model output)
    output_tiff_path = os.path.join(output_dir, 'output_13bands.tif')
    tifffile.imwrite(output_tiff_path, output_np.astype('float32'))
    print(f"✓ Saved 13-band TIFF: {output_tiff_path}")
    print(f"  Note: This TIFF contains all 13 Sentinel-2 bands")
    print(f"  When opened in viewers, you may see it as 13 'pages' - this is correct!")
    
    # Format 2: RGB composite TIFF (true color: Red=B4, Green=B3, Blue=B2)
    if output_np.shape[0] >= 4:
        rgb = np.stack([output_np[3], output_np[2], output_np[1]], axis=0)  # (3, H, W)
        rgb = np.clip(rgb, 0, 10000).astype('float32')
        rgb_tiff_path = os.path.join(output_dir, 'output_rgb.tif')
        tifffile.imwrite(rgb_tiff_path, rgb)
        print(f"✓ Saved RGB composite TIFF: {rgb_tiff_path}")
        print(f"  (Red=Band4, Green=Band3, Blue=Band2)")
    
    # Format 3: Old naming for backward compatibility
    output_tiff_path_legacy = os.path.join(output_dir, 'output_cloudremoved.tif')
    tifffile.imwrite(output_tiff_path_legacy, output_np.astype('float32'))
    print(f"✓ Saved legacy format: {output_tiff_path_legacy}")
    
    
    # Create visualization (RGB from bands 3, 2, 1 for true color)
    # Sentinel-2 bands: B1-B12, so B3=Red, B2=Green, B1=Blue (0-indexed: 2, 1, 0)
    try:
        # Use bands 4, 3, 2 (Red, Green, Blue) if available
        if output_np.shape[0] >= 4:
            rgb = np.stack([output_np[3], output_np[2], output_np[1]], axis=0)  # (3, H, W)
            
            # Normalize for visualization using actual min/max for contrast stretching
            rgb_min = rgb.min()
            rgb_max = rgb.max()
            if rgb_max > rgb_min:
                rgb = (rgb - rgb_min) / (rgb_max - rgb_min)  # Normalize to [0, 1]
            else:
                rgb = np.zeros_like(rgb)
            rgb = np.clip(rgb, 0, 1)
            rgb = (rgb * 255).astype(np.uint8)
            
            # Save RGB visualization
            rgb_pil = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
            
            # Using matplotlib to save
            plt.figure(figsize=(12, 10))
            plt.imshow(rgb_pil)
            plt.title('Cloud-Removed Image (RGB)')
            plt.axis('off')
            
            output_png_path = os.path.join(output_dir, 'output_cloudremoved_rgb.png')
            plt.savefig(output_png_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"✓ Saved PNG visualization: {output_png_path}")
        else:
            print("Warning: Not enough bands for RGB visualization")
    except Exception as e:
        print(f"Warning: Could not create RGB visualization: {e}")
    
    # Also save individual band visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i in range(4):
            if i < output_np.shape[0]:
                ax = axes[i]
                band_data = output_np[i]
                im = ax.imshow(band_data, cmap='viridis')
                ax.set_title(f'Band {i+1}')
                ax.axis('off')
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        band_viz_path = os.path.join(output_dir, 'output_bands_sample.png')
        plt.savefig(band_viz_path, dpi=150)
        plt.close()
        print(f"✓ Saved band visualization: {band_viz_path}")
    except Exception as e:
        print(f"Warning: Could not create band visualization: {e}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\n📋 Output Format Information:")
    print("  • output_13bands.tif: Full 13-band Sentinel-2 output (actual model result)")
    print("  • output_rgb.tif: RGB composite for easy viewing")
    print("  • output_cloudremoved.tif: Legacy naming (same as 13-band)")
    print("\n⚠️  About the '13 bands as 13 separate images':")
    print("  Multi-band TIFF files store each channel as a separate 'page'.")
    print("  This is CORRECT behavior, not an error!")
    print("  To read all bands in Python: tifffile.imread('output_13bands.tif') → (13, H, W)")
    print("="*60)
    
    return output_np


def main():
    parser = argparse.ArgumentParser(
        description='Test CloudRemovalCrossAttention model on a single image'
    )
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the cloudy optical image (S2) with or without extension')
    parser.add_argument('--sar_path', type=str, default=None,
                        help='Path to the SAR image (S1). If not provided, will auto-detect')
    parser.add_argument('--cloudfree_path', type=str, default=None,
                        help='Path to the cloud-free reference image for metrics. If not provided, will auto-detect')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/images_pred',
                        help='Directory to save output images')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Run test
    output = test_single_image(
        image_path=args.image_path,
        sar_path=args.sar_path,
        cloudfree_path=args.cloudfree_path,
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        device=device
    )


if __name__ == '__main__':
    main()
