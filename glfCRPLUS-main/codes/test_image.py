"""
Test a single image with CloudRemovalCrossAttention model
Usage: python test_image.py --image_path <path_to_image> --model_checkpoint <path_to_checkpoint> --output_dir /kaggle/output/images_pred
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


def test_single_image(image_path, model_checkpoint, output_dir, device='cuda'):
    """
    Test a single image with the CloudRemovalCrossAttention model
    
    Args:
        image_path: Path to the image (without extension, will try .tif)
        model_checkpoint: Path to the model checkpoint
        output_dir: Directory to save output images
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        output_image: Predicted cloud-free optical image
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Try different extensions
    base_path = image_path.replace('.tif', '').replace('.TIF', '')
    
    # Find SAR and optical files
    sar_candidates = [
        base_path + '_VV_VH.tif',
        base_path + '_sar.tif',
        base_path + '_VV_VH.TIF',
    ]
    
    optical_candidates = [
        base_path + '_B1_B12.tif',
        base_path + '_optical.tif',
        base_path + '_B1_B12.TIF',
    ]
    
    sar_path = None
    optical_path = None
    
    for candidate in sar_candidates:
        if os.path.exists(candidate):
            sar_path = candidate
            break
    
    for candidate in optical_candidates:
        if os.path.exists(candidate):
            optical_path = candidate
            break
    
    if not sar_path or not optical_path:
        raise FileNotFoundError(f"Could not find SAR or optical images for {base_path}")
    
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
    
    print(f"Output shape: {output_np.shape}")
    print(f"Output range: [{output_np.min():.4f}, {output_np.max():.4f}]")
    
    # Save output TIFF
    output_tiff_path = os.path.join(output_dir, 'output_cloudremoved.tif')
    tifffile.imwrite(output_tiff_path, output_np.astype('float32'))
    print(f"✓ Saved TIFF output: {output_tiff_path}")
    
    # Create visualization (RGB from bands 3, 2, 1 for true color)
    # Sentinel-2 bands: B1-B12, so B3=Red, B2=Green, B1=Blue (0-indexed: 2, 1, 0)
    try:
        # Use bands 3, 2, 1 (Red, Green, Blue) if available
        if output_np.shape[0] >= 3:
            rgb = np.stack([output_np[2], output_np[1], output_np[0]], axis=0)  # (3, H, W)
            
            # Normalize for visualization
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
    
    return output_np


def main():
    parser = argparse.ArgumentParser(
        description='Test CloudRemovalCrossAttention model on a single image'
    )
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image (without extension)')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='/kaggle/output/images_pred',
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
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        device=device
    )


if __name__ == '__main__':
    main()
