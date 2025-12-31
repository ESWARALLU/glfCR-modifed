import os
import torch
import torch.nn as nn
import argparse
import json
from datetime import datetime

# Fix PyTorch 2.6 UnpicklingError
import torch.serialization
import argparse as _argparse
torch.serialization.add_safe_globals([_argparse.Namespace])

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from metrics import PSNR, SSIM
from dataloader import AlignedDataset, get_train_val_test_filelists
from net_CR_RDN import RDN_residual_CR


##########################################################
def test(CR_net, opts):
    """Test the model
    Args:
        CR_net: The model to test
        opts: Configuration options
    """
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    data = AlignedDataset(opts, test_filelist)

    dataloader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=1,               # FORCE batch=1 (model limitation)
        shuffle=False,
        num_workers=0,              # Avoid worker issues
        pin_memory=True
    )

    test_set_size = len(test_filelist)
    print(f"Test set size: {test_set_size} images")

    total_psnr = 0.0
    total_ssim = 0.0
    results_per_image = []
    processed_images = 0

    iterator = tqdm(dataloader, total=len(dataloader), desc='Testing') if tqdm else dataloader

    with torch.no_grad():
        for inputs in iterator:
            cloudy_data = inputs['cloudy_data'].cuda()
            cloudfree_data = inputs['cloudfree_data'].cuda()
            SAR_data = inputs['SAR_data'].cuda()
            file_names = inputs['file_name']

            # Baseline Forward Pass (Standard RDN)
            pred = CR_net(cloudy_data, SAR_data)

            psnr_val = PSNR(pred, cloudfree_data)
            ssim_val = SSIM(pred, cloudfree_data)

            psnr_val = float(psnr_val.item()) if hasattr(psnr_val, "item") else float(psnr_val)
            ssim_val = float(ssim_val.item()) if hasattr(ssim_val, "item") else float(ssim_val)

            total_psnr += psnr_val
            total_ssim += ssim_val

            results_per_image.append({
                "image": file_names,
                "psnr": psnr_val,
                "ssim": ssim_val
            })

            processed_images += 1

            if tqdm:
                iterator.set_postfix({
                    "PSNR": f"{psnr_val:.3f}",
                    "SSIM": f"{ssim_val:.3f}",
                    "Done": processed_images
                })

    avg_psnr = total_psnr / processed_images
    avg_ssim = total_ssim / processed_images

    return avg_psnr, avg_ssim, results_per_image


##########################################################
def main():
    parser = argparse.ArgumentParser()
    # Simplified arguments for Baseline Script
    parser.add_argument('--batch_sz', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter')
    parser.add_argument('--data_list_filepath', type=str, default='/kaggle/working/data.csv')
    parser.add_argument('--checkpoint_path', type=str, required=True)

    # Dataset flags
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    parser.add_argument('--model_name', type=str, default='Baseline_CRNet')

    opts = parser.parse_args()

    # Choose CSV properly
    working_csv = '/kaggle/working/data.csv'
    input_csv = os.path.join(opts.input_data_folder, 'data.csv')

    if opts.data_list_filepath == working_csv:
        if os.path.exists(working_csv):
            opts.data_list_filepath = working_csv
        elif os.path.exists(input_csv):
            print(f"Using dataset CSV: {input_csv}")
            opts.data_list_filepath = input_csv
        else:
            raise FileNotFoundError("No CSV available")

    # Model Initialization (BASELINE RDN)
    print("="*60)
    print("Using single GPU (DataParallel disabled)")
    print(f"Initializing Baseline RDN_residual_CR...")
    print("="*60)
    
    # We explicitly turn off cross_attn in case default changed
    CR_net = RDN_residual_CR(opts.crop_size, use_cross_attn=False).cuda()
    
    CR_net.eval()
    for p in CR_net.parameters():
        p.requires_grad = False

    # Load checkpoint safely (FIXING THE KEY WRAPPING ISSUE)
    print(f"Loading checkpoint: {opts.checkpoint_path}")
    checkpoint = torch.load(
        opts.checkpoint_path,
        map_location="cuda",
        weights_only=False     # REQUIRED FIX for PyTorch 2.6
    )

    # Handle various dictionary structures
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "network" in checkpoint:
        print("Found 'network' key wrapper - Unwraping...")
        state_dict = checkpoint["network"]
    else:
        state_dict = checkpoint

    # Handle module. prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load Weights
    try:
        CR_net.load_state_dict(state_dict, strict=True)
        print("✓ Loaded weights successfully (Strict mode)")
    except Exception as e:
        print(f"Warning: Strict loading failed ({e}). Trying strict=False...")
        CR_net.load_state_dict(state_dict, strict=False)
        print("✓ Loaded weights (Relaxed mode)")

    print("="*60)
    print(f"Testing Model: BASELINE RDN")
    print(f"Input Data: {opts.input_data_folder}")
    print(f"Data CSV: {opts.data_list_filepath}")
    print("="*60)
    
    avg_psnr, avg_ssim, results_per_image = test(CR_net, opts)

    print("-"*65)
    print("="*60)
    print(f"Average Results (BASELINE):")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  Total Images: {len(results_per_image)}")
    print("="*60)

    # Save results
    results_dir = '/kaggle/working/results_baseline'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = os.path.join(results_dir, f"results_baseline_{timestamp}.json")
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": "Baseline_CRNet",
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "num_images": len(results_per_image),
            "per_image": results_per_image
        }, f, indent=4)

    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
