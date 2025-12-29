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
def test(CR_net, opts, model_name='RDN'):
    """Test the model
    Args:
        CR_net: The model to test
        opts: Configuration options
        model_name: 'RDN' or 'CrossAttention'
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
            # Allow flexible key naming in the batch
            def pick(batch, candidates, label):
                for k in candidates:
                    if k in batch:
                        return batch[k]
                raise KeyError(f"Missing {label} in batch. Available keys: {list(batch.keys())}")

            cloudy_data = pick(inputs, ['cloudy_optical', 'cloudy', 'optical_cloudy', 'input_optical'], 'cloudy_optical').cuda()
            cloudfree_data = pick(inputs, ['cloudfree_optical', 'cloudfree', 'optical_cloudfree', 'target_optical'], 'cloudfree_optical').cuda()
            SAR_data = pick(inputs, ['sar', 'SAR', 'sar_data', 'input_sar'], 'sar').cuda()
            file_names = inputs['file_name']

            # Handle different model forward signatures
            if model_name == 'CrossAttention':
                pred = CR_net(cloudy_data, SAR_data)
            else:
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

    parser.add_argument('--model_type', type=str, default='RDN',
                        choices=['RDN', 'CrossAttention'],
                        help='Model type to test')
    parser.add_argument('--batch_sz', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter')
    parser.add_argument('--data_list_filepath', type=str, default='/kaggle/working/data.csv')
    parser.add_argument('--checkpoint_path', type=str, default='/kaggle/input/checkpoint3/pytorch/default/1/checkpoint_epoch_2.pth')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)

    parser.add_argument('--model_name', type=str, default='baseline')
    parser.add_argument('--notes', type=str, default='')

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
    else:
        if not os.path.exists(opts.data_list_filepath):
            raise FileNotFoundError(f"CSV not found: {opts.data_list_filepath}")

    # Model
    print("="*60)
    print("Using single GPU (DataParallel disabled)")
    print("="*60)

    # Load model based on type
    if opts.model_type == 'CrossAttention':
        from net_CR_CrossAttention import CloudRemovalCrossAttention
        CR_net = CloudRemovalCrossAttention().cuda()
    else:
        # Default RDN model
        CR_net = RDN_residual_CR(opts.crop_size).cuda()
    
    CR_net.eval()
    for p in CR_net.parameters():
        p.requires_grad = False

    # Load checkpoint safely
    print(f"Loading checkpoint: {opts.checkpoint_path}")
    checkpoint = torch.load(
        opts.checkpoint_path,
        map_location="cuda",
        weights_only=False     # REQUIRED FIX for PyTorch 2.6
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "network" in checkpoint:
        state_dict = checkpoint["network"]
    else:
        state_dict = checkpoint

    CR_net.load_state_dict(state_dict, strict=False)

    print("="*60)
    print(f"Testing Model: {opts.model_type} - {opts.model_name}")
    print(f"Input Data: {opts.input_data_folder}")
    print(f"Data CSV: {opts.data_list_filepath}")
    print(f"Checkpoint: {opts.checkpoint_path}")
    print("="*60)
    print(f"{'Image':40s} | {'PSNR':>10s} | {'SSIM':>8s}")
    print("-"*65)

    avg_psnr, avg_ssim, results_per_image = test(CR_net, opts, model_name=opts.model_type)

    print("-"*65)
    print("="*60)
    print(f"Average Results:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  Total Images: {len(results_per_image)}")
    print("="*60)

    # Save results
    results_dir = '/kaggle/working/results'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = os.path.join(results_dir, f"results_{opts.model_name}_{timestamp}.json")
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": opts.model_name,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "num_images": len(results_per_image),
            "per_image": results_per_image
        }, f, indent=4)

    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()