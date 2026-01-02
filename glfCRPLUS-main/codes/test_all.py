import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataloader import AlignedDataset, get_train_val_test_filelists
from metrics import PSNR
from net_CR_CrossAttention import CloudRemovalCrossAttention
from net_CR_RDN import RDN_residual_CR


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().float().clamp(0, 1)
    if t.size(0) > 3:
        t = t[:3]
    elif t.size(0) == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


def spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_flat = pred.flatten(2)
    target_flat = target.flatten(2)
    dot = (pred_flat * target_flat).sum(dim=1)
    denom = torch.norm(pred_flat, dim=1) * torch.norm(target_flat, dim=1)
    cos = torch.clamp(dot / (denom + eps), -1.0, 1.0)
    angles = torch.acos(cos)
    return float(torch.rad2deg(angles).mean().item())


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor) -> float:
    (_, channel, _, _) = img1.size()
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
    gauss = torch.exp(-((coords - window_size / 2) ** 2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return {
        "psnr": float(PSNR(pred, target)),
        "ssim": ssim_torch(pred, target),
        "sam": spectral_angle_mapper(pred, target),
        "rmse": rmse(pred, target),
    }


def load_model(model_key: str, checkpoint: str, device: torch.device, crop_size: int):
    if model_key == "base":
        model = RDN_residual_CR(crop_size).to(device)
    else:
        model = CloudRemovalCrossAttention().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "network" in state:
        state = state["network"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_eval(models: dict, loader, device: torch.device, save_dir: Path, save_images: bool, save_every: int):
    per_model = {k: [] for k in models}
    counters = {k: 0 for k in models}

    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        cloudy = batch["cloudy_data"].to(device)
        target = batch["cloudfree_data"].to(device)
        sar = batch["SAR_data"].to(device)
        fname = batch["file_name"][0]

        for key, model in models.items():
            with torch.no_grad():
                pred = model(cloudy, sar)
            metrics = compute_metrics(pred, target)
            per_model[key].append({"image": fname, **metrics})
            counters[key] += 1

            should_save = save_images and (save_every <= 0 or counters[key] % save_every == 0)
            if should_save:
                img = tensor_to_image(pred)
                base = os.path.splitext(os.path.basename(fname))[0]
                out_path = save_dir / key / f"{base}_pred.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path)

    return per_model


def summarize(per_model_results: dict) -> dict:
    summary = {}
    for key, items in per_model_results.items():
        if not items:
            continue
        avg = {
            "psnr": sum(i["psnr"] for i in items) / len(items),
            "ssim": sum(i["ssim"] for i in items) / len(items),
            "sam": sum(i["sam"] for i in items) / len(items),
            "rmse": sum(i["rmse"] for i in items) / len(items),
        }
        summary[key] = {"avg": avg, "count": len(items)}
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate dataset for base and CrossAttention models")
    parser.add_argument("--data_list_filepath", type=str, default="../data/data.csv")
    parser.add_argument("--input_data_folder", type=str, default="../data")
    parser.add_argument("--models", nargs="+", default=["base", "ours"], choices=["base", "ours"], help="Models to evaluate")
    parser.add_argument("--base_checkpoint", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--ours_checkpoint", type=str, required=True, help="Path to our (CrossAttention) checkpoint")
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="../results/test_all")
    parser.add_argument("--save_images", action="store_true", help="Save predicted images")
    parser.add_argument("--save_every", type=int, default=0, help="Save every N images when --save_images is set (0 means all)")
    parser.add_argument("--is_use_cloudmask", action="store_true", help="Enable cloud mask if available")
    parser.add_argument("--cloud_threshold", type=float, default=0.2)
    parser.add_argument("--is_test", type=bool, default=True)
    parser.add_argument("--use_all_splits", action="store_true", help="If set, evaluate train+val+test together")
    opts = parser.parse_args()

    train_files, val_files, test_files = get_train_val_test_filelists(opts.data_list_filepath)

    if opts.use_all_splits:
        combined = train_files + val_files + test_files
        if len(combined) == 0:
            raise RuntimeError("No entries found across splits")
        dataset = AlignedDataset(opts, combined)
    else:
        if len(test_files) == 0:
            raise RuntimeError("No test files found in data list")
        dataset = AlignedDataset(opts, test_files)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opts.batch_sz, shuffle=False, num_workers=opts.num_workers)

    device = torch.device(opts.device if opts.device == "cpu" or torch.cuda.is_available() else "cpu")

    models = {}
    if "base" in opts.models:
        models["base"] = load_model("base", opts.base_checkpoint, device, opts.crop_size)
    if "ours" in opts.models:
        models["ours"] = load_model("ours", opts.ours_checkpoint, device, opts.crop_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(opts.output_dir) / timestamp

    results = run_eval(models, loader, device, save_dir, opts.save_images, opts.save_every)
    summary = summarize(results)

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metrics_per_image.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(save_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Write per-image metrics to CSV
    csv_path = save_dir / "metrics_per_image.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "image", "psnr", "ssim", "sam", "rmse"])
        for model_key, rows in results.items():
            for row in rows:
                writer.writerow([model_key, row["image"], row["psnr"], row["ssim"], row["sam"], row["rmse"]])

    # Write summary averages to CSV
    summary_csv_path = save_dir / "metrics_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "count", "psnr", "ssim", "sam", "rmse"])
        for model_key, info in summary.items():
            avg = info["avg"]
            writer.writerow([model_key, info["count"], avg["psnr"], avg["ssim"], avg["sam"], avg["rmse"]])

    print("\n==== Dataset Summary ====")
    for key, info in summary.items():
        avg = info["avg"]
        print(f"{key}: n={info['count']} PSNR {avg['psnr']:.4f} SSIM {avg['ssim']:.4f} SAM {avg['sam']:.4f} RMSE {avg['rmse']:.6f}")
    print(f"Saved metrics to {save_dir}")


if __name__ == "__main__":
    main()
