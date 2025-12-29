import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import numpy as np
import json
import math
from datetime import datetime
import time
from tqdm import tqdm
import csv

# Enable cuDNN autotuner for optimal performance
torch.backends.cudnn.benchmark = True

# Fix PyTorch 2.6 UnpicklingError
import torch.serialization
import argparse as _argparse
torch.serialization.add_safe_globals([_argparse.Namespace])

from dataloader import *
from model_CR_net import *
from metrics import *

# Import new advanced losses
try:
    from losses import EnhancedLoss
except ImportError:
    # Fallback or local path import
    try:
        from codes.losses import EnhancedLoss
    except ImportError:
        print("Warning: losses.py not found. Advanced losses disabled.")
        EnhancedLoss = None


##===================================================##
##********** Configure training settings ************##
##===================================================##
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='RDN', 
                    choices=['RDN', 'CrossAttention'],
                    help='Model architecture to use: RDN or CrossAttention')
parser.add_argument('--batch_sz', type=int, default=4, help='batch size PER GPU')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')

parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/image1', help='Path to S1 and S2 data')
parser.add_argument('--data_list_filepath', type=str, default='/kaggle/input/image2/data.csv', help='Path to data.csv')
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)
parser.add_argument('--is_test', type=bool, default=False, help='whether in test mode')

# Enhancement arguments
parser.add_argument('--use_cross_attn', action='store_true', default=True, help='Use Cross-Modal Attention in RDN')
parser.add_argument('--use_fft_loss', action='store_true', default=True, help='Use Frequency Domain Loss')
parser.add_argument('--fft_weight', type=float, default=0.1, help='Weight for FFT loss')
parser.add_argument('--use_contrastive_loss', action='store_true', default=True, help='Use Contrastive Loss')
parser.add_argument('--contrastive_weight', type=float, default=0.05, help='Weight for Contrastive loss')



parser.add_argument('--optimizer', type=str, default='Adam', help='Adam optimizer')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for regularization')
parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['step', 'plateau', 'cosine'], help='learning rate scheduler type')
parser.add_argument('--lr_step', type=int, default=5, help='lr decay step (for step scheduler)')
parser.add_argument('--lr_start_epoch_decay', type=int, default=5, help='epoch to start lr decay (for step scheduler)')
parser.add_argument('--lr_patience', type=int, default=5, help='patience for plateau scheduler')
parser.add_argument('--lr_factor', type=float, default=0.5, help='factor to reduce lr')
parser.add_argument('--max_epochs', type=int, default=10, help='maximum training epochs')
parser.add_argument('--early_stop_patience', type=int, default=10, help='early stopping patience (0 to disable)')
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping threshold (0 to disable)')
parser.add_argument('--warmup_epochs', type=int, default=0, help='number of warmup epochs')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint every N epochs')
parser.add_argument('--save_model_dir', type=str, default='/kaggle/working/checkpoints', help='checkpoint directory')

parser.add_argument('--resume_checkpoint', type=str, default=None, help='path to resume checkpoint')
parser.add_argument('--train_from_scratch', action='store_true', default=True, help='train from scratch')

parser.add_argument('--experiment_name', type=str, default='kaggle_training', help='experiment name')
parser.add_argument('--notes', type=str, default='', help='additional notes')

parser.add_argument('--use_ddp', action='store_true', default=False, help='Use DistributedDataParallel for better multi-GPU')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

opts = parser.parse_args()

##===================================================##
##************** Training functions *****************##
##===================================================##
def validate(model, val_dataloader, device):
    """Validate model on validation set with proper error handling"""
    model.net_G.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        # Handle empty validation set gracefully
        try:
            val_len = len(val_dataloader)
        except Exception:
            val_len = 0

        if val_len == 0:
            print("Warning: validation set is empty. Skipping validation.")
            model.net_G.train()
            return 0.0, 0.0

        # Single progress bar for the entire validation loop
        progress_bar = tqdm(val_dataloader, desc="Validating", unit="batch", disable=(opts.local_rank not in [-1, 0]))
        
        for data in progress_bar:
            try:
                model.set_input(data)
                pred = model.forward()
                
                # Ensure tensors are valid (no NaN or Inf)
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    continue
                
                if torch.isnan(model.cloudfree_data).any() or torch.isinf(model.cloudfree_data).any():
                    continue
                
                batch_psnr = PSNR(pred, model.cloudfree_data)
                batch_ssim = SSIM(pred, model.cloudfree_data)
                
                # Check if metrics are valid (they are now floats)
                if math.isnan(batch_psnr) or math.isinf(batch_psnr) or batch_psnr <= 0:
                    continue
                if isinstance(batch_ssim, torch.Tensor):
                    if torch.isnan(batch_ssim) or torch.isinf(batch_ssim):
                        continue
                    batch_ssim = float(batch_ssim.item())
                elif math.isnan(batch_ssim) or math.isinf(batch_ssim) or batch_ssim < 0 or batch_ssim > 1.1:
                    continue
                
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                num_batches += 1
                
                # Update progress bar with running average
                if num_batches > 0:
                    avg_psnr = total_psnr / num_batches
                    avg_ssim = total_ssim / num_batches
                    progress_bar.set_postfix({'PSNR': f'{avg_psnr:.2f}', 'SSIM': f'{avg_ssim:.4f}'})
            
            except Exception as e:
                # Silently skip problematic batches instead of printing error for every batch
                continue
    
    # Avoid division by zero
    if num_batches == 0:
        print("Warning: No valid batches in validation set")
        model.net_G.train()
        return 0.0, 0.0
    
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    model.net_G.train()
    return avg_psnr, avg_ssim

def save_checkpoint_fn(model, epoch, val_psnr, best_val_psnr, opts, is_ddp=False):
    """Save training checkpoint - GUARANTEED to save every epoch"""
    # Only save from main process in DDP
    if is_ddp and opts.local_rank != 0:
        return False
    
    os.makedirs(opts.save_model_dir, exist_ok=True)
    
    # Get the actual model (unwrap DataParallel/DDP if needed)
    if isinstance(model.net_G, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_state = model.net_G.module.state_dict()
    else:
        model_state = model.net_G.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': model.optimizer_G.state_dict(),
        'lr_scheduler_state_dict': model.lr_scheduler.state_dict(),
        'val_psnr': val_psnr,
        'best_val_psnr': best_val_psnr,
        'epochs_without_improvement': 0 if val_psnr > best_val_psnr else epochs_without_improvement,
        'opts': vars(opts)
    }
    
    # ALWAYS save epoch checkpoint
    checkpoint_path = os.path.join(opts.save_model_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Save best model if this is the best so far
    is_best = val_psnr > best_val_psnr
    if is_best:
        best_path = os.path.join(opts.save_model_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model: {best_path} (PSNR: {val_psnr:.2f} dB)")
    
    return is_best

##===================================================##
##******************** Main *************************##
##===================================================##
if __name__ == '__main__':
    ##===================================================##
    ##*************** Setup DDP if requested ************##
    ##===================================================##
    is_ddp = opts.use_ddp
    local_rank = opts.local_rank
    
    if is_ddp:
        if local_rank == -1:
            # Not launched with torch.distributed.launch, fall back to DataParallel
            print("Warning: --use_ddp specified but not launched with torchrun/torch.distributed.launch")
            print("Falling back to DataParallel mode")
            is_ddp = False
        else:
            # Initialize DDP
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            print(f"DDP initialized: rank {local_rank}/{dist.get_world_size()}")
    
    ##===================================================##
    ##*************** Print configuration ***************##
    ##===================================================##
    if not is_ddp or local_rank in [-1, 0]:
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        for arg in vars(opts):
            print(f"{arg:.<30} {getattr(opts, arg)}")
        print("="*60 + "\n")

    ##===================================================##
    ##*** Set GPU devices ***##
    ##===================================================##
    if not is_ddp:
        # Set all available GPUs for DataParallel
        gpu_count = torch.cuda.device_count()
        print(f"GPUs available: {gpu_count}")
        if gpu_count > 1:
            print(f"Using DataParallel on {gpu_count} GPUs")
        print()

    ##===================================================##
    ##*************** Create dataloader *****************##
    ##===================================================##
    seed_torch()

    # Load train/val/test filelists
    def read_csv_rows(path):
        rows = []
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for r in reader:
                    if len(r) == 0:
                        continue
                    rows.append(r)
        except Exception:
            return []
        return rows

    train_filelist, val_filelist, test_filelist = [], [], []
    basename = os.path.basename(opts.data_list_filepath).lower()
    parent_dir = os.path.dirname(opts.data_list_filepath)

    if 'data.csv' in basename:
        train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    else:
        if 'train' in basename:
            train_filelist = read_csv_rows(opts.data_list_filepath)
            val_path = os.path.join(parent_dir, 'val.csv')
            test_path = os.path.join(parent_dir, 'test.csv')
            if os.path.exists(val_path):
                val_filelist = read_csv_rows(val_path)
            if os.path.exists(test_path):
                test_filelist = read_csv_rows(test_path)
            if len(val_filelist) == 0 or len(test_filelist) == 0:
                combined_path = os.path.join(parent_dir, 'data.csv')
                if os.path.exists(combined_path):
                    t, v, te = get_train_val_test_filelists(combined_path)
                    if len(val_filelist) == 0:
                        val_filelist = v
                    if len(test_filelist) == 0:
                        test_filelist = te
        else:
            train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
            if len(train_filelist) == 0 and len(val_filelist) == 0 and len(test_filelist) == 0:
                all_rows = read_csv_rows(opts.data_list_filepath)
                train_filelist = all_rows

    if not is_ddp or local_rank in [-1, 0]:
        print(f"Training samples: {len(train_filelist)}")
        print(f"Validation samples: {len(val_filelist)}")

    # Training dataloader
    train_data = AlignedDataset(opts, train_filelist)
    
    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=dist.get_world_size(),
            rank=local_rank
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opts.batch_sz,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True if opts.num_workers > 0 else False,
        prefetch_factor=2 if opts.num_workers > 0 else None
    )

    # Validation dataloader
    val_data = AlignedDataset(opts, val_filelist)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=opts.batch_sz,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True if opts.num_workers > 0 else False,
        prefetch_factor=2 if opts.num_workers > 0 else None
    )

    ##===================================================##
    ##****************** Create model *******************##
    ##===================================================##
    device = torch.device(f'cuda:{local_rank}' if is_ddp else 'cuda:0')
    
    if opts.model_name == 'CrossAttention':
        from net_CR_CrossAttention import CloudRemovalCrossAttention
        # Create wrapper for compatibility with existing training loop
        class ModelCRNetCrossAttention:
            def __init__(self, opts, device='cuda'):
                self.device = device
                self.net_G = CloudRemovalCrossAttention().to(device)
                self.optimizer_G = torch.optim.Adam(
                    self.net_G.parameters(), 
                    lr=opts.lr,
                    weight_decay=opts.weight_decay
                )
                # Create appropriate learning rate scheduler
                if opts.lr_scheduler == 'plateau':
                    self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer_G, 
                        mode='max',
                        factor=opts.lr_factor, 
                        patience=opts.lr_patience
                    )
                elif opts.lr_scheduler == 'cosine':
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer_G,
                        T_max=opts.max_epochs,
                        eta_min=1e-7
                    )
                else:  # step
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        self.optimizer_G, step_size=opts.lr_step, gamma=opts.lr_factor
                    )
                self.lr_scheduler_type = opts.lr_scheduler
                
                # Use EnhancedLoss if requested
                use_fft = getattr(opts, 'use_fft_loss', False)
                use_contrastive = getattr(opts, 'use_contrastive_loss', False)
                
                if (use_fft or use_contrastive) and EnhancedLoss is not None:
                    fft_w = opts.fft_weight if use_fft else 0.0
                    cont_w = opts.contrastive_weight if use_contrastive else 0.0
                    print(f"Using EnhancedLoss -> FFT: {fft_w}, Contrastive: {cont_w}")
                    self.criterion = EnhancedLoss(fft_weight=fft_w, contrastive_weight=cont_w)
                else:
                    self.criterion = nn.L1Loss()

                    
                self.first_batch = True
            
            def set_input(self, data):
                # Debug: Print available keys and shapes on first batch
                if self.first_batch:
                    print(f"DEBUG: Available data keys: {list(data.keys())}")
                    for key in data.keys():
                        if isinstance(data[key], torch.Tensor):
                            print(f"  {key}: shape {data[key].shape}")
                    self.first_batch = False
                
                # Handle both old and new key names
                cloudy_key = 'cloudy_optical' if 'cloudy_optical' in data else 'cloudy_data'
                sar_key = 'sar' if 'sar' in data else 'SAR_data'
                cloudfree_key = 'cloudfree_optical' if 'cloudfree_optical' in data else 'cloudfree_data'
                
                self.cloudy_optical = data[cloudy_key].to(self.device)  # (B, 13, H, W)
                self.sar_img = data[sar_key].to(self.device)            # (B, 2, H, W)
                self.cloudfree_data = data[cloudfree_key].to(self.device)  # (B, 13, H, W)
            
            def forward(self):
                self.pred_Cloudfree_data = self.net_G(self.cloudy_optical, self.sar_img)
                return self.pred_Cloudfree_data
            
            def optimize_parameters(self, grad_clip=0.0):
                self.optimizer_G.zero_grad()
                pred = self.forward()
                
                # Handle EnhancedLoss
                if isinstance(self.criterion, EnhancedLoss):
                    loss, loss_dict = self.criterion(pred, self.cloudfree_data, cloudy_input=self.cloudy_optical)
                else:
                    loss = self.criterion(pred, self.cloudfree_data)

                
                loss.backward()
                
                # Gradient clipping if enabled
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), grad_clip)
                
                self.optimizer_G.step()
                return loss.item()
        
        model = ModelCRNetCrossAttention(opts, device)
        model = ModelCRNetCrossAttention(opts, device)
    else:
        # Use default RDN model
        model = ModelCRNet(opts)
        
        # Override criterion for RDN model as well
        if getattr(opts, 'use_fft_loss', False) or getattr(opts, 'use_contrastive_loss', False):
            if EnhancedLoss is not None:
                fft_w = opts.fft_weight if getattr(opts, 'use_fft_loss', False) else 0.0
                cont_w = opts.contrastive_weight if getattr(opts, 'use_contrastive_loss', False) else 0.0
                
                if not is_ddp or local_rank == 0:
                    print(f"Using EnhancedLoss for RDN -> FFT: {fft_w}, Contrastive: {cont_w}")
                model.loss_fn = EnhancedLoss(fft_weight=fft_w, contrastive_weight=cont_w)


    
    # Wrap model for multi-GPU
    if is_ddp:
        model.net_G = nn.parallel.DistributedDataParallel(
            model.net_G,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        if not is_ddp or local_rank == 0:
            print("Model wrapped with DistributedDataParallel")
    elif torch.cuda.device_count() > 1:
        model.net_G = nn.DataParallel(model.net_G)
        print(f"Model wrapped with DataParallel on {torch.cuda.device_count()} GPUs")

    # Resume from checkpoint
    start_epoch = 0
    best_val_psnr = 0.0
    epochs_without_improvement = 0

    if opts.resume_checkpoint and os.path.exists(opts.resume_checkpoint):
        if not is_ddp or local_rank in [-1, 0]:
            print(f"\nResuming from checkpoint: {opts.resume_checkpoint}")
        
        checkpoint = torch.load(
            opts.resume_checkpoint,
            map_location=device,
            weights_only=False
        )
        
        # Load model state
        state_dict = checkpoint['model_state_dict']
        
        # Handle DataParallel/DDP wrapping mismatches
        if isinstance(model.net_G, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            # Loading into wrapped model
            if not any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        else:
            # Loading into non-wrapped model
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.net_G.load_state_dict(state_dict, strict=False)
        model.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_psnr' in checkpoint:
            best_val_psnr = checkpoint['best_val_psnr']
        if 'epochs_without_improvement' in checkpoint:
            epochs_without_improvement = checkpoint['epochs_without_improvement']
        
        if not is_ddp or local_rank in [-1, 0]:
            print(f"Resumed from epoch {start_epoch}, best val PSNR: {best_val_psnr:.2f} dB\n")

    ##===================================================##
    ##**************** Train the network ****************##
    ##===================================================##
    if not is_ddp or local_rank in [-1, 0]:
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")

        # Initialize logging
        training_log = {
            'experiment_name': opts.experiment_name,
            'notes': opts.notes,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': vars(opts),
            'epochs': []
        }
        
        # Initialize log path before training loop
        log_dir = os.path.join(opts.save_model_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{opts.experiment_name}_log.json')

    total_steps = 0
    train_start_time = time.time()
    
    # Learning rate warmup function
    def get_warmup_lr(epoch, warmup_epochs, base_lr):
        if warmup_epochs == 0 or epoch >= warmup_epochs:
            return base_lr
        return base_lr * (epoch + 1) / warmup_epochs

    for epoch in range(start_epoch, opts.max_epochs):
        # Apply warmup if needed
        if epoch < opts.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, opts.warmup_epochs, opts.lr)
            for param_group in model.optimizer_G.param_groups:
                param_group['lr'] = warmup_lr
            if not is_ddp or local_rank in [-1, 0]:
                print(f"Warmup: Setting LR to {warmup_lr:.6f}")
        epoch_start_time = time.time()
        
        if not is_ddp or local_rank in [-1, 0]:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{opts.max_epochs-1}")
            print(f"{'='*60}")
        
        # Set epoch for distributed sampler
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.net_G.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Training Epoch {epoch}",
            unit="batch",
            disable=(is_ddp and local_rank != 0)
        )
        
        for batch_idx, data in enumerate(progress_bar):
            total_steps += 1
            
            try:
                model.set_input(data)
                batch_loss = model.optimize_parameters(grad_clip=opts.grad_clip)
                
                epoch_loss += batch_loss
                num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        # Epoch statistics
        if num_batches > 0:
            avg_train_loss = epoch_loss / num_batches
        else:
            avg_train_loss = 0.0
        
        # Training PSNR on last batch
        try:
            with torch.no_grad():
                avg_train_psnr = PSNR(model.pred_Cloudfree_data, model.cloudfree_data)
                # PSNR now returns a float directly
                if isinstance(avg_train_psnr, torch.Tensor):
                    avg_train_psnr = float(avg_train_psnr.item())
                else:
                    avg_train_psnr = float(avg_train_psnr)
        except Exception:
            avg_train_psnr = 0.0
        
        # Validation (only on main process)
        if not is_ddp or local_rank in [-1, 0]:
            print(f"\nRunning validation...")
            val_psnr, val_ssim = validate(model, val_dataloader, device)
            
            # Check if best model
            is_best = val_psnr > best_val_psnr
            if is_best:
                best_val_psnr = val_psnr
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Update learning rate based on scheduler type
            if epoch >= opts.warmup_epochs:  # Only adjust LR after warmup
                if hasattr(model, 'lr_scheduler_type') and model.lr_scheduler_type == 'plateau':
                    # ReduceLROnPlateau needs the metric
                    model.lr_scheduler.step(val_psnr)
                elif opts.lr_scheduler == 'plateau':
                    model.lr_scheduler.step(val_psnr)
                elif epoch >= opts.lr_start_epoch_decay:
                    # StepLR and CosineAnnealingLR
                    model.lr_scheduler.step()
                
                current_lr = model.optimizer_G.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.6f}")
            
            # ALWAYS save checkpoint every epoch (removed frequency check)
            save_checkpoint_fn(model, epoch, val_psnr, best_val_psnr, opts, is_ddp)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            epoch_log = {
                'epoch': epoch,
                'train_loss': float(avg_train_loss),
                'train_psnr': float(avg_train_psnr),
                'val_psnr': float(val_psnr),
                'val_ssim': float(val_ssim),
                'best_val_psnr': float(best_val_psnr),
                'learning_rate': float(model.optimizer_G.param_groups[0]['lr']),
                'epoch_time': float(epoch_time),
                'is_best': is_best,
                'epochs_without_improvement': epochs_without_improvement
            }
            training_log['epochs'].append(epoch_log)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Train PSNR: {avg_train_psnr:.2f} dB")
            print(f"  Val PSNR:   {val_psnr:.2f} dB")
            print(f"  Val SSIM:   {val_ssim:.4f}")
            print(f"  Best Val PSNR: {best_val_psnr:.2f} dB {'(NEW!)' if is_best else ''}")
            print(f"  Epochs without improvement: {epochs_without_improvement}")
            print(f"  Epoch Time: {epoch_time/60:.1f} minutes")
            print(f"{'='*60}")
            
            # Early stopping check
            if opts.early_stop_patience > 0 and epochs_without_improvement >= opts.early_stop_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered!")
                print(f"No improvement for {epochs_without_improvement} epochs")
                print(f"Best validation PSNR: {best_val_psnr:.2f} dB")
                print(f"{'='*60}\n")
                break
            
            # Save training log
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
        
        # Synchronize all processes
        if is_ddp:
            dist.barrier()

    # Training complete
    if not is_ddp or local_rank in [-1, 0]:
        total_time = time.time() - train_start_time
        training_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        training_log['total_time_hours'] = total_time / 3600

        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Total Time: {total_time/3600:.2f} hours")
        print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
        print(f"Training log saved to: {log_path}")
        best_model_path = os.path.join(opts.save_model_dir, 'best_model.pth')
        print(f"Best model saved to: {best_model_path}")
        print(f"{'='*60}\n")
        
        # ========================================
        # Automatic checkpoint download for Kaggle
        # ========================================
        try:
            import shutil
            
            # Create output directory for Kaggle
            output_dir = '/kaggle/output'
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy best model checkpoint
            output_best_model = os.path.join(output_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, output_best_model)
                print(f"✅ Best model copied to: {output_best_model}")
            
            # Copy entire checkpoints folder
            checkpoints_output = os.path.join(output_dir, 'checkpoints')
            if os.path.exists(opts.save_model_dir):
                if os.path.exists(checkpoints_output):
                    shutil.rmtree(checkpoints_output)
                shutil.copytree(opts.save_model_dir, checkpoints_output)
                print(f"✅ All checkpoints copied to: {checkpoints_output}")
            
            # Copy training log
            if os.path.exists(log_path):
                output_log = os.path.join(output_dir, f'{opts.experiment_name}_log.json')
                shutil.copy(log_path, output_log)
                print(f"✅ Training log copied to: {output_log}")
            
            print(f"\n{'='*60}")
            print("📥 Download Instructions (Kaggle Notebooks)")
            print(f"{'='*60}")
            print("1. Click the 'Output' tab in your Kaggle notebook")
            print("2. You should see:")
            print("   - best_model.pth (best checkpoint)")
            print("   - checkpoints/ (all epoch checkpoints)")
            print(f"   - {opts.experiment_name}_log.json (training logs)")
            print("3. Click the download icon next to each file")
            print(f"\n💾 Files are ready to download from Kaggle Output!")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n⚠️  Warning: Could not copy files to /kaggle/output: {e}")
            print(f"Files are still available at: {opts.save_model_dir}")
            print(f"Manually download from /kaggle/working/checkpoints\n")
    
    # Cleanup DDP
    if is_ddp:
        dist.destroy_process_group()