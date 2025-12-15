import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import json
from datetime import datetime
import time

# Enable cuDNN autotuner for optimal performance
torch.backends.cudnn.benchmark = True

from dataloader import *
from model_CR_net import *
from metrics import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='RDN', 
                    choices=['RDN', 'CrossAttention'],
                    help='Model architecture to use: RDN or CrossAttention')
parser.add_argument('--batch_sz', type=int, default=4, help='batch size used for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')

parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--input_data_folder', type=str, default='../data')
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)
parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')
parser.add_argument('--is_test', type=bool, default=False, help='whether in test mode')

parser.add_argument('--optimizer', type=str, default='Adam', help='Adam optimizer')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_step', type=int, default=5, help='lr decay step')
parser.add_argument('--lr_start_epoch_decay', type=int, default=5, help='epoch to start lr decay')
parser.add_argument('--max_epochs', type=int, default=5, help='maximum training epochs')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint every N epochs')
parser.add_argument('--save_model_dir', type=str, default='./checkpoints', help='checkpoint directory')

parser.add_argument('--resume_checkpoint', type=str, default=None, help='path to resume checkpoint')
parser.add_argument('--train_from_scratch', action='store_true', default=True, help='train from scratch (no pretrained weights)')

parser.add_argument('--experiment_name', type=str, default='train_from_scratch', help='experiment name for logging')
parser.add_argument('--notes', type=str, default='', help='additional notes')

parser.add_argument('--gpu_ids', type=str, default='0')

opts = parser.parse_args()

##===================================================##
##************** Training functions *****************##
##===================================================##
def validate(model, val_dataloader):
    """Validate model on validation set"""
    model.net_G.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in val_dataloader:
            model.set_input(data)
            pred = model.forward()
            
            batch_psnr = PSNR(pred, model.cloudfree_data)
            batch_ssim = SSIM(pred, model.cloudfree_data)
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_batches += 1
    
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    model.net_G.train()
    return avg_psnr, avg_ssim

def save_checkpoint_fn(model, epoch, val_psnr, best_val_psnr, opts):
    """Save training checkpoint"""
    os.makedirs(opts.save_model_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.net_G.state_dict(),
        'optimizer_state_dict': model.optimizer_G.state_dict(),
        'lr_scheduler_state_dict': model.lr_scheduler.state_dict(),
        'val_psnr': val_psnr,
        'best_val_psnr': best_val_psnr,
        'opts': opts
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(opts.save_model_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    is_best = val_psnr > best_val_psnr
    if is_best:
        best_path = os.path.join(opts.save_model_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path} (PSNR: {val_psnr:.2f} dB)")
    
    return is_best

##===================================================##
##******************** Main *************************##
##===================================================##
if __name__ == '__main__':
    ##===================================================##
    ##*************** Print configuration ***************##
    ##===================================================##
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    for arg in vars(opts):
        print(f"{arg:.<30} {getattr(opts, arg)}")
    print("="*60 + "\n")

    ##===================================================##
    ##****************** Choose GPU *********************##
    ##===================================================##
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

    ##===================================================##
    ##*************** Create dataloader *****************##
    ##===================================================##
    seed_torch()

    train_filelist, val_filelist, _ = get_train_val_test_filelists(opts.data_list_filepath)

    print(f"Training samples: {len(train_filelist)}")
    print(f"Validation samples: {len(val_filelist)}")

    # Training dataloader with optimizations
    train_data = AlignedDataset(opts, train_filelist)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opts.batch_sz,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True,  # Avoid worker respawn overhead
        prefetch_factor=2  # Prefetch batches for better I/O overlap
    )

    # Validation dataloader
    val_data = AlignedDataset(opts, val_filelist)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=opts.batch_sz,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    ##===================================================##
    ##****************** Create model *******************##
    ##===================================================##
    if opts.model_name == 'CrossAttention':
        from net_CR_CrossAttention import CloudRemovalCrossAttention
        # Create wrapper for compatibility with existing training loop
        class ModelCRNetCrossAttention:
            def __init__(self, opts, device='cuda'):
                self.device = device
                self.net_G = CloudRemovalCrossAttention().to(device)
                self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr)
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer_G, step_size=opts.lr_step, gamma=0.5
                )
                self.criterion = nn.L1Loss()
                self.first_batch = True
            
            def set_input(self, data):
                # Debug: Print available keys on first batch
                if self.first_batch:
                    print(f"DEBUG: Available data keys: {list(data.keys())}")
                    self.first_batch = False
                
                # Handle both old and new key names
                cloudy_key = 'cloudy_optical' if 'cloudy_optical' in data else 'cloudy_data'
                sar_key = 'sar' if 'sar' in data else 'SAR_data'
                cloudfree_key = 'cloudfree_optical' if 'cloudfree_optical' in data else 'cloudfree_data'
                
                self.cloudy_optical = data[cloudy_key].to(self.device)
                self.sar_img = data[sar_key].to(self.device)
                self.cloudfree_data = data[cloudfree_key].to(self.device)
            
            def forward(self):
                self.pred_Cloudfree_data = self.net_G(self.cloudy_optical, self.sar_img)
                return self.pred_Cloudfree_data
            
            def optimize_parameters(self):
                self.optimizer_G.zero_grad()
                pred = self.forward()
                loss = self.criterion(pred, self.cloudfree_data)
                loss.backward()
                self.optimizer_G.step()
                return loss.item()
        
        model = ModelCRNetCrossAttention(opts)
    else:
        # Use default RDN model
        model = ModelCRNet(opts)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_psnr = 0.0

    if opts.resume_checkpoint and os.path.exists(opts.resume_checkpoint):
        print(f"\nResuming from checkpoint: {opts.resume_checkpoint}")
        checkpoint = torch.load(opts.resume_checkpoint)
        model.net_G.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_psnr' in checkpoint:
            best_val_psnr = checkpoint['best_val_psnr']
        print(f"Resuming from epoch {start_epoch}, best val PSNR: {best_val_psnr:.2f} dB\n")
    elif not opts.train_from_scratch:
        print("Warning: No checkpoint found but train_from_scratch=False")

    ##===================================================##
    ##**************** Train the network ****************##
    ##===================================================##
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

    total_steps = 0
    train_start_time = time.time()

    for epoch in range(start_epoch, opts.max_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{opts.max_epochs-1}")
        print(f"{'='*60}")
        
        model.net_G.train()
        
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_dataloader):
            total_steps += 1
            
            model.set_input(data)
            batch_loss = model.optimize_parameters()
            
            epoch_loss += batch_loss
            num_batches += 1
            
            # Print progress every batch with simple progress indicator
            if (batch_idx + 1) % 1 == 0:
                progress_pct = 100 * (batch_idx + 1) / len(train_dataloader)
                # Simple progress bar: [=====>    ]
                bar_length = 30
                filled = int(bar_length * (batch_idx + 1) / len(train_dataloader))
                bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
                print(f"\r  [{bar}] {progress_pct:5.1f}% ({batch_idx+1}/{len(train_dataloader)})", end='', flush=True)
        
        # Epoch statistics
        print()  # New line after progress bar
        avg_train_loss = epoch_loss / num_batches
        
        # Calculate training PSNR on last batch only (for speed)
        with torch.no_grad():
            avg_train_psnr = PSNR(model.pred_Cloudfree_data, model.cloudfree_data)
        
        # Validation (only on last epoch to save time)
        if epoch == opts.max_epochs - 1:
            print(f"\nRunning validation...")
            val_psnr, val_ssim = validate(model, val_dataloader)
        else:
            print(f"\nSkipping validation (will validate on last epoch)")
            val_psnr, val_ssim = 0.0, 0.0
        
        # Update learning rate
        if epoch >= opts.lr_start_epoch_decay:
            model.lr_scheduler.step()
            current_lr = model.optimizer_G.param_groups[0]['lr']
            print(f"Learning rate updated: {current_lr:.6f}")
        
        # Check if best model
        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr
        
        # Save checkpoint at configured frequency and always on last epoch
        if (epoch % opts.save_freq == 0) or (epoch == opts.max_epochs - 1):
            save_checkpoint_fn(model, epoch, val_psnr, best_val_psnr, opts)
        
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
            'is_best': is_best
        }
        training_log['epochs'].append(epoch_log)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train PSNR: {avg_train_psnr:.2f} dB")
        print(f"  Val PSNR:   {val_psnr:.2f} dB")
        print(f"  Val SSIM:   {val_ssim:.4f}")
        print(f"  Best Val PSNR: {best_val_psnr:.2f} dB {'(NEW!)' if is_best else ''}")
        print(f"  Epoch Time: {epoch_time/60:.1f} minutes")
        print(f"{'='*60}")
        
        # Save training log after each epoch
        log_dir = os.path.join(opts.save_model_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{opts.experiment_name}_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

    # Training complete
    total_time = time.time() - train_start_time
    training_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_log['total_time_hours'] = total_time / 3600

    # Final log save
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Training log saved to: {log_path}")
    print(f"Best model saved to: {os.path.join(opts.save_model_dir, 'best_model.pth')}")
    print(f"{'='*60}\n")
