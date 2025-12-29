import torch
import torch.nn as nn

from model_base import *
from net_CR_RDN import *
from metrics import *
from torch.optim import lr_scheduler

class ModelCRNet(ModelBase):
    def __init__(self, opts):
        super(ModelCRNet, self).__init__()
        self.opts = opts
        
        # Create network
        # Create network
        # Pass use_cross_attn if available in opts
        use_cross_attn = getattr(self.opts, 'use_cross_attn', False)
        print(f"Initializing RDN with Cross-Attention: {use_cross_attn}")
        self.net_G = RDN_residual_CR(self.opts.crop_size, use_cross_attn=use_cross_attn)

        
        # Move to GPU - let the training script handle DataParallel/DDP wrapping
        # Just move to the first GPU for now
        if torch.cuda.is_available():
            self.net_G = self.net_G.cuda()
            print(f"Model moved to GPU")
        else:
            print("Warning: CUDA not available, using CPU")
        
        self.print_networks(self.net_G)

        # Initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
        else:
            raise ValueError(f"Optimizer {self.opts.optimizer} not supported")
            
        self.loss_fn = nn.L1Loss()
                        
    def set_input(self, _input):
        """Set input data and move to appropriate device"""
        inputs = _input
        
        # Get device from model (handles DataParallel/DDP automatically)
        if isinstance(self.net_G, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            device = next(self.net_G.module.parameters()).device
        else:
            device = next(self.net_G.parameters()).device
        
        self.cloudy_data = inputs['cloudy_data'].to(device, non_blocking=True)
        self.cloudfree_data = inputs['cloudfree_data'].to(device, non_blocking=True)
        self.SAR_data = inputs['SAR_data'].to(device, non_blocking=True)
        
    def forward(self):
        """Forward pass through the network"""
        pred_CloudFree_data = self.net_G(self.cloudy_data, self.SAR_data)
        return pred_CloudFree_data

    def optimize_parameters(self, grad_clip=1.0):
        """Perform forward pass, compute loss, and update weights"""
        
        self.pred_Cloudfree_data = self.forward()

        # Compute loss
        loss_output = self.loss_fn(self.pred_Cloudfree_data, self.cloudfree_data)
        
        # Handle both scalar and tuple (loss, dict) returns
        if isinstance(loss_output, tuple):
            self.loss_G = loss_output[0]
            # store detailed losses if needed, but for now just use total for backward
        else:
            self.loss_G = loss_output

        # Backpropagation
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        
        # Clip gradients to prevent exploding gradients
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), max_norm=grad_clip)
        
        self.optimizer_G.step()  

        return self.loss_G.item()


    def get_current_scalars(self):
        """Get current training metrics"""
        losses = {}
        try:
            psnr_val = PSNR(self.pred_Cloudfree_data.data, self.cloudfree_data)
            # Handle tensor/float conversion
            if hasattr(psnr_val, 'item'):
                psnr_val = psnr_val.item()
            losses['PSNR_train'] = float(psnr_val)
        except Exception as e:
            print(f"Warning: Could not compute PSNR: {e}")
            losses['PSNR_train'] = 0.0
        return losses

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        self.save_network(self.net_G, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)