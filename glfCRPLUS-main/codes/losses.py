import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FFTLoss(nn.Module):
    """
    Frequency Domain Loss using Fast Fourier Transform.
    Pushes the model to recover high-frequency details (textures/edges)
    by minimizing the difference in the frequency domain.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        # Apply 2D FFT
        # pred, target: (B, C, H, W)
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        
        # Compute loss on amplitude only (robust to phase shifts)
        # Or compute distinct loss on amplitude and phase
        
        # Option A: Complex L1 Loss (Real + Imaginary parts)
        # loss = self.criterion(torch.view_as_real(pred_fft), torch.view_as_real(target_fft))
        
        # Option B: Amplitude + Phase Loss
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        loss = self.criterion(pred_amp, target_amp)
        
        return loss * self.loss_weight

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using pre-trained VGG19.
    Ensures that the features of the reconstructed image match the target.
    """
    def __init__(self, layers=['relu3_2', 'relu4_2', 'relu5_2'], weights=[0.2, 0.4, 0.4]):
        super(PerceptualLoss, self).__init__()
        # Load VGG19 pretrained
        vgg = models.vgg19(pretrained=True).features
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg.eval()
        self.layers = layers
        self.weights = weights
        
        # Dictionary to map layer names to indices
        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_2': 17,
            'relu4_2': 26,
            'relu5_2': 35
        }
        
    def forward(self, pred, target):
        loss = 0.0
        x = pred
        y = target
        
        # Normalize input if needed (VGG expects specific mean/std, but often simple 0-1 works for relative loss)
        # For strict correctness we should normalize, but for training stability often skipping is fine if consistent
        
        current_layer = 0
        for i, (name, module) in enumerate(self.vgg._modules.items()):
            x = module(x)
            y = module(y)
            
            # If this layer is one we want
            for j, layer_name in enumerate(self.layers):
                if i == self.layer_map[layer_name]:
                    loss += self.weights[j] * F.l1_loss(x, y)
                    
            if i >= self.layer_map[self.layers[-1]]:
                break
                
        return loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss to pull prediction closer to cloud-free (pos) 
    and push it away from cloudy (neg).
    """
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, pred, target, negative):
        # Positive distance (pred vs cloud-free) -> minimize
        d_pos = F.l1_loss(pred, target)
        
        # Negative distance (pred vs cloudy) -> maximize
        d_neg = F.l1_loss(pred, negative)
        
        # Loss = d_pos + max(0, margin - d_neg)
        loss = d_pos + torch.clamp(self.margin - d_neg, min=0.0)
        return loss

class EnhancedLoss(nn.Module):
    """
    Composite Loss Function for 'Multi-Domain Optimization' Novelty.
    Combines:
    1. L1 Loss (Pixel-wise accuracy)
    2. FFT Loss (Frequency/Texture domain)
    3. Perceptual Loss (Feature domain)
    4. Contrastive Loss (Optional, requires negative sample)
    """
    def __init__(self, fft_weight=0.1, perceptual_weight=0.1, contrastive_weight=0.0):
        super(EnhancedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.fft_loss = FFTLoss(loss_weight=1.0)
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None
        self.contrastive_loss = ContrastiveLoss() if contrastive_weight > 0 else None
        
        self.fft_weight = fft_weight
        self.perceptual_weight = perceptual_weight
        self.contrastive_weight = contrastive_weight
        
    def forward(self, pred, target, cloudy_input=None):
        # Pixel loss (Base)
        loss_l1 = self.l1(pred, target)
        total_loss = loss_l1
        
        loss_dict = {'l1': loss_l1.item()}
        
        # Frequency Loss
        if self.fft_weight > 0:
            loss_fft = self.fft_loss(pred, target)
            total_loss += self.fft_weight * loss_fft
            loss_dict['fft'] = loss_fft.item()
            
        # Perceptual Loss (Feature)
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            # We need to ensure input is adaptable to VGG (3 channels)
            # Sentinel-2 has 13 bands. We can take RGB (B4, B3, B2) for perceptual loss
            # Assuming channel order is standard S2: B1, B2, B3, B4...
            # Indices: B2=1, B3=2, B4=3 (0-indexed) -> RGB = [3, 2, 1]
            try:
                if pred.shape[1] >= 4:
                    pred_rgb = pred[:, [3, 2, 1], :, :]
                    target_rgb = target[:, [3, 2, 1], :, :]
                    loss_perc = self.perceptual_loss(pred_rgb, target_rgb)
                    total_loss += self.perceptual_weight * loss_perc
                    loss_dict['perc'] = loss_perc.item()
            except Exception as e:
                pass # Skip if channel mapping fails
                
        # Contrastive Loss
        if self.contrastive_weight > 0 and self.contrastive_loss is not None and cloudy_input is not None:
            loss_cont = self.contrastive_loss(pred, target, cloudy_input)
            total_loss += self.contrastive_weight * loss_cont
            loss_dict['cont'] = loss_cont.item()
            
        return total_loss, loss_dict
