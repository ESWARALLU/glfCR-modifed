import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.fft
import torchvision.models as models

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Image Restoration.
    Pulls the restored image closer to the clear image (positive) 
    and pushes it away from the cloudy image (negative) in feature space.
    Uses pre-trained VGG19 features.
    """
    def __init__(self, weight=0.1, ablate_negative=False):
        super(ContrastiveLoss, self).__init__()
        self.weight = weight
        self.ablate_negative = ablate_negative
        
        # Load VGG19
        vgg = models.vgg19(pretrained=True)
        # Use first few layers for feature extraction (up to relu3_1)
        self.vgg_features = nn.Sequential(*list(vgg.features.children())[:12]).eval()
        
        # Freeze VGG parameters
        for param in self.vgg_features.parameters():
            param.requires_grad = False
            
        self.l1 = nn.L1Loss()

    def forward(self, restored, clear, cloudy):
        # Normalize/Preprocess if needed (assuming inputs are [0,1])
        # VGG expects normalized inputs, but for contrastive relative distance 
        # on consistent inputs, raw [0,1] is often acceptable or simple normalization.
        # Here we apply simple mean centering approx.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(restored.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(restored.device)
        
        # Select first 3 channels (RGB) if input is >3 channels (Sentinel-2 is 13)
        # We use bands [3,2,1] for RGB (approximate indices for Sentinel-2: R=3, G=2, B=1)
        if restored.shape[1] > 3:
            res_rgb = restored[:, [3,2,1], :, :]
            clear_rgb = clear[:, [3,2,1], :, :]
            cld_rgb = cloudy[:, [3,2,1], :, :]
        else:
            res_rgb = restored
            clear_rgb = clear
            cld_rgb = cloudy
            
        res_rgb = (res_rgb - mean) / std
        clear_rgb = (clear_rgb - mean) / std
        cld_rgb = (cld_rgb - mean) / std
        
        # Extract features
        res_feat = self.vgg_features(res_rgb)
        clear_feat = self.vgg_features(clear_rgb)
        cld_feat = self.vgg_features(cld_rgb)
        
        # Calculate distances
        d_pos = self.l1(res_feat, clear_feat)
        d_neg = self.l1(res_feat, cld_feat)
        
        # Contrastive loss: minimize d_pos, maximize d_neg
        # loss = d_pos / (d_neg + epsilon)
        loss = d_pos / (d_neg + 1e-7)
        
        return self.weight * loss

class FFTLoss(nn.Module):
    """
    Frequency Domain Loss using Fast Fourier Transform.
    Calculates L1 loss between the FFT of the prediction and the target.
    This encourages the model to recover high-frequency details.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        # Compute 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # Compute L1 loss in frequency domain
        # We separate real and imaginary parts for stable gradient calculation
        loss = self.l1_loss(pred_fft.real, target_fft.real) + \
               self.l1_loss(pred_fft.imag, target_fft.imag)
               
        return self.loss_weight * loss

class EnhancedLoss(nn.Module):
    """
    Enhanced loss for GLF-CR:
    - L1 loss (base)
    - FFT loss (frequency domain)
    - Contrastive loss (feature domain)
    """
    def __init__(self, l1_weight=1.0, fft_weight=0.1, contrastive_weight=0.0):
        super(EnhancedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.fft_weight = fft_weight
        self.contrastive_weight = contrastive_weight
        
        self.l1_loss = nn.L1Loss()
        
        if fft_weight > 0:
            self.fft_loss = FFTLoss(loss_weight=1.0)
        else:
            self.fft_loss = None
            
        if contrastive_weight > 0:
            self.contrastive_loss = ContrastiveLoss(weight=1.0)
        else:
            self.contrastive_loss = None
        
    def forward(self, pred, target, cloudy_input=None):
        loss_dict = {}
        
        # Base L1 loss
        l1 = self.l1_loss(pred, target)
        loss_dict['l1'] = l1.item()
        
        total_loss = self.l1_weight * l1
        
        # FFT loss
        if self.fft_loss is not None:
            fft = self.fft_loss(pred, target)
            loss_dict['fft'] = fft.item()
            total_loss += self.fft_weight * fft
            
        # Contrastive loss
        if self.contrastive_loss is not None and cloudy_input is not None:
            # We need the cloudy input (negative sample) for contrastive loss
            cont = self.contrastive_loss(pred, target, cloudy_input)
            loss_dict['contrastive'] = cont.item()
            total_loss += self.contrastive_weight * cont
            
        return total_loss, loss_dict

