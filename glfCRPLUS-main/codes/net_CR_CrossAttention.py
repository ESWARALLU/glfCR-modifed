"""
Cloud Removal Network with Global Cross-Attention and Speckle-Aware Gating
Architecture: Dual Encoders (Optical + SAR) -> Cross-Attention Fusion -> Speckle-Aware Gating -> Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> ReLU -> Conv -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class OpticalEncoder(nn.Module):
    """Optical image encoder with 3 levels of downsampling"""
    def __init__(self):
        super(OpticalEncoder, self).__init__()
        # E1: input (13, H, W) -> output (64, H, W)
        # Optical data has 13 channels (Sentinel-2 bands)
        self.E1 = ConvBlock(13, 64, kernel_size=3, stride=1, padding=1)
        
        # Downsample 1
        self.down1 = nn.MaxPool2d(2, 2)
        
        # E2: input (64, H/2, W/2) -> output (128, H/2, W/2)
        self.E2 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Downsample 2
        self.down2 = nn.MaxPool2d(2, 2)
        
        # E3: input (128, H/4, W/4) -> output (256, H/4, W/4)
        self.E3 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # E1 features
        feat_opt_1 = self.E1(x)  # (B, 64, H, W)
        
        # Downsample to E2 level
        x = self.down1(feat_opt_1)
        
        # E2 features
        feat_opt_2 = self.E2(x)  # (B, 128, H/2, W/2)
        
        # Downsample to E3 level
        x = self.down2(feat_opt_2)
        
        # E3 features
        feat_opt_3 = self.E3(x)  # (B, 256, H/4, W/4)
        
        return feat_opt_1, feat_opt_2, feat_opt_3


class SAREncoder(nn.Module):
    """SAR image encoder with 3 levels of downsampling"""
    def __init__(self):
        super(SAREncoder, self).__init__()
        # E1: input (2, H, W) -> output (64, H, W)
        # SAR data has 2 channels (VV and VH polarizations)
        self.E1 = ConvBlock(2, 64, kernel_size=3, stride=1, padding=1)
        
        # Downsample 1
        self.down1 = nn.MaxPool2d(2, 2)
        
        # E2: input (64, H/2, W/2) -> output (128, H/2, W/2)
        self.E2 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Downsample 2
        self.down2 = nn.MaxPool2d(2, 2)
        
        # E3: input (128, H/4, W/4) -> output (256, H/4, W/4)
        self.E3 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # E1 features
        feat_sar_1 = self.E1(x)  # (B, 64, H, W)
        
        # Downsample to E2 level
        x = self.down1(feat_sar_1)
        
        # E2 features
        feat_sar_2 = self.E2(x)  # (B, 128, H/2, W/2)
        
        # Downsample to E3 level
        x = self.down2(feat_sar_2)
        
        # E3 features
        feat_sar_3 = self.E3(x)  # (B, 256, H/4, W/4)
        
        return feat_sar_1, feat_sar_2, feat_sar_3


class GlobalCrossAttention(nn.Module):
    """Global cross-attention mechanism for fusing optical and SAR features"""
    def __init__(self, dim=256, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(GlobalCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Query from optical, Key and Value from SAR
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, feat_opt, feat_sar):
        """
        Args:
            feat_opt: (B, C, H, W) - optical features
            feat_sar: (B, C, H, W) - SAR features
        Returns:
            cross_attn_out: (B, C, H, W) - cross-attention refined optical features
        """
        B, C, H, W = feat_opt.shape
        N = H * W
        
        # Reshape to (B, N, C)
        feat_opt_flat = feat_opt.flatten(2).transpose(1, 2)  # (B, N, C)
        feat_sar_flat = feat_sar.flatten(2).transpose(1, 2)  # (B, N, C)
        
        # Generate Q from optical, K and V from SAR
        q = self.q_proj(feat_opt_flat)  # (B, N, C)
        k = self.k_proj(feat_sar_flat)  # (B, N, C)
        v = self.v_proj(feat_sar_flat)  # (B, N, C)
        
        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v)  # (B, num_heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Reshape back to (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class GateNet(nn.Module):
    """Speckle-Aware Gating Network using SAR features to create attention gate"""
    def __init__(self, dim=256):
        super(GateNet, self).__init__()
        # Simple gating: SAR -> Conv -> Sigmoid
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, feat_sar):
        """
        Args:
            feat_sar: (B, C, H, W) - SAR features
        Returns:
            gate: (B, C, H, W) - gating mask in range [0, 1]
        """
        gate = self.gate_conv(feat_sar)
        return gate


class SpeckleAwareGatingModule(nn.Module):
    """Speckle-aware gating that uses SAR to gate optical features"""
    def __init__(self, dim=256):
        super(SpeckleAwareGatingModule, self).__init__()
        self.gate_net = GateNet(dim)
        
    def forward(self, cross_out, feat_sar):
        """
        Args:
            cross_out: (B, C, H, W) - output from cross-attention
            feat_sar: (B, C, H, W) - SAR features
        Returns:
            gated_out: (B, C, H, W) - gated features
        """
        sigma = self.gate_net(feat_sar)  # (B, C, H, W)
        gated_out = cross_out * sigma  # Element-wise multiplication (Hadamard product)
        return gated_out


class Refinement(nn.Module):
    """Refinement block: fuses gated output with original optical features"""
    def __init__(self, dim=256):
        super(Refinement, self).__init__()
        self.refine_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        
    def forward(self, gated_out, feat_opt_3):
        """
        Args:
            gated_out: (B, C, H, W) - gated cross-attention output
            feat_opt_3: (B, C, H, W) - original optical E3 features
        Returns:
            refined: (B, C, H, W) - refined features
        """
        # Concatenate and fuse
        concat = torch.cat([gated_out, feat_opt_3], dim=1)  # (B, 2C, H, W)
        refined = self.refine_conv(concat)
        return refined


class Decoder(nn.Module):
    """Symmetric decoder with upsampling and skip connections"""
    def __init__(self):
        super(Decoder, self).__init__()
        # UpLevel-1: E3 scale -> E2 scale
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        
        # UpLevel-2: E2 scale -> E1 scale
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        
        # Final output layer: E1 scale -> cloud-free image
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1, padding=0)
        
    def forward(self, refined_3, feat_opt_2, feat_opt_1):
        """
        Args:
            refined_3: (B, 256, H/4, W/4) - refined E3 features
            feat_opt_2: (B, 128, H/2, W/2) - encoder E2 features
            feat_opt_1: (B, 64, H, W) - encoder E1 features
        Returns:
            output: (B, 3, H, W) - cloud-free optical image
        """
        # UpLevel-1: upsample to E2 scale
        x = self.up1(refined_3)  # (B, 256, H/2, W/2)
        x = torch.cat([x, feat_opt_2], dim=1)  # (B, 256+128, H/2, W/2)
        x = self.dec1(x)  # (B, 128, H/2, W/2)
        
        # UpLevel-2: upsample to E1 scale
        x = self.up2(x)  # (B, 128, H, W)
        x = torch.cat([x, feat_opt_1], dim=1)  # (B, 128+64, H, W)
        x = self.dec2(x)  # (B, 64, H, W)
        
        # Final output
        output = self.final_conv(x)  # (B, 3, H, W)
        
        return output


class CloudRemovalCrossAttention(nn.Module):
    """
    Complete Cloud Removal Network with Cross-Attention and Speckle-Aware Gating
    
    Architecture:
    1. Dual Encoders: Separate optical (3ch) and SAR (1ch) encoders
    2. Global Cross-Attention: SAR guides optical feature refinement
    3. Speckle-Aware Gating: SAR-based gating for noise mitigation
    4. Refinement: Fuses gated features with original optical
    5. Symmetric Decoder: Upsampling with skip connections
    """
    
    def __init__(self, num_heads=8, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super(CloudRemovalCrossAttention, self).__init__()
        
        # Dual encoders
        self.optical_encoder = OpticalEncoder()
        self.sar_encoder = SAREncoder()
        
        # Global cross-attention fusion at E3 scale
        self.cross_attn = GlobalCrossAttention(
            dim=256, 
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # Speckle-aware gating module
        self.speckle_gating = SpeckleAwareGatingModule(dim=256)
        
        # Refinement block
        self.refinement = Refinement(dim=256)
        
        # Decoder
        self.decoder = Decoder()
        
    def forward(self, optical_img, sar_img):
        """
        Args:
            optical_img: (B, 3, H, W) - cloudy optical image
            sar_img: (B, 1, H, W) - co-registered SAR image
        Returns:
            output: (B, 3, H, W) - cloud-free optical image
        """
        # Encode optical and SAR images
        feat_opt_1, feat_opt_2, feat_opt_3 = self.optical_encoder(optical_img)
        feat_sar_1, feat_sar_2, feat_sar_3 = self.sar_encoder(sar_img)
        
        # Global cross-attention fusion at E3 scale
        cross_out = self.cross_attn(feat_opt_3, feat_sar_3)  # (B, 256, H/4, W/4)
        
        # Speckle-aware gating
        gated_out = self.speckle_gating(cross_out, feat_sar_3)  # (B, 256, H/4, W/4)
        
        # Refinement: fuse with original optical features
        refined_3 = self.refinement(gated_out, feat_opt_3)  # (B, 256, H/4, W/4)
        
        # Symmetric decoder with skip connections
        output = self.decoder(refined_3, feat_opt_2, feat_opt_1)  # (B, 3, H, W)
        
        return output


# Model factory function
def create_model(pretrained=False, **kwargs):
    """
    Create CloudRemovalCrossAttention model
    Args:
        pretrained: bool - whether to load pretrained weights (not implemented)
        **kwargs: additional arguments for model initialization
    Returns:
        model: CloudRemovalCrossAttention instance
    """
    model = CloudRemovalCrossAttention(**kwargs)
    return model


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    height, width = 512, 512
    
    optical_img = torch.randn(batch_size, 3, height, width)  # Cloudy optical image
    sar_img = torch.randn(batch_size, 1, height, width)      # SAR image
    
    model = CloudRemovalCrossAttention()
    output = model(optical_img, sar_img)
    
    print(f"Input optical shape: {optical_img.shape}")
    print(f"Input SAR shape: {sar_img.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
