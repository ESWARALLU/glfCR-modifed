import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Cross-attention between optical and SAR features.
    Maintains GLF-CR's dual-stream architecture while adding explicit attention.
    """
    def __init__(self, dim=96, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Optical queries, SAR keys/values
        self.q_optical = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_sar = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        # SAR queries, optical keys/values
        self.q_sar = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_optical = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Projection and fusion
        self.proj_optical = nn.Linear(dim, dim)
        self.proj_sar = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Gating mechanism (inspired by GLF-CR's fusion gates)
        self.gate_optical = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_sar = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, optical_feat, sar_feat):
        """
        Args:
            optical_feat: [B, N, C] optical features
            sar_feat: [B, N, C] SAR features
        Returns:
            enhanced_optical: [B, N, C]
            enhanced_sar: [B, N, C]
        """
        B, N, C = optical_feat.shape
        
        # Optical attends to SAR
        q_opt = self.q_optical(optical_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_sar = self.kv_sar(sar_feat).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_sar, v_sar = kv_sar[0], kv_sar[1]
        
        attn_opt = (q_opt @ k_sar.transpose(-2, -1)) * self.scale
        attn_opt = attn_opt.softmax(dim=-1)
        attn_opt = self.attn_drop(attn_opt)
        
        optical_attended = (attn_opt @ v_sar).transpose(1, 2).reshape(B, N, C)
        optical_attended = self.proj_drop(self.proj_optical(optical_attended))
        
        # SAR attends to optical
        q_sar = self.q_sar(sar_feat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_opt = self.kv_optical(optical_feat).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_opt, v_opt = kv_opt[0], kv_opt[1]
        
        attn_sar = (q_sar @ k_opt.transpose(-2, -1)) * self.scale
        attn_sar = attn_sar.softmax(dim=-1)
        attn_sar = self.attn_drop(attn_sar)
        
        sar_attended = (attn_sar @ v_opt).transpose(1, 2).reshape(B, N, C)
        sar_attended = self.proj_drop(self.proj_sar(sar_attended))
        
        # Gated fusion (inspired by GLF-CR's gate mechanism)
        gate_opt = self.gate_optical(optical_attended)
        gate_sar = self.gate_sar(sar_attended)
        
        enhanced_optical = optical_feat + gate_opt * optical_attended
        enhanced_sar = sar_feat + gate_sar * sar_attended
        
        return enhanced_optical, enhanced_sar
