# Architecture Fix Summary

## Problem Identified

The model had a **channel mismatch** between what the dataloader provides and what the model expected:

### Data Flow Mismatch (Before Fix)
```
Dataloader Output:
├── cloudy_data: (B, 13, H, W)      ← Sentinel-2 optical bands
├── SAR_data: (B, 2, H, W)           ← VV and VH polarizations
└── cloudfree_data: (B, 13, H, W)   ← Ground truth

Model Input (WRONG):
├── optical_img: Expected (B, 3, H, W)  ← ❌ Only 3 channels
└── sar_img: Expected (B, 1, H, W)      ← ❌ Only 1 channel

Model Output (WRONG):
└── output: (B, 3, H, W)                ← ❌ Only 3 channels, should be 13
```

## Solutions Implemented

### 1. Fixed OpticalEncoder Input (net_CR_CrossAttention.py)
```python
# Before:
self.E1 = ConvBlock(3, 64, ...)      # ❌ Expected 3 channels

# After:
self.E1 = ConvBlock(13, 64, ...)     # ✓ Now accepts 13 Sentinel-2 bands
```

### 2. Fixed SAREncoder Input (net_CR_CrossAttention.py)
```python
# Before:
self.E1 = ConvBlock(1, 64, ...)      # ❌ Expected 1 channel

# After:
self.E1 = ConvBlock(2, 64, ...)      # ✓ Now accepts VV and VH (2 channels)
```

### 3. Fixed Decoder Output (net_CR_CrossAttention.py)
```python
# Before:
self.final_conv = nn.Conv2d(64, 3, kernel_size=1)  # ❌ Outputs 3 channels

# After:
self.final_conv = nn.Conv2d(64, 13, kernel_size=1)  # ✓ Outputs 13 channels
```

### 4. Updated Model Wrapper (train_CR.py, train_CR_kaggle.py)
Added detailed debugging to show actual data shapes:
```python
if self.first_batch:
    print(f"DEBUG: Available data keys: {list(data.keys())}")
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            print(f"  {key}: shape {data[key].shape}")  # ✓ Shows actual dimensions
```

## Complete Data Flow (After Fix)

```
┌─────────────────────────────────────────────┐
│   Dataloader (dataloader.py)                │
├─────────────────────────────────────────────┤
│ cloudy_data:    (B, 13, H, W)  [Sentinel-2]│
│ SAR_data:       (B, 2, H, W)   [VV, VH]    │
│ cloudfree_data: (B, 13, H, W)  [Ground truth]
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  CloudRemovalCrossAttention Model           │
├─────────────────────────────────────────────┤
│ OpticalEncoder(13→64→128→256)               │
│ SAREncoder(2→64→128→256)                    │
│ GlobalCrossAttention(feat_opt_3, feat_sar_3)│
│ SpeckleAwareGatingModule()                  │
│ Refinement()                                │
│ Decoder(256→128→64→13 output)               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│   Model Output                              │
├─────────────────────────────────────────────┤
│ output: (B, 13, H, W) [Cloud-free image]   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│   Loss Calculation                          │
├─────────────────────────────────────────────┤
│ L1Loss(output, cloudfree_data)              │
│ Both are (B, 13, H, W) ✓ Matching          │
└─────────────────────────────────────────────┘
```

## Architecture According to Your Design

The model now correctly implements your cross-attention architecture:

### 1. **Dual Encoders**
- **OpticalEncoder**: 13-channel input (Sentinel-2 multispectral)
  - E1: 13 → 64 channels (H, W)
  - E2: 64 → 128 channels (H/2, W/2)
  - E3: 128 → 256 channels (H/4, W/4)

- **SAREncoder**: 2-channel input (VV, VH polarizations)
  - E1: 2 → 64 channels (H, W)
  - E2: 64 → 128 channels (H/2, W/2)
  - E3: 128 → 256 channels (H/4, W/4)

### 2. **Global Cross-Attention (at E3 scale)**
```python
Q = optical_features_E3  # (B, 256, H/4, W/4)
K = SAR_features_E3      # (B, 256, H/4, W/4)
V = SAR_features_E3      # (B, 256, H/4, W/4)

Attention Output: (B, 256, H/4, W/4)
```
Cross-attention allows SAR features to guide optical feature refinement through attention weights.

### 3. **Speckle-Aware Gating Module**
```python
gate = σ(GateNet(SAR_E3))
output = cross_attention_output ⊙ gate
```
SAR-based gating σ(·) creates attention weights to suppress speckle noise.

### 4. **Refinement Block**
```python
refined = Conv(Concat(gated_output, optical_E3))
```
Fuses gated features with original optical features for preservation.

### 5. **Symmetric Decoder**
```
(B, 256, H/4, W/4) ──upx2──→ (B, 128, H/2, W/2) + skip
                   ──upx2──→ (B, 64, H, W) + skip
                   ──conv──→ (B, 13, H, W) [Output]
```

## Files Modified

1. **net_CR_CrossAttention.py**
   - OpticalEncoder: 3 channels → 13 channels
   - SAREncoder: 1 channel → 2 channels
   - Decoder: 3 output channels → 13 output channels
   - Updated docstrings to reflect actual dimensions

2. **train_CR.py**
   - Enhanced debugging in ModelCRNetCrossAttention.set_input()
   - Shows actual tensor shapes on first batch

3. **train_CR_kaggle.py**
   - Enhanced debugging in ModelCRNetCrossAttention.set_input()
   - Shows actual tensor shapes on first batch

## How to Run Training in Kaggle

```bash
# Clone latest code with all fixes
!rm -rf glfCR-modifed
!git clone https://github.com/ESWARALLU/glfCR-modifed.git

# Run training
!python /kaggle/working/glfCR-modifed/glfCRPLUS-main/codes/train_CR_kaggle.py \
    --model_name CrossAttention \
    --batch_sz 8 \
    --max_epochs 10
```

Expected output on first batch:
```
DEBUG: Available data keys: ['cloudy_data', 'SAR_data', 'cloudfree_data', 'file_name']
  cloudy_data: shape torch.Size([B, 13, 128, 128])
  SAR_data: shape torch.Size([B, 2, 128, 128])
  cloudfree_data: shape torch.Size([B, 13, 128, 128])
```

## Verification Checklist

- ✓ OpticalEncoder accepts 13 channels
- ✓ SAREncoder accepts 2 channels
- ✓ Model output is 13 channels
- ✓ Loss calculation uses matching dimensions
- ✓ Wrapper handles correct data keys
- ✓ Debugging output shows tensor shapes
- ✓ All commits pushed to GitHub

## Next Steps

1. Run training in Kaggle with `--model_name CrossAttention`
2. Monitor metrics (PSNR, SSIM) for improvement
3. Compare with RDN baseline if needed
4. Adjust hyperparameters based on validation results
