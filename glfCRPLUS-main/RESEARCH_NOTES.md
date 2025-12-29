# Research Notes & Novelty Proposal

## Current Codebase Status
- **Baseline**: `net_CR_RDN.py` implements the GLF-CR model using Residual Dense Networks (RDN) with Window Attention.
- **Experimental**: `net_CR_CrossAttention.py` exists as a separate, dual-encoder architecture but isn't integrated into the main training pipeline.
- **Plan**: `IMPLEMENTATION_PLAN.md` outlines a 4-phase strategy to enhance the RDN model with transformers.

## Novelty Assessment
The current plan contributes novelty via "Hybrid CNN-Transformer" architecture for cloud removal. To guarantee paper acceptance, we need to address specific challenges in cloud removal: **Textural Detail Restoration** and **Global Consistency**.

## Proposed Enhancements for "Novelty" & "Results"

### 1. Frequency-Domain Loss (Fast Fourier Transform - FFT)
**Why?** Clouds act as a low-pass filter, removing high-frequency details (textures/edges). Standard L1/MSE loss functions struggle to prioritize these lost frequencies.
**Novelty**: Explicitly optimizing in the frequency domain forces the model to recover high-frequency components.
**Implementation**: Add `FFTLoss` to the loss function mix.
```python
loss_fft = || FFT(prediction) - FFT(target) ||
```

### 2. Adaptive "Cloud-Aware" Attention
**Why?** Attention should be focused on cloudy regions, not clear regions.
**Novelty**: Use the SAR data (which sees through clouds) to generate a "Cloud Confidence Map" that modulates the attention weights.
**Implementation**: Enhance the `SpeckleAwareGating` in `net_CR_CrossAttention.py` or the `DFG` in `net_CR_RDN.py`.

### 3. Mamba / State Space Models (High Novelty / High Risk)
**Why?** Transformers are $O(N^2)$. Mamba is $O(N)$ (linear complexity) and handles long sequences (global context) extremely well.
**Novelty**: "Mamba-based Cloud Removal" is significantly more novel than "Transformer-based".
**Risk**: Requires `mamba_ssm` package (CUDA only, tricky to install).
**Recommendation**: Keep as an optional "Advanced" branch if Kaggle environment supports it.

## Refined Implementation Strategy
I recommend strictly following the `IMPLEMENTATION_PLAN.md` but integrating **FFT Loss** immediately (Phase 1/2) as it's low-risk high-reward for metrics.

**Revised Roadmap:**
1.  **Refine Architecture**: Integrate `CrossModalAttention` into the main `RDN` backbone (Phase 1 of Plan).
2.  **Enhanced Loss**: Implement **FFT Loss** + **Perceptual Loss** (Phase 2 of Plan).
3.  **Training**: Run locally (mock) to verify shapes, then generate script for Kaggle.
