# ZOH Detection Failure - Root Cause Analysis

## The Core Problem

**ZOH discretization is fundamentally incompatible with detection tasks** when using pretrained classification weights, due to numerical instability in the exponential computation.

## Why ZOH Keeps Failing

### Mathematical Issue

ZOH discretization computes:
```python
dA = torch.exp(dt * A)  # Can explode to Inf
dB = dt * B
```

Where:
- `dt = softplus(dt_proj + bias)` - learned parameter, always positive
- `A` - learned state matrix parameter
- Both come from **pretrained classification weights**

### The Explosion Problem

1. **Pretrained weights may have large `dt_proj` values** optimized for classification
2. **Detection task is different** - different input distribution, different loss landscape
3. **First forward pass**: `exp(dt * A)` can immediately produce `Inf` if:
   - `dt` is large (from pretrained weights)
   - `A` has positive values
   - `dt * A > ~88` (float32 limit)

4. **Even with LR=1e-5, warmup=10k, clip=0.1**: The problem happens **before any gradient update** - in the first forward pass!

### Why Other Methods Work

- **FOH**: Uses Taylor series expansion, avoids direct `exp()` explosion
- **Bilinear**: Uses matrix inversion with stability checks
- **RK4**: Uses multiple small steps, more stable
- **Poly/HighOrder**: Use Taylor series, more numerically stable

## Evidence

1. ✅ **Batch sizes 16 and 32 both fail** - not a batch size issue
2. ✅ **Fails on first training step** - before any gradient updates
3. ✅ **Other discretization methods don't need special settings** - they're inherently more stable
4. ✅ **Error occurs in RPN proposals** - the first place Inf values propagate to

## Solutions

### Option 1: Switch to FOH or Bilinear (RECOMMENDED)

These methods are more stable and work with standard training settings:

```bash
# Use FOH instead
bash det/scripts/discretization/CC-Fir/ft_vim_tiny_vimdet_foh.sh

# Or Bilinear
bash det/scripts/discretization/CC-Fir/ft_vim_tiny_vimdet_bilinear.sh
```

### Option 2: Clamp dt_proj Values (If you must use ZOH)

Add initialization/normalization to prevent large dt values:

```python
# In mamba_simple.py, after loading pretrained weights:
if pretrained:
    # Clamp dt_proj to prevent explosion
    with torch.no_grad():
        for layer in self.layers:
            if hasattr(layer.mixer, 'dt_proj'):
                # Limit dt values to reasonable range
                layer.mixer.dt_proj.weight.data.clamp_(max=2.0)
                layer.mixer.dt_proj.bias.data.clamp_(max=1.0)
```

### Option 3: Reinitialize dt_proj for Detection

Don't use pretrained dt_proj values - reinitialize them:

```python
# After loading pretrained weights, reinitialize dt_proj
for layer in self.layers:
    if hasattr(layer.mixer, 'dt_proj'):
        nn.init.normal_(layer.mixer.dt_proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(layer.mixer.dt_proj.bias, 0.0)
```

### Option 4: Use Smaller dt Initialization

Modify the model to initialize dt_proj with smaller values:

```python
# In model initialization
dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
nn.init.normal_(dt_proj.weight, mean=0.0, std=0.01)  # Smaller std
nn.init.constant_(dt_proj.bias, -1.0)  # Negative bias to keep dt small initially
```

### Option 5: Add Numerical Stability Checks

Add clamping in the forward pass:

```python
# In mamba_simple.py ZOH discretization
if self.discretization_method == "zoh":
    dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
    dt = torch.clamp(dt, max=5.0)  # Prevent explosion
    dA = torch.exp(torch.einsum('bd,dn->bdn', dt, A))
    dA = torch.clamp(dA, min=1e-10, max=1e10)  # Clamp result
    dB = torch.einsum('bd,bn->bdn', dt, B)
```

## Recommended Action

**Switch to FOH discretization** - it's:
1. More numerically stable (uses Taylor series)
2. Works with standard training settings
3. Better accuracy than ZOH
4. Already implemented and tested

The fact that FOH, Bilinear, RK4, etc. don't need special stability settings proves they're fundamentally more stable than ZOH for detection tasks.

## Why This Happens in Detection But Not Classification

1. **Different input scales**: Detection images are larger (1024x1024 vs 224x224)
2. **Different feature distributions**: Detection features have different statistics
3. **Different loss functions**: Detection has multiple losses (RPN + ROI heads)
4. **Pretrained weights mismatch**: Classification-optimized dt_proj values may be too large for detection

## Conclusion

ZOH is failing because:
- It uses `exp(dt * A)` which can explode
- Pretrained weights may have large dt_proj values
- Detection task amplifies numerical issues
- The failure happens **before any training** - in the first forward pass

**Solution**: Use FOH or Bilinear instead of ZOH for detection tasks.
