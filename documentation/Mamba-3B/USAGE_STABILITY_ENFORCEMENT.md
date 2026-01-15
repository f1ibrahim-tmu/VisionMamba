# Stability Enforcement Usage Guide

## Overview

The Feature-StabEnforce branch implements three independent stability enforcement mechanisms for Vision Mamba:

1. **Spectral Normalization**: `A ← A / max(1, ρ(A))`
2. **Eigenvalue Clamping**: Clamps eigenvalues to ensure stability
3. **Stability Penalty Loss**: `L_stab = Σ max(0, ℜ(λ_i(A)) - ε)`

Each stabilizer can be enabled/disabled independently for ablation experiments.

## Configuration

### Parameters

When creating a `Mamba` layer, you can configure stability enforcement with the following parameters:

```python
mamba_layer = Mamba(
    d_model=768,
    d_state=16,
    # ... other parameters ...
    
    # Feature-StabEnforce: Stability Enforcement
    use_spectral_normalization=False,  # Enable spectral normalization
    use_eigenvalue_clamping=False,     # Enable eigenvalue clamping
    use_stability_penalty=False,        # Enable stability penalty loss
    stability_epsilon=0.01,             # Threshold for stability (ε)
    stability_penalty_weight=0.1,      # Weight for penalty in loss (λ_stab)
)
```

### Default Values

- `use_spectral_normalization=False`: Disabled by default
- `use_eigenvalue_clamping=False`: Disabled by default
- `use_stability_penalty=False`: Disabled by default
- `stability_epsilon=0.01`: Stability margin threshold
- `stability_penalty_weight=0.1`: Weight for penalty in total loss

## Usage Examples

### Example 1: Enable Only Spectral Normalization

```python
from mamba_ssm.modules.mamba_simple import Mamba

mamba = Mamba(
    d_model=768,
    d_state=16,
    use_spectral_normalization=True,  # Only spectral normalization
    use_eigenvalue_clamping=False,
    use_stability_penalty=False,
)
```

### Example 2: Enable All Stabilizers

```python
mamba = Mamba(
    d_model=768,
    d_state=16,
    use_spectral_normalization=True,
    use_eigenvalue_clamping=True,
    use_stability_penalty=True,
    stability_epsilon=0.01,
    stability_penalty_weight=0.1,
)
```

### Example 3: Using Stability Penalty in Training

```python
import torch
import torch.nn as nn

# Create model with stability penalty enabled
model = YourVisionMambaModel(
    use_stability_penalty=True,
    stability_penalty_weight=0.1,
)

# Training loop
for batch in dataloader:
    # Forward pass
    output = model(batch['input'])
    
    # Compute task loss
    task_loss = criterion(output, batch['target'])
    
    # Compute stability loss from all Mamba layers
    stability_loss = torch.tensor(0.0, device=output.device)
    for module in model.modules():
        if isinstance(module, Mamba):
            stability_loss = stability_loss + module.compute_stability_loss()
    
    # Total loss
    total_loss = task_loss + stability_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

## Ablation Study Configuration

For systematic ablation experiments, you can test all combinations:

### Configuration Matrix

| SN | EC | SP | Description |
|---|---|---|---|
| ❌ | ❌ | ❌ | Baseline (no stabilizers) |
| ✅ | ❌ | ❌ | Spectral Normalization only |
| ❌ | ✅ | ❌ | Eigenvalue Clamping only |
| ❌ | ❌ | ✅ | Stability Penalty only |
| ✅ | ✅ | ❌ | SN + EC |
| ✅ | ❌ | ✅ | SN + SP |
| ❌ | ✅ | ✅ | EC + SP |
| ✅ | ✅ | ✅ | All stabilizers |

### Ablation Script Template

```python
configs = [
    {"use_spectral_normalization": False, "use_eigenvalue_clamping": False, "use_stability_penalty": False, "name": "baseline"},
    {"use_spectral_normalization": True, "use_eigenvalue_clamping": False, "use_stability_penalty": False, "name": "SN_only"},
    {"use_spectral_normalization": False, "use_eigenvalue_clamping": True, "use_stability_penalty": False, "name": "EC_only"},
    {"use_spectral_normalization": False, "use_eigenvalue_clamping": False, "use_stability_penalty": True, "name": "SP_only"},
    {"use_spectral_normalization": True, "use_eigenvalue_clamping": True, "use_stability_penalty": False, "name": "SN_EC"},
    {"use_spectral_normalization": True, "use_eigenvalue_clamping": False, "use_stability_penalty": True, "name": "SN_SP"},
    {"use_spectral_normalization": False, "use_eigenvalue_clamping": True, "use_stability_penalty": True, "name": "EC_SP"},
    {"use_spectral_normalization": True, "use_eigenvalue_clamping": True, "use_stability_penalty": True, "name": "all"},
]

for config in configs:
    print(f"Training with {config['name']}...")
    model = create_model(**{k: v for k, v in config.items() if k != 'name'})
    train(model, config['name'])
```

## Implementation Details

### Spectral Normalization

- Applied during `_construct_A_matrix()` before returning A
- Computes spectral radius `ρ(A) = max(|λ_i|)` for each channel
- Normalizes: `A_normalized = A / max(1, ρ(A))`
- Works for both diagonal and full A matrices

### Eigenvalue Clamping

- Applied during `_construct_A_matrix()` after spectral normalization
- For continuous-time stability: clamps `ℜ(λ_i) ≤ -ε`
- For full matrices: reconstructs matrix from clamped eigenvalues
- For diagonal matrices: clamps diagonal values directly

### Stability Penalty Loss

- Computed via `compute_stability_loss()` method
- Formula: `L_stab = λ_stab * Σ max(0, ℜ(λ_i(A)) - ε)`
- Should be added to training loss during backward pass
- Returns 0 if `use_stability_penalty=False`

## Performance Considerations

1. **Spectral Normalization**: 
   - Adds eigenvalue computation: O(d_state³) per channel for full matrices
   - O(d_state) for diagonal matrices
   - Applied once per forward pass

2. **Eigenvalue Clamping**:
   - Requires eigenvalue decomposition: O(d_state³) per channel
   - Matrix reconstruction: O(d_state³) per channel
   - Applied once per forward pass

3. **Stability Penalty**:
   - Requires eigenvalue computation: O(d_state³) per channel
   - Only computed when `use_stability_penalty=True`
   - Should be called explicitly in training loop

## Best Practices

1. **Start with Baseline**: Always compare against baseline (all stabilizers disabled)

2. **Individual Testing**: Test each stabilizer individually first

3. **Hyperparameter Tuning**:
   - `stability_epsilon`: Start with 0.01, try 0.001, 0.1
   - `stability_penalty_weight`: Start with 0.1, try 0.01, 1.0

4. **Monitoring**: Track stability loss separately from task loss during training

5. **Gradient Flow**: Check that gradients flow properly when stability penalty is enabled

## Troubleshooting

### Issue: Training becomes unstable with stabilizers enabled

**Solution**: 
- Reduce `stability_penalty_weight`
- Increase `stability_epsilon`
- Try enabling only one stabilizer at a time

### Issue: Model performance degrades with stabilizers

**Solution**:
- Stabilizers may be too restrictive
- Try adjusting `stability_epsilon` to allow more flexibility
- Consider using only spectral normalization (least restrictive)

### Issue: Slow training with eigenvalue clamping

**Solution**:
- Eigenvalue decomposition is expensive for large `d_state`
- Consider using only spectral normalization for faster training
- Or use stability penalty instead (computed less frequently)

## References

See `STABILITY_ENFORCEMENT_EXPLANATION.md` for detailed theoretical background.

