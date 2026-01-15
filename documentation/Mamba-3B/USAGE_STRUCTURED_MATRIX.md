# Feature-SST Usage Guide

## Overview

The Feature-SST (Structured State Transitions) branch implements block-diagonal + low-rank A matrices for Vision Mamba:

$$A = \text{blockdiag}(A_1, \ldots, A_K) + UV^T$$

This enables **cross-channel dynamics** while maintaining **computational efficiency** compared to full dense A matrices.

## Configuration

### Parameters

When creating a `Mamba` layer, you can configure Feature-SST with the following parameters:

```python
mamba_layer = Mamba(
    d_model=768,
    d_state=16,
    # ... other parameters ...
    
    # Feature-SST: Structured State Transitions - Block-Diagonal + Low-Rank A matrix
    use_block_diagonal_lowrank=True,  # Enable structured A matrix (default: True)
    block_size=4,                     # Size of each block (e.g., 4x4, 8x8)
    low_rank_rank=2,                  # Rank r for low-rank component UV^T
    discretization_method="zoh",      # Discretization method: zoh, foh, bilinear, rk4, poly, highorder
)
```

### Default Values

- `use_block_diagonal_lowrank=True`: **Enabled by default** (structured A matrix)
- `block_size=4`: Default block size (4×4 blocks)
- `low_rank_rank=2`: Default low-rank rank
- `discretization_method="zoh"`: Default discretization method

### Parameter Constraints

The following constraints must be satisfied:

- `d_state % block_size == 0`: State dimension must be divisible by block size
- `low_rank_rank < d_state`: Low-rank rank must be less than state dimension
- `low_rank_rank > 0`: Low-rank rank must be positive
- `block_size > 0`: Block size must be positive
- `block_size <= 16`: CUDA kernel limitation
- `low_rank_rank <= 16`: CUDA kernel limitation
- `d_state <= 256`: CUDA kernel limitation

**Example Valid Configurations**:
- `d_state=16, block_size=4, low_rank_rank=2` ✅
- `d_state=32, block_size=8, low_rank_rank=4` ✅
- `d_state=64, block_size=4, low_rank_rank=2` ✅
- `d_state=15, block_size=4, low_rank_rank=2` ❌ (15 % 4 != 0)

## Usage Examples

### Example 1: Enable Feature-SST (Default)

```python
from mamba_ssm.modules.mamba_simple import Mamba

# Feature-SST is enabled by default
mamba = Mamba(
    d_model=768,
    d_state=16,
    # use_block_diagonal_lowrank=True is the default
    block_size=4,
    low_rank_rank=2,
)

# Forward pass
output = mamba(input_tensor)  # Uses structured A matrix
```

### Example 2: Disable Feature-SST (Use Diagonal A)

```python
# To use traditional diagonal A matrix (like original Mamba)
mamba = Mamba(
    d_model=768,
    d_state=16,
    use_block_diagonal_lowrank=False,  # Disable structured A
)

# Forward pass
output = mamba(input_tensor)  # Uses diagonal A matrix
```

### Example 3: Custom Block Size and Rank

```python
# Larger blocks for coarser feature interactions
mamba = Mamba(
    d_model=768,
    d_state=64,
    use_block_diagonal_lowrank=True,
    block_size=8,        # 8x8 blocks (8 blocks total)
    low_rank_rank=4,    # Higher rank for more global interactions
)

# Forward pass
output = mamba(input_tensor)
```

### Example 4: Different Discretization Methods

```python
# Zero-Order Hold (default)
mamba_zoh = Mamba(
    d_model=768,
    d_state=16,
    discretization_method="zoh",
)

# First-Order Hold
mamba_foh = Mamba(
    d_model=768,
    d_state=16,
    discretization_method="foh",
)

# Bilinear (Tustin Transform)
mamba_bilinear = Mamba(
    d_model=768,
    d_state=16,
    discretization_method="bilinear",
)

# Runge-Kutta 4th Order
mamba_rk4 = Mamba(
    d_model=768,
    d_state=16,
    discretization_method="rk4",
)

# Polynomial Interpolation
mamba_poly = Mamba(
    d_model=768,
    d_state=16,
    discretization_method="poly",
)

# Higher-Order Hold
mamba_highorder = Mamba(
    d_model=768,
    d_state=16,
    discretization_method="highorder",
)
```

### Example 5: Complete Vision Mamba Model

```python
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class VisionMambaModel(nn.Module):
    def __init__(self, d_model=768, d_state=16):
        super().__init__()
        
        # Feature-SST enabled Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                use_block_diagonal_lowrank=True,  # Enable structured A
                block_size=4,
                low_rank_rank=2,
                discretization_method="zoh",
            )
            for _ in range(12)  # 12 layers
        ])
        
    def forward(self, x):
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        return x

# Create model
model = VisionMambaModel(d_model=768, d_state=16)

# Forward pass
input_tensor = torch.randn(2, 196, 768)  # (batch, seq_len, d_model)
output = model(input_tensor)
```

### Example 6: Training with Feature-SST

```python
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

# Create model with Feature-SST
model = YourVisionMambaModel(
    use_block_diagonal_lowrank=True,
    block_size=4,
    low_rank_rank=2,
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for batch in dataloader:
    # Forward pass
    output = model(batch['input'])
    
    # Compute loss
    loss = criterion(output, batch['target'])
    
    # Backward pass (gradients computed automatically)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Comparison: Structured A vs. Diagonal A

### Code Comparison

#### Using Structured A (Feature-SST)

```python
mamba_structured = Mamba(
    d_model=768,
    d_state=16,
    use_block_diagonal_lowrank=True,  # Enable structured A
    block_size=4,
    low_rank_rank=2,
)
```

**Characteristics**:
- ✅ Cross-channel interactions via blocks
- ✅ Global correlations via low-rank component
- ✅ Rich feature representations
- ✅ Efficient computation (O(K·d_k² + 2Nr) vs O(N²))

#### Using Diagonal A (Original Mamba)

```python
mamba_diagonal = Mamba(
    d_model=768,
    d_state=16,
    use_block_diagonal_lowrank=False,  # Use diagonal A
)
```

**Characteristics**:
- ✅ Very efficient (O(N) storage and computation)
- ❌ No cross-channel interactions
- ❌ Limited expressiveness
- ❌ Independent channel dynamics only

### Performance Comparison

| Configuration | Storage | Matrix Exp | Mat-Vec Mult | Cross-Channel |
|--------------|---------|------------|--------------|---------------|
| Diagonal A | O(N) | O(N) | O(N) | ❌ None |
| Structured A | O(K·d_k² + 2Nr) | O(K·d_k³ + r³) | O(K·d_k² + Nr) | ✅ Yes |
| Full Dense A | O(N²) | O(N³) | O(N²) | ✅ Yes |

**Example** (d_state=64):
- Diagonal A: 64 parameters, O(64) computation
- Structured A (K=8, d_k=8, r=4): 1024 parameters, O(512) computation
- Full Dense A: 4096 parameters, O(262,144) computation

## Ablation Study Configuration

For systematic ablation experiments, you can test different configurations:

### Configuration Matrix

| Config | Structured A | Block Size | Rank | Description |
|--------|--------------|------------|------|-------------|
| Baseline | ❌ | - | - | Diagonal A (original Mamba) |
| Small Blocks | ✅ | 2 | 2 | Fine-grained local interactions |
| Medium Blocks | ✅ | 4 | 2 | Balanced (default) |
| Large Blocks | ✅ | 8 | 4 | Coarser spatial patterns |
| High Rank | ✅ | 4 | 8 | More global interactions |
| Low Rank | ✅ | 4 | 1 | Minimal global interactions |

### Ablation Script Template

```python
configs = [
    {
        "name": "baseline",
        "use_block_diagonal_lowrank": False,
        "block_size": 0,
        "low_rank_rank": 0,
    },
    {
        "name": "small_blocks",
        "use_block_diagonal_lowrank": True,
        "block_size": 2,
        "low_rank_rank": 2,
    },
    {
        "name": "medium_blocks",
        "use_block_diagonal_lowrank": True,
        "block_size": 4,
        "low_rank_rank": 2,
    },
    {
        "name": "large_blocks",
        "use_block_diagonal_lowrank": True,
        "block_size": 8,
        "low_rank_rank": 4,
    },
    {
        "name": "high_rank",
        "use_block_diagonal_lowrank": True,
        "block_size": 4,
        "low_rank_rank": 8,
    },
]

for config in configs:
    print(f"Training with {config['name']}...")
    model = create_model(**{k: v for k, v in config.items() if k != 'name'})
    train(model, config['name'])
```

## Implementation Details

### Automatic CUDA Kernel Selection

Feature-SST automatically uses optimized CUDA kernels when available:

- **CUDA kernels**: Used when `use_block_diagonal_lowrank=True` and CUDA is available
- **No Python fallback**: All structured A operations are implemented in CUDA
- **Direct structured operations**: CUDA kernels operate directly on `A_blocks`, `A_U`, `A_V` without constructing full matrices

### Memory Layout

The structured A matrix is stored as:

- **A_blocks**: `(d_inner, num_blocks, block_size, block_size)` - Block-diagonal components
- **A_U**: `(d_inner, d_state, low_rank_rank)` - Low-rank U factor
- **A_V**: `(d_inner, d_state, low_rank_rank)` - Low-rank V factor

**No full matrix construction**: The full A matrix is never explicitly constructed, ensuring maximum efficiency.

### Discretization Methods

All 6 discretization methods are supported:

1. **ZOH** (Zero-Order Hold): `discretization_method="zoh"` - Default, most efficient
2. **FOH** (First-Order Hold): `discretization_method="foh"` - Higher accuracy
3. **Bilinear** (Tustin Transform): `discretization_method="bilinear"` - Stability-preserving
4. **RK4** (Runge-Kutta 4th Order): `discretization_method="rk4"` - Highest accuracy
5. **Poly** (Polynomial Interpolation): `discretization_method="poly"` - Smooth transitions
6. **Highorder** (Higher-Order Hold): `discretization_method="highorder"` - Very high accuracy

All methods are implemented in CUDA for maximum efficiency.

## Performance Considerations

### Memory Usage

**Structured A** (d_state=64, block_size=4, rank=2):
- Block-diagonal: 8 × 16 = 128 parameters per channel
- Low-rank: 2 × 64 × 2 = 256 parameters per channel
- **Total: 384 parameters per channel**

**Diagonal A** (d_state=64):
- **Total: 64 parameters per channel**

**Full Dense A** (d_state=64):
- **Total: 4096 parameters per channel**

### Computational Cost

**Matrix Exponential** (per time step):
- Diagonal A: O(N) = O(64)
- Structured A: O(K·d_k³ + r³) = O(8·64 + 8) ≈ O(520)
- Full Dense A: O(N³) = O(262,144)

**Matrix-Vector Product** (per time step):
- Diagonal A: O(N) = O(64)
- Structured A: O(K·d_k² + Nr) = O(128 + 128) = O(256)
- Full Dense A: O(N²) = O(4,096)

### When to Use Structured A

**Use Structured A when**:
- ✅ You need cross-channel feature interactions
- ✅ Vision tasks requiring rich feature representations
- ✅ Tasks benefiting from global correlations
- ✅ You can afford slightly more memory/computation

**Use Diagonal A when**:
- ✅ Maximum efficiency is critical
- ✅ Cross-channel interactions are not needed
- ✅ Text/sequence modeling tasks
- ✅ Very limited memory/computation budget

## Best Practices

### 1. Start with Default Configuration

```python
# Start with default settings
mamba = Mamba(
    d_model=768,
    d_state=16,
    # use_block_diagonal_lowrank=True is default
    # block_size=4 is default
    # low_rank_rank=2 is default
)
```

### 2. Tune Block Size Based on Task

- **Small blocks (2×2, 4×4)**: Fine-grained local features (edges, textures)
- **Medium blocks (4×4, 8×8)**: Balanced local and global (default)
- **Large blocks (8×8, 16×16)**: Coarser spatial patterns (objects, regions)

### 3. Tune Low-Rank Rank Based on Global Interactions

- **Low rank (1-2)**: Minimal global interactions, more efficient
- **Medium rank (2-4)**: Balanced global interactions (default)
- **High rank (4-8)**: Rich global interactions, more expressive

### 4. Choose Appropriate Discretization Method

- **ZOH**: Default, most efficient, sufficient for most tasks
- **FOH/Bilinear**: Higher accuracy, slightly more computation
- **RK4**: Highest accuracy, most computation

### 5. Monitor Memory Usage

```python
import torch

# Check memory usage
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    output = model(input_tensor)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    print(f"Peak memory: {peak_memory:.2f} MB")
```

## Troubleshooting

### Issue: `d_state must be divisible by block_size`

**Error**: `AssertionError: d_state (15) must be divisible by block_size (4)`

**Solution**: 
- Choose `block_size` that divides `d_state` evenly
- Common choices: `block_size=1, 2, 4, 8, 16`
- Example: For `d_state=15`, use `block_size=1, 3, 5, or 15`

### Issue: `low_rank_rank must be < d_state`

**Error**: `AssertionError: low_rank_rank (20) must be < d_state (16)`

**Solution**:
- Set `low_rank_rank < d_state`
- Typical values: `low_rank_rank = 1, 2, 4, 8`
- Example: For `d_state=16`, use `low_rank_rank <= 15`

### Issue: CUDA kernel limitations

**Error**: `block_size (32) must be <= 16 (CUDA kernel limitation)`

**Solution**:
- Use `block_size <= 16`
- Use `low_rank_rank <= 16`
- Use `d_state <= 256`

### Issue: Slow training with structured A

**Solution**:
- Structured A is more efficient than full dense A, but slower than diagonal A
- This is expected - you're trading efficiency for expressiveness
- Consider using smaller `block_size` or `low_rank_rank` if speed is critical

### Issue: Out of memory

**Solution**:
- Reduce `d_state` if possible
- Reduce `block_size` (more blocks, smaller blocks)
- Reduce `low_rank_rank`
- Use gradient checkpointing
- Reduce batch size

## References

See `IMPLEMENTATION_SUMMARY_STRUCTURED_MATRIX.md` for detailed implementation information.

See `EXPLANATION_STRUCTURED_MATRIX.md` for theoretical background and mathematical details.
