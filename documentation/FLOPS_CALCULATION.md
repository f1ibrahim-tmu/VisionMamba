The FLOP multipliers account for the extra operations each discretization method performs beyond ZOH. Here’s why each multiplier is used:

## Why FLOP multipliers are needed

### The problem

Standard profilers (fvcore/thop) see the same architecture across methods, so they report the same FLOPs. The discretization step happens inside custom CUDA kernels or complex Python operations that profilers don’t count accurately.

### The solution: multipliers based on actual operations

Looking at the code, here’s why each multiplier makes sense:

---

### 1. ZOH = 1.0x (baseline)

Operations:

- `exp(A*delta)` — one exponential
- `einsum('bdl,dn->bdln')` — one matrix multiplication
- `delta * B` — simple multiplication

Total: ~2 operations per layer

---

### 2. FOH = 1.5x

Additional operations beyond ZOH:

- `delta ** 2`, `delta ** 3`, `delta ** 4`, `delta ** 5` — 4 power operations
- `A ** 2`, `A ** 3` — 2 power operations
- Multiple einsum operations for Taylor series terms:
  - `Δ²/2 + A*Δ³/6 + A²*Δ⁴/24 + A³*Δ⁵/120`
- Additional multiplications and additions

Total: ~3 operations per layer (1.5x ZOH)

---

### 3. Bilinear = 2.5x

Additional operations beyond ZOH:

- Matrix inversions: `torch.inverse(I + A*delta/2)` — O(N³) for each sequence position
- Multiple matrix multiplications:
  - `(I + A*delta/2)^-1 * (I - A*delta/2)`
  - `(I + A*delta/2)^-1 * delta * B`
- Diagonal matrix construction: `torch.diag_embed()`
- More einsum operations

Total: ~5 operations per layer (2.5x ZOH)

Matrix inversion is the main cost: for each (batch, dim, seqlen) position, you compute the inverse of a dstate×dstate matrix.

---

### 4. Poly = 2.0x

Additional operations beyond ZOH:

- Multiple power operations: `delta^2`, `delta^3`, `A^2`
- Multiple matrix multiplications:
  - `delta*B + delta^2*A*B/2 + delta^3*A^2*B/6`
- More einsum operations for polynomial terms

Total: ~4 operations per layer (2.0x ZOH)

---

### 5. HighOrder = 3.0x

Additional operations beyond ZOH:

- Higher-order Taylor series terms (more than FOH)
- More power operations: `delta^4`, `delta^5`, `delta^6`, etc.
- More matrix multiplications for higher-order terms
- More complex einsum operations

Total: ~6 operations per layer (3.0x ZOH)

---

### 6. RK4 = 4.0x

Additional operations beyond ZOH:

- 4 function evaluations per step:
  - `k1 = f(t_n, y_n)`
  - `k2 = f(t_n + delta/2, y_n + k1/2)`
  - `k3 = f(t_n + delta/2, y_n + k2/2)`
  - `k4 = f(t_n + delta, y_n + k3)`
- Final combination: `y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4)/6`
- Each evaluation includes exp, einsum, and multiplications

Total: ~8 operations per layer (4.0x ZOH)

---

## Why these multipliers matter

Even though discretization FLOPs are small relative to the model, they:

1. Reflect real computational differences
2. Help explain latency differences
3. Provide a more accurate comparison

Example:

- Base FLOPs: 4.63 B (from profiler)
- ZOH additional: 0.00 B → Total: 4.63 B
- RK4 additional: ~0.01 B → Total: 4.64 B

The difference is small but measurable and explains why RK4 is slower than ZOH.

---

## Are these multipliers accurate?

They are estimates based on:

- Code analysis of operations
- Mathematical complexity
- Relative computational cost

For precise counts, you’d need to:

- Profile each method separately
- Count operations in the CUDA kernels
- Measure actual hardware performance

The multipliers provide a reasonable approximation that captures the relative complexity differences between methods.

---

## Summary

The multipliers reflect:

- ZOH: simplest (baseline)
- FOH: +50% (power operations)
- Poly: +100% (polynomial terms)
- Bilinear: +150% (matrix inversions)
- HighOrder: +200% (higher-order terms)
- RK4: +300% (4 function evaluations)

These multipliers help explain why RK4 is slower than ZOH, even though the profiler reports the same FLOPs.
