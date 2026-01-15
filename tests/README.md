# Stability Enforcement Test Suite

This directory contains comprehensive tests for the Feature-StabEnforce branch, which implements stability enforcement on state dynamics for Vision Mamba.

## Test Files

- **`test_feature_stab_enforce.py`**: Main test suite with unit tests, integration tests, and performance benchmarks
- **`test_stab_enforce_cuda.py`**: CUDA-specific tests for GPU validation
- **`test_stab_enforce_validation.py`**: Code validation, edge cases, and correctness checks

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/test_feature_stab_enforce.py -v

# Run all stability enforcement tests
pytest tests/test_stab_enforce*.py -v
```

### Run Specific Test Classes

```bash
# Unit tests for stability enforcement functions
pytest tests/test_feature_stab_enforce.py::TestStabilityEnforcementFunctions -v

# Integration tests with Mamba module
pytest tests/test_feature_stab_enforce.py::TestMambaStabilityEnforcement -v

# CUDA tests
pytest tests/test_stab_enforce_cuda.py -v -m cuda

# Validation tests
pytest tests/test_stab_enforce_validation.py -v
```

### Run Specific Tests

```bash
# Test spectral normalization
pytest tests/test_feature_stab_enforce.py::TestStabilityEnforcementFunctions::test_apply_spectral_normalization_diagonal -v

# Test CUDA forward pass
pytest tests/test_stab_enforce_cuda.py::TestCUDAMambaIntegration::test_large_batch_forward -v
```

### Run with Coverage

```bash
pytest tests/test_feature_stab_enforce.py --cov=mamba_ssm.modules.stability_enforcement --cov-report=html
```

## Test Coverage

### Unit Tests (`TestStabilityEnforcementFunctions`)
- Spectral radius computation (diagonal and full matrices)
- Spectral normalization
- Eigenvalue clamping
- Stability penalty computation
- Combined stability enforcement
- Gradient computation

### Integration Tests (`TestMambaStabilityEnforcement`)
- Forward pass with each stabilizer enabled
- Forward pass with all stabilizers enabled
- Backward pass gradient computation
- Stability penalty loss computation
- Stability penalty in training loop
- Numerical stability with extreme values
- Different epsilon values
- Bidirectional Mamba with stability enforcement

### CUDA Tests (`TestCUDASpectralRadius`, `TestCUDANormalization`, etc.)
- Large batch processing
- Full matrix operations
- Long sequence processing
- Gradient accumulation
- Throughput benchmarks
- Memory efficiency

### Validation Tests (`TestInputValidation`, `TestNumericalCorrectness`, etc.)
- Input validation and error handling
- Numerical correctness
- Function composition
- Edge cases and boundary conditions
- Parameter combinations

### Performance Benchmarks (`TestPerformanceBenchmarks`)
- Spectral normalization speed
- Eigenvalue clamping speed
- Mamba forward pass speed
- Memory usage

## Expected Test Results

All tests should pass with:
- ✅ Forward pass: All stabilizers produce valid outputs
- ✅ Backward pass: All gradients computed correctly
- ✅ Numerical stability: No NaN/Inf values
- ✅ CUDA compatibility: All CUDA tests pass on GPU
- ✅ Performance: Reasonable speed and memory usage

## Requirements

- PyTorch (with CUDA support for CUDA tests)
- pytest
- numpy

## Notes

- CUDA tests are skipped if CUDA is not available
- Some tests may have slightly different results due to numerical precision
- Performance benchmarks may vary based on hardware
