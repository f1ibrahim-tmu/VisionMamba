# Feature-SST Test Suite

Comprehensive testing suite for the Feature-SST (Structured State Transitions) implementation, which uses block-diagonal + low-rank A matrices in Vision Mamba.

## Overview

This test suite provides comprehensive coverage for the Feature-SST implementation, including:

- **Unit Tests**: Core functionality, structured A matrix construction, forward/backward passes
- **CUDA Tests**: GPU-specific functionality, large batch processing, memory efficiency
- **Validation Tests**: Input validation, edge cases, parameter combinations, error handling
- **Performance Benchmarks**: Speed and memory usage measurements

## Test Files

### Main Test Suite

- **`test_feature_sst.py`**: Main comprehensive test suite
  - Unit tests for structured A matrix construction
  - Forward pass tests for all discretization methods
  - Backward pass and gradient tests
  - Integration tests with Mamba module
  - Numerical stability tests
  - Memory efficiency tests
  - Performance benchmarks

### CUDA-Specific Tests

- **`test_sst_cuda.py`**: CUDA-specific tests
  - Large batch processing
  - Long sequence processing
  - GPU memory efficiency
  - CUDA kernel validation
  - Throughput benchmarks
  - Mixed precision (FP16) tests

### Validation Tests

- **`test_sst_validation.py`**: Validation and edge case tests
  - Input validation and error handling
  - Edge cases (single element, minimal config, etc.)
  - Boundary conditions
  - Parameter combinations
  - Numerical correctness
  - Error handling

### Supporting Files

- **`conftest.py`**: Pytest configuration and shared fixtures
- **`pytest.ini`**: Pytest configuration file
- **`run_tests.sh`**: Test runner script
- **`TEST_SUITE_SUMMARY.md`**: Detailed test suite summary

## Running Tests

### Using pytest directly

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_feature_sst.py -v

# Run specific test class
pytest tests/test_feature_sst.py::TestFeatureSST -v

# Run specific test
pytest tests/test_feature_sst.py::TestFeatureSST::test_forward_pass_zoh -v

# Run with markers
pytest tests/ -m "cuda" -v  # Only CUDA tests
pytest tests/ -m "not slow" -v  # Skip slow tests
```

### Using the test runner script

```bash
# Run all tests
./tests/run_tests.sh all

# Run main unit tests
./tests/run_tests.sh unit

# Run CUDA-specific tests
./tests/run_tests.sh cuda

# Run validation tests
./tests/run_tests.sh validation

# Run forward pass tests
./tests/run_tests.sh forward

# Run backward pass tests
./tests/run_tests.sh backward

# Run performance benchmarks
./tests/run_tests.sh performance
```

## Test Categories

### Unit Tests

Test core functionality of Feature-SST:

- Structured A matrix component initialization
- `_construct_A_matrix()` correctness
- Block-diagonal structure validation
- Low-rank component validation
- Forward pass with all discretization methods
- Backward pass gradient computation
- Gradient correctness (finite differences)

### Integration Tests

Test Feature-SST integration with Mamba module:

- Variable B/C support
- Bidirectional Mamba
- Multiple layer stacking
- Complex number support (placeholder)

### CUDA Tests

Test GPU-specific functionality:

- CUDA forward/backward passes
- Large batch processing (8, 16, 32 batches)
- Long sequence processing (256, 512, 1024 tokens)
- GPU memory efficiency
- CUDA kernel validation
- Throughput benchmarks
- Mixed precision (FP16)

### Validation Tests

Test input validation and edge cases:

- Invalid parameter handling
- Edge cases (single element, minimal config)
- Boundary conditions
- Parameter combinations
- Numerical correctness
- Error handling

### Performance Benchmarks

Measure performance characteristics:

- Forward pass speed
- Backward pass speed
- Memory usage
- Throughput (tokens/sec)

## Test Coverage

The test suite covers:

✅ **Structured A Matrix Construction**
- Component initialization and shapes
- Block-diagonal structure
- Low-rank component UV^T
- Diagonal computation

✅ **Forward Pass**
- All 6 discretization methods (ZOH, FOH, Bilinear, Poly, Highorder, RK4)
- Deterministic behavior
- Output range validation
- Numerical stability

✅ **Backward Pass**
- Gradient computation for all parameters
- Gradient correctness (finite differences)
- All discretization methods

✅ **Integration**
- Variable B/C support
- Bidirectional Mamba
- Multiple layers
- Complex numbers (placeholder)

✅ **Numerical Stability**
- Extreme values (very small/large)
- Zero input
- NaN/Inf detection

✅ **Memory Efficiency**
- Structured vs. full matrix comparison
- No full matrix construction verification

✅ **Parameter Variations**
- Different block sizes
- Different low-rank ranks
- Different state dimensions

✅ **CUDA Functionality**
- GPU forward/backward passes
- Large batch/sequence processing
- Memory efficiency
- Kernel validation
- Throughput benchmarks

✅ **Validation**
- Input validation
- Error handling
- Edge cases
- Boundary conditions
- Parameter combinations

## Requirements

- Python 3.7+
- PyTorch 1.9+
- pytest
- CUDA (optional, for CUDA tests)

## Installation

```bash
# Install pytest if not already installed
pip install pytest

# Install project dependencies
pip install -r requirements.txt
```

## Troubleshooting

### CUDA tests skipped

If CUDA tests are skipped, ensure:
- CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- CUDA device is properly configured

### Import errors

If you get import errors:
- Ensure you're in the project root directory
- Check that `mamba_ssm` module is installed
- Verify Python path includes project directory

### Memory errors

If tests fail with memory errors:
- Reduce batch sizes in test configurations
- Use CPU instead of CUDA for testing
- Close other applications using GPU memory

## Contributing

When adding new tests:

1. Follow existing test structure and naming conventions
2. Add appropriate docstrings
3. Use fixtures from `conftest.py` when possible
4. Add markers for test categories (cuda, slow, integration)
5. Update this README if adding new test categories

## See Also

- `TEST_SUITE_SUMMARY.md`: Detailed test suite summary
- Feature-SST documentation in `documentation/Mamba-3B/`
