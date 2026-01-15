"""
Feature-StabEnforce: Comprehensive Testing Suite
Tests for Stability Enforcement on State Dynamics

This test suite covers:
1. Unit tests for stability enforcement functions
2. Forward pass correctness with stability enforcement
3. Backward pass gradient correctness
4. Numerical stability and correctness
5. Integration with Mamba module
6. CUDA kernel tests
7. Performance benchmarks
8. Code validation checks
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from typing import Tuple, Optional

# Import Feature-StabEnforce modules
try:
    from mamba_ssm.modules.stability_enforcement import (
        compute_spectral_radius,
        apply_spectral_normalization,
        apply_eigenvalue_clamping,
        compute_stability_penalty,
        apply_stability_enforcement,
    )
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    pytest.skip("Feature-StabEnforce modules not available", allow_module_level=True)


class TestStabilityEnforcementFunctions:
    """Unit tests for stability enforcement utility functions"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self):
        return torch.float32
    
    def test_compute_spectral_radius_diagonal(self, device, dtype):
        """Test spectral radius computation for diagonal matrices"""
        d_inner = 4
        d_state = 8
        
        # Create diagonal matrix with known spectral radius
        A = torch.randn(d_inner, d_state, device=device, dtype=dtype)
        A[0, :] = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0, -2.0, -0.5, -0.1], device=device)
        
        spectral_radius = compute_spectral_radius(A)
        
        assert spectral_radius.shape == (d_inner,)
        assert torch.allclose(spectral_radius[0], torch.tensor(2.0, device=device), atol=1e-5)
        assert torch.all(spectral_radius >= 0), "Spectral radius should be non-negative"
    
    def test_compute_spectral_radius_full_matrix(self, device, dtype):
        """Test spectral radius computation for full matrices"""
        d_inner = 4
        d_state = 4
        
        # Create identity matrix (spectral radius = 1)
        A = torch.eye(d_state, device=device, dtype=dtype).unsqueeze(0).repeat(d_inner, 1, 1)
        
        spectral_radius = compute_spectral_radius(A)
        
        assert spectral_radius.shape == (d_inner,)
        assert torch.allclose(spectral_radius, torch.ones(d_inner, device=device), atol=1e-5)
    
    def test_apply_spectral_normalization_diagonal(self, device, dtype):
        """Test spectral normalization for diagonal matrices"""
        d_inner = 4
        d_state = 8
        
        # Create matrix with spectral radius > 1
        A = torch.randn(d_inner, d_state, device=device, dtype=dtype) * 2.0
        
        A_normalized = apply_spectral_normalization(A)
        
        # Check that spectral radius is <= 1
        spectral_radius = compute_spectral_radius(A_normalized)
        assert torch.all(spectral_radius <= 1.0 + 1e-5), "Spectral radius should be <= 1 after normalization"
        
        # Check that matrices with spectral radius <= 1 are unchanged
        A_small = torch.randn(d_inner, d_state, device=device, dtype=dtype) * 0.5
        A_small_normalized = apply_spectral_normalization(A_small)
        # Should be approximately unchanged (within numerical precision)
        assert torch.allclose(A_small, A_small_normalized, atol=1e-4)
    
    def test_apply_spectral_normalization_full_matrix(self, device, dtype):
        """Test spectral normalization for full matrices"""
        d_inner = 4
        d_state = 4
        
        # Create matrix with spectral radius > 1
        A = torch.randn(d_inner, d_state, d_state, device=device, dtype=dtype) * 2.0
        
        A_normalized = apply_spectral_normalization(A)
        
        # Check that spectral radius is <= 1
        spectral_radius = compute_spectral_radius(A_normalized)
        assert torch.all(spectral_radius <= 1.0 + 1e-5), "Spectral radius should be <= 1 after normalization"
    
    def test_apply_eigenvalue_clamping_diagonal(self, device, dtype):
        """Test eigenvalue clamping for diagonal matrices"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        # Create matrix with some positive values
        A = torch.randn(d_inner, d_state, device=device, dtype=dtype)
        A[0, 0] = 0.1  # Positive value
        
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        
        # Check that all values are <= -epsilon
        assert torch.all(A_clamped <= -epsilon + 1e-5), "All eigenvalues should be <= -epsilon"
        
        # Check that negative values far from threshold are unchanged
        A_negative = torch.randn(d_inner, d_state, device=device, dtype=dtype) * -1.0
        A_negative_clamped = apply_eigenvalue_clamping(A_negative, epsilon=epsilon)
        # Values that are already <= -epsilon should be unchanged
        mask = A_negative <= -epsilon
        assert torch.allclose(A_negative[mask], A_negative_clamped[mask], atol=1e-5)
    
    def test_apply_eigenvalue_clamping_full_matrix(self, device, dtype):
        """Test eigenvalue clamping for full matrices"""
        d_inner = 2
        d_state = 4
        epsilon = 0.01
        
        # Create a matrix (will be converted to full matrix internally)
        A = torch.randn(d_inner, d_state, d_state, device=device, dtype=dtype)
        
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        
        # Check that all real parts of eigenvalues are <= -epsilon
        eigenvalues = torch.linalg.eigvals(A_clamped)
        real_parts = eigenvalues.real
        assert torch.all(real_parts <= -epsilon + 1e-5), "All eigenvalue real parts should be <= -epsilon"
    
    def test_compute_stability_penalty_diagonal(self, device, dtype):
        """Test stability penalty computation for diagonal matrices"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        # Create matrix with some positive values
        A = torch.zeros(d_inner, d_state, device=device, dtype=dtype)
        A[0, 0] = 0.1  # Positive value > epsilon
        A[0, 1] = 0.005  # Positive value < epsilon (should not contribute)
        A[0, 2] = -0.1  # Negative value (should not contribute)
        
        penalty = compute_stability_penalty(A, epsilon=epsilon)
        
        # Should only penalize values > epsilon
        expected_penalty = (0.1 - epsilon)  # Only A[0, 0] contributes
        assert torch.allclose(penalty, torch.tensor(expected_penalty, device=device), atol=1e-5)
        assert penalty >= 0, "Penalty should be non-negative"
    
    def test_compute_stability_penalty_full_matrix(self, device, dtype):
        """Test stability penalty computation for full matrices"""
        d_inner = 2
        d_state = 4
        epsilon = 0.01
        
        # Create identity matrix (eigenvalues = 1, all > epsilon)
        A = torch.eye(d_state, device=device, dtype=dtype).unsqueeze(0).repeat(d_inner, 1, 1)
        
        penalty = compute_stability_penalty(A, epsilon=epsilon)
        
        # Each matrix has d_state eigenvalues of 1, so penalty = d_inner * d_state * (1 - epsilon)
        expected_penalty = d_inner * d_state * (1.0 - epsilon)
        assert torch.allclose(penalty, torch.tensor(expected_penalty, device=device), atol=1e-4)
    
    def test_apply_stability_enforcement_combined(self, device, dtype):
        """Test combined stability enforcement"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        # Create unstable matrix
        A = torch.randn(d_inner, d_state, device=device, dtype=dtype) * 2.0
        
        # Apply both stabilizers
        A_stabilized = apply_stability_enforcement(
            A,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            epsilon=epsilon,
        )
        
        # Check spectral radius
        spectral_radius = compute_spectral_radius(A_stabilized)
        assert torch.all(spectral_radius <= 1.0 + 1e-5), "Spectral radius should be <= 1"
        
        # Check eigenvalue clamping
        assert torch.all(A_stabilized <= -epsilon + 1e-5), "All eigenvalues should be <= -epsilon"
    
    def test_stability_enforcement_gradients(self, device, dtype):
        """Test that stability enforcement functions are differentiable"""
        d_inner = 4
        d_state = 8
        
        A = torch.randn(d_inner, d_state, device=device, dtype=dtype, requires_grad=True)
        
        # Test spectral normalization gradients
        A_normalized = apply_spectral_normalization(A)
        loss = A_normalized.sum()
        loss.backward()
        
        assert A.grad is not None, "Gradient should exist"
        assert not torch.isnan(A.grad).any(), "Gradient should not contain NaN"
        
        # Reset gradients
        A.grad = None
        
        # Test eigenvalue clamping gradients
        A_clamped = apply_eigenvalue_clamping(A, epsilon=0.01)
        loss = A_clamped.sum()
        loss.backward()
        
        assert A.grad is not None, "Gradient should exist"
        assert not torch.isnan(A.grad).any(), "Gradient should not contain NaN"


class TestMambaStabilityEnforcement:
    """Integration tests for Mamba module with stability enforcement"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def dtype(self):
        return torch.float32
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing"""
        return {
            "d_model": 64,
            "d_state": 16,
        }
    
    def test_forward_pass_with_spectral_normalization(self, device, dtype, basic_config):
        """Test forward pass with spectral normalization enabled"""
        batch_size = 2
        seqlen = 32
        
        mamba = Mamba(
            **basic_config,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=False,
            use_stability_penalty=False,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_forward_pass_with_eigenvalue_clamping(self, device, dtype, basic_config):
        """Test forward pass with eigenvalue clamping enabled"""
        batch_size = 2
        seqlen = 32
        
        mamba = Mamba(
            **basic_config,
            use_spectral_normalization=False,
            use_eigenvalue_clamping=True,
            use_stability_penalty=False,
            stability_epsilon=0.01,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_forward_pass_with_all_stabilizers(self, device, dtype, basic_config):
        """Test forward pass with all stabilizers enabled"""
        batch_size = 2
        seqlen = 32
        
        mamba = Mamba(
            **basic_config,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=True,
            stability_epsilon=0.01,
            stability_penalty_weight=0.1,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_backward_pass_gradients(self, device, dtype, basic_config):
        """Test backward pass gradient computation with stability enforcement"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(
            **basic_config,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=False,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
        
        output = mamba(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        
        # Check A_log gradients
        assert mamba.A_log.grad is not None, "A_log gradient is None"
        assert not torch.isnan(mamba.A_log.grad).any(), "A_log gradient contains NaN"
    
    def test_stability_penalty_loss(self, device, dtype, basic_config):
        """Test stability penalty loss computation"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(
            **basic_config,
            use_spectral_normalization=False,
            use_eigenvalue_clamping=False,
            use_stability_penalty=True,
            stability_epsilon=0.01,
            stability_penalty_weight=0.1,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        # Forward pass
        output = mamba(x)
        
        # Compute stability loss
        stability_loss = mamba.compute_stability_loss()
        
        assert stability_loss is not None, "Stability loss should be computed"
        assert stability_loss >= 0, "Stability loss should be non-negative"
        assert not torch.isnan(stability_loss), "Stability loss should not be NaN"
        
        # If stability penalty is disabled, loss should be 0
        mamba_no_penalty = Mamba(
            **basic_config,
            use_stability_penalty=False,
        ).to(device).to(dtype)
        
        stability_loss_disabled = mamba_no_penalty.compute_stability_loss()
        assert torch.allclose(stability_loss_disabled, torch.tensor(0.0, device=device)), \
            "Stability loss should be 0 when disabled"
    
    def test_stability_penalty_in_training(self, device, dtype, basic_config):
        """Test stability penalty loss in training loop"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(
            **basic_config,
            use_stability_penalty=True,
            stability_epsilon=0.01,
            stability_penalty_weight=0.1,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
        
        # Forward pass
        output = mamba(x)
        
        # Compute main loss and stability loss
        main_loss = output.sum()
        stability_loss = mamba.compute_stability_loss()
        
        total_loss = main_loss + stability_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input gradient should exist"
        assert mamba.A_log.grad is not None, "A_log gradient should exist"
        assert not torch.isnan(total_loss), "Total loss should not be NaN"
    
    def test_numerical_stability(self, device, dtype, basic_config):
        """Test numerical stability with extreme values"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(
            **basic_config,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=True,
            stability_epsilon=0.01,
        ).to(device).to(dtype)
        
        # Test with very small values
        x_small = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype) * 1e-6
        output_small = mamba(x_small)
        assert not torch.isnan(output_small).any(), "Small values cause NaN"
        
        # Test with large values
        x_large = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype) * 1e3
        output_large = mamba(x_large)
        assert not torch.isnan(output_large).any(), "Large values cause NaN"
        assert not torch.isinf(output_large).any(), "Large values cause Inf"
    
    def test_different_epsilon_values(self, device, dtype, basic_config):
        """Test with different epsilon values"""
        batch_size = 2
        seqlen = 16
        epsilon_values = [0.001, 0.01, 0.1]
        
        for epsilon in epsilon_values:
            mamba = Mamba(
                **basic_config,
                use_eigenvalue_clamping=True,
                use_stability_penalty=True,
                stability_epsilon=epsilon,
            ).to(device).to(dtype)
            
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            output = mamba(x)
            
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"Epsilon {epsilon} produces NaN"
    
    def test_bidirectional_with_stability_enforcement(self, device, dtype, basic_config):
        """Test bidirectional Mamba with stability enforcement"""
        batch_size = 2
        seqlen = 16
        
        config = basic_config.copy()
        config["bimamba_type"] = "v1"
        
        mamba = Mamba(
            **config,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=True,
            stability_epsilon=0.01,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Bidirectional output contains NaN"


class TestCUDAStabilityEnforcement:
    """CUDA-specific tests for stability enforcement"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    @pytest.fixture
    def dtype(self):
        return torch.float32
    
    def test_cuda_spectral_normalization(self, device, dtype):
        """Test spectral normalization on CUDA"""
        d_inner = 64
        d_state = 32
        
        A = torch.randn(d_inner, d_state, device=device, dtype=dtype) * 2.0
        
        A_normalized = apply_spectral_normalization(A)
        
        spectral_radius = compute_spectral_radius(A_normalized)
        assert torch.all(spectral_radius <= 1.0 + 1e-5), "Spectral radius should be <= 1"
    
    def test_cuda_eigenvalue_clamping(self, device, dtype):
        """Test eigenvalue clamping on CUDA"""
        d_inner = 64
        d_state = 32
        epsilon = 0.01
        
        A = torch.randn(d_inner, d_state, device=device, dtype=dtype)
        
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        
        assert torch.all(A_clamped <= -epsilon + 1e-5), "All eigenvalues should be <= -epsilon"
    
    def test_cuda_mamba_forward(self, device, dtype):
        """Test Mamba forward pass on CUDA with stability enforcement"""
        batch_size = 4
        seqlen = 128
        d_model = 128
        d_state = 32
        
        mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=True,
            stability_epsilon=0.01,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, d_model, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(3):
            _ = mamba(x)
        
        torch.cuda.synchronize()
        
        # Forward pass
        output = mamba(x)
        
        torch.cuda.synchronize()
        
        assert output.shape == (batch_size, seqlen, d_model)
        assert not torch.isnan(output).any(), "CUDA forward pass produces NaN"
        assert not torch.isinf(output).any(), "CUDA forward pass produces Inf"
    
    def test_cuda_gradient_computation(self, device, dtype):
        """Test gradient computation on CUDA"""
        batch_size = 2
        seqlen = 64
        d_model = 64
        d_state = 16
        
        mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=True,
            stability_epsilon=0.01,
            stability_penalty_weight=0.1,
        ).to(device).to(dtype)
        
        x = torch.randn(batch_size, seqlen, d_model, device=device, dtype=dtype, requires_grad=True)
        
        output = mamba(x)
        stability_loss = mamba.compute_stability_loss()
        total_loss = output.sum() + stability_loss
        
        total_loss.backward()
        
        assert x.grad is not None, "Input gradient should exist on CUDA"
        assert mamba.A_log.grad is not None, "A_log gradient should exist on CUDA"
        assert not torch.isnan(x.grad).any(), "CUDA gradient contains NaN"
        assert not torch.isnan(mamba.A_log.grad).any(), "CUDA A_log gradient contains NaN"


class TestCodeValidation:
    """Code validation and correctness checks"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_spectral_normalization_idempotency(self, device):
        """Test that spectral normalization is idempotent (applying twice is same as once)"""
        d_inner = 4
        d_state = 8
        
        A = torch.randn(d_inner, d_state, device=device) * 2.0
        
        A_normalized_once = apply_spectral_normalization(A)
        A_normalized_twice = apply_spectral_normalization(A_normalized_once)
        
        # Should be approximately the same
        assert torch.allclose(A_normalized_once, A_normalized_twice, atol=1e-5), \
            "Spectral normalization should be idempotent"
    
    def test_eigenvalue_clamping_preserves_shape(self, device):
        """Test that eigenvalue clamping preserves tensor shape"""
        d_inner = 4
        d_state = 8
        
        A = torch.randn(d_inner, d_state, device=device)
        A_clamped = apply_eigenvalue_clamping(A, epsilon=0.01)
        
        assert A_clamped.shape == A.shape, "Eigenvalue clamping should preserve shape"
    
    def test_stability_penalty_zero_for_stable_matrix(self, device):
        """Test that stability penalty is zero for a stable matrix"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        # Create a stable matrix (all values << -epsilon)
        A = torch.randn(d_inner, d_state, device=device) * -1.0 - 1.0
        
        penalty = compute_stability_penalty(A, epsilon=epsilon)
        
        assert torch.allclose(penalty, torch.tensor(0.0, device=device), atol=1e-5), \
            "Stability penalty should be zero for stable matrices"
    
    def test_stability_enforcement_parameter_validation(self, device):
        """Test parameter validation for stability enforcement"""
        d_inner = 4
        d_state = 8
        
        A = torch.randn(d_inner, d_state, device=device)
        
        # Test with invalid epsilon (should still work, but warn)
        A_clamped = apply_eigenvalue_clamping(A, epsilon=-0.01)  # Negative epsilon
        
        # Should still work (clamp to negative value)
        assert A_clamped.shape == A.shape
    
    def test_mamba_stability_enforcement_independence(self, device):
        """Test that stabilizers can be enabled independently"""
        batch_size = 2
        seqlen = 16
        d_model = 64
        d_state = 16
        
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        
        # Test each stabilizer independently
        configs = [
            {"use_spectral_normalization": True, "use_eigenvalue_clamping": False, "use_stability_penalty": False},
            {"use_spectral_normalization": False, "use_eigenvalue_clamping": True, "use_stability_penalty": False},
            {"use_spectral_normalization": False, "use_eigenvalue_clamping": False, "use_stability_penalty": True},
        ]
        
        for config in configs:
            mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                **config,
            ).to(device)
            
            output = mamba(x)
            assert output.shape == (batch_size, seqlen, d_model)
            assert not torch.isnan(output).any(), f"Config {config} produces NaN"


class TestPerformanceBenchmarks:
    """Performance benchmarks for stability enforcement"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_spectral_normalization_speed(self, device):
        """Benchmark spectral normalization speed"""
        import time
        
        d_inner = 128
        d_state = 64
        
        A = torch.randn(d_inner, d_state, device=device)
        
        # Warmup
        for _ in range(5):
            _ = apply_spectral_normalization(A)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            _ = apply_spectral_normalization(A)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        avg_time = elapsed / 100
        
        print(f"Average spectral normalization time: {avg_time*1000:.3f} ms")
        assert avg_time < 0.1, f"Spectral normalization too slow: {avg_time*1000:.3f} ms"
    
    def test_eigenvalue_clamping_speed(self, device):
        """Benchmark eigenvalue clamping speed"""
        import time
        
        d_inner = 128
        d_state = 64
        
        A = torch.randn(d_inner, d_state, device=device)
        
        # Warmup
        for _ in range(5):
            _ = apply_eigenvalue_clamping(A, epsilon=0.01)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            _ = apply_eigenvalue_clamping(A, epsilon=0.01)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        avg_time = elapsed / 100
        
        print(f"Average eigenvalue clamping time: {avg_time*1000:.3f} ms")
        assert avg_time < 0.1, f"Eigenvalue clamping too slow: {avg_time*1000:.3f} ms"
    
    def test_mamba_forward_speed_with_stability(self, device):
        """Benchmark Mamba forward pass speed with stability enforcement"""
        import time
        
        config = {
            "d_model": 128,
            "d_state": 32,
            "use_spectral_normalization": True,
            "use_eigenvalue_clamping": True,
            "use_stability_penalty": False,  # Disable penalty for speed test
        }
        
        mamba = Mamba(**config).to(device)
        x = torch.randn(4, 128, config["d_model"], device=device)
        
        # Warmup
        for _ in range(5):
            _ = mamba(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            _ = mamba(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        avg_time = elapsed / 10
        
        print(f"Average forward pass time with stability: {avg_time*1000:.2f} ms")
        # Should be reasonably fast (< 200ms per forward pass for this config)
        assert avg_time < 0.2, f"Forward pass too slow: {avg_time*1000:.2f} ms"
    
    def test_memory_usage(self, device):
        """Benchmark memory usage with stability enforcement"""
        config = {
            "d_model": 256,
            "d_state": 64,
            "use_spectral_normalization": True,
            "use_eigenvalue_clamping": True,
            "use_stability_penalty": True,
        }
        
        mamba = Mamba(**config).to(device)
        x = torch.randn(2, 64, config["d_model"], device=device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = mamba(x)
            stability_loss = mamba.compute_stability_loss()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"Peak memory usage: {peak_memory:.2f} MB")
            
            # Should use reasonable memory (< 2GB for this config)
            assert peak_memory < 2048, f"Memory usage too high: {peak_memory:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
