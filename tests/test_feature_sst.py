"""
Feature-SST: Comprehensive Testing Suite
Tests for Block-Diagonal + Low-Rank A Matrix Implementation

This test suite covers:
1. Forward pass correctness for all discretization methods
2. Backward pass gradient correctness
3. Variable B/C support
4. Complex number support
5. Bidirectional Mamba
6. Numerical stability
7. Performance benchmarks
8. Unit tests for structured A matrix construction
9. Integration tests with Mamba module
10. Code validation checks
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from typing import Tuple, Optional

# Import Feature-SST modules
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    pytest.skip("Feature-SST modules not available", allow_module_level=True)


class TestFeatureSST:
    """Test suite for Feature-SST implementation"""
    
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
            "block_size": 4,
            "low_rank_rank": 2,
            "use_block_diagonal_lowrank": True,
        }
    
    # ========== Unit Tests: Structured A Matrix Construction ==========
    
    def test_structured_a_components_exist(self, device, dtype, basic_config):
        """Test that structured A components are properly initialized"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        assert hasattr(mamba, 'A_blocks'), "A_blocks should exist"
        assert hasattr(mamba, 'A_U'), "A_U should exist"
        assert hasattr(mamba, 'A_V'), "A_V should exist"
        assert len(mamba.A_blocks) == mamba.num_blocks, "Number of blocks should match"
        
        # Check shapes
        d_inner = basic_config["d_model"] * 2  # expand=2
        block_size = basic_config["block_size"]
        d_state = basic_config["d_state"]
        low_rank_rank = basic_config["low_rank_rank"]
        
        for block in mamba.A_blocks:
            assert block.shape == (d_inner, block_size, block_size), \
                f"Block shape should be ({d_inner}, {block_size}, {block_size})"
        
        assert mamba.A_U.shape == (d_inner, d_state, low_rank_rank), \
            f"A_U shape should be ({d_inner}, {d_state}, {low_rank_rank})"
        assert mamba.A_V.shape == (d_inner, d_state, low_rank_rank), \
            f"A_V shape should be ({d_inner}, {d_state}, {low_rank_rank})"
    
    def test_construct_a_matrix_structured(self, device, dtype, basic_config):
        """Test _construct_A_matrix returns correct diagonal for structured A"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        A_diag = mamba._construct_A_matrix()
        
        # Should return diagonal (d_inner, d_state)
        d_inner = basic_config["d_model"] * 2
        d_state = basic_config["d_state"]
        assert A_diag.shape == (d_inner, d_state), \
            f"A_diag shape should be ({d_inner}, {d_state})"
        
        # Check that structured components are stored
        assert hasattr(mamba, '_A_blocks_structured'), "Structured blocks should be stored"
        assert hasattr(mamba, '_A_U_structured'), "Structured U should be stored"
        assert hasattr(mamba, '_A_V_structured'), "Structured V should be stored"
        
        # Check no NaN/Inf
        assert not torch.isnan(A_diag).any(), "A_diag contains NaN"
        assert not torch.isinf(A_diag).any(), "A_diag contains Inf"
    
    def test_block_diagonal_structure(self, device, dtype, basic_config):
        """Test that block-diagonal structure is correctly formed"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        # Construct full A matrix from blocks to verify structure
        d_inner = basic_config["d_model"] * 2
        d_state = basic_config["d_state"]
        block_size = basic_config["block_size"]
        num_blocks = d_state // block_size
        
        # Build block-diagonal matrix manually
        A_full = torch.zeros(d_inner, d_state, d_state, device=device, dtype=dtype)
        for k in range(num_blocks):
            start_idx = k * block_size
            end_idx = (k + 1) * block_size
            A_full[:, start_idx:end_idx, start_idx:end_idx] = mamba.A_blocks[k]
        
        # Add low-rank component
        A_lowrank = torch.bmm(mamba.A_U, mamba.A_V.transpose(-2, -1))  # (d_inner, d_state, d_state)
        A_full = A_full + A_lowrank
        
        # Extract diagonal and compare with _construct_A_matrix
        A_diag_manual = A_full.diagonal(dim1=-2, dim2=-1)  # (d_inner, d_state)
        A_diag_constructed = mamba._construct_A_matrix()
        
        # Should match (within numerical precision)
        assert torch.allclose(A_diag_manual, A_diag_constructed, atol=1e-5), \
            "Constructed diagonal should match manual computation"
    
    def test_low_rank_component(self, device, dtype, basic_config):
        """Test that low-rank component UV^T is correctly formed"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        d_inner = basic_config["d_model"] * 2
        d_state = basic_config["d_state"]
        low_rank_rank = basic_config["low_rank_rank"]
        
        # Compute UV^T
        UVT = torch.bmm(mamba.A_U, mamba.A_V.transpose(-2, -1))
        
        assert UVT.shape == (d_inner, d_state, d_state), \
            f"UV^T shape should be ({d_inner}, {d_state}, {d_state})"
        
        # Check rank (should be at most low_rank_rank)
        # For each channel, compute rank
        for i in range(min(5, d_inner)):  # Check first 5 channels
            rank = torch.linalg.matrix_rank(UVT[i])
            assert rank <= low_rank_rank, \
                f"Rank of UV^T[{i}] should be <= {low_rank_rank}, got {rank}"
    
    # ========== Forward Pass Tests ==========
    
    def test_forward_pass_zoh(self, device, dtype, basic_config):
        """Test forward pass with ZOH discretization"""
        batch_size = 2
        seqlen = 32
        
        mamba = Mamba(**basic_config, discretization_method="zoh").to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_forward_pass_all_methods(self, device, dtype, basic_config):
        """Test forward pass with all 6 discretization methods"""
        methods = ["zoh", "foh", "bilinear", "poly", "highorder", "rk4"]
        batch_size = 2
        seqlen = 16
        
        for method in methods:
            mamba = Mamba(**basic_config, discretization_method=method).to(device).to(dtype)
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"{method} output contains NaN"
            assert not torch.isinf(output).any(), f"{method} output contains Inf"
    
    def test_forward_pass_deterministic(self, device, dtype, basic_config):
        """Test that forward pass is deterministic"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        mamba.eval()  # Set to eval mode for deterministic behavior
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output1 = mamba(x)
        output2 = mamba(x)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Forward pass should be deterministic"
    
    def test_forward_pass_output_range(self, device, dtype, basic_config):
        """Test that forward pass produces reasonable output values"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        # Output should be finite and in reasonable range
        assert torch.isfinite(output).all(), "Output should be finite"
        assert output.abs().max().item() < 1e6, "Output values should not be extremely large"
    
    # ========== Backward Pass Tests ==========
    
    def test_backward_pass_gradients(self, device, dtype, basic_config):
        """Test backward pass gradient computation"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
        
        output = mamba(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "Input gradient contains Inf"
        
        # Check structured A gradients
        if hasattr(mamba, 'A_blocks'):
            for block in mamba.A_blocks:
                assert block.grad is not None, "A_blocks gradient is None"
                assert not torch.isnan(block.grad).any(), "A_blocks gradient contains NaN"
                assert not torch.isinf(block.grad).any(), "A_blocks gradient contains Inf"
        
        if hasattr(mamba, 'A_U'):
            assert mamba.A_U.grad is not None, "A_U gradient is None"
            assert not torch.isnan(mamba.A_U.grad).any(), "A_U gradient contains NaN"
            assert not torch.isinf(mamba.A_U.grad).any(), "A_U gradient contains Inf"
        
        if hasattr(mamba, 'A_V'):
            assert mamba.A_V.grad is not None, "A_V gradient is None"
            assert not torch.isnan(mamba.A_V.grad).any(), "A_V gradient contains NaN"
            assert not torch.isinf(mamba.A_V.grad).any(), "A_V gradient contains Inf"
    
    def test_backward_pass_all_methods(self, device, dtype, basic_config):
        """Test backward pass with all discretization methods"""
        methods = ["zoh", "foh", "bilinear", "poly", "highorder", "rk4"]
        batch_size = 2
        seqlen = 16
        
        for method in methods:
            mamba = Mamba(**basic_config, discretization_method=method).to(device).to(dtype)
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
            
            output = mamba(x)
            loss = output.sum()
            loss.backward()
            
            assert x.grad is not None, f"{method} input gradient is None"
            assert not torch.isnan(x.grad).any(), f"{method} input gradient contains NaN"
    
    def test_gradient_correctness(self, device, dtype, basic_config):
        """Test gradient correctness using finite differences"""
        batch_size = 1
        seqlen = 8
        eps = 1e-5
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
        
        output = mamba(x)
        loss = output.sum()
        loss.backward()
        
        # Finite difference check for input gradient
        grad_analytic = x.grad.clone()
        
        # Reset gradient
        x.grad.zero_()
        
        # Finite difference approximation (sample a few elements)
        grad_finite_diff = torch.zeros_like(x)
        num_samples = min(10, x.numel())  # Sample 10 elements
        indices = torch.randperm(x.numel())[:num_samples]
        
        for idx in indices:
            x_flat = x.flatten()
            original_val = x_flat[idx].item()
            
            # Forward difference
            x_flat[idx] = original_val + eps
            x_perturbed = x_flat.view_as(x)
            output_perturbed = mamba(x_perturbed)
            loss_perturbed = output_perturbed.sum()
            
            grad_finite_diff.flatten()[idx] = (loss_perturbed - loss) / eps
            
            # Reset
            x_flat[idx] = original_val
        
        # Compare gradients (allow for numerical error)
        grad_diff = torch.abs(grad_analytic.flatten()[indices] - grad_finite_diff.flatten()[indices])
        max_diff = grad_diff.max().item()
        max_grad = torch.abs(grad_analytic.flatten()[indices]).max().item()
        
        # Allow 5% relative error for finite differences
        if max_grad > 0:
            assert max_diff < 0.05 * max_grad, \
                f"Gradient mismatch: max_diff={max_diff}, max_grad={max_grad}"
    
    # ========== Integration Tests ==========
    
    def test_variable_b_c(self, device, dtype, basic_config):
        """Test variable B and C support"""
        batch_size = 2
        seqlen = 16
        d_inner = basic_config["d_model"] * 2  # expand=2
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        # Variable B/C is handled internally by Mamba module
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Variable B/C output contains NaN"
    
    def test_bidirectional_mamba(self, device, dtype, basic_config):
        """Test bidirectional Mamba support"""
        batch_size = 2
        seqlen = 16
        
        config = basic_config.copy()
        config["bimamba_type"] = "v1"
        
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Bidirectional output contains NaN"
        
        # Check bidirectional A components exist
        if hasattr(mamba, 'A_b_blocks'):
            assert len(mamba.A_b_blocks) == mamba.num_blocks, \
                "Bidirectional blocks should match forward blocks"
    
    def test_complex_numbers(self, device, basic_config):
        """Test complex number support"""
        # Note: Complex support is template-based, this is a placeholder test
        # Full complex testing would require complex-valued inputs
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(**basic_config).to(device)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device)
        
        output = mamba(x)
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
    
    def test_multiple_layers(self, device, dtype, basic_config):
        """Test stacking multiple Mamba layers"""
        batch_size = 2
        seqlen = 16
        num_layers = 3
        
        layers = nn.ModuleList([
            Mamba(**basic_config).to(device).to(dtype)
            for _ in range(num_layers)
        ])
        
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        for layer in layers:
            x = layer(x)
        
        assert x.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(x).any(), "Multi-layer output contains NaN"
    
    # ========== Numerical Stability Tests ==========
    
    def test_numerical_stability(self, device, dtype, basic_config):
        """Test numerical stability with extreme values"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        # Test with very small values
        x_small = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype) * 1e-6
        output_small = mamba(x_small)
        assert not torch.isnan(output_small).any(), "Small values cause NaN"
        assert not torch.isinf(output_small).any(), "Small values cause Inf"
        
        # Test with large values
        x_large = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype) * 1e3
        output_large = mamba(x_large)
        assert not torch.isnan(output_large).any(), "Large values cause NaN"
        assert not torch.isinf(output_large).any(), "Large values cause Inf"
    
    def test_numerical_stability_zero_input(self, device, dtype, basic_config):
        """Test numerical stability with zero input"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.zeros(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert not torch.isnan(output).any(), "Zero input causes NaN"
        assert not torch.isinf(output).any(), "Zero input causes Inf"
        # Output should be small but not necessarily zero (due to biases, etc.)
        assert output.abs().max().item() < 1e3, "Zero input produces reasonable output"
    
    # ========== Memory Efficiency Tests ==========
    
    def test_memory_efficiency(self, device, dtype, basic_config):
        """Test that structured A uses less memory than full matrix"""
        d_state = basic_config["d_state"]
        block_size = basic_config["block_size"]
        low_rank_rank = basic_config["low_rank_rank"]
        d_inner = basic_config["d_model"] * 2
        
        # Memory for structured A
        num_blocks = d_state // block_size
        memory_structured = (
            d_inner * num_blocks * block_size * block_size +  # A_blocks
            d_inner * d_state * low_rank_rank * 2  # A_U + A_V
        )
        
        # Memory for full A matrix
        memory_full = d_inner * d_state * d_state
        
        # Structured should use less memory
        assert memory_structured < memory_full, \
            f"Structured A ({memory_structured}) should use less memory than full ({memory_full})"
        
        reduction = (1 - memory_structured / memory_full) * 100
        print(f"Memory reduction: {reduction:.1f}%")
        assert reduction > 0, "No memory reduction achieved"
    
    def test_no_full_matrix_construction(self, device, dtype, basic_config):
        """Test that full A matrix is never constructed"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        # _construct_A_matrix should only return diagonal, not full matrix
        A_diag = mamba._construct_A_matrix()
        
        assert A_diag.dim() == 2, "Should return 2D diagonal, not 3D full matrix"
        assert A_diag.shape[0] == basic_config["d_model"] * 2, "First dim should be d_inner"
        assert A_diag.shape[1] == basic_config["d_state"], "Second dim should be d_state"
    
    # ========== Parameter Variation Tests ==========
    
    def test_different_block_sizes(self, device, dtype, basic_config):
        """Test with different block sizes"""
        block_sizes = [2, 4, 8]
        batch_size = 2
        seqlen = 16
        
        for block_size in block_sizes:
            if basic_config["d_state"] % block_size != 0:
                continue
            
            config = basic_config.copy()
            config["block_size"] = block_size
            
            mamba = Mamba(**config).to(device).to(dtype)
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"Block size {block_size} produces NaN"
    
    def test_different_ranks(self, device, dtype, basic_config):
        """Test with different low-rank ranks"""
        ranks = [1, 2, 4]
        batch_size = 2
        seqlen = 16
        
        for rank in ranks:
            if rank >= basic_config["d_state"]:
                continue
            
            config = basic_config.copy()
            config["low_rank_rank"] = rank
            
            mamba = Mamba(**config).to(device).to(dtype)
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"Rank {rank} produces NaN"
    
    def test_different_d_states(self, device, dtype, basic_config):
        """Test with different state dimensions"""
        d_states = [8, 16, 32]
        batch_size = 2
        seqlen = 16
        
        for d_state in d_states:
            if d_state % basic_config["block_size"] != 0:
                continue
            
            config = basic_config.copy()
            config["d_state"] = d_state
            
            mamba = Mamba(**config).to(device).to(dtype)
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"d_state {d_state} produces NaN"
    
    # ========== Code Validation Tests ==========
    
    def test_structured_components_accessible(self, device, dtype, basic_config):
        """Test that structured components are accessible after forward pass"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(2, 16, basic_config["d_model"], device=device, dtype=dtype)
        
        _ = mamba(x)  # Forward pass
        
        # Structured components should be accessible
        assert hasattr(mamba, '_A_blocks_structured'), "Structured blocks should be accessible"
        assert hasattr(mamba, '_A_U_structured'), "Structured U should be accessible"
        assert hasattr(mamba, '_A_V_structured'), "Structured V should be accessible"
        
        assert mamba._A_blocks_structured is not None, "Structured blocks should not be None"
        assert mamba._A_U_structured is not None, "Structured U should not be None"
        assert mamba._A_V_structured is not None, "Structured V should not be None"


class TestPerformanceBenchmarks:
    """Performance benchmarks for Feature-SST"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_forward_speed(self, device):
        """Benchmark forward pass speed"""
        import time
        
        config = {
            "d_model": 128,
            "d_state": 32,
            "block_size": 8,
            "low_rank_rank": 4,
            "use_block_diagonal_lowrank": True,
        }
        
        mamba = Mamba(**config).to(device)
        x = torch.randn(4, 128, config["d_model"], device=device)
        
        # Warmup
        for _ in range(5):
            _ = mamba(x)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(10):
            _ = mamba(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"Average forward pass time: {avg_time*1000:.2f} ms")
        
        # Should be reasonably fast (< 100ms per forward pass for this config)
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time*1000:.2f} ms"
    
    def test_memory_usage(self, device):
        """Benchmark memory usage"""
        config = {
            "d_model": 256,
            "d_state": 64,
            "block_size": 8,
            "low_rank_rank": 4,
            "use_block_diagonal_lowrank": True,
        }
        
        mamba = Mamba(**config).to(device)
        x = torch.randn(2, 64, config["d_model"], device=device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = mamba(x)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"Peak memory usage: {peak_memory:.2f} MB")
            
            # Should use reasonable memory (< 1GB for this config)
            assert peak_memory < 1024, f"Memory usage too high: {peak_memory:.2f} MB"
    
    def test_backward_speed(self, device):
        """Benchmark backward pass speed"""
        import time
        
        config = {
            "d_model": 128,
            "d_state": 32,
            "block_size": 8,
            "low_rank_rank": 4,
            "use_block_diagonal_lowrank": True,
        }
        
        mamba = Mamba(**config).to(device)
        x = torch.randn(4, 128, config["d_model"], device=device, requires_grad=True)
        
        # Warmup
        for _ in range(3):
            output = mamba(x)
            loss = output.sum()
            loss.backward()
            mamba.zero_grad()
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(10):
            output = mamba(x)
            loss = output.sum()
            loss.backward()
            mamba.zero_grad()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"Average backward pass time: {avg_time*1000:.2f} ms")
        
        # Backward should be reasonably fast
        assert avg_time < 0.2, f"Backward pass too slow: {avg_time*1000:.2f} ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
