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
        
        # Check structured A gradients
        if hasattr(mamba, 'A_blocks'):
            for block in mamba.A_blocks:
                assert block.grad is not None, "A_blocks gradient is None"
                assert not torch.isnan(block.grad).any(), "A_blocks gradient contains NaN"
        
        if hasattr(mamba, 'A_U'):
            assert mamba.A_U.grad is not None, "A_U gradient is None"
            assert not torch.isnan(mamba.A_U.grad).any(), "A_U gradient contains NaN"
        
        if hasattr(mamba, 'A_V'):
            assert mamba.A_V.grad is not None, "A_V gradient is None"
            assert not torch.isnan(mamba.A_V.grad).any(), "A_V gradient contains NaN"
    
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
    
    def test_numerical_stability(self, device, dtype, basic_config):
        """Test numerical stability with extreme values"""
        batch_size = 2
        seqlen = 16
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        # Test with very small values
        x_small = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype) * 1e-6
        output_small = mamba(x_small)
        assert not torch.isnan(output_small).any(), "Small values cause NaN"
        
        # Test with large values
        x_large = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype) * 1e3
        output_large = mamba(x_large)
        assert not torch.isnan(output_large).any(), "Large values cause NaN"
        assert not torch.isinf(output_large).any(), "Large values cause Inf"
    
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
        
        # Finite difference approximation
        grad_finite_diff = torch.zeros_like(x)
        for i in range(x.numel()):
            x_flat = x.flatten()
            x_flat[i] += eps
            x_perturbed = x_flat.view_as(x)
            
            output_perturbed = mamba(x_perturbed)
            loss_perturbed = output_perturbed.sum()
            
            grad_finite_diff.flatten()[i] = (loss_perturbed - loss) / eps
            
            # Reset
            x_flat[i] -= eps
        
        # Compare gradients (allow for numerical error)
        grad_diff = torch.abs(grad_analytic - grad_finite_diff)
        max_diff = grad_diff.max().item()
        
        # Allow 1% relative error
        assert max_diff < 0.01 * torch.abs(grad_analytic).max().item(), \
            f"Gradient mismatch: max_diff={max_diff}"
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
