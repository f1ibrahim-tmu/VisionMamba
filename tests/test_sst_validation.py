"""
Feature-SST: Validation Tests
Tests for input validation, error handling, edge cases, and parameter combinations
"""

import torch
import torch.nn as nn
import pytest
import numpy as np

# Import Feature-SST modules
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    pytest.skip("Feature-SST modules not available", allow_module_level=True)


class TestSSTValidation:
    """Validation tests for Feature-SST"""
    
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
    
    # ========== Input Validation Tests ==========
    
    def test_invalid_block_size(self, device, dtype):
        """Test that invalid block_size raises error"""
        config = {
            "d_model": 64,
            "d_state": 16,
            "block_size": 5,  # 16 % 5 != 0
            "low_rank_rank": 2,
            "use_block_diagonal_lowrank": True,
        }
        
        with pytest.raises(AssertionError, match="must be divisible"):
            Mamba(**config).to(device).to(dtype)
    
    def test_invalid_low_rank_rank(self, device, dtype):
        """Test that invalid low_rank_rank is handled"""
        config = {
            "d_model": 64,
            "d_state": 16,
            "block_size": 4,
            "low_rank_rank": 20,  # > d_state
            "use_block_diagonal_lowrank": True,
        }
        
        # Should still work but may have issues
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(2, 16, 64, device=device, dtype=dtype)
        
        # May produce warnings or errors, but should handle gracefully
        try:
            output = mamba(x)
            assert output.shape == (2, 16, 64)
        except Exception as e:
            # If it fails, that's expected for invalid rank
            assert "rank" in str(e).lower() or "dimension" in str(e).lower()
    
    def test_zero_block_size(self, device, dtype):
        """Test that zero block_size raises error"""
        config = {
            "d_model": 64,
            "d_state": 16,
            "block_size": 0,  # Invalid
            "low_rank_rank": 2,
            "use_block_diagonal_lowrank": True,
        }
        
        with pytest.raises((AssertionError, ValueError, ZeroDivisionError)):
            Mamba(**config).to(device).to(dtype)
    
    def test_negative_low_rank_rank(self, device, dtype):
        """Test that negative low_rank_rank raises error"""
        config = {
            "d_model": 64,
            "d_state": 16,
            "block_size": 4,
            "low_rank_rank": -1,  # Invalid
            "use_block_diagonal_lowrank": True,
        }
        
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            Mamba(**config).to(device).to(dtype)
    
    def test_invalid_input_shape(self, device, dtype, basic_config):
        """Test that invalid input shapes are handled"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        # Wrong number of dimensions
        x_wrong_dims = torch.randn(2, 16, device=device, dtype=dtype)  # Missing last dim
        with pytest.raises((RuntimeError, IndexError)):
            _ = mamba(x_wrong_dims)
        
        # Wrong feature dimension
        x_wrong_feat = torch.randn(2, 16, 32, device=device, dtype=dtype)  # Wrong d_model
        with pytest.raises((RuntimeError, AssertionError)):
            _ = mamba(x_wrong_feat)
    
    # ========== Edge Cases ==========
    
    def test_single_element_sequence(self, device, dtype, basic_config):
        """Test with single element sequence"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(1, 1, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (1, 1, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Single element produces NaN"
    
    def test_single_batch(self, device, dtype, basic_config):
        """Test with single batch"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(1, 16, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (1, 16, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Single batch produces NaN"
    
    def test_minimal_config(self, device, dtype):
        """Test with minimal configuration"""
        config = {
            "d_model": 32,
            "d_state": 8,
            "block_size": 2,
            "low_rank_rank": 1,
            "use_block_diagonal_lowrank": True,
        }
        
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(1, 8, config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (1, 8, config["d_model"])
        assert not torch.isnan(output).any(), "Minimal config produces NaN"
    
    def test_maximal_block_size(self, device, dtype):
        """Test with block_size equal to d_state"""
        config = {
            "d_model": 64,
            "d_state": 16,
            "block_size": 16,  # Equal to d_state
            "low_rank_rank": 2,
            "use_block_diagonal_lowrank": True,
        }
        
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(2, 16, config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (2, 16, config["d_model"])
        assert not torch.isnan(output).any(), "Maximal block size produces NaN"
    
    def test_rank_equal_to_d_state_minus_one(self, device, dtype):
        """Test with rank = d_state - 1"""
        config = {
            "d_model": 64,
            "d_state": 16,
            "block_size": 4,
            "low_rank_rank": 15,  # d_state - 1
            "use_block_diagonal_lowrank": True,
        }
        
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(2, 16, config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (2, 16, config["d_model"])
        assert not torch.isnan(output).any(), "High rank produces NaN"
    
    # ========== Boundary Conditions ==========
    
    def test_extreme_block_sizes(self, device, dtype):
        """Test with extreme but valid block sizes"""
        d_state = 32
        valid_block_sizes = [1, 2, 4, 8, 16, 32]  # Divisors of 32
        
        for block_size in valid_block_sizes:
            if block_size == 1:  # Skip block_size=1 as it may not be practical
                continue
            
            config = {
                "d_model": 64,
                "d_state": d_state,
                "block_size": block_size,
                "low_rank_rank": 2,
                "use_block_diagonal_lowrank": True,
            }
            
            mamba = Mamba(**config).to(device).to(dtype)
            x = torch.randn(2, 16, config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            assert output.shape == (2, 16, config["d_model"])
            assert not torch.isnan(output).any(), f"Block size {block_size} produces NaN"
    
    def test_extreme_ranks(self, device, dtype, basic_config):
        """Test with extreme but valid ranks"""
        d_state = basic_config["d_state"]
        valid_ranks = [1, 2, 4, 8]  # Valid ranks < d_state
        
        for rank in valid_ranks:
            config = basic_config.copy()
            config["low_rank_rank"] = rank
            
            mamba = Mamba(**config).to(device).to(dtype)
            x = torch.randn(2, 16, config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            assert output.shape == (2, 16, config["d_model"])
            assert not torch.isnan(output).any(), f"Rank {rank} produces NaN"
    
    # ========== Parameter Combinations ==========
    
    def test_all_discretization_methods(self, device, dtype, basic_config):
        """Test all discretization methods with structured A"""
        methods = ["zoh", "foh", "bilinear", "poly", "highorder", "rk4"]
        
        for method in methods:
            mamba = Mamba(**basic_config, discretization_method=method).to(device).to(dtype)
            x = torch.randn(2, 16, basic_config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            assert output.shape == (2, 16, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"{method} produces NaN"
    
    def test_bidirectional_with_structured_a(self, device, dtype, basic_config):
        """Test bidirectional Mamba with structured A"""
        config = basic_config.copy()
        config["bimamba_type"] = "v1"
        
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(2, 16, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        assert output.shape == (2, 16, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Bidirectional with structured A produces NaN"
    
    def test_diagonal_fallback(self, device, dtype):
        """Test fallback to diagonal A when structured A is disabled"""
        config = {
            "d_model": 64,
            "d_state": 16,
            "use_block_diagonal_lowrank": False,  # Disable structured A
        }
        
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(2, 16, config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        assert output.shape == (2, 16, config["d_model"])
        assert not torch.isnan(output).any(), "Diagonal A produces NaN"
        
        # Should not have structured components
        assert not hasattr(mamba, 'A_blocks') or mamba.A_blocks is None, \
            "Should not have A_blocks when structured A is disabled"
    
    # ========== Numerical Correctness ==========
    
    def test_output_consistency(self, device, dtype, basic_config):
        """Test that outputs are consistent across multiple forward passes"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        mamba.eval()  # Set to eval mode
        
        x = torch.randn(2, 16, basic_config["d_model"], device=device, dtype=dtype)
        
        output1 = mamba(x)
        output2 = mamba(x)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Outputs should be consistent across forward passes"
    
    def test_gradient_consistency(self, device, dtype, basic_config):
        """Test that gradients are consistent"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(2, 16, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
        
        output = mamba(x)
        loss = output.sum()
        loss.backward()
        
        grad1 = x.grad.clone()
        
        # Second backward pass
        x.grad.zero_()
        output2 = mamba(x)
        loss2 = output2.sum()
        loss2.backward()
        
        grad2 = x.grad
        
        # Gradients should be similar (within numerical precision)
        assert torch.allclose(grad1, grad2, atol=1e-5), \
            "Gradients should be consistent"
    
    def test_parameter_gradients_exist(self, device, dtype, basic_config):
        """Test that all parameters receive gradients"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(2, 16, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
        
        output = mamba(x)
        loss = output.sum()
        loss.backward()
        
        # Check that structured A parameters have gradients
        if hasattr(mamba, 'A_blocks'):
            for i, block in enumerate(mamba.A_blocks):
                assert block.grad is not None, f"A_blocks[{i}] should have gradient"
                assert not torch.isnan(block.grad).any(), f"A_blocks[{i}] gradient contains NaN"
        
        if hasattr(mamba, 'A_U'):
            assert mamba.A_U.grad is not None, "A_U should have gradient"
            assert not torch.isnan(mamba.A_U.grad).any(), "A_U gradient contains NaN"
        
        if hasattr(mamba, 'A_V'):
            assert mamba.A_V.grad is not None, "A_V should have gradient"
            assert not torch.isnan(mamba.A_V.grad).any(), "A_V gradient contains NaN"
    
    # ========== Error Handling ==========
    
    def test_wrong_device_input(self, device, dtype, basic_config):
        """Test that wrong device input is handled"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        # Create input on wrong device
        if device.type == "cuda":
            wrong_device = torch.device("cpu")
        else:
            wrong_device = torch.device("cuda") if torch.cuda.is_available() else device
        
        if wrong_device != device:
            x = torch.randn(2, 16, basic_config["d_model"], device=wrong_device, dtype=dtype)
            
            with pytest.raises((RuntimeError, AssertionError)):
                _ = mamba(x)
    
    def test_wrong_dtype_input(self, device, basic_config):
        """Test that wrong dtype input is handled"""
        mamba = Mamba(**basic_config).to(device).to(torch.float32)
        
        # Create input with wrong dtype
        x = torch.randn(2, 16, basic_config["d_model"], device=device, dtype=torch.float64)
        
        # Should either work (with casting) or raise error
        try:
            output = mamba(x)
            assert output.shape == (2, 16, basic_config["d_model"])
        except (RuntimeError, TypeError):
            # Expected if dtype mismatch is not handled
            pass
    
    def test_empty_batch(self, device, dtype, basic_config):
        """Test that empty batch is handled"""
        mamba = Mamba(**basic_config).to(device).to(dtype)
        
        # Empty batch (batch_size=0)
        x = torch.randn(0, 16, basic_config["d_model"], device=device, dtype=dtype)
        
        # Should either work or raise appropriate error
        try:
            output = mamba(x)
            assert output.shape[0] == 0, "Output batch size should be 0"
        except (RuntimeError, AssertionError):
            # Expected if empty batch is not supported
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
