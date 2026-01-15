"""
Feature-SST: CUDA-Specific Tests
Tests for GPU-specific functionality, large batch processing, and CUDA kernel validation
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSSTCUDA:
    """CUDA-specific tests for Feature-SST"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda")
    
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
    
    def test_cuda_forward_pass(self, device, dtype, basic_config):
        """Test forward pass on CUDA"""
        batch_size = 4
        seqlen = 64
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.device.type == "cuda", "Output should be on CUDA"
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "CUDA output contains NaN"
        assert not torch.isinf(output).any(), "CUDA output contains Inf"
    
    def test_large_batch_processing(self, device, dtype, basic_config):
        """Test processing large batches on GPU"""
        batch_sizes = [8, 16, 32]
        seqlen = 128
        
        for batch_size in batch_sizes:
            mamba = Mamba(**basic_config).to(device).to(dtype)
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"Large batch {batch_size} produces NaN"
            assert not torch.isinf(output).any(), f"Large batch {batch_size} produces Inf"
    
    def test_long_sequence_processing(self, device, dtype, basic_config):
        """Test processing long sequences on GPU"""
        batch_size = 4
        seq_lens = [256, 512, 1024]
        
        for seqlen in seq_lens:
            mamba = Mamba(**basic_config).to(device).to(dtype)
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"Long sequence {seqlen} produces NaN"
            assert not torch.isinf(output).any(), f"Long sequence {seqlen} produces Inf"
    
    def test_gpu_memory_efficiency(self, device, dtype, basic_config):
        """Test GPU memory efficiency"""
        batch_size = 8
        seqlen = 256
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        output = mamba(x)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory
        
        print(f"GPU memory used: {memory_used / 1024**2:.2f} MB")
        
        # Should use reasonable memory (< 500MB for this config)
        assert memory_used < 500 * 1024**2, \
            f"GPU memory usage too high: {memory_used / 1024**2:.2f} MB"
        
        # Check that output is correct
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "Output contains NaN"
    
    def test_cuda_kernel_validation(self, device, dtype, basic_config):
        """Test that CUDA kernels are being used correctly"""
        batch_size = 4
        seqlen = 64
        
        # Force CUDA kernel usage
        config = basic_config.copy()
        config["use_cuda_kernel"] = True
        
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.device.type == "cuda", "Output should be on CUDA"
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        assert not torch.isnan(output).any(), "CUDA kernel output contains NaN"
    
    def test_all_methods_cuda(self, device, dtype, basic_config):
        """Test all discretization methods on CUDA"""
        methods = ["zoh", "foh", "bilinear", "poly", "highorder", "rk4"]
        batch_size = 4
        seqlen = 64
        
        for method in methods:
            mamba = Mamba(**basic_config, discretization_method=method).to(device).to(dtype)
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            
            output = mamba(x)
            
            assert output.device.type == "cuda", f"{method} output should be on CUDA"
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"{method} CUDA output contains NaN"
    
    def test_cuda_backward_pass(self, device, dtype, basic_config):
        """Test backward pass on CUDA"""
        batch_size = 4
        seqlen = 64
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
        
        output = mamba(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Input gradient should exist"
        assert x.grad.device.type == "cuda", "Gradient should be on CUDA"
        assert not torch.isnan(x.grad).any(), "CUDA gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "CUDA gradient contains Inf"
    
    def test_throughput_benchmark(self, device, dtype, basic_config):
        """Benchmark throughput on CUDA"""
        import time
        
        batch_size = 8
        seqlen = 256
        num_iterations = 50
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            _ = mamba(x)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_iterations):
            _ = mamba(x)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (batch_size * seqlen * num_iterations) / elapsed
        print(f"Throughput: {throughput:.2f} tokens/sec")
        
        # Should achieve reasonable throughput (> 1000 tokens/sec)
        assert throughput > 1000, f"Throughput too low: {throughput:.2f} tokens/sec"
    
    def test_mixed_precision(self, device, basic_config):
        """Test mixed precision (FP16) on CUDA"""
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
            pytest.skip("FP16 requires compute capability >= 7.0")
        
        batch_size = 4
        seqlen = 64
        
        mamba = Mamba(**basic_config).to(device).half()
        x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=torch.float16)
        
        output = mamba(x)
        
        assert output.dtype == torch.float16, "Output should be FP16"
        assert output.shape == (batch_size, seqlen, basic_config["d_model"])
        # Allow some NaN/Inf in FP16 due to numerical precision
        nan_ratio = torch.isnan(output).float().mean().item()
        assert nan_ratio < 0.01, f"Too many NaN values in FP16: {nan_ratio*100:.2f}%"
    
    def test_concurrent_forward_passes(self, device, dtype, basic_config):
        """Test multiple concurrent forward passes"""
        batch_size = 4
        seqlen = 64
        num_concurrent = 3
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        inputs = [
            torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype)
            for _ in range(num_concurrent)
        ]
        
        outputs = []
        for x in inputs:
            output = mamba(x)
            outputs.append(output)
        
        # All outputs should be valid
        for i, output in enumerate(outputs):
            assert output.shape == (batch_size, seqlen, basic_config["d_model"])
            assert not torch.isnan(output).any(), f"Concurrent pass {i} produces NaN"
    
    def test_large_model_config(self, device, dtype):
        """Test with large model configuration"""
        config = {
            "d_model": 512,
            "d_state": 64,
            "block_size": 8,
            "low_rank_rank": 4,
            "use_block_diagonal_lowrank": True,
        }
        
        batch_size = 2
        seqlen = 128
        
        mamba = Mamba(**config).to(device).to(dtype)
        x = torch.randn(batch_size, seqlen, config["d_model"], device=device, dtype=dtype)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, config["d_model"])
        assert not torch.isnan(output).any(), "Large model produces NaN"
        assert not torch.isinf(output).any(), "Large model produces Inf"
    
    def test_gradient_accumulation(self, device, dtype, basic_config):
        """Test gradient accumulation on CUDA"""
        batch_size = 4
        seqlen = 64
        num_accumulation_steps = 4
        
        mamba = Mamba(**basic_config).to(device).to(dtype)
        optimizer = torch.optim.Adam(mamba.parameters(), lr=1e-4)
        
        # Simulate gradient accumulation
        for step in range(num_accumulation_steps):
            x = torch.randn(batch_size, seqlen, basic_config["d_model"], device=device, dtype=dtype, requires_grad=True)
            
            output = mamba(x)
            loss = output.sum() / num_accumulation_steps
            loss.backward()
        
        # Check gradients exist
        for param in mamba.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Parameter gradient should exist"
                assert not torch.isnan(param.grad).any(), "Parameter gradient contains NaN"
        
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
