"""
CUDA-specific tests for Stability Enforcement

This module contains CUDA kernel tests and GPU-specific validation
for the stability enforcement feature.
"""

import torch
import pytest
import numpy as np

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


@pytest.mark.cuda
class TestCUDASpectralRadius:
    """CUDA tests for spectral radius computation"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    def test_large_batch_spectral_radius(self, device):
        """Test spectral radius computation with large batch on CUDA"""
        d_inner = 512
        d_state = 128
        
        A = torch.randn(d_inner, d_state, device=device)
        
        spectral_radius = compute_spectral_radius(A)
        
        assert spectral_radius.shape == (d_inner,)
        assert torch.all(spectral_radius >= 0)
        assert not torch.isnan(spectral_radius).any()
    
    def test_full_matrix_spectral_radius_cuda(self, device):
        """Test spectral radius for full matrices on CUDA"""
        d_inner = 64
        d_state = 32
        
        A = torch.randn(d_inner, d_state, d_state, device=device)
        
        spectral_radius = compute_spectral_radius(A)
        
        assert spectral_radius.shape == (d_inner,)
        assert torch.all(spectral_radius >= 0)
        assert not torch.isnan(spectral_radius).any()


@pytest.mark.cuda
class TestCUDANormalization:
    """CUDA tests for spectral normalization"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    def test_large_matrix_normalization(self, device):
        """Test normalization with large matrices on CUDA"""
        d_inner = 512
        d_state = 256
        
        A = torch.randn(d_inner, d_state, device=device) * 5.0
        
        A_normalized = apply_spectral_normalization(A)
        
        spectral_radius = compute_spectral_radius(A_normalized)
        assert torch.all(spectral_radius <= 1.0 + 1e-5)
    
    def test_mixed_spectral_radii(self, device):
        """Test normalization with mixed spectral radii"""
        d_inner = 128
        d_state = 64
        
        # Create matrix with some channels having large spectral radius
        A = torch.randn(d_inner, d_state, device=device)
        A[0:32, :] *= 3.0  # First 32 channels have large spectral radius
        A[32:64, :] *= 0.5  # Next 32 channels have small spectral radius
        
        A_normalized = apply_spectral_normalization(A)
        
        spectral_radius = compute_spectral_radius(A_normalized)
        assert torch.all(spectral_radius <= 1.0 + 1e-5)


@pytest.mark.cuda
class TestCUDAEigenvalueClamping:
    """CUDA tests for eigenvalue clamping"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    def test_large_batch_clamping(self, device):
        """Test eigenvalue clamping with large batch on CUDA"""
        d_inner = 512
        d_state = 128
        epsilon = 0.01
        
        A = torch.randn(d_inner, d_state, device=device)
        
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        
        assert torch.all(A_clamped <= -epsilon + 1e-5)
        assert A_clamped.shape == A.shape
    
    def test_full_matrix_clamping_cuda(self, device):
        """Test eigenvalue clamping for full matrices on CUDA"""
        d_inner = 32
        d_state = 16
        epsilon = 0.01
        
        A = torch.randn(d_inner, d_state, d_state, device=device)
        
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        
        # Check eigenvalues
        eigenvalues = torch.linalg.eigvals(A_clamped)
        real_parts = eigenvalues.real
        assert torch.all(real_parts <= -epsilon + 1e-5)


@pytest.mark.cuda
class TestCUDAPenalty:
    """CUDA tests for stability penalty"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    def test_large_batch_penalty(self, device):
        """Test penalty computation with large batch on CUDA"""
        d_inner = 512
        d_state = 128
        epsilon = 0.01
        
        A = torch.randn(d_inner, d_state, device=device)
        
        penalty = compute_stability_penalty(A, epsilon=epsilon)
        
        assert penalty >= 0
        assert not torch.isnan(penalty)
        assert penalty.numel() == 1  # Scalar


@pytest.mark.cuda
class TestCUDAMambaIntegration:
    """CUDA integration tests for Mamba with stability enforcement"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    def test_large_batch_forward(self, device):
        """Test forward pass with large batch on CUDA"""
        batch_size = 16
        seqlen = 512
        d_model = 256
        d_state = 64
        
        mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=True,
            stability_epsilon=0.01,
            stability_penalty_weight=0.1,
        ).to(device)
        
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_long_sequence_forward(self, device):
        """Test forward pass with long sequences on CUDA"""
        batch_size = 4
        seqlen = 2048
        d_model = 128
        d_state = 32
        
        mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=False,
        ).to(device)
        
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, d_model)
        assert not torch.isnan(output).any()
    
    def test_gradient_accumulation(self, device):
        """Test gradient accumulation on CUDA"""
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
            stability_penalty_weight=0.1,
        ).to(device)
        
        optimizer = torch.optim.Adam(mamba.parameters(), lr=1e-3)
        
        for _ in range(3):
            x = torch.randn(batch_size, seqlen, d_model, device=device, requires_grad=True)
            
            output = mamba(x)
            stability_loss = mamba.compute_stability_loss()
            loss = output.sum() + stability_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check that model still works after updates
        x_test = torch.randn(batch_size, seqlen, d_model, device=device)
        output_test = mamba(x_test)
        
        assert output_test.shape == (batch_size, seqlen, d_model)
        assert not torch.isnan(output_test).any()


@pytest.mark.cuda
class TestCUDAPerformance:
    """CUDA performance tests"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")
    
    def test_throughput_benchmark(self, device):
        """Benchmark throughput on CUDA"""
        import time
        
        batch_size = 8
        seqlen = 256
        d_model = 256
        d_state = 64
        
        mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=False,
        ).to(device)
        
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        
        # Warmup
        for _ in range(10):
            _ = mamba(x)
        
        torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        start = time.time()
        for _ in range(num_iterations):
            _ = mamba(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (batch_size * num_iterations) / elapsed
        print(f"Throughput: {throughput:.2f} samples/sec")
        
        assert throughput > 10, f"Throughput too low: {throughput:.2f} samples/sec"
    
    def test_memory_efficiency(self, device):
        """Test memory efficiency on CUDA"""
        batch_size = 4
        seqlen = 512
        d_model = 256
        d_state = 64
        
        mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            use_stability_penalty=True,
        ).to(device)
        
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        
        torch.cuda.reset_peak_memory_stats()
        output = mamba(x)
        stability_loss = mamba.compute_stability_loss()
        _ = output.sum() + stability_loss
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"Peak memory: {peak_memory:.2f} MB")
        
        # Should use reasonable memory
        assert peak_memory < 4096, f"Memory usage too high: {peak_memory:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "cuda"])
