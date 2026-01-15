"""
Code validation and correctness checks for Stability Enforcement

This module contains validation tests, edge cases, and correctness checks.
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


class TestInputValidation:
    """Test input validation and error handling"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_invalid_tensor_shape(self, device):
        """Test that invalid tensor shapes raise appropriate errors"""
        # 1D tensor (invalid)
        A_1d = torch.randn(10, device=device)
        
        with pytest.raises(ValueError, match="Unsupported A matrix shape"):
            compute_spectral_radius(A_1d)
        
        # 4D tensor (invalid)
        A_4d = torch.randn(2, 3, 4, 5, device=device)
        
        with pytest.raises(ValueError, match="Unsupported A matrix shape"):
            compute_spectral_radius(A_4d)
    
    def test_empty_tensor(self, device):
        """Test handling of empty tensors"""
        # Empty diagonal matrix
        A_empty = torch.empty(0, 10, device=device)
        
        # Should handle gracefully or raise appropriate error
        try:
            spectral_radius = compute_spectral_radius(A_empty)
            # If it doesn't raise, check it's valid
            assert spectral_radius.numel() == 0
        except (ValueError, RuntimeError):
            # Acceptable to raise error for empty tensors
            pass
    
    def test_epsilon_validation(self, device):
        """Test epsilon parameter validation"""
        d_inner = 4
        d_state = 8
        
        A = torch.randn(d_inner, d_state, device=device)
        
        # Test with various epsilon values
        for epsilon in [0.0, 0.001, 0.01, 0.1, 1.0]:
            A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
            penalty = compute_stability_penalty(A, epsilon=epsilon)
            
            assert A_clamped.shape == A.shape
            assert penalty >= 0


class TestNumericalCorrectness:
    """Test numerical correctness and edge cases"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_spectral_normalization_identity(self, device):
        """Test that identity matrix normalization works correctly"""
        d_inner = 4
        d_state = 4
        
        # Identity matrix has spectral radius = 1
        A = torch.eye(d_state, device=device).unsqueeze(0).repeat(d_inner, 1, 1)
        
        A_normalized = apply_spectral_normalization(A)
        
        # Should be approximately unchanged (within numerical precision)
        assert torch.allclose(A, A_normalized, atol=1e-4)
    
    def test_spectral_normalization_zero_matrix(self, device):
        """Test normalization of zero matrix"""
        d_inner = 4
        d_state = 8
        
        A = torch.zeros(d_inner, d_state, device=device)
        
        A_normalized = apply_spectral_normalization(A)
        
        # Zero matrix should remain zero
        assert torch.allclose(A_normalized, A, atol=1e-6)
    
    def test_eigenvalue_clamping_already_stable(self, device):
        """Test clamping of already stable matrix"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        # Create matrix that's already stable (all values << -epsilon)
        A = torch.randn(d_inner, d_state, device=device) * -2.0 - 1.0
        
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        
        # Should be approximately unchanged
        assert torch.allclose(A, A_clamped, atol=1e-4)
    
    def test_stability_penalty_zero_for_stable(self, device):
        """Test that penalty is zero for stable matrices"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        # Stable matrix
        A = torch.randn(d_inner, d_state, device=device) * -1.0 - 1.0
        
        penalty = compute_stability_penalty(A, epsilon=epsilon)
        
        assert torch.allclose(penalty, torch.tensor(0.0, device=device), atol=1e-5)
    
    def test_stability_penalty_proportional(self, device):
        """Test that penalty increases with instability"""
        d_inner = 1
        d_state = 8
        epsilon = 0.01
        
        # Create matrices with increasing instability
        A1 = torch.zeros(d_inner, d_state, device=device)
        A1[0, 0] = epsilon + 0.01
        
        A2 = torch.zeros(d_inner, d_state, device=device)
        A2[0, 0] = epsilon + 0.1
        
        penalty1 = compute_stability_penalty(A1, epsilon=epsilon)
        penalty2 = compute_stability_penalty(A2, epsilon=epsilon)
        
        assert penalty2 > penalty1, "Penalty should increase with instability"


class TestCompositionCorrectness:
    """Test correctness of function composition"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_normalization_then_clamping(self, device):
        """Test applying normalization then clamping"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        A = torch.randn(d_inner, d_state, device=device) * 2.0
        
        # Apply normalization first
        A_normalized = apply_spectral_normalization(A)
        
        # Then apply clamping
        A_final = apply_eigenvalue_clamping(A_normalized, epsilon=epsilon)
        
        # Check both properties hold
        spectral_radius = compute_spectral_radius(A_final)
        assert torch.all(spectral_radius <= 1.0 + 1e-5)
        assert torch.all(A_final <= -epsilon + 1e-5)
    
    def test_clamping_then_normalization(self, device):
        """Test applying clamping then normalization"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        A = torch.randn(d_inner, d_state, device=device) * 2.0
        
        # Apply clamping first
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        
        # Then apply normalization
        A_final = apply_spectral_normalization(A_clamped)
        
        # Check both properties hold
        spectral_radius = compute_spectral_radius(A_final)
        assert torch.all(spectral_radius <= 1.0 + 1e-5)
        assert torch.all(A_final <= -epsilon + 1e-5)
    
    def test_combined_enforcement_equivalence(self, device):
        """Test that combined enforcement is equivalent to sequential"""
        d_inner = 4
        d_state = 8
        epsilon = 0.01
        
        A = torch.randn(d_inner, d_state, device=device) * 2.0
        
        # Combined application
        A_combined = apply_stability_enforcement(
            A,
            use_spectral_normalization=True,
            use_eigenvalue_clamping=True,
            epsilon=epsilon,
        )
        
        # Sequential application
        A_seq = apply_spectral_normalization(A)
        A_seq = apply_eigenvalue_clamping(A_seq, epsilon=epsilon)
        
        # Results should be the same
        assert torch.allclose(A_combined, A_seq, atol=1e-5)


class TestMambaIntegrationValidation:
    """Validation tests for Mamba integration"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_default_parameters(self, device):
        """Test that default parameters work correctly"""
        batch_size = 2
        seqlen = 16
        d_model = 64
        d_state = 16
        
        # Create Mamba with defaults (all stabilizers disabled)
        mamba = Mamba(d_model=d_model, d_state=d_state).to(device)
        
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        output = mamba(x)
        
        assert output.shape == (batch_size, seqlen, d_model)
        assert not torch.isnan(output).any()
    
    def test_parameter_combinations(self, device):
        """Test all combinations of stabilizer parameters"""
        batch_size = 2
        seqlen = 16
        d_model = 64
        d_state = 16
        
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        
        # Test all 8 combinations
        for use_spec_norm in [False, True]:
            for use_eig_clamp in [False, True]:
                for use_penalty in [False, True]:
                    mamba = Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        use_spectral_normalization=use_spec_norm,
                        use_eigenvalue_clamping=use_eig_clamp,
                        use_stability_penalty=use_penalty,
                        stability_epsilon=0.01,
                        stability_penalty_weight=0.1,
                    ).to(device)
                    
                    output = mamba(x)
                    assert output.shape == (batch_size, seqlen, d_model)
                    assert not torch.isnan(output).any()
    
    def test_stability_loss_gradient_flow(self, device):
        """Test that stability loss gradients flow correctly"""
        batch_size = 2
        seqlen = 16
        d_model = 64
        d_state = 16
        
        mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            use_stability_penalty=True,
            stability_epsilon=0.01,
            stability_penalty_weight=0.1,
        ).to(device)
        
        x = torch.randn(batch_size, seqlen, d_model, device=device, requires_grad=True)
        
        output = mamba(x)
        stability_loss = mamba.compute_stability_loss()
        total_loss = output.sum() + stability_loss
        
        total_loss.backward()
        
        # Check that A_log receives gradients from stability loss
        assert mamba.A_log.grad is not None
        assert not torch.isnan(mamba.A_log.grad).any()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_very_small_epsilon(self, device):
        """Test with very small epsilon values"""
        d_inner = 4
        d_state = 8
        epsilon = 1e-6
        
        A = torch.randn(d_inner, d_state, device=device)
        
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        penalty = compute_stability_penalty(A, epsilon=epsilon)
        
        assert A_clamped.shape == A.shape
        assert penalty >= 0
    
    def test_very_large_epsilon(self, device):
        """Test with very large epsilon values"""
        d_inner = 4
        d_state = 8
        epsilon = 10.0
        
        A = torch.randn(d_inner, d_state, device=device)
        
        A_clamped = apply_eigenvalue_clamping(A, epsilon=epsilon)
        penalty = compute_stability_penalty(A, epsilon=epsilon)
        
        assert A_clamped.shape == A.shape
        assert penalty >= 0
    
    def test_extreme_matrix_values(self, device):
        """Test with extreme matrix values"""
        d_inner = 4
        d_state = 8
        
        # Very large values
        A_large = torch.randn(d_inner, d_state, device=device) * 1e6
        
        # Very small values
        A_small = torch.randn(d_inner, d_state, device=device) * 1e-6
        
        for A in [A_large, A_small]:
            A_normalized = apply_spectral_normalization(A)
            A_clamped = apply_eigenvalue_clamping(A, epsilon=0.01)
            penalty = compute_stability_penalty(A, epsilon=0.01)
            
            assert not torch.isnan(A_normalized).any()
            assert not torch.isnan(A_clamped).any()
            assert not torch.isnan(penalty)
    
    def test_single_channel(self, device):
        """Test with single channel (d_inner=1)"""
        d_inner = 1
        d_state = 8
        
        A = torch.randn(d_inner, d_state, device=device)
        
        spectral_radius = compute_spectral_radius(A)
        A_normalized = apply_spectral_normalization(A)
        A_clamped = apply_eigenvalue_clamping(A, epsilon=0.01)
        penalty = compute_stability_penalty(A, epsilon=0.01)
        
        assert spectral_radius.shape == (d_inner,)
        assert A_normalized.shape == A.shape
        assert A_clamped.shape == A.shape
        assert penalty >= 0
    
    def test_single_state(self, device):
        """Test with single state dimension (d_state=1)"""
        d_inner = 4
        d_state = 1
        
        A = torch.randn(d_inner, d_state, device=device)
        
        spectral_radius = compute_spectral_radius(A)
        A_normalized = apply_spectral_normalization(A)
        A_clamped = apply_eigenvalue_clamping(A, epsilon=0.01)
        penalty = compute_stability_penalty(A, epsilon=0.01)
        
        assert spectral_radius.shape == (d_inner,)
        assert A_normalized.shape == A.shape
        assert A_clamped.shape == A.shape
        assert penalty >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
