# Copyright (c) 2024, Vision Mamba Stability Enforcement
# Feature-StabEnforce: Stability Enforcement on State Dynamics

"""
Stability enforcement utilities for Vision Mamba state-space models.

This module implements:
1. Spectral normalization: A ← A / max(1, ρ(A))
2. Eigenvalue clamping for block matrices
3. Soft penalty loss: L_stab = Σ max(0, ℜ(λ_i(A)) - ε)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def compute_spectral_radius(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the spectral radius (largest eigenvalue magnitude) of matrix A.
    
    Args:
        A: Tensor of shape (..., d_state, d_state) or (..., d_state) for diagonal
        
    Returns:
        spectral_radius: Tensor of shape (...) with spectral radius for each matrix
    """
    if A.dim() == 2:
        # Diagonal case: (d_inner, d_state)
        # For diagonal matrices, spectral radius is max(|diagonal|)
        return torch.abs(A).max(dim=-1)[0]
    elif A.dim() == 3:
        # Full matrix case: (d_inner, d_state, d_state)
        # Compute eigenvalues for each matrix
        eigenvalues = torch.linalg.eigvals(A)  # (d_inner, d_state) complex
        spectral_radius = torch.abs(eigenvalues).max(dim=-1)[0]  # (d_inner,)
        return spectral_radius
    else:
        raise ValueError(f"Unsupported A matrix shape: {A.shape}")


def apply_spectral_normalization(A: torch.Tensor) -> torch.Tensor:
    """
    Apply spectral normalization: A ← A / max(1, ρ(A))
    
    This ensures the spectral radius ρ(A) ≤ 1, guaranteeing stable dynamics.
    
    Args:
        A: Tensor of shape (d_inner, d_state, d_state) or (d_inner, d_state)
        
    Returns:
        A_normalized: Normalized A matrix with same shape
    """
    if A.dim() == 2:
        # Diagonal case: (d_inner, d_state)
        spectral_radius = compute_spectral_radius(A)  # (d_inner,)
        # Normalize each row (channel) independently
        normalization_factor = torch.clamp(spectral_radius, min=1.0)  # (d_inner,)
        A_normalized = A / normalization_factor.unsqueeze(-1)  # (d_inner, d_state)
        return A_normalized
    elif A.dim() == 3:
        # Full matrix case: (d_inner, d_state, d_state)
        spectral_radius = compute_spectral_radius(A)  # (d_inner,)
        # Normalize each matrix independently
        normalization_factor = torch.clamp(spectral_radius, min=1.0)  # (d_inner,)
        A_normalized = A / normalization_factor.unsqueeze(-1).unsqueeze(-1)  # (d_inner, d_state, d_state)
        return A_normalized
    else:
        raise ValueError(f"Unsupported A matrix shape: {A.shape}")


def apply_eigenvalue_clamping(A: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """
    Apply eigenvalue clamping to ensure stability.
    
    For continuous-time SSMs, stability requires ℜ(λ_i) < 0.
    This function clamps eigenvalues to ensure ℜ(λ_i) ≤ -ε.
    
    Args:
        A: Tensor of shape (d_inner, d_state, d_state) - full matrix
        epsilon: Threshold for stability margin (default: 0.01)
        
    Returns:
        A_clamped: Clamped A matrix with same shape
    """
    if A.dim() == 2:
        # Diagonal case: (d_inner, d_state)
        # For diagonal matrices, clamp the diagonal values directly
        # Ensure real part is ≤ -epsilon
        A_clamped = torch.clamp(A, max=-epsilon)
        return A_clamped
    elif A.dim() == 3:
        # Full matrix case: (d_inner, d_state, d_state)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eig(A)  # eigenvalues: (d_inner, d_state) complex
        
        # Clamp real parts: ℜ(λ_i) ≤ -epsilon
        real_parts = eigenvalues.real
        clamped_real_parts = torch.clamp(real_parts, max=-epsilon)
        # Keep imaginary parts unchanged
        clamped_eigenvalues = torch.complex(clamped_real_parts, eigenvalues.imag)
        
        # Reconstruct matrix: A = V * diag(λ) * V^-1
        # eigenvectors: (d_inner, d_state, d_state) complex
        # For each channel, reconstruct the matrix
        d_inner, d_state = A.shape[0], A.shape[1]
        A_clamped = torch.zeros_like(A, dtype=A.dtype)
        
        for i in range(d_inner):
            V = eigenvectors[i]  # (d_state, d_state) complex
            lambda_diag = torch.diag(clamped_eigenvalues[i])  # (d_state, d_state) complex
            V_inv = torch.linalg.inv(V)  # (d_state, d_state) complex
            A_clamped[i] = (V @ lambda_diag @ V_inv).real  # (d_state, d_state)
        
        return A_clamped
    else:
        raise ValueError(f"Unsupported A matrix shape: {A.shape}")


def compute_stability_penalty(A: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """
    Compute stability penalty loss: L_stab = Σ_i max(0, ℜ(λ_i(A)) - ε)
    
    This penalizes eigenvalues with positive real parts (unstable).
    
    Args:
        A: Tensor of shape (d_inner, d_state, d_state) or (d_inner, d_state)
        epsilon: Threshold for stability margin (default: 0.01)
        
    Returns:
        penalty: Scalar tensor with total penalty
    """
    if A.dim() == 2:
        # Diagonal case: (d_inner, d_state)
        # For diagonal matrices, penalty is sum of max(0, A - epsilon)
        penalty = torch.clamp(A - epsilon, min=0.0).sum()
        return penalty
    elif A.dim() == 3:
        # Full matrix case: (d_inner, d_state, d_state)
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(A)  # (d_inner, d_state) complex
        real_parts = eigenvalues.real  # (d_inner, d_state)
        # Penalty: sum over all eigenvalues and channels
        penalty = torch.clamp(real_parts - epsilon, min=0.0).sum()
        return penalty
    else:
        raise ValueError(f"Unsupported A matrix shape: {A.shape}")


def apply_stability_enforcement(
    A: torch.Tensor,
    use_spectral_normalization: bool = False,
    use_eigenvalue_clamping: bool = False,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    Apply stability enforcement to A matrix.
    
    This function applies the requested stabilizers in order:
    1. Spectral normalization (if enabled)
    2. Eigenvalue clamping (if enabled)
    
    Args:
        A: Tensor of shape (d_inner, d_state, d_state) or (d_inner, d_state)
        use_spectral_normalization: Whether to apply spectral normalization
        use_eigenvalue_clamping: Whether to apply eigenvalue clamping
        epsilon: Threshold for eigenvalue clamping and penalty
        
    Returns:
        A_stabilized: Stabilized A matrix with same shape
    """
    A_stabilized = A
    
    if use_spectral_normalization:
        A_stabilized = apply_spectral_normalization(A_stabilized)
    
    if use_eigenvalue_clamping:
        A_stabilized = apply_eigenvalue_clamping(A_stabilized, epsilon=epsilon)
    
    return A_stabilized

