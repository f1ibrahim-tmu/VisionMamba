"""
Pytest configuration and shared fixtures for Stability Enforcement tests
"""

import pytest
import torch
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available"""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def default_device():
    """Get default device for tests"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def default_dtype():
    """Get default dtype for tests"""
    return torch.float32


@pytest.fixture(autouse=True)
def set_deterministic():
    """Set deterministic behavior for reproducibility"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
