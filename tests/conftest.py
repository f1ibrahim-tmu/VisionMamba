"""
Pytest configuration and shared fixtures for Feature-SST tests
"""

import pytest
import torch
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def device():
    """Get device for testing (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def dtype():
    """Get default dtype for testing"""
    return torch.float32


@pytest.fixture(scope="session")
def seed():
    """Set random seed for reproducibility"""
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    return seed_value


@pytest.fixture(autouse=True)
def reset_seed(seed):
    """Reset random seed before each test"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def basic_sst_config():
    """Basic Feature-SST configuration for testing"""
    return {
        "d_model": 64,
        "d_state": 16,
        "block_size": 4,
        "low_rank_rank": 2,
        "use_block_diagonal_lowrank": True,
    }


@pytest.fixture
def large_sst_config():
    """Large Feature-SST configuration for testing"""
    return {
        "d_model": 256,
        "d_state": 64,
        "block_size": 8,
        "low_rank_rank": 4,
        "use_block_diagonal_lowrank": True,
    }


@pytest.fixture
def minimal_sst_config():
    """Minimal Feature-SST configuration for testing"""
    return {
        "d_model": 32,
        "d_state": 8,
        "block_size": 2,
        "low_rank_rank": 1,
        "use_block_diagonal_lowrank": True,
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark CUDA tests
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.cuda)
        
        # Mark slow tests
        if "benchmark" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)
