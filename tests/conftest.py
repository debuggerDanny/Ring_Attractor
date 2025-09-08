"""
Pytest configuration and shared fixtures for Ring Attractor tests.

This file contains common fixtures, test configuration, and utilities
used across all test modules.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to path so tests can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from attractors import RingAttractorConfig
from model_manager import RingAttractorModelManager


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU/CUDA"
    )
    config.addinivalue_line(
        "markers", "requires_pyflyt: Tests requiring PyFlyt environment"
    )
    config.addinivalue_line(
        "markers", "benchmark: Performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests with 'slow' in name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Device fixtures
@pytest.fixture(scope="session")
def device():
    """Get available device for testing"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture(params=["cpu", "cuda"])
def device_parametrized(request):
    """Parametrized fixture for testing on different devices"""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# Configuration fixtures
@pytest.fixture
def ring_config_small():
    """Small Ring Attractor configuration for fast tests"""
    return RingAttractorConfig(
        num_excitatory=8,
        tau=6.0,
        beta=8.0,
        lambda_decay=0.8,
        trainable_structure=True
    )


@pytest.fixture
def ring_config_medium():
    """Medium Ring Attractor configuration"""
    return RingAttractorConfig(
        num_excitatory=16,
        tau=8.0,
        beta=12.0,
        lambda_decay=0.9,
        trainable_structure=True,
        connectivity_strength=0.1
    )


@pytest.fixture
def ring_config_large():
    """Large Ring Attractor configuration for thorough tests"""
    return RingAttractorConfig(
        num_excitatory=24,
        tau=10.0,
        beta=15.0,
        lambda_decay=0.95,
        trainable_structure=True,
        connectivity_strength=0.15,
        cross_coupling_factor=0.1
    )


@pytest.fixture
def layer_config_single(ring_config_medium):
    """Single axis layer configuration"""
    return {
        'layer_type': 'single',
        'input_dim': 64,
        'output_dim': 1,
        'config': ring_config_medium
    }


@pytest.fixture
def layer_config_multi(ring_config_medium):
    """Multi-axis layer configuration"""
    return {
        'layer_type': 'multi',
        'input_dim': 128,
        'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
        'ring_axes': ['roll', 'pitch', 'yaw'],
        'config': ring_config_medium
    }


@pytest.fixture
def layer_config_coupled(ring_config_medium):
    """Coupled rings layer configuration"""
    return {
        'layer_type': 'coupled',
        'input_dim': 96,
        'control_axes': ['x', 'y', 'z', 'thrust'],
        'num_rings': 3,
        'coupled_axes': ['x', 'y', 'z'],
        'config': ring_config_medium
    }


# Data fixtures
@pytest.fixture
def sample_batch_small():
    """Small batch of sample data for fast tests"""
    return torch.randn(4, 32)


@pytest.fixture
def sample_batch_medium():
    """Medium batch of sample data"""
    return torch.randn(16, 64)


@pytest.fixture
def sample_batch_large():
    """Large batch of sample data for stress tests"""
    return torch.randn(128, 128)


@pytest.fixture(params=[1, 4, 16])
def batch_sizes_parametrized(request):
    """Parametrized batch sizes for testing"""
    return request.param


@pytest.fixture(params=[32, 64, 128])
def input_dims_parametrized(request):
    """Parametrized input dimensions for testing"""
    return request.param


# Temporary directory fixtures
@pytest.fixture
def temp_dir():
    """Temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_model_dir():
    """Temporary directory specifically for model testing"""
    temp_path = tempfile.mkdtemp(prefix="ring_attractor_models_")
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def model_manager(temp_model_dir):
    """Model manager with temporary directory"""
    return RingAttractorModelManager(base_save_dir=temp_model_dir)


# Mock model fixtures
class MockModel:
    """Mock RL model for testing"""
    def __init__(self):
        self.policy = MockPolicy()
        self.device = torch.device('cpu')
        
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def predict(self, obs, deterministic=True):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0) if len(obs.shape) == 1 else torch.FloatTensor(obs)
        action_tensor = self.policy(obs_tensor)
        action = action_tensor.detach().numpy()
        return action.squeeze(), None


class MockPolicy:
    """Mock policy network"""
    def __init__(self, input_dim=64, output_dim=4):
        import torch.nn as nn
        self.mu = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(), 
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.mu(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def state_dict(self):
        return self.mu.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.mu.load_state_dict(state_dict)


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    return MockModel()


@pytest.fixture
def mock_policy():
    """Mock policy for testing"""
    return MockPolicy()


# Environment fixtures
class MockEnvironment:
    """Mock environment for testing"""
    def __init__(self, obs_dim=64, action_dim=4, max_steps=100):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        obs = np.random.randn(self.obs_dim)
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(self.obs_dim)
        reward = np.random.rand() - 0.5
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {'step': self.step_count}
        return obs, reward, terminated, truncated, info


@pytest.fixture
def mock_env():
    """Mock environment for testing"""
    return MockEnvironment()


@pytest.fixture(params=[32, 64, 128])
def mock_env_parametrized(request):
    """Parametrized mock environment with different observation dimensions"""
    return MockEnvironment(obs_dim=request.param)


# Utility fixtures
@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def deterministic_setup():
    """Setup for deterministic testing"""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    # Reset to non-deterministic
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer utility for performance tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time or time.perf_counter()
            return end - self.start_time
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, *args):
            self.stop()
    
    return Timer()


# Skip conditions
skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), 
    reason="CUDA not available"
)

skip_if_no_pyflyt = pytest.mark.skipif(
    not pytest.importorskip("PyFlyt", reason="PyFlyt not available"),
    reason="PyFlyt not installed"
)


# Test data generation utilities
def generate_test_observations(batch_size=8, obs_dim=64, noise_level=0.1):
    """Generate test observations with optional noise"""
    obs = torch.randn(batch_size, obs_dim)
    if noise_level > 0:
        obs += torch.randn_like(obs) * noise_level
    return obs


def generate_test_actions(batch_size=8, action_dim=4, action_range=(-1, 1)):
    """Generate test actions within specified range"""
    actions = torch.rand(batch_size, action_dim)
    actions = actions * (action_range[1] - action_range[0]) + action_range[0]
    return actions


# Custom assertions for testing
def assert_valid_tensor(tensor, expected_shape=None, finite_only=True):
    """Assert tensor is valid with optional shape and finite checks"""
    assert torch.is_tensor(tensor), "Input is not a tensor"
    
    if expected_shape is not None:
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    if finite_only:
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"


def assert_model_weights_changed(model_before, model_after, tolerance=1e-8):
    """Assert that model weights have changed (for training tests)"""
    params_before = dict(model_before.named_parameters())
    params_after = dict(model_after.named_parameters())
    
    changes_found = False
    for name in params_before:
        if name in params_after:
            diff = torch.abs(params_before[name] - params_after[name]).max()
            if diff > tolerance:
                changes_found = True
                break
    
    assert changes_found, "No significant weight changes detected"


# Add custom assertions to pytest namespace
@pytest.fixture(autouse=True)
def add_custom_assertions(request):
    """Add custom assertion functions to test namespace"""
    request.cls.assert_valid_tensor = assert_valid_tensor if request.cls else None
    request.cls.assert_model_weights_changed = assert_model_weights_changed if request.cls else None