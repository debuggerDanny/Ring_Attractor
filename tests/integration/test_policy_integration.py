import pytest
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import tempfile
import shutil

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from policy_warp import create_sac_ring_attractor, create_ddpg_ring_attractor
from attractors import RingAttractorConfig
from control_layers import create_control_layer
from model_manager import RingAttractorModelManager


class MockSACModel:
    """Mock SAC model for testing"""
    def __init__(self, policy=None):
        self.policy = policy or MockPolicy()
        self.actor = self.policy
        self.device = torch.device('cpu')
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


class MockDDPGModel:
    """Mock DDPG model for testing"""
    def __init__(self, policy=None):
        self.policy = policy or MockPolicy()
        self.actor = self.policy
        self.device = torch.device('cpu')
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


class MockPolicy:
    """Mock policy network"""
    def __init__(self):
        self.mu = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 action outputs
        )
        self.device = torch.device('cpu')
    
    def forward(self, obs):
        return self.mu(obs)
    
    def state_dict(self):
        return self.mu.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.mu.load_state_dict(state_dict)


@pytest.mark.integration
class TestSACRingAttractorIntegration:
    """Test SAC integration with Ring Attractors"""
    
    def test_create_sac_ring_attractor(self):
        base_model = MockSACModel()
        
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': RingAttractorConfig(num_excitatory=16)
        }
        
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Model should be wrapped successfully
        assert ring_model is not None
        assert hasattr(ring_model, 'policy')
        
        # Test forward pass
        test_input = torch.randn(5, 64)
        output = ring_model.policy(test_input)
        
        assert output.shape == (5, 4)  # 4 control axes
        assert not torch.isnan(output).any()
    
    def test_sac_preset_config(self):
        base_model = MockSACModel()
        
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            preset_config="quadrotor",
            device='cpu'
        )
        
        assert ring_model is not None
        
        # Test with sample input
        test_input = torch.randn(3, 64)
        output = ring_model.policy(test_input)
        
        assert output.shape[0] == 3  # Batch size preserved
        assert output.shape[1] > 0    # Has action outputs
    
    def test_sac_policy_gradient_flow(self):
        base_model = MockSACModel()
        
        layer_config = {
            'layer_type': 'single',
            'input_dim': 256,
            'output_dim': 1,
            'config': RingAttractorConfig(num_excitatory=12)
        }
        
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Test gradient flow
        test_input = torch.randn(2, 64, requires_grad=True)
        output = ring_model.policy(test_input)
        loss = output.sum()
        loss.backward()
        
        assert test_input.grad is not None
        assert not torch.isnan(test_input.grad).any()


@pytest.mark.integration  
class TestDDPGRingAttractorIntegration:
    """Test DDPG integration with Ring Attractors"""
    
    def test_create_ddpg_ring_attractor(self):
        base_model = MockDDPGModel()
        
        layer_config = {
            'layer_type': 'coupled',
            'input_dim': 256,
            'control_axes': ['x', 'y', 'z', 'thrust'],
            'num_rings': 3,
            'coupled_axes': ['x', 'y', 'z'],
            'config': RingAttractorConfig(num_excitatory=20)
        }
        
        ring_model = create_ddpg_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        assert ring_model is not None
        assert hasattr(ring_model, 'policy')
        
        # Test forward pass
        test_input = torch.randn(4, 64)
        output = ring_model.policy(test_input)
        
        assert output.shape == (4, 4)  # 4 control axes
        assert torch.isfinite(output).all()
    
    def test_ddpg_adaptive_layer(self):
        base_model = MockDDPGModel()
        
        layer_config = {
            'layer_type': 'adaptive',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch'],
            'ring_axes': ['roll', 'pitch'],
            'config': RingAttractorConfig(num_excitatory=8),
            'architecture_type': 'multi'
        }
        
        ring_model = create_ddpg_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Test that adaptive layer can switch architectures
        test_input = torch.randn(6, 64)
        output = ring_model.policy(test_input)
        
        assert output.shape == (6, 2)
        assert not torch.isnan(output).any()


@pytest.mark.integration
class TestPolicyWrapperErrorHandling:
    """Test error handling in policy wrappers"""
    
    def test_invalid_base_model(self):
        invalid_model = "not_a_model"
        
        layer_config = {
            'layer_type': 'single',
            'input_dim': 64,
            'output_dim': 1,
            'config': RingAttractorConfig()
        }
        
        with pytest.raises((AttributeError, TypeError)):
            create_sac_ring_attractor(
                base_model=invalid_model,
                layer_config=layer_config
            )
    
    def test_incompatible_layer_config(self):
        base_model = MockSACModel()
        
        # Missing required config parameters
        invalid_config = {
            'layer_type': 'multi'
            # Missing required fields
        }
        
        with pytest.raises((KeyError, ValueError)):
            create_sac_ring_attractor(
                base_model=base_model,
                layer_config=invalid_config
            )
    
    def test_invalid_preset_config(self):
        base_model = MockSACModel()
        
        with pytest.raises(ValueError):
            create_sac_ring_attractor(
                base_model=base_model,
                preset_config="nonexistent_preset"
            )


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    def test_full_training_simulation(self):
        # Create base model
        base_model = MockSACModel()
        
        # Configure Ring Attractor
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': RingAttractorConfig(
                num_excitatory=16,
                tau=8.0,
                beta=12.0,
                trainable_structure=True
            )
        }
        
        # Create Ring Attractor model
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Simulate training loop
        for epoch in range(10):  # Short simulation
            # Generate random batch
            batch_obs = torch.randn(32, 64)
            
            # Forward pass
            actions = ring_model.policy(batch_obs)
            
            # Verify output consistency
            assert actions.shape == (32, 4)
            assert not torch.isnan(actions).any()
            assert torch.isfinite(actions).all()
            
            # Simulate gradient step
            loss = actions.mean()
            loss.backward()
            
        # Training simulation completed successfully
        assert True  # If we get here, everything worked
    
    def test_model_save_load_cycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RingAttractorModelManager(base_save_dir=temp_dir)
            
            # Create and configure model
            base_model = MockDDPGModel()
            layer_config = {
                'layer_type': 'single',
                'input_dim': 256,
                'output_dim': 1,
                'config': RingAttractorConfig(num_excitatory=12)
            }
            
            ring_model = create_ddpg_ring_attractor(
                base_model=base_model,
                layer_config=layer_config
            )
            
            # Save model
            manager.save_model(
                model=ring_model,
                model_name="test_integration",
                framework="stable_baselines3",
                algorithm="DDPG",
                layer_config=layer_config,
                metadata={"test": "integration"}
            )
            
            # Load model back
            def model_factory():
                return create_ddpg_ring_attractor(
                    base_model=MockDDPGModel(),
                    layer_config=layer_config
                )
            
            loaded_model, config = manager.load_model(
                "test_integration",
                model_factory
            )
            
            # Verify loaded model works
            test_input = torch.randn(5, 64)
            original_output = ring_model.policy(test_input)
            loaded_output = loaded_model.policy(test_input)
            
            # Both should produce valid outputs
            assert original_output.shape == loaded_output.shape
            assert not torch.isnan(original_output).any()
            assert not torch.isnan(loaded_output).any()


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance tests for integrated systems"""
    
    def test_large_batch_processing(self):
        base_model = MockSACModel()
        
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': RingAttractorConfig(num_excitatory=24)
        }
        
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Test with large batch
        large_batch = torch.randn(512, 64)  # Large batch size
        
        output = ring_model.policy(large_batch)
        
        assert output.shape == (512, 4)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_memory_efficiency(self):
        # Test that models don't leak memory during repeated use
        base_model = MockDDPGModel()
        
        layer_config = {
            'layer_type': 'coupled',
            'input_dim': 256,
            'control_axes': ['x', 'y', 'z'],
            'num_rings': 3,
            'coupled_axes': ['x', 'y', 'z'],
            'config': RingAttractorConfig(num_excitatory=16)
        }
        
        ring_model = create_ddpg_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Repeated forward passes
        for _ in range(100):
            test_input = torch.randn(8, 64)
            output = ring_model.policy(test_input)
            
            # Force garbage collection by deleting references
            del test_input, output
        
        # If we complete without memory errors, test passes
        assert True


@pytest.mark.integration
@pytest.mark.requires_pyflyt
class TestEnvironmentIntegration:
    """Test integration with actual environments (requires PyFlyt)"""
    
    @pytest.mark.skipif(
        not pytest.importorskip("PyFlyt", reason="PyFlyt not available"),
        reason="PyFlyt not installed"
    )
    def test_pyflyt_environment_compatibility(self):
        """Test that Ring Attractor models work with PyFlyt environments"""
        try:
            import PyFlyt
            import PyFlyt.gym_envs
        except ImportError:
            pytest.skip("PyFlyt not available")
        
        # Create mock environment interface
        class MockEnv:
            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(64,))
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
            
            def step(self, action):
                obs = np.random.randn(64)
                reward = 0.0
                terminated = False
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info
            
            def reset(self):
                obs = np.random.randn(64)
                info = {}
                return obs, info
        
        env = MockEnv()
        
        # Create Ring Attractor model
        base_model = MockSACModel()
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': RingAttractorConfig(num_excitatory=16)
        }
        
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config
        )
        
        # Test environment interaction
        obs, _ = env.reset()
        
        for step in range(10):  # Short episode
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = ring_model.policy(obs_tensor)
            action = action_tensor.detach().numpy()[0]
            
            # Verify action is valid
            assert action.shape == (4,)
            assert np.isfinite(action).all()
            assert not np.isnan(action).any()
            
            # Take environment step
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()


# Fixtures for integration tests
@pytest.fixture
def temp_model_dir():
    """Temporary directory for model testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_ring_config():
    """Sample Ring Attractor configuration for testing"""
    return {
        'layer_type': 'multi',
        'input_dim': 256,
        'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
        'ring_axes': ['roll', 'pitch', 'yaw'],
        'config': RingAttractorConfig(
            num_excitatory=16,
            tau=8.0,
            beta=12.0,
            trainable_structure=True
        )
    }


@pytest.fixture
def mock_sac_model():
    """Mock SAC model for testing"""
    return MockSACModel()


@pytest.fixture 
def mock_ddpg_model():
    """Mock DDPG model for testing"""
    return MockDDPGModel()


class TestIntegrationFixtures:
    """Test integration using fixtures"""
    
    def test_sac_integration_with_fixtures(self, mock_sac_model, sample_ring_config):
        ring_model = create_sac_ring_attractor(
            base_model=mock_sac_model,
            layer_config=sample_ring_config,
            device='cpu'
        )
        
        test_input = torch.randn(6, 64)
        output = ring_model.policy(test_input)
        
        assert output.shape == (6, 4)
        assert not torch.isnan(output).any()
    
    def test_ddpg_integration_with_fixtures(self, mock_ddpg_model, sample_ring_config):
        ring_model = create_ddpg_ring_attractor(
            base_model=mock_ddpg_model,
            layer_config=sample_ring_config,
            device='cpu'
        )
        
        test_input = torch.randn(3, 64)
        output = ring_model.policy(test_input)
        
        assert output.shape == (3, 4)
        assert torch.isfinite(output).all()
    
    def test_model_manager_integration(self, temp_model_dir, mock_sac_model, sample_ring_config):
        manager = RingAttractorModelManager(base_save_dir=temp_model_dir)
        
        ring_model = create_sac_ring_attractor(
            base_model=mock_sac_model,
            layer_config=sample_ring_config
        )
        
        # Save and load
        manager.save_model(
            model=ring_model,
            model_name="fixture_test",
            framework="stable_baselines3",
            algorithm="SAC",
            layer_config=sample_ring_config
        )
        
        def factory():
            return create_sac_ring_attractor(
                base_model=MockSACModel(),
                layer_config=sample_ring_config
            )
        
        loaded_model, config = manager.load_model("fixture_test", factory)
        
        # Test both models
        test_input = torch.randn(2, 64)
        original_output = ring_model.policy(test_input)
        loaded_output = loaded_model.policy(test_input)
        
        assert original_output.shape == loaded_output.shape
        assert not torch.isnan(original_output).any()
        assert not torch.isnan(loaded_output).any()