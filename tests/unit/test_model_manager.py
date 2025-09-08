import pytest
import torch
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_manager import (
    RingAttractorModelManager,
    ModelRegistry,
    save_ring_attractor_model,
    load_ring_attractor_model
)
from attractors import RingAttractorConfig


class TestRingAttractorModelManager:
    """Test RingAttractorModelManager class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_model(self):
        """Create a mock model for testing"""
        model = MagicMock()
        model.save = MagicMock()
        model.policy = MagicMock()
        model.policy.state_dict = MagicMock(return_value={'param1': torch.tensor([1.0])})
        return model
    
    @pytest.fixture
    def sample_layer_config(self):
        """Sample layer configuration for testing"""
        return {
            'layer_type': 'multi',
            'input_dim': 128,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': RingAttractorConfig(
                num_excitatory=16,
                tau=8.0,
                beta=12.0
            )
        }
    
    def test_initialization(self, temp_dir):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        assert manager.base_save_dir == Path(temp_dir)
        assert manager.base_save_dir.exists()
    
    def test_initialization_creates_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_path = os.path.join(temp_dir, "new_models")
            manager = RingAttractorModelManager(base_save_dir=non_existent_path)
            
            assert manager.base_save_dir.exists()
    
    def test_save_model_full(self, temp_dir, sample_model, sample_layer_config):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        manager.save_model(
            model=sample_model,
            model_name="test_model",
            framework="stable_baselines3",
            algorithm="DDPG",
            layer_config=sample_layer_config,
            metadata={"test": "data"}
        )
        
        # Check that model directory was created
        model_dir = manager.base_save_dir / "test_model"
        assert model_dir.exists()
        
        # Check that config file was created
        config_file = model_dir / "config.json"
        assert config_file.exists()
        
        # Check that model.save was called
        sample_model.save.assert_called_once()
    
    def test_save_model_policy_only(self, temp_dir, sample_model, sample_layer_config):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        manager.save_model(
            model=sample_model,
            model_name="policy_model",
            framework="stable_baselines3",
            algorithm="SAC",
            layer_config=sample_layer_config,
            save_policy_only=True
        )
        
        model_dir = manager.base_save_dir / "policy_model"
        assert model_dir.exists()
        
        # Check that policy weights were saved
        policy_file = model_dir / "policy_weights.pt"
        assert policy_file.exists()
        
        # Model.save should not be called for policy-only saves
        sample_model.save.assert_not_called()
    
    def test_save_model_config_serialization(self, temp_dir, sample_model, sample_layer_config):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        metadata = {
            "training_steps": 100000,
            "performance": {"mean_reward": 450.2},
            "hyperparameters": {"learning_rate": 3e-4}
        }
        
        manager.save_model(
            model=sample_model,
            model_name="config_test",
            framework="stable_baselines3",
            algorithm="DDPG",
            layer_config=sample_layer_config,
            metadata=metadata
        )
        
        # Load and verify config
        config_file = manager.base_save_dir / "config_test" / "config.json"
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config['framework'] == "stable_baselines3"
        assert saved_config['algorithm'] == "DDPG"
        assert saved_config['metadata']['training_steps'] == 100000
        assert saved_config['layer_config']['layer_type'] == "multi"
    
    def test_load_model_with_factory(self, temp_dir, sample_model, sample_layer_config):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        # First save a model
        manager.save_model(
            model=sample_model,
            model_name="load_test",
            framework="stable_baselines3",
            algorithm="DDPG",
            layer_config=sample_layer_config
        )
        
        # Create a model factory
        def model_factory():
            return sample_model
        
        # Load the model
        loaded_model, config = manager.load_model(
            model_name="load_test",
            model_factory=model_factory
        )
        
        assert loaded_model is sample_model
        assert config['framework'] == "stable_baselines3"
        assert config['algorithm'] == "DDPG"
    
    def test_load_nonexistent_model(self, temp_dir):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        def model_factory():
            return MagicMock()
        
        with pytest.raises(FileNotFoundError):
            manager.load_model("nonexistent_model", model_factory)
    
    def test_list_models(self, temp_dir, sample_model, sample_layer_config):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        # Save multiple models
        for i, algorithm in enumerate(["DDPG", "SAC", "TD3"]):
            manager.save_model(
                model=sample_model,
                model_name=f"model_{i}",
                framework="stable_baselines3",
                algorithm=algorithm,
                layer_config=sample_layer_config
            )
        
        models = manager.list_models()
        
        assert len(models) == 3
        model_names = [model['name'] for model in models]
        assert "model_0" in model_names
        assert "model_1" in model_names
        assert "model_2" in model_names
    
    def test_delete_model(self, temp_dir, sample_model, sample_layer_config):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        # Save a model
        manager.save_model(
            model=sample_model,
            model_name="delete_test",
            framework="stable_baselines3",
            algorithm="DDPG",
            layer_config=sample_layer_config
        )
        
        # Verify it exists
        assert (manager.base_save_dir / "delete_test").exists()
        
        # Delete it
        manager.delete_model("delete_test")
        
        # Verify it's gone
        assert not (manager.base_save_dir / "delete_test").exists()
    
    def test_get_model_info(self, temp_dir, sample_model, sample_layer_config):
        manager = RingAttractorModelManager(base_save_dir=temp_dir)
        
        metadata = {"training_steps": 50000, "mean_reward": 123.45}
        
        manager.save_model(
            model=sample_model,
            model_name="info_test",
            framework="stable_baselines3",
            algorithm="SAC",
            layer_config=sample_layer_config,
            metadata=metadata
        )
        
        info = manager.get_model_info("info_test")
        
        assert info['framework'] == "stable_baselines3"
        assert info['algorithm'] == "SAC"
        assert info['metadata']['training_steps'] == 50000
        assert info['layer_config']['layer_type'] == "multi"


class TestModelRegistry:
    """Test ModelRegistry class"""
    
    def test_initialization(self):
        registry = ModelRegistry()
        assert len(registry.factories) == 0
    
    def test_register_factory(self):
        registry = ModelRegistry()
        
        def sample_factory():
            return "test_model"
        
        registry.register_factory(
            name="test_factory",
            factory_fn=sample_factory,
            description="Test factory"
        )
        
        assert "test_factory" in registry.factories
        assert registry.factories["test_factory"]["description"] == "Test factory"
        assert callable(registry.factories["test_factory"]["factory"])
    
    def test_get_factory(self):
        registry = ModelRegistry()
        
        def sample_factory():
            return "created_model"
        
        registry.register_factory("test", sample_factory)
        
        factory = registry.get_factory("test")
        assert factory() == "created_model"
    
    def test_get_nonexistent_factory(self):
        registry = ModelRegistry()
        
        with pytest.raises(KeyError):
            registry.get_factory("nonexistent")
    
    def test_list_factories(self):
        registry = ModelRegistry()
        
        registry.register_factory("factory1", lambda: "model1", "First factory")
        registry.register_factory("factory2", lambda: "model2", "Second factory")
        
        factories = registry.list_factories()
        
        assert len(factories) == 2
        assert "factory1" in factories
        assert "factory2" in factories
        assert factories["factory1"]["description"] == "First factory"
    
    def test_register_duplicate_factory(self):
        registry = ModelRegistry()
        
        registry.register_factory("duplicate", lambda: "model1")
        
        with pytest.warns(UserWarning, match="Factory 'duplicate' already exists"):
            registry.register_factory("duplicate", lambda: "model2")
        
        # Should overwrite the previous factory
        factory = registry.get_factory("duplicate")
        assert factory() == "model2"


class TestStandaloneFunctions:
    """Test standalone save/load functions"""
    
    @pytest.fixture
    def temp_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_save_ring_attractor_model(self, temp_file):
        model = MagicMock()
        model.save = MagicMock()
        
        config = RingAttractorConfig(num_excitatory=12, tau=6.0)
        layer_config = {
            'layer_type': 'single',
            'input_dim': 64,
            'output_dim': 1,
            'config': config
        }
        
        save_ring_attractor_model(
            model=model,
            filepath=temp_file,
            layer_config=layer_config,
            framework="stable_baselines3",
            algorithm="DDPG"
        )
        
        # Verify file was created and contains expected data
        assert os.path.exists(temp_file)
        
        with open(temp_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['framework'] == "stable_baselines3"
        assert saved_data['algorithm'] == "DDPG"
        assert saved_data['layer_config']['layer_type'] == "single"
    
    def test_load_ring_attractor_model(self, temp_file):
        # Create test data
        config_data = {
            'framework': "stable_baselines3",
            'algorithm': "SAC",
            'layer_config': {
                'layer_type': 'multi',
                'input_dim': 128,
                'control_axes': ['roll', 'pitch'],
                'config': RingAttractorConfig(num_excitatory=16).to_dict()
            }
        }
        
        with open(temp_file, 'w') as f:
            json.dump(config_data, f, default=str)
        
        loaded_config = load_ring_attractor_model(temp_file)
        
        assert loaded_config['framework'] == "stable_baselines3"
        assert loaded_config['algorithm'] == "SAC"
        assert loaded_config['layer_config']['layer_type'] == "multi"


class TestModelManagerErrorHandling:
    """Test error handling in ModelManager"""
    
    def test_save_model_invalid_model(self, tmp_path):
        manager = RingAttractorModelManager(base_save_dir=tmp_path)
        
        with pytest.raises(AttributeError):
            manager.save_model(
                model=None,  # Invalid model
                model_name="test",
                framework="stable_baselines3",
                algorithm="DDPG",
                layer_config={}
            )
    
    def test_save_model_permission_error(self, tmp_path):
        manager = RingAttractorModelManager(base_save_dir=tmp_path)
        model = MagicMock()
        
        # Mock os.makedirs to raise PermissionError
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                manager.save_model(
                    model=model,
                    model_name="permission_test",
                    framework="stable_baselines3",
                    algorithm="DDPG",
                    layer_config={}
                )
    
    def test_load_model_corrupted_config(self, tmp_path):
        manager = RingAttractorModelManager(base_save_dir=tmp_path)
        
        # Create model directory with corrupted config
        model_dir = tmp_path / "corrupted_model"
        model_dir.mkdir()
        
        config_file = model_dir / "config.json"
        with open(config_file, 'w') as f:
            f.write("{invalid json")  # Corrupted JSON
        
        def factory():
            return MagicMock()
        
        with pytest.raises(json.JSONDecodeError):
            manager.load_model("corrupted_model", factory)


class TestModelManagerIntegration:
    """Integration tests for ModelManager"""
    
    def test_full_save_load_cycle(self, tmp_path):
        manager = RingAttractorModelManager(base_save_dir=tmp_path)
        
        # Create mock model with realistic attributes
        model = MagicMock()
        model.save = MagicMock()
        model.policy.state_dict = MagicMock(return_value={
            'features_extractor.weight': torch.randn(64, 32),
            'mlp.0.weight': torch.randn(128, 64)
        })
        
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 128,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': RingAttractorConfig(
                num_excitatory=20,
                tau=10.0,
                beta=8.0,
                trainable_structure=True
            )
        }
        
        metadata = {
            'training_steps': 200000,
            'environment': 'PyFlyt/QuadX-Hover-v2',
            'performance': {
                'mean_reward': 567.8,
                'std_reward': 23.4,
                'success_rate': 0.87
            }
        }
        
        # Save model
        manager.save_model(
            model=model,
            model_name="integration_test",
            framework="stable_baselines3",
            algorithm="SAC",
            layer_config=layer_config,
            metadata=metadata,
            save_policy_only=True
        )
        
        # Verify all files exist
        model_dir = manager.base_save_dir / "integration_test"
        assert model_dir.exists()
        assert (model_dir / "config.json").exists()
        assert (model_dir / "policy_weights.pt").exists()
        
        # Load model
        def model_factory():
            return model
        
        loaded_model, config = manager.load_model(
            "integration_test",
            model_factory,
            device="cpu"
        )
        
        # Verify loaded data
        assert loaded_model is model
        assert config['algorithm'] == "SAC"
        assert config['metadata']['training_steps'] == 200000
        assert config['layer_config']['config']['num_excitatory'] == 20
        
        # Test model info
        info = manager.get_model_info("integration_test")
        assert info['framework'] == "stable_baselines3"
        assert info['metadata']['performance']['success_rate'] == 0.87
    
    def test_multiple_models_management(self, tmp_path):
        manager = RingAttractorModelManager(base_save_dir=tmp_path)
        
        # Create and save multiple models
        configs = [
            ("model_ddpg", "DDPG", {'layer_type': 'single'}),
            ("model_sac", "SAC", {'layer_type': 'multi'}),
            ("model_td3", "TD3", {'layer_type': 'coupled'})
        ]
        
        for name, algorithm, layer_config in configs:
            model = MagicMock()
            model.save = MagicMock()
            
            manager.save_model(
                model=model,
                model_name=name,
                framework="stable_baselines3",
                algorithm=algorithm,
                layer_config=layer_config
            )
        
        # List all models
        models = manager.list_models()
        assert len(models) == 3
        
        model_names = {model['name'] for model in models}
        assert model_names == {"model_ddpg", "model_sac", "model_td3"}
        
        # Test individual model info
        ddpg_info = manager.get_model_info("model_ddpg")
        assert ddpg_info['algorithm'] == "DDPG"
        assert ddpg_info['layer_config']['layer_type'] == "single"
        
        # Delete one model
        manager.delete_model("model_sac")
        
        # Verify it's gone
        remaining_models = manager.list_models()
        assert len(remaining_models) == 2
        remaining_names = {model['name'] for model in remaining_models}
        assert remaining_names == {"model_ddpg", "model_td3"}