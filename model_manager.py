"""
Model management utilities for Ring Attractor models.

This module provides utilities for saving, loading, and managing Ring Attractor
models across different RL frameworks.
"""

import torch
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class RingAttractorModelManager:
    """
    Manager for Ring Attractor model persistence and loading.
    
    This class handles saving and loading of Ring Attractor models with their
    configurations, ensuring compatibility across different sessions and environments.
    """
    
    def __init__(self, base_save_dir: Union[str, Path] = "./models"):
        """
        Initialize model manager.
        
        Args:
            base_save_dir: Base directory for saving models
        """
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        framework: str,
        algorithm: str,
        layer_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        save_policy_only: bool = False
    ) -> Path:
        """
        Save a Ring Attractor model with its configuration.
        
        Args:
            model: The model to save
            model_name: Name for the saved model
            framework: RL framework name
            algorithm: Algorithm name
            layer_config: Ring Attractor layer configuration
            metadata: Additional metadata to save
            save_policy_only: If True, save only policy weights
            
        Returns:
            Path to saved model directory
        """
        model_dir = self.base_save_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model configuration
        config = {
            'framework': framework,
            'algorithm': algorithm,
            'layer_config': layer_config,
            'metadata': metadata or {},
            'model_name': model_name,
            'save_policy_only': save_policy_only
        }
        
        config_path = model_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save model weights
        if framework == 'stable_baselines3':
            self._save_sb3_model(model, model_dir, save_policy_only)
        elif framework == 'torchrl':
            self._save_torchrl_model(model, model_dir, save_policy_only)
        else:
            # Generic PyTorch model saving
            model_path = model_dir / 'model_weights.pth'
            torch.save(model.state_dict(), model_path)
        
        logger.info(f"Saved Ring Attractor model '{model_name}' to {model_dir}")
        return model_dir
    
    def _save_sb3_model(
        self, 
        model: Any, 
        model_dir: Path, 
        save_policy_only: bool
    ) -> None:
        """Save Stable Baselines3 model."""
        if save_policy_only:
            # Save only policy components
            policy_weights = {}
            
            # Actor weights
            if hasattr(model, 'actor'):
                policy_weights.update({
                    f'actor.{k}': v for k, v in model.actor.state_dict().items()
                })
            
            # Actor target weights (for off-policy algorithms)
            if hasattr(model, 'actor_target'):
                policy_weights.update({
                    f'actor_target.{k}': v for k, v in model.actor_target.state_dict().items()
                })
            
            # Policy weights (for on-policy algorithms)
            if hasattr(model, 'policy'):
                policy_weights.update({
                    f'policy.{k}': v for k, v in model.policy.state_dict().items()
                })
            
            weights_path = model_dir / 'policy_weights.pth'
            torch.save({'state_dict': policy_weights}, weights_path)
        else:
            # Save full model
            model_path = model_dir / 'full_model.zip'
            model.save(str(model_path))
    
    def _save_torchrl_model(
        self, 
        model: Any, 
        model_dir: Path, 
        save_policy_only: bool
    ) -> None:
        """Save TorchRL model."""
        if save_policy_only:
            # Extract policy components
            policy_state = {}
            if hasattr(model, 'actor_network'):
                policy_state['actor_network'] = model.actor_network.state_dict()
            
            weights_path = model_dir / 'policy_weights.pth'
            torch.save(policy_state, weights_path)
        else:
            # Save full model state
            model_path = model_dir / 'full_model.pth'
            torch.save(model.state_dict(), model_path)
    
    def load_model(
        self,
        model_name: str,
        model_factory: Callable[[], Any],
        device: str = "cpu"
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Load a Ring Attractor model.
        
        Args:
            model_name: Name of the model to load
            model_factory: Function that creates the base model architecture
            device: Device to load the model on
            
        Returns:
            Tuple of (loaded_model, config)
        """
        model_dir = self.base_save_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found in {self.base_save_dir}")
        
        # Load configuration
        config_path = model_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create base model
        model = model_factory()
        
        # Load weights
        framework = config['framework']
        save_policy_only = config.get('save_policy_only', False)
        
        if framework == 'stable_baselines3':
            model = self._load_sb3_model(model, model_dir, save_policy_only, device)
        elif framework == 'torchrl':
            model = self._load_torchrl_model(model, model_dir, save_policy_only, device)
        else:
            # Generic PyTorch loading
            weights_path = model_dir / 'model_weights.pth'
            model.load_state_dict(torch.load(weights_path, map_location=device))
        
        model = model.to(device)
        
        logger.info(f"Loaded Ring Attractor model '{model_name}' on {device}")
        return model, config
    
    def _load_sb3_model(
        self, 
        model: Any, 
        model_dir: Path, 
        save_policy_only: bool, 
        device: str
    ) -> Any:
        """Load Stable Baselines3 model."""
        if save_policy_only:
            weights_path = model_dir / 'policy_weights.pth'
            saved_weights = torch.load(weights_path, map_location=device)
            
            # Separate weights by component
            component_state_dicts = {
                "actor": OrderedDict(),
                "actor_target": OrderedDict(),
                "policy": OrderedDict()
            }
            
            for key, value in saved_weights['state_dict'].items():
                if key.startswith("actor."):
                    new_key = key.replace("actor.", "")
                    component_state_dicts["actor"][new_key] = value
                elif key.startswith("actor_target."):
                    new_key = key.replace("actor_target.", "")
                    component_state_dicts["actor_target"][new_key] = value
                elif key.startswith("policy."):
                    new_key = key.replace("policy.", "")
                    component_state_dicts["policy"][new_key] = value
            
            # Load weights into appropriate components
            if hasattr(model, 'actor') and component_state_dicts["actor"]:
                model.actor.load_state_dict(component_state_dicts["actor"])
            if hasattr(model, 'actor_target') and component_state_dicts["actor_target"]:
                model.actor_target.load_state_dict(component_state_dicts["actor_target"])
            if hasattr(model, 'policy') and component_state_dicts["policy"]:
                model.policy.load_state_dict(component_state_dicts["policy"])
        else:
            # Load full model
            model_path = model_dir / 'full_model.zip'
            model = model.load(str(model_path), device=device)
        
        return model
    
    def _load_torchrl_model(
        self, 
        model: Any, 
        model_dir: Path, 
        save_policy_only: bool, 
        device: str
    ) -> Any:
        """Load TorchRL model."""
        if save_policy_only:
            weights_path = model_dir / 'policy_weights.pth'
            policy_state = torch.load(weights_path, map_location=device)
            
            if 'actor_network' in policy_state and hasattr(model, 'actor_network'):
                model.actor_network.load_state_dict(policy_state['actor_network'])
        else:
            model_path = model_dir / 'full_model.pth'
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        return model
    
    def list_models(self) -> list[str]:
        """List all saved models."""
        if not self.base_save_dir.exists():
            return []
        
        models = []
        for item in self.base_save_dir.iterdir():
            if item.is_dir() and (item / 'config.json').exists():
                models.append(item.name)
        
        return sorted(models)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a saved model."""
        model_dir = self.base_save_dir / model_name
        config_path = model_dir / 'config.json'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add file size information
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        config['total_size_mb'] = total_size / (1024 * 1024)
        
        return config
    
    def delete_model(self, model_name: str) -> None:
        """Delete a saved model."""
        model_dir = self.base_save_dir / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found")
        
        import shutil
        shutil.rmtree(model_dir)
        
        logger.info(f"Deleted model '{model_name}'")


class ModelRegistry:
    """
    Registry for model factory functions.
    
    This class maintains a registry of factory functions that can create
    base model architectures for different RL algorithms and frameworks.
    """
    
    def __init__(self):
        self._factories = {}
    
    def register_factory(
        self, 
        name: str, 
        factory_fn: Callable[[], Any],
        description: str = ""
    ) -> None:
        """
        Register a model factory function.
        
        Args:
            name: Name for the factory
            factory_fn: Function that creates a model
            description: Optional description
        """
        self._factories[name] = {
            'factory': factory_fn,
            'description': description
        }
    
    def get_factory(self, name: str) -> Callable[[], Any]:
        """Get a registered factory function."""
        if name not in self._factories:
            raise KeyError(f"Factory '{name}' not registered. Available: {list(self._factories.keys())}")
        
        return self._factories[name]['factory']
    
    def list_factories(self) -> Dict[str, str]:
        """List all registered factories with their descriptions."""
        return {
            name: info['description'] 
            for name, info in self._factories.items()
        }
    
    def remove_factory(self, name: str) -> None:
        """Remove a factory from the registry."""
        if name in self._factories:
            del self._factories[name]
            logger.info(f"Removed factory '{name}' from registry")
        else:
            logger.warning(f"Factory '{name}' not found in registry")


# Global registry instance
model_registry = ModelRegistry()


def register_ring_attractor_factories():
    """Register standard Ring Attractor model factories."""
    from ..utils.control_layers import (
        SingleAxisRingAttractorLayer, 
        MultiAxisRingAttractorLayer,
        CoupledRingAttractorLayer,
        AdaptiveRingAttractorLayer,
        get_quadrotor_config,
        get_drone_navigation_config
    )
    from ..utils.attractors import RingAttractorConfig
    
    # Single-axis control factory
    def create_single_axis_model():
        config = RingAttractorConfig()
        return SingleAxisRingAttractorLayer(
            input_dim=64,
            output_dim=4,
            config=config
        )
    
    model_registry.register_factory(
        'single_axis_control',
        create_single_axis_model,
        'Single-axis Ring Attractor for basic control tasks'
    )
    
    # Multi-axis control factory
    def create_multi_axis_model():
        quad_config = get_quadrotor_config()
        return MultiAxisRingAttractorLayer(
            input_dim=64,
            control_axes=quad_config['control_axes'],
            config=quad_config['config'],
            ring_axes=quad_config['ring_axes']
        )
    
    model_registry.register_factory(
        'multi_axis_control',
        create_multi_axis_model,
        'Multi-axis Ring Attractor for quadrotor control'
    )
    
    # Coupled control factory
    def create_coupled_model():
        quad_config = get_quadrotor_config()
        return CoupledRingAttractorLayer(
            input_dim=64,
            control_axes=quad_config['control_axes'],
            num_rings=3,
            config=quad_config['config'],
            coupled_axes=quad_config['coupled_axes']
        )
    
    model_registry.register_factory(
        'coupled_control',
        create_coupled_model,
        'Coupled Ring Attractor for integrated quadrotor control'
    )
    
    # Navigation control factory
    def create_navigation_model():
        nav_config = get_drone_navigation_config()
        return AdaptiveRingAttractorLayer(
            input_dim=64,
            control_axes=nav_config['control_axes'],
            architecture_type='coupled',
            config=nav_config['config'],
            coupled_axes=nav_config['coupled_axes']
        )
    
    model_registry.register_factory(
        'navigation_control',
        create_navigation_model,
        'Adaptive Ring Attractor for drone navigation'
    )


def create_ddpg_factory(
    layer_type: str = 'multi',
    env_name: str = 'quadrotor',
    policy_kwargs: Optional[Dict[str, Any]] = None
) -> Callable[[], Any]:
    """
    Create a factory function for DDPG models with Ring Attractor layers.
    
    Args:
        layer_type: Type of Ring Attractor layer ('single', 'multi', 'coupled', 'adaptive')
        env_name: Environment name for configuration selection
        policy_kwargs: Additional policy arguments
        
    Returns:
        Factory function that creates DDPG models
    """
    def ddpg_factory():
        try:
            from stable_baselines3 import DDPG
            from stable_baselines3.common.policies import BasePolicy
            from ..utils.control_layers import create_control_layer, get_quadrotor_config
            
            # Select configuration based on environment
            if env_name == 'quadrotor':
                config_dict = get_quadrotor_config()
            else:
                # Default configuration
                config_dict = {
                    'control_axes': ['action_0', 'action_1', 'action_2', 'action_3'],
                    'config': RingAttractorConfig()
                }
            
            # Create custom policy with Ring Attractor layers
            class RingAttractorPolicy(BasePolicy):
                def __init__(self, *args, **kwargs):
                    super(RingAttractorPolicy, self).__init__(*args, **kwargs)
                    
                    # Create Ring Attractor control layer
                    self.ring_layer = create_control_layer(
                        layer_type=layer_type,
                        input_dim=64,  # Typical hidden layer size
                        control_axes=config_dict['control_axes'],
                        config=config_dict['config'],
                        **{k: v for k, v in config_dict.items() if k not in ['control_axes', 'config']}
                    )
                
                def forward(self, obs, deterministic=False):
                    # This would be implemented based on the specific policy structure
                    # For now, return a placeholder
                    return self.ring_layer(obs), None
            
            # Default policy arguments
            default_policy_kwargs = {
                'net_arch': [64, 64],
                'activation_fn': torch.nn.ReLU
            }
            
            if policy_kwargs:
                default_policy_kwargs.update(policy_kwargs)
            
            # Create DDPG model (without environment for factory pattern)
            model = DDPG(
                policy='MlpPolicy',  # This would be replaced with custom policy
                env=None,  # Environment will be set when loading
                policy_kwargs=default_policy_kwargs,
                verbose=1
            )
            
            return model
            
        except ImportError:
            logger.warning("Stable Baselines3 not available, creating placeholder model")
            return None
    
    return ddpg_factory


def register_rl_factories():
    """Register reinforcement learning model factories."""
    
    # DDPG with different Ring Attractor configurations
    for layer_type in ['single', 'multi', 'coupled', 'adaptive']:
        factory_name = f'ddpg_{layer_type}_ring'
        factory_fn = create_ddpg_factory(layer_type=layer_type)
        
        model_registry.register_factory(
            factory_name,
            factory_fn,
            f'DDPG with {layer_type} Ring Attractor layer'
        )


def get_model_manager(base_save_dir: Union[str, Path] = "./models") -> RingAttractorModelManager:
    """
    Get a configured model manager instance.
    
    Args:
        base_save_dir: Directory for saving models
        
    Returns:
        Configured RingAttractorModelManager instance
    """
    manager = RingAttractorModelManager(base_save_dir)
    
    # Register standard factories if not already done
    if not model_registry._factories:
        register_ring_attractor_factories()
        register_rl_factories()
        logger.info("Registered standard Ring Attractor model factories")
    
    return manager


# Convenience functions for common operations
def save_ring_attractor_model(
    model: Any,
    model_name: str,
    framework: str = 'pytorch',
    algorithm: str = 'ddpg',
    layer_config: Optional[Dict[str, Any]] = None,
    save_dir: Union[str, Path] = "./models",
    **kwargs
) -> Path:
    """
    Convenience function to save a Ring Attractor model.
    
    Args:
        model: Model to save
        model_name: Name for the saved model
        framework: Framework name
        algorithm: Algorithm name
        layer_config: Ring Attractor layer configuration
        save_dir: Save directory
        **kwargs: Additional arguments for model manager
        
    Returns:
        Path to saved model
    """
    manager = get_model_manager(save_dir)
    
    if layer_config is None:
        layer_config = get_quadrotor_config()
    
    return manager.save_model(
        model=model,
        model_name=model_name,
        framework=framework,
        algorithm=algorithm,
        layer_config=layer_config,
        **kwargs
    )


def load_ring_attractor_model(
    model_name: str,
    factory_name: str,
    save_dir: Union[str, Path] = "./models",
    device: str = "cpu"
) -> tuple[Any, Dict[str, Any]]:
    """
    Convenience function to load a Ring Attractor model.
    
    Args:
        model_name: Name of saved model
        factory_name: Name of factory function to use
        save_dir: Save directory
        device: Device to load on
        
    Returns:
        Tuple of (loaded_model, config)
    """
    manager = get_model_manager(save_dir)
    
    if factory_name not in model_registry._factories:
        raise KeyError(f"Factory '{factory_name}' not registered")
    
    factory_fn = model_registry.get_factory(factory_name)
    
    return manager.load_model(
        model_name=model_name,
        model_factory=factory_fn,
        device=device
    )