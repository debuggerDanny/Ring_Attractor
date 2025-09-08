"""
Policy wrappers for integrating Ring Attractor layers with different RL algorithms.

This module provides a modular and extensible way to integrate Ring Attractor
architectures with various reinforcement learning frameworks without tight coupling.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from copy import deepcopy
import logging

from  src.utils.control_layers import (
    BaseControlLayer, 
    create_control_layer, 
    RingAttractorConfig,
    get_quadrotor_config
)

logger = logging.getLogger(__name__)


class PolicyWrapper(ABC):
    """
    Abstract base class for policy wrappers.
    
    This class defines the interface for wrapping different RL algorithm policies
    with Ring Attractor layers, ensuring modularity and extensibility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def wrap_policy(self, model: Any, layer_config: Dict[str, Any]) -> Any:
        """
        Wrap a policy with Ring Attractor layers.
        
        Args:
            model: The RL model to wrap
            layer_config: Configuration for the Ring Attractor layer
            
        Returns:
            The wrapped model
        """
        pass
    
    @abstractmethod
    def extract_policy_layers(self, model: Any) -> List[nn.Module]:
        """
        Extract the policy layers from a model.
        
        Args:
            model: The RL model
            
        Returns:
            List of policy layers
        """
        pass
    
    def validate_layer_config(self, layer_config: Dict[str, Any]) -> None:
        """Validate the layer configuration."""
        required_keys = ['layer_type', 'input_dim', 'control_axes']
        for key in required_keys:
            if key not in layer_config:
                raise ValueError(f"Missing required key in layer_config: {key}")


class StableBaselines3Wrapper(PolicyWrapper):
    """
    Wrapper for Stable Baselines3 algorithms (DDPG, SAC, TD3, etc.).
    
    This wrapper provides a unified interface for integrating Ring Attractor
    layers with different SB3 algorithms without algorithm-specific code.
    """
    
    SUPPORTED_ALGORITHMS = ['DDPG', 'SAC', 'TD3', 'A2C', 'PPO']
    
    def __init__(self, algorithm: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Supported: {self.SUPPORTED_ALGORITHMS}")
        self.algorithm = algorithm
    
    def wrap_policy(self, model: Any, layer_config: Dict[str, Any]) -> Any:
        """
        Wrap SB3 policy with Ring Attractor layers.
        
        Args:
            model: SB3 model instance
            layer_config: Ring Attractor layer configuration
            
        Returns:
            Modified model with Ring Attractor layers
        """
        self.validate_layer_config(layer_config)
        
        # Algorithm-specific wrapping
        if self.algorithm in ['DDPG', 'TD3']:
            return self._wrap_deterministic_policy(model, layer_config)
        elif self.algorithm == 'SAC':
            return self._wrap_sac_policy(model, layer_config)
        elif self.algorithm in ['A2C', 'PPO']:
            return self._wrap_actor_critic_policy(model, layer_config)
        else:
            raise NotImplementedError(f"Wrapping not implemented for {self.algorithm}")
    
    def _wrap_deterministic_policy(self, model: Any, layer_config: Dict[str, Any]) -> Any:
        """Wrap deterministic policies (DDPG, TD3)."""
        # Extract existing layers (all except last few)
        existing_layers = self.extract_policy_layers(model.actor.mu)[:-2]
        
        # Create Ring Attractor layer
        ring_layer = create_control_layer(**layer_config)
        
        # Reconstruct actor network
        model.actor.mu = nn.Sequential(*existing_layers, ring_layer)
        model.actor.mu = model.actor.mu.to(model.device)
        
        # Update optimizers
        self._update_optimizers(model, 'actor')
        
        # Update target network (for DDPG/TD3)
        model.actor_target.mu = deepcopy(model.actor.mu)
        model.actor_target.mu = model.actor_target.mu.to(model.device)
        
        logger.info(f"Wrapped {self.algorithm} deterministic policy with Ring Attractor")
        return model
    
    def _wrap_sac_policy(self, model: Any, layer_config: Dict[str, Any]) -> Any:
        """Wrap SAC policy (stochastic)."""
        # For SAC, we need to modify both mu and log_std networks
        existing_layers = self.extract_policy_layers(model.actor.mu)[:-2]
        
        # we need the output of the ring layer to be the same 
        # as the input of the mu layer
        
        # Check if layer_config exists and is mutable
        print("Before modification:")
        print(f"layer_config type: {type(layer_config)}")
        print(f"layer_config: {layer_config}")

        output_shape = model.actor.mu.in_features
        print(f"output_shape: {output_shape}")

        layer_config["output_dim"] = output_shape

        print("After modification:")
        print(f"layer_config: {layer_config}")
        print(f"'output_dim' in layer_config: {'output_dim' in layer_config}")

        # Create Ring Attractor layer
        ring_layer = create_control_layer(**layer_config)
        
        # Modify the latent network (shared between mu and log_std)
        if hasattr(model.actor, 'latent_pi'):
            # Modify shared latent network
            existing_latent_layers = list(model.actor.latent_pi.children())[:-2]
            model.actor.latent_pi = nn.Sequential(*existing_latent_layers, ring_layer)

            # TODO: make the output of ring be the same size as the mu layer
        else:
            # Modify mu network directly
            logger.info("Modified the MU layerr instead of the latent_pi")
            model.actor.mu = nn.Sequential(*existing_layers, ring_layer)
        
        model.actor = model.actor.to(model.device)
        
        # Update optimizer
        self._update_optimizers(model, 'actor')
        
        logger.info("Wrapped SAC policy with Ring Attractor")
        return model
    
    def _wrap_actor_critic_policy(self, model: Any, layer_config: Dict[str, Any]) -> Any:
        """Wrap actor-critic policies (A2C, PPO)."""
        # For actor-critic, modify the policy network
        existing_layers = self.extract_policy_layers(model.policy.action_net)[:-2]
        
        # Create Ring Attractor layer
        ring_layer = create_control_layer(**layer_config)
        
        # Reconstruct action network
        model.policy.action_net = nn.Sequential(*existing_layers, ring_layer)
        model.policy = model.policy.to(model.device)
        
        # Update optimizer
        self._update_optimizers(model, 'policy')
        
        logger.info(f"Wrapped {self.algorithm} actor-critic policy with Ring Attractor")
        return model
    
    def _update_optimizers(self, model: Any, component: str) -> None:
        """Update optimizers with new parameters."""
        if component == 'actor':
            model.actor.optimizer = model.actor.optimizer_class(
                model.actor.parameters(),
                **model.actor.optimizer_kwargs
            )
            if hasattr(model, 'actor_target'):
                model.actor_target.optimizer = model.actor_target.optimizer_class(
                    model.actor_target.parameters(),
                    **model.actor_target.optimizer_kwargs
                )
        elif component == 'policy':
            model.policy.optimizer = model.policy.optimizer_class(
                model.policy.parameters(),
                lr=model.learning_rate,
                **model.policy.optimizer_kwargs
            )
    
    def extract_policy_layers(self, policy_net: nn.Module) -> List[nn.Module]:
        """Extract layers from SB3 policy network."""
        return list(policy_net.children())


class TorchRLWrapper(PolicyWrapper):
    """
    Wrapper for TorchRL algorithms.
    
    This wrapper provides integration with TorchRL's modular RL framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
    
    def wrap_policy(self, model: Any, layer_config: Dict[str, Any]) -> Any:
        """Wrap TorchRL policy with Ring Attractor layers."""
        self.validate_layer_config(layer_config)
        
        # TorchRL uses modular networks - modify the actor module
        if hasattr(model, 'actor_network'):
            existing_layers = self.extract_policy_layers(model.actor_network)[:-1]
            ring_layer = create_control_layer(**layer_config)
            model.actor_network = nn.Sequential(*existing_layers, ring_layer)
        
        logger.info("Wrapped TorchRL policy with Ring Attractor")
        return model
    
    def extract_policy_layers(self, policy_net: nn.Module) -> List[nn.Module]:
        """Extract layers from TorchRL policy network."""
        return list(policy_net.children())


class CustomPolicyWrapper(PolicyWrapper):
    """
    Wrapper for custom policy implementations.
    
    This wrapper allows users to define their own policy modification logic
    through callback functions.
    """
    
    def __init__(
        self, 
        wrap_fn: Callable[[Any, Dict[str, Any]], Any],
        extract_fn: Callable[[Any], List[nn.Module]],
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.wrap_fn = wrap_fn
        self.extract_fn = extract_fn
    
    def wrap_policy(self, model: Any, layer_config: Dict[str, Any]) -> Any:
        """Wrap policy using custom function."""
        self.validate_layer_config(layer_config)
        return self.wrap_fn(model, layer_config)
    
    def extract_policy_layers(self, policy_net: nn.Module) -> List[nn.Module]:
        """Extract layers using custom function."""
        return self.extract_fn(policy_net)


class PolicyWrapperFactory:
    """
    Factory for creating policy wrappers.
    
    This factory provides a centralized way to create policy wrappers
    for different RL frameworks and algorithms.
    """
    
    _WRAPPERS = {
        'stable_baselines3': StableBaselines3Wrapper,
        'sb3': StableBaselines3Wrapper,  # Alias
        'torchrl': TorchRLWrapper,
        'custom': CustomPolicyWrapper
    }
    
    @classmethod
    def create_wrapper(
        self,
        framework: str,
        algorithm: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PolicyWrapper:
        """
        Create a policy wrapper for the specified framework.
        
        Args:
            framework (str): RL framework name ('stable_baselines3', 'torchrl', 'custom')
            algorithm (str): Algorithm name (required for some frameworks)
            config (Dict): Additional configuration
            **kwargs: Additional arguments for wrapper initialization
            
        Returns:
            PolicyWrapper: The created wrapper instance
        """
        if framework not in self._WRAPPERS:
            raise ValueError(f"Unsupported framework: {framework}. "
                           f"Available: {list(self._WRAPPERS.keys())}")
        
        wrapper_class = self._WRAPPERS[framework]
        
        if framework in ['stable_baselines3', 'sb3']:
            if algorithm is None:
                raise ValueError("Algorithm must be specified for Stable Baselines3")
            return wrapper_class(algorithm, config)
        elif framework == 'custom':
            return wrapper_class(config=config, **kwargs)
        else:
            return wrapper_class(config)
    
    @classmethod
    def register_wrapper(cls, name: str, wrapper_class: type) -> None:
        """Register a new wrapper class."""
        cls._WRAPPERS[name] = wrapper_class


def create_ring_attractor_model(
    base_model: Any,
    framework: str,
    algorithm: str,
    layer_config: Optional[Dict[str, Any]] = None,
    preset_config: Optional[str] = None,
    wrapper_config: Optional[Dict[str, Any]] = None,
    device: str = "cpu"
) -> Any:
    """
    High-level function to create a Ring Attractor model from any base RL model.
    
    This function provides a simple interface for wrapping existing RL models
    with Ring Attractor layers, regardless of the underlying framework.
    
    Args:
        base_model: The base RL model to wrap
        framework (str): RL framework ('stable_baselines3', 'torchrl', etc.)
        algorithm (str): Algorithm name
        layer_config (Dict): Custom layer configuration
        preset_config (str): Use preset configuration ('quadrotor', 'drone', 'arm')
        wrapper_config (Dict): Additional wrapper configuration
        device (str): Device for computation
        
    Returns:
        Wrapped model with Ring Attractor layers
        
    Example:
        >>> import gym
        >>> from stable_baselines3 import DDPG
        >>> env = gym.make("Pendulum-v1")
        >>> base_model = DDPG("MlpPolicy", env)
        >>> ra_model = create_ring_attractor_model(
        ...     base_model=base_model,
        ...     framework="stable_baselines3",
        ...     algorithm="DDPG",
        ...     preset_config="quadrotor"
        ... )
    """
    # Use preset configuration if specified
    if preset_config:
        preset_configs = {
            'quadrotor': get_quadrotor_config(),
            # 'drone': get_drone_navigation_config(),
            # 'arm': get_robotic_arm_config()
        }
        if preset_config not in preset_configs:
            raise ValueError(f"Unknown preset config: {preset_config}")
        layer_config = preset_configs[preset_config]
    
    # Validate layer configuration
    if layer_config is None:
        raise ValueError("Either layer_config or preset_config must be provided")
    
    # Create wrapper
    wrapper = PolicyWrapperFactory.create_wrapper(
        framework=framework,
        algorithm=algorithm,
        config=wrapper_config
    )
    
    # Wrap the model
    wrapped_model = wrapper.wrap_policy(base_model, layer_config)
    # print(wrapped_model)
    # wrapped_model = wrapped_model.to(device)
    
    logger.info(f"Created Ring Attractor model for {framework}/{algorithm} on {device}")
    return wrapped_model


# Convenience functions for popular combinations
def create_ddpg_ring_attractor(
    base_model,
    layer_config: Optional[Dict[str, Any]] = None,
    preset_config: Optional[str] = None,
    device: str = "cpu"
):
    """Create DDPG model with Ring Attractor layers."""
    return create_ring_attractor_model(
        base_model=base_model,
        framework="stable_baselines3",
        algorithm="DDPG",
        layer_config=layer_config,
        preset_config=preset_config,
        device=device
    )


def create_sac_ring_attractor(
    base_model,
    layer_config: Optional[Dict[str, Any]] = None,
    preset_config: Optional[str] = None,
    device: str = "cpu"
):
    """Create SAC model with Ring Attractor layers."""
    return create_ring_attractor_model(
        base_model=base_model,
        framework="stable_baselines3",
        algorithm="SAC",
        layer_config=layer_config,
        preset_config=preset_config,
        device=device
    )


def create_td3_ring_attractor(
    base_model,
    layer_config: Optional[Dict[str, Any]] = None,
    preset_config: Optional[str] = None,
    device: str = "cpu"
):
    """Create TD3 model with Ring Attractor layers."""
    return create_ring_attractor_model(
        base_model=base_model,
        framework="stable_baselines3",
        algorithm="TD3",
        layer_config=layer_config,
        preset_config=preset_config,
        device=device
    )