"""
Layered Neural Architecture with Ring Attractors

This module provides a flexible system for creating neural networks with 
Ring Attractor layers interspersed between regular neural network layers.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from abc import ABC, abstractmethod

from .attractors import RingAttractor, MultiRingAttractor, RingAttractorConfig
from .control_layers import MultiAxisRingAttractorLayer

logger = logging.getLogger(__name__)


class LayerSpec:
    """Specification for a single layer in the network."""
    
    def __init__(
        self,
        layer_type: str,
        **kwargs
    ):
        """
        Initialize layer specification.
        
        Args:
            layer_type: Type of layer ('linear', 'ring', 'multi_ring', 'activation', 'dropout')
            **kwargs: Layer-specific parameters
        """
        self.layer_type = layer_type
        self.params = kwargs
    
    def __repr__(self):
        return f"LayerSpec(type={self.layer_type}, params={self.params})"


class LayeredNetwork(nn.Module):
    """
    Flexible neural network with Ring Attractor layers interspersed between regular layers.
    
    Allows specification of arbitrary layer sequences like:
    Linear -> RingAttractor -> Linear -> Activation -> RingAttractor -> Linear
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_specs: List[LayerSpec],
        name: str = "LayeredNetwork"
    ):
        """
        Initialize layered network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension  
            layer_specs: List of layer specifications
            name: Network name for logging
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_specs = layer_specs
        self.name = name
        
        # Build the network
        self.layers = nn.ModuleList()
        self._build_network()
        
        logger.info(f"Created {name} with {len(self.layers)} layers")
    
    def _build_network(self):
        """Build the network from layer specifications."""
        current_dim = self.input_dim
        
        for i, spec in enumerate(self.layer_specs):
            layer = self._create_layer(spec, current_dim, i)
            
            if layer is not None:
                self.layers.append(layer)
                # Update current dimension for next layer
                current_dim = self._get_output_dim(layer, current_dim)
        
        # Add final output layer if needed
        if current_dim != self.output_dim:
            final_layer = nn.Linear(current_dim, self.output_dim)
            self.layers.append(final_layer)
            logger.debug(f"Added final linear layer: {current_dim} -> {self.output_dim}")
    
    def _create_layer(self, spec: LayerSpec, input_dim: int, layer_idx: int) -> Optional[nn.Module]:
        """Create a layer from specification."""
        layer_type = spec.layer_type.lower()
        params = spec.params.copy()
        
        if layer_type == 'linear':
            output_dim = params.get('output_dim', input_dim)
            layer = nn.Linear(input_dim, output_dim)
            logger.debug(f"Layer {layer_idx}: Linear {input_dim} -> {output_dim}")
            
        elif layer_type == 'ring':
            # Single Ring Attractor
            config = params.get('config', RingAttractorConfig())
            num_excitatory = params.get('num_excitatory', config.num_excitatory)
            
            layer = RingAttractor(
                input_dim=input_dim,
                num_excitatory=num_excitatory,
                tau=config.tau,
                beta=config.beta,
                lambda_decay=config.lambda_decay,
                trainable_structure=config.trainable_structure,
                scale=config.scale
            )
            logger.debug(f"Layer {layer_idx}: RingAttractor {input_dim} -> {num_excitatory}")
            
        elif layer_type == 'multi_ring':
            # Multi Ring Attractor
            config = params.get('config', RingAttractorConfig())
            output_size = params.get('output_size', 16)
            num_rings = params.get('num_rings', 3)
            
            layer = MultiRingAttractor(
                input_size=input_dim // num_rings,  # Divide input among rings
                output_size=output_size,
                num_rings=num_rings,
                trainable_structure=config.trainable_structure,
                connectivity_strength=config.connectivity_strength,
                tau=config.tau,
                beta=config.beta,
                cross_coupling_factor=config.cross_coupling_factor,
                lambda_decay=config.lambda_decay
            )
            logger.debug(f"Layer {layer_idx}: MultiRingAttractor {input_dim} -> {output_size * num_rings}")
            
        elif layer_type == 'control_ring':
            # Control-specific Ring Attractor layer
            control_axes = params.get('control_axes', ['x', 'y', 'z', 'w'])
            ring_axes = params.get('ring_axes', control_axes[:-1])
            config = params.get('config', RingAttractorConfig())
            
            layer = MultiAxisRingAttractorLayer(
                input_dim=input_dim,
                control_axes=control_axes,
                ring_axes=ring_axes,
                config=config
            )
            logger.debug(f"Layer {layer_idx}: ControlRingAttractor {input_dim} -> {len(control_axes)}")
            
        elif layer_type == 'activation':
            # Activation function
            activation_type = params.get('type', 'relu').lower()
            if activation_type == 'relu':
                layer = nn.ReLU()
            elif activation_type == 'tanh':
                layer = nn.Tanh()
            elif activation_type == 'sigmoid':
                layer = nn.Sigmoid()
            elif activation_type == 'leaky_relu':
                layer = nn.LeakyReLU(params.get('negative_slope', 0.01))
            elif activation_type == 'gelu':
                layer = nn.GELU()
            else:
                logger.warning(f"Unknown activation type: {activation_type}, using ReLU")
                layer = nn.ReLU()
            logger.debug(f"Layer {layer_idx}: {activation_type.upper()}")
            
        # should never need these alst ones for RL
        elif layer_type == 'dropout':
            # Dropout layer
            p = params.get('p', 0.5)
            layer = nn.Dropout(p)
            logger.debug(f"Layer {layer_idx}: Dropout(p={p})")
            
        elif layer_type == 'batch_norm':
            # Batch normalization
            layer = nn.BatchNorm1d(input_dim)
            logger.debug(f"Layer {layer_idx}: BatchNorm1d({input_dim})")
            
        elif layer_type == 'layer_norm':
            # Layer normalization
            layer = nn.LayerNorm(input_dim)
            logger.debug(f"Layer {layer_idx}: LayerNorm({input_dim})")
            
        else:
            logger.warning(f"Unknown layer type: {layer_type}")
            return None
            
        return layer
    
    def _get_output_dim(self, layer: nn.Module, input_dim: int) -> int:
        """Get output dimension of a layer."""
        if isinstance(layer, nn.Linear):
            return layer.out_features
        elif isinstance(layer, RingAttractor):
            return layer.num_excitatory
        elif isinstance(layer, MultiRingAttractor):
            return layer.output_size * layer.num_rings
        elif isinstance(layer, MultiAxisRingAttractorLayer):
            return layer.output_dim
        else:
            # For activations, dropout, etc. - dimension doesn't change
            return input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layered network."""
        for i, layer in enumerate(self.layers):
            try:
                x = layer(x)
            except Exception as e:
                logger.error(f"Error in layer {i} ({type(layer).__name__}): {e}")
                logger.error(f"Input shape: {x.shape}")
                raise
        return x


class NetworkBuilder:
    """Builder class for creating layered networks easily."""
    
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.specs = []
    
    def add_linear(self, output_dim: int) -> 'NetworkBuilder':
        """Add a linear layer."""
        self.specs.append(LayerSpec('linear', output_dim=output_dim))
        return self
    
    def add_ring(
        self, 
        num_excitatory: int = 16, 
        config: Optional[RingAttractorConfig] = None
    ) -> 'NetworkBuilder':
        """Add a Ring Attractor layer."""
        if config is None:
            config = RingAttractorConfig()
        self.specs.append(LayerSpec(
            'ring', 
            num_excitatory=num_excitatory, 
            config=config
        ))
        return self
    
    def add_multi_ring(
        self,
        output_size: int = 16,
        num_rings: int = 3,
        config: Optional[RingAttractorConfig] = None
    ) -> 'NetworkBuilder':
        """Add a Multi Ring Attractor layer."""
        if config is None:
            config = RingAttractorConfig()
        self.specs.append(LayerSpec(
            'multi_ring',
            output_size=output_size,
            num_rings=num_rings,
            config=config
        ))
        return self
    
    def add_control_ring(
        self,
        control_axes: List[str],
        ring_axes: Optional[List[str]] = None,
        config: Optional[RingAttractorConfig] = None
    ) -> 'NetworkBuilder':
        """Add a control-specific Ring Attractor layer."""
        if config is None:
            config = RingAttractorConfig()
        if ring_axes is None:
            ring_axes = control_axes[:-1]  # All but last axis
        self.specs.append(LayerSpec(
            'control_ring',
            control_axes=control_axes,
            ring_axes=ring_axes,
            config=config
        ))
        return self
    
    def add_activation(self, activation_type: str = 'relu', **kwargs) -> 'NetworkBuilder':
        """Add an activation layer."""
        self.specs.append(LayerSpec('activation', type=activation_type, **kwargs))
        return self
    
    def add_dropout(self, p: float = 0.5) -> 'NetworkBuilder':
        """Add a dropout layer."""
        self.specs.append(LayerSpec('dropout', p=p))
        return self
    
    def add_batch_norm(self) -> 'NetworkBuilder':
        """Add a batch normalization layer."""
        self.specs.append(LayerSpec('batch_norm'))
        return self
    
    def add_layer_norm(self) -> 'NetworkBuilder':
        """Add a layer normalization layer."""
        self.specs.append(LayerSpec('layer_norm'))
        return self
    
    def build(self, name: str = "CustomNetwork") -> LayeredNetwork:
        """Build the network."""
        return LayeredNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            layer_specs=self.specs,
            name=name
        )


class LayeredSACPolicy:
    """
    Custom SAC policy using layered architecture with Ring Attractors.
    
    This provides a clean interface for creating SAC policies with arbitrary
    layer combinations including Ring Attractors.
    """
    
    @staticmethod
    def create_policy_network(
        input_dim: int,
        output_dim: int,
        architecture: str = "default",
        ring_config: Optional[RingAttractorConfig] = None
    ) -> LayeredNetwork:
        """
        Create a policy network with predefined or custom architecture.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (action space)
            architecture: Architecture type ('default', 'ring_sandwich', 'multi_ring', 'custom')
            ring_config: Ring Attractor configuration
            
        Returns:
            Configured layered network
        """
        if ring_config is None:
            ring_config = RingAttractorConfig()
        
        builder = NetworkBuilder(input_dim, output_dim)
        
        if architecture == "default":
            # Standard MLP: Linear -> ReLU -> Linear -> ReLU -> Ring -> Output
            return (builder
                    .add_linear(256)
                    .add_activation('relu')
                    .add_linear(128)
                    .add_activation('relu')
                    .add_ring(num_excitatory=16, config=ring_config)
                    .build("DefaultRingPolicy"))
        
        elif architecture == "ring_sandwich":
            # Ring layers sandwiched between linear layers
            # Linear -> Ring -> Linear -> Ring -> Linear
            return (builder
                    .add_linear(128)
                    .add_ring(num_excitatory=20, config=ring_config)
                    .add_linear(64)
                    .add_activation('tanh')
                    .add_ring(num_excitatory=12, config=ring_config)
                    .add_linear(32)
                    .build("RingSandwichPolicy"))
        
        elif architecture == "multi_ring":
            # Multiple Ring Attractor layers with different configurations
            return (builder
                    .add_linear(256)
                    .add_activation('relu')
                    .add_multi_ring(output_size=16, num_rings=3, config=ring_config)
                    .add_linear(64)
                    .add_activation('tanh')
                    .add_ring(num_excitatory=8, config=ring_config)
                    .build("MultiRingPolicy"))
        
        elif architecture == "deep_ring":
            # Deep network with Ring Attractors at multiple levels
            return (builder
                    .add_linear(512)
                    .add_activation('relu')
                    .add_dropout(0.1)
                    .add_ring(num_excitatory=32, config=ring_config)
                    .add_linear(256)
                    .add_activation('relu')
                    .add_ring(num_excitatory=16, config=ring_config)
                    .add_linear(128)
                    .add_activation('tanh')
                    .add_ring(num_excitatory=8, config=ring_config)
                    .build("DeepRingPolicy"))
        
        elif architecture == "quadcopter_control":
            # Specialized for quadcopter control
            control_axes = ['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate']
            return (builder
                    .add_linear(256)
                    .add_activation('relu')
                    .add_linear(128)
                    .add_activation('relu')
                    .add_control_ring(
                        control_axes=control_axes,
                        ring_axes=['roll_rate', 'pitch_rate', 'yaw_rate'],
                        config=ring_config
                    )
                    .build("QuadcopterControlPolicy"))
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")


# Convenience functions for common patterns
def create_simple_ring_network(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = [256, 128],
    ring_positions: List[int] = [1],  # Insert rings after these layer indices
    ring_config: Optional[RingAttractorConfig] = None
) -> LayeredNetwork:
    """
    Create a simple network with Ring Attractors at specified positions.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions
        ring_positions: Positions to insert Ring Attractor layers (0-indexed after layers)
        ring_config: Ring Attractor configuration
        
    Returns:
        Configured layered network
        
    Example:
        # Creates: Linear(64->256) -> ReLU -> Ring -> Linear(16->128) -> ReLU -> Linear(128->4)
        net = create_simple_ring_network(64, 4, [256, 128], [1])
    """
    if ring_config is None:
        ring_config = RingAttractorConfig()
    
    builder = NetworkBuilder(input_dim, output_dim)
    
    layer_count = 0
    for i, hidden_dim in enumerate(hidden_dims):
        builder.add_linear(hidden_dim)
        layer_count += 1
        
        builder.add_activation('relu')
        layer_count += 1
        
        # Add Ring Attractor if this position is specified
        if layer_count - 1 in ring_positions:  # -1 because we want after activation
            builder.add_ring(num_excitatory=16, config=ring_config)
    
    return builder.build("SimpleRingNetwork")


def create_quadcopter_policy_network(
    input_dim: int,
    ring_config: Optional[RingAttractorConfig] = None,
    architecture: str = "standard"
) -> LayeredNetwork:
    """
    Create a policy network specifically designed for quadcopter control.
    
    Args:
        input_dim: Input dimension (observation space)
        ring_config: Ring Attractor configuration
        architecture: Architecture variant ('standard', 'deep', 'multi_ring')
        
    Returns:
        Quadcopter control policy network
    """
    if ring_config is None:
        ring_config = RingAttractorConfig(
            num_excitatory=16,
            tau=8.0,
            beta=12.0,
            trainable_structure=True
        )
    
    control_axes = ['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate']
    builder = NetworkBuilder(input_dim, 4)  # 4 control outputs
    
    if architecture == "standard":
        return (builder
                .add_linear(256)
                .add_activation('relu')
                .add_linear(128)
                .add_activation('relu')
                .add_control_ring(control_axes=control_axes, config=ring_config)
                .build("StandardQuadcopterPolicy"))
    
    elif architecture == "deep":
        return (builder
                .add_linear(512)
                .add_activation('relu')
                .add_dropout(0.1)
                .add_ring(num_excitatory=32, config=ring_config)
                .add_linear(256)
                .add_activation('relu')
                .add_ring(num_excitatory=16, config=ring_config)
                .add_linear(128)
                .add_activation('tanh')
                .add_control_ring(control_axes=control_axes, config=ring_config)
                .build("DeepQuadcopterPolicy"))
    
    elif architecture == "multi_ring":
        return (builder
                .add_linear(384)  # Divisible by 3 for multi-ring
                .add_activation('relu')
                .add_multi_ring(output_size=16, num_rings=3, config=ring_config)
                .add_linear(128)
                .add_activation('relu')
                .add_control_ring(control_axes=control_axes, config=ring_config)
                .build("MultiRingQuadcopterPolicy"))
    
    else:
        raise ValueError(f"Unknown quadcopter architecture: {architecture}")


if __name__ == "__main__":
    # Example usage and testing
    from .attractors import RingAttractorConfig
    
    # Test basic network creation
    config = RingAttractorConfig(num_excitatory=16, trainable_structure=True)
    
    # Create a simple ring network
    net1 = create_simple_ring_network(
        input_dim=64,
        output_dim=4,
        hidden_dims=[256, 128],
        ring_positions=[1, 3],
        ring_config=config
    )
    
    print(f"Network 1: {net1}")
    
    # Test with dummy input
    x = torch.randn(32, 64)  # Batch of 32 samples
    try:
        output = net1(x)
        print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Create quadcopter policy
    quad_net = create_quadcopter_policy_network(
        input_dim=20,  # Typical quadcopter observation dimension
        architecture="deep",
        ring_config=config
    )
    
    print(f"Quadcopter network: {quad_net}")