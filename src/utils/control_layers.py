"""
Control-specific Ring Attractor layers.

This module contains specialized Ring Attractor layers designed for control tasks,
such as multi-axis control systems (roll, yaw, pitch, thrust).
"""

import torch
import torch.nn as nn
from typing import List, Optional
import logging

from src.utils.attractors import RingAttractor, MultiRingAttractor, RingAttractorConfig

logger = logging.getLogger(__name__)


class BaseControlLayer(nn.Module):
    """Base class for control-oriented Ring Attractor layers."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(BaseControlLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_transform = nn.Linear(input_dim, input_dim)
        self.activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError


class SingleAxisRingAttractorLayer(BaseControlLayer):
    """
    Single-axis Ring Attractor layer for simple control tasks.
    
    Uses a single ring attractor followed by a linear output layer.
    
    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output dimension (number of actions)
        config (RingAttractorConfig): Ring attractor configuration
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        config: Optional[RingAttractorConfig] = None
    ):
        super(SingleAxisRingAttractorLayer, self).__init__(input_dim, output_dim)
        
        if config is None:
            config = RingAttractorConfig()
        
        self.ring_attractor = RingAttractor(
            input_dim=input_dim,
            num_excitatory=config.num_excitatory,
            tau=config.tau,
            beta=config.beta,
            lambda_decay=config.lambda_decay,
            trainable_structure=config.trainable_structure,
            scale=config.scale
        )
        
        self.output_layer = nn.Linear(config.num_excitatory, output_dim)
        
        logger.info(f"Initialized SingleAxisRingAttractorLayer: "
                   f"input_dim={input_dim}, output_dim={output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through single-axis ring attractor."""
        x_transformed = self.input_transform(x)
        ring_output = self.ring_attractor(x_transformed)
        action_values = self.output_layer(ring_output)
        return self.activation(action_values)


class MultiAxisRingAttractorLayer(BaseControlLayer):
    """
    Multi-axis Ring Attractor layer for complex control tasks.
    
    Uses separate ring attractors for different control axes (e.g., roll, yaw, pitch)
    plus linear layers for other controls (e.g., thrust).
    
    Args:
        input_dim (int): Input feature dimension
        control_axes (List[str]): Names of control axes (e.g., ['roll', 'yaw', 'pitch', 'thrust'])
        config (RingAttractorConfig): Ring attractor configuration
        ring_axes (List[str]): Which axes use ring attractors (others use linear layers)
    """
    
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        control_axes: List[str],
        config: Optional[RingAttractorConfig] = None,
        ring_axes: Optional[List[str]] = None
    ):
        super(MultiAxisRingAttractorLayer, self).__init__(input_dim, output_dim)
        
        if config is None:
            config = RingAttractorConfig()
        
        self.control_axes = control_axes
        self.ring_axes = ring_axes or control_axes[:-1]  # Default: all but last axis (usually the thrust)
        self.linear_axes = [axis for axis in control_axes if axis not in self.ring_axes]
        
        # Calculate split sizes
        num_ring_axes = len(self.ring_axes)
        num_linear_axes = len(self.linear_axes)
        
        if num_ring_axes > 0:
            self.ring_split_size = input_dim // (num_ring_axes + 1)  # +1 for linear part
            self.linear_split_size = input_dim - (num_ring_axes * self.ring_split_size)
        else:
            self.ring_split_size = 0
            self.linear_split_size = input_dim
        
        # Create ring attractors for specified axes
        self.ring_attractors = nn.ModuleDict()
        for axis in self.ring_axes:
            self.ring_attractors[axis] = RingAttractor(
                input_dim=self.ring_split_size,
                num_excitatory=config.num_excitatory,
                tau=config.tau,
                beta=config.beta,
                lambda_decay=config.lambda_decay,
                trainable_structure=config.trainable_structure,
                scale=config.scale
            )
        
        # Create linear layers for remaining axes
        self.linear_layers = nn.ModuleDict()
        for axis in self.linear_axes:
            self.linear_layers[axis] = nn.Linear(self.linear_split_size, 1)
        
        # Final linear layer to combine outputs and match output_dim
        intermediate_dim = len(self.ring_axes) + len(self.linear_axes)
        self.final_linear = nn.Linear(intermediate_dim, output_dim)
        
        logger.info(f"Initialized MultiAxisRingAttractorLayer: "
                   f"ring_axes={self.ring_axes}, linear_axes={self.linear_axes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-axis ring attractor."""
        x_transformed = self.input_transform(x)
        
        outputs = []
        
        # Process ring attractor axes
        if self.ring_axes:
            ring_splits = torch.split(x_transformed, self.ring_split_size, dim=1)
            
            for i, axis in enumerate(self.ring_axes):
                ring_output = self.ring_attractors[axis](ring_splits[i])
                # Use a simple linear layer to get single output per axis
                axis_output = torch.mean(ring_output, dim=1, keepdim=True)  # Simple pooling
                outputs.append(axis_output)
        
        # Process linear axes
        if self.linear_axes:
            linear_input = x_transformed[:, -self.linear_split_size:] if self.ring_axes else x_transformed
            
            for axis in self.linear_axes:
                linear_output = self.linear_layers[axis](linear_input)
                outputs.append(linear_output)
        
        # Combine all outputs
        combined_output = torch.cat(outputs, dim=1)
        
        # Pass through final linear layer to get desired output dimension
        final_output = self.final_linear(combined_output)
        return self.activation(final_output)


class CoupledRingAttractorLayer(BaseControlLayer):
    """
    Coupled Ring Attractor layer using multi-ring architecture.
    
    Uses multiple coupled ring attractors for integrated control,
    with optional linear layers for additional control axes.
    
    Args:
        input_dim (int): Input feature dimension
        control_axes (List[str]): Names of control axes
        num_rings (int): Number of coupled rings for the main control axes
        config (RingAttractorConfig): Ring attractor configuration
        coupled_axes (List[str]): Which axes use coupled rings (others use linear layers)
    """
    
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        control_axes: List[str],
        num_rings: int = 3,
        config: Optional[RingAttractorConfig] = None,
        coupled_axes: Optional[List[str]] = None
    ):
        super(CoupledRingAttractorLayer, self).__init__(input_dim, output_dim)
        
        if config is None:
            config = RingAttractorConfig()
        
        self.control_axes = control_axes
        self.num_rings = num_rings
        self.coupled_axes = coupled_axes or control_axes[:num_rings]  # Default: first N axes
        self.linear_axes = [axis for axis in control_axes if axis not in self.coupled_axes]
        
        # Calculate input splits
        num_coupled_axes = len(self.coupled_axes)
        if num_coupled_axes > 0:
            self.coupled_input_size = input_dim * 3 // 4  # 3/4 for coupled rings
            self.linear_input_size = input_dim - self.coupled_input_size
        else:
            self.coupled_input_size = 0
            self.linear_input_size = input_dim
        
        # Create multi-ring attractor for coupled axes
        if self.coupled_axes:
            self.multi_ring_attractor = MultiRingAttractor(
                input_size=self.coupled_input_size // num_rings,
                output_size=config.num_excitatory,
                num_rings=num_rings,
                trainable_structure=config.trainable_structure,
                connectivity_strength=config.connectivity_strength,
                tau=config.tau,
                beta=config.beta,
                cross_coupling_factor=config.cross_coupling_factor,
                lambda_decay=config.lambda_decay
            )
            
            # Output layer for multi-ring
            self.coupled_output_layer = nn.Linear(
                config.num_excitatory * num_rings, 
                len(self.coupled_axes)
            )
        
        # Create linear layers for remaining axes
        self.linear_layers = nn.ModuleDict()
        for axis in self.linear_axes:
            self.linear_layers[axis] = nn.Linear(self.linear_input_size, 1)
        
        logger.info(f"Initialized CoupledRingAttractorLayer: "
                   f"coupled_axes={self.coupled_axes}, linear_axes={self.linear_axes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through coupled ring attractor."""
        x_transformed = self.input_transform(x)
        
        outputs = []
        
        # Process coupled ring attractor axes
        if self.coupled_axes:
            # Split input for coupled vs linear processing
            coupled_input, linear_input = torch.split(
                x_transformed, 
                [self.coupled_input_size, self.linear_input_size], 
                dim=-1
            )
            
            # Repeat coupled input for multi-ring processing
            coupled_input_repeated = coupled_input.repeat(1, self.num_rings)
            
            # Process through multi-ring attractor
            multi_ring_output = self.multi_ring_attractor(coupled_input_repeated)
            coupled_output = self.coupled_output_layer(multi_ring_output)
            outputs.append(coupled_output)
            
            # Process linear axes
            for axis in self.linear_axes:
                linear_output = self.linear_layers[axis](linear_input)
                outputs.append(linear_output)
        else:
            # Only linear processing
            for axis in self.linear_axes:
                linear_output = self.linear_layers[axis](x_transformed)
                outputs.append(linear_output)
        
        # Combine all outputs
        if len(outputs) == 1:
            combined_output = outputs[0]
        else:
            combined_output = torch.cat(outputs, dim=1)
        
        return self.activation(combined_output)


# Factory functions for easy instantiation
def create_control_layer(
    layer_type: str,
    input_dim: int,
    output_dim:int, 
    control_axes: List[str],
    config: Optional[RingAttractorConfig] = None,
    **kwargs
) -> BaseControlLayer:
    """
    Factory function to create control layers.
    
    Args:
        layer_type (str): Type of layer ('single', 'multi', 'coupled', 'adaptive')
        input_dim (int): Input dimension
        control_axes (List[str]): Control axis names
        config (RingAttractorConfig): Configuration
        **kwargs: Additional arguments
        
    Returns:
        BaseControlLayer: The created control layer
    """
    if layer_type == 'single':
        return SingleAxisRingAttractorLayer(input_dim, output_dim, len(control_axes), config)
    elif layer_type == 'multi':
        return MultiAxisRingAttractorLayer(input_dim, output_dim, control_axes, config, **kwargs)
    elif layer_type == 'coupled':
        return CoupledRingAttractorLayer(input_dim, output_dim, control_axes, config=config, **kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


# Predefined configurations for common control tasks
def get_quadrotor_config() -> dict:
    """Configuration for quadrotor control (roll, yaw, pitch, thrust)."""
    return {
        'control_axes': ['roll', 'yaw', 'pitch', 'thrust'],
        'ring_axes': ['roll', 'yaw', 'pitch'],
        'coupled_axes': ['roll', 'yaw', 'pitch'],
        'config': RingAttractorConfig(
            num_excitatory=16,
            tau=10.0,
            beta=10.0,
            trainable_structure=False
        )
    }

