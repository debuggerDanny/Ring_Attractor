"""
Core Ring Attractor implementations.

This module contains the fundamental Ring Attractor neural network architectures
without any dependencies on specific RL frameworks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class RingAttractor(nn.Module):
    """
    Single Ring Attractor implementation using RNN architecture.
    
    Implements the ring attractor model as described in Section 3.2, which uses
    a RNN to maintain stable spatial representations through circular connectivity patterns.
    
    The model implements the equation:
        Q(s,a) = β * tanh((1/τ) * Φθ(st)^T * V + h(v)^T * U)
    
    Where:
        - V(s): Fixed input-to-hidden connections preserving ring topology
        - U(v): Learnable hidden-to-hidden connections for action relationships
        - τ: Time integration constant
        - β: Scaling factor preventing tanh saturation
    
    Args:
        input_dim (int): Dimension of input features Φθ(s)
        num_excitatory (int): Number of excitatory neurons arranged in ring topology
        tau (float): Initial time integration constant controlling temporal evolution
        beta (float): Initial scaling factor for preventing tanh saturation
        lambda_decay (float): Decay parameter for distance-dependent weights
        trainable_structure (bool): If False, maintains fixed ring structure
        scale (float): Scaling factor for weight initialization
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_excitatory: int, 
        tau: float = 10.0, 
        beta: float = 10.0, 
        lambda_decay: float = 0.9,
        trainable_structure: bool = True,
        scale: float = 0.000025
    ):
        super(RingAttractor, self).__init__()
        
        # Store hyperparameters
        self.input_dim = input_dim
        self.num_excitatory = num_excitatory
        self.lambda_decay = lambda_decay
        self.scale = scale
        
        # Learnable parameters from Section 3.2.1
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        
        # RNN implementation of ring attractor dynamics
        self.rnn = nn.RNN(input_dim, num_excitatory, bias=False)
        
        # Initialize connectivity matrices
        if not trainable_structure:
            self._initialize_fixed_structure()
        
        logger.debug(f"Initialized RingAttractor: {num_excitatory} neurons, "
                    f"input_dim={input_dim}, trainable={trainable_structure}")
    
    def _initialize_fixed_structure(self) -> None:
        """Initialize ring attractor with fixed topological structure."""
        # V(s): Fixed input-to-hidden connections preserving ring topology
        input_weights = self._create_distance_weights(self.num_excitatory, self.input_dim)
        self.rnn.weight_ih_l0 = nn.Parameter(
            torch.Tensor(input_weights), requires_grad=False
        )
        
        # U(v): Learnable hidden-to-hidden connections for action relationships
        hidden_weights = self._create_distance_weights(self.num_excitatory, self.num_excitatory)
        self.rnn.weight_hh_l0 = nn.Parameter(
            torch.Tensor(hidden_weights), requires_grad=True
        )
    
    def _create_distance_weights(self, output_size: int, input_size: int) -> np.ndarray:
        """
        Create distance-dependent weight matrix implementing:
        w_{m,n} = scale * e^{-d(m,n)/λ}
        
        where d(m,n) is the circular distance between neurons m and n.
        
        Args:
            output_size (int): Number of output neurons
            input_size (int): Number of input neurons
            
        Returns:
            np.ndarray: Weight matrix with circular topology
        """
        weights = np.zeros((output_size, input_size))
        
        for m in range(output_size):
            for n in range(input_size):
                if output_size == input_size:
                    # Same size: direct circular distance
                    d_mn = min(abs(m - n), output_size - abs(m - n))
                else:
                    # Different sizes: scale appropriately
                    ratio = input_size / output_size
                    scaled_n = n / ratio
                    d_mn = min(abs(m - scaled_n), output_size - abs(m - scaled_n))
                
                # Apply exponential decay based on distance
                weights[m, n] = self.scale * np.exp(-d_mn / self.lambda_decay)
        
        return weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing ring attractor dynamics.
        
        Args:
            x (torch.Tensor): Input features representing state
            
        Returns:
            torch.Tensor: Ring attractor output
        """
        # Scale input by learnable temporal integration constant τ
        x_scaled = (1.0 / self.tau) * x
        
        # Apply RNN dynamics (ring attractor)
        ring_output, _ = self.rnn(x_scaled)
        
        # Apply scaling factor β
        return self.beta * ring_output


class MultiRingAttractor(nn.Module):
    """
    Multiple coupled Ring Attractor implementation.
    
    Features multiple coupled ring attractors with cross-connections between rings,
    allowing for complex multi-dimensional representations.
    
    Args:
        input_size (int): Size of input feature vector
        output_size (int): Number of neurons in each ring attractor
        num_rings (int): Number of coupled rings
        trainable_structure (bool): Whether to use trainable connectivity
        connectivity_strength (float): Strength of synaptic connections
        tau (float): Forward pass input integration constant
        beta (float): Scaling factor preventing output saturation
        cross_coupling_factor (float): Strength of coupling between rings
        lambda_decay (float): Decay parameter for distance-dependent weights
    """
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        num_rings: int = 3,
        trainable_structure: bool = True,
        connectivity_strength: float = 0.1, 
        tau: float = 10.0, 
        beta: float = 10.0,
        cross_coupling_factor: float = 0.1,
        lambda_decay: float = 0.9
    ):
        super(MultiRingAttractor, self).__init__()
        
        # Store hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.num_rings = num_rings
        self.connectivity_strength = connectivity_strength
        self.lambda_decay = lambda_decay
        self.cross_coupling_factor_K = cross_coupling_factor
        
        # Learnable parameters
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        
        # Initialize RNN for multiple rings
        total_input_size = input_size * num_rings
        total_output_size = output_size * num_rings
        self.rnn = nn.RNN(total_input_size, total_output_size, bias=False, batch_first=True)
        
        # Create connectivity matrices
        if not trainable_structure:
            self._initialize_multi_ring_connectivity()
        
        logger.debug(f"Initialized MultiRingAttractor: {num_rings} rings, "
                    f"{output_size} neurons each")
    
    def _initialize_multi_ring_connectivity(self) -> None:
        """Initialize connectivity matrices for multi-ring architecture."""
        # Create base connectivity matrices for single ring
        input_to_hidden_single = self._create_input_connectivity()
        hidden_to_hidden_single = self._create_hidden_connectivity()
        
        # Expand to multi-ring matrices with cross-connections
        input_to_hidden_multi = self._create_multi_ring_matrix(
            input_to_hidden_single, self.input_size, self.output_size
        )
        hidden_to_hidden_multi = self._create_multi_ring_matrix(
            hidden_to_hidden_single, self.output_size, self.output_size
        )
        
        # Set RNN weights
        self.rnn.weight_ih_l0 = nn.Parameter(
            torch.Tensor(input_to_hidden_multi), requires_grad=False
        )
        self.rnn.weight_hh_l0 = nn.Parameter(
            torch.Tensor(hidden_to_hidden_multi), requires_grad=True
        )
    
    def _create_multi_ring_matrix(
        self, 
        single_matrix: np.ndarray, 
        dim1: int, 
        dim2: int
    ) -> np.ndarray:
        """
        Create multi-ring matrix with cross-connections from single ring matrix.
        
        Args:
            single_matrix (np.ndarray): Base connectivity matrix for single ring
            dim1 (int): First dimension size
            dim2 (int): Second dimension size
            
        Returns:
            np.ndarray: Multi-ring matrix with cross-connections
        """
        multi_matrix = np.zeros((dim2 * self.num_rings, dim1 * self.num_rings))
        
        # Main diagonal blocks (primary connections within each ring)
        for i in range(self.num_rings):
            start_row, end_row = i * dim2, (i + 1) * dim2
            start_col, end_col = i * dim1, (i + 1) * dim1
            multi_matrix[start_row:end_row, start_col:end_col] = single_matrix
        
        # Off-diagonal blocks (cross-connections between rings)
        for i in range(self.num_rings):
            for j in range(self.num_rings):
                if i != j:
                    start_row, end_row = i * dim2, (i + 1) * dim2
                    start_col, end_col = j * dim1, (j + 1) * dim1
                    multi_matrix[start_row:end_row, start_col:end_col] = (
                        single_matrix * self.cross_coupling_factor_K
                    )
        
        return multi_matrix
    
    def _create_input_connectivity(self) -> np.ndarray:
        """
        Create connectivity matrix for input-to-hidden connections maintaining
        circular topology.
        
        Returns:
            np.ndarray: Input-to-hidden connectivity matrix
        """
        matrix = np.zeros((self.output_size, self.input_size))
        ratio = self.input_size / self.output_size
        
        for i in range(self.output_size):
            for j in range(self.input_size):
                # Calculate circular distance considering input/output size ratio
                scaled_j = j / ratio
                distance = min(abs(i - scaled_j), self.output_size - abs(i - scaled_j))
                matrix[i, j] = self.connectivity_strength * np.exp(-distance / self.lambda_decay)
        
        return matrix
    
    def _create_hidden_connectivity(self) -> np.ndarray:
        """
        Create connectivity matrix for hidden-to-hidden connections within each ring.
        
        Returns:
            np.ndarray: Hidden-to-hidden connectivity matrix
        """
        matrix = np.zeros((self.output_size, self.output_size))
        
        for i in range(self.output_size):
            for j in range(self.output_size):
                # Calculate circular distance
                distance = min(abs(i - j), self.output_size - abs(i - j))
                matrix[i, j] = self.connectivity_strength * np.exp(-distance / self.lambda_decay)
        
        return matrix
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing multi-ring attractor dynamics.
        
        Args:
            x (torch.Tensor): Input tensor representing state
            
        Returns:
            torch.Tensor: Output after multi-ring attractor dynamics
        """
        # Scale input by learnable temporal integration constant
        x_scaled = (1.0 / self.tau) * x
        
        # Apply RNN dynamics (multi-ring attractor)
        ring_output, _ = self.rnn(x_scaled)
        
        # Apply scaling
        return self.beta * ring_output


class RingAttractorConfig:
    """Configuration class for Ring Attractor parameters."""
    
    def __init__(
        self,
        num_excitatory: int = 16,
        tau: float = 10.0,
        beta: float = 10.0,
        lambda_decay: float = 0.9,
        trainable_structure: bool = True,
        connectivity_strength: float = 0.1,
        cross_coupling_factor: float = 0.1,
        scale: float = 0.000025
    ):
        self.num_excitatory = num_excitatory
        self.tau = tau
        self.beta = beta
        self.lambda_decay = lambda_decay
        self.trainable_structure = trainable_structure
        self.connectivity_strength = connectivity_strength
        self.cross_coupling_factor = cross_coupling_factor
        self.scale = scale
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'num_excitatory': self.num_excitatory,
            'tau': self.tau,
            'beta': self.beta,
            'lambda_decay': self.lambda_decay,
            'trainable_structure': self.trainable_structure,
            'connectivity_strength': self.connectivity_strength,
            'cross_coupling_factor': self.cross_coupling_factor,
            'scale': self.scale
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RingAttractorConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)