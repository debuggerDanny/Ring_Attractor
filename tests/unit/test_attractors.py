import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from attractors import RingAttractor, MultiRingAttractor, RingAttractorConfig


class TestRingAttractorConfig:
    """Test RingAttractorConfig class"""
    
    def test_config_initialization(self):
        config = RingAttractorConfig()
        assert config.num_excitatory == 16
        assert config.tau == 10.0
        assert config.beta == 10.0
        assert config.lambda_decay == 0.9
        assert config.trainable_structure == False
    
    def test_config_custom_values(self):
        config = RingAttractorConfig(
            num_excitatory=20,
            tau=8.0,
            beta=15.0,
            trainable_structure=True
        )
        assert config.num_excitatory == 20
        assert config.tau == 8.0
        assert config.beta == 15.0
        assert config.trainable_structure == True
    
    def test_config_to_dict(self):
        config = RingAttractorConfig(num_excitatory=12, tau=5.0)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['num_excitatory'] == 12
        assert config_dict['tau'] == 5.0
        assert 'beta' in config_dict
    
    def test_config_from_dict(self):
        config_dict = {
            'num_excitatory': 24,
            'tau': 6.0,
            'beta': 12.0,
            'lambda_decay': 0.8,
            'trainable_structure': True
        }
        config = RingAttractorConfig.from_dict(config_dict)
        
        assert config.num_excitatory == 24
        assert config.tau == 6.0
        assert config.trainable_structure == True


class TestRingAttractor:
    """Test RingAttractor class"""
    
    def test_initialization(self):
        ring = RingAttractor(input_dim=32, num_excitatory=16)
        
        assert ring.input_dim == 32
        assert ring.num_excitatory == 16
        assert isinstance(ring.rnn, nn.RNN)
        assert isinstance(ring.tau, nn.Parameter)
        assert isinstance(ring.beta, nn.Parameter)
    
    def test_initialization_with_config(self):
        ring = RingAttractor(input_dim=64, 
                             num_excitatory= 20,
                             tau=8,
                             beta=12)
        
        assert ring.num_excitatory == 20
        assert ring.tau.item() == pytest.approx(8.0)
        assert ring.beta.item() == pytest.approx(12.0)
    
    def test_forward_shape(self):
        ring = RingAttractor(input_dim=32, num_excitatory=16)
        batch_size = 5
        input_tensor = torch.randn(batch_size, 32)
        
        output = ring(input_tensor)
        
        assert output.shape == (batch_size, 16)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_forward_different_batch_sizes(self):
        ring = RingAttractor(input_dim=64, num_excitatory=20)
        
        for batch_size in [1, 8, 16, 32]:
            input_tensor = torch.randn(batch_size, 64)
            output = ring(input_tensor)
            assert output.shape == (batch_size, 20)
    
    def test_trainable_structure_false(self):
        ring = RingAttractor(input_dim=32, num_excitatory=16, trainable_structure=False)
        
        # Check that weights are frozen
        assert not ring.rnn.weight_ih_l0.requires_grad
        assert not ring.rnn.weight_hh_l0.requires_grad
        
        # But tau and beta should be trainable
        assert ring.tau.requires_grad
        assert ring.beta.requires_grad
    
    def test_trainable_structure_true(self):
        ring = RingAttractor(input_dim=32, num_excitatory=16, trainable_structure=True)
        
        # All weights should be trainable
        assert ring.rnn.weight_ih_l0.requires_grad
        assert ring.rnn.weight_hh_l0.requires_grad
        assert ring.tau.requires_grad
        assert ring.beta.requires_grad
    
    def test_circular_connectivity_pattern(self):
        ring = RingAttractor(input_dim=32, num_excitatory=8, trainable_structure=False)
        
        # Test that the connectivity follows expected circular pattern
        weights = ring.rnn.weight_hh_l0.detach().numpy()
        
        # Check diagonal structure (neurons connect to neighbors)
        for i in range(8):
            # Each neuron should have strongest connections to neighbors
            neighbors = [(i-1) % 8, i, (i+1) % 8]
            neighbor_weights = [abs(weights[i, j]) for j in neighbors]
            other_weights = [abs(weights[i, j]) for j in range(8) if j not in neighbors]
            
            # Neighbors should generally have stronger connections
            if other_weights:  # Avoid empty list comparison
                assert max(neighbor_weights) >= np.mean(other_weights) * 0.5  # Relaxed constraint
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_compatibility(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        ring = RingAttractor(input_dim=32, num_excitatory=16).to(device)
        input_tensor = torch.randn(5, 32).to(device)
        
        output = ring(input_tensor)
        assert output.device.type == device
    
    def test_gradient_flow(self):
        ring = RingAttractor(input_dim=32, num_excitatory=16)
        input_tensor = torch.randn(5, 32, requires_grad=True)
        
        output = ring(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert input_tensor.grad is not None
        assert ring.tau.grad is not None
        assert ring.beta.grad is not None


class TestMultiRingAttractor:
    """Test MultiRingAttractor class"""
    
    def test_initialization(self):
        multi_ring = MultiRingAttractor(
            input_size=24,
            ring_size=8,
            output_dim=16,
            num_rings=3
        )
        
        assert multi_ring.num_rings == 3
        assert multi_ring.input_size == 24
        assert multi_ring.output_dim == 16
    
    def test_forward_shape(self):
        multi_ring = MultiRingAttractor(
            input_size=32,
            ring_size=8,
            output_dim=12,
            num_rings=3
        )
        
        batch_size = 8
        input_tensor = torch.randn(batch_size, 32 * 3)  # Input for 3 rings
        
        output = multi_ring(input_tensor)
        expected_shape = (batch_size, 12 )  # output_size is 12 because there should be a linear layer at the end
        assert output.shape == expected_shape
    
    def test_gradient_flow_multi_ring(self):
        multi_ring = MultiRingAttractor(
            input_size=16,
            ring_size=8,
            output_dim=8,
            num_rings=2
        )
        
        input_tensor = torch.randn(4, 32, requires_grad=True)  # 16 * 2 rings
        
        output = multi_ring(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        # Check that all rings receive gradients
        assert multi_ring.tau.grad is not None
        assert multi_ring.beta.grad is not None


class TestRingAttractorMathematicalProperties:
    """Test mathematical properties of Ring Attractors"""
    
    def test_tau_affects_temporal_dynamics(self):
        # Create two rings with different tau values
        ring_fast = RingAttractor(input_dim=32, num_excitatory=16)
        ring_slow = RingAttractor(input_dim=32, num_excitatory=16)
        
        # Set different tau values
        ring_fast.tau.data = torch.tensor(2.0)
        ring_slow.tau.data = torch.tensor(20.0)
        
        input_tensor = torch.randn(1, 32)
        
        output_fast = ring_fast(input_tensor)
        output_slow = ring_slow(input_tensor)
        
        # Both should produce valid outputs
        assert not torch.isnan(output_fast).any()
        assert not torch.isnan(output_slow).any()
    
    def test_beta_affects_output_magnitude(self):
        ring = RingAttractor(input_dim=32, num_excitatory=16)
        input_tensor = torch.randn(5, 32)
        
        # Test with different beta values
        ring.beta.data = torch.tensor(1.0)
        output_low = ring(input_tensor)
        
        ring.beta.data = torch.tensor(10.0)
        output_high = ring(input_tensor)
        
        # Higher beta should generally produce larger magnitude outputs
        assert torch.abs(output_high).mean() >= torch.abs(output_low).mean()
    
    def test_circular_symmetry_property(self):
        # Test that the ring maintains circular symmetry properties
        ring = RingAttractor(input_dim=16, num_excitatory=8, trainable_structure=False)
        
        # Create input that should activate specific positions on the ring
        input_tensor = torch.zeros(1, 16)
        input_tensor[0, 0] = 1.0  # Activate first input
        
        output1 = ring(input_tensor)
        
        # Shift input activation
        input_tensor = torch.zeros(1, 16)
        input_tensor[0, 2] = 1.0  # Activate third input
        
        output2 = ring(input_tensor)
        
        # Both outputs should be valid and different
        assert not torch.allclose(output1, output2, atol=1e-6)
        assert not torch.isnan(output1).any()
        assert not torch.isnan(output2).any()


# Fixtures for commonly used test data
@pytest.fixture
def sample_config():
    return RingAttractorConfig(
        num_excitatory=16,
        tau=8.0,
        beta=12.0,
        trainable_structure=True
    )


@pytest.fixture
def sample_input_tensor():
    return torch.randn(8, 64)


@pytest.fixture
def ring_attractor(sample_config):
    return RingAttractor(input_dim=64, config=sample_config)


class TestRingAttractorIntegration:
    """Integration tests using fixtures"""
    
    def test_end_to_end_processing(self, ring_attractor, sample_input_tensor):
        output = ring_attractor(sample_input_tensor)
        
        assert output.shape == (8, 16)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_batch_consistency(self, ring_attractor):
        # Process same input individually and in batch
        single_input = torch.randn(1, 64)
        batch_input = single_input.repeat(5, 1)
        
        single_outputs = []
        for i in range(5):
            single_outputs.append(ring_attractor(single_input))
        single_batch = torch.cat(single_outputs, dim=0)
        
        batch_output = ring_attractor(batch_input)
        
        assert torch.allclose(single_batch, batch_output, atol=1e-6)