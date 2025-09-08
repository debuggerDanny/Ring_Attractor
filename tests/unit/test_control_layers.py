import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from control_layers import (
    SingleAxisRingAttractorLayer,
    MultiAxisRingAttractorLayer, 
    CoupledRingAttractorLayer,
    create_control_layer
)
from attractors import RingAttractorConfig


class TestSingleAxisRingAttractorLayer:
    """Test SingleAxisRingAttractorLayer class"""
    
    def test_initialization(self):
        config = RingAttractorConfig(num_excitatory=16)
        layer = SingleAxisRingAttractorLayer(
            input_dim=64,
            output_dim=1,
            config=config
        )
        
        assert layer.input_dim == 64
        assert layer.output_dim == 1
        assert hasattr(layer, 'ring_attractor')
        assert hasattr(layer, 'output_projection')
    
    def test_forward_shape(self):
        config = RingAttractorConfig(num_excitatory=12)
        layer = SingleAxisRingAttractorLayer(
            input_dim=32,
            output_dim=1,
            config=config
        )
        
        batch_size = 8
        input_tensor = torch.randn(batch_size, 32)
        output = layer(input_tensor)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_gradient_flow(self):
        config = RingAttractorConfig(num_excitatory=16)
        layer = SingleAxisRingAttractorLayer(
            input_dim=64,
            output_dim=1,
            config=config
        )
        
        input_tensor = torch.randn(5, 64, requires_grad=True)
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert layer.ring_attractor.tau.grad is not None


class TestMultiAxisRingAttractorLayer:
    """Test MultiAxisRingAttractorLayer class"""
    
    def test_initialization(self):
        config = RingAttractorConfig(num_excitatory=16)
        layer = MultiAxisRingAttractorLayer(
            input_dim=128,
            control_axes=['roll', 'pitch', 'yaw', 'thrust'],
            ring_axes=['roll', 'pitch', 'yaw'],
            config=config
        )
        
        assert layer.input_dim == 128
        assert len(layer.control_axes) == 4
        assert len(layer.ring_axes) == 3
        assert len(layer.non_ring_axes) == 1  # thrust
        assert 'thrust' in layer.non_ring_axes
    
    def test_forward_shape(self):
        config = RingAttractorConfig(num_excitatory=12)
        layer = MultiAxisRingAttractorLayer(
            input_dim=64,
            control_axes=['x', 'y', 'z', 'thrust'],
            ring_axes=['x', 'y', 'z'],
            config=config
        )
        
        batch_size = 6
        input_tensor = torch.randn(batch_size, 64)
        output = layer(input_tensor)
        
        assert output.shape == (batch_size, 4)  # 4 control axes
        assert not torch.isnan(output).any()
    
    def test_ring_vs_non_ring_axes(self):
        config = RingAttractorConfig(num_excitatory=8)
        layer = MultiAxisRingAttractorLayer(
            input_dim=32,
            control_axes=['roll', 'pitch', 'thrust'],
            ring_axes=['roll', 'pitch'],  # Only 2 use ring attractors
            config=config
        )
        
        # Check that correct number of components are created
        assert len(layer.ring_layers) == 2  # roll, pitch
        assert len(layer.linear_layers) == 1  # thrust
        
        input_tensor = torch.randn(4, 32)
        output = layer(input_tensor)
        assert output.shape == (4, 3)  # 3 control axes total
    
    def test_all_ring_axes(self):
        config = RingAttractorConfig(num_excitatory=16)
        layer = MultiAxisRingAttractorLayer(
            input_dim=64,
            control_axes=['roll', 'pitch', 'yaw'],
            ring_axes=['roll', 'pitch', 'yaw'],  # All axes use rings
            config=config
        )
        
        assert len(layer.ring_layers) == 3
        assert len(layer.linear_layers) == 0
        assert len(layer.non_ring_axes) == 0
    
    def test_no_ring_axes(self):
        config = RingAttractorConfig(num_excitatory=16)
        layer = MultiAxisRingAttractorLayer(
            input_dim=64,
            control_axes=['thrust', 'power'],
            ring_axes=[],  # No ring attractors
            config=config
        )
        
        assert len(layer.ring_layers) == 0
        assert len(layer.linear_layers) == 2
        
        input_tensor = torch.randn(3, 64)
        output = layer(input_tensor)
        assert output.shape == (3, 2)


class TestCoupledRingAttractorLayer:
    """Test CoupledRingAttractorLayer class"""
    
    def test_initialization(self):
        config = RingAttractorConfig(num_excitatory=12)
        layer = CoupledRingAttractorLayer(
            input_dim=128,
            control_axes=['x', 'y', 'z', 'thrust'],
            num_rings=3,
            coupled_axes=['x', 'y', 'z'],
            config=config
        )
        
        assert layer.num_rings == 3
        assert len(layer.coupled_axes) == 3
        assert len(layer.uncoupled_axes) == 1
        assert hasattr(layer, 'multi_ring')
    
    def test_forward_shape(self):
        config = RingAttractorConfig(num_excitatory=16)
        layer = CoupledRingAttractorLayer(
            input_dim=64,
            control_axes=['roll', 'pitch', 'yaw'],
            num_rings=3,
            coupled_axes=['roll', 'pitch', 'yaw'],
            config=config
        )
        
        batch_size = 5
        input_tensor = torch.randn(batch_size, 64)
        output = layer(input_tensor)
        
        assert output.shape == (batch_size, 3)
        assert not torch.isnan(output).any()
    
    def test_mixed_coupled_uncoupled_axes(self):
        config = RingAttractorConfig(num_excitatory=10)
        layer = CoupledRingAttractorLayer(
            input_dim=96,
            control_axes=['x', 'y', 'z', 'thrust', 'power'],
            num_rings=2,
            coupled_axes=['x', 'y'],  # Only x, y coupled
            config=config
        )
        
        assert len(layer.coupled_axes) == 2
        assert len(layer.uncoupled_axes) == 3  # z, thrust, power
        
        input_tensor = torch.randn(7, 96)
        output = layer(input_tensor)
        assert output.shape == (7, 5)  # All 5 control axes



class TestCreateControlLayer:
    """Test create_control_layer factory function"""
    
    def test_create_single_layer(self):
        config = RingAttractorConfig(num_excitatory=16)
        
        layer = create_control_layer(
            layer_type='single',
            input_dim=64,
            output_dim=1,
            config=config
        )
        
        assert isinstance(layer, SingleAxisRingAttractorLayer)
        assert layer.input_dim == 64
        assert layer.output_dim == 1
    
    def test_create_multi_layer(self):
        config = RingAttractorConfig(num_excitatory=12)
        
        layer = create_control_layer(
            layer_type='multi',
            input_dim=128,
            control_axes=['roll', 'pitch', 'yaw', 'thrust'],
            ring_axes=['roll', 'pitch', 'yaw'],
            config=config
        )
        
        assert isinstance(layer, MultiAxisRingAttractorLayer)
        assert len(layer.control_axes) == 4
    
    def test_create_coupled_layer(self):
        config = RingAttractorConfig(num_excitatory=20)
        
        layer = create_control_layer(
            layer_type='coupled',
            input_dim=96,
            control_axes=['x', 'y', 'z'],
            num_rings=3,
            coupled_axes=['x', 'y', 'z'],
            config=config
        )
        
        assert isinstance(layer, CoupledRingAttractorLayer)
        assert layer.num_rings == 3
    
    def test_invalid_layer_type(self):
        config = RingAttractorConfig()
        
        with pytest.raises(ValueError, match="Unknown layer_type"):
            create_control_layer(
                layer_type='invalid_type',
                input_dim=64,
                config=config
            )


class TestControlLayerMathematicalProperties:
    """Test mathematical properties of control layers"""
    
    def test_output_range_single_axis(self):
        config = RingAttractorConfig(num_excitatory=16, beta=1.0)
        layer = SingleAxisRingAttractorLayer(
            input_dim=32,
            output_dim=1,
            config=config
        )
        
        # Test with various input magnitudes
        for magnitude in [0.1, 1.0, 10.0]:
            input_tensor = torch.randn(10, 32) * magnitude
            output = layer(input_tensor)
            
            # Output should be finite and reasonable
            assert torch.isfinite(output).all()
            assert not torch.isnan(output).any()
    
    def test_multi_axis_independence(self):
        config = RingAttractorConfig(num_excitatory=8)
        layer = MultiAxisRingAttractorLayer(
            input_dim=32,
            control_axes=['axis1', 'axis2', 'axis3'],
            ring_axes=['axis1', 'axis2'],  # axis3 is linear
            config=config
        )
        
        input_tensor = torch.randn(5, 32)
        output = layer(input_tensor)
        
        # Each axis should produce independent outputs
        assert output.shape == (5, 3)
        
        # Test that changing one input affects outputs appropriately
        input_tensor2 = input_tensor.clone()
        input_tensor2[:, 0] += 1.0  # Modify first input dimension
        
        output2 = layer(input_tensor2)
        assert not torch.allclose(output, output2)
    
    def test_coupling_effects(self):
        config = RingAttractorConfig(
            num_excitatory=12,
            cross_coupling_factor=0.1
        )
        
        layer = CoupledRingAttractorLayer(
            input_dim=48,
            control_axes=['x', 'y'],
            num_rings=2,
            coupled_axes=['x', 'y'],
            config=config
        )
        
        input_tensor = torch.randn(6, 48)
        output = layer(input_tensor)
        
        # Coupled outputs should show some correlation due to cross-coupling
        assert output.shape == (6, 2)
        assert not torch.isnan(output).any()


class TestControlLayerConsistency:
    """Test consistency across different layer types"""
    
    @pytest.fixture
    def common_config(self):
        return RingAttractorConfig(num_excitatory=16, tau=8.0, beta=10.0)
    
    @pytest.fixture 
    def common_input(self):
        return torch.randn(8, 64)
    
    def test_single_vs_multi_axis_equivalence(self, common_config, common_input):
        # Single axis layer
        single_layer = SingleAxisRingAttractorLayer(
            input_dim=64,
            output_dim=1,
            config=common_config
        )
        
        # Multi axis layer with one axis
        multi_layer = MultiAxisRingAttractorLayer(
            input_dim=64,
            control_axes=['single_axis'],
            ring_axes=['single_axis'],
            config=common_config
        )
        
        # Both should produce valid outputs of same shape
        single_output = single_layer(common_input)
        multi_output = multi_layer(common_input)
        
        assert single_output.shape == multi_output.shape == (8, 1)
        assert not torch.isnan(single_output).any()
        assert not torch.isnan(multi_output).any()
    
    def test_layer_parameter_consistency(self, common_config):
        layers = [
            create_control_layer('single', input_dim=64, output_dim=1, config=common_config),
            create_control_layer('multi', input_dim=64, control_axes=['axis1'], 
                               ring_axes=['axis1'], config=common_config),
        ]
        
        # All layers should have consistent config parameters
        for layer in layers:
            ring_component = (layer.ring_attractor if hasattr(layer, 'ring_attractor') 
                            else layer.ring_layers['axis1'])
            
            assert ring_component.tau.item() == pytest.approx(common_config.tau)
            assert ring_component.beta.item() == pytest.approx(common_config.beta)


# # Performance benchmarks (optional)
# class TestControlLayerPerformance:
#     """Performance tests for control layers"""
    
#     @pytest.mark.benchmark
#     def test_single_axis_forward_speed(self, benchmark):
#         config = RingAttractorConfig(num_excitatory=16)
#         layer = SingleAxisRingAttractorLayer(input_dim=64, output_dim=1, config=config)
#         input_tensor = torch.randn(32, 64)
        
#         result = benchmark(layer, input_tensor)
#         assert result.shape == (32, 1)
    
#     @pytest.mark.benchmark
#     def test_multi_axis_forward_speed(self, benchmark):
#         config = RingAttractorConfig(num_excitatory=16)
#         layer = MultiAxisRingAttractorLayer(
#             input_dim=128,
#             control_axes=['roll', 'pitch', 'yaw', 'thrust'],
#             ring_axes=['roll', 'pitch', 'yaw'],
#             config=config
#         )
#         input_tensor = torch.randn(32, 128)
        
#         result = benchmark(layer, input_tensor)
#         assert result.shape == (32, 4)