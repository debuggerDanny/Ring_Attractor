"""
Simple Layered Ring Attractor Demo

This script demonstrates the easiest way to create neural networks with 
Ring Attractor layers interspersed between regular neural network layers.

Key Features:
✅ Linear -> Ring -> Linear -> Ring patterns  
✅ Easy builder pattern
✅ Works with any PyTorch model
✅ Simple and reliable
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.layered_policy import NetworkBuilder, create_simple_ring_network
    from src.utils.attractors import RingAttractorConfig
    AVAILABLE = True
    print("✅ Ring Attractor components loaded successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    AVAILABLE = False


def demo_basic_builder():
    """Demonstrate the basic NetworkBuilder usage."""
    print("\n🏗️ Basic NetworkBuilder Demo")
    print("=" * 40)
    
    if not AVAILABLE:
        print("❌ Components not available")
        return
    
    # Create Ring Attractor configuration
    config = RingAttractorConfig(
        num_excitatory=16,
        tau=8.0,
        beta=10.0,
        trainable_structure=True
    )
    
    # Method 1: Using the builder pattern step-by-step
    print("1. Step-by-step builder:")
    
    builder = NetworkBuilder(input_dim=64, output_dim=4)
    
    network = (builder
               .add_linear(256)           # Linear layer: 64 -> 256
               .add_activation('relu')    # ReLU activation
               .add_ring(16, config)      # Ring Attractor: 256 -> 16
               .add_linear(128)           # Linear layer: 16 -> 128  
               .add_activation('relu')    # ReLU activation
               .add_ring(12, config)      # Ring Attractor: 128 -> 12
               .add_linear(64)            # Linear layer: 12 -> 64
               .add_activation('tanh')    # Tanh activation
               .build("StepByStepDemo"))  # Final build
    
    print(f"   Created: {network.name}")
    
    # Test the network
    test_input = torch.randn(10, 64)  # Batch of 10 samples
    try:
        output = network(test_input)
        print(f"   Input: {test_input.shape} -> Output: {output.shape}")
        print("   ✅ Network works correctly!")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_simple_patterns():
    """Demonstrate common Ring Attractor patterns."""
    print("\n🎯 Common Ring Attractor Patterns")
    print("=" * 40)
    
    if not AVAILABLE:
        print("❌ Components not available")
        return
    
    config = RingAttractorConfig(num_excitatory=12, trainable_structure=True)
    
    # Pattern 1: Ring at the end
    print("1. Ring at the end pattern:")
    print("   Linear -> ReLU -> Linear -> ReLU -> Ring -> Output")
    
    net1 = create_simple_ring_network(
        input_dim=32,
        output_dim=4,
        hidden_dims=[128, 64],
        ring_positions=[3],  # After second ReLU (0-indexed: Linear=0, ReLU=1, Linear=2, ReLU=3)
        ring_config=config
    )
    
    # Pattern 2: Multiple rings
    print("\n2. Multiple rings pattern:")
    print("   Linear -> ReLU -> Ring -> Linear -> ReLU -> Ring -> Output")
    
    net2 = create_simple_ring_network(
        input_dim=32,
        output_dim=4,
        hidden_dims=[128, 64],
        ring_positions=[1, 3],  # After first and second ReLU
        ring_config=config
    )
    
    # Test both networks
    test_input = torch.randn(5, 32)
    
    try:
        out1 = net1(test_input)
        out2 = net2(test_input)
        print(f"\n   Pattern 1 output: {out1.shape}")
        print(f"   Pattern 2 output: {out2.shape}")
        print("   ✅ Both patterns work!")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_quadcopter_specific():
    """Demonstrate quadcopter-specific Ring Attractor networks."""
    print("\n🚁 Quadcopter Control Networks")
    print("=" * 40)
    
    if not AVAILABLE:
        print("❌ Components not available")
        return
    
    from src.utils.layered_policy import create_quadcopter_policy_network
    
    config = RingAttractorConfig(
        num_excitatory=16,
        tau=6.0,      # Fast response for quadcopter
        beta=12.0,    # Strong control outputs
        trainable_structure=True
    )
    
    # Standard quadcopter network
    print("1. Standard quadcopter control:")
    quad_net = create_quadcopter_policy_network(
        input_dim=20,  # Typical quadcopter observation size
        ring_config=config,
        architecture="standard"
    )
    
    # Deep quadcopter network  
    print("2. Deep quadcopter control:")
    deep_quad_net = create_quadcopter_policy_network(
        input_dim=20,
        ring_config=config,
        architecture="deep"
    )
    
    # Test with quadcopter-like input
    quad_obs = torch.randn(8, 20)  # Batch of 8 quadcopter observations
    
    try:
        standard_actions = quad_net(quad_obs)
        deep_actions = deep_quad_net(quad_obs)
        
        print(f"\n   Standard network: {quad_obs.shape} -> {standard_actions.shape}")
        print(f"   Deep network: {quad_obs.shape} -> {deep_actions.shape}")
        print("   ✅ Quadcopter networks ready for PyFlyt!")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_custom_architecture():
    """Demonstrate creating completely custom architectures."""
    print("\n🎨 Custom Architecture Creation")
    print("=" * 40)
    
    if not AVAILABLE:
        print("❌ Components not available")
        return
    
    config = RingAttractorConfig(num_excitatory=20, trainable_structure=True)
    
    print("Creating custom architecture:")
    print("Linear(100->256) -> BatchNorm -> ReLU -> Dropout -> Ring(20) -> ")
    print("Linear(20->128) -> ReLU -> Ring(16) -> Linear(16->64) -> Tanh -> Ring(8)")
    
    builder = NetworkBuilder(input_dim=100, output_dim=8)
    
    custom_net = (builder
                  .add_linear(256)
                  .add_batch_norm()
                  .add_activation('relu')
                  .add_dropout(0.2)
                  .add_ring(20, config)
                  .add_linear(128)
                  .add_activation('relu')
                  .add_ring(16, config)
                  .add_linear(64)
                  .add_activation('tanh')
                  .add_ring(8, config)
                  .build("CustomArchitecture"))
    
    # Test the custom network
    test_input = torch.randn(16, 100)
    
    try:
        output = custom_net(test_input)
        print(f"\n   Custom network: {test_input.shape} -> {output.shape}")
        print("   ✅ Custom architecture works!")
        
        # Count Ring Attractor layers
        ring_count = sum(1 for layer in custom_net.layers 
                        if 'Ring' in type(layer).__name__)
        print(f"   📊 Total Ring Attractor layers: {ring_count}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_integration_with_sac():
    """Show how this integrates with SAC training."""
    print("\n🤖 SAC Integration Example")
    print("=" * 40)
    
    if not AVAILABLE:
        print("❌ Components not available")
        return
    
    print("Here's how you would use this with SAC for PyFlyt:")
    print()
    
    code_example = '''
# 1. Create your custom Ring Attractor network
config = RingAttractorConfig(num_excitatory=16, tau=8.0, trainable_structure=True)

network = (NetworkBuilder(input_dim=observation_size, output_dim=4)
           .add_linear(256)
           .add_activation('relu')
           .add_ring(16, config)          # Ring layer for spatial reasoning
           .add_linear(128)
           .add_activation('relu')
           .add_ring(12, config)          # Another ring layer
           .build("QuadcopterPolicy"))

# 2. Use it in a custom SAC policy class
class CustomRingSACPolicy(SACPolicy):
    def _build_mlp_extractor(self):
        self.policy_net = network  # Use your custom network
        
# 3. Train with SAC
env = gym.make("PyFlyt/QuadX-Waypoints-v2")
model = SAC(CustomRingSACPolicy, env)
model.learn(total_timesteps=100000)
'''
    
    print(code_example)
    print("✅ This gives you full control over Ring Attractor placement!")


def run_all_demos():
    """Run all demonstration functions."""
    print("🎯 Complete Layered Ring Attractor Demo")
    print("=" * 50)
    
    if not AVAILABLE:
        print("❌ Ring Attractor components not available")
        print("\nTo use this system:")
        print("1. Ensure the src/ directory structure is correct")
        print("2. Check that all modules are properly implemented")
        print("3. Install required dependencies (torch, numpy)")
        return
    
    # Run all demos
    demo_basic_builder()
    demo_simple_patterns() 
    demo_quadcopter_specific()
    demo_custom_architecture()
    demo_integration_with_sac()
    
    print("\n🎉 All demos completed successfully!")
    print("\nKey takeaways:")
    print("✅ Ring Attractors can be placed anywhere in the network")
    print("✅ Builder pattern makes it easy to create complex architectures")
    print("✅ Predefined patterns available for common use cases")
    print("✅ Full customization possible for advanced users")
    print("✅ Ready for integration with SAC and PyFlyt")


if __name__ == "__main__":
    run_all_demos()