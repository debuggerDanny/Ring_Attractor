"""
Layered PyFlyt Ring Attractor Integration

This script demonstrates how to create neural networks with Ring Attractor layers
interspersed between regular neural network layers for PyFlyt quadcopter control.

Features:
- Flexible layer ordering: Linear -> Ring -> Linear -> Ring -> etc.
- Multiple Ring Attractor types: single, multi, control-specific
- Easy-to-use builder pattern
- Predefined architectures for common use cases
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.sac.policies import SACPolicy
import PyFlyt.gym_envs
import logging
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Ring Attractor components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.layered_policy import (
        NetworkBuilder, 
        LayeredNetwork, 
        LayerSpec,
        create_simple_ring_network,
        create_quadcopter_policy_network
    )
    from src.utils.attractors import RingAttractorConfig
    LAYERED_POLICY_AVAILABLE = True
    logger.info("✅ Layered Policy system loaded successfully")
except ImportError as e:
    logger.warning(f"❌ Layered Policy components not available: {e}")
    LAYERED_POLICY_AVAILABLE = False


class LayeredRingAttractorSACPolicy(SACPolicy):
    """
    SAC Policy with flexible Ring Attractor layer architecture.
    
    This policy allows you to specify exactly where Ring Attractor layers
    should be placed in the network architecture.
    """
    
    def __init__(
        self,
        observation_space,
        action_space, 
        lr_schedule,
        network_architecture: str = "quadcopter_standard",
        ring_config: Optional[RingAttractorConfig] = None,
        custom_network: Optional[LayeredNetwork] = None,
        **kwargs
    ):
        """
        Initialize policy with layered Ring Attractor architecture.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            lr_schedule: Learning rate schedule
            network_architecture: Predefined architecture name
            ring_config: Ring Attractor configuration
            custom_network: Custom network (overrides architecture)
        """
        self.network_architecture = network_architecture
        self.ring_config = ring_config or RingAttractorConfig()
        self.custom_network = custom_network
        
        # Initialize parent class
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """Build MLP extractor with layered Ring Attractor architecture."""
        if not LAYERED_POLICY_AVAILABLE:
            logger.warning("Layered Policy not available, using standard MLP")
            super()._build_mlp_extractor()
            return
        
        input_dim = self.features_extractor.features_dim
        action_dim = self.action_space.shape[0]
        
        # Create custom layered network
        if self.custom_network is not None:
            self.policy_net = self.custom_network
        else:
            self.policy_net = self._create_predefined_network(input_dim, action_dim)
        
        logger.info(f"Created layered policy network: {self.policy_net.name}")
    
    def _create_predefined_network(self, input_dim: int, action_dim: int) -> LayeredNetwork:
        """Create predefined network architecture."""
        arch = self.network_architecture.lower()
        
        if "quadcopter" in arch:
            if "standard" in arch:
                return create_quadcopter_policy_network(
                    input_dim=input_dim,
                    ring_config=self.ring_config,
                    architecture="standard"
                )
            elif "deep" in arch:
                return create_quadcopter_policy_network(
                    input_dim=input_dim,
                    ring_config=self.ring_config,
                    architecture="deep"
                )
            elif "multi" in arch:
                return create_quadcopter_policy_network(
                    input_dim=input_dim,
                    ring_config=self.ring_config,
                    architecture="multi_ring"
                )
        
        elif "simple" in arch:
            return create_simple_ring_network(
                input_dim=input_dim,
                output_dim=action_dim,
                hidden_dims=[256, 128, 64],
                ring_positions=[1, 3],  # After first and third layers
                ring_config=self.ring_config
            )
        
        elif "ring_sandwich" in arch:
            # Ring layers between every linear layer
            builder = NetworkBuilder(input_dim, action_dim)
            return (builder
                    .add_linear(256)
                    .add_activation('relu')
                    .add_ring(num_excitatory=20, config=self.ring_config)
                    .add_linear(128)
                    .add_activation('relu')
                    .add_ring(num_excitatory=16, config=self.ring_config)
                    .add_linear(64)
                    .add_activation('tanh')
                    .add_ring(num_excitatory=8, config=self.ring_config)
                    .build("RingSandwichPolicy"))
        
        elif "deep_ring" in arch:
            # Deep network with multiple Ring Attractor layers
            builder = NetworkBuilder(input_dim, action_dim)
            return (builder
                    .add_linear(512)
                    .add_activation('relu')
                    .add_dropout(0.1)
                    .add_ring(num_excitatory=32, config=self.ring_config)
                    .add_linear(256)
                    .add_activation('relu')
                    .add_ring(num_excitatory=20, config=self.ring_config)
                    .add_linear(128)
                    .add_activation('relu')
                    .add_ring(num_excitatory=16, config=self.ring_config)
                    .add_linear(64)
                    .add_activation('tanh')
                    .add_ring(num_excitatory=8, config=self.ring_config)
                    .build("DeepRingPolicy"))
        
        else:
            logger.warning(f"Unknown architecture {arch}, using simple ring network")
            return create_simple_ring_network(
                input_dim=input_dim,
                output_dim=action_dim,
                ring_config=self.ring_config
            )
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """Forward pass through the layered network."""
        # Extract features
        features = self.extract_features(obs)
        
        # Pass through layered policy network
        if hasattr(self, 'policy_net'):
            latent_pi = self.policy_net(features)
        else:
            # Fallback to standard MLP
            latent_pi, _ = self.mlp_extractor(features)
        
        # Get action distribution
        return self._get_action_dist_from_latent(latent_pi, deterministic=deterministic)


def create_layered_ring_sac(
    env,
    network_architecture: str = "quadcopter_standard",
    ring_config: Optional[RingAttractorConfig] = None,
    custom_network: Optional[LayeredNetwork] = None,
    **sac_kwargs
) -> SAC:
    """
    Create SAC model with layered Ring Attractor architecture.
    
    Args:
        env: Training environment
        network_architecture: Predefined architecture name
        ring_config: Ring Attractor configuration
        custom_network: Custom layered network
        **sac_kwargs: Additional SAC arguments
        
    Returns:
        SAC model with layered Ring Attractor policy
    """
    if not LAYERED_POLICY_AVAILABLE:
        logger.error("Layered Policy system not available!")
        return None
    
    # Default SAC configuration
    default_sac_config = {
        'learning_rate': 3e-4,
        'batch_size': 256,
        'policy_kwargs': {
            'network_architecture': network_architecture,
            'ring_config': ring_config,
            'custom_network': custom_network
        },
        'verbose': 1
    }
    
    # Update with user-provided arguments
    default_sac_config.update(sac_kwargs)
    
    # Create SAC model
    model = SAC(
        policy=LayeredRingAttractorSACPolicy,
        env=env,
        **default_sac_config
    )
    
    logger.info(f"Created SAC with {network_architecture} Ring Attractor architecture")
    return model


def demonstrate_architecture_examples():
    """Demonstrate different Ring Attractor architectures."""
    if not LAYERED_POLICY_AVAILABLE:
        logger.error("Cannot demonstrate architectures - Layered Policy not available")
        return
    
    print("🧠 Ring Attractor Architecture Examples")
    print("=" * 50)
    
    # Configuration for demonstrations
    config = RingAttractorConfig(
        num_excitatory=16,
        tau=8.0,
        beta=12.0,
        trainable_structure=True
    )
    
    input_dim, output_dim = 64, 4
    
    # 1. Simple Ring Network
    print("1. Simple Ring Network:")
    print("   Linear(64->256) -> ReLU -> Ring(16) -> Linear(128) -> ReLU -> Linear(4)")
    
    net1 = create_simple_ring_network(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[256, 128],
        ring_positions=[1],  # After first ReLU
        ring_config=config
    )
    
    # Test forward pass
    x = torch.randn(10, input_dim)
    try:
        y1 = net1(x)
        print(f"   ✅ Output shape: {y1.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 2. Ring Sandwich Network
    print("2. Ring Sandwich Network:")
    print("   Linear -> ReLU -> Ring -> Linear -> ReLU -> Ring -> Linear -> Tanh -> Ring")
    
    builder = NetworkBuilder(input_dim, output_dim)
    net2 = (builder
            .add_linear(128)
            .add_activation('relu')
            .add_ring(num_excitatory=16, config=config)
            .add_linear(64)
            .add_activation('relu')
            .add_ring(num_excitatory=12, config=config)
            .add_linear(32)
            .add_activation('tanh')
            .add_ring(num_excitatory=8, config=config)
            .build("RingSandwich"))
    
    try:
        y2 = net2(x)
        print(f"   ✅ Output shape: {y2.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 3. Multi-Ring Network
    print("3. Multi-Ring Network:")
    print("   Linear -> ReLU -> MultiRing(3 rings) -> Linear -> ReLU -> Ring")
    
    builder = NetworkBuilder(input_dim, output_dim)
    net3 = (builder
            .add_linear(192)  # Divisible by 3 for multi-ring
            .add_activation('relu')
            .add_multi_ring(output_size=12, num_rings=3, config=config)
            .add_linear(32)
            .add_activation('relu')
            .add_ring(num_excitatory=8, config=config)
            .build("MultiRing"))
    
    try:
        y3 = net3(x)
        print(f"   ✅ Output shape: {y3.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 4. Quadcopter Control Network
    print("4. Quadcopter Control Network:")
    print("   Linear -> ReLU -> Linear -> ReLU -> ControlRing(thrust,roll,pitch,yaw)")
    
    net4 = create_quadcopter_policy_network(
        input_dim=input_dim,
        ring_config=config,
        architecture="standard"
    )
    
    try:
        y4 = net4(x)
        print(f"   ✅ Output shape: {y4.shape}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)


def train_with_custom_architecture():
    """Train PyFlyt with a custom Ring Attractor architecture."""
    print("\n🚁 Training PyFlyt with Custom Ring Attractor Architecture")
    print("=" * 60)
    
    # Create environment
    try:
        env = gym.make(
            "PyFlyt/QuadX-Waypoints-v2",
            sparse_reward=False,
            num_targets=4,
            use_yaw_targets=True,
            goal_reach_distance=0.3,
            flight_dome_size=6.0,
            max_duration_seconds=15.0,
            angle_representation="quaternion",
            agent_hz=30
        )
        logger.info("✅ Created PyFlyt QuadX-Waypoints-v2 environment")
    except Exception as e:
        logger.warning(f"Failed to create waypoints environment: {e}")
        env = gym.make("PyFlyt/QuadX-Hover-v2")
        logger.info("✅ Fallback to QuadX-Hover-v2")
    
    if not LAYERED_POLICY_AVAILABLE:
        logger.error("Cannot train - Layered Policy not available")
        env.close()
        return
    
    # Custom Ring Attractor configuration
    ring_config = RingAttractorConfig(
        num_excitatory=20,
        tau=6.0,          # Fast temporal dynamics
        beta=15.0,        # Strong outputs
        lambda_decay=0.7,
        trainable_structure=True,
        connectivity_strength=0.15,
        cross_coupling_factor=0.1
    )
    
    print("🔧 Architecture Options:")
    print("1. quadcopter_standard - Standard quadcopter control architecture")
    print("2. quadcopter_deep - Deep network with multiple Ring layers")
    print("3. ring_sandwich - Ring layers between every linear layer")
    print("4. deep_ring - Deep network with Ring layers at multiple levels")
    print("5. custom - Build your own architecture")
    
    choice = input("\nSelect architecture (1-5): ").strip()
    
    architecture_map = {
        '1': 'quadcopter_standard',
        '2': 'quadcopter_deep',  
        '3': 'ring_sandwich',
        '4': 'deep_ring'
    }
    
    if choice == '5':
        # Custom architecture building
        print("\n🏗️ Building Custom Architecture:")
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.shape[0]
        
        builder = NetworkBuilder(input_dim, output_dim)
        
        # Example: Let user build step by step (simplified)
        print("Creating: Linear(512) -> ReLU -> Ring(24) -> Linear(256) -> ReLU -> Ring(16) -> ControlRing")
        
        custom_net = (builder
                     .add_linear(512)
                     .add_activation('relu')
                     .add_ring(num_excitatory=24, config=ring_config)
                     .add_linear(256)
                     .add_activation('relu')
                     .add_ring(num_excitatory=16, config=ring_config)
                     .add_control_ring(
                         control_axes=['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'],
                         config=ring_config
                     )
                     .build("CustomQuadcopterPolicy"))
        
        # Create SAC with custom network
        model = create_layered_ring_sac(
            env=env,
            custom_network=custom_net,
            ring_config=ring_config,
            tensorboard_log="./tensorboard_logs/custom_ring"
        )
        
    else:
        architecture = architecture_map.get(choice, 'quadcopter_standard')
        
        # Create SAC with predefined architecture
        model = create_layered_ring_sac(
            env=env,
            network_architecture=architecture,
            ring_config=ring_config,
            tensorboard_log=f"./tensorboard_logs/{architecture}"
        )
    
    if model is None:
        logger.error("Failed to create model")
        env.close()
        return
    
    print(f"\n🚀 Training with architecture: {model.policy.network_architecture}")
    
    # Train the model
    try:
        model.learn(total_timesteps=50000, progress_bar=True)
        logger.info("✅ Training completed successfully!")
        
        # Quick evaluation
        print("\n📊 Quick Evaluation:")
        rewards = []
        for episode in range(5):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            
            rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"\n📈 Results: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Save model
        model.save("./models/layered_ring_attractor_sac")
        print("💾 Model saved to ./models/layered_ring_attractor_sac")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
    
    finally:
        env.close()


if __name__ == "__main__":
    print("🎯 Layered Ring Attractor Integration for PyFlyt")
    print("=" * 50)
    
    # Demonstrate different architectures
    demonstrate_architecture_examples()
    
    # Interactive training
    if input("\nWould you like to train a model? (y/n): ").lower().startswith('y'):
        train_with_custom_architecture()
    
    print("\n✅ Demo completed!")
    print("\nKey benefits of layered Ring Attractor architectures:")
    print("- 🧠 Better spatial reasoning for continuous control")
    print("- 🎯 Flexible placement of Ring layers throughout the network")  
    print("- 🚁 Specialized architectures for quadcopter control")
    print("- 📈 Improved learning efficiency and stability")
    print("- 🔧 Easy customization with builder pattern")