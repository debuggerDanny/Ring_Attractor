# Ring Attractor Deep Reinforcement Learning - Usage Guide

This guide explains how to use the modular Ring Attractor system with different RL frameworks and algorithms.

## 📁 Project Structure

```
ring_attractor/
├── core/
│   └── attractors.py          # Core Ring Attractor implementations
├── layers/
│   └── control_layers.py      # Control-specific layers
├── adapters/
│   └── policy_wrappers.py     # Framework integration wrappers
├── utils/
│   └── model_manager.py       # Model saving/loading utilities
└── __init__.py
```

## 🚀 Quick Start

### 1. Basic Usage with Stable Baselines3 DDPG

```python
import gym
from stable_baselines3 import DDPG
from ring_attractor.adapters.policy_wrappers import create_ddpg_ring_attractor

# Create base environment and model
env = gym.make("Pendulum-v1")
base_model = DDPG("MlpPolicy", env, verbose=1)

# Convert to Ring Attractor model using preset configuration
ra_model = create_ddpg_ring_attractor(
    base_model=base_model,
    preset_config="quadrotor",  # Use quadrotor preset
    device="cpu"
)

# Train the model
ra_model.learn(total_timesteps=10000)
```

### 2. Custom Configuration

```python
from ring_attractor.core.attractors import RingAttractorConfig
from ring_attractor.adapters.policy_wrappers import create_ring_attractor_model

# Define custom layer configuration
custom_config = {
    'layer_type': 'multi',  # 'single', 'multi', 'coupled', 'adaptive'
    'input_dim': 64,
    'control_axes': ['roll', 'yaw', 'pitch', 'thrust'],
    'ring_axes': ['roll', 'yaw', 'pitch'],  # Which axes use ring attractors
    'config': RingAttractorConfig(
        num_excitatory=20,
        tau=12.0,
        beta=8.0,
        trainable_structure=False
    )
}

# Create model with custom configuration
ra_model = create_ring_attractor_model(
    base_model=base_model,
    framework="stable_baselines3",
    algorithm="DDPG",
    layer_config=custom_config,
    device="cuda"
)
```

## 🏗️ Core Components

### 1. Core Attractors (`ring_attractor/core/attractors.py`)

This module contains the fundamental Ring Attractor implementations:

#### `RingAttractor`
Single ring attractor for basic spatial representations:

```python
from ring_attractor.core.attractors import RingAttractor

# Create single ring attractor
ring = RingAttractor(
    input_dim=32,
    num_excitatory=16,
    tau=10.0,                    # Time integration constant
    beta=10.0,                   # Scaling factor
    lambda_decay=0.9,            # Distance decay parameter
    trainable_structure=False    # Fixed topology
)

# Forward pass
output = ring(input_tensor)  # Shape: [batch_size, num_excitatory]
```

#### `MultiRingAttractor`
Multiple coupled rings with cross-connections:

```python
from ring_attractor.core.attractors import MultiRingAttractor

# Create triple ring system
multi_ring = MultiRingAttractor(
    input_size=24,
    output_size=16,
    num_rings=3,                 # Number of coupled rings
    cross_coupling_factor=0.1    # Inter-ring connection strength
)

output = multi_ring(input_tensor)  # Shape: [batch_size, output_size * num_rings]
```

#### `RingAttractorConfig`
Configuration management:

```python
from ring_attractor.core.attractors import RingAttractorConfig

# Create configuration
config = RingAttractorConfig(
    num_excitatory=16,
    tau=10.0,
    beta=10.0,
    lambda_decay=0.9,
    trainable_structure=True,
    connectivity_strength=0.1,
    cross_coupling_factor=0.1
)

# Save/load configuration
config_dict = config.to_dict()
loaded_config = RingAttractorConfig.from_dict(config_dict)
```

### 2. Control Layers (`ring_attractor/layers/control_layers.py`)

Specialized layers for control tasks:

#### `SingleAxisRingAttractorLayer`
Simple single-axis control:

```python
from ring_attractor.layers.control_layers import SingleAxisRingAttractorLayer

layer = SingleAxisRingAttractorLayer(
    input_dim=64,
    output_dim=1,  # Single control output
    config=config
)

control_output = layer(state_input)
```

#### `MultiAxisRingAttractorLayer`
Multi-axis control with separate rings:

```python
from ring_attractor.layers.control_layers import MultiAxisRingAttractorLayer

layer = MultiAxisRingAttractorLayer(
    input_dim=64,
    control_axes=['roll', 'yaw', 'pitch', 'thrust'],
    ring_axes=['roll', 'yaw', 'pitch'],  # These use ring attractors
    config=config
)

control_outputs = layer(state_input)  # Shape: [batch_size, 4]
```

#### `CoupledRingAttractorLayer`
Coupled rings for integrated control:

```python
from ring_attractor.layers.control_layers import CoupledRingAttractorLayer

layer = CoupledRingAttractorLayer(
    input_dim=64,
    control_axes=['x', 'y', 'z', 'thrust'],
    num_rings=3,
    coupled_axes=['x', 'y', 'z'],  # These use coupled rings
    config=config
)
```

#### Factory Function
Easy layer creation:

```python
from ring_attractor.layers.control_layers import create_control_layer

layer = create_control_layer(
    layer_type='adaptive',  # 'single', 'multi', 'coupled', 'adaptive'
    input_dim=64,
    control_axes=['roll', 'yaw', 'pitch', 'thrust'],
    config=config,
    architecture_type='multi'  # For adaptive layers
)
```

## 🔌 Framework Integration

### 3. Policy Wrappers (`ring_attractor/adapters/policy_wrappers.py`)

This module provides framework-agnostic integration:

#### Stable Baselines3 Integration

```python
from ring_attractor.adapters.policy_wrappers import (
    create_ddpg_ring_attractor,
    create_sac_ring_attractor,
    create_td3_ring_attractor
)

# DDPG with Ring Attractors
ddpg_model = create_ddpg_ring_attractor(base_ddpg_model, preset_config="drone")

# SAC with Ring Attractors  
sac_model = create_sac_ring_attractor(base_sac_model, layer_config=custom_config)

# TD3 with Ring Attractors
td3_model = create_td3_ring_attractor(base_td3_model, preset_config="arm")
```

#### Generic Model Creation

```python
from ring_attractor.adapters.policy_wrappers import create_ring_attractor_model

# Works with any supported algorithm
model = create_ring_attractor_model(
    base_model=your_model,
    framework="stable_baselines3",  # or "torchrl", "custom"
    algorithm="SAC",                # Algorithm name
    preset_config="quadrotor",      # Preset configuration
    device="cuda"
)
```

#### Custom Framework Integration

```python
from ring_attractor.adapters.policy_wrappers import CustomPolicyWrapper

def my_wrap_function(model, layer_config):
    # Your custom wrapping logic
    layer = create_control_layer(**layer_config)
    model.policy_network = nn.Sequential(model.policy_network, layer)
    return model

def my_extract_function(policy_net):
    return list(policy_net.children())

wrapper = CustomPolicyWrapper(
    wrap_fn=my_wrap_function,
    extract_fn=my_extract_function
)

wrapped_model = wrapper.wrap_policy(base_model, layer_config)
```

## 💾 Model Management

### 4. Model Manager (`ring_attractor/utils/model_manager.py`)

Utilities for saving and loading models:

#### Saving Models

```python
from ring_attractor.utils.model_manager import RingAttractorModelManager

manager = RingAttractorModelManager(base_save_dir="./my_models")

# Save full model
manager.save_model(
    model=trained_model,
    model_name="quadrotor_controller_v1",
    framework="stable_baselines3",
    algorithm="DDPG",
    layer_config=layer_config,
    metadata={
        "training_steps": 100000,
        "environment": "QuadrotorEnv-v1",
        "performance": {"mean_reward": 450.2}
    }
)

# Save only policy weights (smaller file)
manager.save_model(
    model=trained_model,
    model_name="quadrotor_policy_only",
    framework="stable_baselines3", 
    algorithm="DDPG",
    layer_config=layer_config,
    save_policy_only=True
)
```

#### Loading Models

```python
# Create factory function for your model architecture
def create_base_model():
    env = gym.make("QuadrotorEnv-v1")
    return DDPG("MlpPolicy", env)

# Load model
loaded_model, config = manager.load_model(
    model_name="quadrotor_controller_v1",
    model_factory=create_base_model,
    device="cuda"
)

# Access configuration
print(f"Algorithm: {config['algorithm']}")
print(f"Layer config: {config['layer_config']}")
print(f"Metadata: {config['metadata']}")
```

#### Model Registry

```python
from ring_attractor.utils.model_manager import ModelRegistry

registry = ModelRegistry()

# Register model factories
registry.register_factory(
    name="quadrotor_ddpg",
    factory_fn=lambda: create_ddpg_ring_attractor(
        DDPG("MlpPolicy", gym.make("QuadrotorEnv-v1")),
        preset_config="quadrotor"
    ),
    description="DDPG with quadrotor ring attractors"
)

# Use registered factory
factory = registry.get_factory("quadrotor_ddpg")
model = factory()
```

## 🎯 Preset Configurations

### Built-in Presets

#### Quadrotor Configuration
```python
from ring_attractor.layers.control_layers import get_quadrotor_config

config = get_quadrotor_config()
# Returns:
# {
#     'control_axes': ['roll', 'yaw', 'pitch', 'thrust'],
#     'ring_axes': ['roll', 'yaw', 'pitch'],
#     'coupled_axes': ['roll', 'yaw', 'pitch'],
#     'config': RingAttractorConfig(...)
# }
```

#### Drone Navigation Configuration
```python
from ring_attractor.layers.control_layers import get_drone_navigation_config

config = get_drone_navigation_config()
# For forward/right/up/yaw control
```

#### Robotic Arm Configuration
```python
from ring_attractor.layers.control_layers import get_robotic_arm_config

config = get_robotic_arm_config()
# For multi-joint robotic arm control
```

## 🔧 Advanced Usage

### Custom Ring Attractor Architecture

```python
from ring_attractor.core.attractors import RingAttractor, MultiRingAttractor
import torch.nn as nn

class CustomControlNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Preprocessing layers
        self.preprocessing = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Ring attractors for different control aspects
        self.spatial_ring = MultiRingAttractor(
            input_size=21,  # 64/3 rounded
            output_size=16,
            num_rings=3
        )
        
        self.orientation_ring = RingAttractor(
            input_dim=22,
            num_excitatory=12
        )
        
        self.thrust_control = nn.Linear(21, 1)
        
    def forward(self, x):
        x = self.preprocessing(x)
        
        # Split input for different control aspects
        spatial_input, orient_input, thrust_input = torch.chunk(x, 3, dim=1)
        
        # Process through different ring attractors
        spatial_out = self.spatial_ring(spatial_input.repeat(1, 3))
        orient_out = self.orientation_ring(orient_input)
        thrust_out = self.thrust_control(thrust_input)
        
        # Combine outputs
        spatial_controls = torch.mean(spatial_out.view(-1, 3, 16), dim=2)  # [batch, 3]
        orient_controls = torch.mean(orient_out, dim=1, keepdim=True)      # [batch, 1]
        
        return torch.cat([spatial_controls, orient_controls, thrust_out], dim=1)

# Use custom network with policy wrapper
def custom_wrap_fn(model, layer_config):
    existing_layers = list(model.actor.mu.children())[:-2]
    custom_layer = CustomControlNetwork()
    model.actor.mu = nn.Sequential(*existing_layers, custom_layer)
    return model

wrapper = CustomPolicyWrapper(wrap_fn=custom_wrap_fn, extract_fn=lambda x: list(x.children()))
```

### Training with Custom Callbacks

```python
from stable_baselines3.common.callbacks import BaseCallback

class RingAttractorCallback(BaseCallback):
    def __init__(self, save_freq=10000):
        super().__init__()
        self.save_freq = save_freq
        self.manager = RingAttractorModelManager()
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # Save model checkpoint
            self.manager.save_model(
                model=self.model,
                model_name=f"checkpoint_{self.num_timesteps}",
                framework="stable_baselines3",
                algorithm="DDPG",
                layer_config=self.layer_config,
                save_policy_only=True
            )
        return True

# Use callback during training
callback = RingAttractorCallback()
callback.layer_config = layer_config

model.learn(total_timesteps=100000, callback=callback)
```

### Hyperparameter Tuning

```python
from ring_attractor.core.attractors import RingAttractorConfig

def tune_ring_attractor_params():
    configs_to_test = [
        RingAttractorConfig(num_excitatory=12, tau=8.0, beta=12.0),
        RingAttractorConfig(num_excitatory=16, tau=10.0, beta=10.0),
        RingAttractorConfig(num_excitatory=20, tau=12.0, beta=8.0),
    ]
    
    results = []
    for i, config in enumerate(configs_to_test):
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 64,
            'control_axes': ['roll', 'yaw', 'pitch', 'thrust'],
            'config': config
        }
        
        # Create and train model
        model = create_ring_attractor_model(
            base_model=create_base_model(),
            framework="stable_baselines3",
            algorithm="DDPG", 
            layer_config=layer_config
        )
        
        model.learn(total_timesteps=50000)
        
        # Evaluate performance
        mean_reward = evaluate_model(model)
        results.append((config, mean_reward))
        
        # Save best performing model
        if mean_reward == max(r[1] for r in results):
            manager.save_model(
                model=model,
                model_name="best_tuned_model",
                framework="stable_baselines3",
                algorithm="DDPG",
                layer_config=layer_config
            )
    
    return results
```

## 🧪 Testing and Validation

### Unit Testing Ring Attractors

```python
import torch
from ring_attractor.core.attractors import RingAttractor

def test_ring_attractor_output_shape():
    ring = RingAttractor(input_dim=32, num_excitatory=16)
    input_tensor = torch.randn(5, 32)  # Batch size 5
    output = ring(input_tensor)
    
    assert output.shape == (5, 16), f"Expected (5, 16), got {output.shape}"
    print("✅ Ring Attractor output shape test passed")

def test_ring_attractor_parameters():
    ring = RingAttractor(input_dim=32, num_excitatory=16, trainable_structure=False)
    
    # Check that input weights are frozen
    assert not ring.rnn.weight_ih_l0.requires_grad, "Input weights should be frozen"
    
    # Check that tau and beta are learnable
    assert ring.tau.requires_grad, "Tau should be learnable"
    assert ring.beta.requires_grad, "Beta should be learnable"
    
    print("✅ Ring Attractor parameters test passed")

test_ring_attractor_output_shape()
test_ring_attractor_parameters()
```

### Integration Testing

```python
def test_end_to_end_workflow():
    # Create environment
    env = gym.make("Pendulum-v1")
    
    # Create base model
    base_model = DDPG("MlpPolicy", env, verbose=0)
    
    # Create Ring Attractor model
    ra_model = create_ddpg_ring_attractor(
        base_model=base_model,
        preset_config="quadrotor"
    )
    
    # Quick training
    ra_model.learn(total_timesteps=1000)
    
    # Test saving
    manager = RingAttractorModelManager()
    manager.save_model(
        model=ra_model,
        model_name="test_model",
        framework="stable_baselines3",
        algorithm="DDPG",
        layer_config=get_quadrotor_config()
    )
    
    # Test loading
    def factory():
        return create_ddpg_ring_attractor(
            DDPG("MlpPolicy", env, verbose=0),
            preset_config="quadrotor"
        )
    
    loaded_model, config = manager.load_model("test_model", factory)
    
    print("✅ End-to-end workflow test passed")

test_end_to_end_workflow()
```

## 🎭 Common Patterns and Best Practices

### 1. Choosing the Right Architecture
- **Single Ring**: Simple 1D control tasks, basic navigation
- **Multi Ring**: Multi-axis independent control (quadrotor, robotic arm)
- **Coupled Ring**: Integrated multi-dimensional control (3D navigation)
- **Adaptive**: When you need to switch between architectures

### 2. Configuration Guidelines
- Start with preset configurations for common tasks
- Use `trainable_structure=False` for biological plausibility
- Adjust `tau` for temporal dynamics (lower = faster adaptation)
- Adjust `beta` to prevent saturation (higher = more linear responses)

### 3. Performance Tips
- Use GPU (`device="cuda"`) for larger networks
- Save only policy weights for faster loading during inference
- Use model registry for consistent model creation across experiments

This modular design allows you to easily extend the system for new RL algorithms, custom control architectures, and different application domains while maintaining clean separation of concerns.