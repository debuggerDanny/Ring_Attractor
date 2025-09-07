# Layered Ring Attractor System

A **simple and reliable** way to create neural networks with Ring Attractor layers interspersed between regular neural network layers.

## 🎯 Key Features

✅ **Flexible Layer Ordering**: Place Ring Attractors anywhere in your network  
✅ **Builder Pattern**: Easy, readable network construction  
✅ **Predefined Architectures**: Ready-to-use patterns for common tasks  
✅ **PyTorch Compatible**: Works with any PyTorch model  
✅ **SAC Integration**: Seamless integration with Stable Baselines3  

## 🚀 Quick Start

### Basic Usage

```python
from src.utils.layered_policy import NetworkBuilder
from src.utils.attractors import RingAttractorConfig

# Configure Ring Attractors
config = RingAttractorConfig(
    num_excitatory=16,
    tau=8.0,
    trainable_structure=True
)

# Build network: Linear -> ReLU -> Ring -> Linear -> ReLU -> Ring -> Output
builder = NetworkBuilder(input_dim=64, output_dim=4)

network = (builder
           .add_linear(256)           # Linear layer
           .add_activation('relu')    # ReLU activation  
           .add_ring(16, config)      # Ring Attractor layer
           .add_linear(128)           # Linear layer
           .add_activation('relu')    # ReLU activation
           .add_ring(12, config)      # Ring Attractor layer
           .build("MyRingNetwork"))

# Use it!
output = network(torch.randn(32, 64))  # Batch of 32 samples
```

### Quadcopter Control

```python
from src.utils.layered_policy import create_quadcopter_policy_network

# Specialized network for quadcopter control
quad_net = create_quadcopter_policy_network(
    input_dim=20,  # Observation space size
    architecture="standard",  # or "deep" or "multi_ring"
    ring_config=config
)

# Ready for PyFlyt!
actions = quad_net(quadcopter_observations)
```

### Simple Patterns

```python
from src.utils.layered_policy import create_simple_ring_network

# Ring at the end: Linear -> ReLU -> Linear -> ReLU -> Ring -> Output
net = create_simple_ring_network(
    input_dim=32,
    output_dim=4,
    hidden_dims=[128, 64],
    ring_positions=[3],  # After second ReLU
    ring_config=config
)
```

## 🏗️ Architecture Types

### 1. **Ring at the End** (Recommended for beginners)
```
Linear -> ReLU -> Linear -> ReLU -> Ring -> Output
```
- Ring processes final latent features
- Good for adding spatial reasoning to existing networks

### 2. **Ring Sandwich** 
```
Linear -> ReLU -> Ring -> Linear -> ReLU -> Ring -> Linear -> Tanh -> Ring
```
- Multiple Ring layers throughout the network
- Better spatial processing at different abstraction levels

### 3. **Control-Specific Rings**
```
Linear -> ReLU -> Linear -> ReLU -> ControlRing(thrust,roll,pitch,yaw)
```
- Specialized for quadcopter/drone control
- Separate rings for spatial axes (roll, pitch, yaw)
- Linear layer for thrust (magnitude control)

### 4. **Multi-Ring Architecture**
```
Linear -> ReLU -> MultiRing(3 coupled rings) -> Linear -> ReLU -> Ring
```
- Coupled rings with cross-connections
- Good for complex 3D spatial tasks

## 🎯 SAC Integration

### Method 1: Custom Policy Class

```python
from stable_baselines3.sac.policies import SACPolicy

class LayeredRingSACPolicy(SACPolicy):
    def _build_mlp_extractor(self):
        # Replace standard MLP with your layered network
        self.policy_net = create_quadcopter_policy_network(
            input_dim=self.features_extractor.features_dim,
            architecture="standard",
            ring_config=my_config
        )
    
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi = self.policy_net(features)
        return self._get_action_dist_from_latent(latent_pi, deterministic)

# Use with SAC
model = SAC(LayeredRingSACPolicy, env)
```

### Method 2: Pre-built Integration

```python
from examples.layered_pyflyt_integration import create_layered_ring_sac

# Automatic SAC integration
model = create_layered_ring_sac(
    env=env,
    network_architecture="quadcopter_standard",  # or "deep_ring", "ring_sandwich"
    ring_config=config
)
```

## 🔧 Available Layer Types

| Layer Type | Description | Example |
|------------|-------------|---------|
| `linear` | Standard linear layer | `.add_linear(256)` |
| `ring` | Single Ring Attractor | `.add_ring(16, config)` |
| `multi_ring` | Multiple coupled rings | `.add_multi_ring(output_size=16, num_rings=3)` |
| `control_ring` | Control-specific rings | `.add_control_ring(['thrust', 'roll', 'pitch', 'yaw'])` |
| `activation` | Activation function | `.add_activation('relu')` |
| `dropout` | Dropout layer | `.add_dropout(0.2)` |
| `batch_norm` | Batch normalization | `.add_batch_norm()` |
| `layer_norm` | Layer normalization | `.add_layer_norm()` |

## 📊 Predefined Architectures

### For Quadcopters:
- `"quadcopter_standard"`: Linear -> ReLU -> Linear -> ReLU -> ControlRing
- `"quadcopter_deep"`: Deep network with multiple Ring layers + ControlRing
- `"quadcopter_multi"`: Uses MultiRing architecture for complex spatial control

### General Purpose:
- `"ring_sandwich"`: Ring layers between every linear layer
- `"deep_ring"`: Deep network with Ring layers at multiple levels
- `"simple"`: Basic network with rings at specified positions

## 🎯 PyFlyt Integration Examples

### Basic Training
```python
import gymnasium as gym
from examples.layered_pyflyt_integration import create_layered_ring_sac

# Create environment
env = gym.make("PyFlyt/QuadX-Waypoints-v2", sparse_reward=False)

# Create SAC with Ring Attractors
model = create_layered_ring_sac(
    env=env,
    network_architecture="quadcopter_standard",
    ring_config=RingAttractorConfig(tau=6.0, beta=15.0)
)

# Train
model.learn(total_timesteps=500000)
```

### Custom Architecture
```python
# Build custom network
custom_net = (NetworkBuilder(input_dim=env.observation_space.shape[0], output_dim=4)
              .add_linear(512)
              .add_activation('relu')
              .add_ring(24, config)
              .add_linear(256) 
              .add_activation('relu')
              .add_ring(16, config)
              .add_control_ring(['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'])
              .build("CustomQuadcopter"))

# Use with SAC
model = create_layered_ring_sac(env=env, custom_network=custom_net)
```

## 🧪 Testing Your Networks

```python
# Always test your networks before training!
test_input = torch.randn(10, input_dim)  # Batch of 10 samples

try:
    output = network(test_input)
    print(f"✅ Network works! {test_input.shape} -> {output.shape}")
except Exception as e:
    print(f"❌ Network error: {e}")
```

## 🎯 Best Practices

### 1. **Start Simple**
- Begin with `ring_positions=[3]` (ring at the end)
- Use standard Ring Attractor configurations
- Test with small networks first

### 2. **Ring Placement**
- **Early rings**: Process raw features (good for feature extraction)
- **Middle rings**: Process intermediate representations 
- **Late rings**: Process high-level features (good for control)

### 3. **Configuration Guidelines**
- `tau=6.0-10.0`: Fast response for real-time control
- `tau=10.0-15.0`: Stable, smooth control
- `num_excitatory=12-20`: Good balance of expressiveness and efficiency
- `trainable_structure=True`: Let the network learn optimal connectivity

### 4. **Architecture Selection**
- **Quadcopter waypoints**: Use `"quadcopter_standard"` or `"quadcopter_deep"`
- **Complex 3D navigation**: Use `"multi_ring"` architectures
- **Research/experimentation**: Build custom with `NetworkBuilder`

## 🚀 Run the Demos

```bash
# See all examples in action
python examples/simple_layered_demo.py

# Interactive training with different architectures  
python examples/layered_pyflyt_integration.py

# Full training pipeline
python examples/pyflyt_sac_waypoints.py
```

## 🎉 Benefits

✅ **Better Spatial Reasoning**: Ring Attractors provide biologically-inspired spatial processing  
✅ **Flexible Integration**: Place rings anywhere in your network  
✅ **Easy to Use**: Builder pattern makes complex architectures simple  
✅ **Proven for Drones**: Optimized configurations for quadcopter control  
✅ **Research Ready**: Full customization for advanced experiments  

---

**The layered Ring Attractor system gives you complete control over where and how Ring Attractors are integrated into your neural networks, making it easy to experiment with biologically-inspired architectures for continuous control tasks!** 🧠🚁