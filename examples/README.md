# PyFlyt Ring Attractor SAC Integration

This directory contains a complete implementation for integrating Ring Attractor structures with SAC (Soft Actor-Critic) for quadcopter waypoint navigation in the PyFlyt simulation environment.

## 🚁 Overview

The Ring Attractor integration enhances SAC's spatial reasoning capabilities for continuous control tasks, specifically designed for quadcopter waypoint navigation. The biologically-inspired Ring Attractor layers provide better spatial representations and smoother control policies.

## 📁 Files

- **`pyflyt_sac_waypoints.py`**: Main training script with Ring Attractor SAC integration
- **`pyflyt_visualization.py`**: Comprehensive analysis and visualization tools
- **`requirements.txt`**: Required Python packages
- **`README.md`**: This documentation

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt

# Install PyFlyt (if not already installed)
pip install PyFlyt

# Verify installation
python -c "import PyFlyt; print('PyFlyt installed successfully')"
```

### 2. Basic Training

```python
from pyflyt_sac_waypoints import PyFlytRingAttractorTrainer

# Create trainer with default Ring Attractor configuration
trainer = PyFlytRingAttractorTrainer(
    env_id="PyFlyt/QuadX-Waypoints-v4",
    save_dir="./my_models"
)

# Train the model
model = trainer.train(
    total_timesteps=500000,
    n_envs=4
)

# Evaluate the trained model
metrics = trainer.evaluate_model(model, n_episodes=20, render=True)
```

### 3. Run Complete Training Pipeline

```bash
# Run the full training script
python pyflyt_sac_waypoints.py
```

## 🔧 Configuration

### Ring Attractor Configuration

The system uses multi-axis Ring Attractor layers specifically designed for quadcopter control:

```python
ring_config = {
    'layer_type': 'multi',                          # Multi-axis control
    'control_axes': ['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'],
    'ring_axes': ['roll_rate', 'pitch_rate', 'yaw_rate'],  # Spatial control
    'config': RingAttractorConfig(
        num_excitatory=20,      # Neurons per ring
        tau=6.0,               # Temporal dynamics (lower = faster)
        beta=15.0,             # Output scaling
        lambda_decay=0.7,      # Spatial decay
        trainable_structure=True,
        connectivity_strength=0.2,
        cross_coupling_factor=0.15  # Inter-ring coupling
    )
}
```

### Environment Configuration

The PyFlyt environment is configured for optimal waypoint navigation:

```python
env_kwargs = {
    'sparse_reward': False,          # Dense rewards for better learning
    'num_targets': 6,               # Multiple waypoints
    'use_yaw_targets': True,        # Include yaw control
    'goal_reach_distance': 0.3,     # Waypoint reach tolerance
    'goal_reach_angle': 0.2,        # Yaw tolerance
    'flight_dome_size': 8.0,        # Flight area size
    'max_duration_seconds': 20.0,   # Episode duration
    'angle_representation': 'quaternion',  # More stable than Euler
    'agent_hz': 30                  # Control frequency
}
```

### SAC Hyperparameters

Optimized for continuous control:

```python
sac_kwargs = {
    'learning_rate': 3e-4,
    'buffer_size': int(1e6),
    'batch_size': 256,
    'tau': 0.02,                    # Soft update coefficient
    'gamma': 0.98,                  # Discount factor
    'policy_kwargs': {
        'net_arch': [256, 256, 128],
        'ring_config': ring_config
    }
}
```

## 🏗️ Architecture Details

### Custom SAC Policy with Ring Attractors

The `RingAttractorSACPolicy` integrates Ring Attractor layers into the SAC policy network:

1. **Standard SAC layers**: Feature extraction and MLP layers
2. **Ring Attractor layer**: Multi-axis spatial control layer
3. **Action distribution**: Uses Ring Attractor output for action generation

### Ring Attractor Structure

- **Thrust control**: Linear layer (magnitude control)
- **Spatial control**: Ring attractors for roll, pitch, yaw rates
- **Cross-coupling**: Coordination between control axes
- **Biological constraints**: Distance-based connectivity patterns

## 📊 Analysis and Visualization

### Training Analysis

```python
from pyflyt_visualization import PyFlytRingAttractorAnalyzer

analyzer = PyFlytRingAttractorAnalyzer("./my_models")

# Plot training curves
analyzer.plot_training_curves()

# Analyze Ring Attractor activations
ring_patterns = analyzer.analyze_ring_attractor_activations(
    model=trained_model,
    test_observations=test_obs
)

# Compare with baseline
analyzer.compare_with_baseline(
    baseline_results=baseline_metrics,
    ring_attractor_results=ring_metrics
)
```

### Trajectory Visualization

```python
# Visualize flight trajectories and waypoint navigation
analyzer.create_waypoint_trajectory_plot(trajectory_data)

# Generate comprehensive report
analyzer.generate_full_report(
    model=trained_model,
    test_observations=test_obs,
    evaluation_results=metrics
)
```

## 🛠️ Customization

### Custom Ring Configurations

```python
# High-precision navigation
precision_config = RingAttractorConfig(
    num_excitatory=24,
    tau=4.0,           # Fast response
    beta=20.0,         # Strong outputs
    lambda_decay=0.6   # Tight spatial coupling
)

# Smooth, stable control
stable_config = RingAttractorConfig(
    num_excitatory=16,
    tau=10.0,          # Slow, stable dynamics
    beta=8.0,          # Moderate outputs
    lambda_decay=0.9   # Broad spatial coupling
)
```

### Custom Environments

```python
# High-difficulty navigation
difficult_env_kwargs = {
    'num_targets': 10,              # More waypoints
    'goal_reach_distance': 0.15,    # Tighter tolerance
    'flight_dome_size': 12.0,       # Larger area
    'max_duration_seconds': 30.0    # Longer episodes
}

trainer = PyFlytRingAttractorTrainer(
    env_id="PyFlyt/QuadX-Waypoints-v2",
    ring_config=precision_config
)
trainer.env_kwargs.update(difficult_env_kwargs)
```
