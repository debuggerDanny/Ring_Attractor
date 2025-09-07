"""
Simple PyFlyt Ring Attractor Integration Example

This script shows the easiest way to integrate Ring Attractor structures
with SAC for PyFlyt quadcopter waypoint navigation using the policy wrapper system.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import PyFlyt.gym_envs
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Ring Attractor components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.policy_warp import create_sac_ring_attractor
    from src.utils.attractors import RingAttractorConfig
    RING_ATTRACTOR_AVAILABLE = True
except ImportError:
    logger.warning("Ring Attractor components not available. Using standard SAC.")
    RING_ATTRACTOR_AVAILABLE = False


def create_pyflyt_environment(env_id: str = "PyFlyt/QuadX-Waypoints-v3"):
    """
    Create PyFlyt quadcopter waypoint environment.
    
    Args:
        env_id: Environment identifier
        
    Returns:
        PyFlyt environment
    """
    try:
        env = gym.make(
            env_id,
            sparse_reward=False,
            num_targets=4,
            use_yaw_targets=True,
            goal_reach_distance=0.3,
            goal_reach_angle=0.2,
            flight_dome_size=6.0,
            max_duration_seconds=15.0,
            angle_representation="quaternion",
            agent_hz=30
        )
        logger.info(f"Created environment: {env_id}")
        return env
    except Exception as e:
        logger.error(f"Failed to create {env_id}: {e}")
        # Fallback to hover environment
        env = gym.make("PyFlyt/QuadX-Waypoints-v3")
        logger.info("Fallback to QuadX-Waypoints-v3")
        return env


def train_baseline_sac():
    """Train baseline SAC without Ring Attractors."""
    logger.info("Training baseline SAC...")
    
    # Create environment
    env = create_pyflyt_environment()
    
    # Create baseline SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./tensorboard_logs/baseline_sac"
    )
    
    # Train
    model.learn(total_timesteps=100000, progress_bar=True)
    
    # Save model
    model.save("./models/baseline_sac_pyflyt")
    
    # Evaluate
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    
    baseline_performance = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards)
    }
    
    logger.info(f"Baseline SAC Performance: {baseline_performance}")
    env.close()
    return model, baseline_performance


def train_ring_attractor_sac():
    """Train SAC with Ring Attractor layers."""
    if not RING_ATTRACTOR_AVAILABLE:
        logger.error("Ring Attractor components not available!")
        return None, {}
    
    logger.info("Training Ring Attractor SAC...")
    
    # Create environment
    env = create_pyflyt_environment()
    
    # Create baseline SAC model first
    base_model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./tensorboard_logs/ring_attractor_sac"
    )
    
    # Configure Ring Attractor for quadcopter control
    ring_config = {
        'layer_type': 'multi',
        'input_dim': 256,  # Should match the last layer of the MLP
        'control_axes': ['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'],
        'ring_axes': ['roll_rate', 'pitch_rate', 'yaw_rate'],  # Spatial axes
        'config': RingAttractorConfig(
            num_excitatory=16,
            tau=8.0,
            beta=12.0,
            lambda_decay=0.8,
            trainable_structure=True,
            connectivity_strength=0.1,
            cross_coupling_factor=0.05
        )
    }
    
    # Wrap with Ring Attractor layers
    try:
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=ring_config,
            device="cpu"
        )
        logger.info("Successfully created Ring Attractor SAC model")
    except Exception as e:
        logger.error(f"Failed to create Ring Attractor model: {e}")
        logger.info("Falling back to baseline SAC")
        ring_model = base_model
    
    # Train
    ring_model.learn(total_timesteps=100000, progress_bar=True)
    
    # Save model
    ring_model.save("./models/ring_attractor_sac_pyflyt")
    
    # Evaluate
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = ring_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    
    ring_performance = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards)
    }
    
    logger.info(f"Ring Attractor SAC Performance: {ring_performance}")
    env.close()
    return ring_model, ring_performance


def compare_models():
    """Compare baseline SAC vs Ring Attractor SAC."""
    logger.info("Starting model comparison...")
    
    # Train baseline
    baseline_model, baseline_perf = train_baseline_sac()
    
    # Train Ring Attractor version
    ring_model, ring_perf = train_ring_attractor_sac()
    
    if ring_perf:
        # Print comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"Baseline SAC:")
        print(f"  Mean Reward: {baseline_perf['mean_reward']:.2f} ± {baseline_perf['std_reward']:.2f}")
        
        print(f"Ring Attractor SAC:")
        print(f"  Mean Reward: {ring_perf['mean_reward']:.2f} ± {ring_perf['std_reward']:.2f}")
        
        improvement = ring_perf['mean_reward'] - baseline_perf['mean_reward']
        improvement_pct = (improvement / baseline_perf['mean_reward']) * 100
        print(f"Improvement: {improvement:.2f} ({improvement_pct:+.1f}%)")
        print("="*60)
    
    return baseline_model, ring_model


def demonstrate_ring_attractor_integration():
    """Demonstrate the simplest Ring Attractor integration."""
    logger.info("Demonstrating Ring Attractor integration...")
    
    if not RING_ATTRACTOR_AVAILABLE:
        logger.warning("Ring Attractor not available. Showing baseline training only.")
        baseline_model, _ = train_baseline_sac()
        return baseline_model
    
    # Create environment
    env = create_pyflyt_environment()
    
    # Step 1: Create standard SAC
    logger.info("Step 1: Creating standard SAC model")
    base_sac = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128])  # Smaller network for demo
    )
    
    # Step 2: Define Ring Attractor configuration
    logger.info("Step 2: Configuring Ring Attractor for quadcopter control")
    quadcopter_ring_config = {
        'layer_type': 'multi',
        'input_dim': 128,
        'control_axes': ['thrust', 'roll', 'pitch', 'yaw'],
        'ring_axes': ['roll', 'pitch', 'yaw'],  # Use rings for spatial control
        'config': RingAttractorConfig(
            num_excitatory=12,      # Compact for demo
            tau=10.0,
            beta=8.0,
            trainable_structure=True
        )
    }
    
    # Step 3: Apply Ring Attractor transformation
    logger.info("Step 3: Applying Ring Attractor transformation")
    try:
        ring_sac = create_sac_ring_attractor(
            base_model=base_sac,
            layer_config=quadcopter_ring_config
        )
        logger.info("✅ Ring Attractor integration successful!")
        
        # Step 4: Train the enhanced model
        logger.info("Step 4: Training Ring Attractor enhanced SAC")
        ring_sac.learn(total_timesteps=50000, progress_bar=True)
        
        # Step 5: Quick evaluation
        logger.info("Step 5: Quick evaluation")
        obs, _ = env.reset()
        for step in range(100):
            action, _ = ring_sac.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        logger.info("✅ Ring Attractor SAC training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Ring Attractor integration failed: {e}")
        logger.info("Continuing with baseline SAC")
        ring_sac = base_sac
    
    env.close()
    return ring_sac


if __name__ == "__main__":
    print("PyFlyt Ring Attractor Integration Demo")
    print("=====================================")
    
    # Choose demonstration mode
    mode = input("Choose mode:\n1. Simple demo\n2. Full comparison\nEnter (1 or 2): ").strip()
    
    if mode == "2":
        # Full comparison
        compare_models()
    else:
        # Simple demonstration
        model = demonstrate_ring_attractor_integration()
        print("\n✅ Demo completed! Check the logs for details.")
        
        if RING_ATTRACTOR_AVAILABLE:
            print("\nNext steps:")
            print("- Adjust Ring Attractor parameters in the config")
            print("- Increase training timesteps for better performance")
            print("- Use visualization tools to analyze results")
        else:
            print("\nTo use Ring Attractors:")
            print("- Ensure the src/ directory is in your Python path")
            print("- Check that all Ring Attractor modules are available")
            print("- Install required dependencies")