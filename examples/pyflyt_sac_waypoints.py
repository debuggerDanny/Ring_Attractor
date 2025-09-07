"""
PyFlyt QuadX-Waypoints SAC Training with Ring Attractor Integration

This script demonstrates how to train a SAC agent with Ring Attractor structures 
for quadcopter waypoint navigation in PyFlyt simulation environment.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import SACPolicy
import PyFlyt.gym_envs
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional

# Import Ring Attractor components (adjust path as needed)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.attractors import RingAttractorConfig
    from src.utils.control_layers import MultiAxisRingAttractorLayer
    from src.utils.model_manager import save_ring_attractor_model
    from src.utils.policy_warp import create_sac_ring_attractor
except ImportError as e:
    print(f"Warning: Could not import Ring Attractor components: {e}")
    print("Please ensure the Ring Attractor project is properly set up")
    # Create dummy classes for demonstration
    class RingAttractorConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MultiAxisRingAttractorLayer:
        def __init__(self, *args, **kwargs):
            pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RingAttractorSACPolicy(SACPolicy):
    """
    Custom SAC policy with Ring Attractor layers for spatial control.
    
    Integrates Ring Attractor structures into the actor network to provide
    biologically-inspired spatial reasoning for quadcopter waypoint navigation.
    """
    
    def __init__(
        self, 
        observation_space, 
        action_space, 
        lr_schedule, 
        ring_config: Optional[Dict[str, Any]] = None,
        net_arch: Optional[List[int]] = None,
        **kwargs
    ):
        # Default Ring Attractor configuration for quadcopter waypoints
        if ring_config is None:
            ring_config = {
                'layer_type': 'multi',
                'control_axes': ['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'],
                'ring_axes': ['roll_rate', 'pitch_rate', 'yaw_rate'],  # Spatial control
                'config': RingAttractorConfig(
                    num_excitatory=16,
                    tau=8.0,
                    beta=12.0,
                    lambda_decay=0.8,
                    trainable_structure=True,
                    connectivity_strength=0.15,
                    cross_coupling_factor=0.1
                )
            }
        
        self.ring_config = ring_config
        
        # Default network architecture if not provided
        if net_arch is None:
            net_arch = [256, 256]
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """Build the MLP feature extractor with Ring Attractor integration."""
        super()._build_mlp_extractor()
        
        # Get the input dimension for Ring Attractor layer
        if hasattr(self.mlp_extractor, 'policy_net'):
            # Get output dimension of the last shared layer
            if len(self.mlp_extractor.policy_net) > 0:
                last_layer = self.mlp_extractor.policy_net[-1]
                if hasattr(last_layer, 'out_features'):
                    ring_input_dim = last_layer.out_features
                else:
                    ring_input_dim = self.net_arch[-1]
            else:
                ring_input_dim = self.features_extractor.features_dim
        else:
            ring_input_dim = self.features_extractor.features_dim
        
        # Create Ring Attractor layer for spatial control
        self.ring_attractor_layer = MultiAxisRingAttractorLayer(
            input_dim=ring_input_dim,
            control_axes=self.ring_config['control_axes'],
            ring_axes=self.ring_config['ring_axes'],
            config=self.ring_config['config']
        )
        
        logger.info(f"Integrated Ring Attractor layer with input_dim={ring_input_dim}")
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> torch.distributions.Distribution:
        """
        Override action distribution computation to use Ring Attractor output.
        
        Args:
            latent_pi: Latent policy features
            
        Returns:
            Action distribution with Ring Attractor spatial reasoning
        """
        # Pass through Ring Attractor layer for spatial control
        ring_output = self.ring_attractor_layer(latent_pi)
        
        # Create action distribution from Ring Attractor output
        mean_actions = self.action_net(ring_output)
        log_std = self.log_std
        
        return self.action_dist.proba_distribution(mean_actions, log_std)


class PyFlytRingAttractorTrainer:
    """
    Trainer class for PyFlyt quadcopter waypoint navigation with Ring Attractors.
    
    Provides complete training pipeline with environment setup, Ring Attractor
    integration, and evaluation utilities.
    """
    
    def __init__(
        self,
        env_id: str = "PyFlyt/QuadX-Waypoints-v2",
        save_dir: str = "./pyflyt_ring_models",
        ring_config: Optional[Dict[str, Any]] = None
    ):
        self.env_id = env_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ring_config = ring_config
        
        # Environment configuration optimized for waypoint navigation
        self.env_kwargs = {
            'sparse_reward': False,          # Dense rewards for better learning
            'num_targets': 6,               # Multiple waypoints for complex navigation
            'use_yaw_targets': True,        # Include yaw targets for full 3D control
            'goal_reach_distance': 0.3,     # Reasonable waypoint reach distance
            'goal_reach_angle': 0.2,        # Yaw tolerance
            'flight_dome_size': 8.0,        # Larger flight area
            'max_duration_seconds': 20.0,   # Extended episode time
            'angle_representation': 'quaternion',  # More stable than Euler
            'agent_hz': 30                  # Control frequency
        }
        
        # SAC hyperparameters optimized for continuous control
        self.sac_kwargs = {
            'learning_rate': 3e-4,
            'buffer_size': int(1e6),
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.02,                    # Soft update coefficient
            'gamma': 0.98,                  # Discount factor
            'train_freq': 1,
            'gradient_steps': 1,
            'target_update_interval': 1,
            'use_sde': False,               # No state-dependent exploration
            'policy_kwargs': {
                'net_arch': [256, 256, 128],
                'ring_config': ring_config
            }
        }
    
    def create_environment(self, render_mode: Optional[str] = None) -> gym.Env:
        """
        Create and configure PyFlyt quadcopter waypoint environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            
        Returns:
            Configured PyFlyt environment
        """
        try:
            env_kwargs = self.env_kwargs.copy()
            if render_mode:
                env_kwargs['render_mode'] = render_mode
            
            env = gym.make(self.env_id, **env_kwargs)
            env = Monitor(env)
            
            logger.info(f"Created PyFlyt environment: {self.env_id}")
            logger.info(f"Observation space: {env.observation_space}")
            logger.info(f"Action space: {env.action_space}")
            
            return env
            
        except Exception as e:
            logger.error(f"Failed to create environment {self.env_id}: {e}")
            # Fallback to basic environment
            logger.info("Falling back to basic QuadX-Hover environment")
            env = gym.make("PyFlyt/QuadX-Hover-v2", render_mode=render_mode)
            return Monitor(env)
    
    def create_vectorized_env(self, n_envs: int = 4) -> VecNormalize:
        """
        Create vectorized and normalized environment for efficient training.
        
        Args:
            n_envs: Number of parallel environments
            
        Returns:
            Vectorized and normalized environment
        """
        def make_env():
            return self.create_environment()
        
        # Create vectorized environment
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # Normalize observations and rewards
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.sac_kwargs['gamma']
        )
        
        logger.info(f"Created vectorized environment with {n_envs} parallel environments")
        return env
    
    def create_model(self, env: gym.Env) -> SAC:
        """
        Create SAC model with Ring Attractor policy.
        
        Args:
            env: Training environment
            
        Returns:
            SAC model with Ring Attractor integration
        """
        model = SAC(
            policy=RingAttractorSACPolicy,
            env=env,
            verbose=1,
            tensorboard_log=str(self.save_dir / "tensorboard"),
            **self.sac_kwargs
        )
        
        logger.info("Created SAC model with Ring Attractor policy")
        return model
    
    def setup_callbacks(self, eval_env: gym.Env) -> List:
        """
        Setup training callbacks for evaluation and checkpointing.
        
        Args:
            eval_env: Environment for evaluation
            
        Returns:
            List of configured callbacks
        """
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.save_dir / "best_model"),
            log_path=str(self.save_dir / "eval_logs"),
            eval_freq=10000,
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=str(self.save_dir / "checkpoints"),
            name_prefix='sac_ring_attractor'
        )
        
        return [eval_callback, checkpoint_callback]
    
    def train(
        self, 
        total_timesteps: int = 500000,
        n_envs: int = 4,
        eval_episodes: int = 10
    ) -> SAC:
        """
        Complete training pipeline for Ring Attractor SAC on PyFlyt waypoints.
        
        Args:
            total_timesteps: Total training timesteps
            n_envs: Number of parallel environments
            eval_episodes: Episodes for evaluation
            
        Returns:
            Trained SAC model
        """
        logger.info("Starting Ring Attractor SAC training on PyFlyt waypoints")
        
        # Create environments
        train_env = self.create_vectorized_env(n_envs=n_envs)
        eval_env = self.create_environment()
        
        # Create model
        model = self.create_model(train_env)
        
        # Setup callbacks
        callbacks = self.setup_callbacks(eval_env)
        
        # Train the model
        logger.info(f"Training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True
        )
        
        # Save final model with Ring Attractor configuration
        final_model_path = save_ring_attractor_model(
            model=model,
            model_name="pyflyt_sac_waypoints_final",
            framework="stable_baselines3",
            algorithm="SAC",
            layer_config=self.ring_config or {},
            save_dir=self.save_dir,
            metadata={
                'environment': self.env_id,
                'total_timesteps': total_timesteps,
                'n_envs': n_envs,
                'env_kwargs': self.env_kwargs,
                'sac_kwargs': self.sac_kwargs
            }
        )
        
        logger.info(f"Training completed! Final model saved to {final_model_path}")
        
        # Cleanup
        train_env.close()
        eval_env.close()
        
        return model
    
    def evaluate_model(
        self, 
        model: SAC, 
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate trained model performance.
        
        Args:
            model: Trained SAC model
            n_episodes: Number of evaluation episodes
            render: Whether to render evaluation
            
        Returns:
            Evaluation metrics
        """
        env = self.create_environment(render_mode="human" if render else None)
        
        episode_rewards = []
        episode_lengths = []
        waypoints_reached = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            waypoints = 0
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Count waypoints reached (if available in info)
                if 'waypoints_reached' in info:
                    waypoints = info['waypoints_reached']
                
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            waypoints_reached.append(waypoints)
            
            logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                       f"Length={episode_length}, Waypoints={waypoints}")
        
        env.close()
        
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_waypoints_reached': np.mean(waypoints_reached),
            'success_rate': np.mean(np.array(waypoints_reached) >= self.env_kwargs['num_targets'])
        }
        
        logger.info("Evaluation Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")
        
        return metrics


def create_custom_ring_config() -> Dict[str, Any]:
    """
    Create custom Ring Attractor configuration optimized for quadcopter waypoints.
    
    Returns:
        Ring Attractor configuration dictionary
    """
    return {
        'layer_type': 'multi',
        'control_axes': ['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate'],
        'ring_axes': ['roll_rate', 'pitch_rate', 'yaw_rate'],  # Spatial control axes
        'config': RingAttractorConfig(
            num_excitatory=16,              # More neurons for complex spatial reasoning
            tau=6.0,                        # Faster temporal dynamics for responsive control
            beta=15.0,                      # Higher scaling for stronger outputs
            lambda_decay=0.7,               # Moderate spatial decay
            trainable_structure=True,       # Allow learning of connectivity
            connectivity_strength=0.2,      # Stronger base connections
            cross_coupling_factor=0.15      # Coordinate between control axes
        )
    }


def main():
    """Main training script."""
    logger.info("PyFlyt Ring Attractor SAC Training")
    
    # Create custom Ring Attractor configuration
    ring_config = create_custom_ring_config()
    
    # Create trainer
    trainer = PyFlytRingAttractorTrainer(
        env_id="PyFlyt/QuadX-Waypoints-v2",
        save_dir="./pyflyt_ring_models",
        ring_config=ring_config
    )
    
    try:
        # Train the model
        model = trainer.train(
            total_timesteps=500000,
            n_envs=4,
            eval_episodes=10
        )
        
        # Evaluate the trained model
        logger.info("Evaluating trained model...")
        metrics = trainer.evaluate_model(model, n_episodes=20, render=False)
        
        # Print final results
        print("\n" + "="*50)
        print("FINAL EVALUATION RESULTS")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value:.3f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()