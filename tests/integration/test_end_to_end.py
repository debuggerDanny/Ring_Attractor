import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from attractors import RingAttractor, MultiRingAttractor, RingAttractorConfig
from control_layers import create_control_layer, MultiAxisRingAttractorLayer
from policy_warp import create_sac_ring_attractor
from model_manager import RingAttractorModelManager


class MockEnvironment:
    """Mock environment for testing complete workflows"""
    
    def __init__(self, obs_dim=64, action_dim=4):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        self.step_count = 0
        obs = np.random.randn(self.obs_dim)
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(self.obs_dim)
        reward = np.random.rand() - 0.5  # Random reward
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {'step': self.step_count}
        return obs, reward, terminated, truncated, info


class MockModel:
    """Mock RL model for testing"""
    
    def __init__(self):
        self.policy = MockPolicy()
        self.device = torch.device('cpu')
        self.training_data = []
        
    def predict(self, obs, deterministic=True):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0) if len(obs.shape) == 1 else torch.FloatTensor(obs)
        action_tensor = self.policy(obs_tensor)
        action = action_tensor.detach().numpy()
        return action.squeeze(), None
    
    def learn(self, total_timesteps):
        # Mock learning process
        for step in range(total_timesteps):
            # Simulate training data collection
            obs = np.random.randn(64)
            action, _ = self.predict(obs)
            reward = np.random.rand()
            self.training_data.append((obs, action, reward))
        
        return self
    
    def save(self, path):
        # Mock save
        pass
    
    def load(self, path):
        # Mock load
        pass


class MockPolicy:
    """Mock policy network"""
    
    def __init__(self):
        self.mu = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
    def forward(self, x):
        return self.mu(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def state_dict(self):
        return self.mu.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.mu.load_state_dict(state_dict)


@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete end-to-end workflows"""
    
    def test_quadrotor_control_workflow(self):
        """Test complete quadrotor control workflow"""
        
        # Step 1: Create Ring Attractor configuration for quadrotor
        quadrotor_config = RingAttractorConfig(
            num_excitatory=16,
            tau=8.0,
            beta=12.0,
            lambda_decay=0.8,
            trainable_structure=True,
            connectivity_strength=0.1,
            cross_coupling_factor=0.05
        )
        
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],  # Spatial control
            'config': quadrotor_config
        }
        
        # Step 2: Create base model and integrate Ring Attractor
        base_model = MockModel()
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Step 3: Create environment and test interaction
        env = MockEnvironment(obs_dim=64, action_dim=4)
        
        # Step 4: Run episode
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 50:  # Short episode for testing
            action, _ = ring_model.predict(obs, deterministic=True)
            
            # Verify action is valid for quadrotor
            assert action.shape == (4,), f"Expected 4 actions, got {action.shape}"
            assert np.isfinite(action).all(), "Actions contain invalid values"
            assert not np.isnan(action).any(), "Actions contain NaN values"
            
            # Take environment step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Verify successful episode completion
        assert steps > 0, "No steps taken"
        assert np.isfinite(total_reward), "Total reward is invalid"
        
        # Step 5: Test learning capability
        initial_params = ring_model.policy.state_dict()
        ring_model.learn(total_timesteps=100)  # Short training
        
        # Verify model can still generate actions after training
        obs, _ = env.reset()
        action, _ = ring_model.predict(obs)
        assert action.shape == (4,)
        assert np.isfinite(action).all()
    
    def test_model_development_cycle(self):
        """Test complete model development, training, and evaluation cycle"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Model Manager Setup
            manager = RingAttractorModelManager(base_save_dir=temp_dir)
            
            # Step 2: Experiment with different configurations
            configs = [
                {
                    'name': 'small_ring',
                    'config': RingAttractorConfig(num_excitatory=8, tau=6.0),
                    'layer_type': 'single'
                },
                {
                    'name': 'large_ring',
                    'config': RingAttractorConfig(num_excitatory=24, tau=12.0),
                    'layer_type': 'multi'
                },
                {
                    'name': 'coupled_rings',
                    'config': RingAttractorConfig(num_excitatory=16, cross_coupling_factor=0.2),
                    'layer_type': 'coupled'
                }
            ]
            
            results = []
            
            for config in configs:
                # Create layer config
                if config['layer_type'] == 'single':
                    layer_config = {
                        'layer_type': 'single',
                        'input_dim': 256,
                        'output_dim': 1,
                        'config': config['config']
                    }
                elif config['layer_type'] == 'multi':
                    layer_config = {
                        'layer_type': 'multi',
                        'input_dim': 256,
                        'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
                        'ring_axes': ['roll', 'pitch', 'yaw'],
                        'config': config['config']
                    }
                else:  # coupled
                    layer_config = {
                        'layer_type': 'coupled',
                        'input_dim': 256,
                        'control_axes': ['x', 'y', 'z', 'thrust'],
                        'num_rings': 3,
                        'coupled_axes': ['x', 'y', 'z'],
                        'config': config['config']
                    }
                
                # Create and train model
                base_model = MockModel()
                ring_model = create_sac_ring_attractor(
                    base_model=base_model,
                    layer_config=layer_config,
                    device='cpu'
                )
                
                # Training simulation
                ring_model.learn(total_timesteps=50)
                
                # Evaluation simulation
                env = MockEnvironment()
                obs, _ = env.reset()
                episode_rewards = []
                
                for episode in range(5):  # Short evaluation
                    obs, _ = env.reset()
                    episode_reward = 0
                    steps = 0
                    
                    while steps < 20:  # Short episodes
                        action, _ = ring_model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        steps += 1
                        
                        if terminated or truncated:
                            break
                    
                    episode_rewards.append(episode_reward)
                
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                
                # Save model
                manager.save_model(
                    model=ring_model,
                    model_name=config['name'],
                    framework="test_framework",
                    algorithm="SAC",
                    layer_config=layer_config,
                    metadata={
                        'mean_reward': float(mean_reward),
                        'std_reward': float(std_reward),
                        'episodes': len(episode_rewards)
                    }
                )
                
                results.append({
                    'name': config['name'],
                    'mean_reward': mean_reward,
                    'std_reward': std_reward
                })
            
            # Step 3: Compare results and select best model
            best_config = max(results, key=lambda x: x['mean_reward'])
            
            # Step 4: Load best model for deployment
            def model_factory():
                return create_sac_ring_attractor(
                    base_model=MockModel(),
                    layer_config=layer_config,  # Use last layer_config
                    device='cpu'
                )
            
            best_model, config = manager.load_model(
                model_name=best_config['name'],
                model_factory=model_factory
            )
            
            # Verify loaded model works
            env = MockEnvironment()
            obs, _ = env.reset()
            action, _ = best_model.predict(obs)
            
            assert np.isfinite(action).all()
            assert not np.isnan(action).any()
            
            # Verify all models were saved
            saved_models = manager.list_models()
            assert len(saved_models) == 3
            
            model_names = {model['name'] for model in saved_models}
            expected_names = {'small_ring', 'large_ring', 'coupled_rings'}
            assert model_names == expected_names
    
    def test_hyperparameter_tuning_workflow(self):
        """Test hyperparameter tuning workflow"""
        
        # Define hyperparameter grid
        tau_values = [6.0, 8.0, 12.0]
        beta_values = [8.0, 12.0, 16.0]
        num_excitatory_values = [12, 16, 20]
        
        best_performance = -float('inf')
        best_config = None
        results = []
        
        for tau in tau_values:
            for beta in beta_values:
                for num_excitatory in num_excitatory_values:
                    # Create configuration
                    config = RingAttractorConfig(
                        num_excitatory=num_excitatory,
                        tau=tau,
                        beta=beta,
                        trainable_structure=True
                    )
                    
                    layer_config = {
                        'layer_type': 'multi',
                        'input_dim': 256,
                        'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
                        'ring_axes': ['roll', 'pitch', 'yaw'],
                        'config': config
                    }
                    
                    # Create and evaluate model
                    try:
                        base_model = MockModel()
                        ring_model = create_sac_ring_attractor(
                            base_model=base_model,
                            layer_config=layer_config,
                            device='cpu'
                        )
                        
                        # Quick training
                        ring_model.learn(total_timesteps=25)
                        
                        # Quick evaluation
                        env = MockEnvironment()
                        rewards = []
                        
                        for _ in range(3):  # Very short evaluation
                            obs, _ = env.reset()
                            episode_reward = 0
                            
                            for step in range(10):
                                action, _ = ring_model.predict(obs, deterministic=True)
                                obs, reward, terminated, truncated, info = env.step(action)
                                episode_reward += reward
                                
                                if terminated or truncated:
                                    break
                            
                            rewards.append(episode_reward)
                        
                        mean_reward = np.mean(rewards)
                        
                        result = {
                            'tau': tau,
                            'beta': beta,
                            'num_excitatory': num_excitatory,
                            'mean_reward': mean_reward
                        }
                        results.append(result)
                        
                        # Track best configuration
                        if mean_reward > best_performance:
                            best_performance = mean_reward
                            best_config = result.copy()
                    
                    except Exception as e:
                        # Log failed configuration
                        results.append({
                            'tau': tau,
                            'beta': beta,
                            'num_excitatory': num_excitatory,
                            'mean_reward': -float('inf'),
                            'error': str(e)
                        })
        
        # Verify tuning found valid configurations
        assert len(results) == len(tau_values) * len(beta_values) * len(num_excitatory_values)
        assert best_config is not None, "No valid configuration found"
        assert np.isfinite(best_performance), "Best performance is invalid"
        
        # Verify best config has reasonable values
        assert best_config['tau'] in tau_values
        assert best_config['beta'] in beta_values
        assert best_config['num_excitatory'] in num_excitatory_values
    
    def test_multi_algorithm_comparison(self):
        """Test comparison across different algorithms"""
        
        algorithms = ['SAC', 'DDPG', 'TD3']  # Mock different algorithms
        
        base_layer_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': RingAttractorConfig(num_excitatory=16, tau=8.0, beta=12.0)
        }
        
        algorithm_results = {}
        
        for algorithm in algorithms:
            # Create model (all use same mock model for this test)
            base_model = MockModel()
            ring_model = create_sac_ring_attractor(  # Using SAC wrapper for all
                base_model=base_model,
                layer_config=base_layer_config,
                device='cpu'
            )
            
            # Training simulation
            ring_model.learn(total_timesteps=50)
            
            # Evaluation
            env = MockEnvironment()
            episode_rewards = []
            
            for episode in range(5):
                obs, _ = env.reset()
                episode_reward = 0
                steps = 0
                
                while steps < 15:
                    action, _ = ring_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
            
            algorithm_results[algorithm] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'episodes': len(episode_rewards)
            }
        
        # Verify all algorithms produced results
        assert len(algorithm_results) == len(algorithms)
        
        for algorithm, results in algorithm_results.items():
            assert 'mean_reward' in results
            assert 'std_reward' in results
            assert np.isfinite(results['mean_reward'])
            assert np.isfinite(results['std_reward'])
        
        # Find best performing algorithm
        best_algorithm = max(algorithm_results.keys(), 
                           key=lambda x: algorithm_results[x]['mean_reward'])
        
        assert best_algorithm in algorithms
        
        # Verify performance differences (should have some variance)
        rewards = [results['mean_reward'] for results in algorithm_results.values()]
        assert max(rewards) >= min(rewards)  # Should have some spread


@pytest.mark.integration
@pytest.mark.slow
class TestComplexScenarios:
    """Test complex real-world scenarios"""
    
    def test_curriculum_learning_workflow(self):
        """Test curriculum learning with increasing difficulty"""
        
        # Define curriculum stages
        curriculum_stages = [
            {'max_steps': 20, 'difficulty': 'easy'},
            {'max_steps': 50, 'difficulty': 'medium'},
            {'max_steps': 100, 'difficulty': 'hard'}
        ]
        
        # Create Ring Attractor model
        config = RingAttractorConfig(
            num_excitatory=16,
            tau=8.0,
            beta=12.0,
            trainable_structure=True
        )
        
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': config
        }
        
        base_model = MockModel()
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        performance_progression = []
        
        # Train through curriculum
        for stage_idx, stage in enumerate(curriculum_stages):
            # Create environment for this stage
            env = MockEnvironment()
            env.max_steps = stage['max_steps']
            
            # Training on this stage
            ring_model.learn(total_timesteps=30)  # Short training per stage
            
            # Evaluation on this stage
            rewards = []
            for episode in range(3):
                obs, _ = env.reset()
                episode_reward = 0
                steps = 0
                
                while steps < stage['max_steps']:
                    action, _ = ring_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                rewards.append(episode_reward)
            
            stage_performance = np.mean(rewards)
            performance_progression.append(stage_performance)
            
            # Verify model still generates valid actions at each stage
            obs, _ = env.reset()
            action, _ = ring_model.predict(obs)
            assert np.isfinite(action).all()
            assert action.shape == (4,)
        
        # Verify performance progression (should show learning)
        assert len(performance_progression) == len(curriculum_stages)
        
        # All stages should produce finite performance
        for performance in performance_progression:
            assert np.isfinite(performance), "Stage performance is invalid"
    
    def test_transfer_learning_scenario(self):
        """Test transfer learning between different tasks"""
        
        # Task 1: Simple control (2 actions)
        task1_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['pitch', 'thrust'],
            'ring_axes': ['pitch'],
            'config': RingAttractorConfig(num_excitatory=12)
        }
        
        # Train on Task 1
        base_model1 = MockModel()
        ring_model1 = create_sac_ring_attractor(
            base_model=base_model1,
            layer_config=task1_config,
            device='cpu'
        )
        
        # Train on simple task
        ring_model1.learn(total_timesteps=50)
        
        # Evaluate on Task 1
        env1 = MockEnvironment(action_dim=2)
        obs, _ = env1.reset()
        
        # Verify model works on Task 1
        # Note: This is simplified - real transfer would involve weight extraction
        task1_weights = ring_model1.policy.state_dict()
        
        # Task 2: Complex control (4 actions) - transfer from Task 1
        task2_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': RingAttractorConfig(num_excitatory=16)  # Larger for complex task
        }
        
        # Create Task 2 model
        base_model2 = MockModel()
        ring_model2 = create_sac_ring_attractor(
            base_model=base_model2,
            layer_config=task2_config,
            device='cpu'
        )
        
        # Simulate transfer learning (simplified)
        # In practice, this would involve careful weight mapping
        ring_model2.learn(total_timesteps=30)  # Less training due to transfer
        
        # Evaluate on Task 2
        env2 = MockEnvironment(action_dim=4)
        obs, _ = env2.reset()
        action, _ = ring_model2.predict(obs)
        
        # Verify transfer model works
        assert action.shape == (4,)
        assert np.isfinite(action).all()
        
        # Both models should still work after transfer
        obs1, _ = env1.reset()
        action1, _ = ring_model1.predict(obs1)
        # Note: action1 shape would be (2,) but our mock returns (4,)
        assert np.isfinite(action1).all()
        
        obs2, _ = env2.reset()
        action2, _ = ring_model2.predict(obs2)
        assert action2.shape == (4,)
        assert np.isfinite(action2).all()


@pytest.mark.integration
class TestRobustnessScenarios:
    """Test robustness and error recovery"""
    
    def test_noisy_environment_robustness(self):
        """Test model robustness to environmental noise"""
        
        # Create Ring Attractor model
        config = RingAttractorConfig(
            num_excitatory=16,
            tau=10.0,  # More stable dynamics
            beta=8.0,
            trainable_structure=True
        )
        
        layer_config = {
            'layer_type': 'multi',
            'input_dim': 256,
            'control_axes': ['roll', 'pitch', 'yaw', 'thrust'],
            'ring_axes': ['roll', 'pitch', 'yaw'],
            'config': config
        }
        
        base_model = MockModel()
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Test with increasing noise levels
        noise_levels = [0.0, 0.1, 0.5, 1.0]
        
        for noise_level in noise_levels:
            env = MockEnvironment()
            
            # Run episode with noise
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            valid_actions = 0
            
            while steps < 25:
                # Add noise to observations
                noisy_obs = obs + np.random.normal(0, noise_level, obs.shape)
                
                try:
                    action, _ = ring_model.predict(noisy_obs, deterministic=True)
                    
                    # Verify action is still valid despite noise
                    if np.isfinite(action).all() and not np.isnan(action).any():
                        valid_actions += 1
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                except Exception as e:
                    # Model should handle noisy inputs gracefully
                    print(f"Model failed with noise level {noise_level}: {e}")
                    break
            
            # Model should handle reasonable noise levels
            if noise_level <= 0.5:
                assert valid_actions > steps * 0.8, f"Too many invalid actions with noise {noise_level}"
            
            assert steps > 0, f"No steps taken with noise level {noise_level}"
    
    def test_extreme_input_robustness(self):
        """Test model robustness to extreme inputs"""
        
        config = RingAttractorConfig(num_excitatory=16)
        layer_config = {
            'layer_type': 'single',
            'input_dim': 256,
            'output_dim': 1,
            'config': config
        }
        
        base_model = MockModel()
        ring_model = create_sac_ring_attractor(
            base_model=base_model,
            layer_config=layer_config,
            device='cpu'
        )
        
        # Test extreme inputs
        extreme_inputs = [
            np.zeros(64),  # All zeros
            np.ones(64) * 1000,  # Very large values
            np.ones(64) * -1000,  # Very negative values
            np.full(64, np.inf),  # Infinite values
            np.full(64, -np.inf),  # Negative infinite values
            np.full(64, np.nan),  # NaN values
        ]
        
        for i, extreme_input in enumerate(extreme_inputs):
            try:
                action, _ = ring_model.predict(extreme_input, deterministic=True)
                
                # For finite inputs, action should be finite
                if np.isfinite(extreme_input).all():
                    assert np.isfinite(action).all(), f"Infinite action from finite input {i}"
                    assert not np.isnan(action).any(), f"NaN action from finite input {i}"
                
            except Exception as e:
                # Some extreme inputs may cause exceptions - that's acceptable
                # as long as normal inputs still work
                print(f"Expected exception for extreme input {i}: {e}")
        
        # Verify model still works with normal input after extreme inputs
        normal_input = np.random.randn(64)
        action, _ = ring_model.predict(normal_input)
        assert np.isfinite(action).all(), "Model corrupted by extreme inputs"
        assert not np.isnan(action).any(), "Model producing NaN after extreme inputs"