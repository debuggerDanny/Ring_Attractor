"""
Visualization and analysis utilities for PyFlyt Ring Attractor training results.

This module provides tools for visualizing training progress, analyzing Ring Attractor
behavior, and creating performance comparisons for quadcopter waypoint navigation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Any
import torch
import logging
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logger = logging.getLogger(__name__)


class PyFlytRingAttractorAnalyzer:
    """
    Analyzer for PyFlyt Ring Attractor training results and model behavior.
    
    Provides comprehensive analysis tools for understanding Ring Attractor
    performance in quadcopter waypoint navigation tasks.
    """
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.tensorboard_dir = self.results_dir / "tensorboard"
        self.eval_logs_dir = self.results_dir / "eval_logs"
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_tensorboard_data(self, tags: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load training data from TensorBoard logs.
        
        Args:
            tags: Specific tags to load (default: all available)
            
        Returns:
            Dictionary mapping tag names to DataFrames
        """
        if not self.tensorboard_dir.exists():
            logger.warning(f"TensorBoard directory not found: {self.tensorboard_dir}")
            return {}
        
        data = {}
        
        # Find all event files
        event_files = list(self.tensorboard_dir.rglob("events.out.tfevents.*"))
        
        for event_file in event_files:
            try:
                ea = EventAccumulator(str(event_file))
                ea.Reload()
                
                available_tags = ea.Tags()['scalars']
                tags_to_load = tags if tags else available_tags
                
                for tag in tags_to_load:
                    if tag in available_tags:
                        scalar_events = ea.Scalars(tag)
                        df = pd.DataFrame([
                            {'step': s.step, 'value': s.value, 'wall_time': s.wall_time}
                            for s in scalar_events
                        ])
                        data[tag] = df
                        
            except Exception as e:
                logger.warning(f"Failed to load {event_file}: {e}")
        
        return data
    
    def plot_training_curves(
        self, 
        save_path: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Plot training curves from TensorBoard logs.
        
        Args:
            save_path: Path to save the plot
            tags: Specific tags to plot
        """
        data = self.load_tensorboard_data(tags)
        
        if not data:
            logger.warning("No TensorBoard data found for plotting")
            return
        
        # Common tags for SAC training
        common_tags = [
            'rollout/ep_rew_mean',
            'rollout/ep_len_mean',
            'train/actor_loss',
            'train/critic_loss',
            'train/entropy_loss'
        ]
        
        available_tags = [tag for tag in common_tags if tag in data]
        
        if not available_tags:
            available_tags = list(data.keys())[:5]  # Take first 5 available tags
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, tag in enumerate(available_tags[:6]):
            if tag in data:
                df = data[tag]
                axes[i].plot(df['step'], df['value'], linewidth=2, alpha=0.8)
                axes[i].set_title(tag.replace('/', ' ').title(), fontsize=12)
                axes[i].set_xlabel('Training Steps')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
                
                # Add smoothed trend line
                if len(df) > 10:
                    window = max(1, len(df) // 20)
                    smoothed = df['value'].rolling(window=window, center=True).mean()
                    axes[i].plot(df['step'], smoothed, '--', alpha=0.7, linewidth=3)
        
        # Hide unused subplots
        for i in range(len(available_tags), 6):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('PyFlyt Ring Attractor SAC Training Progress', 
                    fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def analyze_ring_attractor_activations(
        self, 
        model,
        test_observations: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze Ring Attractor layer activations for given observations.
        
        Args:
            model: Trained SAC model with Ring Attractor layers
            test_observations: Test observation samples
            save_path: Path to save activation plots
            
        Returns:
            Dictionary of activation patterns
        """
        if not hasattr(model.policy, 'ring_attractor_layer'):
            logger.warning("Model does not have Ring Attractor layers")
            return {}
        
        # Extract Ring Attractor layer
        ring_layer = model.policy.ring_attractor_layer
        
        # Get activations for test observations
        with torch.no_grad():
            observations = torch.FloatTensor(test_observations)
            
            # Forward pass through feature extractor and MLP
            features = model.policy.features_extractor(observations)
            if hasattr(model.policy, 'mlp_extractor'):
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                latent_pi = features
            
            # Get Ring Attractor activations
            ring_activations = ring_layer(latent_pi)
            
            # If multi-ring, analyze each ring separately
            if hasattr(ring_layer, 'ring_attractors'):
                ring_patterns = {}
                for axis, ring in ring_layer.ring_attractors.items():
                    # Get individual ring activations
                    axis_input = latent_pi[:, :ring_layer.ring_split_size]
                    axis_activation = ring(axis_input)
                    ring_patterns[f'{axis}_ring'] = axis_activation.numpy()
                
                ring_patterns['combined'] = ring_activations.numpy()
                
            else:
                ring_patterns = {'single_ring': ring_activations.numpy()}
        
        # Create visualization
        if ring_patterns and save_path:
            self.plot_ring_activations(ring_patterns, save_path)
        
        return ring_patterns
    
    def plot_ring_activations(
        self, 
        ring_patterns: Dict[str, np.ndarray],
        save_path: str
    ) -> None:
        """
        Create visualizations of Ring Attractor activation patterns.
        
        Args:
            ring_patterns: Dictionary of activation patterns
            save_path: Path to save the visualization
        """
        n_patterns = len(ring_patterns)
        fig, axes = plt.subplots(2, (n_patterns + 1) // 2, figsize=(15, 10))
        
        if n_patterns == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (name, patterns) in enumerate(ring_patterns.items()):
            if i >= len(axes):
                break
                
            # Plot activation heatmap
            im = axes[i].imshow(patterns.T, aspect='auto', cmap='viridis')
            axes[i].set_title(f'{name.replace("_", " ").title()} Activations')
            axes[i].set_xlabel('Observation Sample')
            axes[i].set_ylabel('Neuron Index')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(n_patterns, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Ring Attractor Activation Patterns', fontsize=16, y=0.98)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Ring activation plot saved to {save_path}")
        plt.show()
    
    def compare_with_baseline(
        self,
        baseline_results: Dict[str, float],
        ring_attractor_results: Dict[str, float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create comparison plot between baseline SAC and Ring Attractor SAC.
        
        Args:
            baseline_results: Baseline model evaluation results
            ring_attractor_results: Ring Attractor model results
            save_path: Path to save comparison plot
        """
        # Prepare data for plotting
        metrics = list(baseline_results.keys())
        baseline_values = [baseline_results[m] for m in metrics]
        ring_values = [ring_attractor_results[m] for m in metrics]
        
        # Create comparison bar plot
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, baseline_values, width, 
                      label='Baseline SAC', alpha=0.8)
        bars2 = ax.bar(x + width/2, ring_values, width,
                      label='Ring Attractor SAC', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10)
        
        ax.set_xlabel('Evaluation Metrics', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title('Performance Comparison: Baseline vs Ring Attractor SAC', 
                    fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
        # Print numerical comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        for metric in metrics:
            baseline_val = baseline_results[metric]
            ring_val = ring_attractor_results[metric]
            improvement = ((ring_val - baseline_val) / baseline_val) * 100
            
            print(f"{metric.replace('_', ' ').title():25}: "
                  f"Baseline={baseline_val:.3f}, "
                  f"Ring Attractor={ring_val:.3f}, "
                  f"Change={improvement:+.1f}%")
        print("="*60)
    
    def create_waypoint_trajectory_plot(
        self,
        trajectory_data: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize quadcopter trajectories and waypoint navigation.
        
        Args:
            trajectory_data: List of trajectory dictionaries with positions and waypoints
            save_path: Path to save trajectory plot
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        for i, traj in enumerate(trajectory_data[:5]):  # Limit to 5 trajectories
            positions = np.array(traj['positions'])
            waypoints = np.array(traj['waypoints'])
            
            # Plot trajectory
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    alpha=0.7, linewidth=2, label=f'Trajectory {i+1}')
            
            # Plot waypoints
            ax1.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                       s=100, marker='*', alpha=0.8)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.set_title('3D Flight Trajectories')
        ax1.legend()
        
        # XY trajectory plot
        ax2 = fig.add_subplot(222)
        for i, traj in enumerate(trajectory_data[:5]):
            positions = np.array(traj['positions'])
            waypoints = np.array(traj['waypoints'])
            
            ax2.plot(positions[:, 0], positions[:, 1], 
                    alpha=0.7, linewidth=2, label=f'Trajectory {i+1}')
            ax2.scatter(waypoints[:, 0], waypoints[:, 1],
                       s=100, marker='*', alpha=0.8)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('XY Plane Trajectories')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Distance to waypoints over time
        ax3 = fig.add_subplot(223)
        for i, traj in enumerate(trajectory_data[:5]):
            if 'distances_to_target' in traj:
                distances = traj['distances_to_target']
                time_steps = range(len(distances))
                ax3.plot(time_steps, distances, alpha=0.7, linewidth=2)
        
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Distance to Target (m)')
        ax3.set_title('Distance to Waypoints Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Control inputs over time
        ax4 = fig.add_subplot(224)
        if trajectory_data and 'actions' in trajectory_data[0]:
            actions = np.array(trajectory_data[0]['actions'])  # Show first trajectory
            time_steps = range(len(actions))
            
            control_names = ['Thrust', 'Roll Rate', 'Pitch Rate', 'Yaw Rate']
            for i in range(min(4, actions.shape[1])):
                ax4.plot(time_steps, actions[:, i], 
                        label=control_names[i], alpha=0.8, linewidth=2)
            
            ax4.set_xlabel('Time Steps')
            ax4.set_ylabel('Control Input')
            ax4.set_title('Control Commands Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {save_path}")
        
        plt.show()
    
    def generate_full_report(
        self,
        model,
        test_observations: np.ndarray,
        evaluation_results: Dict[str, float],
        baseline_results: Optional[Dict[str, float]] = None,
        trajectory_data: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Generate comprehensive analysis report.
        
        Args:
            model: Trained model
            test_observations: Test observation samples
            evaluation_results: Model evaluation results
            baseline_results: Optional baseline comparison results
            trajectory_data: Optional trajectory visualization data
        """
        report_dir = self.results_dir / "analysis_report"
        report_dir.mkdir(exist_ok=True)
        
        logger.info("Generating comprehensive analysis report...")
        
        # 1. Training curves
        self.plot_training_curves(
            save_path=str(report_dir / "training_curves.png")
        )
        
        # 2. Ring Attractor activations
        self.analyze_ring_attractor_activations(
            model=model,
            test_observations=test_observations,
            save_path=str(report_dir / "ring_activations.png")
        )
        
        # 3. Baseline comparison
        if baseline_results:
            self.compare_with_baseline(
                baseline_results=baseline_results,
                ring_attractor_results=evaluation_results,
                save_path=str(report_dir / "performance_comparison.png")
            )
        
        # 4. Trajectory visualization
        if trajectory_data:
            self.create_waypoint_trajectory_plot(
                trajectory_data=trajectory_data,
                save_path=str(report_dir / "flight_trajectories.png")
            )
        
        # 5. Save evaluation results as JSON
        with open(report_dir / "evaluation_results.json", 'w') as f:
            json.dump({
                'ring_attractor_results': evaluation_results,
                'baseline_results': baseline_results
            }, f, indent=2)
        
        logger.info(f"Analysis report generated in {report_dir}")


def create_sample_baseline_results() -> Dict[str, float]:
    """Create sample baseline results for comparison."""
    return {
        'mean_reward': 245.3,
        'std_reward': 45.2,
        'mean_episode_length': 280.5,
        'mean_waypoints_reached': 4.2,
        'success_rate': 0.65
    }


def create_sample_trajectory_data() -> List[Dict[str, Any]]:
    """Create sample trajectory data for visualization."""
    trajectories = []
    
    for i in range(3):
        # Generate sample trajectory
        n_steps = 200
        positions = np.cumsum(np.random.randn(n_steps, 3) * 0.1, axis=0)
        waypoints = np.random.uniform(-3, 3, (6, 3))
        actions = np.random.uniform(-1, 1, (n_steps, 4))
        
        # Calculate distances to current target
        distances = np.linalg.norm(
            positions - waypoints[i % len(waypoints)], axis=1
        )
        
        trajectories.append({
            'positions': positions.tolist(),
            'waypoints': waypoints.tolist(),
            'actions': actions.tolist(),
            'distances_to_target': distances.tolist()
        })
    
    return trajectories


if __name__ == "__main__":
    # Example usage
    analyzer = PyFlytRingAttractorAnalyzer("./pyflyt_ring_models")
    
    # Create sample data for demonstration
    sample_test_obs = np.random.randn(50, 20)  # 50 test observations
    sample_results = {
        'mean_reward': 287.5,
        'std_reward': 38.1,
        'mean_episode_length': 245.2,
        'mean_waypoints_reached': 5.1,
        'success_rate': 0.82
    }
    
    baseline_results = create_sample_baseline_results()
    trajectory_data = create_sample_trajectory_data()
    
    # Generate plots (without model for demonstration)
    analyzer.plot_training_curves()
    
    analyzer.compare_with_baseline(
        baseline_results=baseline_results,
        ring_attractor_results=sample_results
    )
    
    analyzer.create_waypoint_trajectory_plot(trajectory_data)