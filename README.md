# Ring_Attractor
Implementation and adaptation of ring attractors for Pyflyt Sim for quadcoptor from this paper: https://arxiv.org/pdf/2410.03119. It aims to reduce training time for RL agents by giving it ability to reason with spatial information.

# Over view
A PyTorch implementation of biologically-inspired Ring Attractor networks integrated with Deep Reinforcement Learning for continuous control tasks. This implementation provides neural architectures that maintain stable spatial representations through circular connectivity patterns, making them particularly suitable for navigation and control tasks.
Ring Attractors are neural circuit motifs found in biological systems that maintain stable representations of continuous variables (like heading direction or spatial position). This implementation combines these biologically-inspired architectures with modern deep reinforcement learning, specifically DDPG (Deep Deterministic Policy Gradient), for continuous control tasks.

Key Features:
- Biologically-Inspired Architecture: Implements ring attractor dynamics with circular connectivity patterns
- Multiple Configurations: Single ring and triple ring attractor variants
- DDPG Integration: Seamless integration with Stable Baselines3 DDPG implementation
- Flexible Control: Separate control for roll, yaw, pitch, and thrust in multi-axis systems
- Configurable Structure: Choose between fixed topological structure or trainable connectivity
- Comprehensive Logging: TensorBoard integration for training