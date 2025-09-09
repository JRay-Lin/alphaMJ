"""DQN Agent implementation for Mahjong AI."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from typing import Dict, Tuple, Optional, List
from collections import deque

from ai.dqn_model import DuelingDQN, MahjongStateEncoder
from ai.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from ai.utils.config import dqn_config


class DQNAgent:
    """
    Deep Q-Network agent for Mahjong.
    
    Implements DQN with experience replay, target networks, and epsilon-greedy exploration.
    Can be configured for both training and evaluation modes.
    """
    
    def __init__(self, state_size: int = None, action_size: int = 14,
                 learning_rate: float = None, device: torch.device = None,
                 use_prioritized_replay: bool = False, model_path: str = None):
        """
        Initialize DQN agent.
        
        Args:
            state_size: State vector dimension
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            device: PyTorch device
            use_prioritized_replay: Whether to use prioritized experience replay
            model_path: Path to load pre-trained model
        """
        # Configuration
        self.state_size = state_size or dqn_config.STATE_SIZE
        self.action_size = action_size or dqn_config.ACTION_SIZE
        self.learning_rate = learning_rate or dqn_config.LEARNING_RATE
        # Device selection: CUDA > MPS > CPU
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        
        # Training parameters
        self.batch_size = dqn_config.BATCH_SIZE
        self.gamma = dqn_config.GAMMA
        self.tau = dqn_config.TAU
        self.update_every = dqn_config.UPDATE_EVERY
        
        # Exploration parameters
        self.epsilon = dqn_config.EPSILON_START
        self.epsilon_min = dqn_config.EPSILON_END
        self.epsilon_decay = dqn_config.EPSILON_DECAY
        
        # Networks
        self.q_network = DuelingDQN(
            self.state_size, self.action_size, dqn_config.HIDDEN_SIZES
        ).to(self.device)
        
        self.target_network = DuelingDQN(
            self.state_size, self.action_size, dqn_config.HIDDEN_SIZES
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                dqn_config.BUFFER_SIZE, self.batch_size, device=self.device
            )
        else:
            self.memory = ReplayBuffer(
                dqn_config.BUFFER_SIZE, self.batch_size, device=self.device
            )
        
        # Training tracking
        self.t_step = 0
        self.training = True
        self.loss_history = deque(maxlen=1000)
        
        # State encoder
        self.state_encoder = MahjongStateEncoder()
        
        # Load model if specified
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize target network with same weights as main network
            self.soft_update(1.0)
    
    def get_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
                   epsilon: Optional[float] = None) -> Tuple[int, torch.Tensor]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            action_mask: Mask for valid actions
            epsilon: Override epsilon value
            
        Returns:
            Tuple of (action, q_values)
        """
        if epsilon is None:
            epsilon = self.epsilon if self.training else 0.0
        
        return self.q_network.get_action(state, action_mask, epsilon)
    
    def step(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Save experience and potentially learn from batch.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn from experiences
        self.t_step += 1
        if (self.t_step % self.update_every == 0 and 
            self.memory.is_ready(dqn_config.MIN_BUFFER_SIZE) and 
            self.training):
            self.learn()
    
    def learn(self):
        """Update Q-network using batch of experiences."""
        if isinstance(self.memory, PrioritizedReplayBuffer):
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones, weights, indices = experiences
        else:
            experiences = self.memory.sample()
            states, actions, rewards, next_states, dones = experiences
            weights = torch.ones_like(rewards)
            indices = None
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Use main network to select actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate Q-values
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        td_errors = current_q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update priorities for prioritized replay
        if isinstance(self.memory, PrioritizedReplayBuffer) and indices is not None:
            priorities = torch.abs(td_errors).detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # Soft update target network
        self.soft_update()
        
        # Track loss
        self.loss_history.append(loss.item())
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def soft_update(self, tau: Optional[float] = None):
        """
        Soft update target network parameters.
        
        Args:
            tau: Update rate (default uses config value)
        """
        tau = tau if tau is not None else self.tau
        
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
    
    def save_model(self, filepath: str):
        """
        Save model and training state.
        
        Args:
            filepath: Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            't_step': self.t_step,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'hidden_sizes': dqn_config.HIDDEN_SIZES
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, load_optimizer: bool = True):
        """
        Load model and training state.
        
        Args:
            filepath: Path to model file
            load_optimizer: Whether to load optimizer state
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        # Load training state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        
        if 't_step' in checkpoint:
            self.t_step = checkpoint['t_step']
        
        print(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training: bool):
        """Set training/evaluation mode."""
        self.training = training
        if training:
            self.q_network.train()
            self.target_network.train()
        else:
            self.q_network.eval()
            self.target_network.eval()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'average_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'memory_size': len(self.memory),
            'training_steps': self.t_step
        }
    
    def reset_epsilon(self, epsilon: float):
        """Reset epsilon for new training phase."""
        self.epsilon = epsilon


class MultiAgentDQN:
    """
    Multi-agent DQN manager for separate agent training.
    
    Each agent has its own model and replay buffer for better interpretability
    and independent learning behavior.
    """
    
    def __init__(self, num_agents: int = 4, centralized_training: bool = False, 
                 model_dir: str = "ai/models", device: str = None):
        """
        Initialize multi-agent DQN system with individual agents.
        
        Args:
            num_agents: Number of agents to manage
            centralized_training: Ignored - always use individual training
            model_dir: Directory for saving models
            device: Computing device (cuda/mps/cpu)
        """
        self.num_agents = num_agents
        self.model_dir = model_dir
        self.centralized_training = False  # Always use individual agents
        
        # Individual agents with separate models
        self.agents = []
        for i in range(num_agents):
            agent = DQNAgent(device=device)
            self.agents.append(agent)
        
        print(f"🤖 Initialized {num_agents} individual agents with separate models")
        
        # Performance tracking per agent
        self.agent_performances = [0.0] * num_agents
        self.current_generation = 0
        
    def get_agent(self, index: int) -> DQNAgent:
        """Get agent by index."""
        return self.agents[index % len(self.agents)]
    
    def update_agent_performance(self, index: int, reward: float):
        """Update agent performance tracking."""
        alpha = 0.1  # Learning rate for performance update
        self.agent_performances[index] = (
            (1 - alpha) * self.agent_performances[index] + alpha * reward
        )
    
    def get_best_agent(self) -> DQNAgent:
        """Get the best performing agent."""
        best_idx = np.argmax(self.agent_performances)
        return self.agents[best_idx]
    
    def save_all_agents(self, generation: int):
        """Save all individual agent models."""
        for i, agent in enumerate(self.agents):
            filepath = os.path.join(self.model_dir, f"agent_{i}_gen_{generation}.pth")
            agent.save_model(filepath)
        print(f"💾 Saved {self.num_agents} individual agent models")
    
    def load_generation(self, generation: int):
        """Load all individual agent models from a specific generation."""
        loaded_count = 0
        for i, agent in enumerate(self.agents):
            filepath = os.path.join(self.model_dir, f"agent_{i}_gen_{generation}.pth")
            if os.path.exists(filepath):
                agent.load_model(filepath)
                loaded_count += 1
        print(f"📥 Loaded {loaded_count}/{self.num_agents} agent models from generation {generation}")
    
    def tournament_evaluation(self, num_games: int = 100) -> Dict[int, float]:
        """
        Evaluate individual agent performance in self-play tournament.
        
        Args:
            num_games: Number of games to play
            
        Returns:
            Dictionary mapping agent index to win rate
        """
        return self._evaluate_traditional_tournament(num_games)
    
    
    def _create_baseline_agent(self):
        """Create a simple random baseline agent for evaluation."""
        class BaselineAgent:
            def __init__(self):
                self.epsilon = 0.0
                # Create a dummy q_network for compatibility
                self.q_network = DummyNetwork()
                
            def get_action(self, state, action_mask, epsilon=0.0):
                import torch
                import numpy as np
                
                # Random valid action selection
                valid_actions = torch.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions.cpu().numpy())
                else:
                    action = 0
                
                # Return dummy q_values
                dummy_q = torch.zeros(len(action_mask))
                return action, dummy_q
                
            def set_training_mode(self, training):
                pass
                
        class DummyNetwork:
            def parameters(self):
                # Return a dummy parameter with the same device as the shared agent
                dummy_param = torch.tensor(0.0, requires_grad=True)
                if hasattr(self, '_device'):
                    dummy_param = dummy_param.to(self._device)
                yield dummy_param
                
            def set_device(self, device):
                self._device = device
                
        baseline = BaselineAgent()
        # Set the device to match the shared agent
        if hasattr(self.shared_agent, 'device'):
            baseline.q_network.set_device(self.shared_agent.device)
        return baseline
    
    def _evaluate_traditional_tournament(self, num_games: int) -> Dict[int, float]:
        """Traditional tournament evaluation for individual agents."""
        from ai.training_env import MultiAgentTrainingEnv
        
        # Set all agents to evaluation mode
        for agent in self.agents:
            agent.set_training_mode(False)
        
        # Create tournament environment
        env = MultiAgentTrainingEnv(reward_system="stage1")
        
        # Register agents
        for i, agent in enumerate(self.agents):
            env.register_agent(i, agent)
        
        # Run tournament games
        wins = {i: 0 for i in range(self.num_agents)}
        
        for game_num in range(num_games):
            result = env.run_episode()
            winner = result['info'].get('winner', -1)
            
            if winner is not None and 0 <= winner < self.num_agents:
                wins[winner] += 1
        
        # Calculate win rates
        win_rates = {i: wins[i] / num_games for i in range(self.num_agents)}
        
        # Restore training mode
        for agent in self.agents:
            agent.set_training_mode(True)
        
        return win_rates
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics for all individual agents."""
        stats = {
            'generation': self.current_generation,
            'agent_performances': self.agent_performances.copy(),
            'centralized_training': False,
            'agent_stats': []
        }
        
        # Individual agent statistics
        for i, agent in enumerate(self.agents):
            agent_stats = agent.get_statistics()
            agent_stats['index'] = i
            agent_stats['model_type'] = 'individual'
            stats['agent_stats'].append(agent_stats)
        
        return stats