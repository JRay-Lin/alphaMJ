"""Experience replay buffer for Deep Q-Learning."""

import random
import numpy as np
import torch
from collections import deque, namedtuple
from typing import List, Tuple, Optional


# Experience tuple for storing transitions
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Circular buffer for storing and sampling experiences for DQN training.
    
    Supports uniform random sampling and priority-based sampling.
    """
    
    def __init__(self, buffer_size: int = 100000, batch_size: int = 32, 
                 device: torch.device = None):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Size of sampling batches
            device: PyTorch device for tensor operations
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # Device selection: CUDA > MPS > CPU
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Size of batch to sample (uses default if None)
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        batch_size = batch_size or self.batch_size
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays first
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=np.bool_)
        
        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = None) -> bool:
        """Check if buffer has enough experiences for training."""
        min_size = min_size or self.batch_size
        return len(self.buffer) >= min_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer.
    
    Samples experiences with probability proportional to their TD error,
    giving more weight to surprising/important experiences.
    """
    
    def __init__(self, buffer_size: int = 100000, batch_size: int = 32,
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001,
                 device: torch.device = None):
        """
        Initialize prioritized replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences
            batch_size: Sampling batch size
            alpha: Prioritization exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (compensates for bias)
            beta_increment: Amount to increment beta each step
            device: PyTorch device
        """
        super().__init__(buffer_size, batch_size, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Sum tree for efficient priority sampling
        self.priorities = deque(maxlen=buffer_size)
        
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, priority: Optional[float] = None):
        """Add experience with priority."""
        super().add(state, action, reward, next_state, done)
        
        # Add priority (use max priority for new experiences)
        priority = priority or self.max_priority
        self.priorities.append(priority)
        
    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        batch_size = batch_size or self.batch_size
        
        # Convert priorities to probabilities
        priorities = np.array(list(self.priorities))
        priorities = priorities ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Convert experiences to tensors
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=np.bool_)
        
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).to(self.device)
        weights = torch.from_numpy(weights.astype(np.float32)).to(self.device)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)


class EpisodeBuffer:
    """
    Buffer for storing complete episodes for trajectory-based learning.
    
    Useful for algorithms that need full episode information like n-step returns.
    """
    
    def __init__(self, max_episodes: int = 1000):
        """
        Initialize episode buffer.
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode = []
        
    def add_step(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Add step to current episode."""
        step = {
            'state': state,
            'action': action, 
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.current_episode.append(step)
        
        if done:
            self.end_episode()
    
    def end_episode(self):
        """Finish current episode and start new one."""
        if self.current_episode:
            self.episodes.append(self.current_episode.copy())
            self.current_episode.clear()
    
    def get_episode(self, index: int = -1) -> List[dict]:
        """Get episode by index (default: most recent)."""
        if not self.episodes:
            return []
        return self.episodes[index]
    
    def get_random_episode(self) -> List[dict]:
        """Get random complete episode."""
        if not self.episodes:
            return []
        return random.choice(self.episodes)
    
    def calculate_returns(self, episode: List[dict], gamma: float = 0.99) -> List[float]:
        """Calculate discounted returns for an episode."""
        returns = []
        G = 0
        
        # Calculate returns backwards
        for step in reversed(episode):
            G = step['reward'] + gamma * G
            returns.append(G)
        
        return list(reversed(returns))
    
    def to_replay_buffer(self, replay_buffer: ReplayBuffer, gamma: float = 0.99):
        """Convert episodes to replay buffer format with calculated returns."""
        for episode in self.episodes:
            returns = self.calculate_returns(episode, gamma)
            
            for step, G in zip(episode, returns):
                replay_buffer.add(
                    step['state'],
                    step['action'],
                    G,  # Use calculated return instead of immediate reward
                    step['next_state'],
                    step['done']
                )
    
    def __len__(self) -> int:
        """Return number of completed episodes."""
        return len(self.episodes)