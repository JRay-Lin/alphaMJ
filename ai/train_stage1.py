"""Stage 1 training script for Mahjong DQN - Basic win/loss learning."""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.dqn_agent import DQNAgent, MultiAgentDQN
from ai.training_env import MultiAgentTrainingEnv
from ai.utils.config import dqn_config
from ai.reward_system import Stage1RewardSystem


class Stage1Trainer:
    """
    Stage 1 trainer for Mahjong DQN.
    
    Focuses on learning basic game knowledge:
    - Win/loss outcomes
    - Basic game mechanics
    - Turn efficiency
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize Stage 1 trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Device selection: CUDA > MPS > CPU
        device_name = config.get('device', 'auto')
        if device_name == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_name)
        print(f"Training on device: {self.device}")
        
        # Create multi-agent system
        self.multi_agent = MultiAgentDQN(
            num_agents=4,
            centralized_training=self.config['centralized_training'],
            model_dir=self.config['model_dir'],
            device=self.device.type
        )
        
        # Create training environment
        self.env = MultiAgentTrainingEnv(
            rule_name=self.config['rule_name'],
            reward_system="stage1"
        )
        
        # Register agents with environment
        for i in range(4):
            self.env.register_agent(i, self.multi_agent.get_agent(i))
        
        # Training statistics
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.win_rates = deque(maxlen=100)
        self.loss_history = []
        
        # Setup logging
        self.setup_logging()
    
    def _get_default_config(self) -> dict:
        """Get default training configuration."""
        return {
            'num_episodes': 10000,
            'max_steps_per_episode': 200,
            'save_frequency': 200,   # Save less frequently to reduce I/O
            'eval_frequency': 300,   # More frequent evaluation for better monitoring
            'log_frequency': 20,     # Less frequent logging to reduce output
            'model_dir': 'ai/models/stage1',
            'log_dir': 'ai/logs/stage1',
            'rule_name': 'standard',
            'centralized_training': True,  # Use CTDE by default
            'target_win_rate': 0.22,     # Slightly lower target (more realistic)
            'early_stopping_patience': 1500  # Reduced patience for faster iteration
        }
    
    def setup_logging(self):
        """Setup logging directories and files."""
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Save config
        with open(os.path.join(self.config['log_dir'], 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def train(self):
        """Run Stage 1 training."""
        print("Starting Stage 1 Training: Basic Win/Loss Learning")
        print(f"Target episodes: {self.config['num_episodes']}")
        print(f"Rule: {self.config['rule_name']}")
        print("-" * 60)
        
        start_time = time.time()
        best_avg_reward = -float('inf')
        patience_counter = 0
        
        # Training loop
        for episode in tqdm(range(self.config['num_episodes']), desc="Training"):
            # Run episode
            episode_result = self.env.run_episode()
            
            # Process episode data
            self._process_episode(episode_result, episode)
            
            # Logging
            if episode % self.config['log_frequency'] == 0:
                self._log_progress(episode)
            
            # Save models
            if episode % self.config['save_frequency'] == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Evaluation
            if episode % self.config['eval_frequency'] == 0 and episode > 0:
                eval_results = self._evaluate()
                self._log_evaluation(episode, eval_results)
                
                # Early stopping check
                avg_reward = np.mean(self.episode_rewards)
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    patience_counter = 0
                    self._save_best_model(episode)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience'] / self.config['eval_frequency']:
                    print(f"Early stopping at episode {episode}")
                    break
        
        # Final save and evaluation
        self._save_checkpoint(self.config['num_episodes'])
        final_eval = self._evaluate()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final evaluation: {final_eval}")
        
        # Generate training report
        self._generate_training_report(training_time)
    
    def _process_episode(self, episode_result: dict, episode_num: int):
        """Process episode results and update agents."""
        episode_data = episode_result['episode_data']
        info = episode_result['info']
        
        # Track episode statistics
        episode_reward = 0
        episode_length = len(episode_data)
        
        # Process transitions for each agent
        agent_transitions = {i: [] for i in range(4)}
        
        for transition in episode_data:
            player_idx = transition['player_idx']
            agent_transitions[player_idx].append(transition)
            episode_reward += transition['reward']
        
        # Debug: Print episode outcome
        winner = info.get('winner', -1)
        if episode_num % 50 == 0:  # Print every 50th episode
            print(f"Episode {episode_num}: Winner={winner}, Length={episode_length}, Reward={episode_reward:.3f}")
            if winner is not None:
                if winner >= 0:
                    print(f"  🏆 Player {winner} won!")
                else:
                    print(f"  🤝 Draw game")
        
        # Update each agent
        for player_idx, transitions in agent_transitions.items():
            agent = self.multi_agent.get_agent(player_idx)
            
            # Add experiences to agent's replay buffer
            for transition in transitions:
                agent.step(
                    transition['state'],
                    transition['action'],
                    transition['reward'],
                    transition['next_state'],
                    transition['done']
                )
            
            # Update agent performance tracking
            if transitions:
                avg_reward = np.mean([t['reward'] for t in transitions])
                self.multi_agent.update_agent_performance(player_idx, avg_reward)
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward / 4)  # Average across agents
        self.episode_lengths.append(episode_length)
        
        # Track win rates
        winner = info.get('winner', -1)
        if winner is not None and winner >= 0:
            # Calculate recent win rate for winner
            recent_episodes = min(100, len(self.episode_rewards))
            recent_wins = sum(1 for _ in range(recent_episodes) 
                            if episode_num >= recent_episodes - 1)
            win_rate = recent_wins / recent_episodes if recent_episodes > 0 else 0.0
            self.win_rates.append(win_rate)
    
    def _log_progress(self, episode: int):
        """Log training progress."""
        if not self.episode_rewards:
            return
        
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        # Get agent statistics
        agent_stats = self.multi_agent.get_statistics()
        avg_epsilon = np.mean([stat['epsilon'] for stat in agent_stats['agent_stats']])
        avg_loss = np.mean([stat['average_loss'] for stat in agent_stats['agent_stats']])
        
        print(f"Episode {episode:5d} | "
              f"Avg Reward: {avg_reward:6.3f} | "
              f"Avg Length: {avg_length:5.1f} | "
              f"Epsilon: {avg_epsilon:.3f} | "
              f"Loss: {avg_loss:.4f}")
    
    def _evaluate(self) -> dict:
        """Run evaluation tournament."""
        print("Running evaluation tournament...")
        
        # Set agents to evaluation mode
        for i in range(4):
            self.multi_agent.get_agent(i).set_training_mode(False)
        
        # Run tournament
        win_rates = self.multi_agent.tournament_evaluation(num_games=10)
        
        # Calculate additional metrics
        eval_results = {
            'win_rates': win_rates,
            'balance_score': 1.0 - np.std(list(win_rates.values())),  # Higher is more balanced
            'total_games': 100,
            'agent_performances': self.multi_agent.agent_performances.copy()
        }
        
        # Restore training mode
        for i in range(4):
            self.multi_agent.get_agent(i).set_training_mode(True)
        
        return eval_results
    
    def _log_evaluation(self, episode: int, eval_results: dict):
        """Log evaluation results."""
        win_rates = eval_results['win_rates']
        balance = eval_results['balance_score']
        
        print(f"\nEvaluation at episode {episode}:")
        print(f"Win rates: {[f'{rate:.3f}' for rate in win_rates.values()]}")
        print(f"Balance score: {balance:.3f}")
        print(f"Agent performances: {[f'{perf:.3f}' for perf in eval_results['agent_performances']]}")
        print("-" * 40)
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        # Save all agents
        self.multi_agent.save_all_agents(episode)
        
        # Save training statistics
        stats = {
            'episode': episode,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'win_rates': list(self.win_rates),
            'multi_agent_stats': self.multi_agent.get_statistics()
        }
        
        stats_path = os.path.join(self.config['log_dir'], f'training_stats_{episode}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Checkpoint saved at episode {episode}")
    
    def _save_best_model(self, episode: int):
        """Save the best performing model."""
        best_agent = self.multi_agent.get_best_agent()
        best_path = os.path.join(self.config['model_dir'], 'best_model.pth')
        best_agent.save_model(best_path)
        
        # Save metadata
        metadata = {
            'episode': episode,
            'performance': max(self.multi_agent.agent_performances),
            'avg_reward': np.mean(self.episode_rewards),
            'timestamp': time.time(),
            'centralized_training': self.multi_agent.centralized_training,
            'model_type': 'shared' if self.multi_agent.centralized_training else 'individual'
        }
        
        metadata_path = os.path.join(self.config['model_dir'], 'best_model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.multi_agent.centralized_training:
            print(f"💾 Saved best shared model to {best_path}")
    
    def _generate_training_report(self, training_time: float):
        """Generate final training report with visualizations."""
        print("\nGenerating training report...")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward progress
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Win rates
        if self.win_rates:
            axes[1, 0].plot(self.win_rates)
            axes[1, 0].set_title('Win Rates')
            axes[1, 0].set_xlabel('Evaluation')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].axhline(y=0.25, color='r', linestyle='--', label='Target (25%)')
            axes[1, 0].legend()
        
        # Agent performance comparison
        performances = self.multi_agent.agent_performances
        axes[1, 1].bar(range(len(performances)), performances)
        axes[1, 1].set_title('Final Agent Performances')
        axes[1, 1].set_xlabel('Agent Index')
        axes[1, 1].set_ylabel('Performance Score')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['log_dir'], 'training_report.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate text report
        report = {
            'training_time': training_time,
            'total_episodes': len(self.episode_rewards),
            'final_avg_reward': np.mean(self.episode_rewards),
            'final_avg_length': np.mean(self.episode_lengths),
            'best_agent_performance': max(self.multi_agent.agent_performances),
            'agent_balance': 1.0 - np.std(self.multi_agent.agent_performances),
            'config': self.config
        }
        
        report_path = os.path.join(self.config['log_dir'], 'final_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training report saved to {self.config['log_dir']}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Stage 1 Mahjong DQN Training')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--rule', choices=['standard', 'taiwan'], default='standard',
                       help='Mahjong rule variant')
    parser.add_argument('--model-dir', type=str, default='ai/models/stage1',
                       help='Model save directory')
    parser.add_argument('--log-dir', type=str, default='ai/logs/stage1',
                       help='Log directory')
    parser.add_argument('--centralized', action='store_true', default=True,
                       help='Use centralized training (CTDE) - default: True')
    parser.add_argument('--individual-agents', action='store_true',
                       help='Use individual agents instead of CTDE')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Configure device: CUDA > MPS > CPU
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    # Determine training mode
    centralized_training = not args.individual_agents  # Default to CTDE unless explicitly disabled
    
    # Create training configuration
    config = {
        'num_episodes': args.episodes,
        'rule_name': args.rule,
        'model_dir': args.model_dir,
        'log_dir': args.log_dir,
        'centralized_training': centralized_training,
        'max_steps_per_episode': 200,
        'save_frequency': 100,
        'eval_frequency': 500,
        'log_frequency': 10,
        'target_win_rate': 0.25,
        'early_stopping_patience': 2000
    }
    
    # Create and run trainer
    trainer = Stage1Trainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer._save_checkpoint(len(trainer.episode_rewards))
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    print("\nStage 1 training completed!")


if __name__ == "__main__":
    main()